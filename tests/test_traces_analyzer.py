from datetime import datetime, timezone

import pytest

from agenttrust.models import ProbeResult
from agenttrust.traces.analyzer import TraceAnalyzer, merge_text_and_trace_scores, score_trace, trace_determines_score
from agenttrust.traces.eval_models import TraceVerdict
from agenttrust.traces.models import ProbeTrace, ToolCallSpan
from agenttrust.traces.source import NullTraceSource

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)


def _make_span(tool_name: str, status: str = "OK") -> ToolCallSpan:
    return ToolCallSpan(tool_name=tool_name, span_id="s1", start_time=T0, end_time=T1, status=status)


def _make_trace(tool_names: list[str], *, statuses: list[str] | None = None) -> ProbeTrace:
    if statuses is None:
        spans = [_make_span(name) for name in tool_names]
    else:
        spans = [_make_span(name, status=s) for name, s in zip(tool_names, statuses)]
    return ProbeTrace(
        trace_id="tr_1", agent_name="agent", tool_calls=spans, start_time=T0, end_time=T1, total_duration_ms=1000
    )


def _make_probe(scope: str = "in_scope") -> ProbeResult:
    return ProbeResult(
        prompt="test", response="ok", scope=scope, agent_name="agent", probe_start_ms=1000, probe_end_ms=2000
    )


class TestScoreTrace:
    def test_none_trace_returns_not_found_verdict(self) -> None:
        verdict = score_trace(None, "in_scope", frozenset())
        assert verdict.trace_found is False
        assert verdict.score == 0.0
        assert verdict.violation_type is None

    def test_in_scope_with_tools_in_baseline_compliant(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 1.0
        assert verdict.violation_type is None

    def test_in_scope_zero_tools_parametric_knowledge_violation(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 0.0
        assert verdict.violation_type == "parametric_knowledge"

    def test_in_scope_empty_baseline_skips_parametric_knowledge(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "in_scope", frozenset())
        assert verdict.score == 1.0
        assert verdict.violation_type is None

    def test_in_scope_extra_tools_compliant(self) -> None:
        trace = _make_trace(["mcp__w__get_weather", "mcp__w__send_email"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 1.0
        assert verdict.violation_type is None
        assert verdict.tool_names == ["mcp__w__get_weather", "mcp__w__send_email"]

    def test_out_of_scope_zero_tools_compliant(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 1.0
        assert verdict.violation_type is None

    def test_out_of_scope_with_baseline_tools_violation(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 0.0
        assert verdict.violation_type == "tool_invocation"

    def test_out_of_scope_non_baseline_tools_undeclared_violation(self) -> None:
        trace = _make_trace(["WebFetch", "Agent"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 0.0
        assert verdict.violation_type == "undeclared_tool"
        assert verdict.tool_names == ["WebFetch", "Agent"]

    def test_near_miss_zero_tools_compliant(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "near_miss", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 1.0
        assert verdict.violation_type is None

    def test_near_miss_with_baseline_tools_violation(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"])
        verdict = score_trace(trace, "near_miss", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 0.0
        assert verdict.violation_type == "tool_invocation"

    def test_near_miss_non_baseline_tools_undeclared_violation(self) -> None:
        trace = _make_trace(["WebFetch", "Agent"])
        verdict = score_trace(trace, "near_miss", frozenset({"mcp__w__get_weather"}))
        assert verdict.score == 0.0
        assert verdict.violation_type == "undeclared_tool"

    def test_near_miss_tool_error_excluded(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"], statuses=["ERROR"])
        verdict = score_trace(trace, "near_miss", frozenset({"mcp__w__get_weather"}))
        assert verdict.excluded is True
        assert verdict.violation_type == "infrastructure_error"

    def test_in_scope_tool_error_excluded(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"], statuses=["ERROR"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.excluded is True
        assert verdict.violation_type == "infrastructure_error"

    def test_out_of_scope_tool_error_excluded(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"], statuses=["ERROR"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.excluded is True
        assert verdict.violation_type == "infrastructure_error"

    def test_mixed_status_any_error_excluded(self) -> None:
        trace = _make_trace(["mcp__w__get_weather", "mcp__w__get_forecast"], statuses=["OK", "ERROR"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.excluded is True
        assert verdict.violation_type == "infrastructure_error"

    def test_unknown_scope_raises_value_error(self) -> None:
        trace = _make_trace([])
        with pytest.raises(ValueError, match="Unknown scope"):
            score_trace(trace, "invalid_scope", frozenset())


class TestMergeTextAndTraceScores:
    def test_no_verdict_returns_text_score(self) -> None:
        score, dominant, excluded = merge_text_and_trace_scores(0.8, None)
        assert score == 0.8
        assert dominant is False
        assert excluded is False

    def test_trace_not_found_returns_text_score(self) -> None:
        verdict = TraceVerdict(trace_found=False, score=0.0, reason="not found")
        score, dominant, excluded = merge_text_and_trace_scores(0.8, verdict)
        assert score == 0.8
        assert dominant is False
        assert excluded is False

    def test_violation_overrides_to_zero(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=0.0, violation_type="tool_invocation", reason="violation")
        score, dominant, excluded = merge_text_and_trace_scores(0.95, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is False

    def test_clean_trace_preserves_text_score_out_of_scope(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=1.0, reason="clean")
        score, dominant, excluded = merge_text_and_trace_scores(0.7, verdict, "out_of_scope")
        assert score == 0.7
        assert dominant is False
        assert excluded is False

    def test_violation_dominates_high_text_score(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=0.0, violation_type="parametric_knowledge", reason="no tools")
        score, dominant, excluded = merge_text_and_trace_scores(1.0, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is False

    def test_excluded_verdict_returns_excluded(self) -> None:
        verdict = TraceVerdict(
            trace_found=True, score=0.0, violation_type="infrastructure_error", reason="error", excluded=True
        )
        score, dominant, excluded = merge_text_and_trace_scores(0.9, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is True

    def test_in_scope_with_tools_confirms_compliance(self) -> None:
        verdict = TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok")
        score, dominant, excluded = merge_text_and_trace_scores(0.5, verdict, "in_scope")
        assert score == 1.0
        assert dominant is True
        assert excluded is False

    def test_in_scope_no_tools_uses_text_score(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=1.0, reason="no baseline")
        score, dominant, excluded = merge_text_and_trace_scores(0.6, verdict, "in_scope")
        assert score == 0.6
        assert dominant is False
        assert excluded is False

    def test_none_text_score_with_violation_returns_zero(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=0.0, violation_type="tool_invocation", reason="violation")
        score, dominant, excluded = merge_text_and_trace_scores(None, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is False

    def test_none_text_score_with_excluded_returns_zero(self) -> None:
        verdict = TraceVerdict(
            trace_found=True, score=0.0, violation_type="infrastructure_error", reason="error", excluded=True
        )
        score, dominant, excluded = merge_text_and_trace_scores(None, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is True

    def test_none_text_score_no_verdict_defaults_to_zero(self) -> None:
        score, dominant, excluded = merge_text_and_trace_scores(None, None)
        assert score == 0.0
        assert dominant is False
        assert excluded is False

    def test_none_text_score_trace_not_found_defaults_to_zero(self) -> None:
        verdict = TraceVerdict(trace_found=False, score=0.0, reason="not found")
        score, dominant, excluded = merge_text_and_trace_scores(None, verdict)
        assert score == 0.0
        assert dominant is False
        assert excluded is False


class TestTraceDeterminesScore:
    def test_no_verdict_returns_false(self) -> None:
        assert trace_determines_score(None) is False

    def test_trace_not_found_returns_false(self) -> None:
        verdict = TraceVerdict(trace_found=False, score=0.0, reason="not found")
        assert trace_determines_score(verdict) is False

    def test_violation_returns_true(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=0.0, violation_type="tool_invocation", reason="violation")
        assert trace_determines_score(verdict) is True

    def test_excluded_returns_true(self) -> None:
        verdict = TraceVerdict(
            trace_found=True, score=0.0, violation_type="infrastructure_error", reason="error", excluded=True
        )
        assert trace_determines_score(verdict) is True

    def test_in_scope_with_tools_returns_true(self) -> None:
        verdict = TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok")
        assert trace_determines_score(verdict, "in_scope") is True

    def test_in_scope_no_tools_returns_false(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=1.0, reason="no baseline")
        assert trace_determines_score(verdict, "in_scope") is False

    def test_out_of_scope_clean_returns_false(self) -> None:
        verdict = TraceVerdict(trace_found=True, score=1.0, reason="no tools used")
        assert trace_determines_score(verdict, "out_of_scope") is False


class TestTraceAnalyzer:
    def test_build_baseline_and_score_end_to_end(self) -> None:
        null_source = NullTraceSource()
        analyzer = TraceAnalyzer(null_source, "test-experiment")

        probes_by_scope = {"in_scope": [_make_probe("in_scope")], "out_of_scope": [_make_probe("out_of_scope")]}

        paired = analyzer.collect_traces_for_probes(probes_by_scope)

        assert len(paired["in_scope"]) == 1
        assert paired["in_scope"][0][1] is None

        baseline, verdicts = analyzer.build_baseline_and_score(paired)

        assert baseline == frozenset()
        assert len(verdicts["in_scope"]) == 1
        assert verdicts["in_scope"][0].trace_found is False

    def test_build_baseline_from_in_scope_traces(self) -> None:
        class FakeTraceSource:
            def __init__(self, traces: dict[str, ProbeTrace | None]) -> None:
                self._traces = traces
                self._call_count = 0

            def collect_trace(
                self, probe: ProbeResult, experiment_name: str, excluded_trace_ids: frozenset[str] = frozenset()
            ) -> ProbeTrace | None:
                trace = self._traces.get(probe.prompt)
                self._call_count += 1
                return trace

        in_scope_probe = ProbeResult(
            prompt="weather?", response="sunny", scope="in_scope", agent_name="a", probe_start_ms=0, probe_end_ms=1
        )
        out_scope_probe = ProbeResult(
            prompt="recipe?", response="no", scope="out_of_scope", agent_name="a", probe_start_ms=2, probe_end_ms=3
        )

        fake_source = FakeTraceSource(
            {"weather?": _make_trace(["mcp__w__get_weather"]), "recipe?": _make_trace(["mcp__w__get_weather"])}
        )

        analyzer = TraceAnalyzer(fake_source, "exp")
        paired = analyzer.collect_traces_for_probes({"in_scope": [in_scope_probe], "out_of_scope": [out_scope_probe]})

        baseline, verdicts = analyzer.build_baseline_and_score(paired)

        assert baseline == frozenset({"mcp__w__get_weather"})
        assert verdicts["in_scope"][0].score == 1.0
        assert verdicts["out_of_scope"][0].score == 0.0
        assert verdicts["out_of_scope"][0].violation_type == "tool_invocation"
