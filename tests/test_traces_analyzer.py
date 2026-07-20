import pytest

from agenttrust.models import ProbeResult
from agenttrust.scoring import merge_text_and_trace_scores, trace_determines_score
from agenttrust.traces.analyzer import TraceAnalyzer, score_trace
from agenttrust.traces.eval_models import TraceVerdict
from agenttrust.traces.trace_models import ProbeTrace, ToolCallSpan
from agenttrust.traces.source import NullTraceSource


def _make_span(tool_name: str, status: str = "OK") -> ToolCallSpan:
    return ToolCallSpan(tool_name=tool_name, status=status)


def _make_trace(tool_names: list[str], *, statuses: list[str] | None = None) -> ProbeTrace:
    if statuses is None:
        spans = [_make_span(name) for name in tool_names]
    else:
        spans = [_make_span(name, status=s) for name, s in zip(tool_names, statuses)]
    return ProbeTrace(trace_id="tr_1", tool_calls=spans, total_duration_ms=1000)


def _make_probe() -> ProbeResult:
    return ProbeResult(prompt="test", response="ok", agent_name="agent", probe_start_ms=1000, probe_end_ms=2000)


class TestScoreTrace:
    def test_none_trace_returns_not_found_verdict(self) -> None:
        verdict = score_trace(None, "in_scope", frozenset())
        assert verdict.trace_found is False
        assert verdict.violation_type is None

    def test_in_scope_with_tools_in_baseline_compliant(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type is None

    def test_in_scope_zero_tools_parametric_knowledge_violation(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type == "parametric_knowledge"

    def test_in_scope_empty_baseline_skips_parametric_knowledge(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "in_scope", frozenset())
        assert verdict.violation_type is None

    def test_in_scope_undeclared_tools_violation(self) -> None:
        trace = _make_trace(["mcp__w__get_weather", "mcp__w__send_email"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type == "undeclared_tool"

    def test_out_of_scope_zero_tools_compliant(self) -> None:
        trace = _make_trace([])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type is None

    def test_out_of_scope_with_baseline_tools_violation(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type == "tool_invocation"

    def test_out_of_scope_non_baseline_tools_undeclared_violation(self) -> None:
        trace = _make_trace(["WebFetch", "Agent"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type == "undeclared_tool"

    def test_out_of_scope_mixed_declared_undeclared_is_undeclared(self) -> None:
        trace = _make_trace(["mcp__w__get_weather", "send_email"])
        verdict = score_trace(trace, "out_of_scope", frozenset({"mcp__w__get_weather"}))
        assert verdict.violation_type == "undeclared_tool"

    def test_in_scope_tool_error_excluded(self) -> None:
        trace = _make_trace(["mcp__w__get_weather"], statuses=["ERROR"])
        verdict = score_trace(trace, "in_scope", frozenset({"mcp__w__get_weather"}))
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
        verdict = TraceVerdict(trace_found=False, reason="not found")
        score, dominant, excluded = merge_text_and_trace_scores(0.8, verdict)
        assert score == 0.8
        assert dominant is False
        assert excluded is False

    def test_undeclared_tool_overrides_to_zero(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="undeclared_tool", reason="violation")
        score, dominant, excluded = merge_text_and_trace_scores(0.95, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is False

    def test_tool_invocation_applies_penalty(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="tool_invocation", reason="violation")
        score, dominant, excluded = merge_text_and_trace_scores(0.9, verdict)
        assert score == pytest.approx(0.45)
        assert dominant is True
        assert excluded is False

    def test_tool_invocation_none_text_defaults_zero(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="tool_invocation", reason="violation")
        score, dominant, excluded = merge_text_and_trace_scores(None, verdict)
        assert score == 0.0
        assert dominant is True

    def test_clean_trace_preserves_text_score_out_of_scope(self) -> None:
        verdict = TraceVerdict(trace_found=True, reason="clean")
        score, dominant, excluded = merge_text_and_trace_scores(0.7, verdict)
        assert score == 0.7
        assert dominant is False
        assert excluded is False

    def test_excluded_verdict_returns_excluded(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="infrastructure_error", reason="error", excluded=True)
        score, dominant, excluded = merge_text_and_trace_scores(0.9, verdict)
        assert score == 0.0
        assert dominant is True
        assert excluded is True

    def test_none_text_score_no_verdict_defaults_to_zero(self) -> None:
        score, dominant, excluded = merge_text_and_trace_scores(None, None)
        assert score == 0.0
        assert dominant is False
        assert excluded is False


class TestTraceDeterminesScore:
    def test_no_verdict_returns_false(self) -> None:
        assert trace_determines_score(None) is False

    def test_trace_not_found_returns_false(self) -> None:
        verdict = TraceVerdict(trace_found=False, reason="not found")
        assert trace_determines_score(verdict) is False

    def test_undeclared_tool_violation_returns_true(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="undeclared_tool", reason="violation")
        assert trace_determines_score(verdict) is True

    def test_tool_invocation_violation_returns_false(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="tool_invocation", reason="violation")
        assert trace_determines_score(verdict) is False

    def test_excluded_returns_true(self) -> None:
        verdict = TraceVerdict(trace_found=True, violation_type="infrastructure_error", reason="error", excluded=True)
        assert trace_determines_score(verdict) is True


class TestTraceAnalyzer:
    def test_discover_and_score_end_to_end(self) -> None:
        null_source = NullTraceSource()
        analyzer = TraceAnalyzer(null_source, "test-experiment")

        probes_by_scope = {"in_scope": [_make_probe()], "out_of_scope": [_make_probe()]}

        paired = analyzer.collect_traces_for_probes(probes_by_scope)

        assert len(paired["in_scope"]) == 1
        assert paired["in_scope"][0][1] is None

        observed = analyzer.discover_observed_tools(paired)
        assert observed == frozenset()

        verdicts = analyzer.score_probes(paired, frozenset())
        assert len(verdicts["in_scope"]) == 1
        assert verdicts["in_scope"][0].trace_found is False

    def test_discover_tools_and_score_with_allowlist(self) -> None:
        class FakeTraceSource:
            def __init__(self, traces: dict[str, ProbeTrace | None]) -> None:
                self._traces = traces

            def collect_trace(
                self, probe: ProbeResult, experiment_name: str, excluded_trace_ids: frozenset[str] = frozenset()
            ) -> ProbeTrace | None:
                return self._traces.get(probe.prompt)

        in_scope_probe = ProbeResult(
            prompt="weather?", response="sunny", agent_name="a", probe_start_ms=0, probe_end_ms=1
        )
        out_scope_probe = ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=2, probe_end_ms=3)

        fake_source = FakeTraceSource(
            {"weather?": _make_trace(["mcp__w__get_weather"]), "recipe?": _make_trace(["mcp__w__get_weather"])}
        )

        analyzer = TraceAnalyzer(fake_source, "exp")
        paired = analyzer.collect_traces_for_probes({"in_scope": [in_scope_probe], "out_of_scope": [out_scope_probe]})

        observed = analyzer.discover_observed_tools(paired)
        assert observed == frozenset({"mcp__w__get_weather"})

        allowed = frozenset({"mcp__w__get_weather"})
        verdicts = analyzer.score_probes(paired, allowed)

        assert verdicts["in_scope"][0].violation_type is None
        assert verdicts["out_of_scope"][0].violation_type == "tool_invocation"
