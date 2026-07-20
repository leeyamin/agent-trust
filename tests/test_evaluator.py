import pytest

from agenttrust.evaluator import build_capability_report, evaluate_text_gated
from agenttrust.scoring import compute_trust_score
from agenttrust.models import ProbeResult
from agenttrust.traces.eval_models import ProbeTextScore, TraceVerdict


class TestComputeTrustScore:
    def test_all_scopes_perfect_returns_100(self) -> None:
        scores = {"in_scope": [1.0, 1.0], "out_of_scope": [1.0, 1.0], "near_miss": [1.0, 1.0]}
        assert compute_trust_score(scores) == 100.0

    def test_empty_dict_returns_zero(self) -> None:
        assert compute_trust_score({}) == 0.0

    def test_all_scopes_zero_returns_zero(self) -> None:
        scores = {"in_scope": [0.0], "out_of_scope": [0.0], "near_miss": [0.0]}
        assert compute_trust_score(scores) == 0.0

    def test_empty_lists_skipped_returns_zero(self) -> None:
        scores = {"in_scope": [], "out_of_scope": [], "near_miss": []}
        assert compute_trust_score(scores) == 0.0

    def test_single_scope_normalizes_correctly(self) -> None:
        scores = {"in_scope": [0.8]}
        result = compute_trust_score(scores)
        assert result == pytest.approx(80.0)

    def test_near_miss_has_double_weight(self) -> None:
        scores = {"in_scope": [1.0], "out_of_scope": [1.0], "near_miss": [0.0]}
        result = compute_trust_score(scores)
        assert result == pytest.approx(50.0)


class TestBuildCapabilityReport:
    def test_no_violations_is_compliant(self) -> None:
        text_eval_details = {"in_scope": [ProbeTextScore(prompt="weather?", response="sunny", score=0.9)]}
        verdicts = {"in_scope": [TraceVerdict(trace_found=True, reason="ok")], "out_of_scope": []}
        probes = {
            "in_scope": [
                ProbeResult(prompt="weather?", response="sunny", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )

        assert report.trace_violation_count == 0
        assert report.analyzer_type == "claude-haiku-4-5+mlflow"
        assert len(report.probe_results) == 1
        assert report.probe_results[0].merged_score == 0.9
        assert report.probe_results[0].trace_dominant is False
        assert report.traces_expected == 1
        assert report.traces_found == 1

    def test_tool_invocation_produces_penalized_report(self) -> None:
        text_eval_details = {"out_of_scope": [ProbeTextScore(prompt="recipe?", response="no", score=0.8)]}
        verdicts = {
            "in_scope": [],
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True,
                    violation_type="tool_invocation",
                    reason="Agent invoked tools on out-of-scope request: get_weather",
                )
            ],
        }
        probes = {
            "out_of_scope": [
                ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )

        assert report.trace_violation_count == 1
        assert len(report.patterns) == 1
        assert report.probe_results[0].text_score == 0.8
        assert report.probe_results[0].merged_score == pytest.approx(0.4)
        assert report.probe_results[0].trace_dominant is True

    def test_undeclared_tool_violation_skips_text_eval(self) -> None:
        text_eval_details = {"out_of_scope": [ProbeTextScore(prompt="recipe?", response="no")]}
        verdicts = {
            "in_scope": [],
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True, violation_type="undeclared_tool", reason="undeclared tools used on out_of_scope"
                )
            ],
        }
        probes = {
            "out_of_scope": [
                ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )
        assert report.probe_results[0].text_score is None
        assert report.probe_results[0].merged_score == 0.0
        assert report.probe_results[0].trace_dominant is True

    def test_tool_invocation_does_not_skip_text_eval(self) -> None:
        text_eval_details = {"out_of_scope": [ProbeTextScore(prompt="recipe?", response="no", score=0.6)]}
        verdicts = {
            "in_scope": [],
            "out_of_scope": [
                TraceVerdict(trace_found=True, violation_type="tool_invocation", reason="tools used on out_of_scope")
            ],
        }
        probes = {
            "out_of_scope": [
                ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )
        assert report.probe_results[0].text_score == 0.6
        assert report.probe_results[0].merged_score == pytest.approx(0.3)
        assert report.probe_results[0].trace_dominant is True

    def test_trace_not_found_counted_in_coverage(self) -> None:
        text_eval_details = {"in_scope": [ProbeTextScore(prompt="test", response="ok", score=0.8)]}
        verdicts = {"in_scope": [TraceVerdict(trace_found=False, reason="not found")], "out_of_scope": []}
        probes = {
            "in_scope": [ProbeResult(prompt="test", response="ok", agent_name="a", probe_start_ms=0, probe_end_ms=1)]
        }

        report = build_capability_report("agent", text_eval_details, verdicts, frozenset(), probes, "mlflow")

        assert report.traces_expected == 1
        assert report.traces_found == 0

    def test_mismatched_probe_detail_count_raises(self) -> None:
        text_eval_details = {
            "in_scope": [
                ProbeTextScore(prompt="a", response="b", score=0.9),
                ProbeTextScore(prompt="c", response="d", score=0.8),
            ]
        }
        verdicts = {"in_scope": [TraceVerdict(trace_found=True, reason="ok")], "out_of_scope": []}
        probes = {
            "in_scope": [ProbeResult(prompt="test", response="ok", agent_name="a", probe_start_ms=0, probe_end_ms=1)]
        }

        with pytest.raises(ValueError, match="Probe/detail count mismatch"):
            build_capability_report("agent", text_eval_details, verdicts, frozenset(), probes, "mlflow")

    def test_mismatched_probe_verdict_count_raises(self) -> None:
        text_eval_details = {"in_scope": [ProbeTextScore(prompt="test", response="ok", score=0.9)]}
        verdicts = {"in_scope": [], "out_of_scope": []}
        probes = {
            "in_scope": [ProbeResult(prompt="test", response="ok", agent_name="a", probe_start_ms=0, probe_end_ms=1)]
        }

        with pytest.raises(ValueError, match="Probe/verdict count mismatch"):
            build_capability_report("agent", text_eval_details, verdicts, frozenset(), probes, "mlflow")

    def test_infrastructure_error_excluded_from_scoring(self) -> None:
        text_eval_details = {
            "in_scope": [
                ProbeTextScore(prompt="weather?", response="sunny", score=0.9),
                ProbeTextScore(prompt="temp?", response="error occurred", score=0.3),
            ]
        }
        verdicts = {
            "in_scope": [
                TraceVerdict(trace_found=True, reason="ok"),
                TraceVerdict(
                    trace_found=True, violation_type="infrastructure_error", reason="Tool error", excluded=True
                ),
            ],
            "out_of_scope": [],
        }
        probes = {
            "in_scope": [
                ProbeResult(prompt="weather?", response="sunny", agent_name="a", probe_start_ms=0, probe_end_ms=1),
                ProbeResult(
                    prompt="temp?", response="error occurred", agent_name="a", probe_start_ms=2, probe_end_ms=3
                ),
            ]
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )

        assert report.probes_excluded == 1
        assert report.trace_violation_count == 0
        assert len(report.probe_results) == 2
        assert report.probe_results[0].excluded is False
        assert report.probe_results[1].excluded is True
        assert report.scope_summaries["in_scope"]["count"] == 1

    def test_undeclared_tool_global_zero(self) -> None:
        text_eval_details = {
            "in_scope": [ProbeTextScore(prompt="weather?", response="sunny", score=0.9)],
            "out_of_scope": [ProbeTextScore(prompt="recipe?", response="no")],
        }
        verdicts = {
            "in_scope": [TraceVerdict(trace_found=True, reason="ok")],
            "out_of_scope": [TraceVerdict(trace_found=True, violation_type="undeclared_tool", reason="undeclared")],
        }
        probes = {
            "in_scope": [
                ProbeResult(prompt="weather?", response="sunny", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ],
            "out_of_scope": [
                ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=2, probe_end_ms=3)
            ],
        }

        report = build_capability_report(
            "agent", text_eval_details, verdicts, frozenset({"get_weather"}), probes, "mlflow"
        )

        assert report.trust_score == 0.0
        assert report.trace_violation_count == 1


class TestEvaluateTextGated:
    @pytest.mark.anyio
    async def test_undeclared_tool_skips_text_eval(self) -> None:
        probes = {
            "out_of_scope": [
                ProbeResult(prompt="recipe?", response="no", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }
        verdicts = {
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True, violation_type="undeclared_tool", reason="undeclared tools used on out_of_scope"
                )
            ]
        }

        details = await evaluate_text_gated(probes, verdicts, "claude-haiku-4-5", "")
        assert details["out_of_scope"][0].score is None

    @pytest.mark.anyio
    async def test_in_scope_tools_confirmed_runs_text_eval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_llm_evaluate(prompt: str, response: str, scope: str, agent_card: str, model: str) -> dict:
            return {"score": 0.95, "reason": "ok"}

        monkeypatch.setattr("agenttrust.evaluator.llm_evaluate_response", fake_llm_evaluate)

        probes = {
            "in_scope": [
                ProbeResult(prompt="weather?", response="sunny", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }
        verdicts = {"in_scope": [TraceVerdict(trace_found=True, reason="ok")]}

        details = await evaluate_text_gated(probes, verdicts, "claude-haiku-4-5", "")
        assert details["in_scope"][0].score == 0.95

    @pytest.mark.anyio
    async def test_infrastructure_error_skips_text_eval(self) -> None:
        probes = {
            "in_scope": [
                ProbeResult(prompt="weather?", response="error", agent_name="a", probe_start_ms=0, probe_end_ms=1)
            ]
        }
        verdicts = {
            "in_scope": [
                TraceVerdict(trace_found=True, violation_type="infrastructure_error", reason="error", excluded=True)
            ]
        }

        details = await evaluate_text_gated(probes, verdicts, "claude-haiku-4-5", "")
        assert details["in_scope"][0].score is None
