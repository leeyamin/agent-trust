from unittest.mock import MagicMock

import pytest

from agenttrust.evaluator import (
    DEBERTA_LABELS,
    build_capability_report,
    compute_trust_score,
    deberta_score_entry,
    evaluate_text_gated,
)
from agenttrust.models import ProbeResult
from agenttrust.traces.eval_models import TraceVerdict


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


class TestDebertaScoreEntry:
    def test_in_scope_returns_compliance(self) -> None:
        scores = {"compliance": 0.9, "refusal": 0.1}
        assert deberta_score_entry(scores, "in_scope") == 0.9

    def test_out_of_scope_returns_refusal(self) -> None:
        scores = {"compliance": 0.9, "refusal": 0.1}
        assert deberta_score_entry(scores, "out_of_scope") == 0.1

    def test_near_miss_returns_refusal(self) -> None:
        scores = {"compliance": 0.9, "refusal": 0.1}
        assert deberta_score_entry(scores, "near_miss") == 0.1


class TestBuildCapabilityReport:
    def test_no_violations_is_compliant(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": 0.9}]}}
        verdicts = {
            "in_scope": [TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok")],
            "out_of_scope": [],
        }
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="sunny",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }

        report = build_capability_report(
            "agent", text_eval_data, verdicts, frozenset({"get_weather"}), probes, "deberta", "mlflow"
        )

        assert report.compliant is True
        assert report.violation_count == 0
        assert report.analyzer_type == "deberta+mlflow"
        assert len(report.probe_results) == 1
        assert report.probe_results[0].merged_score == 1.0
        assert report.probe_results[0].trace_dominant is True
        assert report.traces_expected == 1
        assert report.traces_found == 1

    def test_violation_produces_non_compliant_report(self) -> None:
        text_eval_data = {"details": {"out_of_scope": [{"score": None}]}}
        verdicts = {
            "in_scope": [],
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True,
                    tool_names=["get_weather"],
                    score=0.0,
                    violation_type="tool_invocation",
                    reason="Agent invoked tools on out-of-scope request: get_weather",
                )
            ],
        }
        probes = {
            "out_of_scope": [
                ProbeResult(
                    prompt="recipe?",
                    response="no",
                    scope="out_of_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }

        report = build_capability_report(
            "agent", text_eval_data, verdicts, frozenset({"get_weather"}), probes, "deberta", "mlflow"
        )

        assert report.compliant is False
        assert report.violation_count == 1
        assert len(report.patterns) == 1
        assert report.probe_results[0].text_score is None
        assert report.probe_results[0].merged_score == 0.0
        assert report.probe_results[0].trace_dominant is True

    def test_trace_violation_skips_text_eval(self) -> None:
        text_eval_data = {"details": {"out_of_scope": [{"score": None}]}}
        verdicts = {
            "in_scope": [],
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True,
                    tool_names=["get_weather"],
                    score=0.0,
                    violation_type="tool_invocation",
                    reason="tools used on out_of_scope",
                )
            ],
        }
        probes = {
            "out_of_scope": [
                ProbeResult(
                    prompt="recipe?",
                    response="no",
                    scope="out_of_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }

        report = build_capability_report(
            "agent", text_eval_data, verdicts, frozenset({"get_weather"}), probes, "deberta", "mlflow"
        )
        assert report.probe_results[0].text_score is None
        assert report.probe_results[0].merged_score == 0.0
        assert report.probe_results[0].trace_dominant is True

    def test_in_scope_tools_confirmed_skips_text_eval(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": None}]}}
        verdicts = {
            "in_scope": [TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok")],
            "out_of_scope": [],
        }
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="sunny",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }

        report = build_capability_report(
            "agent", text_eval_data, verdicts, frozenset({"get_weather"}), probes, "deberta", "mlflow"
        )
        assert report.probe_results[0].text_score is None
        assert report.probe_results[0].merged_score == 1.0
        assert report.probe_results[0].trace_dominant is True

    def test_trace_not_found_counted_in_coverage(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": 0.8}]}}
        verdicts = {"in_scope": [TraceVerdict(trace_found=False, score=0.0, reason="not found")], "out_of_scope": []}
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="test", response="ok", scope="in_scope", agent_name="a", probe_start_ms=0, probe_end_ms=1
                )
            ]
        }

        report = build_capability_report("agent", text_eval_data, verdicts, frozenset(), probes, "deberta", "mlflow")

        assert report.traces_expected == 1
        assert report.traces_found == 0

    def test_mismatched_probe_detail_count_raises(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": 0.9}, {"score": 0.8}]}}
        verdicts = {"in_scope": [TraceVerdict(trace_found=True, score=1.0, reason="ok")], "out_of_scope": []}
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="test", response="ok", scope="in_scope", agent_name="a", probe_start_ms=0, probe_end_ms=1
                )
            ]
        }

        with pytest.raises(ValueError, match="Probe/detail count mismatch"):
            build_capability_report("agent", text_eval_data, verdicts, frozenset(), probes, "deberta", "mlflow")

    def test_mismatched_probe_verdict_count_raises(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": 0.9}]}}
        verdicts = {"in_scope": [], "out_of_scope": []}
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="test", response="ok", scope="in_scope", agent_name="a", probe_start_ms=0, probe_end_ms=1
                )
            ]
        }

        with pytest.raises(ValueError, match="Probe/verdict count mismatch"):
            build_capability_report("agent", text_eval_data, verdicts, frozenset(), probes, "deberta", "mlflow")

    def test_infrastructure_error_excluded_from_scoring(self) -> None:
        text_eval_data = {"details": {"in_scope": [{"score": 0.9}, {"score": 0.3}]}}
        verdicts = {
            "in_scope": [
                TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok"),
                TraceVerdict(
                    trace_found=True,
                    tool_names=["get_weather"],
                    score=0.0,
                    violation_type="infrastructure_error",
                    reason="Tool error",
                    excluded=True,
                ),
            ],
            "out_of_scope": [],
        }
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="sunny",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                ),
                ProbeResult(
                    prompt="temp?",
                    response="error occurred",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=2,
                    probe_end_ms=3,
                ),
            ]
        }

        report = build_capability_report(
            "agent", text_eval_data, verdicts, frozenset({"get_weather"}), probes, "deberta", "mlflow"
        )

        assert report.probes_excluded == 1
        assert report.violation_count == 0
        assert len(report.probe_results) == 2
        assert report.probe_results[0].excluded is False
        assert report.probe_results[1].excluded is True
        assert report.scope_summaries["in_scope"]["count"] == 1

    def test_none_trace_source_type_sets_analyzer_to_method_only(self) -> None:
        report = build_capability_report(
            "a", {"details": {}}, {"in_scope": [], "out_of_scope": []}, frozenset(), {}, "deberta", "none"
        )
        assert report.analyzer_type == "deberta"


class TestEvaluateTextGated:
    def test_violation_skips_text_eval(self) -> None:
        probes = {
            "out_of_scope": [
                ProbeResult(
                    prompt="recipe?",
                    response="no",
                    scope="out_of_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }
        verdicts = {
            "out_of_scope": [
                TraceVerdict(
                    trace_found=True,
                    tool_names=["get_weather"],
                    score=0.0,
                    violation_type="tool_invocation",
                    reason="tools used on out_of_scope",
                )
            ]
        }

        details = evaluate_text_gated(probes, verdicts, "deberta")
        assert details["out_of_scope"][0]["score"] is None

    def test_in_scope_tools_confirmed_skips_text_eval(self) -> None:
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="sunny",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }
        verdicts = {"in_scope": [TraceVerdict(trace_found=True, tool_names=["get_weather"], score=1.0, reason="ok")]}

        details = evaluate_text_gated(probes, verdicts, "deberta")
        assert details["in_scope"][0]["score"] is None

    def test_clean_trace_runs_text_eval(self) -> None:
        mock_classifier = MagicMock()
        mock_classifier.return_value = {"labels": [DEBERTA_LABELS[0], DEBERTA_LABELS[1]], "scores": [0.3, 0.7]}
        probes = {
            "out_of_scope": [
                ProbeResult(
                    prompt="recipe?",
                    response="I cannot help",
                    scope="out_of_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }
        verdicts = {"out_of_scope": [TraceVerdict(trace_found=True, score=1.0, reason="no tools used")]}

        details = evaluate_text_gated(probes, verdicts, "deberta", classifier=mock_classifier)
        assert details["out_of_scope"][0]["score"] == pytest.approx(0.7, abs=0.001)
        mock_classifier.assert_called_once()

    def test_excluded_outcome_not_sent_to_classifier(self) -> None:
        mock_classifier = MagicMock()
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                    outcome="timeout",
                )
            ]
        }
        verdicts = {"in_scope": [TraceVerdict(trace_found=False, score=0.0, reason="not found")]}

        details = evaluate_text_gated(probes, verdicts, "deberta", classifier=mock_classifier)
        assert details["in_scope"][0]["score"] == 0.0
        assert details["in_scope"][0]["excluded"] is True
        mock_classifier.assert_not_called()

    def test_infrastructure_error_skips_text_eval(self) -> None:
        probes = {
            "in_scope": [
                ProbeResult(
                    prompt="weather?",
                    response="error",
                    scope="in_scope",
                    agent_name="a",
                    probe_start_ms=0,
                    probe_end_ms=1,
                )
            ]
        }
        verdicts = {
            "in_scope": [
                TraceVerdict(
                    trace_found=True,
                    tool_names=["get_weather"],
                    score=0.0,
                    violation_type="infrastructure_error",
                    reason="error",
                    excluded=True,
                )
            ]
        }

        details = evaluate_text_gated(probes, verdicts, "deberta")
        assert details["in_scope"][0]["score"] is None
