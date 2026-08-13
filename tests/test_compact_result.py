from agenttrust.compact_result import build_compact_result
from agenttrust.traces.eval_models import CapabilityReport


def _make_report(trust_score: float) -> CapabilityReport:
    return CapabilityReport(
        summary="test",
        trace_violation_count=0,
        timestamp="now",
        run_id="1",
        analyzer_type="test",
        agent_name="weather_agent",
        trust_score=trust_score,
    )


class TestBuildCompactResult:
    def test_aligned_at_exact_threshold(self) -> None:
        result = build_compact_result(_make_report(70.0), "eval-1", "sha256:abc", 70.0)
        assert result.alignment_passed is True
        assert result.outcome == "completed"
        assert result.score == 70

    def test_json_keys_are_camel_case(self) -> None:
        result = build_compact_result(_make_report(85.0), "eval-1", "sha256:abc", 70.0)
        json_str = result.model_dump_json(by_alias=True)
        assert "schemaVersion" in json_str
        assert "evaluationId" in json_str
        assert "cardHash" in json_str
        assert "alignmentPassed" in json_str
        assert "evidenceMode" in json_str
        assert "completedAt" in json_str
