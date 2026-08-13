from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agenttrust.traces.eval_models import CapabilityReport


class CompactResult(BaseModel):
    schema_version: str = Field(default="v1", alias="schemaVersion")
    evaluation_id: str = Field(alias="evaluationId")
    agent: str
    card_hash: str = Field(alias="cardHash")
    outcome: Literal["completed", "error"]
    alignment_passed: bool | None = Field(alias="alignmentPassed")
    score: int | None
    evidence_mode: str = Field(alias="evidenceMode")
    completed_at: str = Field(alias="completedAt")
    report_uri: str | None = Field(default=None, alias="reportURI")

    model_config = ConfigDict(populate_by_name=True)


def build_compact_result(
    report: CapabilityReport,
    evaluation_id: str,
    card_hash: str,
    alignment_threshold: float,
    evidence_mode: str = "text",
) -> CompactResult:
    """Build a compact result from a completed evaluation report."""
    return CompactResult(
        evaluation_id=evaluation_id,
        agent=report.agent_name,
        card_hash=card_hash,
        outcome="completed",
        alignment_passed=report.trust_score >= alignment_threshold,
        score=round(report.trust_score),
        evidence_mode=evidence_mode,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def build_error_result(
    evaluation_id: str, agent_name: str, card_hash: str, evidence_mode: str = "text"
) -> CompactResult:
    """Build a compact result for a failed evaluation run."""
    return CompactResult(
        evaluation_id=evaluation_id,
        agent=agent_name,
        card_hash=card_hash,
        outcome="error",
        alignment_passed=None,
        score=None,
        evidence_mode=evidence_mode,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
