from typing import Literal

from pydantic import BaseModel, Field

ViolationType = Literal["tool_invocation", "parametric_knowledge", "undeclared_tool", "infrastructure_error"]


class TraceVerdict(BaseModel):
    trace_found: bool
    violation_type: ViolationType | None = None
    reason: str
    excluded: bool = False


class ProbeScore(BaseModel):
    prompt: str
    response: str
    text_score: float | None = None
    trace_verdict: TraceVerdict | None = None
    merged_score: float
    trace_dominant: bool = False
    excluded: bool = False


class ToolValidation(BaseModel):
    tool_name: str
    verdict: Literal["MAPPED", "UNMAPPED"]


class BaselineValidation(BaseModel):
    tools_evaluated: list[ToolValidation] = Field(default_factory=list)


class ProbeTextScore(BaseModel):
    prompt: str
    response: str
    score: float | None = None


class CapabilityReport(BaseModel):
    summary: str
    trace_violation_count: int
    patterns: list[str] = Field(default_factory=list)
    timestamp: str
    run_id: str
    analyzer_type: str
    agent_name: str
    trust_score: float
    probes_excluded: int = 0
    traces_expected: int = 0
    traces_found: int = 0
    scope_summaries: dict[str, dict] = Field(default_factory=dict)
    probe_results: list[ProbeScore] = Field(default_factory=list)
    baseline_validation: BaselineValidation | None = None
