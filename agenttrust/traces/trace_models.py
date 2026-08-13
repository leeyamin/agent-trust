from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallSpan(BaseModel):
    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    status: Literal["OK", "ERROR", "UNSET"] = Field(default="OK")


class ProbeTrace(BaseModel):
    trace_id: str
    tool_calls: list[ToolCallSpan] = Field(default_factory=list)
    total_duration_ms: int
    timestamp_ms: int = 0


class TraceRetrievalResult(BaseModel):
    probes: list[ProbeTrace] = Field(default_factory=list)
