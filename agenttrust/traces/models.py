from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallSpan(BaseModel):
    tool_name: str
    span_id: str
    parent_span_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    start_time: datetime
    end_time: datetime
    status: Literal["OK", "ERROR", "UNSET"] = Field(default="OK")


class ProbeTrace(BaseModel):
    trace_id: str
    agent_name: str
    tool_calls: list[ToolCallSpan] = Field(default_factory=list)
    start_time: datetime
    end_time: datetime
    total_duration_ms: int


class TraceRetrievalResult(BaseModel):
    probes: list[ProbeTrace] = Field(default_factory=list)
