from typing import Literal

from pydantic import BaseModel

SCOPES = ["in_scope", "out_of_scope", "near_miss"]


class ProbeResult(BaseModel):
    prompt: str
    response: str
    agent_name: str
    probe_start_ms: int
    probe_end_ms: int
    outcome: Literal["response", "error", "timeout"] = "response"
