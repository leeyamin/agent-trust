import re
from typing import Any, Literal

import httpx2 as httpx
from pydantic import BaseModel

SCOPES = ["in_scope", "out_of_scope", "near_miss"]
AGENT_CARD_PATH = "/.well-known/agent-card.json"


def strip_markdown_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", cleaned)


async def fetch_agent_card(client: httpx.AsyncClient, agent_url: str) -> dict[str, Any]:
    resp = await client.get(f"{agent_url.rstrip('/')}{AGENT_CARD_PATH}")
    resp.raise_for_status()
    return resp.json()


class ProbeResult(BaseModel):
    prompt: str
    response: str
    scope: str
    agent_name: str
    probe_start_ms: int
    probe_end_ms: int
    outcome: Literal["response", "error", "timeout"] = "response"
