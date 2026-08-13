import hashlib
import json
import re
from typing import Any

import httpx2 as httpx

AGENT_CARD_PATH = "/.well-known/agent-card.json"


async def fetch_agent_card(client: httpx.AsyncClient, agent_url: str) -> dict[str, Any]:
    """Fetch and parse the agent's card from its .well-known endpoint."""
    resp = await client.get(f"{agent_url.rstrip('/')}{AGENT_CARD_PATH}")
    resp.raise_for_status()
    return resp.json()


def strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences from LLM output."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", cleaned)


def compute_card_hash(card: dict[str, Any]) -> str:
    """Compute a sha256-prefixed digest of the canonical JSON representation of an agent card."""
    canonical = json.dumps(card, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
