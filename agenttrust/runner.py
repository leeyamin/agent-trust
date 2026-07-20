import argparse
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import httpx2 as httpx

from agenttrust.utils import fetch_agent_card
from agenttrust.models import SCOPES, ProbeResult

logger = logging.getLogger(__name__)

A2A_HEADERS = {"A2A-Version": "1.0"}


async def send_message(client: httpx.AsyncClient, agent_url: str, text: str, agent_name: str) -> ProbeResult:
    """Send a prompt to the agent via A2A JSON-RPC and parse the response parts with timing."""
    probe_start_ms = int(time.time() * 1000)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": {"messageId": str(uuid4()), "role": "ROLE_USER", "parts": [{"text": text}]}},
    }
    resp = await client.post(agent_url, json=payload, headers=A2A_HEADERS, timeout=120)
    data = resp.json()

    outcome: str = "response"

    if "result" in data:
        result = data["result"]
        parts = result.get("message", {}).get("parts", [])
        if not parts:
            task = result.get("task", result)
            for artifact in task.get("artifacts", []):
                parts.extend(artifact.get("parts", []))
        text_parts = [p["text"] for p in parts if "text" in p]
        response_text = "\n".join(text_parts) if text_parts else ""
        if not response_text:
            outcome = "error"
    else:
        response_text = json.dumps(data)
        outcome = "error"

    probe_end_ms = int(time.time() * 1000)

    return ProbeResult(
        prompt=text,
        response=response_text,
        agent_name=agent_name,
        probe_start_ms=probe_start_ms,
        probe_end_ms=probe_end_ms,
        outcome=outcome,
    )


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompts.append(entry["prompt"])
    return prompts


def save_results(results: list[ProbeResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(entry.model_dump_json() + "\n")


async def run(args: argparse.Namespace) -> None:
    scopes = [args.scope] if args.scope else SCOPES
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)

    async with httpx.AsyncClient() as client:
        card = await fetch_agent_card(client, args.agent_url)
        agent_name = card["name"].lower().replace(" ", "_")
        logger.info("Agent: %s", agent_name)

        cards_dir = output_dir / "agent_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        with open(cards_dir / f"{agent_name}.json", "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

        for scope in scopes:
            prompt_file = prompts_dir / scope / f"{agent_name}.jsonl"
            if not prompt_file.exists():
                continue

            prompts = load_prompts(prompt_file)
            results: list[ProbeResult] = []

            logger.info("%s: %d prompts", scope, len(prompts))

            for i, prompt in enumerate(prompts, 1):
                logger.info("  [%d/%d] %s...", i, len(prompts), prompt[:60])
                try:
                    result = await send_message(client, args.agent_url, prompt, agent_name)
                except httpx.ReadTimeout:
                    logger.warning("Timeout on prompt %d, recording empty response", i)
                    result = ProbeResult(
                        prompt=prompt,
                        response="",
                        agent_name=agent_name,
                        probe_start_ms=int(time.time() * 1000),
                        probe_end_ms=int(time.time() * 1000),
                        outcome="timeout",
                    )
                results.append(result)

            output_path = output_dir / scope / f"{agent_name}.jsonl"
            save_results(results, output_path)
            logger.info("  -> %s", output_path)
