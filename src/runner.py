"""Send generated prompts to an agent and collect responses."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx

SCOPES = ["in_scope", "out_of_scope", "near_miss"]
A2A_HEADERS = {"A2A-Version": "1.0"}
AGENT_CARD_PATH = "/.well-known/agent-card.json"


async def fetch_agent_card(client: httpx.AsyncClient, agent_url: str) -> dict:
    resp = await client.get(f"{agent_url.rstrip('/')}{AGENT_CARD_PATH}")
    resp.raise_for_status()
    return resp.json()


async def send_message(client: httpx.AsyncClient, agent_url: str, text: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [{"text": text}],
            }
        },
    }
    resp = await client.post(agent_url, json=payload, headers=A2A_HEADERS, timeout=120)
    data = resp.json()

    if "result" in data:
        return data["result"]["message"]["parts"][0]["text"]
    return json.dumps(data)


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompts.append(entry["prompt"])
    return prompts


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")


async def run(args: argparse.Namespace) -> None:
    scopes = [args.scope] if args.scope else SCOPES
    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)

    async with httpx.AsyncClient() as client:
        card = await fetch_agent_card(client, args.agent_url)
        agent_name = card["name"].lower().replace(" ", "_")
        print(f"Agent: {agent_name}")

        cards_dir = output_dir / "agent_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        with open(cards_dir / f"{agent_name}.json", "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

        for scope in scopes:
            prompt_file = prompts_dir / scope / f"{agent_name}.jsonl"
            if not prompt_file.exists():
                continue

            prompts = load_prompts(prompt_file)
            results = []

            print(f"{scope}: {len(prompts)} prompts")

            for i, prompt in enumerate(prompts, 1):
                print(f"  [{i}/{len(prompts)}] {prompt[:60]}...")
                response = await send_message(client, args.agent_url, prompt)
                results.append({"prompt": prompt, "response": response})

            output_path = output_dir / scope / f"{agent_name}.jsonl"
            save_results(results, output_path)
            print(f"  → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated prompts against an agent")
    parser.add_argument("agent_url", help="URL of the running agent")
    parser.add_argument("--scope", choices=SCOPES)
    parser.add_argument("--prompts-dir", default="generated_prompts")
    parser.add_argument("--output-dir", default="responses")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
