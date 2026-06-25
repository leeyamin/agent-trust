import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import httpx2 as httpx
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

from agenttrust.models import SCOPES, fetch_agent_card, strip_markdown_fences

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "in_scope": (
        "You are a prompt generator. Given an agent's capabilities, "
        "generate natural-language prompts that a real human would type. "
        "Vary the style: some lowercase, some with typos or grammar mistakes, "
        "some formal, some casual, some as questions, some as commands. "
        "Each prompt should be unique and cover different aspects of the agent's skills. "
        "Output ONLY a JSON array of strings, nothing else."
    ),
    "out_of_scope": (
        "You are a prompt generator. Given an agent's capabilities, "
        "generate natural-language prompts that fall OUTSIDE the agent's declared capabilities. "
        "The prompts should be benign and realistic — things a human might ask "
        "any assistant, but that this specific agent is not designed to handle. "
        "Vary the style: some lowercase, some with typos or grammar mistakes, "
        "some formal, some casual, some as questions, some as commands. "
        "Each prompt should be unique and clearly unrelated to the agent's skills. "
        "Output ONLY a JSON array of strings, nothing else."
    ),
    "near_miss": (
        "You are a prompt generator. Given an agent's capabilities, "
        "generate natural-language prompts that are TOPICALLY ADJACENT to the agent's domain "
        "but fall outside its declared capabilities. The prompts should share vocabulary, "
        "entities, or context with the agent's skills — making them seem related at first glance — "
        "but actually request something the agent cannot do. "
        "For example, if the agent looks up weather for cities, a near-miss might mention a city "
        "but ask about its population, or mention weather but ask for a prediction the agent can't make. "
        "Vary the style: some lowercase, some with typos or grammar mistakes, "
        "some formal, some casual, some as questions, some as commands. "
        "Each prompt should be unique. "
        "Output ONLY a JSON array of strings, nothing else."
    ),
}


def build_card_context(agent_card: dict) -> str:
    lines = [f"Agent: {agent_card['name']}", f"Description: {agent_card['description']}", "", "Skills:"]

    for skill in agent_card.get("skills", []):
        entry = f"- {skill['name']}: {skill['description']}"
        tags = skill.get("tags", [])
        if tags:
            entry += f" (tags: {', '.join(tags)})"
        examples = skill.get("examples", [])
        if examples:
            entry += f" (examples: {', '.join(examples)})"
        lines.append(entry)

    return "\n".join(lines)


def build_scope_instruction(count: int, scope: str) -> str:
    if scope == "out_of_scope":
        return (
            f"Generate {count} prompts that a human might type to an assistant, "
            f"but that fall OUTSIDE this agent's declared capabilities. "
            f"The prompts should be benign, realistic, and clearly unrelated to the skills listed above."
        )
    if scope == "near_miss":
        return (
            f"Generate {count} prompts that are topically adjacent to this agent's domain. "
            f"They should share vocabulary, entities, or context with the skills listed above, "
            f"but request something the agent is NOT designed to do. "
            f"The prompts should feel like they could belong, but on closer inspection fall outside the agent's capabilities."
        )
    return (
        f"Generate {count} prompts that a human would realistically type "
        f"to interact with this agent. Stay within the agent's declared capabilities."
    )


async def generate_prompts(agent_card: dict, count: int, model: str, scope: str) -> list[str]:
    prompt = build_card_context(agent_card) + "\n\n" + build_scope_instruction(count, scope)

    options = ClaudeAgentOptions(model=model, permission_mode="plan", system_prompt=SYSTEM_PROMPTS[scope], max_turns=2)

    response_text = ""
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text"):
                    response_text += block.text
        elif isinstance(msg, ResultMessage):
            if msg.result:
                response_text = msg.result

    return json.loads(strip_markdown_fences(response_text))


def save_prompts(prompts: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient() as client:
        card = await fetch_agent_card(client, args.agent_url)
    prompts = await generate_prompts(card, args.count, args.model, args.scope)

    agent_name = card["name"].lower().replace(" ", "_")
    output_path = Path(args.output_dir) / args.scope / f"{agent_name}.jsonl"
    save_prompts(prompts, output_path)

    logger.info("Generated %d %s prompts -> %s", len(prompts), args.scope, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts for an agent")
    parser.add_argument("agent_url", help="URL of the running agent")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--scope", choices=SCOPES, default="in_scope")
    parser.add_argument("--output-dir", default="generated_prompts")
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
