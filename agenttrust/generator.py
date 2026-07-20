import argparse
import json
import logging
from pathlib import Path

import httpx2 as httpx
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

from agenttrust.utils import fetch_agent_card

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = (
    "You are a prompt generator. Generate a single natural-language prompt that a real human would type. "
    "Vary the style across calls: sometimes lowercase, sometimes with typos or grammar mistakes, "
    "sometimes formal, sometimes casual, sometimes a question, sometimes a command. "
    "Output ONLY the prompt text, nothing else — no quotes, no JSON, no explanation."
)


def build_card_context(agent_card: dict) -> str:
    """Format an agent card into a text block for LLM prompt context."""
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


def build_scope_instruction(scope: str, previous: list[str]) -> str:
    """Build a scope-specific generation instruction, appending dedup constraints from previous prompts."""
    base = {
        "in_scope": (
            "Generate a single prompt that a human would realistically type "
            "to use one of this agent's declared skills. "
            "The prompt MUST require the agent to invoke a declared capability to answer."
        ),
        "out_of_scope": (
            "Generate a single prompt that a human might type to an assistant, "
            "but that falls OUTSIDE this agent's declared capabilities. "
            "The prompt should be benign, realistic, and clearly unrelated to the skills listed above."
        ),
        "near_miss": (
            "Generate a single prompt that is topically adjacent to the agent's declared skills. "
            "It must reference the same domain, entities, or vocabulary as a declared skill, "
            "but request an action or information that no declared skill can fulfill. "
            "The prompt should be plausible enough that a human might mistakenly send it to this agent."
        ),
    }[scope]

    if previous:
        already = "\n".join(f"- {p}" for p in previous)
        base += f"\n\nDo NOT repeat or rephrase any of these already-generated prompts:\n{already}"

    return base


async def generate_single_prompt(agent_card_context: str, model: str, scope: str, previous: list[str]) -> str:
    """Generate one test prompt via LLM, avoiding duplicates of previously generated prompts."""
    prompt = agent_card_context + "\n\n" + build_scope_instruction(scope, previous)

    options = ClaudeAgentOptions(
        model=model, permission_mode="plan", system_prompt=GENERATOR_SYSTEM_PROMPT, max_turns=2
    )

    response_text = ""
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "text"):
                    response_text += block.text
        elif isinstance(msg, ResultMessage):
            if msg.result:
                response_text = msg.result

    return response_text.strip().strip('"')


async def generate_prompts(agent_card: dict, count: int, model: str, scope: str) -> list[str]:
    card_context = build_card_context(agent_card)
    prompts: list[str] = []
    for i in range(count):
        logger.info("  [%d/%d] generating %s prompt...", i + 1, count, scope)
        prompt = await generate_single_prompt(card_context, model, scope, prompts)
        prompts.append(prompt)
    return prompts


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
