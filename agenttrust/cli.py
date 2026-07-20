import argparse
import asyncio
import logging
import os
from pathlib import Path

from agenttrust.models import SCOPES


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")


def _get_agent_url(agent: str) -> str:
    if agent.startswith("http://") or agent.startswith("https://"):
        return agent
    config_path = Path("agents.yaml")
    if not config_path.exists():
        raise SystemExit(f"Agent '{agent}' is not a URL and no agents.yaml config found")
    import yaml

    agents_config: dict[str, str] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if agent not in agents_config:
        raise SystemExit(f"Agent '{agent}' not found in agents.yaml. Available: {list(agents_config.keys())}")
    return agents_config[agent]


async def _run_pipeline(args: argparse.Namespace) -> None:
    from agenttrust.evaluator import run_async
    from agenttrust.generator import run as generator_run
    from agenttrust.runner import run as runner_run

    agent_url = _get_agent_url(args.agent)

    for scope in SCOPES:
        gen_args = argparse.Namespace(
            agent_url=agent_url,
            count=args.num_probes,
            scope=scope,
            output_dir="generated_prompts",
            model=args.gen_model,
        )
        await generator_run(gen_args)

    run_args = argparse.Namespace(
        agent_url=agent_url, scope=None, prompts_dir="generated_prompts", output_dir="responses",
    )
    await runner_run(run_args)

    eval_args = argparse.Namespace(
        responses_dir="responses",
        output_dir="evaluations",
        llm_model=args.judge_model,
        trace_source=args.trace_source,
        experiment=args.experiment,
    )
    await run_async(eval_args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenttrust", description="Agent trust evaluation framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_default = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")

    gen = subparsers.add_parser("generate", help="Generate test prompts from an agent card")
    gen.add_argument("agent", help="Agent URL or name (resolved via agents.yaml)")
    gen.add_argument("--num-probes", type=int, default=20)
    gen.add_argument("--scope", choices=SCOPES, default="in_scope")
    gen.add_argument("--output-dir", default="generated_prompts")
    gen.add_argument("--gen-model", default=model_default)
    _add_common_args(gen)

    probe = subparsers.add_parser("probe", help="Send generated prompts to an agent and collect responses")
    probe.add_argument("agent", help="Agent URL or name (resolved via agents.yaml)")
    probe.add_argument("--scope", choices=SCOPES)
    probe.add_argument("--input-dir", default="generated_prompts")
    probe.add_argument("--output-dir", default="responses")
    _add_common_args(probe)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate agent responses for scope compliance")
    evaluate.add_argument("--responses-dir", default="responses")
    evaluate.add_argument("--output-dir", default="evaluations")
    evaluate.add_argument("--judge-model", default=model_default)
    evaluate.add_argument("--trace-source", choices=["mlflow", "none"], default="none")
    evaluate.add_argument("--experiment", default="agent-trust")
    _add_common_args(evaluate)

    pipeline = subparsers.add_parser("pipeline", help="Run full pipeline: generate, run, evaluate")
    pipeline.add_argument("agent", help="Agent URL or name (resolved via config)")
    pipeline.add_argument("--num-probes", type=int, default=5, help="Probes per scope (default: 5)")
    pipeline.add_argument("--gen-model", default=model_default, help="LLM for prompt generation")
    pipeline.add_argument("--judge-model", default=model_default, help="LLM for evaluation judge")
    pipeline.add_argument("--trace-source", choices=["mlflow", "none"], default="none")
    pipeline.add_argument("--experiment", default="agent-trust")
    _add_common_args(pipeline)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.command == "generate":
        from agenttrust.generator import run as generator_run

        args.agent_url = _get_agent_url(args.agent)
        args.count = args.num_probes
        args.model = args.gen_model
        asyncio.run(generator_run(args))

    elif args.command == "probe":
        from agenttrust.runner import run as runner_run

        args.agent_url = _get_agent_url(args.agent)
        args.prompts_dir = args.input_dir
        asyncio.run(runner_run(args))

    elif args.command == "evaluate":
        from agenttrust.evaluator import run_async

        args.llm_model = args.judge_model
        asyncio.run(run_async(args))

    elif args.command == "pipeline":
        asyncio.run(_run_pipeline(args))


if __name__ == "__main__":
    main()
