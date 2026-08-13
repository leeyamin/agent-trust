import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agenttrust.models import SCOPES

EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_INFRA_ERROR = 2

logger = logging.getLogger(__name__)


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


async def _run_pipeline(args: argparse.Namespace) -> int:
    import httpx2 as httpx
    from pydantic import ValidationError

    from agenttrust.config import build_pipeline_config
    from agenttrust.evaluator import run_async, save_capability_report
    from agenttrust.generator import run as generator_run
    from agenttrust.report_upload import ReportUploadError, upload_report
    from agenttrust.compact_result import build_compact_result, build_error_result
    from agenttrust.runner import run as runner_run
    from agenttrust.utils import compute_card_hash, fetch_agent_card

    agent_ref = args.agent or os.environ.get("AGENT_URL")
    if not agent_ref:
        logger.error("No agent specified: pass as argument or set AGENT_URL")
        return EXIT_CONFIG_ERROR

    agent_url = _get_agent_url(agent_ref)

    try:
        config = build_pipeline_config(args, agent_url)
    except (ValueError, ValidationError) as e:
        logger.error("Configuration error: %s", e)
        return EXIT_CONFIG_ERROR

    evaluation_id = str(uuid4())
    evidence_mode = "text+trace" if config.trace_source == "mlflow" else "text"

    try:
        async with httpx.AsyncClient() as client:
            card = await fetch_agent_card(client, config.agent_url)
        card_hash = compute_card_hash(card)
        agent_name = card.get("name", args.agent).lower().replace(" ", "_")
    except Exception as e:
        logger.error("Failed to fetch agent card: %s", e)
        error_result = build_error_result(evaluation_id, agent_ref, "unknown", evidence_mode)
        print(error_result.model_dump_json(by_alias=True, indent=2))
        return EXIT_INFRA_ERROR

    started_at = datetime.now(timezone.utc).isoformat()
    work_dir = config.work_dir

    try:

        async def _execute() -> list:
            for scope in SCOPES:
                gen_args = argparse.Namespace(
                    agent_url=config.agent_url,
                    count=config.num_probes,
                    scope=scope,
                    output_dir=str(work_dir / "generated_prompts"),
                    model=config.gen_model,
                )
                await generator_run(gen_args)

            run_args = argparse.Namespace(
                agent_url=config.agent_url,
                scope=None,
                prompts_dir=str(work_dir / "generated_prompts"),
                output_dir=str(work_dir / "responses"),
                probe_timeout=config.probe_timeout_s,
            )
            await runner_run(run_args)

            if config.trace_source != "none":
                logger.info("Waiting for OTEL batch export flush before trace collection...")
                await asyncio.sleep(6)

            eval_args = argparse.Namespace(
                responses_dir=str(work_dir / "responses"),
                output_dir=str(work_dir / "evaluations"),
                llm_model=config.judge_model,
                trace_source=config.trace_source,
                experiment=config.experiment,
                evaluation_id=evaluation_id,
                card_hash=card_hash,
            )
            return await run_async(eval_args)

        reports = await asyncio.wait_for(_execute(), timeout=config.job_deadline_s)

    except asyncio.TimeoutError:
        logger.error("Pipeline exceeded deadline of %ds", config.job_deadline_s)
        error_result = build_error_result(evaluation_id, agent_name, card_hash, evidence_mode)
        print(error_result.model_dump_json(by_alias=True, indent=2))
        return EXIT_INFRA_ERROR
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        error_result = build_error_result(evaluation_id, agent_name, card_hash, evidence_mode)
        print(error_result.model_dump_json(by_alias=True, indent=2))
        return EXIT_INFRA_ERROR

    if not reports:
        logger.error("No evaluation reports produced")
        error_result = build_error_result(evaluation_id, agent_name, card_hash, evidence_mode)
        print(error_result.model_dump_json(by_alias=True, indent=2))
        return EXIT_INFRA_ERROR

    report = reports[0]

    completed_at = datetime.now(timezone.utc).isoformat()
    report = report.model_copy(
        update={
            "evidence_mode": evidence_mode,
            "judge_model": config.judge_model,
            "alignment_threshold": config.alignment_threshold,
            "alignment_passed": report.trust_score >= config.alignment_threshold,
            "started_at": started_at,
            "completed_at": completed_at,
        }
    )
    save_capability_report(report, work_dir / "evaluations")

    compact = build_compact_result(report, evaluation_id, card_hash, config.alignment_threshold, evidence_mode)

    if config.mlflow_tracking_uri:
        try:
            report_uri = upload_report(report, config.experiment, config.mlflow_tracking_uri)
            compact = compact.model_copy(update={"report_uri": report_uri})
        except ReportUploadError as e:
            logger.error("%s", e)
            error_result = build_error_result(evaluation_id, agent_name, card_hash, evidence_mode)
            print(error_result.model_dump_json(by_alias=True, indent=2))
            return EXIT_INFRA_ERROR

    print(compact.model_dump_json(by_alias=True, indent=2))
    return EXIT_SUCCESS


def main() -> None:
    parser = argparse.ArgumentParser(prog="agenttrust", description="Agent trust evaluation framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate test prompts from an agent card")
    gen.add_argument("agent", help="Agent URL or name (resolved via agents.yaml)")
    gen.add_argument("--num-probes", type=int, default=20)
    gen.add_argument("--scope", choices=SCOPES, default="in_scope")
    gen.add_argument("--output-dir", default="generated_prompts")
    gen.add_argument("--gen-model", default="claude-haiku-4-5")
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
    evaluate.add_argument("--judge-model", default="claude-haiku-4-5")
    evaluate.add_argument("--trace-source", choices=["mlflow", "none"], default="none")
    evaluate.add_argument("--experiment", default="agent-trust")
    _add_common_args(evaluate)

    pipeline = subparsers.add_parser("pipeline", help="Run full pipeline: generate, run, evaluate")
    pipeline.add_argument("agent", nargs="?", default=None, help="Agent URL or name (falls back to AGENT_URL env var)")
    pipeline.add_argument("--num-probes", type=int, default=None, help="Probes per scope (default: 5)")
    pipeline.add_argument("--gen-model", default=None, help="LLM for prompt generation (default: claude-haiku-4-5)")
    pipeline.add_argument("--judge-model", default=None, help="LLM for evaluation judge (default: claude-haiku-4-5)")
    pipeline.add_argument(
        "--trace-source", default=None, choices=["mlflow", "none"], help="Trace backend (default: none)"
    )
    pipeline.add_argument("--experiment", default=None, help="MLflow experiment name (default: agent-trust)")
    pipeline.add_argument("--work-dir", default=None, help="Working directory for pipeline output (default: .)")
    pipeline.add_argument("--probe-timeout", type=int, default=None, help="Per-probe timeout in seconds (default: 120)")
    pipeline.add_argument(
        "--job-deadline", type=int, default=None, help="Overall pipeline deadline in seconds (default: 900)"
    )
    pipeline.add_argument(
        "--alignment-threshold", type=float, default=None, help="Score threshold for alignment pass (default: 70.0)"
    )
    _add_common_args(pipeline)

    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG if args.verbose else logging.INFO)

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
        exit_code = asyncio.run(_run_pipeline(args))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
