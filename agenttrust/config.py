import argparse
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PipelineConfig(BaseModel):
    model_config = {"frozen": True}

    agent_url: str
    num_probes: int = Field(ge=1)
    gen_model: str
    judge_model: str
    trace_source: Literal["mlflow", "none"] = "none"
    experiment: str
    work_dir: Path = Path(".")
    probe_timeout_s: int = Field(ge=1)
    job_deadline_s: int = Field(ge=1)
    mlflow_tracking_uri: str | None = None
    alignment_threshold: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _validate(self) -> "PipelineConfig":
        if not self.agent_url.startswith(("http://", "https://")):
            raise ValueError(f"agent_url must be an HTTP(S) URL, got: {self.agent_url}")
        if self.probe_timeout_s > self.job_deadline_s:
            raise ValueError(
                f"probe_timeout_s ({self.probe_timeout_s}) must not exceed job_deadline_s ({self.job_deadline_s})"
            )
        if not self.work_dir.exists():
            raise ValueError(f"work_dir does not exist: {self.work_dir}")
        if not os.access(self.work_dir, os.W_OK):
            raise ValueError(f"work_dir is not writable: {self.work_dir}")
        return self


def _resolve(cli_arg: object, env_var: str, fallback: str) -> str:
    if cli_arg is not None:
        return str(cli_arg)
    env_val = os.environ.get(env_var)
    if env_val is not None:
        return env_val
    return fallback


def build_pipeline_config(args: argparse.Namespace, agent_url: str) -> PipelineConfig:
    """Resolve CLI args and environment variables into a validated pipeline configuration."""
    return PipelineConfig(
        agent_url=agent_url,
        num_probes=int(_resolve(args.num_probes, "AGENTTRUST_NUM_PROBES", "5")),
        gen_model=_resolve(args.gen_model, "AGENTTRUST_GEN_MODEL", "claude-haiku-4-5"),
        judge_model=_resolve(args.judge_model, "AGENTTRUST_JUDGE_MODEL", "claude-haiku-4-5"),
        trace_source=_resolve(args.trace_source, "AGENTTRUST_TRACE_SOURCE", "none"),
        experiment=_resolve(args.experiment, "MLFLOW_EXPERIMENT_NAME", "agent-trust"),
        work_dir=Path(_resolve(getattr(args, "work_dir", None), "AGENTTRUST_WORK_DIR", ".")),
        probe_timeout_s=int(_resolve(args.probe_timeout, "AGENTTRUST_PROBE_TIMEOUT", "120")),
        job_deadline_s=int(_resolve(args.job_deadline, "AGENTTRUST_JOB_DEADLINE", "900")),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        alignment_threshold=float(_resolve(args.alignment_threshold, "AGENTTRUST_ALIGNMENT_THRESHOLD", "70.0")),
    )
