import logging
import tempfile
from pathlib import Path

import mlflow

from agenttrust.traces.eval_models import CapabilityReport

logger = logging.getLogger(__name__)


class ReportUploadError(Exception):
    pass


def upload_report(report: CapabilityReport, experiment_name: str, tracking_uri: str | None = None) -> str:
    """Upload a capability report to MLflow."""
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix=f"{report.agent_name}_report_", delete=False
            ) as f:
                f.write(report.model_dump_json(indent=2))
                temp_path = Path(f.name)

            mlflow.log_artifact(str(temp_path))
            temp_path.unlink()

            mlflow.log_params(
                {
                    "evaluation_id": report.evaluation_id or "",
                    "agent_name": report.agent_name,
                    "trust_score": str(report.trust_score),
                    "alignment_passed": str(report.alignment_passed),
                    "evidence_mode": report.evidence_mode or "",
                    "schema_version": report.schema_version,
                }
            )

            run_id = run.info.run_id
            logger.info("Report uploaded to MLflow: experiment=%s run=%s", experiment_name, run_id)
            return f"mlflow://{experiment_name}/runs/{run_id}"

    except Exception as e:
        raise ReportUploadError(f"Failed to upload report to MLflow: {e}") from e
