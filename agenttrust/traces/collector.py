import logging
from typing import Literal, cast

import mlflow
from mlflow.entities import SpanType

from agenttrust.models import ProbeResult
from agenttrust.traces.trace_models import ProbeTrace, ToolCallSpan, TraceRetrievalResult

logger = logging.getLogger(__name__)


def collect_traces(experiment_name: str, start_time_ms: int, end_time_ms: int) -> TraceRetrievalResult:
    """Query MLflow for traces within a time window, extracting tool call spans."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("MLflow experiment not found: %s", experiment_name)
        return TraceRetrievalResult()

    filter_string = f"trace.timestamp_ms >= {start_time_ms} AND trace.timestamp_ms <= {end_time_ms}"

    traces = mlflow.search_traces(locations=[experiment.experiment_id], filter_string=filter_string, return_type="list")

    probes: list[ProbeTrace] = []

    for trace in traces:
        tool_span_type: SpanType = SpanType.TOOL  # type: ignore[assignment]
        tool_spans = trace.search_spans(span_type=tool_span_type)

        tool_calls: list[ToolCallSpan] = []
        for span in tool_spans:
            tool_call = ToolCallSpan(
                tool_name=span.name,
                inputs=span.inputs if isinstance(span.inputs, dict) else {},
                status=cast(Literal["OK", "ERROR", "UNSET"], span.status.status_code) if span.status else "UNSET",
            )
            tool_calls.append(tool_call)

        execution_ms = trace.info.execution_time_ms or 0

        probe = ProbeTrace(
            trace_id=trace.info.trace_id,
            tool_calls=tool_calls,
            total_duration_ms=execution_ms,
            timestamp_ms=trace.info.timestamp_ms or 0,
        )
        probes.append(probe)

    return TraceRetrievalResult(probes=probes)


def collect_trace_for_probe(
    probe: ProbeResult, experiment_name: str, buffer_ms: int = 5000, excluded_trace_ids: frozenset[str] = frozenset()
) -> ProbeTrace | None:
    """Find the best-matching trace for a probe by time window, selecting the longest if ambiguous."""
    probe_duration_ms = probe.probe_end_ms - probe.probe_start_ms
    effective_buffer = min(buffer_ms, max(1000, probe_duration_ms))

    result = collect_traces(
        experiment_name=experiment_name,
        start_time_ms=probe.probe_start_ms - effective_buffer,
        end_time_ms=probe.probe_end_ms + effective_buffer,
    )

    candidates = [p for p in result.probes if p.trace_id not in excluded_trace_ids]
    return select_best_trace(candidates, probe.probe_start_ms, probe.prompt)


def select_best_trace(candidates: list[ProbeTrace], probe_start_ms: int, prompt: str = "") -> ProbeTrace | None:
    if not candidates:
        logger.warning("No trace found for probe: %s", prompt[:60])
        return None

    if len(candidates) > 1:
        logger.warning(
            "Multiple traces (%d) matched probe: %s — selecting closest by timestamp", len(candidates), prompt[:60]
        )

    return min(candidates, key=lambda p: abs(p.timestamp_ms - probe_start_ms))
