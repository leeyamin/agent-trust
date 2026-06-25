import logging
from datetime import datetime, timezone
from typing import Literal, cast

import mlflow
from mlflow.entities import SpanType

from agenttrust.models import ProbeResult
from agenttrust.traces.models import ProbeTrace, ToolCallSpan, TraceRetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_MS = 5000


def collect_traces(
    experiment_name: str, start_time_ms: int, end_time_ms: int, agent_name: str | None = None
) -> TraceRetrievalResult:
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
            start_ns = span.start_time_ns or 0
            end_ns = span.end_time_ns or 0
            tool_call = ToolCallSpan(
                tool_name=span.name,
                span_id=span.span_id,
                parent_span_id=span.parent_id,
                inputs=span.inputs if isinstance(span.inputs, dict) else {},
                outputs=span.outputs if isinstance(span.outputs, dict) else {},
                start_time=datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc),
                end_time=datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc),
                status=cast(Literal["OK", "ERROR", "UNSET"], span.status.status_code) if span.status else "UNSET",
            )
            tool_calls.append(tool_call)

        execution_ms = trace.info.execution_time_ms or 0
        timestamp_ms = trace.info.timestamp_ms or 0

        probe = ProbeTrace(
            trace_id=trace.info.trace_id,
            agent_name=agent_name or "unknown",
            tool_calls=tool_calls,
            start_time=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc),
            end_time=datetime.fromtimestamp((timestamp_ms + execution_ms) / 1000, tz=timezone.utc),
            total_duration_ms=execution_ms,
        )
        probes.append(probe)

    return TraceRetrievalResult(probes=probes)


def collect_trace_for_probe(
    probe: ProbeResult,
    experiment_name: str,
    buffer_ms: int = DEFAULT_BUFFER_MS,
    excluded_trace_ids: frozenset[str] = frozenset(),
) -> ProbeTrace | None:
    probe_duration_ms = probe.probe_end_ms - probe.probe_start_ms
    effective_buffer = min(buffer_ms, max(1000, probe_duration_ms))

    result = collect_traces(
        experiment_name=experiment_name,
        start_time_ms=probe.probe_start_ms - effective_buffer,
        end_time_ms=probe.probe_end_ms + effective_buffer,
        agent_name=probe.agent_name,
    )

    candidates = [p for p in result.probes if p.trace_id not in excluded_trace_ids]

    if not candidates:
        logger.warning("No trace found for probe: %s", probe.prompt[:60])
        return None

    if len(candidates) > 1:
        logger.warning("Multiple traces (%d) matched probe: %s — selecting longest", len(candidates), probe.prompt[:60])

    best = max(candidates, key=lambda p: p.total_duration_ms)
    return best
