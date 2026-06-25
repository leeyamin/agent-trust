from typing import Protocol

from agenttrust.models import ProbeResult
from agenttrust.traces.collector import DEFAULT_BUFFER_MS, collect_trace_for_probe
from agenttrust.traces.models import ProbeTrace


class TraceSource(Protocol):
    def collect_trace(
        self, probe: ProbeResult, experiment_name: str, excluded_trace_ids: frozenset[str] = frozenset()
    ) -> ProbeTrace | None: ...


class MlflowTraceSource:
    def __init__(self, buffer_ms: int = DEFAULT_BUFFER_MS) -> None:
        self.buffer_ms = buffer_ms

    def collect_trace(
        self, probe: ProbeResult, experiment_name: str, excluded_trace_ids: frozenset[str] = frozenset()
    ) -> ProbeTrace | None:
        return collect_trace_for_probe(
            probe, experiment_name, buffer_ms=self.buffer_ms, excluded_trace_ids=excluded_trace_ids
        )


class NullTraceSource:
    def collect_trace(
        self, probe: ProbeResult, experiment_name: str, excluded_trace_ids: frozenset[str] = frozenset()
    ) -> ProbeTrace | None:
        return None


def create_trace_source(source_type: str, buffer_ms: int = DEFAULT_BUFFER_MS) -> TraceSource:
    if source_type == "mlflow":
        return MlflowTraceSource(buffer_ms=buffer_ms)
    if source_type == "none":
        return NullTraceSource()
    raise ValueError(f"Unknown trace source type: {source_type}")
