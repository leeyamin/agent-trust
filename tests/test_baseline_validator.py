import pytest

from agenttrust.traces.baseline_validator import calculate_baseline_compliance
from agenttrust.traces.eval_models import BaselineValidation, ToolValidation


def _make_validation(tools: list[ToolValidation]) -> BaselineValidation:
    return BaselineValidation(tools_evaluated=tools)


def _make_tool(name: str, verdict: str) -> ToolValidation:
    return ToolValidation(tool_name=name, verdict=verdict)


class TestCalculateBaselineCompliance:
    def test_no_tools_returns_one(self) -> None:
        validation = _make_validation([])
        assert calculate_baseline_compliance(validation) == 1.0

    def test_all_mapped_returns_one(self) -> None:
        tools = [_make_tool("t1", "MAPPED"), _make_tool("t2", "MAPPED")]
        validation = _make_validation(tools)
        assert calculate_baseline_compliance(validation) == 1.0

    def test_all_unmapped_returns_zero(self) -> None:
        tools = [_make_tool("t1", "UNMAPPED"), _make_tool("t2", "UNMAPPED")]
        validation = _make_validation(tools)
        assert calculate_baseline_compliance(validation) == 0.0

    def test_mixed_mapped_unmapped(self) -> None:
        tools = [_make_tool("t1", "MAPPED"), _make_tool("t2", "UNMAPPED")]
        validation = _make_validation(tools)
        assert calculate_baseline_compliance(validation) == pytest.approx(0.5)
