import json

import httpx2 as httpx
import pytest

from agenttrust.models import ProbeResult
from agenttrust.runner import load_prompts, save_results, send_message


class TestLoadPrompts:
    def test_reads_jsonl(self, tmp_path) -> None:
        f = tmp_path / "prompts.jsonl"
        f.write_text('{"prompt": "hello"}\n{"prompt": "world"}\n', encoding="utf-8")
        assert load_prompts(f) == ["hello", "world"]


class TestSaveResults:
    def test_writes_probe_results_as_jsonl(self, tmp_path) -> None:
        results = [ProbeResult(prompt="q", response="a", agent_name="test", probe_start_ms=0, probe_end_ms=1)]
        output = tmp_path / "results.jsonl"
        save_results(results, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["prompt"] == "q"
        assert parsed["response"] == "a"


class TestSendMessage:
    @pytest.mark.anyio
    async def test_parses_message_parts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"message": {"parts": [{"text": "sunny"}, {"text": " and warm"}]}},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await send_message(client, "http://agent", "weather?", "test")

        assert result.response == "sunny\n and warm"
        assert result.outcome == "response"

    @pytest.mark.anyio
    async def test_parses_artifact_parts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "task": {"artifacts": [{"parts": [{"text": "from artifact"}]}]},
                        "message": {"parts": []},
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await send_message(client, "http://agent", "test", "test")

        assert result.response == "from artifact"

    @pytest.mark.anyio
    async def test_empty_response_marks_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"message": {"parts": []}}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await send_message(client, "http://agent", "test", "test")

        assert result.outcome == "error"

    @pytest.mark.anyio
    async def test_no_result_key_marks_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"message": "fail"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await send_message(client, "http://agent", "test", "test")

        assert result.outcome == "error"
        assert "fail" in result.response
