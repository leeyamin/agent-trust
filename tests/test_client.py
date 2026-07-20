import httpx2 as httpx
import pytest

from agenttrust.utils import fetch_agent_card


class TestFetchAgentCard:
    @pytest.mark.anyio
    async def test_returns_parsed_json(self) -> None:
        card = {"name": "Weather Agent", "description": "Weather", "skills": []}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/.well-known/agent-card.json"
            return httpx.Response(200, json=card)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_agent_card(client, "http://agent:8000")

        assert result == card
