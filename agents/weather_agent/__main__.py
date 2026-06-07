"""Weather agent server — A2A wiring, agent card, and startup."""

import argparse
import os

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette
from agents.weather_agent.agent_executor import WeatherAgentExecutor

weather_skill = AgentSkill(
    id="weather_lookup",
    name="Weather Lookup",
    description="Look up current weather conditions for cities",
    tags=["weather", "temperature", "forecast"],
    examples=["What's the weather in Tel Aviv?"],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather Agent Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_PORT", "8000")))
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"))
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Weather Agent",
        description="A weather assistant that provides real-time weather information.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=f"http://localhost:{args.port}/", protocol_binding="JSONRPC"),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[weather_skill],
    )

    handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(args.model),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, "/"))

    app = Starlette(routes=routes)

    print(f"Starting weather agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
