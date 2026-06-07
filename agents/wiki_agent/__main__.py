"""Wikipedia agent server — A2A wiring, agent card, and startup."""

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
from agents.wiki_agent.agent_executor import WikiAgentExecutor

search_skill = AgentSkill(
    id="wikipedia_search",
    name="Wikipedia Search",
    description="Search for Wikipedia articles by keyword",
    tags=["wikipedia", "search", "knowledge"],
    examples=["Search for articles about quantum computing"],
)

summary_skill = AgentSkill(
    id="article_summary",
    name="Article Summary",
    description="Get a summary of a specific Wikipedia article",
    tags=["wikipedia", "summary", "knowledge"],
    examples=["Tell me about the history of the internet"],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia Agent Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_PORT", "8001")))
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"))
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Wikipedia Agent",
        description="A knowledge lookup agent that searches and summarizes Wikipedia articles.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=f"http://localhost:{args.port}/", protocol_binding="JSONRPC"),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[search_skill, summary_skill],
    )

    handler = DefaultRequestHandler(
        agent_executor=WikiAgentExecutor(args.model),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, "/"))

    app = Starlette(routes=routes)

    print(f"Starting Wikipedia agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
