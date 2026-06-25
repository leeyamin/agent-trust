import argparse
import json
import os
from pathlib import Path

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from agents.wiki_agent.agent_executor import WikiAgentExecutor

CARD_PATH = Path(__file__).parent / "agent_card.json"


def load_agent_card(port: int) -> AgentCard:
    card_data = json.loads(CARD_PATH.read_text())
    card_data["supportedInterfaces"] = [{"url": f"http://localhost:{port}/", "protocolBinding": "JSONRPC"}]
    return ParseDict(card_data, AgentCard())


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia Agent Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_PORT", "8001")))
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"))
    args = parser.parse_args()

    agent_card = load_agent_card(args.port)

    handler = DefaultRequestHandler(
        agent_executor=WikiAgentExecutor(args.model), task_store=InMemoryTaskStore(), agent_card=agent_card
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, "/"))

    app = Starlette(routes=routes)

    print(f"Starting Wikipedia agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
