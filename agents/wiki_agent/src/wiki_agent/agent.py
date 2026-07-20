import logging
import os

import uvicorn
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Message, Part, Role

from wiki_agent.configuration import Configuration
from wiki_agent.graph import get_graph, get_mcpclient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LangChainInstrumentor().instrument()


def get_agent_card(host: str, port: int) -> AgentCard:
    capabilities = AgentCapabilities(streaming=True)
    skills = [
        AgentSkill(
            id="wikipedia_search",
            name="Wikipedia Search",
            description="Search for Wikipedia articles by keyword",
            tags=["wikipedia", "search", "knowledge"],
            examples=["Search for articles about quantum computing"],
        ),
        AgentSkill(
            id="article_summary",
            name="Article Summary",
            description="Get a summary of a specific Wikipedia article",
            tags=["wikipedia", "summary", "knowledge"],
            examples=["Tell me about the history of the internet"],
        ),
    ]
    return AgentCard(
        name="Wikipedia Agent",
        description="A knowledge lookup agent that searches and summarizes Wikipedia articles.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=capabilities,
        skills=skills,
        supported_interfaces=[
            AgentInterface(
                url=os.getenv("AGENT_ENDPOINT", f"http://{host}:{port}").rstrip("/") + "/", protocol_binding="JSONRPC"
            )
        ],
    )


class WikiAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.config = Configuration()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = context.get_user_input()
        messages = [HumanMessage(content=prompt)]

        mcpclient = get_mcpclient()
        graph = await get_graph(mcpclient)

        output = None
        async for event in graph.astream({"messages": messages}, stream_mode="updates"):
            output = event

        final_answer = output.get("assistant", {}).get("final_answer") if output else None
        response_text = str(final_answer) if final_answer else "No response generated."

        await event_queue.enqueue_event(Message(role=Role.ROLE_AGENT, parts=[Part(text=response_text)]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise RuntimeError("WikiAgentExecutor does not support cancellation")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    agent_card = get_agent_card(host, port)

    request_handler = DefaultRequestHandler(
        agent_executor=WikiAgentExecutor(), task_store=InMemoryTaskStore(), agent_card=agent_card
    )

    routes = [Route("/health", health, methods=["GET"])]
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True))
    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)
