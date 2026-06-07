"""Wikipedia lookup agent executor."""

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

SYSTEM_PROMPT = (
    "You are a Wikipedia lookup agent. "
    "Use the search_wikipedia tool to find relevant articles, "
    "then use the get_article_summary tool to retrieve details. "
    "Present the information in a clear, readable way, and cite the article title."
)

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
HTTP_HEADERS = {
    "User-Agent": "AgentTrust/1.0 (https://github.com/example/agent-trust; agent-trust@example.com)"
}


@tool(
    "search_wikipedia",
    "Search Wikipedia for articles matching a query. Returns a list of article titles.",
    {"query": str},
)
async def search_wikipedia(args: dict) -> dict:
    async with httpx.AsyncClient(headers=HTTP_HEADERS) as client:
        resp = await client.get(SEARCH_URL, params={
            "action": "opensearch",
            "search": args["query"],
            "limit": 5,
            "format": "json",
        })
        data = resp.json()
        titles = data[1] if len(data) > 1 else []

        if not titles:
            return {"content": [{"type": "text", "text": f"No articles found for: {args['query']}"}]}

        return {"content": [{"type": "text", "text": "\n".join(titles)}]}


@tool(
    "get_article_summary",
    "Get the summary of a Wikipedia article by its exact title.",
    {"title": str},
)
async def get_article_summary(args: dict) -> dict:
    title = args["title"].replace(" ", "_")
    async with httpx.AsyncClient(headers=HTTP_HEADERS) as client:
        resp = await client.get(f"{SUMMARY_URL}/{title}")

        if resp.status_code != 200:
            return {"content": [{"type": "text", "text": f"Article not found: {args['title']}"}]}

        data = resp.json()
        extract = data.get("extract", "")
        article_title = data.get("title", args["title"])
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"{article_title}\n\n{extract}"
        if url:
            result += f"\n\nSource: {url}"

        return {"content": [{"type": "text", "text": result}]}


wiki_server = create_sdk_mcp_server(
    name="wiki",
    version="1.0.0",
    tools=[search_wikipedia, get_article_summary],
)


class WikiAgentExecutor(AgentExecutor):
    def __init__(self, model: str) -> None:
        self.model = model

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text") and part.text:
                    prompt = part.text
                    break

        options = ClaudeAgentOptions(
            model=self.model,
            permission_mode="plan",
            system_prompt=SYSTEM_PROMPT,
            max_turns=5,
            mcp_servers={"wiki": wiki_server},
            allowed_tools=["mcp__wiki__search_wikipedia", "mcp__wiki__get_article_summary"],
        )

        response_text = ""
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text += block.text
            elif isinstance(msg, ResultMessage):
                if msg.result:
                    response_text = msg.result
        await event_queue.enqueue_event(
            Message(role=Role.ROLE_AGENT, parts=[Part(text=response_text)])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")
