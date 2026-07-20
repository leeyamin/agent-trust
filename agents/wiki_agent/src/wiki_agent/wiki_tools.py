import httpx2 as httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
HTTP_HEADERS = {"User-Agent": "AgentTrust/1.0 (https://github.com/example/agent-trust; agent-trust@example.com)"}

SYSTEM_PROMPT = (
    "You are a Wikipedia lookup agent. "
    "Use the search_wikipedia tool to find relevant articles, "
    "then use the get_article_summary tool to retrieve details. "
    "Present the information in a clear, readable way, and cite the article title."
)


@tool(
    "search_wikipedia",
    "Search Wikipedia for articles matching a query. Returns a list of article titles.",
    {"query": str},
)
async def search_wikipedia(args: dict) -> dict:
    async with httpx.AsyncClient(headers=HTTP_HEADERS) as client:
        resp = await client.get(
            SEARCH_URL, params={"action": "opensearch", "search": args["query"], "limit": 5, "format": "json"}
        )
        data = resp.json()
        titles = data[1] if len(data) > 1 else []

        if not titles:
            return {"content": [{"type": "text", "text": f"No articles found for: {args['query']}"}]}

        return {"content": [{"type": "text", "text": "\n".join(titles)}]}


@tool("get_article_summary", "Get the summary of a Wikipedia article by its exact title.", {"title": str})
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


wiki_server = create_sdk_mcp_server(name="wiki", version="1.0.0", tools=[search_wikipedia, get_article_summary])
