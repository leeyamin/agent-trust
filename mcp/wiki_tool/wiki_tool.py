"Wikipedia MCP tool: search and summarize articles"

import json
import logging
import os
import sys

import httpx
import uvicorn
from fastmcp import FastMCP
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

mcp = FastMCP("Wiki")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format="%(levelname)s: %(message)s")

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
HTTP_HEADERS = {"User-Agent": "AgentTrust/1.0 (https://github.com/example/agent-trust; agent-trust@example.com)"}

_REQUEST_TIMEOUT = int(os.getenv("WIKI_REQUEST_TIMEOUT", "30"))


def setup_tracing() -> None:
    service_name = os.getenv("OTEL_SERVICE_NAME", "wiki-mcp-tool")

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()]))

    logger.info("Tracing initialized: service=%s", service_name)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia for articles matching a query. Returns a list of article titles."""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", "search_wikipedia")
    span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"query": query}))

    logger.debug("Searching Wikipedia for '%s'", query)

    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(
                SEARCH_URL, params={"action": "opensearch", "search": query, "limit": 5, "format": "json"}
            )
            resp.raise_for_status()
            data = resp.json()
            titles = data[1] if len(data) > 1 else []

        if not titles:
            result = f"No articles found for: {query}"
        else:
            result = "\n".join(titles)

        span.set_attribute("gen_ai.tool.call.result", result)
        span.set_status(Status(StatusCode.OK))
        return result

    except httpx.HTTPError as e:
        logger.warning("Wikipedia search error for '%s': %s", query, e)
        span.set_attribute("error.type", type(e).__name__)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        return f"Wikipedia search temporarily unavailable for {query}"


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def get_article_summary(title: str) -> str:
    """Get the summary of a Wikipedia article by its exact title."""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", "get_article_summary")
    span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"title": title}))

    logger.debug("Getting article summary for '%s'", title)
    url_title = title.replace(" ", "_")

    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{SUMMARY_URL}/{url_title}")

        if resp.status_code != 200:
            result = f"Article not found: {title}"
            span.set_attribute("gen_ai.tool.call.result", result)
            span.set_status(Status(StatusCode.OK))
            return result

        data = resp.json()
        extract_text = data.get("extract", "")
        article_title = data.get("title", title)
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"{article_title}\n\n{extract_text}"
        if page_url:
            result += f"\n\nSource: {page_url}"

        span.set_attribute("gen_ai.tool.call.result", result)
        span.set_status(Status(StatusCode.OK))
        return result

    except httpx.HTTPError as e:
        logger.warning("Wikipedia summary error for '%s': %s", title, e)
        span.set_attribute("error.type", type(e).__name__)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        return f"Wikipedia summary temporarily unavailable for {title}"


async def _trace_propagation_middleware(request, call_next):
    incoming_ctx = extract(dict(request.headers))
    token = otel_context.attach(incoming_ctx)
    try:
        return await call_next(request)
    finally:
        otel_context.detach(token)


def run_server() -> None:
    setup_tracing()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8005"))
    app = mcp.http_app(middleware=[Middleware(BaseHTTPMiddleware, dispatch=_trace_propagation_middleware)])
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
