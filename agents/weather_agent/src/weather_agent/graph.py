import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from weather_agent.configuration import Configuration

config = Configuration()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a weather reporting agent. "
    "Use the get_weather tool to fetch current weather data for any city the user asks about. "
    "Interpret the WMO weather code and present the weather in a clear, readable way."
)


class ExtendedMessagesState(MessagesState):
    final_answer: str = ""


def get_mcpclient() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {"weather": {"url": config.weather_mcp_url, "transport": config.mcp_transport, "timeout": config.mcp_timeout}}
    )


async def get_graph(client: MultiServerMCPClient) -> StateGraph:
    llm = ChatOpenAI(model=config.llm_model, api_key=config.llm_api_key, base_url=config.llm_api_base, temperature=0)

    try:
        tools = await client.get_tools()
        if tools:
            logger.info("Loaded %d MCP tool(s)", len(tools))
        else:
            logger.warning("No MCP tools available")
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:
        logger.warning("Failed to load MCP tools: %s", e)
        tools = []
        llm_with_tools = llm

    sys_msg = SystemMessage(content=SYSTEM_PROMPT)

    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        updated_state = {"messages": state["messages"] + [result]}
        if isinstance(result, AIMessage) and not result.tool_calls:
            updated_state["final_answer"] = result.content
        return updated_state

    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
