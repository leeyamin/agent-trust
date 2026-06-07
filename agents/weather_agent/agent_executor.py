"""Weather agent executor."""

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
    "You are a weather reporting agent. "
    "Use the get_weather tool to fetch current weather data for any city the user asks about. "
    "Interpret the WMO weather code and present the weather in a clear, readable way."
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@tool(
    "get_weather",
    "Get current weather for a city. Returns temperature, WMO weather code, and wind speed.",
    {"city": str},
)
async def get_weather(args: dict) -> dict:
    city = args["city"]
    async with httpx.AsyncClient() as client:
        geo_resp = await client.get(GEOCODING_URL, params={"name": city, "count": 1})
        geo_data = geo_resp.json()

        results = geo_data.get("results")
        if not results:
            return {"content": [{"type": "text", "text": f"City not found: {city}"}]}

        location = results[0]
        lat, lon = location["latitude"], location["longitude"]
        name = location.get("name", city)
        country = location.get("country", "")

        weather_resp = await client.get(FORECAST_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        })
        weather_data = weather_resp.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        wind = current.get("wind_speed_10m")

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{name}, {country}: {temp}°F, WMO code {code}, wind {wind} mph",
                }
            ]
        }


weather_server = create_sdk_mcp_server(
    name="weather",
    version="1.0.0",
    tools=[get_weather],
)


class WeatherAgentExecutor(AgentExecutor):
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
            mcp_servers={"weather": weather_server},
            allowed_tools=["mcp__weather__get_weather"],
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
