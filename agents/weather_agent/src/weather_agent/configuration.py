from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    llm_model: str = "llama3.2:latest"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"
    weather_mcp_url: str = "http://localhost:8000/mcp"
    mcp_transport: str = "streamable_http"
    mcp_timeout: int = 600
