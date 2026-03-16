"""Load configuration from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
LANGSMITH_TRACING: bool = os.environ.get("LANGSMITH_TRACING", "false").lower() in (
    "true",
    "1",
)
LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "research-drafting-agent")
USE_ARXIV: bool = os.environ.get("USE_ARXIV", "true").lower() in ("true", "1")
DB_PATH: str = os.environ.get(
    "DB_PATH", str(Path(__file__).parent / "research_agent.db")
)
