"""LLM: critique sources, summarize, draft (with streaming)."""

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from state import Source

CRITIQUE_PROMPT = """You are a critical analyst. Given the following search results for the topic "{topic}", write a short critique (2–4 sentences) covering:
- Relevance to the topic
- Possible bias or limitations
- Recency and reliability

If sources are clearly insufficient or off-topic, say so and suggest what would be needed.

Search results:
{sources_text}
"""

SUMMARIZE_PROMPT = """Using the search results and critique below, write a concise summary (bullet points or 2–3 short paragraphs) of the key points relevant to the topic "{topic}".

Critique:
{critique}

Search results:
{sources_text}
"""

DRAFT_PROMPT = """Write a short, engaging {output_format} post based on the summary below. Keep it concise and suitable for the chosen format. Do not use hashtags unless the format is LinkedIn (then 1–3 are fine).

Summary:
{summary}
"""


def _sources_to_text(sources: list[Source]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "")
        snippet = s.get("snippet", "")[:600]
        url = s.get("url", "")
        parts.append(f"[{i}] {title}\n{snippet}\nURL: {url}")
    return "\n\n".join(parts) if parts else "(No sources)"


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY or None,
        temperature=0.3,
    )


def critique_sources(topic: str, sources: list[Source]) -> str:
    """Return a short critique of the sources."""
    if not sources:
        return "No sources found. Consider refining the search query or adding more sources."
    sources_text = _sources_to_text(sources)
    llm = _get_llm()
    msg = HumanMessage(
        content=CRITIQUE_PROMPT.format(topic=topic, sources_text=sources_text)
    )
    response = llm.invoke([msg])
    return response.content if hasattr(response, "content") else str(response)


def summarize(topic: str, critique: str, sources: list[Source]) -> str:
    """Return a concise summary from critique and sources."""
    sources_text = _sources_to_text(sources)
    llm = _get_llm()
    msg = HumanMessage(
        content=SUMMARIZE_PROMPT.format(
            topic=topic, critique=critique, sources_text=sources_text
        )
    )
    response = llm.invoke([msg])
    return response.content if hasattr(response, "content") else str(response)


def draft_post(
    summary: str,
    output_format: str,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    """Draft a LinkedIn or email post; optionally stream tokens to callback."""
    output_format = output_format.lower() if output_format else "linkedin"
    if output_format not in ("linkedin", "email"):
        output_format = "linkedin"
    llm = _get_llm()
    msg = HumanMessage(
        content=DRAFT_PROMPT.format(output_format=output_format, summary=summary)
    )
    if stream_callback:
        acc: list[str] = []
        for chunk in llm.stream([msg]):
            if hasattr(chunk, "content") and chunk.content:
                acc.append(chunk.content)
                stream_callback(chunk.content)
        return "".join(acc)
    response = llm.invoke([msg])
    return response.content if hasattr(response, "content") else str(response)
