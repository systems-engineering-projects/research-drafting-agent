"""Search: Tavily, Wikipedia, optional arXiv. Returns unified list of sources."""

import logging
from typing import Any

from state import Source

from config import TAVILY_API_KEY, USE_ARXIV

logger = logging.getLogger(__name__)


def _tavily_sources(query: str, max_results: int = 8) -> list[Source]:
    sources: list[Source] = []
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set; skipping Tavily search")
        return sources
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        for r in getattr(response, "results", []) or []:
            title = getattr(r, "title", "") or ""
            content = getattr(r, "content", "") or getattr(r, "snippet", "") or ""
            url = getattr(r, "url", "") or ""
            sources.append({"title": title, "snippet": content[:1000], "url": url})
    except Exception as e:
        logger.exception("Tavily search failed: %s", e)
    return sources


def _wikipedia_sources(query: str, max_results: int = 3) -> list[Source]:
    sources: list[Source] = []
    try:
        import wikipedia

        titles = wikipedia.search(query, results=max_results)
        for title in titles[:max_results]:
            try:
                summary = wikipedia.summary(title, auto_suggest=False, sentences=5)
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                sources.append({"title": title, "snippet": summary[:1000], "url": url})
            except Exception:
                continue
    except Exception as e:
        logger.exception("Wikipedia search failed: %s", e)
    return sources


def _arxiv_sources(query: str, max_results: int = 3) -> list[Source]:
    sources: list[Source] = []
    if not USE_ARXIV:
        return sources
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        for result in client.results(search):
            title = getattr(result, "title", "") or ""
            abstract = getattr(result, "abstract", "") or ""
            url = (
                getattr(result, "entry_id", "") or getattr(result, "pdf_url", "") or ""
            )
            sources.append({"title": title, "snippet": abstract[:1000], "url": url})
    except Exception as e:
        logger.exception("arXiv search failed: %s", e)
    return sources


def search(query: str, max_results_per_source: int = 8) -> list[Source]:
    """Run Tavily + Wikipedia + (optional) arXiv; return unified list of sources."""
    all_sources: list[Source] = []
    all_sources.extend(_tavily_sources(query, max_results=max_results_per_source))
    all_sources.extend(_wikipedia_sources(query, max_results=3))
    all_sources.extend(_arxiv_sources(query, max_results=3))
    return all_sources
