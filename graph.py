"""LangGraph: StateGraph with search, critique, summarize, draft, human_approve, persist."""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from llm import critique_sources as llm_critique
from llm import draft_post
from llm import summarize as llm_summarize
from search import search as run_search
from state import ResearchDraftingState
from store import approve_run

MAX_SEARCH_REFINEMENTS = 2


def _route_after_critique(
    state: ResearchDraftingState,
) -> Literal["need_more_sources", "ok"]:
    """Route to search again if sources are weak and we haven't hit the cap."""
    critique = (state.get("critique") or "").lower()
    search_count = state.get("search_count") or 0
    weak = any(
        phrase in critique
        for phrase in ("insufficient", "weak", "need more", "off-topic", "not enough")
    )
    if weak and search_count < MAX_SEARCH_REFINEMENTS:
        return "need_more_sources"
    return "ok"


def search_node(state: ResearchDraftingState) -> dict:
    """Run Tavily + Wikipedia + optional arXiv; update sources and search_count."""
    topic = state.get("topic") or ""
    current_count = state.get("search_count") or 0
    sources = run_search(topic)
    return {"sources": sources, "search_count": current_count + 1}


def critique_node(state: ResearchDraftingState) -> dict:
    """LLM critique of sources."""
    topic = state.get("topic") or ""
    sources = state.get("sources") or []
    critique = llm_critique(topic, sources)
    return {"critique": critique}


def summarize_node(state: ResearchDraftingState) -> dict:
    """LLM summarize from critique and sources."""
    topic = state.get("topic") or ""
    critique = state.get("critique") or ""
    sources = state.get("sources") or []
    summary = llm_summarize(topic, critique, sources)
    return {"summary": summary}


def draft_node(state: ResearchDraftingState) -> dict:
    """LLM draft LinkedIn/email post (no streaming in graph; CLI can stream via graph.stream())."""
    summary = state.get("summary") or ""
    output_format = state.get("output_format") or "linkedin"
    draft = draft_post(summary, output_format, stream_callback=None)
    return {"draft": draft}


def human_approve_node(
    state: ResearchDraftingState,
) -> Command[Literal["persist", "draft"]]:
    """Interrupt for human approval; on resume, route to persist, draft, or END."""
    draft = state.get("draft") or ""
    payload = {
        "draft": draft,
        "prompt": "Approve (a), Regenerate (r), Reject (x)",
        "topic": state.get("topic"),
        "sources": state.get("sources"),
        "critique": state.get("critique"),
        "summary": state.get("summary"),
    }
    result = interrupt(payload)
    # result is the value passed to Command(resume=...) when CLI resumes
    if isinstance(result, dict):
        action = (result.get("action") or "reject").lower()
    else:
        action = "reject"
    update = {
        "approval_result": result if isinstance(result, dict) else {"action": "reject"}
    }
    if action == "approve":
        return Command(update=update, goto="persist")
    if action == "regenerate":
        return Command(update=update, goto="draft")
    return Command(update=update, goto="__end__")


def persist_node(state: ResearchDraftingState) -> dict:
    """Save approved draft to store (run_id and final_text come from approval_result)."""
    approval = state.get("approval_result") or {}
    if isinstance(approval, dict):
        run_id = approval.get("run_id")
        final_text = approval.get("final_text") or state.get("draft") or ""
        if run_id:
            approve_run(run_id, final_text)
    return {}


def build_graph() -> StateGraph:
    """Build and return the compiled StateGraph."""
    builder = StateGraph(ResearchDraftingState)

    builder.add_node("search", search_node)
    builder.add_node("critique", critique_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("draft", draft_node)
    builder.add_node("human_approve", human_approve_node)
    builder.add_node("persist", persist_node)

    builder.add_edge(START, "search")
    builder.add_edge("search", "critique")
    builder.add_conditional_edges(
        "critique",
        _route_after_critique,
        path_map={"need_more_sources": "search", "ok": "summarize"},
    )
    builder.add_edge("summarize", "draft")
    builder.add_edge("draft", "human_approve")
    # human_approve routes via Command to persist, draft, or END (no static edge)
    builder.add_edge("persist", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
