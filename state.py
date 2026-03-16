"""State schema for the research-drafting LangGraph."""

from typing import Any, TypedDict


class Source(TypedDict, total=False):
    """A single search result."""

    title: str
    snippet: str
    url: str


class ApprovalResult(TypedDict, total=False):
    """Result from human approval (interrupt resume)."""

    action: str  # "approve" | "regenerate" | "reject"
    final_text: str  # optional, for approve


class ResearchDraftingState(TypedDict, total=False):
    """Graph state: topic, search results, critique, summary, draft, approval."""

    topic: str
    output_format: str  # "linkedin" | "email"
    sources: list[Source]
    critique: str
    summary: str
    draft: str
    search_count: int
    approval_result: ApprovalResult
