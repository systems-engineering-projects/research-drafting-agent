"""CLI: parse topic, invoke/stream graph, handle interrupt and resume, call store on approve."""

import argparse
import sys
import uuid

from langgraph.types import Command

from graph import build_graph
from store import create_run, approve_run


def _get_config(thread_id: str | None = None) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Research & Drafting Agent (LangGraph)"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="",
        help="Topic or question to research and draft",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("linkedin", "email"),
        default="linkedin",
        help="Output format: linkedin or email",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread ID for checkpointing (default: new UUID)",
    )
    args = parser.parse_args()
    topic = (args.topic or "").strip()
    if not topic:
        parser.error("Topic is required (e.g. python cli.py 'What is RAG?')")
        return 1

    graph = build_graph()
    config = _get_config(args.thread_id)
    thread_id = config["configurable"]["thread_id"]
    initial_state: dict = {"topic": topic, "output_format": args.format}

    # First run: stream until we hit an interrupt or finish
    interrupt_payload = None
    try:
        for chunk in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_out in chunk.items():
                if isinstance(node_out, dict) and "__interrupt__" in node_out:
                    interrupt_payload = node_out["__interrupt__"]
                    break
                if node_name == "draft" and isinstance(node_out, dict):
                    draft = node_out.get("draft", "")
                    if draft:
                        print("\n--- Draft ---\n")
                        print(draft)
                        print("\n--- End draft ---\n")
            if interrupt_payload is not None:
                break
    except Exception as e:
        print(f"Error during graph run: {e}", file=sys.stderr)
        return 1

    if interrupt_payload is None:
        print("Graph finished without interrupt.")
        return 0

    # We have an interrupt: create run, show draft, prompt user
    draft = interrupt_payload.get("draft") or ""
    prompt = (
        interrupt_payload.get("prompt") or "Approve (a), Regenerate (r), Reject (x)"
    )
    topic_s = interrupt_payload.get("topic") or topic
    sources = interrupt_payload.get("sources") or []
    critique = interrupt_payload.get("critique") or ""
    summary = interrupt_payload.get("summary") or ""

    run_id = create_run(
        topic=topic_s,
        sources=sources,
        critique=critique,
        summary=summary,
        draft=draft,
        status="draft",
    )
    print("\n--- Draft ---\n")
    print(draft)
    print("\n--- End draft ---\n")
    print(prompt)
    final_text = ""
    while True:
        choice = input("Choice [a/r/x]: ").strip().lower() or "x"
        if choice in ("a", "approve"):
            action = "approve"
            final_text = draft
            break
        if choice in ("r", "regenerate"):
            action = "regenerate"
            break
        if choice in ("x", "reject"):
            action = "reject"
            break
        print("Invalid. Use a (approve), r (regenerate), or x (reject).")

    resume_value = {
        "action": action,
        "run_id": run_id,
        "final_text": final_text,
    }
    # Resume the graph
    try:
        graph.invoke(Command(resume=resume_value), config)
    except Exception as e:
        print(f"Error on resume: {e}", file=sys.stderr)
        return 1

    if action == "approve":
        approve_run(run_id, final_text)
        print(f"Approved and saved. Run ID: {run_id}")
    elif action == "reject":
        print("Rejected.")
    else:
        print("Regenerated (graph ran to completion after draft).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
