# Autonomous Research & Drafting Agent (LangGraph)

A LangGraph-based agent that researches a topic (Tavily, Wikipedia, optional arXiv), critiques sources, summarizes, and drafts a short LinkedIn or email post—then pauses for **human approval** before persisting. Built to demonstrate **state graphs**, **conditional edges**, **cycles**, and **human-in-the-loop** via `interrupt()`.

## Flow

1. **Input**: Topic and output format (linkedin | email).
2. **Search**: Tavily + Wikipedia + (optional) arXiv → unified list of sources.
3. **Critique**: LLM critiques relevance, bias, recency.
4. **Conditional**: If sources are weak and under the refinement cap → search again; else → summarize.
5. **Summarize**: LLM produces a short summary from critique + sources.
6. **Draft**: LLM writes a LinkedIn or email post.
7. **Human approve**: Graph **interrupts**; CLI shows the draft and prompts: Approve (a), Regenerate (r), Reject (x).
8. **Resume**: CLI resumes with `Command(resume={...})`; graph routes to persist, back to draft, or end.
9. **Persist**: Approved drafts are saved to SQLite (run ID, topic, sources, critique, summary, draft, approved_final).

## What this demonstrates (LangGraph)

- **State**: TypedDict state (`state.py`) passed through the graph and updated by each node.
- **Conditional edges**: After `critique`, a routing function decides `need_more_sources` → search or `ok` → summarize (with `path_map`).
- **Cycles**: The graph can loop back to search when the critique says sources are weak (up to a cap).
- **Human-in-the-loop**: The `human_approve` node calls `interrupt(payload)`; the CLI prompts the user and resumes with `Command(resume=...)` so the graph continues.
- **Checkpointer**: `MemorySaver()` is used so interrupt/resume works with a stable `thread_id`.

## Run

```bash
cd research-drafting-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set TAVILY_API_KEY, OPENAI_API_KEY
python cli.py "Your topic here"
```

Options:

- `--format` / `-f`: `linkedin` (default) or `email`.
- `--thread-id`: Optional thread ID for checkpointing (default: new UUID).

Example:

```bash
python cli.py "What is RAG?" -f linkedin
```

When the graph interrupts, you’ll see the draft and:

```
Choice [a/r/x]: a
Approved and saved. Run ID: 1
```

## Env vars

| Variable              | Required | Description                          |
|-----------------------|----------|--------------------------------------|
| `TAVILY_API_KEY`      | Yes      | Tavily search API key                 |
| `OPENAI_API_KEY`      | Yes      | OpenAI API key for LLM                |
| `LANGSMITH_TRACING`   | No       | Set to `true` to enable LangSmith     |
| `LANGSMITH_API_KEY`   | No       | LangSmith API key                     |
| `LANGSMITH_PROJECT`   | No       | Project name (default: research-drafting-agent) |
| `USE_ARXIV`           | No       | Set to `false` to disable arXiv       |
| `DB_PATH`             | No       | SQLite path (default: ./research_agent.db)     |

## Project layout

| File               | Purpose                                           |
|--------------------|---------------------------------------------------|
| `cli.py`           | Entrypoint; invokes graph, handles interrupt/resume |
| `config.py`        | Env-based config                                  |
| `graph.py`         | LangGraph StateGraph and nodes                     |
| `llm.py`           | Critique, summarize, draft (LangChain)            |
| `search.py`        | Tavily, Wikipedia, optional arXiv                 |
| `state.py`         | TypedDict state schema                            |
| `store.py`         | SQLite create_run, approve_run                    |

## Lint / format

```bash
black .
python -m py_compile cli.py config.py graph.py llm.py search.py state.py store.py
```
