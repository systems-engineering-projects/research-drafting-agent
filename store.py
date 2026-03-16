"""SQLite persistence: create_run, approve_run."""

import json
import logging
import sqlite3
from pathlib import Path

from config import DB_PATH
from state import Source

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            sources_json TEXT,
            critique TEXT,
            summary TEXT,
            draft TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            approved_final TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """)
    conn.commit()


def create_run(
    topic: str,
    sources: list[Source],
    critique: str = "",
    summary: str = "",
    draft: str = "",
    status: str = "draft",
) -> int:
    """Insert a run; return run_id."""
    conn = _get_conn()
    try:
        _init_db(conn)
        sources_json = json.dumps(sources, default=str)
        cur = conn.execute(
            """
            INSERT INTO runs (topic, sources_json, critique, summary, draft, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (topic, sources_json, critique, summary, draft, status),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def approve_run(run_id: int, final_text: str) -> None:
    """Set approved_final and status='approved' for the run."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE runs SET approved_final = ?, status = 'approved' WHERE id = ?",
            (final_text, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_run(run_id: int) -> dict | None:
    """Return run row as dict or None."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
