"""Session history - a local SQLite DB, one row per completed session.
Local-only by design (no hosting), so a file-based DB is the right
amount of infrastructure, not a reason to reach for a real server."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,              -- "yoga" or "workout"
    name TEXT NOT NULL,              -- pose name or exercise name
    started_at TEXT NOT NULL,
    duration_seconds REAL,           -- yoga: hold time; workout: session length
    reps INTEGER,                    -- workout only
    form_warnings INTEGER            -- workout only
);
"""


@contextmanager
def _connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class SessionRecord:
    mode: str
    name: str
    duration_seconds: float | None = None
    reps: int | None = None
    form_warnings: int | None = None


def log_session(record: SessionRecord) -> None:
    with _connection() as conn:
        conn.execute(
            "INSERT INTO sessions (mode, name, started_at, duration_seconds, reps, form_warnings) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.mode,
                record.name,
                datetime.now(timezone.utc).isoformat(),
                record.duration_seconds,
                record.reps,
                record.form_warnings,
            ),
        )


def recent_sessions(limit: int = 20) -> list[dict]:
    with _connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def reps_over_time(exercise: str) -> list[tuple[str, int]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT started_at, reps FROM sessions WHERE mode = 'workout' AND name = ? "
            "ORDER BY started_at",
            (exercise,),
        ).fetchall()
        return [(started_at, reps) for started_at, reps in rows]
