"""SQLite review queue with immutable status-change audit events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dahua_cup.semantic_teacher.schemas import LABELS


SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY, artifact_path TEXT NOT NULL, suggested_label TEXT NOT NULL,
  score REAL NOT NULL, status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
  versions_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
  reviewer TEXT NOT NULL, final_label TEXT NOT NULL, reason_code TEXT NOT NULL,
  note TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
  old_status TEXT, new_status TEXT NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewQueue:
    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enqueue(self, sample_id, artifact_path, suggested_label, score, versions, priority=0) -> None:
        if suggested_label not in LABELS or not 0 <= score <= 1:
            raise ValueError("invalid suggested label or score")
        created_at = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (sample_id, str(artifact_path), suggested_label, score, priority,
                 json.dumps(versions, sort_keys=True), created_at),
            )
            connection.execute(
                "INSERT INTO events(sample_id, old_status, new_status, actor, created_at) VALUES (?, NULL, 'pending', 'system', ?)",
                (sample_id, created_at),
            )

    def submit_review(self, sample_id, reviewer, final_label, reason_code, note="") -> None:
        if final_label not in set(LABELS) | {"unknown", "damaged"}:
            raise ValueError("invalid final label")
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM samples WHERE sample_id = ?", (sample_id,)).fetchone()
            if row is None:
                raise KeyError(sample_id)
            if row["status"] != "pending":
                raise ValueError("sample has already been reviewed")
            new_status = "reviewed" if final_label in LABELS else final_label
            created_at = _now()
            connection.execute(
                "INSERT INTO reviews(sample_id, reviewer, final_label, reason_code, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sample_id, reviewer, final_label, reason_code, note, created_at),
            )
            connection.execute("UPDATE samples SET status = ? WHERE sample_id = ?", (new_status, sample_id))
            connection.execute(
                "INSERT INTO events(sample_id, old_status, new_status, actor, created_at) VALUES (?, 'pending', ?, ?, ?)",
                (sample_id, new_status, reviewer, created_at),
            )

    def pending(self, limit=100) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM samples WHERE status = 'pending' ORDER BY priority DESC, created_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]
