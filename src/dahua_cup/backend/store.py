"""SQLite persistence for Web samples, reviews, audit events, and jobs."""

from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from dahua_cup.semantic_teacher.schemas import LABELS
from dahua_cup.semantic_teacher.pseudo_label.repository import (
    revert_review as revert_pseudo_review,
    write_back_review,
)


SPECIAL_LABELS = ("unknown", "damaged", "out_of_scope")
REVIEW_LABELS = frozenset(LABELS) | frozenset(SPECIAL_LABELS)
SUGGESTION_TO_LABEL = {
    "normal_walk": "normal_walk",
    "normal_run": "normal_run",
}
INTERRUPTED_JOB_MESSAGE = "Web 服务重启，任务已安全中止"


SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY,
  video_path TEXT NOT NULL,
  source_dataset TEXT NOT NULL DEFAULT '',
  source_label TEXT NOT NULL DEFAULT '',
  suggested_coarse_label TEXT NOT NULL DEFAULT '',
  suggested_label TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  priority INTEGER NOT NULL DEFAULT 0,
  manual_label TEXT,
  reviewer TEXT,
  reason_code TEXT,
  note TEXT NOT NULL DEFAULT '',
  quality_score REAL,
  hard_score REAL,
  pseudo_dataset_path TEXT NOT NULL DEFAULT '',
  pseudo_record_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
CREATE INDEX IF NOT EXISTS idx_samples_dataset ON samples(source_dataset);

CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  previous_status TEXT NOT NULL,
  previous_label TEXT,
  final_label TEXT NOT NULL,
  reason_code TEXT NOT NULL,
  note TEXT NOT NULL,
  student_snapshot_json TEXT NOT NULL DEFAULT '{}',
  teacher_snapshot_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  reverted_at TEXT,
  FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id TEXT,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  sample_id TEXT NOT NULL,
  action TEXT NOT NULL,
  status TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  log_text TEXT NOT NULL DEFAULT '',
  pose_device TEXT NOT NULL DEFAULT '',
  student_device TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_sample ON jobs(sample_id, created_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(value: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(value) if value is not None else None


def _sample_record(value: dict) -> dict:
    result = dict(value)
    raw = result.pop("pseudo_record_json", "{}") or "{}"
    try:
        result["pseudo_record"] = json.loads(raw)
    except (TypeError, ValueError):
        result["pseudo_record"] = {"status": "invalid_pseudo_record"}
    return result


class ReviewStore:
    def __init__(self, database: Union[str, Path]):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_schema(connection)

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        sample_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(samples)")
        }
        sample_additions = {
            "quality_score": "REAL",
            "hard_score": "REAL",
            "pseudo_dataset_path": "TEXT NOT NULL DEFAULT ''",
            "pseudo_record_json": "TEXT NOT NULL DEFAULT '{}'",
            "incremental_pool": "INTEGER NOT NULL DEFAULT 0",
            "incremental_reason": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in sample_additions.items():
            if name not in sample_columns:
                connection.execute(
                    "ALTER TABLE samples ADD COLUMN {} {}".format(name, definition)
                )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(reviews)")
        }
        additions = {
            "student_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
            "teacher_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    "ALTER TABLE reviews ADD COLUMN {} {}".format(name, definition)
                )
        job_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        for name in ("pose_device", "student_device"):
            if name not in job_columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN {} TEXT NOT NULL DEFAULT ''".format(name)
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def import_manifest(self, path: Union[str, Path]) -> Dict[str, int]:
        source = Path(path)
        if not source.is_file():
            return {"inserted": 0, "existing": 0}
        inserted = 0
        existing = 0
        created_at = _now()
        with source.open("r", encoding="utf-8-sig", newline="") as stream, self.connect() as connection:
            for item in csv.DictReader(stream):
                sample_id = str(item.get("clip_id", "")).strip()
                video_path = str(item.get("path", "")).strip()
                if not sample_id or not video_path:
                    continue
                suggestion = str(item.get("suggested_coarse_label", "")).strip()
                manual_label = str(item.get("manual_label", "")).strip() or None
                if manual_label and manual_label not in REVIEW_LABELS:
                    manual_label = None
                status = (
                    "reviewed" if manual_label in LABELS
                    else manual_label if manual_label in SPECIAL_LABELS
                    else "pending"
                )
                dataset = str(item.get("source_dataset", "")).strip()
                source_label = str(item.get("source_label", "")).strip()
                note = str(item.get("notes", "")).strip()
                previous = connection.execute(
                    "SELECT video_path FROM samples WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()
                if previous is not None:
                    connection.execute(
                        """
                        UPDATE samples SET video_path = ?, source_dataset = ?,
                          source_label = ?, suggested_coarse_label = ?,
                          suggested_label = ?, updated_at = ? WHERE sample_id = ?
                        """,
                        (
                            video_path, dataset, source_label, suggestion,
                            SUGGESTION_TO_LABEL.get(suggestion), created_at, sample_id,
                        ),
                    )
                    existing += 1
                    if previous["video_path"] != video_path:
                        connection.execute(
                            "INSERT INTO events(sample_id, event_type, actor, payload_json, created_at) VALUES (?, 'manifest_path_refreshed', 'system', ?, ?)",
                            (
                                sample_id,
                                json.dumps(
                                    {
                                        "old_path": previous["video_path"],
                                        "new_path": video_path,
                                        "manifest": str(source),
                                    },
                                    ensure_ascii=False,
                                ),
                                created_at,
                            ),
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO samples(
                      sample_id, video_path, source_dataset, source_label,
                      suggested_coarse_label, suggested_label, status, manual_label,
                      note, created_at, updated_at, reviewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        video_path,
                        dataset,
                        source_label,
                        suggestion,
                        SUGGESTION_TO_LABEL.get(suggestion),
                        status,
                        manual_label,
                        note,
                        created_at,
                        created_at,
                        created_at if status == "reviewed" else None,
                    ),
                )
                inserted += 1
                connection.execute(
                    "INSERT INTO events(sample_id, event_type, actor, payload_json, created_at) VALUES (?, 'imported', 'system', ?, ?)",
                    (sample_id, json.dumps({"manifest": str(source)}, ensure_ascii=False), created_at),
                )
        return {"inserted": inserted, "existing": existing}

    def import_baseline_records(self, records: list[dict]) -> Dict[str, int]:
        """Index existing official labels without copying any model artefact."""
        inserted = existing = 0
        now = _now()
        with self.connect() as connection:
            for item in records:
                sample_id = str(item["sample_id"])
                previous = connection.execute(
                    "SELECT 1 FROM samples WHERE sample_id = ?", (sample_id,)
                ).fetchone()
                if previous:
                    # Never overwrite a later human decision on restart.
                    # Existing rows only need their immutable source index
                    # refreshed; labels/reviewer/reason remain auditable.
                    connection.execute(
                        """
                        UPDATE samples SET video_path=?, source_dataset=?,
                          source_label=?, updated_at=? WHERE sample_id=?
                        """,
                        (
                            str(item["video_path"]),
                            str(item["source_dataset"]),
                            str(item["source_label"]),
                            now, sample_id,
                        ),
                    )
                    existing += 1
                else:
                    connection.execute(
                        """
                        INSERT INTO samples(
                          sample_id,video_path,source_dataset,source_label,
                          status,manual_label,reviewer,reason_code,note,
                          created_at,updated_at,reviewed_at
                        ) VALUES (?,?,?,?,'reviewed',?,?,?,?,?,?,?)
                        """,
                        (
                            sample_id,
                            str(item["video_path"]),
                            str(item["source_dataset"]),
                            str(item["source_label"]),
                            str(item["manual_label"]),
                            str(item["reviewer"]),
                            str(item["reason_code"]),
                            str(item.get("note", "")),
                            now, now, now,
                        ),
                    )
                    inserted += 1
        return {"inserted": inserted, "existing": existing}

    def add_sample(
        self,
        sample_id: str,
        video_path: Union[str, Path],
        *,
        source_dataset: str = "uploaded",
        source_label: str = "",
        suggested_coarse_label: str = "",
    ) -> dict:
        sample_id = sample_id.strip()
        if not sample_id:
            raise ValueError("sample_id is required")
        created_at = _now()
        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO samples(
                      sample_id, video_path, source_dataset, source_label,
                      suggested_coarse_label, suggested_label, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id, str(Path(video_path).resolve()), source_dataset,
                        source_label, suggested_coarse_label,
                        SUGGESTION_TO_LABEL.get(suggested_coarse_label),
                        created_at, created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"sample already exists: {sample_id}") from exc
            connection.execute(
                "INSERT INTO events(sample_id, event_type, actor, payload_json, created_at) VALUES (?, 'uploaded', 'system', '{}', ?)",
                (sample_id, created_at),
            )
        return self.get_sample(sample_id)

    def enqueue_pseudo_record(
        self,
        record: dict,
        video_path: Union[str, Path],
        *,
        pseudo_dataset_path: Union[str, Path],
        priority: int = 0,
        hard_score: Optional[float] = None,
        source_dataset: str = "pseudo_label",
    ) -> dict:
        """Insert or refresh one filter-routed item in the Web review queue."""
        sample_id = str(record.get("sample_id", "")).strip()
        label = str(record.get("label", "")).strip()
        score = float(record.get("quality_score", -1))
        if not sample_id or label not in LABELS or not 0 <= score <= 1:
            raise ValueError("invalid pseudo-label queue record")
        source = Path(video_path).expanduser().resolve()
        dataset_path = Path(pseudo_dataset_path).expanduser().resolve()
        created_at = _now()
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        priority = max(0, int(priority))
        hard_value = None if hard_score is None else float(hard_score)
        if hard_value is not None and not 0 <= hard_value <= 1:
            raise ValueError("hard_score must be in [0, 1]")
        with self.connect() as connection:
            previous = connection.execute(
                "SELECT status FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if previous is None:
                connection.execute(
                    """
                    INSERT INTO samples(
                      sample_id, video_path, source_dataset, suggested_label,
                      status, priority, quality_score, hard_score,
                      pseudo_dataset_path, pseudo_record_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id, str(source), source_dataset, label, priority,
                        score, hard_value, str(dataset_path), payload,
                        created_at, created_at,
                    ),
                )
                event_type = "pseudo_review_enqueued"
            else:
                connection.execute(
                    """
                    UPDATE samples SET video_path = ?, source_dataset = ?,
                      suggested_label = ?, status = 'pending',
                      priority = ?, manual_label = NULL, reviewer = NULL,
                      reason_code = NULL, note = '', reviewed_at = NULL,
                      quality_score = ?, hard_score = ?,
                      pseudo_dataset_path = ?, pseudo_record_json = ?,
                      updated_at = ?
                    WHERE sample_id = ?
                    """,
                    (
                        str(source), source_dataset, label, priority, score,
                        hard_value, str(dataset_path), payload, created_at,
                        sample_id,
                    ),
                )
                event_type = "pseudo_review_refreshed"
            connection.execute(
                """
                INSERT INTO events(
                  sample_id, event_type, actor, payload_json, created_at
                ) VALUES (?, ?, 'pseudo_filter', ?, ?)
                """,
                (
                    sample_id,
                    event_type,
                    json.dumps(
                        {
                            "quality_score": score,
                            "hard_score": hard_value,
                            "priority": priority,
                            "pseudo_dataset_path": str(dataset_path),
                        },
                        ensure_ascii=False,
                    ),
                    created_at,
                ),
            )
        return self.get_sample(sample_id)

    def escalate_hard_sample(
        self,
        sample_id: str,
        *,
        hard_score: float,
        priority: int,
        reason: str,
        payload: Optional[dict] = None,
    ) -> dict:
        """Place an online hard case at the front of the human-review queue."""
        score = float(hard_score)
        if not 0 <= score <= 1:
            raise ValueError("hard_score must be in [0, 1]")
        reason = reason.strip()
        if not reason:
            raise ValueError("hard-sample reason is required")
        priority = max(0, int(priority))
        created_at = _now()
        with self.connect() as connection:
            previous = connection.execute(
                "SELECT status FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if previous is None:
                raise KeyError(sample_id)
            connection.execute(
                """
                UPDATE samples SET status = 'pending',
                  priority = CASE WHEN priority > ? THEN priority ELSE ? END,
                  hard_score = CASE
                    WHEN hard_score IS NULL OR hard_score < ? THEN ?
                    ELSE hard_score
                  END,
                  updated_at = ?
                WHERE sample_id = ?
                """,
                (
                    priority,
                    priority,
                    score,
                    score,
                    created_at,
                    sample_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO events(
                  sample_id, event_type, actor, payload_json, created_at
                ) VALUES (?, 'hard_sample_escalated', 'inference_router', ?, ?)
                """,
                (
                    sample_id,
                    json.dumps(
                        {
                            "reason": reason,
                            "hard_score": score,
                            "priority": priority,
                            "previous_status": previous["status"],
                            **(payload or {}),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    created_at,
                ),
            )
        return self.get_sample(sample_id)

    def list_samples(
        self,
        *,
        status: Optional[str] = None,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        workflow_status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
        summary: bool = False,
    ) -> dict:
        where = []
        values: List[object] = []
        if status and status != "all":
            where.append("status = ?")
            values.append(status)
        if dataset and dataset != "all":
            where.append("source_dataset = ?")
            values.append(dataset)
        if query:
            where.append("(sample_id LIKE ? OR source_label LIKE ? OR video_path LIKE ?)")
            pattern = f"%{query}%"
            values.extend((pattern, pattern, pattern))
        if workflow_status and workflow_status != "all":
            active_teacher_job = (
                "EXISTS (SELECT 1 FROM jobs "
                "WHERE jobs.sample_id = samples.sample_id "
                "AND jobs.action = 'teacher' "
                "AND jobs.status IN ('queued', 'running'))"
            )
            if workflow_status == "complete":
                where.append("status != 'pending'")
            elif workflow_status == "waiting_teacher":
                where.extend(("status = 'pending'", active_teacher_job))
            elif workflow_status == "waiting_human":
                where.extend(("status = 'pending'", f"NOT {active_teacher_job}"))
            else:
                raise ValueError(f"unsupported workflow status: {workflow_status}")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        columns = (
            "sample_id, source_dataset, source_label, suggested_coarse_label, "
            "suggested_label, status, priority, manual_label, reviewer, "
            "updated_at, reviewed_at, "
            "CASE WHEN status != 'pending' THEN 'complete' "
            "WHEN EXISTS (SELECT 1 FROM jobs "
            "WHERE jobs.sample_id = samples.sample_id "
            "AND jobs.action = 'teacher' "
            "AND jobs.status IN ('queued', 'running')) THEN 'waiting_teacher' "
            "ELSE 'waiting_human' END AS workflow_status"
            if summary
            else "*"
        )
        with self.connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM samples {clause}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT {columns} FROM samples {clause}
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                         priority DESC, sample_id ASC LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        return {
            "items": [
                dict(item) if summary else _sample_record(dict(item))
                for item in rows
            ],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def get_sample(self, sample_id: str) -> dict:
        with self.connect() as connection:
            sample = _row(connection.execute(
                "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone())
            if sample is None:
                raise KeyError(sample_id)
            sample = _sample_record(sample)
            reviews = [self._review_record(item) for item in connection.execute(
                "SELECT * FROM reviews WHERE sample_id = ? ORDER BY review_id DESC LIMIT 20",
                (sample_id,),
            ).fetchall()]
            jobs = [dict(item) for item in connection.execute(
                "SELECT * FROM jobs WHERE sample_id = ? ORDER BY created_at DESC LIMIT 20",
                (sample_id,),
            ).fetchall()]
        sample["reviews"] = reviews
        sample["jobs"] = jobs
        return sample

    @staticmethod
    def _review_record(row: sqlite3.Row) -> dict:
        value = dict(row)
        for column, target in (
            ("student_snapshot_json", "student_snapshot"),
            ("teacher_snapshot_json", "teacher_snapshot"),
        ):
            raw = value.pop(column, "{}") or "{}"
            try:
                value[target] = json.loads(raw)
            except (TypeError, ValueError):
                value[target] = {"status": "invalid_snapshot"}
        return value

    @staticmethod
    def _student_top1_label(snapshot: object) -> str:
        """Read the stored student Top-1 label without trusting UI input."""
        if not isinstance(snapshot, dict):
            return ""
        topk = snapshot.get("top6") or snapshot.get("topk") or []
        if not isinstance(topk, list) or not topk:
            return ""
        first = topk[0]
        return str(first.get("label") or "") if isinstance(first, dict) else ""

    @classmethod
    def _is_incremental_hard_correction(
        cls, final_label: str, student_snapshot: object
    ) -> bool:
        """A human six-class correction is a durable incremental hard case."""
        student_label = cls._student_top1_label(student_snapshot)
        return (
            final_label in LABELS
            and student_label in LABELS
            and student_label != final_label
        )

    @classmethod
    def _refresh_incremental_pool(
        cls, connection: sqlite3.Connection, sample_id: str
    ) -> bool:
        """Rebuild the pool flag from active review history, including undo."""
        rows = connection.execute(
            """
            SELECT final_label, student_snapshot_json
            FROM reviews
            WHERE sample_id = ? AND reverted_at IS NULL
            ORDER BY review_id DESC
            """,
            (sample_id,),
        ).fetchall()
        enrolled = False
        for row in rows:
            try:
                snapshot = json.loads(row["student_snapshot_json"] or "{}")
            except (TypeError, ValueError):
                snapshot = {}
            if cls._is_incremental_hard_correction(row["final_label"], snapshot):
                enrolled = True
                break
        connection.execute(
            """
            UPDATE samples
            SET incremental_pool = ?,
                incremental_reason = ?
            WHERE sample_id = ?
            """,
            (
                int(enrolled),
                "human_label_disagrees_with_student" if enrolled else "",
                sample_id,
            ),
        )
        return enrolled

    def submit_review(
        self,
        sample_id: str,
        reviewer: str,
        final_label: str,
        reason_code: str,
        note: str = "",
        student_snapshot: Optional[dict] = None,
        teacher_snapshot: Optional[dict] = None,
    ) -> dict:
        reviewer = reviewer.strip()
        reason_code = reason_code.strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        if final_label not in REVIEW_LABELS:
            raise ValueError("invalid final label")
        if not reason_code:
            raise ValueError("reason_code is required")
        created_at = _now()
        status = "reviewed" if final_label in LABELS else final_label
        student_snapshot = student_snapshot or {
            "status": "not_recorded", "top6": []
        }
        teacher_snapshot = teacher_snapshot or {"status": "not_recorded"}
        student_snapshot_json = json.dumps(student_snapshot, ensure_ascii=False)
        teacher_snapshot_json = json.dumps(teacher_snapshot, ensure_ascii=False)
        with self.connect() as connection:
            previous = connection.execute(
                "SELECT status, manual_label FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if previous is None:
                raise KeyError(sample_id)
            cursor = connection.execute(
                """
                INSERT INTO reviews(
                  sample_id, reviewer, previous_status, previous_label,
                  final_label, reason_code, note, student_snapshot_json,
                  teacher_snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id, reviewer, previous["status"], previous["manual_label"],
                    final_label, reason_code, note.strip(), student_snapshot_json,
                    teacher_snapshot_json, created_at,
                ),
            )
            connection.execute(
                """
                UPDATE samples SET status = ?, manual_label = ?, reviewer = ?,
                  reason_code = ?, note = ?, reviewed_at = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (status, final_label, reviewer, reason_code, note.strip(), created_at, created_at, sample_id),
            )
            enrolled_in_incremental_pool = self._refresh_incremental_pool(
                connection, sample_id
            )
            connection.execute(
                "INSERT INTO events(sample_id, event_type, actor, payload_json, created_at) VALUES (?, 'review_submitted', ?, ?, ?)",
                (
                    sample_id,
                    reviewer,
                    json.dumps(
                        {
                            "review_id": cursor.lastrowid,
                            "final_label": final_label,
                            "reason_code": reason_code,
                            "student_top1_label": self._student_top1_label(
                                student_snapshot
                            ),
                            "student_status": student_snapshot.get("status"),
                            "teacher_status": teacher_snapshot.get("status"),
                            "incremental_hard_enqueued": enrolled_in_incremental_pool,
                        },
                        ensure_ascii=False,
                    ),
                    created_at,
                ),
            )
            if enrolled_in_incremental_pool:
                connection.execute(
                    """
                    INSERT INTO events(
                      sample_id, event_type, actor, payload_json, created_at
                    ) VALUES (?, 'incremental_hard_sample_enqueued', ?, ?, ?)
                    """,
                    (
                        sample_id,
                        reviewer,
                        json.dumps(
                            {
                                "reason": "human_label_disagrees_with_student",
                                "final_label": final_label,
                                "student_top1_label": self._student_top1_label(
                                    student_snapshot
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        created_at,
                    ),
                )
            dataset_path = connection.execute(
                "SELECT pseudo_dataset_path FROM samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()["pseudo_dataset_path"]
            if dataset_path:
                reviewed_record = write_back_review(
                    dataset_path,
                    sample_id,
                    {
                        "review_id": cursor.lastrowid,
                        "reviewer": reviewer,
                        "final_label": final_label,
                        "reason_code": reason_code,
                        "note": note.strip(),
                        "created_at": created_at,
                    },
                )
                connection.execute(
                    "UPDATE samples SET pseudo_record_json = ? WHERE sample_id = ?",
                    (
                        json.dumps(
                            reviewed_record, ensure_ascii=False, sort_keys=True
                        ),
                        sample_id,
                    ),
                )
        return self.get_sample(sample_id)

    def undo_last_review(self, sample_id: str, actor: str) -> dict:
        actor = actor.strip()
        if not actor:
            raise ValueError("actor is required")
        created_at = _now()
        with self.connect() as connection:
            review = connection.execute(
                "SELECT * FROM reviews WHERE sample_id = ? AND reverted_at IS NULL ORDER BY review_id DESC LIMIT 1",
                (sample_id,),
            ).fetchone()
            if review is None:
                raise ValueError("sample has no review to undo")
            connection.execute(
                "UPDATE reviews SET reverted_at = ? WHERE review_id = ?",
                (created_at, review["review_id"]),
            )
            restored_review = connection.execute(
                """
                SELECT * FROM reviews
                WHERE sample_id = ? AND reverted_at IS NULL
                ORDER BY review_id DESC LIMIT 1
                """,
                (sample_id,),
            ).fetchone()
            connection.execute(
                """
                UPDATE samples SET status = ?, manual_label = ?, reviewer = ?,
                  reason_code = ?, note = ?, reviewed_at = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (
                    review["previous_status"],
                    review["previous_label"],
                    restored_review["reviewer"] if restored_review else None,
                    restored_review["reason_code"] if restored_review else None,
                    restored_review["note"] if restored_review else "",
                    restored_review["created_at"] if restored_review else None,
                    created_at,
                    sample_id,
                ),
            )
            self._refresh_incremental_pool(connection, sample_id)
            connection.execute(
                "INSERT INTO events(sample_id, event_type, actor, payload_json, created_at) VALUES (?, 'review_reverted', ?, ?, ?)",
                (sample_id, actor, json.dumps({"review_id": review["review_id"]}), created_at),
            )
            dataset_path = connection.execute(
                "SELECT pseudo_dataset_path FROM samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()["pseudo_dataset_path"]
            if dataset_path:
                reverted_record = revert_pseudo_review(dataset_path, sample_id)
                connection.execute(
                    "UPDATE samples SET pseudo_record_json = ? WHERE sample_id = ?",
                    (
                        json.dumps(
                            reverted_record, ensure_ascii=False, sort_keys=True
                        ),
                        sample_id,
                    ),
                )
        return self.get_sample(sample_id)

    def undo_latest_review(self, actor: str) -> dict:
        with self.connect() as connection:
            review = connection.execute(
                "SELECT sample_id FROM reviews WHERE reverted_at IS NULL ORDER BY review_id DESC LIMIT 1"
            ).fetchone()
        if review is None:
            raise ValueError("there is no review to undo")
        return self.undo_last_review(review["sample_id"], actor)

    def dashboard(self) -> dict:
        with self.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            statuses = {
                row["status"]: row["count"] for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM samples GROUP BY status"
                )
            }
            labels = {
                row["manual_label"]: row["count"] for row in connection.execute(
                    "SELECT manual_label, COUNT(*) AS count FROM samples WHERE manual_label IS NOT NULL GROUP BY manual_label"
                )
            }
            datasets = {
                row["source_dataset"]: row["count"] for row in connection.execute(
                    "SELECT source_dataset, COUNT(*) AS count FROM samples GROUP BY source_dataset"
                )
            }
            jobs = [dict(row) for row in connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 8"
            ).fetchall()]
        return {"total": total, "statuses": statuses, "labels": labels, "datasets": datasets, "recent_jobs": jobs}

    def recent_events(self, limit: int = 50) -> List[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY event_id DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(
        self, sample_id: str, action: str, *, pose_device: str = "",
        student_device: str = "",
    ) -> dict:
        job_id = uuid.uuid4().hex
        created_at = _now()
        with self.connect() as connection:
            if connection.execute("SELECT 1 FROM samples WHERE sample_id = ?", (sample_id,)).fetchone() is None:
                raise KeyError(sample_id)
            connection.execute(
                """
                INSERT INTO jobs(
                  job_id, sample_id, action, status, pose_device,
                  student_device, created_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, sample_id, action, pose_device, student_device, created_at),
            )
        return self.get_job(job_id)

    def recover_incomplete_jobs(self) -> int:
        """Fail jobs orphaned by a previous Web process restart."""
        finished_at = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    message = ?,
                    finished_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (INTERRUPTED_JOB_MESSAGE, finished_at),
            )
            return int(cursor.rowcount)

    def redact_verbose_teacher_failures(self) -> int:
        """Remove historical model-load traces from reviewer-facing job data."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET message = 'Qwen 教师分析失败，请稍后重试',
                    log_text = ''
                WHERE action = 'teacher'
                  AND status = 'failed'
                  AND (
                    instr(log_text, 'Traceback') > 0
                    OR instr(log_text, 'Loading weights') > 0
                    OR instr(message, 'Traceback') > 0
                    OR instr(message, '命令执行失败') > 0
                  )
                """
            )
            return int(cursor.rowcount)

    def update_job(self, job_id: str, **changes) -> dict:
        allowed = {"status", "progress", "message", "log_text", "started_at", "finished_at"}
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get_job(job_id)
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                (*values.values(), job_id),
            )
            if not cursor.rowcount:
                raise KeyError(job_id)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict:
        with self.connect() as connection:
            value = _row(connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone())
        if value is None:
            raise KeyError(job_id)
        return value

    def export_manifest(self, destination: Union[str, Path]) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "clip_id", "path", "source_dataset", "source_label",
            "suggested_coarse_label", "manual_label", "keep", "notes",
            "reviewer", "reason_code", "reviewed_at",
            "student_status", "student_model", "student_task", "student_modality",
            "student_checkpoint_sha256", "student_generated_at",
            "student_top1_label", "student_top1_score",
            "student_top2_label", "student_top2_score",
            "student_top3_label", "student_top3_score",
            "student_top4_label", "student_top4_score",
            "student_top5_label", "student_top5_score",
            "student_top6_label", "student_top6_score",
            "teacher_status", "teacher_model", "teacher_suggested_label",
            "teacher_confidence",
        )
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT samples.*,
                  reviews.student_snapshot_json AS export_student_snapshot_json,
                  reviews.teacher_snapshot_json AS export_teacher_snapshot_json
                FROM samples
                LEFT JOIN reviews ON reviews.review_id = (
                  SELECT MAX(active.review_id) FROM reviews AS active
                  WHERE active.sample_id = samples.sample_id
                    AND active.reverted_at IS NULL
                )
                ORDER BY samples.sample_id
                """
            ).fetchall()
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in rows:
                value = dict(item)
                try:
                    student = json.loads(
                        value.get("export_student_snapshot_json") or "{}"
                    )
                except (TypeError, ValueError):
                    student = {"status": "invalid_snapshot"}
                try:
                    teacher = json.loads(
                        value.get("export_teacher_snapshot_json") or "{}"
                    )
                except (TypeError, ValueError):
                    teacher = {"status": "invalid_snapshot"}
                top6 = list(
                    student.get("top6") or student.get("top5") or []
                )[:6]
                teacher_result = teacher.get("result") or {}
                row = {
                    "clip_id": value["sample_id"],
                    "path": value["video_path"],
                    "source_dataset": value["source_dataset"],
                    "source_label": value["source_label"],
                    "suggested_coarse_label": value["suggested_coarse_label"],
                    "manual_label": value["manual_label"] or "",
                    "keep": "yes" if value["manual_label"] in LABELS else "no" if value["status"] != "pending" else "",
                    "notes": value["note"],
                    "reviewer": value["reviewer"] or "",
                    "reason_code": value["reason_code"] or "",
                    "reviewed_at": value["reviewed_at"] or "",
                    "student_status": student.get("status", ""),
                    "student_model": student.get("model", ""),
                    "student_task": student.get("task", ""),
                    "student_modality": student.get("modality", ""),
                    "student_checkpoint_sha256": student.get("checkpoint_sha256", ""),
                    "student_generated_at": student.get("generated_at", ""),
                    "teacher_status": teacher.get("status", ""),
                    "teacher_model": teacher.get("model", ""),
                    "teacher_suggested_label": teacher_result.get("suggested_label", ""),
                    "teacher_confidence": teacher_result.get("confidence", ""),
                }
                for index in range(6):
                    prediction = top6[index] if index < len(top6) else {}
                    row["student_top{}_label".format(index + 1)] = prediction.get("label", "")
                    row["student_top{}_score".format(index + 1)] = prediction.get("score", "")
                writer.writerow(row)
        return output

    def count_incremental_samples(self, since: Optional[str]) -> int:
        """Count post-production human-confirmed Campus6 samples."""
        placeholders = ",".join("?" for _ in LABELS)
        values: List[object] = [*LABELS, "official_initial_annotation"]
        time_clause = ""
        if since:
            time_clause = "AND reviews.created_at > ?"
            values.append(since)
        with self.connect() as connection:
            return int(connection.execute(
                f"""
                SELECT COUNT(DISTINCT reviews.sample_id)
                FROM reviews
                WHERE reviews.reverted_at IS NULL
                  AND reviews.final_label IN ({placeholders})
                  AND reviews.reviewer != ?
                  {time_clause}
                """,
                values,
            ).fetchone()[0])
