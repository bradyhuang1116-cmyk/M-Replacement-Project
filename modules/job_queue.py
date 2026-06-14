"""SQLite 任务队列。"""
from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import QUEUE_DB_PATH

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    drawing_no   TEXT,
    revision     TEXT,
    docnumber    TEXT,
    work_seq     TEXT,
    status       TEXT NOT NULL,
    retry_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error_msg    TEXT,
    result_path  TEXT,
    plm_delivery_status      TEXT NOT NULL DEFAULT 'not_applicable',
    plm_remote_path          TEXT,
    plm_uploaded_path        TEXT,
    plm_delivery_error       TEXT,
    plm_delivery_finished_at TEXT
);
"""

_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_status ON job_queue(status)",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON job_queue(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_plm_delivery_status ON job_queue(plm_delivery_status)",
)

_VALID_STATUS = {"pending", "running", "done", "failed"}
_VALID_SOURCE = {"api", "watch_folder"}
_VALID_PLM_DELIVERY_STATUS = {
    "not_applicable",
    "pending",
    "uploaded",
    "complete",
    "failed",
}


@dataclass
class Job:
    id: int
    source: str
    source_file: str
    file_path: str
    drawing_no: str | None
    revision: str | None
    docnumber: str | None
    work_seq: str | None
    status: str
    retry_count: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    error_msg: str | None
    result_path: str | None
    plm_delivery_status: str
    plm_remote_path: str | None
    plm_uploaded_path: str | None
    plm_delivery_error: str | None
    plm_delivery_finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            source=row["source"],
            source_file=row["source_file"],
            file_path=row["file_path"],
            drawing_no=row["drawing_no"],
            revision=row["revision"],
            docnumber=row["docnumber"],
            work_seq=row["work_seq"],
            status=row["status"],
            retry_count=row["retry_count"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error_msg=row["error_msg"],
            result_path=row["result_path"],
            plm_delivery_status=row["plm_delivery_status"],
            plm_remote_path=row["plm_remote_path"],
            plm_uploaded_path=row["plm_uploaded_path"],
            plm_delivery_error=row["plm_delivery_error"],
            plm_delivery_finished_at=row["plm_delivery_finished_at"],
        )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobQueue:
    """SQLite FIFO 任务队列。"""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or QUEUE_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bootstrap_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _bootstrap_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_TABLE_SCHEMA)
            self._ensure_delivery_columns(conn)
            for ddl in _INDEX_STATEMENTS:
                conn.execute(ddl)

    @staticmethod
    def _ensure_delivery_columns(conn: sqlite3.Connection) -> None:
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(job_queue)").fetchall()
        }
        required = {
            "plm_delivery_status": "TEXT NOT NULL DEFAULT 'not_applicable'",
            "plm_remote_path": "TEXT",
            "plm_uploaded_path": "TEXT",
            "plm_delivery_error": "TEXT",
            "plm_delivery_finished_at": "TEXT",
        }
        for column, ddl in required.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE job_queue ADD COLUMN {column} {ddl}")

    def enqueue(
        self,
        source: str,
        source_file: str,
        file_path: str,
        drawing_no: str | None = None,
        revision: str | None = None,
        docnumber: str | None = None,
        work_seq: str | None = None,
    ) -> int:
        if source not in _VALID_SOURCE:
            raise ValueError(f"invalid source: {source!r}")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO job_queue (
                    source, source_file, file_path,
                    drawing_no, revision, docnumber,
                    work_seq, status, retry_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)
                """,
                (
                    source,
                    source_file,
                    file_path,
                    drawing_no,
                    revision,
                    docnumber,
                    work_seq,
                    _now_iso(),
                ),
            )
            return cur.lastrowid

    def claim_next_pending(self) -> Job | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status='pending'
                ORDER BY created_at, id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            now = _now_iso()
            conn.execute(
                "UPDATE job_queue SET status='running', started_at=? WHERE id=?",
                (now, row["id"]),
            )
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (row["id"],),
            ).fetchone()
            return Job.from_row(row)

    def mark_done(self, job_id: int, result_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status='done', finished_at=?, result_path=?, error_msg=NULL
                WHERE id=?
                """,
                (_now_iso(), result_path, job_id),
            )

    def mark_failed(self, job_id: int, error_msg: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status='failed', finished_at=?, error_msg=?
                WHERE id=?
                """,
                (_now_iso(), error_msg, job_id),
            )

    def requeue_for_retry(self, job_id: int, max_retry: int) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT retry_count FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None or row["retry_count"] >= max_retry:
                return False
            conn.execute(
                """
                UPDATE job_queue
                SET status='pending',
                    retry_count = retry_count + 1,
                    started_at = NULL,
                    finished_at = NULL,
                    error_msg = NULL
                WHERE id=?
                """,
                (job_id,),
            )
            return True

    def update_plm_delivery(
        self,
        job_id: int,
        delivery_status: str,
        remote_path: str | None = None,
        uploaded_path: str | None = None,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        if delivery_status not in _VALID_PLM_DELIVERY_STATUS:
            raise ValueError(f"invalid plm delivery status: {delivery_status!r}")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET plm_delivery_status=?,
                    plm_remote_path=?,
                    plm_uploaded_path=?,
                    plm_delivery_error=?,
                    plm_delivery_finished_at=?
                WHERE id=?
                """,
                (
                    delivery_status,
                    remote_path,
                    uploaded_path,
                    error_msg,
                    _now_iso() if finished else None,
                    job_id,
                ),
            )

    def mark_delivery_pending(
        self,
        job_id: int,
        remote_path: str | None,
        uploaded_path: str | None,
    ) -> None:
        self.update_plm_delivery(
            job_id,
            "pending",
            remote_path=remote_path,
            uploaded_path=uploaded_path,
            error_msg=None,
            finished=False,
        )

    def mark_delivery_uploaded(
        self,
        job_id: int,
        remote_path: str | None,
        uploaded_path: str | None,
        error_msg: str | None = None,
    ) -> None:
        self.update_plm_delivery(
            job_id,
            "uploaded",
            remote_path=remote_path,
            uploaded_path=uploaded_path,
            error_msg=error_msg,
            finished=False,
        )

    def mark_delivery_complete(
        self,
        job_id: int,
        remote_path: str | None,
        uploaded_path: str | None,
    ) -> None:
        self.update_plm_delivery(
            job_id,
            "complete",
            remote_path=remote_path,
            uploaded_path=uploaded_path,
            error_msg=None,
            finished=True,
        )

    def mark_delivery_failed(
        self,
        job_id: int,
        error_msg: str,
        remote_path: str | None = None,
        uploaded_path: str | None = None,
    ) -> None:
        self.update_plm_delivery(
            job_id,
            "failed",
            remote_path=remote_path,
            uploaded_path=uploaded_path,
            error_msg=error_msg,
            finished=False,
        )

    def retry_failed_job(self, job_id: int) -> Job:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"job not found: {job_id}")
            job = Job.from_row(row)
            if job.source != "api":
                raise ValueError("only api jobs support manual retry")
            if job.status != "failed":
                raise ValueError("only failed jobs can retry processing")
            if not job.file_path or not os.path.isfile(job.file_path):
                raise ValueError(f"source file not found: {job.file_path}")

            conn.execute(
                """
                UPDATE job_queue
                SET status='pending',
                    retry_count = retry_count + 1,
                    started_at = NULL,
                    finished_at = NULL,
                    error_msg = NULL,
                    result_path = NULL,
                    plm_delivery_status = 'not_applicable',
                    plm_remote_path = NULL,
                    plm_uploaded_path = NULL,
                    plm_delivery_error = NULL,
                    plm_delivery_finished_at = NULL
                WHERE id=?
                """,
                (job_id,),
            )
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            return Job.from_row(row)

    def retry_failed_plm_delivery(self, job_id: int) -> Job:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"job not found: {job_id}")
            job = Job.from_row(row)
            if job.source != "api":
                raise ValueError("only api jobs support manual retry")
            if job.status != "done" or job.plm_delivery_status != "failed":
                raise ValueError("only failed PLM deliveries can retry")

            conn.execute(
                """
                UPDATE job_queue
                SET retry_count = retry_count + 1,
                    plm_delivery_status = 'pending',
                    plm_delivery_error = NULL,
                    plm_delivery_finished_at = NULL
                WHERE id=?
                """,
                (job_id,),
            )
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            return Job.from_row(row)

    def get(self, job_id: int) -> Job | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?",
                (job_id,),
            ).fetchone()
            return Job.from_row(row) if row else None

    def list_by_status(self, status: str, limit: int = 100) -> list[Job]:
        if status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {status!r}")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status=?
                ORDER BY created_at, id
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
            return [Job.from_row(r) for r in rows]

    def list_jobs(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        if source is not None and source not in _VALID_SOURCE:
            raise ValueError(f"invalid source: {source!r}")
        if status is not None and status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {status!r}")

        where: list[str] = []
        params: list[object] = []
        if source is not None:
            where.append("source=?")
            params.append(source)
        if status is not None:
            where.append("status=?")
            params.append(status)

        sql = "SELECT * FROM job_queue"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [Job.from_row(r) for r in rows]

    def count_jobs(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
    ) -> int:
        if source is not None and source not in _VALID_SOURCE:
            raise ValueError(f"invalid source: {source!r}")
        if status is not None and status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {status!r}")

        where: list[str] = []
        params: list[object] = []
        if source is not None:
            where.append("source=?")
            params.append(source)
        if status is not None:
            where.append("status=?")
            params.append(status)

        sql = "SELECT COUNT(*) AS n FROM job_queue"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return int(row["n"]) if row else 0

    def list_recoverable_plm_deliveries(self, limit: int = 100) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status='done'
                  AND source='api'
                  AND COALESCE(docnumber, '') <> ''
                  AND COALESCE(work_seq, '') <> ''
                  AND (
                        plm_delivery_status IN ('pending', 'uploaded', 'failed')
                        OR plm_delivery_status IS NULL
                        OR plm_delivery_status = 'not_applicable'
                  )
                ORDER BY finished_at, id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [Job.from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM job_queue GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    def reset_stale_running(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE job_queue SET status='pending' WHERE status='running'"
            )
            return cur.rowcount
