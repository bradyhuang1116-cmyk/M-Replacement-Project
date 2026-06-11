"""SQLite 任务队列 — §12 Phase 2。

单进程 + Worker 单线程顺序消费，故 SQLite 即可满足并发需求。
表结构对齐 [docs/plm_integration_design.md](../docs/plm_integration_design.md) §8.1。
"""
from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import QUEUE_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    drawing_no   TEXT,
    revision     TEXT,
    status       TEXT NOT NULL,
    retry_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error_msg    TEXT,
    result_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_created_at ON job_queue(created_at);
"""

_VALID_STATUS = {"pending", "running", "done", "failed"}
_VALID_SOURCE = {"api", "watch_folder"}


@dataclass
class Job:
    id: int
    source: str
    source_file: str
    file_path: str
    drawing_no: str | None
    revision: str | None
    status: str
    retry_count: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    error_msg: str | None
    result_path: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            source=row["source"],
            source_file=row["source_file"],
            file_path=row["file_path"],
            drawing_no=row["drawing_no"],
            revision=row["revision"],
            status=row["status"],
            retry_count=row["retry_count"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error_msg=row["error_msg"],
            result_path=row["result_path"],
        )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobQueue:
    """SQLite-backed FIFO 队列。

    线程安全：所有写操作通过 `_lock` 串行；读操作 SQLite 自身保证一致性。
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or QUEUE_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bootstrap_schema()

    # ── schema ──
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _bootstrap_schema(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── 写操作 ──
    def enqueue(
        self,
        source: str,
        source_file: str,
        file_path: str,
        drawing_no: str | None = None,
        revision: str | None = None,
    ) -> int:
        if source not in _VALID_SOURCE:
            raise ValueError(f"invalid source: {source!r}")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO job_queue (source, source_file, file_path,
                                       drawing_no, revision, status,
                                       retry_count, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?)
                """,
                (source, source_file, file_path, drawing_no, revision, _now_iso()),
            )
            return cur.lastrowid

    def claim_next_pending(self) -> Job | None:
        """取一条 pending 任务并原子地标为 running。

        Returns the claimed Job or None when queue is empty.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status = 'pending'
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
            # 重新拉一次保证字段与 DB 同步
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?", (row["id"],)
            ).fetchone()
            return Job.from_row(row)

    def mark_done(self, job_id: int, result_path: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE job_queue
                SET status='done', finished_at=?, result_path=?, error_msg=NULL
                WHERE id=?
                """,
                (_now_iso(), result_path, job_id),
            )

    def mark_failed(self, job_id: int, error_msg: str):
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
        """failed 后再排队；超过 max_retry 则保持 failed 不动。返回是否真的 requeue。"""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT retry_count FROM job_queue WHERE id=?", (job_id,)
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

    # ── 读操作 ──
    def get(self, job_id: int) -> Job | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_queue WHERE id=?", (job_id,)
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

    def counts(self) -> dict[str, int]:
        """各状态下的任务数。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM job_queue GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    # ── 维护 ──
    def reset_stale_running(self) -> int:
        """启动时把残留的 running（进程崩溃留下的）转回 pending。"""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE job_queue SET status='pending' WHERE status='running'"
            )
            return cur.rowcount
