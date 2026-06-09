"""图纸处理日志记录表 — 开放给 PLM 系统后台访问。

字段对齐客户文档《光栅图处理软件与PLM系统的交互机制》中的 R_V_TD_FILEPATH 表。
处理方式标记 OCR 字段：
  - O = 使用了 OCR 识别（扫描件路径，method='ocr'）
  - N = 未使用 OCR（矢量 PDF 直接处理，method='vector'）

PLM 来源字段（docnumber/work_seq）单机处理时拿不到，留空；
对方 PLM 对接层可在回写客户 Oracle 时补齐。
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from config import PROCESS_LOG_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS process_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    docnumber     TEXT,            -- 文件ID (R_V_TD_FILEPATH.DOCNUMBER)，PLM 来源，可空
    drawing_no    TEXT,            -- 图号 (TH)
    revision      TEXT,            -- 版本 (BBH)
    filename      TEXT,            -- 文件名 (FILENAME)
    ocr_flag      TEXT NOT NULL,   -- 处理方式：O=用OCR / N=未用OCR
    work_seq      TEXT,            -- 接收批次号 (WORK_SEQ)，PLM 来源，可空
    status        TEXT NOT NULL,   -- success / failed
    process_date  TEXT NOT NULL    -- 处理日期时间
);

CREATE INDEX IF NOT EXISTS idx_pl_drawing ON process_log(drawing_no);
CREATE INDEX IF NOT EXISTS idx_pl_date ON process_log(process_date);
"""


def method_to_ocr_flag(method: str) -> str:
    """处理方式 → OCR 标记。method='ocr'→O（用了OCR）；'vector'→N（未用OCR）。"""
    return "O" if method == "ocr" else "N"


class ProcessLog:
    """SQLite 图纸处理日志表。线程安全（写操作加锁）。"""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or PROCESS_LOG_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bootstrap()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _bootstrap(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(
        self,
        drawing_no: str | None,
        revision: str | None,
        ocr_flag: str,
        status: str = "success",
        filename: str | None = None,
        docnumber: str | None = None,
        work_seq: str | None = None,
    ) -> int:
        """写入一条处理日志。返回行 id。"""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO process_log
                    (docnumber, drawing_no, revision, filename,
                     ocr_flag, work_seq, status, process_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (docnumber, drawing_no, revision, filename,
                 ocr_flag, work_seq, status,
                 datetime.now().isoformat(timespec="seconds")),
            )
            return cur.lastrowid

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM process_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
