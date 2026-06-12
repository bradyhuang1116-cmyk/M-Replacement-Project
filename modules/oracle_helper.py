"""Oracle 数据库助手 — 连接管理与 PLM 双表 CRUD。

使用 oracledb thin mode，无需 Oracle 客户端库。
"""
from __future__ import annotations

import logging
import threading

import oracledb

from config import (
    ORACLE_HOST,
    ORACLE_MAX_POOL,
    ORACLE_MIN_POOL,
    ORACLE_PASSWORD,
    ORACLE_PORT,
    ORACLE_SERVICE_NAME,
    ORACLE_USER,
)

logger = logging.getLogger(__name__)


class OracleError(Exception):
    """Oracle 操作异常基类。"""


class OracleHelper:
    """Oracle 数据库助手（连接池 + PLM 双表操作）。

    线程安全（读写通过连接池+锁）。连接池延迟初始化。
    """

    def __init__(self) -> None:
        self._pool: oracledb.ConnectionPool | None = None
        self._lock = threading.Lock()

    # ── 连接管理 ────────────────────────────────────────────────

    def _dsn(self) -> str:
        return oracledb.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE_NAME)

    def _get_pool(self) -> oracledb.ConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    if not ORACLE_HOST or not ORACLE_USER:
                        raise OracleError(
                            "Oracle 未配置：请设置 ORACLE_HOST / ORACLE_USER"
                        )
                    self._pool = oracledb.create_pool(
                        user=ORACLE_USER,
                        password=ORACLE_PASSWORD,
                        dsn=self._dsn(),
                        min=ORACLE_MIN_POOL,
                        max=ORACLE_MAX_POOL,
                        timeout=30,
                    )
                    logger.info(
                        f"Oracle 连接池已建立: {ORACLE_USER}@{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE_NAME}"
                    )
        return self._pool

    def _execute(self, sql: str, params: list | dict | None = None) -> list[dict]:
        """执行 SQL 并返回 dict 列表（自动归还连接到池）。"""
        pool = self._get_pool()
        with pool.acquire() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                if cur.description:
                    cols = [d[0].lower() for d in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                return []

    # ── R_V_TD_FILEPATH 操作 ────────────────────────────────────

    def fetch_pending_tasks(self, limit: int = 50) -> list[dict]:
        """查询未处理的 PLM 任务（ISPROCESS='0'）。

        返回每行含: docnumber, th, bbh, filename, work_seq
        """
        sql = """
            SELECT DOCNUMBER, TH, BBH, FILENAME, WORK_SEQ
            FROM R_V_TD_FILEPATH
            WHERE ISPROCESS = '0' AND ROWNUM <= :limit
            ORDER BY UPD_TIMESTAMP ASC
        """
        return self._execute(sql, {"limit": limit})

    def mark_task_processing(self, docnumber: str, th: str, bbh: str,
                             filename: str, work_seq: str) -> None:
        """标记任务为处理中（ISPROCESS='9'），防止重复拉取。"""
        sql = """
            UPDATE R_V_TD_FILEPATH
            SET ISPROCESS = '9', UPD_USER = 'LMT', UPD_TIMESTAMP = SYSDATE
            WHERE DOCNUMBER = :doc AND TH = :th AND BBH = :bbh
              AND FILENAME = :fn AND WORK_SEQ = :ws
              AND ISPROCESS = '0'
        """
        self._execute(sql, {
            "doc": docnumber, "th": th, "bbh": bbh,
            "fn": filename, "ws": work_seq,
        })

    # ── SIPM197 操作 ────────────────────────────────────────────

    def fetch_file_location(self, th: str, bbh: str, fname: str,
                            docnumber: str, work_seq: str) -> str | None:
        """查询 SIPM197 获取文件相对路径 LOCATION。"""
        sql = """
            SELECT LOCATION FROM SIPM197
            WHERE DEL = 0 AND WKAID <> '3'
              AND TH = :th AND BBH = :bbh AND FNAME = :fn
              AND DOCNUMBER = :doc AND WORK_SEQ = :ws
        """
        rows = self._execute(sql, {
            "th": th, "bbh": bbh, "fn": fname,
            "doc": docnumber, "ws": work_seq,
        })
        return rows[0]["location"] if rows else None

    # ── 回写操作 ───────────────────────────────────────────────

    def update_sipm197(self, th: str, bbh: str, fname: str,
                       docnumber: str, work_seq: str,
                       ocr_flag: str) -> None:
        """处理完成后回写 SIPM197。"""
        sql = """
            UPDATE SIPM197
            SET OCR = :ocr, PTIME = SYSDATE
            WHERE DEL = 0 AND WKAID <> '3'
              AND TH = :th AND BBH = :bbh AND FNAME = :fn
              AND DOCNUMBER = :doc AND WORK_SEQ = :ws
        """
        self._execute(sql, {
            "ocr": ocr_flag, "th": th, "bbh": bbh, "fn": fname,
            "doc": docnumber, "ws": work_seq,
        })

    def update_filepath(self, th: str, bbh: str, filename: str,
                        docnumber: str, work_seq: str,
                        ocr_flag: str) -> None:
        """处理完成后回写 R_V_TD_FILEPATH。"""
        sql = """
            UPDATE R_V_TD_FILEPATH
            SET ISPROCESS = '1', OCR = :ocr,
                UPD_USER = 'LMT', UPD_TIMESTAMP = SYSDATE
            WHERE TH = :th AND BBH = :bbh AND FILENAME = :fn
              AND DOCNUMBER = :doc AND WORK_SEQ = :ws
        """
        self._execute(sql, {
            "ocr": ocr_flag, "th": th, "bbh": bbh, "fn": filename,
            "doc": docnumber, "ws": work_seq,
        })

    # ── 种子数据 ────────────────────────────────────────────────

    def seed_dev_data(self, entries: list[dict]) -> None:
        """批量插入种子数据到 R_V_TD_FILEPATH 和 SIPM197。

        entries 每个 dict 需含：
            docnumber, th, bbh, filename, work_seq, location
        """
        pool = self._get_pool()
        with pool.acquire() as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                for e in entries:
                    # 插入 R_V_TD_FILEPATH（ISPROCESS='0' 待处理）
                    cur.execute("""
                        INSERT INTO R_V_TD_FILEPATH
                            (DOCNUMBER, TH, BBH, FILENAME, WORK_SEQ,
                             ISPROCESS, OCR, UPD_USER, UPD_TIMESTAMP)
                        VALUES (:doc, :th, :bbh, :fn, :ws,
                                '0', 'N', 'SEED', SYSDATE)
                    """, {
                        "doc": e["docnumber"], "th": e["th"],
                        "bbh": e["bbh"], "fn": e["filename"],
                        "ws": e["work_seq"],
                    })
                    # 插入 SIPM197
                    cur.execute("""
                        INSERT INTO SIPM197
                            (DOCNUMBER, TH, BBH, FNAME, LOCATION,
                             OCR, WORK_SEQ)
                        VALUES (:doc, :th, :bbh, :fn, :loc,
                                'N', :ws)
                    """, {
                        "doc": e["docnumber"], "th": e["th"],
                        "bbh": e["bbh"], "fn": e["filename"],
                        "loc": e["location"], "ws": e["work_seq"],
                    })
                conn.commit()
        logger.info(f"种子数据插入完成: {len(entries)} 条")
