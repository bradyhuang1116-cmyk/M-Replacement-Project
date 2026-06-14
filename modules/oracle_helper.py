"""Oracle 数据库助手 — 连接管理与 PLM 双表 CRUD。

兼容 Oracle 11g/12c+：
- 默认使用 oracledb thin mode
- 当 ORACLE_THICK_MODE=true 时切换到 thick mode
- 同时支持 ORACLE_SERVICE_NAME 和 ORACLE_SID 两种 DSN 形式
"""
from __future__ import annotations

import logging
import threading

import oracledb

import config as cfg

logger = logging.getLogger(__name__)


_GLOBAL_POOL: oracledb.ConnectionPool | None = None
_GLOBAL_LOCK = threading.Lock()
_THICK_INIT_STATE = {"initialized": False, "lib_dir": None}


class OracleError(Exception):
    """Oracle 操作异常基类。"""


class OracleHelper:
    """Oracle 数据库助手（连接池 + PLM 双表操作）。

    线程安全（读写通过连接池+锁）。连接池为进程级共享实例，
    普通连接参数修改后可通过 reset_global_pool() 热重建。
    """

    def __init__(self) -> None:
        self._pool: oracledb.ConnectionPool | None = None
        self._lock = _GLOBAL_LOCK

    # ── 连接管理 ────────────────────────────────────────────────

    @staticmethod
    def reset_global_pool() -> None:
        global _GLOBAL_POOL
        with _GLOBAL_LOCK:
            if _GLOBAL_POOL is not None:
                try:
                    _GLOBAL_POOL.close(force=True)
                except Exception as e:
                    logger.warning("关闭 Oracle 连接池失败（忽略）: %s", e)
                finally:
                    _GLOBAL_POOL = None
            logger.info("Oracle 全局连接池已重置；下次访问将按新配置重建")

    @staticmethod
    def requires_restart_for_overrides(overrides: dict[str, object]) -> bool:
        return any(key in overrides for key in ("ORACLE_THICK_MODE", "ORACLE_CLIENT_LIB_DIR"))

    def _init_oracle_client_if_needed(self) -> None:
        if not cfg.ORACLE_THICK_MODE:
            return
        lib_dir = cfg.ORACLE_CLIENT_LIB_DIR or None
        if not _THICK_INIT_STATE["initialized"]:
            kwargs = {}
            if lib_dir:
                kwargs["lib_dir"] = lib_dir
            oracledb.init_oracle_client(**kwargs)
            _THICK_INIT_STATE["initialized"] = True
            _THICK_INIT_STATE["lib_dir"] = lib_dir
            return
        if _THICK_INIT_STATE["lib_dir"] != lib_dir:
            raise OracleError(
                "Oracle thick mode 客户端目录已在当前进程初始化；修改 ORACLE_CLIENT_LIB_DIR 或 ORACLE_THICK_MODE 后需要重启后端"
            )

    def _dsn_label(self) -> str:
        if cfg.ORACLE_SERVICE_NAME:
            return cfg.ORACLE_SERVICE_NAME
        if cfg.ORACLE_SID:
            return f"SID:{cfg.ORACLE_SID}"
        return "<missing-service-or-sid>"

    def connection_summary(self) -> str:
        return (
            f"{cfg.ORACLE_USER}@{cfg.ORACLE_HOST}:{cfg.ORACLE_PORT}/{self._dsn_label()} "
            f"(mode={'thick' if cfg.ORACLE_THICK_MODE else 'thin'})"
        )

    def _dsn(self) -> str:
        if cfg.ORACLE_SERVICE_NAME:
            return oracledb.makedsn(cfg.ORACLE_HOST, cfg.ORACLE_PORT, service_name=cfg.ORACLE_SERVICE_NAME)
        if cfg.ORACLE_SID:
            return oracledb.makedsn(cfg.ORACLE_HOST, cfg.ORACLE_PORT, sid=cfg.ORACLE_SID)
        raise OracleError(
            "Oracle 未配置：请设置 ORACLE_SERVICE_NAME 或 ORACLE_SID"
        )

    def _get_pool(self) -> oracledb.ConnectionPool:
        global _GLOBAL_POOL
        if _GLOBAL_POOL is None:
            with self._lock:
                if _GLOBAL_POOL is None:
                    if not cfg.ORACLE_HOST or not cfg.ORACLE_USER:
                        raise OracleError(
                            "Oracle 未配置：请设置 ORACLE_HOST / ORACLE_USER"
                        )
                    self._init_oracle_client_if_needed()
                    _GLOBAL_POOL = oracledb.create_pool(
                        user=cfg.ORACLE_USER,
                        password=cfg.ORACLE_PASSWORD,
                        dsn=self._dsn(),
                        min=cfg.ORACLE_MIN_POOL,
                        max=cfg.ORACLE_MAX_POOL,
                        timeout=30,
                    )
                    logger.info(
                        "Oracle 连接池已建立: %s@%s:%s/%s (mode=%s)",
                        cfg.ORACLE_USER,
                        cfg.ORACLE_HOST,
                        cfg.ORACLE_PORT,
                        self._dsn_label(),
                        "thick" if cfg.ORACLE_THICK_MODE else "thin",
                    )
        self._pool = _GLOBAL_POOL
        return _GLOBAL_POOL

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
