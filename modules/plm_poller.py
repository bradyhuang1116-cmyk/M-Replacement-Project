"""PLM Oracle 任务轮询器 — 从 Oracle 拉取待处理任务，下载文件并入队。

流程：
  1. 查询 R_V_TD_FILEPATH 中 ISPROCESS='0' 的待处理任务
  2. 对每个任务查询 SIPM197.LOCATION
  3. 下载文件到 processing 目录（WinSCP 或本地复制）
  4. 复制镜像到 ORACLE_PATH_PREFIX 路径（本地镜像远程结构）
  5. 直接入队到 JobQueue（带完整 Oracle 元数据）
  6. 标记任务为处理中 (ISPROCESS='9')
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

from config import ORACLE_PATH_PREFIX, WATCH_PROCESSING_DIR

logger = logging.getLogger(__name__)


class PlmPoller:
    """PLM Oracle 任务轮询器。

    生命周期：
      - start() 起一个 daemon 线程，定期扫 Oracle 拉取新任务
      - stop() 设置 stop event，线程在下一轮退出
    """

    def __init__(
        self,
        processing_dir: str | Path | None = None,
        poll_interval: float = 10.0,
        task_limit: int = 50,
    ):
        self.processing_dir = str(processing_dir or WATCH_PROCESSING_DIR)
        self.poll_interval = poll_interval
        self.task_limit = task_limit

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 已拉取过的 (DOCNUMBER, WORK_SEQ) 集合，避免同一批重复拷贝
        self._seen: set[tuple[str, str]] = set()

    # ── 生命周期 ──

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        Path(self.processing_dir).mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="plm-poller", daemon=True)
        self._thread.start()
        logger.info(
            f"PlmPoller 已启动；processing={self.processing_dir} "
            f"interval={self.poll_interval}s limit={self.task_limit}"
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("PlmPoller 已停止")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def get_seen_count(self) -> int:
        return len(self._seen)

    # ── 主循环 ──

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                logger.error(f"PlmPoller 轮询异常: {e}", exc_info=True)
            if self._stop.wait(self.poll_interval):
                break

    def _poll_once(self) -> None:
        """单次轮询：查 Oracle → 下载 → 镜像 → 入队 → 标记。"""
        from modules.oracle_helper import OracleHelper

        oracle = OracleHelper()
        logger.info("Oracle poll config: %s", oracle.connection_summary())

        try:
            tasks = oracle.fetch_pending_tasks(limit=self.task_limit)
        except Exception as e:
            logger.warning(f"Oracle 查询待处理任务失败（下轮重试）: {e}")
            return

        if not tasks:
            return

        logger.info(f"PlmPoller: 发现 {len(tasks)} 个待处理任务")

        for task in tasks:
            docnumber = task.get("docnumber", "").strip()
            th = task.get("th", "").strip()
            bbh = task.get("bbh", "").strip()
            filename = task.get("filename", "").strip()
            work_seq = task.get("work_seq", "").strip()

            if not all([docnumber, th, bbh, filename]):
                logger.warning(f"跳过无效任务: {task}")
                continue

            # 去重（同一批次下同一文件只拉一次）
            seen_key = (docnumber, work_seq) if work_seq else (docnumber, filename)
            if seen_key in self._seen:
                continue

            try:
                # 查询 SIPM197 获取文件相对路径
                location = oracle.fetch_file_location(th, bbh, filename, docnumber, work_seq)
                if not location:
                    logger.warning(f"未找到文件记录: {th}/{bbh}/{filename} ({docnumber})")
                    continue

                # 拼接远程绝对路径
                remote_path = os.path.join(ORACLE_PATH_PREFIX, location.strip().lstrip("\\/"))

                # 1. 下载到 processing 目录
                local_path = os.path.join(self.processing_dir, filename)
                from modules.winscp_client import WinSCPClient
                win = WinSCPClient()
                if win.is_available():
                    ok, err = win.download(remote_path, local_path)
                    if not ok:
                        logger.error(f"SFTP 下载失败: {remote_path} → {local_path}: {err}")
                        continue
                    logger.info(f"已下载 (SFTP) 到 processing: {remote_path} → {local_path}")
                else:
                    if not os.path.isfile(remote_path):
                        logger.warning(f"源文件不存在: {remote_path}")
                        continue
                    shutil.copy2(remote_path, local_path)
                    logger.info(f"已复制 (local) 到 processing: {filename}")

                # 2. 镜像副本到 ORACLE_PATH_PREFIX 路径（本地镜像远程结构）
                mirror_path = os.path.join(ORACLE_PATH_PREFIX, location.strip().lstrip("\\/"))
                mirror_dir = os.path.dirname(mirror_path)
                try:
                    Path(mirror_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path, mirror_path)
                    logger.info(f"已镜像到: {mirror_path}")
                except OSError as e:
                    logger.warning(f"镜像副本写入失败（不影响主流程）: {e}")

                # 3. 直接入队到 JobQueue
                try:
                    from modules.job_queue import JobQueue
                    queue = JobQueue()
                    queue.enqueue(
                        source="api",
                        source_file=filename,
                        file_path=local_path,
                        drawing_no=th,
                        revision=bbh,
                        docnumber=docnumber,
                        work_seq=work_seq,
                    )
                    logger.info(f"已入队: {filename} (drawing_no={th}, doc={docnumber})")
                except Exception as e:
                    logger.warning(f"入队失败（不影响主流程）: {e}")

                # 4. 标记为处理中
                oracle.mark_task_processing(docnumber, th, bbh, filename, work_seq)
                self._seen.add(seen_key)

            except Exception as e:
                logger.error(f"处理任务失败 ({th}/{bbh}/{filename}): {e}", exc_info=True)
                continue
