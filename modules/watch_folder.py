"""Watch folder daemon — 文件夹监听入口。

⚠️ 定位说明（v4 方案）：
  本模块是【测试 / 备用入口】，不是生产主入口。
  - 生产 PLM 对接 = Oracle 双表直连（R_V_TD_FILEPATH + SIPM197），
    由对方实现，对方的对接层直接 import 并调用 modules.batch_processor
    .process_single_file()，不经过本模块。
  - 本模块（丢文件进 inbox 文件夹即处理）保留用于【本地端到端测试】：
    在没有真实 Oracle 环境时，是验证"检测→OCR→替换→写O/N日志"整条
    处理链路的唯一便捷入口。
  - 两个入口共用同一处理核心 process_single_file，切换只是"启动谁"，
    不需要改代码、不需要重新打包。生产部署时本 watcher 不启动即可。

职责（单线程，轮询模式，不依赖 watchdog）：
  1. 每 `WATCH_SCAN_INTERVAL` 秒扫描 inbox\
  2. 跟踪每个新文件的 (size, mtime)；连续 `WATCH_STABILITY_SECONDS` 秒不变 → 视为稳定
  3. 把稳定文件移到 processing\（防止重复触发）
  4. 解析 drawing_no / revision，入 SQLite 队列 (source='watch_folder')
  5. 失败的文件交给 Worker 在处理失败时移到 failed\（不归 watcher 管）
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    SUPPORTED_EXTENSIONS,
    WATCH_INBOX_DIR,
    WATCH_PROCESSING_DIR,
    WATCH_SCAN_INTERVAL,
    WATCH_STABILITY_SECONDS,
)
from modules.filename_parser import parse as parse_filename
from modules.job_queue import JobQueue

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    """正在做稳定性观察的文件。"""
    size: int
    mtime: float
    first_seen: float
    last_change: float


class WatchFolder:
    """Inbox 监视器。

    Lifecycle:
      - start() 起一个 daemon 线程
      - 线程内：轮询 inbox → 稳定性判定 → 移到 processing → enqueue
      - stop() 设置 stop event，线程在下一轮退出
    """

    def __init__(
        self,
        queue: JobQueue,
        inbox_dir: str | Path | None = None,
        processing_dir: str | Path | None = None,
        scan_interval: float | None = None,
        stability_seconds: float | None = None,
        prefixes: list[str] | None = None,
    ):
        self.queue = queue
        self.inbox_dir = str(inbox_dir or WATCH_INBOX_DIR)
        self.processing_dir = str(processing_dir or WATCH_PROCESSING_DIR)
        self.scan_interval = scan_interval if scan_interval is not None else WATCH_SCAN_INTERVAL
        self.stability_seconds = stability_seconds if stability_seconds is not None else WATCH_STABILITY_SECONDS
        self.prefixes = prefixes

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: dict[str, _Pending] = {}

    # ── 生命周期 ──
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        Path(self.inbox_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processing_dir).mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="watch-folder", daemon=True)
        self._thread.start()
        logger.info(
            f"WatchFolder 已启动；inbox={self.inbox_dir} processing={self.processing_dir} "
            f"scan={self.scan_interval}s stability={self.stability_seconds}s"
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("WatchFolder 已停止")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── 主循环 ──
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except Exception as e:  # 保险层：扫描异常不让线程死
                logger.error(f"WatchFolder scan 异常: {e}", exc_info=True)
            if self._stop.wait(self.scan_interval):
                break

    def _scan_once(self) -> None:
        try:
            names = os.listdir(self.inbox_dir)
        except FileNotFoundError:
            # inbox 被删了；重新建一次再下轮扫
            Path(self.inbox_dir).mkdir(parents=True, exist_ok=True)
            return

        now = time.time()
        seen: set[str] = set()

        for name in names:
            full = os.path.join(self.inbox_dir, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            seen.add(full)
            try:
                st = os.stat(full)
            except OSError:
                continue

            prev = self._pending.get(full)
            if prev is None:
                self._pending[full] = _Pending(
                    size=st.st_size, mtime=st.st_mtime,
                    first_seen=now, last_change=now,
                )
                continue
            # 文件还在写 → 更新 last_change
            if prev.size != st.st_size or prev.mtime != st.st_mtime:
                prev.size = st.st_size
                prev.mtime = st.st_mtime
                prev.last_change = now
                continue
            # 稳定窗口达成 → 入队
            if (now - prev.last_change) >= self.stability_seconds:
                self._promote(full)

        # 清理已不在 inbox 的 pending 条目（被外部删除/已被本 watcher 移走）
        for stale in list(self._pending.keys()):
            if stale not in seen:
                self._pending.pop(stale, None)

    def _promote(self, src_path: str) -> None:
        """稳定文件：移到 processing，入队 (source='watch_folder')。"""
        name = os.path.basename(src_path)
        dst_name = self._unique_name(self.processing_dir, name)
        dst_path = os.path.join(self.processing_dir, dst_name)

        try:
            shutil.move(src_path, dst_path)
        except OSError as e:
            logger.warning(f"WatchFolder 移动 {name} 到 processing 失败（下轮重试）: {e}")
            return
        finally:
            self._pending.pop(src_path, None)

        parsed = parse_filename(name, prefixes=self.prefixes)
        try:
            job_id = self.queue.enqueue(
                source="watch_folder",
                source_file=name,
                file_path=dst_path,
                drawing_no=parsed.drawing_no,
                revision=parsed.revision,
            )
        except Exception as e:
            logger.error(f"WatchFolder enqueue 失败 ({name}): {e}", exc_info=True)
            return

        logger.info(
            f"WatchFolder enqueued job={job_id} file={name} "
            f"drawing_no={parsed.drawing_no} rev={parsed.revision} "
            f"matched={parsed.matched}"
        )

    @staticmethod
    def _unique_name(folder: str, name: str) -> str:
        """若 folder 下已有同名文件，追加 _NNN 以免覆盖。"""
        if not os.path.exists(os.path.join(folder, name)):
            return name
        stem, ext = os.path.splitext(name)
        i = 1
        while True:
            candidate = f"{stem}_{i:03d}{ext}"
            if not os.path.exists(os.path.join(folder, candidate)):
                return candidate
            i += 1


# ── 独立运行入口（仅 watcher，不带 Worker）──
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    queue = JobQueue()
    watcher = WatchFolder(queue)
    watcher.start()
    logger.info("WatchFolder 主循环运行中；Ctrl+C 退出")
    try:
        while watcher.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在停止 WatchFolder ...")
        watcher.stop()


if __name__ == "__main__":
    _main()
