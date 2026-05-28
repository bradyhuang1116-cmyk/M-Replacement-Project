"""队列 Worker — §12 Phase 2。

单进程单线程顺序消费 `JobQueue`，独占 GPU。
- 不在本阶段自动挂到 FastAPI startup（保证当前 V2 后端不被破坏）
- 入口：`Worker(queue).start()` / `worker.stop()`，或独立脚本 `python -m modules.worker`
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import threading
import time
import traceback
from pathlib import Path

from config import (
    OUTPUT_SUBDIR,
    PDF_REPLACEMENT_SUBDIR,
    VLMOCR_SUBDIR,
    WATCH_FAILED_DIR,
    WATCH_OUTPUT_DIR,
    Y_BOXES_CSV_NAME,
    WORKER_MAX_RETRY,
    WORKER_OUTPUT_DIR,
    WORKER_POLL_INTERVAL,
)
from modules.job_queue import Job, JobQueue

logger = logging.getLogger(__name__)


class Worker:
    """单线程队列消费者。

    生命周期：
      - `start()` 起一个 daemon 线程
      - 线程内部循环：claim_next_pending → 处理 → mark_done / mark_failed
      - 队列空时 sleep `WORKER_POLL_INTERVAL` 秒
      - `stop()` 设置 stop event，线程在下一轮循环退出
    """

    def __init__(
        self,
        queue: JobQueue,
        output_root: str | Path | None = None,
        poll_interval: float | None = None,
        max_retry: int | None = None,
        ensure_vlm: bool = True,
        watch_output_dir: str | Path | None = None,
        watch_failed_dir: str | Path | None = None,
    ):
        self.queue = queue
        self.output_root = str(output_root or WORKER_OUTPUT_DIR)
        self.poll_interval = poll_interval if poll_interval is not None else WORKER_POLL_INTERVAL
        self.max_retry = max_retry if max_retry is not None else WORKER_MAX_RETRY
        self.ensure_vlm = ensure_vlm
        # watch_folder 任务的最终去向；source='api' 的任务不会用到
        self.watch_output_dir = str(watch_output_dir or WATCH_OUTPUT_DIR)
        self.watch_failed_dir = str(watch_failed_dir or WATCH_FAILED_DIR)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._vlm_ready = False

    # ── 生命周期 ──
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        # 把上一次没跑完就被强杀的 running 转回 pending
        n = self.queue.reset_stale_running()
        if n:
            logger.info(f"Worker 启动：将 {n} 条残留 running 任务转回 pending")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="job-worker", daemon=True)
        self._thread.start()
        logger.info(f"Worker 已启动；output_root={self.output_root}")

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("Worker 已停止")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── 主循环 ──
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            job = self.queue.claim_next_pending()
            if job is None:
                if self._stop.wait(self.poll_interval):
                    break
                continue
            try:
                self._process_one(job)
            except Exception:  # 保险层：任何异常都不让线程死掉
                logger.error(f"Worker 处理任务 {job.id} 时发生未捕获异常\n{traceback.format_exc()}")
                self._safe_mark_failed(job.id, "uncaught worker exception")

    # ── 单任务处理 ──
    def _process_one(self, job: Job) -> None:
        logger.info(f"[job={job.id}] 开始处理 {job.file_path} (source={job.source})")
        if not os.path.isfile(job.file_path):
            self._handle_failure(job, f"file not found: {job.file_path}")
            return

        if self.ensure_vlm and not self._vlm_ready:
            ok, msg = self._ensure_vlm_ready()
            if not ok:
                self._handle_failure(job, f"VLM 启动失败: {msg}")
                return
            self._vlm_ready = True

        # 输出目录结构：<output_root>/<OUTPUT_SUBDIR>/{VLMOCR|PDF_Replacement}/
        base_output = os.path.join(self.output_root, OUTPUT_SUBDIR)
        vlmocr_dir = os.path.join(base_output, VLMOCR_SUBDIR)
        pdf_dir = os.path.join(base_output, PDF_REPLACEMENT_SUBDIR)
        os.makedirs(vlmocr_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        start = time.time()
        try:
            from modules.batch_processor import process_single_file
            result = process_single_file(job.file_path, self.output_root)
        except Exception as e:
            logger.error(f"[job={job.id}] 处理失败: {e}\n{traceback.format_exc()}")
            self._handle_failure(job, str(e))
            return
        finally:
            gc.collect()
            try:
                import paddle
                paddle.device.cuda.empty_cache()
            except Exception:
                pass

        # 把产物从 process_single_file 的 vector/ocr 临时子目录搬到最终位置
        out_path = result.get("output_path", "")
        method = result.get("method", "ocr")
        final_path = out_path
        if out_path and os.path.exists(out_path):
            target_dir = pdf_dir if method == "vector" else vlmocr_dir
            target_path = os.path.join(target_dir, os.path.basename(out_path))
            try:
                shutil.move(out_path, target_path)
                final_path = target_path
            except OSError as e:
                logger.warning(f"[job={job.id}] 移动产物失败（保留原路径）: {e}")

        # 清理空临时目录
        for sub in ("vector", "ocr"):
            d = os.path.join(self.output_root, sub)
            try:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass

        # y_boxes.csv 汇总（与 API 路径一致；多任务会累积写入）
        try:
            from modules.factory_note_pixel import flush_y_boxes_csv
            flush_y_boxes_csv(os.path.join(base_output, Y_BOXES_CSV_NAME))
        except Exception as e:
            logger.warning(f"[job={job.id}] y_boxes.csv 写入失败（不影响结果）: {e}")

        elapsed = time.time() - start
        n_repl = result.get("total", 0)
        logger.info(
            f"[job={job.id}] 完成 method={method} replacements={n_repl} "
            f"elapsed={elapsed:.1f}s → {final_path}"
        )

        # watch_folder 任务：把产物再拷贝一份到 WATCH_OUTPUT_DIR（PLM 监听这里）
        if job.source == "watch_folder" and final_path and os.path.exists(final_path):
            try:
                Path(self.watch_output_dir).mkdir(parents=True, exist_ok=True)
                out_name = os.path.basename(final_path)
                watch_out_path = os.path.join(self.watch_output_dir, out_name)
                # 同名冲突时加时间戳后缀（§3.3 "同名加时间戳"）
                if os.path.exists(watch_out_path):
                    stem, ext = os.path.splitext(out_name)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    watch_out_path = os.path.join(self.watch_output_dir, f"{stem}_{ts}{ext}")
                shutil.copy2(final_path, watch_out_path)
                logger.info(f"[job={job.id}] 已复制到 watch_output: {watch_out_path}")
                final_path = watch_out_path
            except OSError as e:
                logger.warning(f"[job={job.id}] 复制到 watch_output 失败（保留内部路径）: {e}")

        self.queue.mark_done(job.id, final_path)

    # ── 失败处理 ──
    def _handle_failure(self, job: Job, err: str) -> None:
        # 先标 failed，再视重试策略尝试 requeue
        self.queue.mark_failed(job.id, err)
        if self.max_retry > 0:
            requeued = self.queue.requeue_for_retry(job.id, self.max_retry)
            if requeued:
                logger.warning(f"[job={job.id}] 失败已重新排队 (max_retry={self.max_retry}): {err}")
                return
        logger.error(f"[job={job.id}] 失败（不再重试）: {err}")
        # watch_folder 任务彻底失败：把 processing\ 里的源文件搬到 failed\
        if job.source == "watch_folder" and os.path.isfile(job.file_path):
            try:
                Path(self.watch_failed_dir).mkdir(parents=True, exist_ok=True)
                dst = os.path.join(self.watch_failed_dir, os.path.basename(job.file_path))
                # 同名追加时间戳避免覆盖历史失败
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(os.path.basename(job.file_path))
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    dst = os.path.join(self.watch_failed_dir, f"{stem}_{ts}{ext}")
                shutil.move(job.file_path, dst)
                logger.info(f"[job={job.id}] 源文件已移到 failed: {dst}")
            except OSError as e:
                logger.warning(f"[job={job.id}] 移动源文件到 failed 失败: {e}")

    def _safe_mark_failed(self, job_id: int, err: str) -> None:
        try:
            self.queue.mark_failed(job_id, err)
        except Exception:
            logger.error(f"标记 job={job_id} failed 时发生异常\n{traceback.format_exc()}")

    # ── VLM 预热（仅 OCR 路径需要；矢量 PDF 不需要，但启动一次开销可接受）──
    def _ensure_vlm_ready(self) -> tuple[bool, str]:
        try:
            from modules.docker_manager import ensure_vlm_ready
            return ensure_vlm_ready()
        except Exception as e:
            return False, f"ensure_vlm_ready 异常: {e}"


# ── 独立运行入口 ──
def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    queue = JobQueue()
    worker = Worker(queue)
    worker.start()
    logger.info("Worker 主循环运行中；Ctrl+C 退出")
    try:
        while worker.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在停止 Worker ...")
        worker.stop()


if __name__ == "__main__":
    _main()
