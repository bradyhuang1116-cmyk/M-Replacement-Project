"""图纸处理队列 Worker，负责本地处理与 PLM 交付。"""
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
    WORKER_MAX_RETRY,
    WORKER_OUTPUT_DIR,
    WORKER_POLL_INTERVAL,
    Y_BOXES_CSV_NAME,
)
from modules.job_queue import Job, JobQueue

logger = logging.getLogger(__name__)


class Worker:
    """单线程队列消费者。"""

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
        self.watch_output_dir = str(watch_output_dir or WATCH_OUTPUT_DIR)
        self.watch_failed_dir = str(watch_failed_dir or WATCH_FAILED_DIR)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._vlm_ready = False

        from modules.process_log import ProcessLog

        self._process_log = ProcessLog()

    @staticmethod
    def build_plm_remote_output_path(job: Job, artifact_path: str) -> str:
        """统一构建 PLM 远端与本地镜像使用的输出路径。"""
        import config as cfg

        return os.path.join(
            cfg.PLM_OUTPUT_BASE_DIR,
            job.work_seq or "",
            f"{job.drawing_no}-{job.revision}",
            os.path.basename(artifact_path),
        )

    @staticmethod
    def _resolve_output_base_dir(output_root: str | Path, artifact_path: str) -> str:
        """优先按实际产物路径反推 OUTPUT 根目录，兼容历史恢复。"""
        configured_base = os.path.join(str(output_root), OUTPUT_SUBDIR)
        artifact_abs = os.path.abspath(artifact_path)
        configured_abs = os.path.abspath(configured_base)
        if artifact_abs == configured_abs or artifact_abs.startswith(configured_abs + os.sep):
            return configured_base

        parts = Path(artifact_path).parts
        output_index = next(
            (index for index, part in enumerate(parts) if part.lower() == OUTPUT_SUBDIR.lower()),
            None,
        )
        if output_index is not None:
            return os.path.join(*parts[: output_index + 1])
        return configured_base

    @classmethod
    def build_uploaded_archive_path(cls, output_root: str | Path, artifact_path: str) -> str:
        """构建 uploaded 归档路径，并保持与正常发送流程一致。"""
        base_output = cls._resolve_output_base_dir(output_root, artifact_path)
        uploaded_root = os.path.join(os.path.dirname(base_output.rstrip("\\/")), "uploaded")
        output_relative = os.path.relpath(artifact_path, base_output)
        return os.path.join(uploaded_root, output_relative)

    @staticmethod
    def infer_method_from_artifact_path(artifact_path: str) -> str:
        """根据产物路径推断处理方式，供恢复和补发时复用。"""
        normalized = artifact_path.replace("/", os.sep).replace("\\", os.sep)
        parts = {part.lower() for part in normalized.split(os.sep) if part}
        return "vector" if PDF_REPLACEMENT_SUBDIR.lower() in parts else "ocr"

    @staticmethod
    def _is_plm_delivery_job(job: Job) -> bool:
        """只有自动 PLM 入队任务参与远端交付与恢复。"""
        return job.source == "api" and bool(job.docnumber and job.work_seq)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        stale = self.queue.reset_stale_running()
        if stale:
            logger.info("Worker 启动时将 %s 条残留 running 任务重置为 pending", stale)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="job-worker", daemon=True)
        self._thread.start()
        logger.info("Worker 已启动；output_root=%s", self.output_root)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("Worker 已停止")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _sync_config(self) -> None:
        import config as cfg

        self.poll_interval = cfg.WORKER_POLL_INTERVAL
        self.max_retry = cfg.WORKER_MAX_RETRY
        self.watch_output_dir = str(cfg.WATCH_OUTPUT_DIR)
        self.watch_failed_dir = str(cfg.WATCH_FAILED_DIR)
        self.output_root = str(cfg.WORKER_OUTPUT_DIR)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self._sync_config()
            job = self.queue.claim_next_pending()
            if job is None:
                if self._stop.wait(self.poll_interval):
                    break
                continue
            try:
                self._process_one(job)
            except Exception:
                logger.error(
                    "Worker 处理任务 %s 时发生未捕获异常\n%s",
                    job.id,
                    traceback.format_exc(),
                )
                self._safe_mark_failed(job.id, "未捕获的 Worker 异常")

    def _write_plm_oracle_updates(self, job: Job, method: str) -> None:
        """执行 PLM 交付完成后的 Oracle 双表回写。"""
        from modules.oracle_helper import OracleHelper
        from modules.process_log import method_to_ocr_flag

        ocr_flag = method_to_ocr_flag(method)
        oracle = OracleHelper()
        oracle.update_sipm197(
            th=job.drawing_no or "",
            bbh=job.revision or "",
            fname=job.source_file,
            docnumber=job.docnumber or "",
            work_seq=job.work_seq or "",
            ocr_flag=ocr_flag,
        )
        oracle.update_filepath(
            th=job.drawing_no or "",
            bbh=job.revision or "",
            filename=job.source_file,
            docnumber=job.docnumber or "",
            work_seq=job.work_seq or "",
            ocr_flag=ocr_flag,
        )

    def _complete_plm_delivery(
        self,
        job: Job,
        method: str,
        remote_path: str,
        uploaded_path: str,
    ) -> tuple[bool, str]:
        """产物已上传归档后补做 Oracle 回写。"""
        try:
            self._write_plm_oracle_updates(job, method)
        except Exception as exc:
            err = str(exc)
            self.queue.mark_delivery_uploaded(
                job.id,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
                error_msg=err,
            )
            logger.warning("[job=%s] Oracle PLM 回写失败：%s", job.id, err)
            return False, err

        self.queue.mark_delivery_complete(
            job.id,
            remote_path=remote_path,
            uploaded_path=uploaded_path,
        )
        logger.info("[job=%s] PLM 交付完成", job.id)
        return True, ""

    def _deliver_plm_output(
        self,
        job: Job,
        artifact_path: str,
        method: str,
        remote_path: str | None = None,
        uploaded_path: str | None = None,
    ) -> tuple[bool, str]:
        """执行 PLM 交付：上传远端、归档 uploaded、回写 Oracle。"""
        remote_path = remote_path or self.build_plm_remote_output_path(job, artifact_path)
        uploaded_path = uploaded_path or self.build_uploaded_archive_path(self.output_root, artifact_path)
        self.queue.mark_delivery_pending(job.id, remote_path, uploaded_path)

        if not artifact_path or not os.path.isfile(artifact_path):
            err = f"产物文件不存在：{artifact_path}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM 交付失败：%s", job.id, err)
            return False, err

        local_plm_path = remote_path
        try:
            Path(os.path.dirname(local_plm_path)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, local_plm_path)
            logger.info("[job=%s] 已复制产物到 PLM 本地镜像目录：%s", job.id, local_plm_path)
        except OSError as exc:
            err = f"复制产物到 PLM 本地镜像目录失败：{exc}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM 交付失败：%s", job.id, err)
            return False, err

        from modules.winscp_client import WinSCPClient

        ok, err = WinSCPClient().upload(local_plm_path, remote_path)
        if not ok:
            err = err or "WinSCP 上传失败"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM 上传失败：%s", job.id, err)
            return False, err
        logger.info("[job=%s] 已上传产物到 PLM：%s", job.id, remote_path)

        try:
            Path(os.path.dirname(uploaded_path)).mkdir(parents=True, exist_ok=True)
            shutil.move(artifact_path, uploaded_path)
            logger.info("[job=%s] 已归档产物到 uploaded：%s", job.id, uploaded_path)
        except OSError as exc:
            err = f"移动产物到 uploaded 失败：{exc}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM 归档失败：%s", job.id, err)
            return False, err

        self.queue.mark_delivery_uploaded(
            job.id,
            remote_path=remote_path,
            uploaded_path=uploaded_path,
        )
        return self._complete_plm_delivery(job, method, remote_path, uploaded_path)

    def _resolve_plm_retry_paths(self, job: Job) -> tuple[str, str, str | None]:
        reference_path = job.plm_uploaded_path or job.result_path
        if not reference_path:
            return "", "", None

        remote_path = job.plm_remote_path or self.build_plm_remote_output_path(job, reference_path)
        uploaded_path = job.plm_uploaded_path or self.build_uploaded_archive_path(
            self.output_root,
            job.result_path or reference_path,
        )
        return remote_path, uploaded_path, reference_path

    def can_retry_plm_delivery(self, job: Job) -> tuple[bool, str]:
        """检查当前任务是否仍具备 PLM 手动补发条件。"""
        if not self._is_plm_delivery_job(job):
            return False, "当前任务不是可补发的 PLM 自动任务"

        _, uploaded_path, reference_path = self._resolve_plm_retry_paths(job)
        if not reference_path:
            return False, "缺少结果路径，无法恢复 PLM 交付"
        if uploaded_path and os.path.isfile(uploaded_path):
            return True, ""
        if job.result_path and os.path.isfile(job.result_path):
            return True, ""
        return False, "OUTPUT 与 uploaded 中都未找到产物文件"

    def recover_single_plm_delivery(self, job_id: int) -> tuple[bool, str]:
        """按任务 ID 补发单条失败的 PLM 交付。"""
        self._sync_config()
        job = self.queue.get(job_id)
        if job is None:
            return False, f"job not found: {job_id}"
        return self._recover_one_plm_delivery(job)

    def recover_pending_plm_deliveries(self, limit: int = 100) -> None:
        """重启后恢复未完成的 PLM 交付任务。"""
        jobs = self.queue.list_recoverable_plm_deliveries(limit=limit)
        if not jobs:
            return

        logger.info("开始恢复 %s 条未完成的 PLM 交付任务", len(jobs))
        for job in jobs:
            self._sync_config()
            try:
                self._recover_one_plm_delivery(job)
            except Exception:
                logger.warning(
                    "[job=%s] PLM 交付恢复时发生未预期异常\n%s",
                    job.id,
                    traceback.format_exc(),
                )

    def _recover_one_plm_delivery(self, job: Job) -> tuple[bool, str]:
        """按现有文件状态继续补发，避免路径规则分叉。"""
        if not self._is_plm_delivery_job(job):
            return False, "当前任务不是可补发的 PLM 自动任务"

        remote_path, uploaded_path, reference_path = self._resolve_plm_retry_paths(job)
        if not reference_path:
            err = "缺少结果路径，无法恢复 PLM 交付"
            logger.warning("[job=%s] %s", job.id, err)
            self.queue.mark_delivery_failed(job.id, err)
            return False, err

        if uploaded_path and os.path.isfile(uploaded_path):
            method = self.infer_method_from_artifact_path(uploaded_path)
            self.queue.mark_delivery_uploaded(
                job.id,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
                error_msg=job.plm_delivery_error,
            )
            logger.info("[job=%s] 检测到 uploaded 产物，仅继续补做 Oracle 回写", job.id)
            return self._complete_plm_delivery(job, method, remote_path, uploaded_path)

        if job.result_path and os.path.isfile(job.result_path):
            method = self.infer_method_from_artifact_path(job.result_path)
            logger.info("[job=%s] 检测到 OUTPUT 产物，继续执行 PLM 上传流程", job.id)
            return self._deliver_plm_output(
                job,
                job.result_path,
                method,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )

        err = "OUTPUT 与 uploaded 中都未找到产物文件"
        logger.warning("[job=%s] 无法恢复 PLM 交付：%s", job.id, err)
        self.queue.mark_delivery_failed(
            job.id,
            err,
            remote_path=remote_path,
            uploaded_path=uploaded_path,
        )
        return False, err

    def _process_one(self, job: Job) -> None:
        logger.info("[job=%s] 开始处理 %s（source=%s）", job.id, job.file_path, job.source)
        if not os.path.isfile(job.file_path):
            self._handle_failure(job, f"文件不存在：{job.file_path}")
            return

        if self.ensure_vlm and not self._vlm_ready:
            ok, msg = self._ensure_vlm_ready()
            if not ok:
                self._handle_failure(job, f"VLM 启动失败：{msg}")
                return
            self._vlm_ready = True

        base_output = os.path.join(self.output_root, OUTPUT_SUBDIR)
        vlmocr_dir = os.path.join(base_output, VLMOCR_SUBDIR)
        pdf_dir = os.path.join(base_output, PDF_REPLACEMENT_SUBDIR)
        os.makedirs(vlmocr_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        start = time.time()
        try:
            from modules.batch_processor import process_single_file

            result = process_single_file(job.file_path, self.output_root)
        except Exception as exc:
            logger.error("[job=%s] 处理失败：%s\n%s", job.id, exc, traceback.format_exc())
            self._handle_failure(job, str(exc))
            return
        finally:
            gc.collect()
            try:
                import paddle

                paddle.device.cuda.empty_cache()
            except Exception:
                pass

        out_path = result.get("output_path", "")
        method = result.get("method", "ocr")
        final_path = out_path
        if out_path and os.path.exists(out_path):
            target_dir = pdf_dir if method == "vector" else vlmocr_dir
            target_path = os.path.join(target_dir, os.path.basename(out_path))
            try:
                shutil.move(out_path, target_path)
                final_path = target_path
            except OSError as exc:
                logger.warning("[job=%s] 移动产物失败，保留原路径：%s", job.id, exc)

        for sub in ("vector", "ocr"):
            tmp_dir = os.path.join(self.output_root, sub)
            try:
                if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                    os.rmdir(tmp_dir)
            except OSError:
                pass

        try:
            from modules.factory_note_pixel import flush_y_boxes_csv

            flush_y_boxes_csv(os.path.join(base_output, Y_BOXES_CSV_NAME))
        except Exception as exc:
            logger.warning("[job=%s] 写入 y_boxes.csv 失败：%s", job.id, exc)

        elapsed = time.time() - start
        replacements = result.get("total", 0)
        logger.info(
            "[job=%s] 处理完成 method=%s replacements=%s elapsed=%.1fs -> %s",
            job.id,
            method,
            replacements,
            elapsed,
            final_path,
        )

        if job.source == "watch_folder" and final_path and os.path.exists(final_path):
            try:
                Path(self.watch_output_dir).mkdir(parents=True, exist_ok=True)
                out_name = os.path.basename(final_path)
                watch_out_path = os.path.join(self.watch_output_dir, out_name)
                if os.path.exists(watch_out_path):
                    stem, ext = os.path.splitext(out_name)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    watch_out_path = os.path.join(self.watch_output_dir, f"{stem}_{ts}{ext}")
                shutil.copy2(final_path, watch_out_path)
                logger.info("[job=%s] 已复制到 watch_output：%s", job.id, watch_out_path)
                final_path = watch_out_path
            except OSError as exc:
                logger.warning("[job=%s] 复制到 watch_output 失败：%s", job.id, exc)

        self.queue.mark_done(job.id, final_path)

        try:
            from modules.process_log import method_to_ocr_flag

            self._process_log.record(
                drawing_no=job.drawing_no,
                revision=job.revision,
                ocr_flag=method_to_ocr_flag(method),
                status="success",
                filename=job.source_file,
            )
        except Exception as exc:
            logger.warning("[job=%s] 写入处理日志失败：%s", job.id, exc)

        if self._is_plm_delivery_job(job):
            self._deliver_plm_output(job, final_path, method)

        if job.file_path and os.path.isfile(job.file_path):
            try:
                os.remove(job.file_path)
            except OSError as exc:
                logger.warning("[job=%s] 清理 processing 源文件失败：%s", job.id, exc)

    def _handle_failure(self, job: Job, err: str) -> None:
        self.queue.mark_failed(job.id, err)
        if self.max_retry > 0 and self.queue.requeue_for_retry(job.id, self.max_retry):
            logger.warning(
                "[job=%s] 处理失败，已重新入队（max_retry=%s）：%s",
                job.id,
                self.max_retry,
                err,
            )
            return

        logger.error("[job=%s] 处理失败且不再重试：%s", job.id, err)
        try:
            self._process_log.record(
                drawing_no=job.drawing_no,
                revision=job.revision,
                ocr_flag="N",
                status="failed",
                filename=job.source_file,
            )
        except Exception as exc:
            logger.warning("[job=%s] 写入失败日志失败：%s", job.id, exc)

        if job.source == "watch_folder" and os.path.isfile(job.file_path):
            try:
                Path(self.watch_failed_dir).mkdir(parents=True, exist_ok=True)
                dst = os.path.join(self.watch_failed_dir, os.path.basename(job.file_path))
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(os.path.basename(job.file_path))
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    dst = os.path.join(self.watch_failed_dir, f"{stem}_{ts}{ext}")
                shutil.move(job.file_path, dst)
                logger.info("[job=%s] 已移动源文件到 failed：%s", job.id, dst)
            except OSError as exc:
                logger.warning("[job=%s] 移动源文件到 failed 失败：%s", job.id, exc)

    def _safe_mark_failed(self, job_id: int, err: str) -> None:
        try:
            self.queue.mark_failed(job_id, err)
        except Exception:
            logger.error("标记 job=%s 为 failed 时发生异常\n%s", job_id, traceback.format_exc())

    def _ensure_vlm_ready(self) -> tuple[bool, str]:
        try:
            from modules.docker_manager import ensure_vlm_ready

            return ensure_vlm_ready()
        except Exception as exc:
            return False, f"ensure_vlm_ready 异常：{exc}"


def _main() -> None:
    """独立运行 Worker 的调试入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from config import PADDLEOCR_API_URL, VLM_PROVIDER

    logger.info(
        "VLM 模式：%s%s",
        VLM_PROVIDER,
        f" -> {PADDLEOCR_API_URL}" if VLM_PROVIDER == "paddleocr_api" else " -> 本地 Docker",
    )
    queue = JobQueue()
    worker = Worker(queue)
    worker.start()
    logger.info("Worker 主循环运行中，按 Ctrl+C 退出")
    try:
        while worker.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在停止 Worker")
        worker.stop()


if __name__ == "__main__":
    _main()
