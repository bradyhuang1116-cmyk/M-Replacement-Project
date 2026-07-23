"""Background worker for queue processing and PLM delivery."""
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
from modules.job_queue import Job, JobQueue, review_status_allows_plm_delivery

logger = logging.getLogger(__name__)


class Worker:
    """Queue consumer — single or multi-threaded (WORKER_COUNT env)."""

    def __init__(
        self,
        queue: JobQueue,
        output_root: str | Path | None = None,
        poll_interval: float | None = None,
        max_retry: int | None = None,
        ensure_vlm: bool = True,
        watch_output_dir: str | Path | None = None,
        watch_failed_dir: str | Path | None = None,
        worker_count: int | None = None,
    ):
        self.queue = queue
        self.output_root = str(output_root or WORKER_OUTPUT_DIR)
        self.poll_interval = poll_interval if poll_interval is not None else WORKER_POLL_INTERVAL
        self.max_retry = max_retry if max_retry is not None else WORKER_MAX_RETRY
        self.ensure_vlm = ensure_vlm
        self.worker_count = worker_count if worker_count is not None else 1
        self.watch_output_dir = str(watch_output_dir or WATCH_OUTPUT_DIR)
        self.watch_failed_dir = str(watch_failed_dir or WATCH_FAILED_DIR)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._vlm_ready = False
        self._vlm_lock = threading.Lock()

        from modules.process_log import ProcessLog

        self._process_log = ProcessLog()

    @staticmethod
    def build_plm_remote_output_path(job: Job, artifact_path: str) -> str:
        """Build the target PLM output path."""
        import config as cfg

        return os.path.join(
            cfg.PLM_OUTPUT_BASE_DIR,
            job.work_seq or "",
            f"{job.drawing_no}-{job.revision}",
            os.path.basename(artifact_path),
        )

    @staticmethod
    def _resolve_output_base_dir(output_root: str | Path, artifact_path: str) -> str:
        """Resolve the OUTPUT root for a produced artifact."""
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
        """Build the uploaded archive path for a produced artifact."""
        base_output = cls._resolve_output_base_dir(output_root, artifact_path)
        uploaded_root = os.path.join(os.path.dirname(base_output.rstrip("\\/")), "uploaded")
        output_relative = os.path.relpath(artifact_path, base_output)
        return os.path.join(uploaded_root, output_relative)

    @staticmethod
    def infer_method_from_artifact_path(artifact_path: str) -> str:
        """Infer the processing method from the artifact path."""
        normalized = artifact_path.replace("/", os.sep).replace("\\", os.sep)
        parts = {part.lower() for part in normalized.split(os.sep) if part}
        return "vector" if PDF_REPLACEMENT_SUBDIR.lower() in parts else "ocr"

    @staticmethod
    def method_to_result_method(method: str) -> str:
        from modules.process_log import method_to_ocr_flag

        return method_to_ocr_flag(method)

    @staticmethod
    def result_method_to_processing_method(result_method: str | None) -> str:
        return "ocr" if result_method == "O" else "vector"

    @classmethod
    def is_review_output_path(cls, output_root: str | Path, artifact_path: str | None) -> bool:
        return JobQueue.is_review_artifact_path(artifact_path, output_root)

    @staticmethod
    def _is_plm_delivery_job(job: Job) -> bool:
        """Return whether the job participates in PLM delivery."""
        return job.source == "api" and bool(job.docnumber and job.work_seq)

    def _get_processing_method_for_job(self, job: Job, artifact_path: str | None = None) -> str:
        if job.result_method in {"O", "N"}:
            return self.result_method_to_processing_method(job.result_method)
        if artifact_path:
            return self.infer_method_from_artifact_path(artifact_path)
        return "ocr"

    def start(self) -> None:
        if self._threads and any(t.is_alive() for t in self._threads):
            return
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        stale = self.queue.reset_stale_running()
        if stale:
            logger.info("Worker startup reset %s stale running jobs to pending", stale)

        # VLM 就绪检查只做一次（多线程共享同一个 vLLM 服务）
        if self.ensure_vlm and not self._vlm_ready:
            ok, msg = self._ensure_vlm_ready()
            if not ok:
                logger.error("VLM startup failed; Worker not started: %s", msg)
                return
            self._vlm_ready = True

        self._stop.clear()
        self._threads = []
        for i in range(self.worker_count):
            t = threading.Thread(
                target=self._run_loop,
                args=(i,),
                name=f"job-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        if self.worker_count == 1:
            logger.info("Worker started (single-thread); output_root=%s", self.output_root)
        else:
            logger.info("Worker started (%s threads); output_root=%s", self.worker_count, self.output_root)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=timeout)
        self._threads = []
        logger.info("Worker stopped")

    def is_running(self) -> bool:
        return bool(self._threads and any(t.is_alive() for t in self._threads))

    def _sync_config(self) -> None:
        import config as cfg

        self.poll_interval = cfg.WORKER_POLL_INTERVAL
        self.max_retry = cfg.WORKER_MAX_RETRY
        self.watch_output_dir = str(cfg.WATCH_OUTPUT_DIR)
        self.watch_failed_dir = str(cfg.WATCH_FAILED_DIR)
        self.output_root = str(cfg.WORKER_OUTPUT_DIR)

    def _run_loop(self, worker_id: int = 0) -> None:
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
                    "Unhandled Worker exception while processing job %s\n%s",
                    job.id,
                    traceback.format_exc(),
                )
                self._safe_mark_failed(job.id, "Unhandled Worker exception")

    def _write_plm_oracle_updates(self, job: Job, method: str) -> None:
        """Write Oracle callbacks after PLM delivery completes."""
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
        """Complete Oracle callbacks for a delivered PLM artifact."""
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
            logger.warning("[job=%s] Oracle callback failed: %s", job.id, err)
            return False, err

        self.queue.mark_delivery_complete(
            job.id,
            remote_path=remote_path,
            uploaded_path=uploaded_path,
        )
        logger.info("[job=%s] PLM delivery completed", job.id)
        return True, ""

    def _deliver_plm_output(
        self,
        job: Job,
        artifact_path: str,
        method: str,
        remote_path: str | None = None,
        uploaded_path: str | None = None,
    ) -> tuple[bool, str]:
        """Perform PLM delivery: mirror locally, upload, archive, callback."""
        remote_path = remote_path or self.build_plm_remote_output_path(job, artifact_path)
        uploaded_path = uploaded_path or self.build_uploaded_archive_path(self.output_root, artifact_path)
        self.queue.mark_delivery_pending(job.id, remote_path, uploaded_path)

        if not artifact_path or not os.path.isfile(artifact_path):
            err = f"artifact file not found: {artifact_path}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM delivery failed: %s", job.id, err)
            return False, err

        local_plm_path = remote_path
        try:
            Path(os.path.dirname(local_plm_path)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, local_plm_path)
            logger.info("[job=%s] Mirrored artifact to local PLM path: %s", job.id, local_plm_path)
        except OSError as exc:
            err = f"failed to mirror artifact to local PLM path: {exc}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM delivery failed: %s", job.id, err)
            return False, err

        from modules.winscp_client import WinSCPClient

        ok, err = WinSCPClient().upload(local_plm_path, remote_path)
        if not ok:
            err = err or "WinSCP upload failed"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM upload failed: %s", job.id, err)
            return False, err
        logger.info("[job=%s] Uploaded artifact to PLM: %s", job.id, remote_path)

        try:
            Path(os.path.dirname(uploaded_path)).mkdir(parents=True, exist_ok=True)
            shutil.move(artifact_path, uploaded_path)
            logger.info("[job=%s] Archived artifact to uploaded: %s", job.id, uploaded_path)
        except OSError as exc:
            err = f"failed to move artifact to uploaded archive: {exc}"
            self.queue.mark_delivery_failed(
                job.id,
                err,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )
            logger.warning("[job=%s] PLM archive failed: %s", job.id, err)
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
        """Return whether a PLM delivery can be retried."""
        if not self._is_plm_delivery_job(job):
            return False, "current job is not a retryable PLM job"
        if not review_status_allows_plm_delivery(job.result_method, job.review_status):
            return False, "review approval is required before retrying PLM delivery"

        _, uploaded_path, reference_path = self._resolve_plm_retry_paths(job)
        if not reference_path:
            return False, "missing result path; cannot recover PLM delivery"
        if uploaded_path and os.path.isfile(uploaded_path):
            return True, ""
        if job.result_path and os.path.isfile(job.result_path):
            return True, ""
        return False, "artifact file was not found in OUTPUT or uploaded"

    def recover_single_plm_delivery(self, job_id: int) -> tuple[bool, str]:
        """Recover a single PLM delivery by job ID."""
        self._sync_config()
        job = self.queue.get(job_id)
        if job is None:
            return False, f"job not found: {job_id}"
        return self._recover_one_plm_delivery(job)

    def deliver_reviewed_plm_job(self, job_id: int) -> tuple[bool, str]:
        """Deliver an approved O-type PLM job."""
        self._sync_config()
        job = self.queue.get(job_id)
        if job is None:
            return False, f"job not found: {job_id}"
        if not self._is_plm_delivery_job(job):
            return False, "current job is not a PLM delivery job"
        if job.review_status != "approved":
            return False, "review approval required before PLM delivery"
        if job.result_method != "O":
            return False, "only O-type reviewed jobs can be delivered manually"
        if not self.is_review_output_path(self.output_root, job.result_path):
            return False, "review artifact is not under the VLMOCR review directory"
        if not job.result_path or not os.path.isfile(job.result_path):
            return False, "review artifact file not found"

        return self._deliver_plm_output(job, job.result_path, "ocr")

    def recover_pending_plm_deliveries(self, limit: int = 100) -> None:
        """Recover incomplete PLM deliveries on startup."""
        jobs = self.queue.list_recoverable_plm_deliveries(limit=limit)
        if not jobs:
            return

        logger.info("Recovering %s incomplete PLM deliveries", len(jobs))
        for job in jobs:
            self._sync_config()
            try:
                self._recover_one_plm_delivery(job)
            except Exception:
                logger.warning(
                    "[job=%s] Unexpected error while recovering PLM delivery\n%s",
                    job.id,
                    traceback.format_exc(),
                )

    def _recover_one_plm_delivery(self, job: Job) -> tuple[bool, str]:
        """Recover PLM delivery using the current file state."""
        if not self._is_plm_delivery_job(job):
            return False, "current job is not a retryable PLM job"
        if not review_status_allows_plm_delivery(job.result_method, job.review_status):
            return False, "review approval required before recovering PLM delivery"

        remote_path, uploaded_path, reference_path = self._resolve_plm_retry_paths(job)
        if not reference_path:
            err = "missing result path; cannot recover PLM delivery"
            logger.warning("[job=%s] %s", job.id, err)
            self.queue.mark_delivery_failed(job.id, err)
            return False, err

        if uploaded_path and os.path.isfile(uploaded_path):
            method = self._get_processing_method_for_job(job, uploaded_path)
            self.queue.mark_delivery_uploaded(
                job.id,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
                error_msg=job.plm_delivery_error,
            )
            logger.info("[job=%s] Found uploaded artifact; resuming Oracle callback only", job.id)
            return self._complete_plm_delivery(job, method, remote_path, uploaded_path)

        if job.result_path and os.path.isfile(job.result_path):
            method = self._get_processing_method_for_job(job, job.result_path)
            logger.info("[job=%s] Found OUTPUT artifact; resuming PLM upload flow", job.id)
            return self._deliver_plm_output(
                job,
                job.result_path,
                method,
                remote_path=remote_path,
                uploaded_path=uploaded_path,
            )

        err = "artifact file was not found in OUTPUT or uploaded"
        logger.warning("[job=%s] Unable to recover PLM delivery: %s", job.id, err)
        self.queue.mark_delivery_failed(
            job.id,
            err,
            remote_path=remote_path,
            uploaded_path=uploaded_path,
        )
        return False, err

    def _process_one(self, job: Job) -> None:
        logger.info("[job=%s] Start processing %s (source=%s)", job.id, job.file_path, job.source)
        if not os.path.isfile(job.file_path):
            self._handle_failure(job, f"file not found: {job.file_path}")
            return

        if self.ensure_vlm and not self._vlm_ready:
            # VLM 应在 start() 阶段已就绪；未就绪时跳过本次（下轮重试）
            logger.warning("[job=%s] VLM not ready; requeuing", job.id)
            self.queue.mark_failed(job.id, "VLM not ready")
            return

        base_output = os.path.join(self.output_root, OUTPUT_SUBDIR)
        vlmocr_dir = os.path.join(base_output, VLMOCR_SUBDIR)
        pdf_dir = os.path.join(base_output, PDF_REPLACEMENT_SUBDIR)
        os.makedirs(vlmocr_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        # 处理日志落盘到 <output_root>/OUTPUT/job_<时间戳>.log（与 API 路径一致）
        log_handler = None
        try:
            log_path = os.path.join(
                base_output, f"job_{time.strftime('%Y%m%d_%H%M%S')}.log")
            log_handler = logging.FileHandler(log_path, encoding="utf-8")
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
            logging.getLogger().addHandler(log_handler)
        except Exception as exc:
            logger.warning("[job=%s] Failed to create job log file: %s", job.id, exc)
            log_handler = None

        try:
            self._process_one_inner(job, base_output, vlmocr_dir, pdf_dir)
        finally:
            if log_handler is not None:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()

    def _process_one_inner(self, job: Job, base_output: str,
                           vlmocr_dir: str, pdf_dir: str) -> None:
        start = time.time()
        try:
            from modules.batch_processor import process_single_file

            result = process_single_file(
                job.file_path, self.output_root, drawing_no=job.drawing_no)
        except Exception as exc:
            logger.error("[job=%s] Processing failed: %s\n%s", job.id, exc, traceback.format_exc())
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
        result_method = self.method_to_result_method(method)
        final_path = out_path
        if out_path and os.path.exists(out_path):
            target_dir = pdf_dir if method == "vector" else vlmocr_dir
            target_path = os.path.join(target_dir, os.path.basename(out_path))
            try:
                shutil.move(out_path, target_path)
                final_path = target_path
            except OSError as exc:
                logger.warning("[job=%s] Failed to move artifact; keeping original path: %s", job.id, exc)

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
            logger.warning("[job=%s] Failed to write y_boxes.csv: %s", job.id, exc)

        elapsed = time.time() - start
        replacements = result.get("total", 0)
        logger.info(
            "[job=%s] Processing completed method=%s replacements=%s elapsed=%.1fs -> %s",
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
                logger.info("[job=%s] Copied artifact to watch_output: %s", job.id, watch_out_path)
                final_path = watch_out_path
            except OSError as exc:
                logger.warning("[job=%s] Failed to copy artifact to watch_output: %s", job.id, exc)

        self.queue.mark_done(job.id, final_path, result_method=result_method)

        try:
            self._process_log.record(
                drawing_no=job.drawing_no,
                revision=job.revision,
                ocr_flag=result_method,
                status="success",
                filename=job.source_file,
                docnumber=job.docnumber,
                work_seq=job.work_seq,
            )
        except Exception as exc:
            logger.warning("[job=%s] Failed to record process log: %s", job.id, exc)

        if self._is_plm_delivery_job(job):
            if result_method == "O":
                self.queue.mark_review_pending(job.id)
                logger.info("[job=%s] O-type PLM artifact moved to pending review: %s", job.id, final_path)
            else:
                self.queue.mark_review_not_required(job.id)
                self._deliver_plm_output(job, final_path, method)

        if job.file_path and os.path.isfile(job.file_path):
            try:
                os.remove(job.file_path)
            except OSError as exc:
                logger.warning("[job=%s] Failed to remove processing source file: %s", job.id, exc)

    def _handle_failure(self, job: Job, err: str) -> None:
        self.queue.mark_failed(job.id, err)
        if self.max_retry > 0 and self.queue.requeue_for_retry(job.id, self.max_retry):
            logger.warning(
                "[job=%s] Processing failed and was requeued (max_retry=%s): %s",
                job.id,
                self.max_retry,
                err,
            )
            return

        logger.error("[job=%s] Processing failed without further retry: %s", job.id, err)
        try:
            self._process_log.record(
                drawing_no=job.drawing_no,
                revision=job.revision,
                ocr_flag="N",
                status="failed",
                filename=job.source_file,
                docnumber=job.docnumber,
                work_seq=job.work_seq,
            )
        except Exception as exc:
            logger.warning("[job=%s] Failed to record failure log: %s", job.id, exc)

        if job.source == "watch_folder" and os.path.isfile(job.file_path):
            try:
                Path(self.watch_failed_dir).mkdir(parents=True, exist_ok=True)
                dst = os.path.join(self.watch_failed_dir, os.path.basename(job.file_path))
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(os.path.basename(job.file_path))
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    dst = os.path.join(self.watch_failed_dir, f"{stem}_{ts}{ext}")
                shutil.move(job.file_path, dst)
                logger.info("[job=%s] Moved source file to failed: %s", job.id, dst)
            except OSError as exc:
                logger.warning("[job=%s] Failed to move source file to failed: %s", job.id, exc)

    def _safe_mark_failed(self, job_id: int, err: str) -> None:
        try:
            self.queue.mark_failed(job_id, err)
        except Exception:
            logger.error("Failed to mark job=%s as failed\n%s", job_id, traceback.format_exc())

    def _ensure_vlm_ready(self) -> tuple[bool, str]:
        try:
            from modules.docker_manager import ensure_vlm_ready

            return ensure_vlm_ready()
        except Exception as exc:
            return False, f"ensure_vlm_ready exception: {exc}"


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from config import PADDLEOCR_API_URL, VLM_PROVIDER

    logger.info(
        "VLM mode: %s%s",
        VLM_PROVIDER,
        f" -> {PADDLEOCR_API_URL}" if VLM_PROVIDER == "paddleocr_api" else " -> local Docker",
    )
    queue = JobQueue()
    worker = Worker(queue)
    worker.start()
    logger.info("Worker main loop running; press Ctrl+C to exit")
    try:
        while worker.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Received Ctrl+C; stopping Worker")
        worker.stop()


if __name__ == "__main__":
    _main()
