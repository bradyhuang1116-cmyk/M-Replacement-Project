"""任务管理接口 — 启动/停止/SSE状态推送"""

import gc
import os
import json
import time
import shutil
import threading
import logging
import asyncio
import collections
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    SUPPORTED_EXTENSIONS,
    DEFAULT_PREFIXES,
    OUTPUT_SUBDIR,
    VLMOCR_SUBDIR,
    PDF_REPLACEMENT_SUBDIR,
    Y_BOXES_CSV_NAME,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ── 全局任务状态 ──

_job_lock = threading.Lock()
_job: dict | None = None  # { id, thread, cancel, phase, files, ... }


class StartRequest(BaseModel):
    input_dir: str
    output_dir: str
    prefixes: list[str] = DEFAULT_PREFIXES
    selected_files: list[str] | None = None


def _scan_input(input_dir: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(input_dir, name))
    return files


def _run_job(job: dict):
    input_dir = job["input_dir"]
    output_dir = job["output_dir"]
    prefixes = job["prefixes"]
    cancel: threading.Event = job["cancel"]
    file_paths: list[str] = job["file_paths"]

    base_output = os.path.join(output_dir, OUTPUT_SUBDIR)
    vlmocr_dir = os.path.join(base_output, VLMOCR_SUBDIR)
    pdf_dir = os.path.join(base_output, PDF_REPLACEMENT_SUBDIR)
    os.makedirs(vlmocr_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    log_path = os.path.join(
        base_output,
        f"job_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)

    log_buffer = collections.deque(maxlen=200)
    job["log_buffer"] = log_buffer

    class _BufferHandler(logging.Handler):
        def emit(self, record):
            log_buffer.append(self.format(record))

    buf_handler = _BufferHandler()
    buf_handler.setLevel(logging.INFO)
    buf_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(buf_handler)

    try:
        # Phase 1: Start VLM service
        job["phase"] = "starting_vlm"
        try:
            from modules.docker_manager import ensure_vlm_ready
            ok, msg = ensure_vlm_ready()
            if not ok:
                logger.error(f"VLM 启动失败: {msg}")
                return
            logger.info(f"VLM 就绪: {msg}")
        except Exception as e:
            logger.error(f"VLM 启动异常: {e}")
            return

        if cancel.is_set():
            return

        # Phase 2: Warm up models
        job["phase"] = "warming_up"
        try:
            from modules.region_detector import _get_ocr_v5
            _get_ocr_v5("en")
            logger.info("OCR 引擎预热完成")
        except Exception as e:
            logger.warning(f"模型预热异常(继续处理): {e}")

        if cancel.is_set():
            return

        # Phase 3: Process files
        job["phase"] = "processing"
        from modules.batch_processor import process_single_file
        from modules.text_replacer import clear_ocr_cache
        from modules.factory_note_pixel import clear_y_box_records, flush_y_boxes_csv
        clear_ocr_cache()
        clear_y_box_records()

        for i, file_path in enumerate(file_paths):
            if cancel.is_set():
                break

            fname = os.path.basename(file_path)
            job["current_file"] = i + 1
            job["files"][i]["status"] = "processing"
            job["files"][i]["start_time"] = time.time()

            try:
                result = process_single_file(
                    file_path, output_dir, prefixes=prefixes,
                )
                elapsed = time.time() - job["files"][i]["start_time"]
                method = result.get("method", "ocr")
                out_path = result.get("output_path", "")

                if out_path and os.path.exists(out_path):
                    target_dir = pdf_dir if method == "vector" else vlmocr_dir
                    target_path = os.path.join(target_dir, os.path.basename(out_path))
                    shutil.move(out_path, target_path)
                    out_path = target_path

                method_label = "PDF Replacement" if method == "vector" else "VLMOCR"
                m = int(elapsed // 60)
                s = int(elapsed % 60)

                job["files"][i].update({
                    "status": "completed",
                    "duration": f"{m}m {s:02d}s",
                    "method": method_label,
                    "replacedFilename": os.path.basename(out_path) if out_path else "",
                })
                logger.info(f"  OK [{i+1}/{len(file_paths)}] {fname}: {result.get('total', 0)} replacements ({method_label})")

                # 写处理日志（O/N）：O=用OCR(ocr)，N=未用OCR(vector)
                try:
                    from modules.process_log import ProcessLog, method_to_ocr_flag
                    from modules.filename_parser import parse as _parse_fn
                    _pf = _parse_fn(fname)
                    ProcessLog().record(
                        drawing_no=_pf.drawing_no,
                        revision=_pf.revision,
                        ocr_flag=method_to_ocr_flag(method),
                        status="success",
                        filename=fname,
                    )
                except Exception as _e:
                    logger.warning(f"  写处理日志失败（不影响结果）: {_e}")

            except Exception as e:
                logger.error(f"  FAIL [{i+1}/{len(file_paths)}] {fname}: {e}")
                job["files"][i]["status"] = "failed"
                job["files"][i]["error"] = str(e)

            gc.collect()
            from modules.batch_processor import clear_gpu_cache
            clear_gpu_cache()

    finally:
        job["phase"] = "idle"
        job["is_running"] = False

        for subdir in ("vector", "ocr"):
            d = os.path.join(output_dir, subdir)
            try:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass

        try:
            from modules.factory_note_pixel import flush_y_boxes_csv
            n = flush_y_boxes_csv(os.path.join(base_output, Y_BOXES_CSV_NAME))
            if n:
                logger.info(f"y_boxes.csv 已写入 {n} 条记录")
        except Exception as e:
            logger.warning(f"y_boxes.csv 写入失败（不影响结果）: {e}")

        root_logger.removeHandler(fh)
        root_logger.removeHandler(buf_handler)
        fh.close()
        logger.info("任务完成")


@router.post("/jobs/start")
async def start_job(req: StartRequest):
    global _job

    with _job_lock:
        if _job and _job.get("is_running"):
            return {"error": "A job is already running"}

    input_dir = os.path.abspath(req.input_dir)
    output_dir = os.path.abspath(req.output_dir)

    if not os.path.isdir(input_dir):
        return {"error": f"Input directory not found: {input_dir}"}

    file_paths = _scan_input(input_dir)
    if req.selected_files:
        selected = set(req.selected_files)
        file_paths = [fp for fp in file_paths if os.path.basename(fp) in selected]
    if not file_paths:
        return {"error": "No supported files found in input directory"}

    os.makedirs(output_dir, exist_ok=True)

    files_state = [
        {
            "filename": os.path.basename(fp),
            "status": "pending",
            "duration": "",
            "method": "",
            "replacedFilename": "",
            "start_time": None,
        }
        for fp in file_paths
    ]

    job_id = str(uuid4())[:8]
    job = {
        "id": job_id,
        "is_running": True,
        "phase": "starting_vlm",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "prefixes": req.prefixes,
        "file_paths": file_paths,
        "files": files_state,
        "current_file": 0,
        "start_time": time.time(),
        "cancel": threading.Event(),
    }

    with _job_lock:
        _job = job

    t = threading.Thread(target=_run_job, args=(job,), daemon=True)
    job["thread"] = t
    t.start()

    return {
        "job_id": job_id,
        "total_files": len(file_paths),
        "files": [{"filename": f["filename"], "status": f["status"]} for f in files_state],
    }


@router.post("/jobs/stop")
async def stop_job():
    global _job
    with _job_lock:
        if _job and _job.get("is_running"):
            _job["cancel"].set()
            _job["phase"] = "stopping"
            return {"status": "stopping"}
    return {"status": "no_job_running"}


@router.get("/jobs/status")
async def job_status_sse():
    async def event_stream():
        while True:
            with _job_lock:
                job = _job

            if job:
                elapsed = int(time.time() - job["start_time"]) if job.get("start_time") else 0
                total = len(job["files"])
                completed = sum(1 for f in job["files"] if f["status"] == "completed")
                progress = int((completed / total) * 100) if total > 0 else 0

                # Calculate current file elapsed
                current_elapsed = 0
                for f in job["files"]:
                    if f["status"] == "processing" and f.get("start_time"):
                        current_elapsed = int(time.time() - f["start_time"])

                data = {
                    "phase": job.get("phase", "idle"),
                    "isRunning": job.get("is_running", False),
                    "currentFile": job.get("current_file", 0),
                    "totalFiles": total,
                    "elapsedSeconds": current_elapsed,
                    "totalElapsed": elapsed,
                    "progress": progress,
                    "logs": list(job.get("log_buffer", [])),
                    "files": [
                        {
                            "filename": f["filename"],
                            "status": f["status"],
                            "duration": f.get("duration", ""),
                            "method": f.get("method", ""),
                            "replacedFilename": f.get("replacedFilename", ""),
                            "error": f.get("error", ""),
                        }
                        for f in job["files"]
                    ],
                }
            else:
                data = {
                    "phase": "idle",
                    "isRunning": False,
                    "currentFile": 0,
                    "totalFiles": 0,
                    "elapsedSeconds": 0,
                    "totalElapsed": 0,
                    "progress": 0,
                    "logs": [],
                    "files": [],
                }

            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
