"""Job management routes: manual batch processing, queue inspection, and review flow."""
from __future__ import annotations

import asyncio
import collections
import ctypes
import gc
import ipaddress
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import config as config_module
from config import (
    OUTPUT_SUBDIR,
    PDF_REPLACEMENT_SUBDIR,
    SUPPORTED_EXTENSIONS,
    VLMOCR_SUBDIR,
    WORKER_OUTPUT_DIR,
    Y_BOXES_CSV_NAME,
)
from modules.job_queue import JobQueue

router = APIRouter()
logger = logging.getLogger(__name__)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_UNICODE_ENVIRONMENT = 0x00000400
NORMAL_PRIORITY_CLASS = 0x00000020

_job_lock = threading.Lock()
_job: dict | None = None
_manual_editor_launch_lock = threading.Lock()
_manual_editor_last_launch_at = 0.0


class StartRequest(BaseModel):
    input_dir: str
    output_dir: str
    prefixes: list[str] = Field(default_factory=lambda: list(config_module.DEFAULT_PREFIXES))
    selected_files: list[str] | None = None


def _serialize_queue_job(job) -> dict:
    return {
        "id": job.id,
        "source": job.source,
        "source_file": job.source_file,
        "file_path": job.file_path,
        "drawing_no": job.drawing_no,
        "revision": job.revision,
        "docnumber": job.docnumber,
        "work_seq": job.work_seq,
        "status": job.status,
        "retry_count": job.retry_count,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error_msg": job.error_msg,
        "result_path": job.result_path,
        "result_method": job.result_method,
        "review_status": job.review_status,
        "reviewed_at": job.reviewed_at,
        "plm_delivery_status": job.plm_delivery_status,
        "plm_remote_path": job.plm_remote_path,
        "plm_uploaded_path": job.plm_uploaded_path,
        "plm_delivery_error": job.plm_delivery_error,
        "plm_delivery_finished_at": job.plm_delivery_finished_at,
    }


def _scan_input(input_dir: str) -> list[str]:
    files: list[str] = []
    for name in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(input_dir, name))
    return files


def _review_output_root() -> str:
    return JobQueue.get_review_output_dir(WORKER_OUTPUT_DIR)


def _validate_reviewable_job(job, *, require_pending: bool) -> None:
    if job.source != "api":
        raise HTTPException(status_code=400, detail="only api jobs can enter review flow")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job is not ready for review")
    if job.result_method != "O":
        raise HTTPException(status_code=409, detail="only O-type jobs can be reviewed")
    if require_pending and job.review_status != "pending":
        raise HTTPException(status_code=409, detail="job is not pending review")
    if not JobQueue.is_review_artifact_path(job.result_path, WORKER_OUTPUT_DIR):
        raise HTTPException(status_code=409, detail="result file is not under the VLMOCR review directory")
    if not job.result_path or not os.path.isfile(job.result_path):
        raise HTTPException(status_code=404, detail="review artifact file not found")


def _render_preview_png(image_path: str, max_dim: int = 2200) -> bytes:
    from PIL import Image

    with Image.open(image_path) as img:
        preview = img.convert("RGB")
        preview.thumbnail((max_dim, max_dim))
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        return buffer.getvalue()


def _manual_editor_csv_path() -> str:
    return os.path.abspath(
        os.path.join(
            config_module.WORKER_OUTPUT_DIR,
            config_module.OUTPUT_SUBDIR,
            config_module.Y_BOXES_CSV_NAME,
        )
    )


def _resolve_manual_editor_original_file(job) -> str:
    if not job.docnumber or not job.work_seq:
        raise HTTPException(status_code=409, detail="PLM metadata is missing; cannot resolve original backup file")

    from modules.oracle_helper import OracleHelper

    location = OracleHelper().fetch_file_location(
        job.drawing_no or "",
        job.revision or "",
        job.source_file,
        job.docnumber,
        job.work_seq,
    )
    if not location:
        raise HTTPException(status_code=404, detail="original backup file location was not found in SIPM197")

    original_file = os.path.abspath(
        os.path.join(
            config_module.ORACLE_PATH_PREFIX,
            location.strip().lstrip("\\/"),
        )
    )
    if not os.path.isfile(original_file):
        raise HTTPException(status_code=404, detail=f"original backup file not found: {original_file}")
    return original_file


def _quote_powershell(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _format_win_error(code: int) -> str:
    try:
        return ctypes.FormatError(code).strip()
    except Exception:
        return f"Windows error {code}"


def _build_manual_editor_args(
    *,
    original_dir: str,
    replaced_dir: str,
    filename: str,
    csv_dir: str,
) -> list[str]:
    return ["-i", original_dir, "-o", replaced_dir, "-f", filename, "--csv", csv_dir]


def _current_process_session_id() -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32
    session_id = ctypes.c_uint()
    if not kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session_id)):
        return None
    return int(session_id.value)


def _run_process_in_active_session(executable: str, args: list[str]) -> None:
    if os.name != "nt":
        raise OSError("active-session launch is only supported on Windows")

    wintypes = ctypes.wintypes
    kernel32 = ctypes.windll.kernel32
    advapi32 = ctypes.windll.advapi32
    userenv = ctypes.windll.userenv
    wtsapi32 = ctypes.windll.wtsapi32

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    active_session_id = kernel32.WTSGetActiveConsoleSessionId()
    if active_session_id == 0xFFFFFFFF:
        raise OSError("no active desktop session was found")

    user_token = wintypes.HANDLE()
    env_block = ctypes.c_void_p()
    proc_info = PROCESS_INFORMATION()

    command_line = subprocess.list2cmdline([executable, *args])
    working_dir = os.path.dirname(executable) or None

    startup = STARTUPINFOW()
    startup.cb = ctypes.sizeof(STARTUPINFOW)
    startup.lpDesktop = "winsta0\\default"

    try:
        if not wtsapi32.WTSQueryUserToken(active_session_id, ctypes.byref(user_token)):
            code = ctypes.get_last_error()
            raise OSError(f"WTSQueryUserToken failed: {_format_win_error(code)}")

        if not userenv.CreateEnvironmentBlock(ctypes.byref(env_block), user_token, False):
            code = ctypes.get_last_error()
            raise OSError(f"CreateEnvironmentBlock failed: {_format_win_error(code)}")

        created = advapi32.CreateProcessAsUserW(
            user_token,
            None,
            command_line,
            None,
            None,
            False,
            CREATE_UNICODE_ENVIRONMENT | NORMAL_PRIORITY_CLASS,
            env_block,
            working_dir,
            ctypes.byref(startup),
            ctypes.byref(proc_info),
        )
        if not created:
            code = ctypes.get_last_error()
            raise OSError(f"CreateProcessAsUserW failed: {_format_win_error(code)}")
    finally:
        if proc_info.hThread:
            kernel32.CloseHandle(proc_info.hThread)
        if proc_info.hProcess:
            kernel32.CloseHandle(proc_info.hProcess)
        if env_block:
            userenv.DestroyEnvironmentBlock(env_block)
        if user_token:
            kernel32.CloseHandle(user_token)


def _run_powershell_in_active_session(command: str) -> bool:
    try:
        _run_process_in_active_session(
            "powershell.exe",
            ["-NoProfile", "-NonInteractive", "-Command", command],
        )
        return True
    except OSError:
        logger.exception("Failed to run PowerShell in active desktop session")
        return False


def _normalize_host(value: str) -> str:
    host = (value or "").strip().lower()
    if host.startswith("::ffff:"):
        host = host[7:]
    if "%" in host:
        host = host.split("%", 1)[0]
    return host


@lru_cache(maxsize=1)
def _local_host_candidates() -> tuple[set[str], set[str]]:
    hostnames: set[str] = {"localhost"}
    addresses: set[str] = {"127.0.0.1", "::1"}

    for name in filter(None, {socket.gethostname(), socket.getfqdn(), "localhost"}):
        hostnames.add(name.lower())
        try:
            for info in socket.getaddrinfo(name, None):
                addr = _normalize_host(info[4][0])
                if addr:
                    addresses.add(addr)
        except OSError:
            continue

    return hostnames, addresses


def _is_local_request_host(host: str | None) -> bool:
    normalized = _normalize_host(host or "")
    if not normalized:
        return False

    hostnames, addresses = _local_host_candidates()
    if normalized in hostnames or normalized in addresses:
        return True

    try:
        addr = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return addr.is_loopback or normalized in addresses


def _is_manual_editor_running(exe_path: str) -> bool:
    image_name = os.path.basename(exe_path).strip()
    if not image_name:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError:
        logger.exception("Failed to inspect ManualEditor.exe process list")
        return False

    if result.returncode != 0:
        logger.warning("tasklist returned %s while checking ManualEditor.exe", result.returncode)
        return False

    output = (result.stdout or "").strip().lower()
    if not output or "no tasks are running" in output:
        return False
    return image_name.lower() in output


def _focus_manual_editor_window(exe_path: str) -> bool:
    process_name = os.path.splitext(os.path.basename(exe_path).strip())[0]
    if not process_name:
        return False

    command = (
        "$sig = '[DllImport(\"user32.dll\")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow); "
        "[DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr hWnd);'; "
        "Add-Type -MemberDefinition $sig -Name Win32Show -Namespace ManualEditorFocus -ErrorAction SilentlyContinue | Out-Null; "
        f"$proc = Get-Process -Name {_quote_powershell(process_name)} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.MainWindowHandle -ne 0 } | Sort-Object StartTime | Select-Object -First 1; "
        "if ($null -eq $proc) { exit 1 }; "
        "[ManualEditorFocus.Win32Show]::ShowWindowAsync($proc.MainWindowHandle, 9) | Out-Null; "
        "[ManualEditorFocus.Win32Show]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null; "
        "exit 0"
    )

    if _current_process_session_id() == 0:
        return _run_powershell_in_active_session(command)

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError:
        logger.exception("Failed to focus ManualEditor.exe window")
        return False

    return result.returncode == 0


def _launch_manual_editor_elevated(
    *,
    exe_path: str,
    original_dir: str,
    replaced_dir: str,
    filename: str,
    csv_dir: str,
) -> None:
    arg_values = _build_manual_editor_args(
        original_dir=original_dir,
        replaced_dir=replaced_dir,
        filename=filename,
        csv_dir=csv_dir,
    )
    if _current_process_session_id() == 0:
        _run_process_in_active_session(exe_path, arg_values)
        return

    args_literal = ", ".join(_quote_powershell(value) for value in arg_values)
    command = (
        f"$argList = @({args_literal}); "
        f"Start-Process -FilePath {_quote_powershell(exe_path)} "
        "-ArgumentList $argList -Verb RunAs"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_job(job: dict) -> None:
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

    log_path = os.path.join(base_output, f"job_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)

    log_buffer = collections.deque(maxlen=200)
    job["log_buffer"] = log_buffer

    class _BufferHandler(logging.Handler):
        def emit(self, record):
            log_buffer.append(self.format(record))

    buf_handler = _BufferHandler()
    buf_handler.setLevel(logging.INFO)
    buf_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(buf_handler)

    try:
        job["phase"] = "starting_vlm"
        try:
            from modules.docker_manager import ensure_vlm_ready

            ok, msg = ensure_vlm_ready()
            if not ok:
                logger.error("VLM startup failed: %s", msg)
                return
            logger.info("VLM ready: %s", msg)
        except Exception as exc:
            logger.error("VLM startup exception: %s", exc)
            return

        if cancel.is_set():
            return

        job["phase"] = "warming_up"
        try:
            from modules.region_detector import _get_ocr_v5

            _get_ocr_v5("en")
            logger.info("OCR warmup finished")
        except Exception as exc:
            logger.warning("Warmup warning (processing continues): %s", exc)

        if cancel.is_set():
            return

        job["phase"] = "processing"
        from modules.batch_processor import clear_gpu_cache, process_single_file
        from modules.factory_note_pixel import clear_y_box_records
        from modules.text_replacer import clear_ocr_cache

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
                result = process_single_file(file_path, output_dir, prefixes=prefixes)
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

                job["files"][i].update(
                    {
                        "status": "completed",
                        "duration": f"{m}m {s:02d}s",
                        "method": method_label,
                        "replacedFilename": os.path.basename(out_path) if out_path else "",
                    }
                )
                logger.info(
                    "OK [%s/%s] %s: %s replacements (%s)",
                    i + 1,
                    len(file_paths),
                    fname,
                    result.get("total", 0),
                    method_label,
                )

                try:
                    from modules.filename_parser import parse as parse_filename
                    from modules.process_log import ProcessLog, method_to_ocr_flag

                    parsed = parse_filename(fname)
                    ProcessLog().record(
                        drawing_no=parsed.drawing_no,
                        revision=parsed.revision,
                        ocr_flag=method_to_ocr_flag(method),
                        status="success",
                        filename=fname,
                    )
                except Exception as exc:
                    logger.warning("Process log warning: %s", exc)

            except Exception as exc:
                logger.error("FAIL [%s/%s] %s: %s", i + 1, len(file_paths), fname, exc)
                job["files"][i]["status"] = "failed"
                job["files"][i]["error"] = str(exc)

            gc.collect()
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
                logger.info("y_boxes.csv written with %s rows", n)
        except Exception as exc:
            logger.warning("y_boxes.csv warning: %s", exc)

        root_logger.removeHandler(fh)
        root_logger.removeHandler(buf_handler)
        fh.close()
        logger.info("Job finished")


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


@router.get("/jobs/queue")
async def list_queue_jobs(
    source: str | None = Query(default=None, description="api or watch_folder"),
    status: str | None = Query(default=None, description="pending/running/done/failed"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    queue = JobQueue()
    items = queue.list_jobs(source=source, status=status, limit=limit, offset=offset)
    total = queue.count_jobs(source=source, status=status)

    counts = {
        key: queue.count_jobs(source=source, status=key)
        for key in ("pending", "running", "done", "failed")
    }

    return {
        "items": [_serialize_queue_job(job) for job in items],
        "total": total,
        "counts": counts,
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/review")
async def list_review_jobs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    queue = JobQueue()
    items = queue.list_review_jobs(limit=limit, offset=offset, output_root=WORKER_OUTPUT_DIR)
    total = queue.count_review_jobs(output_root=WORKER_OUTPUT_DIR)
    return {
        "items": [_serialize_queue_job(job) for job in items],
        "total": total,
        "limit": limit,
        "offset": offset,
        "review_root": _review_output_root(),
    }


@router.post("/jobs/review/{job_id}/approve")
async def approve_review_job(job_id: int, background_tasks: BackgroundTasks):
    from modules.worker import Worker

    queue = JobQueue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")

    _validate_reviewable_job(job, require_pending=True)

    queue.mark_review_approved(job_id)
    worker = Worker(queue, ensure_vlm=False)
    background_tasks.add_task(worker.deliver_reviewed_plm_job, job_id)
    return {"status": "started"}


@router.get("/jobs/review/{job_id}/preview")
async def preview_review_job(job_id: int):
    queue = JobQueue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")

    _validate_reviewable_job(job, require_pending=True)
    try:
        preview_bytes = _render_preview_png(job.result_path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to render preview: {exc}") from exc
    return Response(content=preview_bytes, media_type="image/png")


@router.post("/jobs/review/{job_id}/manual-process")
async def launch_manual_process(job_id: int, request: Request):
    queue = JobQueue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")

    _validate_reviewable_job(job, require_pending=True)

    if os.name != "nt":
        raise HTTPException(status_code=501, detail="manual process launch is only supported on Windows")
    client_host = request.client.host if request.client else None
    if not _is_local_request_host(client_host):
        raise HTTPException(
            status_code=403,
            detail="Remote editing is unavailable. Please use this feature on the local machine.",
        )

    exe_path_raw = str(getattr(config_module, "MANUAL_EDITOR_EXE_PATH", "") or "").strip()
    if not exe_path_raw:
        raise HTTPException(status_code=400, detail="MANUAL_EDITOR_EXE_PATH is not configured")
    exe_path = os.path.abspath(exe_path_raw)
    if not os.path.isfile(exe_path):
        raise HTTPException(status_code=404, detail=f"ManualEditor.exe not found: {exe_path}")

    original_file = _resolve_manual_editor_original_file(job)
    original_dir = os.path.dirname(original_file)
    if not os.path.isdir(original_dir):
        raise HTTPException(status_code=404, detail=f"manual editor original directory not found: {original_dir}")

    result_path = os.path.abspath(job.result_path)
    replaced_dir = os.path.dirname(result_path)
    if not os.path.isdir(replaced_dir):
        raise HTTPException(status_code=404, detail=f"review artifact directory not found: {replaced_dir}")
    filename = os.path.basename(result_path)

    csv_path = _manual_editor_csv_path()
    csv_dir = os.path.dirname(csv_path)
    if not os.path.isdir(csv_dir):
        raise HTTPException(status_code=404, detail=f"y_boxes.csv directory not found: {csv_dir}")
    if not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail=f"y_boxes.csv not found: {csv_path}")

    try:
        with _manual_editor_launch_lock:
            global _manual_editor_last_launch_at
            now = time.monotonic()
            if _is_manual_editor_running(exe_path):
                if _focus_manual_editor_window(exe_path):
                    logger.info("ManualEditor was already running; brought window to front for job %s", job_id)
                    return {"status": "focused"}
                raise HTTPException(status_code=409, detail="ManualEditor is already open.")
            if now - _manual_editor_last_launch_at < 8:
                raise HTTPException(status_code=409, detail="ManualEditor is starting. Please try again in a moment.")
            _launch_manual_editor_elevated(
                exe_path=exe_path,
                original_dir=original_dir,
                replaced_dir=replaced_dir,
                filename=filename,
                csv_dir=csv_dir,
            )
            _manual_editor_last_launch_at = now
    except OSError as exc:
        logger.exception("Failed to launch manual editor for review job %s", job_id)
        raise HTTPException(status_code=500, detail=f"failed to launch ManualEditor.exe: {exc}") from exc

    logger.info(
        "ManualEditor launch requested for review job %s: exe=%s original=%s file=%s",
        job_id,
        exe_path,
        original_file,
        filename,
    )
    return {"status": "started"}


@router.post("/jobs/queue/{job_id}/retry")
async def retry_queue_job(job_id: int, background_tasks: BackgroundTasks):
    from modules.worker import Worker

    queue = JobQueue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    if job.source != "api":
        raise HTTPException(status_code=400, detail="only source=api jobs support manual retry")

    if job.status == "failed":
        try:
            queue.retry_failed_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "queued", "retry_type": "processing"}

    if job.status == "done" and job.plm_delivery_status == "failed":
        worker = Worker(queue, ensure_vlm=False)
        ok, msg = worker.can_retry_plm_delivery(job)
        if not ok:
            raise HTTPException(status_code=409, detail=msg)
        try:
            queue.retry_failed_plm_delivery(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(worker.recover_single_plm_delivery, job_id)
        return {"status": "started", "retry_type": "plm_delivery"}

    raise HTTPException(status_code=409, detail="current job state does not allow manual retry")
