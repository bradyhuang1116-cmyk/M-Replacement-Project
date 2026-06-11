"""系统生命周期管理 — 停止全部服务 / 重启 / 完全关机"""

import os
import sys
import re
import subprocess
import threading
import time
import logging

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _kill_docker_desktop():
    subprocess.run(
        ["taskkill", "/F", "/IM", "Docker Desktop.exe"],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )


def _shutdown_wsl():
    subprocess.run(
        ["wsl", "--shutdown"],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )


def _kill_frontend():
    """Find the process listening on port 3000 and kill it by PID."""
    try:
        r = subprocess.run(
            ["netstat", "-aon"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        for line in r.stdout.splitlines():
            if ":3000" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and pid != "0":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", pid],
                        capture_output=True, creationflags=CREATE_NO_WINDOW,
                    )
                    logger.info(f"Killed frontend process PID {pid}")
                    return
    except Exception as e:
        logger.warning(f"Failed to kill frontend: {e}")


@router.post("/system/stop-all")
async def stop_all():
    """Stop running job + VLM container only. Does NOT kill Docker/WSL."""
    from api.routes.jobs import _job, _job_lock

    with _job_lock:
        if _job and _job.get("is_running"):
            _job["cancel"].set()
            _job["phase"] = "stopping"

    from modules.docker_manager import stop_vllm_container
    try:
        stop_vllm_container()
    except Exception as e:
        logger.warning(f"Container stop failed: {e}")

    return {"status": "stopping_all"}


class ShutdownRequest(BaseModel):
    kill_docker: bool = True
    kill_wsl: bool = True


@router.post("/system/shutdown")
async def shutdown(req: ShutdownRequest):
    """Full system shutdown with optional Docker/WSL cleanup."""
    from api.routes.jobs import _job, _job_lock

    with _job_lock:
        if _job and _job.get("is_running"):
            _job["cancel"].set()
            _job["phase"] = "stopping"

    def _do_shutdown():
        time.sleep(1.0)

        from modules.docker_manager import stop_vllm_container
        try:
            stop_vllm_container()
        except Exception:
            pass

        if req.kill_docker:
            _kill_docker_desktop()
        if req.kill_wsl:
            _shutdown_wsl()

        _kill_frontend()
        logger.info("System shutdown complete, exiting backend")
        os._exit(0)

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return {"status": "shutting_down"}


@router.post("/system/restart")
async def restart():
    """停止当前任务，原地重启后端进程。"""
    from api.routes.jobs import _job, _job_lock

    with _job_lock:
        if _job and _job.get("is_running"):
            _job["cancel"].set()
            _job["phase"] = "stopping"

    def _do_restart():
        time.sleep(1.5)
        logger.info("Backend restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do_restart, daemon=True).start()
    return {"status": "restarting"}
