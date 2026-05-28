"""Docker 容器管理 — 启动/停止/检查 vLLM VLM 服务"""

import os
import time
import subprocess
import logging

import requests

from config import (
    DOCKER_CONTAINER_NAME,
    DOCKER_CONTAINER_PORT,
    DOCKER_IMAGE as _CFG_DOCKER_IMAGE,
    VLLM_MODEL_DIR,
    VLLM_CONFIG_PATH,
)

logger = logging.getLogger(__name__)

# 保留模块内导出名（旧代码可能 import）；值取自 config，可通过 .env 覆盖
CONTAINER_NAME = DOCKER_CONTAINER_NAME
CONTAINER_PORT = DOCKER_CONTAINER_PORT
DOCKER_IMAGE = _CFG_DOCKER_IMAGE

_MODEL_PATH = VLLM_MODEL_DIR
_CONFIG_PATH = VLLM_CONFIG_PATH


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except FileNotFoundError:
        return -1, "docker not found"


def is_docker_running() -> bool:
    code, _ = _run(["docker", "info"])
    return code == 0


def is_container_running() -> bool:
    code, out = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME])
    return code == 0 and out.strip().lower() == "true"


def _vlm_health_check() -> bool:
    try:
        resp = requests.get(
            f"http://localhost:{CONTAINER_PORT}/v1/models", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def start_docker_desktop() -> tuple[bool, str]:
    if is_docker_running():
        return True, "Docker 已在运行"

    docker_exe = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if not os.path.exists(docker_exe):
        return False, f"Docker Desktop 未安装: {docker_exe}"

    logger.info("启动 Docker Desktop ...")
    subprocess.Popen(
        [docker_exe],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    for i in range(60):
        time.sleep(2)
        if is_docker_running():
            logger.info(f"Docker daemon 就绪 (等待 {(i+1)*2}s)")
            return True, f"Docker 已启动 (等待 {(i+1)*2}s)"
    return False, "Docker 启动超时 (120s)"


def start_vllm_container() -> tuple[bool, str]:
    if _vlm_health_check():
        return True, "VLM 服务已在运行"

    if is_container_running():
        logger.info("容器在运行但 API 未就绪，等待模型加载...")
    else:
        _run(["docker", "rm", "-f", CONTAINER_NAME])

        model_mount = _MODEL_PATH.replace("\\", "/")
        config_mount = _CONFIG_PATH.replace("\\", "/")

        cmd = [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--gpus", "all",
            "-p", f"{CONTAINER_PORT}:8080",
            "-v", f"{model_mount}:/models/PaddleOCR-VL-1.5:ro",
            "-v", f"{config_mount}:/config/vllm_config.yaml:ro",
            DOCKER_IMAGE,
            "bash", "-c",
            "mkdir -p /home/paddleocr/.paddlex/official_models "
            "&& cp -r /models/PaddleOCR-VL-1.5 /home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.5 "
            "&& paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8080 --backend vllm --backend_config /config/vllm_config.yaml",
        ]

        code, out = _run(cmd, timeout=60)
        if code != 0:
            logger.error(f"容器启动失败: {out}")
            return False, f"容器启动失败: {out}"
        logger.info("容器已创建，等待模型加载...")

    for i in range(60):
        time.sleep(3)
        if _vlm_health_check():
            logger.info(f"VLM 服务就绪 (等待 {(i+1)*3}s)")
            return True, f"VLM 服务已启动 (模型加载 {(i+1)*3}s)"
    return False, "VLM 模型加载超时 (180s)"


def stop_vllm_container() -> tuple[bool, str]:
    if not is_container_running():
        _run(["docker", "rm", "-f", CONTAINER_NAME])
        return True, "VLM 容器已停止"
    code, out = _run(["docker", "stop", CONTAINER_NAME], timeout=30)
    _run(["docker", "rm", "-f", CONTAINER_NAME])
    if code == 0:
        logger.info("VLM 容器已停止并移除")
        return True, "VLM 容器已停止"
    return False, f"停止失败: {out}"


def ensure_vlm_ready() -> tuple[bool, str]:
    if _vlm_health_check():
        return True, "VLM 服务: 运行中"

    if not is_docker_running():
        ok, msg = start_docker_desktop()
        if not ok:
            return False, msg

    return start_vllm_container()
