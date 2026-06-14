"""PLM 对接编排层 — 对方 Oracle 对接层调用入口。

流程：获取远程原图 → 本地处理 → 回写远程归档路径。
"""

from __future__ import annotations

import logging
import os
import shutil

from app_config.config_service import ConfigService
from config import DEFAULT_PREFIXES
from modules.batch_processor import process_single_file
from modules.factory_note_pixel import clear_y_box_records

logger = logging.getLogger(__name__)


def _use_local_stubs() -> bool:
    """联调用：PLM_LOCAL_STUBS=1 时启用本地文件复制占位（非生产逻辑）。"""
    return os.getenv("PLM_LOCAL_STUBS", "").lower() in ("1", "true", "yes")


def _find_latest_output(output_dir_local: str) -> str:
    """在本地 output 的 vector/ocr 子目录中找最近修改的产物文件。"""
    candidates: list[str] = []
    for sub in ("vector", "ocr"):
        subdir = os.path.join(output_dir_local, sub)
        if not os.path.isdir(subdir):
            continue
        for name in os.listdir(subdir):
            path = os.path.join(subdir, name)
            if os.path.isfile(path):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"本地 output 目录无产物: {output_dir_local}")
    return max(candidates, key=os.path.getmtime)


def fetch_remote_file(file_path: str, file_path_local: str) -> str:
    """从远程路径获取原图到本地 inbox（WinSCP SFTP download，不可用时回退本地复制）。

    Args:
        file_path: 远程/PLM 侧原图路径
        file_path_local: 本地 inbox 目录（Settings → Inbox Directory）

    Returns:
        本地可处理的完整文件路径
    """
    if _use_local_stubs():
        os.makedirs(file_path_local, exist_ok=True)
        dest = os.path.join(file_path_local, os.path.basename(file_path))
        shutil.copy2(file_path, dest)
        return dest

    os.makedirs(file_path_local, exist_ok=True)
    dest = os.path.join(file_path_local, os.path.basename(file_path))

    from modules.winscp_client import WinSCPClient

    win = WinSCPClient()
    if win.is_available():
        ok, err = win.download(file_path, dest)
        if not ok:
            raise RuntimeError(f"SFTP 下载失败: {file_path} → {dest}: {err}")
        logger.info("已下载 (SFTP) 到 inbox: %s → %s", file_path, dest)
        return dest

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"源文件不存在且 WinSCP 不可用: {file_path!r}")
    shutil.copy2(file_path, dest)
    logger.info("已复制 (local) 到 inbox: %s → %s", file_path, dest)
    return dest


def write_back_to_remote(output_dir: str, output_dir_local: str) -> str:
    """将处理后的文件从本地 output 回写到远程归档路径（WinSCP SFTP upload，不可用时回退本地复制）。

    Args:
        output_dir: 远程/PLM 侧归档目录
        output_dir_local: 本地 output 目录（Settings → Output Directory）

    Returns:
        远程侧最终文件完整路径（作为返回值 file_path）
    """
    if _use_local_stubs():
        local_output = _find_latest_output(output_dir_local)
        os.makedirs(output_dir, exist_ok=True)
        remote_path = os.path.join(output_dir, os.path.basename(local_output))
        shutil.copy2(local_output, remote_path)
        return remote_path

    local_output = _find_latest_output(output_dir_local)
    remote_path = os.path.join(output_dir, os.path.basename(local_output))

    from modules.winscp_client import WinSCPClient

    win = WinSCPClient()
    if win.is_available():
        ok, err = win.upload(local_output, remote_path)
        if not ok:
            raise RuntimeError(f"SFTP 上传失败: {local_output} → {remote_path}: {err}")
        logger.info("已上传 (SFTP) 到远程: %s → %s", local_output, remote_path)
        return remote_path

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(local_output, remote_path)
    logger.info("已复制 (local) 到远程路径: %s → %s", local_output, remote_path)
    return remote_path


def plm_process_drawing(file_path: str, output_dir: str) -> dict:
    """PLM 图纸处理入口。

    Args:
        file_path: 远程/PLM 侧原图路径
        output_dir: 远程/PLM 侧归档目录

    Returns:
        {status, method, total, file_path}
        - method: "ocr" | "vector"
        - file_path: 回写后的远程文件路径
    """
    cfg = ConfigService.get_instance()
    file_path_local = str(cfg.get_effective("WATCH_INBOX_DIR"))
    output_dir_local = str(cfg.get_effective("WATCH_OUTPUT_DIR"))

    os.makedirs(file_path_local, exist_ok=True)
    os.makedirs(output_dir_local, exist_ok=True)

    logger.info(
        "PLM 处理开始: remote=%s → inbox=%s, remote_out=%s → local_out=%s",
        file_path, file_path_local, output_dir, output_dir_local,
    )

    try:
        local_input_path = fetch_remote_file(file_path, file_path_local)
    except NotImplementedError:
        raise
    except Exception as e:
        logger.error("获取远程文件失败: %s", e)
        return {
            "status": "error",
            "method": "",
            "total": 0,
            "file_path": "",
        }

    if not local_input_path or not os.path.isfile(local_input_path):
        return {
            "status": "error",
            "method": "",
            "total": 0,
            "file_path": "",
        }

    clear_y_box_records()

    try:
        result = process_single_file(
            local_input_path,
            output_dir_local,
            generate_debug=False,
            prefixes=list(DEFAULT_PREFIXES),
        )
    except Exception as e:
        logger.error("图纸处理失败: %s", e, exc_info=True)
        return {
            "status": "error",
            "method": "",
            "total": 0,
            "file_path": "",
        }

    if result.get("status") != "success":
        return {
            "status": result.get("status", "error"),
            "method": result.get("method", ""),
            "total": result.get("total", 0),
            "file_path": "",
        }

    try:
        remote_file_path = write_back_to_remote(output_dir, output_dir_local)
    except NotImplementedError:
        raise
    except Exception as e:
        logger.error("回写远程路径失败: %s", e)
        return {
            "status": "error",
            "method": result.get("method", ""),
            "total": result.get("total", 0),
            "file_path": "",
        }

    logger.info(
        "PLM 处理完成: status=%s method=%s total=%s file_path=%s",
        result["status"], result["method"], result["total"], remote_file_path,
    )

    return {
        "status": result["status"],
        "method": result["method"],
        "total": result.get("total", 0),
        "file_path": remote_file_path,
    }
