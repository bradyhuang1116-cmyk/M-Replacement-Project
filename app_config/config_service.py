"""配置管理服务 — 运行时读取/持久化配置覆盖层。

三层优先级（高→低）：
  1. config_overrides.json（持久化覆盖值）
  2. 系统环境变量（os.environ）
  3. config.py 模块默认值

覆盖值持久化到 data/config_overrides.json（原子写入），不修改 config.py。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import config as config_module

logger = logging.getLogger(__name__)

# ── 覆盖文件路径（相对于项目根目录） ───────────────────────────
OVERRIDES_DIR = Path(config_module.DATA_DIR)
OVERRIDES_PATH = OVERRIDES_DIR / "config_overrides.json"

# ── 配置文件路径（config.py 所在目录） ──
CONFIG_DIR = Path(os.path.dirname(os.path.abspath(config_module.__file__)))


# ── 元数据：驱动前端渲染每个配置字段 ─────────────────────────

CONFIG_META: dict[str, dict] = {
    # Worker
    "WORKER_POLL_INTERVAL": {
        "category": "Worker",
        "type": "number",
        "label": "Poll Interval (s)",
        "description": "Worker idle polling interval in seconds",
        "step": 0.1,
    },
    "WORKER_MAX_RETRY": {
        "category": "Worker",
        "type": "number",
        "label": "Max Retry Count",
        "description": "Maximum retry attempts for a failed task",
        "step": 1,
    },
    # Watch Folder
    "WATCH_INBOX_DIR": {
        "category": "Watch Folder",
        "type": "text",
        "label": "Inbox Directory",
        "description": "PLM file drop directory",
    },
    "WATCH_OUTPUT_DIR": {
        "category": "Watch Folder",
        "type": "text",
        "label": "Output Directory",
        "description": "Processed file output directory",
    },
    "WATCH_FAILED_DIR": {
        "category": "Watch Folder",
        "type": "text",
        "label": "Failed Directory",
        "description": "Failed file output directory",
    },
    "WATCH_STABILITY_SECONDS": {
        "category": "Watch Folder",
        "type": "number",
        "label": "Stability Window (s)",
        "description": "Seconds to wait after last file write before enqueuing",
        "step": 0.5,
    },
    "WATCH_SCAN_INTERVAL": {
        "category": "Watch Folder",
        "type": "number",
        "label": "Scan Interval (s)",
        "description": "Interval between inbox directory scans",
        "step": 0.5,
    },
    # Oracle Database
    "ORACLE_HOST": {
        "category": "Oracle Database",
        "type": "text",
        "label": "Host",
        "description": "Oracle database server hostname or IP",
    },
    "ORACLE_PORT": {
        "category": "Oracle Database",
        "type": "number",
        "label": "Port",
        "description": "Oracle listener port (default: 1521)",
        "step": 1,
    },
    "ORACLE_SERVICE_NAME": {
        "category": "Oracle Database",
        "type": "text",
        "label": "Service Name",
        "description": "Oracle service name or SID",
    },
    "ORACLE_USER": {
        "category": "Oracle Database",
        "type": "text",
        "label": "Username",
        "description": "Oracle database login user",
    },
    "ORACLE_PASSWORD": {
        "category": "Oracle Database",
        "type": "text",
        "label": "Password",
        "description": "Oracle database login password",
    },
    "ORACLE_MIN_POOL": {
        "category": "Oracle Database",
        "type": "number",
        "label": "Min Pool Size",
        "description": "Oracle connection pool minimum connections",
        "step": 1,
    },
    "ORACLE_MAX_POOL": {
        "category": "Oracle Database",
        "type": "number",
        "label": "Max Pool Size",
        "description": "Oracle connection pool maximum connections",
        "step": 1,
    },
    "ORACLE_PATH_PREFIX": {
        "category": "Oracle Database",
        "type": "text",
        "label": "File Path Prefix",
        "description": "Prefix for SIPM197.LOCATION relative path (e.g. D:\\PLM719\\filedata)",
    },
    "PLM_OUTPUT_BASE_DIR": {
        "category": "Oracle Database",
        "type": "text",
        "label": "PLM Output Base",
        "description": "Base directory for PLM processed output (e.g. D:\\SMEC)",
    },
    # WinSCP SFTP
    "WINSCP_ENABLED": {
        "category": "WinSCP SFTP",
        "type": "text",
        "label": "Enable SFTP",
        "description": "Set to 'true' to enable WinSCP remote file transfer",
    },
    "WINSCP_HOST": {
        "category": "WinSCP SFTP",
        "type": "text",
        "label": "SFTP Host",
        "description": "Remote SFTP server hostname or IP address",
    },
    "WINSCP_PORT": {
        "category": "WinSCP SFTP",
        "type": "number",
        "label": "SFTP Port",
        "description": "Remote SFTP server port (default: 22)",
        "step": 1,
    },
    "WINSCP_USER": {
        "category": "WinSCP SFTP",
        "type": "text",
        "label": "SFTP Username",
        "description": "SFTP login username",
    },
    "WINSCP_PASSWORD": {
        "category": "WinSCP SFTP",
        "type": "text",
        "label": "SFTP Password",
        "description": "SFTP login password",
    },
    "WINSCP_EXE_PATH": {
        "category": "WinSCP SFTP",
        "type": "text",
        "label": "WinSCP Executable Path",
        "description": "Full path to WinSCP.exe on this machine",
    },
}

CATEGORIES: dict[str, dict] = {
    "Worker": {
        "description": "Background worker queue consumer behavior",
        "icon": "Cog",
    },
    "Watch Folder": {
        "description": "PLM folder monitoring and document drop settings",
        "icon": "HardDrive",
    },
    "Oracle Database": {
        "description": "PLM Oracle database connection configuration",
        "icon": "Database",
    },
    "WinSCP SFTP": {
        "description": "SFTP connection settings for remote file transfer via WinSCP",
        "icon": "Server",
    },
}


class ConfigService:
    """配置管理服务单例。

    职责：
    - 从 config.py / os.environ / config_overrides.json 合并读取配置
    - 将用户修改持久化到 config_overrides.json
    - 返回按 category 分组的完整配置数据供前端渲染
    """

    _instance: ConfigService | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if ConfigService._instance is not None:
            raise RuntimeError("Use ConfigService.get_instance() instead")
        self._overrides: dict[str, Any] = {}
        self._overrides_lock = threading.Lock()
        self._load_overrides()

    # ── 单例 ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> ConfigService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 覆盖文件读写 ─────────────────────────────────────────────

    def _overrides_path(self) -> Path:
        return OVERRIDES_PATH

    def _load_overrides(self) -> None:
        path = self._overrides_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._overrides = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("读取配置覆盖文件失败，使用空覆盖: %s", e)
                self._overrides = {}
        else:
            self._overrides = {}

        # 启动时将覆盖值同步到 config 模块属性
        # 使 from config import X 或 import config; config.X 能读到持久化的值
        if self._overrides:
            for k, v in self._overrides.items():
                setattr(config_module, k, v)
            logger.info("已将 %d 个覆盖值同步到 config 模块", len(self._overrides))

    def _save_overrides(self) -> None:
        path = self._overrides_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写入：写 .tmp → rename
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._overrides, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        logger.info("配置覆盖已持久化: %s", path)

    # ── 值读取 / 类型转换 ────────────────────────────────────────

    @staticmethod
    def _get_default(key: str) -> Any:
        """从 config.py 模块读取默认值。"""
        return getattr(config_module, key, None)

    @staticmethod
    def _coerce(key: str, raw: Any) -> Any:
        """将原始值按 config.py 默认值的类型做转换。"""
        default = ConfigService._get_default(key)
        if isinstance(default, list):
            if isinstance(raw, str):
                return [s.strip() for s in raw.split(",") if s.strip()]
            if isinstance(raw, list):
                return [str(s).strip() for s in raw if str(s).strip()]
            return list(raw) if raw else list(default)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, int):
            return int(raw)
        return str(raw)

    def _read_env(self, key: str) -> Any:
        """从环境变量读取，做类型转换。"""
        raw = os.environ.get(key)
        if raw is None:
            return None
        try:
            return self._coerce(key, raw)
        except (ValueError, TypeError):
            return raw

    def get_effective(self, key: str) -> Any:
        """返回生效值（override > env > config.py default）。"""
        with self._overrides_lock:
            if key in self._overrides:
                return self._overrides[key]
        env_val = self._read_env(key)
        if env_val is not None:
            return env_val
        return self._get_default(key)

    def get_all_effective(self) -> dict[str, Any]:
        """返回所有暴露配置的生效值。"""
        return {key: self.get_effective(key) for key in CONFIG_META}

    # ── 更新覆盖值 ──────────────────────────────────────────────

    def update_overrides(self, overrides: dict[str, Any]) -> dict[str, str]:
        """更新覆盖值并持久化。

        Args:
            overrides: {key: value} 字典，只包含需要修改的项。

        Returns:
            { "status": "ok" }，或遇到无效键时抛出 ValueError。
        """
        # 验证键
        unknown = [k for k in overrides if k not in CONFIG_META]
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

        # 类型转换
        coerced = {}
        for key, raw in overrides.items():
            try:
                coerced[key] = self._coerce(key, raw)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {key}: {e}")

        with self._overrides_lock:
            self._overrides.update(coerced)
            self._save_overrides()

            # 热应用：更新 config 模块的属性，使 import config; config.X 立即生效
            for k, v in coerced.items():
                setattr(config_module, k, v)

        return {"status": "ok"}

    # ── 前端完整数据 ────────────────────────────────────────────

    def get_full_config(self) -> dict:
        """返回前端渲染所需的完整配置数据。"""
        return {
            "current": self.get_all_effective(),
            "meta": CONFIG_META,
            "categories": CATEGORIES,
        }
