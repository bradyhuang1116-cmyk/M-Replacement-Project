"""WinSCP client wrapper for SFTP file transfers."""
from __future__ import annotations

import locale
import logging
import os
import subprocess
import tempfile
import urllib.parse
from pathlib import Path

from config import (
    DATA_DIR,
    WINSCP_ENABLED,
    WINSCP_EXE_PATH,
    WINSCP_HOST,
    WINSCP_PASSWORD,
    WINSCP_PORT,
    WINSCP_USER,
)

logger = logging.getLogger(__name__)

_WINSCP_LOG = Path(DATA_DIR) / "winscp.log"


class WinSCPError(Exception):
    """Raised when a WinSCP operation fails."""


class WinSCPClient:
    """SFTP transfer client backed by WinSCP."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        exe_path: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._host = host or WINSCP_HOST
        self._port = port or WINSCP_PORT
        self._username = username or WINSCP_USER
        self._password = password or WINSCP_PASSWORD
        self._exe_path = exe_path or WINSCP_EXE_PATH
        self._command_path = self._resolve_command_path(self._exe_path)
        self._enabled = WINSCP_ENABLED
        self._timeout = timeout
        self._log_path = str(_WINSCP_LOG)

    def is_available(self) -> bool:
        """Return whether WinSCP is enabled and executable exists."""
        if not self._enabled:
            return False
        if not self._command_path or not os.path.isfile(self._command_path):
            logger.debug("WinSCP executable not found: %s", self._command_path)
            return False
        return True

    def download(self, remote_path: str, local_path: str) -> tuple[bool, str]:
        """Download one file from the remote SFTP server."""
        if not self.is_available():
            return False, "WinSCP unavailable"

        remote_sftp = self._normalize_remote_path(remote_path)
        local_target = self._normalize_local_path(local_path)
        Path(local_target).parent.mkdir(parents=True, exist_ok=True)
        script_lines = self._build_script_lines(
            f'get "{remote_sftp}" "{local_target}"',
        )
        return self._run(script_lines, action=f'get "{remote_sftp}" "{local_target}"')

    def upload(self, local_path: str, remote_path: str) -> tuple[bool, str]:
        """Upload one file to the remote SFTP server."""
        if not self.is_available():
            return False, "WinSCP unavailable"

        local_source = self._normalize_local_path(local_path)
        remote_sftp = self._normalize_remote_path(remote_path)
        remote_dir_sftp = self._normalize_remote_path(os.path.dirname(remote_path))

        script_lines = self._build_script_lines(
            *self._build_remote_mkdir_lines(remote_dir_sftp),
            f'put "{local_source}" "{remote_sftp}"',
        )
        return self._run(script_lines, action=f'put "{local_source}" "{remote_sftp}"')

    def _sanitized_log(self) -> str:
        """Return a sanitized connection string for logs."""
        return f"sftp://{self._username}:***@{self._host}:{self._port}"

    def _resolve_command_path(self, exe_path: str) -> str:
        """Prefer WinSCP.com to avoid spawning a visible console window."""
        if not exe_path:
            return exe_path

        path = Path(exe_path)
        if path.suffix.lower() == ".com":
            return str(path)

        console_path = path.with_suffix(".com")
        if console_path.is_file():
            return str(console_path)
        return str(path)

    def _base_args(self) -> list[str]:
        """Build common WinSCP launch arguments."""
        args = [self._command_path]
        if self._command_path.lower().endswith(".exe"):
            args.extend(["/console", "/nointeractiveinput", "/nointeractive"])
        args.append("/ini=nul")
        return args

    def _normalize_remote_path(self, remote_path: str) -> str:
        """Normalize Windows-style remote paths for WinSCP SFTP syntax."""
        normalized = remote_path.replace("\\", "/")
        if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
            normalized = f"/{normalized}"
        return normalized

    def _normalize_local_path(self, local_path: str) -> str:
        """Normalize local Windows paths for WinSCP script commands."""
        return str(Path(local_path).resolve())

    def _build_remote_mkdir_lines(self, remote_dir: str) -> list[str]:
        """Build mkdir commands for each missing remote directory level."""
        normalized = remote_dir.replace("\\", "/").rstrip("/")
        if not normalized:
            return []

        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return []

        current = f"/{parts[0]}"
        commands = ["option batch continue"]
        for part in parts[1:]:
            current = f"{current}/{part}"
            commands.append(f'mkdir "{current}"')
        commands.append("option batch abort")
        return commands

    def _encode_password(self) -> str:
        """URL-encode password special characters."""
        return urllib.parse.quote(self._password, safe="")

    def _build_session_url(self) -> str:
        """Build WinSCP session URL with credentials."""
        password_enc = self._encode_password()
        return f"sftp://{self._username}:{password_enc}@{self._host}:{self._port}"

    def _build_script_lines(self, *commands: str) -> list[str]:
        """Build a WinSCP script file body."""
        return [
            "option batch abort",
            "option confirm off",
            f"open {self._build_session_url()} -hostkey=*",
            *commands,
            "exit",
        ]

    def _create_script_file(self, script_lines: list[str]) -> str:
        """Write the WinSCP script to a temporary file."""
        script_dir = Path(DATA_DIR)
        script_dir.mkdir(parents=True, exist_ok=True)
        fd, script_path = tempfile.mkstemp(
            prefix="winscp_",
            suffix=".txt",
            dir=script_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="\n") as handle:
                handle.write("\n".join(script_lines))
                handle.write("\n")
        except Exception:
            raise
        return script_path

    def _build_command_args(self, script_path: str) -> list[str]:
        """Build WinSCP command-line arguments."""
        return [
            *self._base_args(),
            f"/log={self._log_path}",
            "/loglevel=1",
            f"/script={script_path}",
        ]

    def _decode_output(self, data: bytes | None) -> str:
        """Decode subprocess output without crashing on mixed encodings."""
        if not data:
            return ""

        encodings: list[str] = ["utf-8"]
        preferred = locale.getpreferredencoding(False)
        if preferred and preferred.lower() not in {"utf-8", "utf8"}:
            encodings.append(preferred)
        encodings.append("gb18030")

        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode(encodings[0], errors="replace")

    def _run(self, script_lines: list[str], action: str) -> tuple[bool, str]:
        """Run WinSCP and inspect the exit code."""
        logger.info("WinSCP: %s %s ...", self._sanitized_log(), action)
        logger.debug("WinSCP script: %s", " | ".join(
            line.replace(self._password, "***") for line in script_lines
        ))

        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        script_path = self._create_script_file(script_lines)
        args = self._build_command_args(script_path)
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=False,
                timeout=self._timeout,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        except FileNotFoundError:
            return False, f"WinSCP executable not found: {self._command_path}"
        except subprocess.TimeoutExpired:
            return False, f"WinSCP timeout ({self._timeout}s)"
        except OSError as exc:
            return False, f"WinSCP execution failed: {exc}"
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass

        if result.returncode != 0:
            stderr = self._decode_output(result.stderr).strip()
            stdout = self._decode_output(result.stdout).strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            logger.warning("WinSCP failed (rc=%d): %s", result.returncode, detail)
            return False, detail

        logger.info("WinSCP succeeded")
        return True, ""
