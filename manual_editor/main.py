"""手动编号框编辑器 — 入口。

启动方式
--------
    # 任一种都可
    /c/Users/Brady\\ Huang/miniconda3/envs/mitsubishi/python.exe manual_editor/main.py
    /c/Users/Brady\\ Huang/miniconda3/envs/mitsubishi/python.exe -m manual_editor.main
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 直接 `python manual_editor/main.py` 时 sys.path[0] 是 manual_editor/ 本身，
# 需要把项目根加入路径，包导入才能找到 manual_editor 包。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from manual_editor.app.main_window import MainWindow  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
