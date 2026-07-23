"""定位项目字体；找不到回落到 PIL 默认。"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_FONT_NAME = "basictitlefont-1.ttf"


def find_font() -> str | None:
    """返回字体绝对路径，找不到返回 None。

    搜索顺序：
    1. PyInstaller 打包环境：sys._MEIPASS/fonts/<name>（exe 内置）
    2. <repo_root>/fonts/<name>（开发态，manual_editor 是 repo 下的子目录）
    3. 当前工作目录下的 fonts/<name>
    """
    here = Path(__file__).resolve()
    candidates = []
    # PyInstaller 打包后，资源解压到 sys._MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "fonts" / _PROJECT_FONT_NAME)
    candidates += [
        here.parent.parent.parent / "fonts" / _PROJECT_FONT_NAME,
        Path.cwd() / "fonts" / _PROJECT_FONT_NAME,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
