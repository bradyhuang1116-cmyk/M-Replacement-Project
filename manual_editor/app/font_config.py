"""定位项目字体；找不到回落到 PIL 默认。"""
from __future__ import annotations

from pathlib import Path

_PROJECT_FONT_NAME = "dingliesongtypeface20241217-2.ttf"


def find_font() -> str | None:
    """返回字体绝对路径，找不到返回 None。

    搜索顺序：
    1. <repo_root>/fonts/<name>（manual_editor 是 repo 下的子目录）
    2. 当前工作目录下的 fonts/<name>
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "fonts" / _PROJECT_FONT_NAME,
        Path.cwd() / "fonts" / _PROJECT_FONT_NAME,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
