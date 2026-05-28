"""文件名 → drawing_no / revision 解析。§12 Phase 3。

兜底策略（[docs/plm_integration_design.md](../docs/plm_integration_design.md) §3.7 L1）：
- 优先匹配项目通用 Y 编号正则（config.make_pattern）
- 同时尝试识别版本号片段：常见形态 `_R1`、`_Rev2`、`-A`、`_v3`
- 全部失败 → `drawing_no = 原文件名 stem`, `revision = "N/A"`
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from config import DEFAULT_PREFIXES, make_pattern


@dataclass
class ParsedFilename:
    drawing_no: str
    revision: str
    matched: bool  # 是否成功命中项目编号正则


# 版本号探测——按优先级，命中即停
_REV_PATTERNS = [
    re.compile(r"[_\-](?:Rev|REV|rev)([0-9A-Za-z]{1,4})"),    # _Rev2, -REV01
    re.compile(r"[_\-]R([0-9]{1,3})(?![A-Za-z])"),             # _R1, -R03（避免吞 -RA）
    re.compile(r"[_\-]v([0-9]{1,3})(?![A-Za-z])", re.I),       # _v3, -V01
    re.compile(r"-([A-Z])(?=[\._]|$)"),                        # -A 结尾（不含字母后缀）
]


def parse(filename: str, prefixes: list[str] | None = None) -> ParsedFilename:
    """从文件名（含或不含路径）解析 drawing_no / revision。

    Args:
        filename: 文件名或完整路径
        prefixes: 图号首字母候选，默认 config.DEFAULT_PREFIXES

    Returns:
        ParsedFilename：matched=False 表示走兜底，drawing_no=stem，revision='N/A'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    pattern = re.compile(make_pattern(prefixes or DEFAULT_PREFIXES))
    m = pattern.search(stem.upper())
    drawing_no = m.group(0) if m else None

    revision: str | None = None
    for pat in _REV_PATTERNS:
        rm = pat.search(stem)
        if rm:
            revision = rm.group(1)
            break

    if drawing_no is None:
        return ParsedFilename(drawing_no=stem, revision=revision or "N/A", matched=False)

    return ParsedFilename(
        drawing_no=drawing_no,
        revision=revision or "N/A",
        matched=True,
    )
