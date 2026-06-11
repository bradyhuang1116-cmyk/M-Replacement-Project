"""HXXX-R.tif <-> XXX.tif <-> XXX (CSV source_file stem) 的命名规则。

Examples
--------
>>> replaced_to_stem('HYA116A226-1_Eg-脱敏-R.tif')
'YA116A226-1_Eg-脱敏'
>>> replaced_to_original_name('HYA116A226-1_Eg-脱敏-R.tif')
'YA116A226-1_Eg-脱敏.tif'
"""
from __future__ import annotations

from pathlib import Path


_TIF_EXTS = {".tif", ".tiff"}


def _split_ext(name: str) -> tuple[str, str]:
    p = Path(name)
    return p.stem, p.suffix


def replaced_to_stem(replaced_name: str) -> str:
    """`HXXX-R.tif` -> `XXX`（用于匹配 CSV 的 source_file 列）。

    规则：去首字符 `H`，去末尾 `-R`，丢扩展名。
    若文件名不符合规则，按尽力剥离原则返回 stem。
    """
    stem, _ = _split_ext(replaced_name)
    if stem.startswith("H"):
        stem = stem[1:]
    if stem.endswith("-R"):
        stem = stem[:-2]
    return stem


def replaced_to_original_name(replaced_name: str) -> str:
    """`HXXX-R.tif` -> `XXX.tif`（保持原扩展名大小写，找原图文件）。

    不在 _TIF_EXTS 内的扩展名归一为 .tif。
    """
    _, ext = _split_ext(replaced_name)
    orig_stem = replaced_to_stem(replaced_name)
    out_ext = ext if ext.lower() in _TIF_EXTS else ".tif"
    return orig_stem + out_ext


def is_replaced_name(name: str) -> bool:
    """是否符合 `H...-R.tif/.tiff` 模式。"""
    stem, ext = _split_ext(name)
    if ext.lower() not in _TIF_EXTS:
        return False
    return stem.startswith("H") and stem.endswith("-R")
