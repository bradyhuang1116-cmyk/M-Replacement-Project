"""加载 y_boxes.csv -> {source_file_stem: list[BoxRecord]}。

CSV 列：source_file, token, x1, y1, x2, y2
（与项目主流水线 modules/factory_note_pixel.flush_y_boxes_csv 的输出一致）
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoxRecord:
    token: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


def find_csv(folder: str | Path) -> Path | None:
    """优先 y_boxes.csv；否则取目录下任意一个 .csv（按名字排序取第一个）。"""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    primary = folder / "y_boxes.csv"
    if primary.exists():
        return primary
    candidates = sorted(folder.glob("*.csv"))
    return candidates[0] if candidates else None


def load_csv(folder: str | Path) -> dict[str, list[BoxRecord]]:
    """读取 folder 下的 CSV，按 source_file 分组返回。

    Returns
    -------
    dict[stem, list[BoxRecord]] —— stem 是 CSV 中 source_file 列的值
    （不带扩展名）。
    """
    csv_path = find_csv(folder)
    if csv_path is None:
        logger.warning("未在 %s 找到 .csv", folder)
        return {}

    out: dict[str, list[BoxRecord]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = BoxRecord(
                    token=row.get("token", ""),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    x2=int(row["x2"]),
                    y2=int(row["y2"]),
                )
            except (KeyError, ValueError) as e:
                logger.warning("跳过无效 CSV 行 %s: %s", row, e)
                continue
            raw = row.get("source_file", "").strip()
            if not raw:
                continue
            # 兼容两种历史格式：带扩展名 (旧 factory_note_pixel_v6) 和裸 stem (新 text_replacer)
            stem = Path(raw).stem if raw.lower().endswith((".tif", ".tiff")) else raw
            out.setdefault(stem, []).append(rec)

    logger.info("从 %s 读到 %d 个文件, %d 条记录",
                csv_path, len(out), sum(len(v) for v in out.values()))
    return out
