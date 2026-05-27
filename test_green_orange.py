"""绿框 / 橙框最小化检测脚本。

只调当前主代码里的绿框和橙框定位逻辑：
  - _locate_bottom_right_number  (绿框, modules/region_detector.py)
  - _locate_top_left_number      (橙框, modules/region_detector.py)

跳过：
  - 红框 (_locate_material_code_column)
  - 工厂注意 (Phase C, factory_note_pixel.detect_factory_note_codes_v6)
  - text_replacer 替换

橙框搜索区域采用 detect_all_regions 中"红框未检测到"时的 fallback 公式
(左上 2/8 × 2/12)，路径与主流程在缺红框时完全一致。

输出：
  - 控制台打印 (绿框 bbox/文本, 橙框 bbox/文本)
  - test_output/green_orange_debug/<file>_debug.jpg 把两框画在原图上
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES
from modules.file_ingestion import load_file
from modules.docker_manager import ensure_vlm_ready
from modules.region_detector import (
    BBox,
    _auto_rotate_portrait,
    _crop_and_scale,
    _map_bbox_back,
    _locate_bottom_right_number,
    _locate_top_left_number,
)


# ── 配置 ──────────────────────────────────────────────────────
INPUT_DIR = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
KEYWORDS: list[str] = []    # 空 = 全跑；否则只跑包含其中关键字的文件
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "green_orange_debug")
PREFIXES = DEFAULT_PREFIXES


def _collect_files() -> list[str]:
    files = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if os.path.splitext(f)[1].lower() not in (".tif", ".tiff"):
            continue
        if not KEYWORDS:
            files.append(os.path.join(INPUT_DIR, f))
            continue
        for kw in KEYWORDS:
            if kw in f:
                files.append(os.path.join(INPUT_DIR, f))
                break
    assert files, f"no test files in {INPUT_DIR}"
    return files


def _detect_green(image: np.ndarray, img_h: int, img_w: int):
    """绿框搜索区域 + 定位。参数与 detect_all_regions 同。"""
    green_x = int(img_w * (5.0 / 8.0))
    green_y = int(img_h * (5.0 / 6.0))
    green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)
    logger.info(f"  绿框搜索区域: {green_search}")

    green_sub, green_scale = _crop_and_scale(image, green_search)
    logger.info(f"  绿框裁切: {green_sub.shape[1]}x{green_sub.shape[0]} (scale={green_scale:.3f})")

    res = _locate_bottom_right_number(green_sub, prefixes=PREFIXES)
    if res is None:
        return None, None, green_search
    text, bbox_sub = res
    bbox = _map_bbox_back(bbox_sub, green_search, green_scale)
    return text, bbox, green_search


def _detect_orange(image: np.ndarray, img_h: int, img_w: int):
    """橙框搜索区域 + 定位。使用红框未命中时的 fallback 区域 (左上 2/8 × 2/12)，
    路径与 detect_all_regions 在 mat_bbox is None 时一致。"""
    orange_search = BBox(0, 0, int(img_w * (2.0 / 8.0)), int(img_h * (2.0 / 12.0)))
    logger.info(f"  橙框搜索区域(fallback): {orange_search}")

    orange_sub, orange_scale = _crop_and_scale(image, orange_search)
    logger.info(f"  橙框裁切: {orange_sub.shape[1]}x{orange_sub.shape[0]} (scale={orange_scale:.3f})")

    res = _locate_top_left_number(orange_sub, None, prefixes=PREFIXES)
    if res is None:
        return None, None, orange_search
    text, bbox_sub = res
    bbox = _map_bbox_back(bbox_sub, orange_search, orange_scale)
    return text, bbox, orange_search


def _draw_debug(
    image: np.ndarray,
    green_bbox: BBox | None,
    orange_bbox: BBox | None,
    green_search: BBox,
    orange_search: BBox,
    out_path: str,
) -> None:
    vis = image.copy()
    # 搜索区域：细线，淡色
    cv2.rectangle(vis,
                  (green_search.x, green_search.y),
                  (green_search.x2, green_search.y2),
                  (180, 255, 180), 4)
    cv2.rectangle(vis,
                  (orange_search.x, orange_search.y),
                  (orange_search.x2, orange_search.y2),
                  (255, 220, 180), 4)
    # 最终框：粗线，正色
    if green_bbox is not None:
        cv2.rectangle(vis,
                      (green_bbox.x, green_bbox.y),
                      (green_bbox.x2, green_bbox.y2),
                      (0, 220, 0), 8)
    if orange_bbox is not None:
        cv2.rectangle(vis,
                      (orange_bbox.x, orange_bbox.y),
                      (orange_bbox.x2, orange_bbox.y2),
                      (255, 140, 0), 8)
    Image.fromarray(vis).save(out_path, quality=85)
    logger.info(f"  debug 图: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 绿框 OCR 用 V5；橙框 OCR 也走 V5（_locate_top_left_number_core 内部）
    # 所以这里不需要 VLM。但保持启动 VLM 以便和主流程环境一致；启动失败也不阻塞。
    ok, msg = ensure_vlm_ready()
    logger.info(f"VLM ready={ok}: {msg}")

    files = _collect_files()
    logger.info(f"测试文件: {[os.path.basename(f) for f in files]}")

    summary = []
    for fp in files:
        base = os.path.basename(fp)
        print("=" * 70)
        print(f"处理: {base}")

        img, _ = load_file(fp)
        img_h, img_w = img.shape[:2]

        # 纵向自动旋转 — 与 detect_all_regions 行为一致
        if img_h > img_w:
            img, rot = _auto_rotate_portrait(img, prefixes=PREFIXES)
            img_h, img_w = img.shape[:2]
            logger.info(f"  纵向图纸已旋转 (rot_code={rot})")

        green_text, green_bbox, green_search = _detect_green(img, img_h, img_w)
        if green_bbox is not None:
            print(f"  绿框: text='{green_text}', bbox={green_bbox}")
        else:
            print("  绿框: 未检测到")

        orange_text, orange_bbox, orange_search = _detect_orange(img, img_h, img_w)
        if orange_bbox is not None:
            print(f"  橙框: text='{orange_text}', bbox={orange_bbox}")
        else:
            print("  橙框: 未检测到")

        out_path = os.path.join(OUT_DIR, os.path.splitext(base)[0] + "_debug.jpg")
        _draw_debug(img, green_bbox, orange_bbox, green_search, orange_search, out_path)
        summary.append((base, green_bbox is not None, orange_bbox is not None))

    print("=" * 70)
    print("汇总:")
    for base, g_ok, o_ok in summary:
        print(f"  {base}: green={'OK' if g_ok else 'MISS'}, orange={'OK' if o_ok else 'MISS'}")


if __name__ == "__main__":
    main()
