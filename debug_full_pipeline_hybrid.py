"""全流程调试脚本（混合模式：红框v5定位 + 绿/橙框VLM检测）
   重点可视化绿框/橙框的检测过程"""
import os
import sys
import logging
import subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── GPU 显存监控工具 ──────────────────────────────────────────
_gpu_peak_mib = 0.0


def _get_gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=5,
        )
        parts = out.strip().split(",")
        used, total, free = float(parts[0]), float(parts[1]), float(parts[2])
        return used, total, free
    except Exception:
        return 0, 0, 0


def _log_gpu(tag: str):
    global _gpu_peak_mib
    used, total, free = _get_gpu_mem()
    if used > _gpu_peak_mib:
        _gpu_peak_mib = used
    logger.info(f"  [GPU] {tag}: {used:.0f}/{total:.0f} MiB (空闲 {free:.0f}), 峰值 {_gpu_peak_mib:.0f} MiB")

# ── 混合模式：红框用 v5 定位，绿框/橙框走 VLM（region_detector 内部自动切换）──

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, make_pattern
from modules.file_ingestion import load_file
from modules.region_detector import (
    detect_all_regions, draw_regions_debug, _enhance_vertical_lines,
    BBox,
    _crop_and_scale, _map_bbox_back,
    _locate_bottom_right_number_core,
    _detect_morph_lines, _find_cell_from_lines, _make_green_bbox,
    _ocr_region,
)
from modules.text_replacer import (
    detect_cyan_boxes, replace_in_all_regions,
)

_default_input = r"C:\Users\Brady Huang\Downloads\TIF_Undo\YA070A191P7933-1_0-脱敏.tif"
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else _default_input
_input_stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "test_output", f"debug_pipeline_hybrid_{_input_stem}")
os.makedirs(DEBUG_DIR, exist_ok=True)

import glob as _glob
for _f in _glob.glob(os.path.join(DEBUG_DIR, "*.jpg")):
    os.remove(_f)

prefixes = DEFAULT_PREFIXES

# ── 0. GPU 初始状态 ──
logger.info("=" * 60)
_log_gpu("启动前")

# ── 1. 加载文件 ──
logger.info("Step 1: 加载文件")
img_array, meta = load_file(INPUT_FILE)
logger.info(f"  尺寸: {img_array.shape[1]}x{img_array.shape[0]}")
Image.fromarray(img_array).save(os.path.join(DEBUG_DIR, "01_original.jpg"), quality=90)

# ── 2. 竖线增强 ──
logger.info("Step 2: 竖线增强")
enhanced = _enhance_vertical_lines(img_array)
Image.fromarray(enhanced).save(os.path.join(DEBUG_DIR, "02_enhanced.jpg"), quality=90)

# ── 3. 区域检测 ──
logger.info("Step 3: 区域检测（红/绿/橙框 + 工厂注意）")
_log_gpu("区域检测前")
regions = detect_all_regions(enhanced, prefixes=prefixes)
_log_gpu("区域检测后(v5)")

rot_code = regions.get("_metadata", {}).get("rotation")
if rot_code is not None:
    img_array = cv2.rotate(img_array, rot_code)
    enhanced = cv2.rotate(enhanced, rot_code)
    logger.info(f"  应用旋转: rot_code={rot_code}")

debug_img = draw_regions_debug(img_array, regions)
Image.fromarray(debug_img).save(os.path.join(DEBUG_DIR, "03_regions_debug.jpg"), quality=90)

img_h, img_w = img_array.shape[:2]
metadata = regions.get("_metadata", {})
search_areas = metadata.get("search_areas", {})
crop_scales = metadata.get("crop_scales", {})

# ── 搜索区总览 ──
overview = img_array.copy()
sa_colors = {"red_search": (255, 0, 0), "green_search": (0, 200, 0), "orange_search": (255, 165, 0)}
for name, color in sa_colors.items():
    sa = search_areas.get(name)
    if sa:
        cv2.rectangle(overview, (sa.x, sa.y), (sa.x2, sa.y2), color, 4)
        cv2.putText(overview, name, (sa.x + 5, sa.y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
Image.fromarray(overview).save(os.path.join(DEBUG_DIR, "03_search_areas_overview.jpg"), quality=90)


# ══════════════════════════════════════════════════════════════════
#  绿框（bottom_right_number）过程可视化 — 暂停，跳过执行
# ══════════════════════════════════════════════════════════════════
SKIP_GREEN_VIS = True
logger.info("=" * 40)
logger.info("Step G: 绿框检测过程可视化" + ("（已跳过）" if SKIP_GREEN_VIS else ""))

if not SKIP_GREEN_VIS:
    green_search = search_areas.get("green_search")
    green_scale = crop_scales.get("green", 1.0)
    green_bbox = regions.get("bottom_right_number")

if not SKIP_GREEN_VIS and green_search:
    # G1: 搜索区裁切
    green_sub, green_scale = _crop_and_scale(enhanced, green_search)
    green_sub_vis = green_sub.copy()
    gh, gw = green_sub.shape[:2]
    logger.info(f"  绿框搜索区: {green_search}, 裁切尺寸={gw}x{gh}, scale={green_scale:.3f}")
    Image.fromarray(green_sub).save(os.path.join(DEBUG_DIR, "G1_green_search_crop.jpg"), quality=90)

    # G2: 在搜索区裁切上做 OCR，显示所有 OCR 结果
    gray_g = cv2.cvtColor(green_sub, cv2.COLOR_RGB2GRAY)
    ocr_all = _ocr_region(green_sub, BBox(0, 0, gw, gh), engine="vlm")
    green_ocr_vis = green_sub.copy()
    for i, (text, conf, poly) in enumerate(ocr_all):
        if poly and len(poly) >= 4:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cv2.rectangle(green_ocr_vis, (x1, y1), (x2, y2), (200, 200, 0), 1)
            cv2.putText(green_ocr_vis, f"{text[:20]}",
                        (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
    Image.fromarray(green_ocr_vis).save(os.path.join(DEBUG_DIR, "G2_green_all_ocr.jpg"), quality=90)
    logger.info(f"  绿框 OCR 结果: {len(ocr_all)} 条")
    for i, (t, c, _) in enumerate(ocr_all):
        logger.info(f"    [{i}] '{t}' (conf={c:.2f})")

    # G3: core 检测结果（第一轮 OCR bbox）
    core_result = _locate_bottom_right_number_core(green_sub, gray_g, gh, gw, prefixes)
    if core_result:
        core_text, core_bbox = core_result
        green_core_vis = green_sub.copy()
        cv2.rectangle(green_core_vis, (core_bbox.x, core_bbox.y),
                      (core_bbox.x2, core_bbox.y2), (0, 0, 255), 2)
        cv2.putText(green_core_vis, f"L1 OCR: '{core_text}'",
                    (core_bbox.x, core_bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        Image.fromarray(green_core_vis).save(os.path.join(DEBUG_DIR, "G3_green_core_ocr_bbox.jpg"), quality=90)
        logger.info(f"  绿框 core OCR: '{core_text}' at {core_bbox}")

        # G4: morph 线检测
        vlines, hlines = _detect_morph_lines(gray_g)
        green_morph_vis = green_sub.copy()
        for x, y1m, y2m in vlines:
            cv2.line(green_morph_vis, (x, y1m), (x, y2m), (255, 0, 0), 1)
        for y, x1m, x2m in hlines:
            cv2.line(green_morph_vis, (x1m, y), (x2m, y), (0, 0, 255), 1)
        cv2.rectangle(green_morph_vis, (core_bbox.x, core_bbox.y),
                      (core_bbox.x2, core_bbox.y2), (0, 255, 255), 2)
        Image.fromarray(green_morph_vis).save(os.path.join(DEBUG_DIR, "G4_green_morph_lines.jpg"), quality=90)
        logger.info(f"  绿框 morph 线: {len(vlines)}竖 {len(hlines)}横")

        # G5: cell 边界
        cell = _find_cell_from_lines(core_bbox, vlines, hlines, gw)
        green_cell_vis = green_sub.copy()
        cv2.rectangle(green_cell_vis, (cell.x, cell.y), (cell.x2, cell.y2), (255, 255, 0), 2)
        cv2.putText(green_cell_vis, "Cell", (cell.x, cell.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.rectangle(green_cell_vis, (core_bbox.x, core_bbox.y),
                      (core_bbox.x2, core_bbox.y2), (0, 0, 255), 2)
        cv2.putText(green_cell_vis, "OCR", (core_bbox.x, core_bbox.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        Image.fromarray(green_cell_vis).save(os.path.join(DEBUG_DIR, "G5_green_cell_vs_ocr.jpg"), quality=90)
        logger.info(f"  绿框 Cell: {cell}")

        # G6: 最终绿框
        final_green = _make_green_bbox(core_bbox, cell)
        green_final_vis = green_sub.copy()
        cv2.rectangle(green_final_vis, (cell.x, cell.y), (cell.x2, cell.y2), (255, 255, 0), 1)
        cv2.putText(green_final_vis, "Cell", (cell.x, cell.y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.rectangle(green_final_vis, (core_bbox.x, core_bbox.y),
                      (core_bbox.x2, core_bbox.y2), (0, 0, 255), 1)
        cv2.putText(green_final_vis, "OCR", (core_bbox.x, core_bbox.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.rectangle(green_final_vis, (final_green.x, final_green.y),
                      (final_green.x2, final_green.y2), (0, 255, 0), 3)
        cv2.putText(green_final_vis, f"GREEN: '{core_text}'",
                    (final_green.x, final_green.y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        Image.fromarray(green_final_vis).save(os.path.join(DEBUG_DIR, "G6_green_final_box.jpg"), quality=90)
        logger.info(f"  绿框最终: {final_green}")

        # G7: 把 O6 的 final_green 映射回原图坐标
        mapped_green = _map_bbox_back(final_green, green_search, green_scale)
        green_orig_vis = img_array.copy()
        cv2.rectangle(green_orig_vis, (green_search.x, green_search.y),
                      (green_search.x2, green_search.y2), (0, 200, 0), 2)
        cv2.putText(green_orig_vis, "search area", (green_search.x + 5, green_search.y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.rectangle(green_orig_vis, (mapped_green.x, mapped_green.y),
                      (mapped_green.x2, mapped_green.y2), (0, 255, 0), 4)
        cv2.putText(green_orig_vis, f"GREEN: '{core_text}'",
                    (mapped_green.x, mapped_green.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        pad = 100
        crop_x1 = max(0, green_search.x - pad)
        crop_y1 = max(0, green_search.y - pad)
        green_orig_crop = green_orig_vis[crop_y1:img_h, crop_x1:img_w]
        Image.fromarray(green_orig_crop).save(os.path.join(DEBUG_DIR, "G7_green_on_original.jpg"), quality=90)
    else:
        logger.info("  绿框: core 检测未命中")


# ── 完成 ──
_log_gpu("完成")

logger.info("=" * 60)
logger.info(f"全部调试图已保存到: {DEBUG_DIR}")
logger.info(f"共 {len(os.listdir(DEBUG_DIR))} 个文件")
logger.info(f"GPU 显存峰值: {_gpu_peak_mib:.0f} MiB")
