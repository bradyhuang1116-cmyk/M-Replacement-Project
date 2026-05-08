"""调试脚本：测试全部 Y 字头 TIF 文件，仅输出 03/04 图"""
import os
import sys
import logging
import subprocess
import glob as _glob

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, make_pattern
from modules.file_ingestion import load_file
from modules.region_detector import (
    detect_all_regions, draw_regions_debug, _enhance_vertical_lines,
    BBox,
)
from modules.text_replacer import (
    detect_cyan_boxes, detect_row_ys_for_red_box,
)

# ── 输入 ──
_default_dir = r"C:\Users\Brady Huang\Downloads\TIF_Undo"

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else _default_dir

if os.path.isdir(INPUT_PATH):
    INPUT_FILES = sorted([
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
        and f.upper().startswith("Y")
    ])
else:
    INPUT_FILES = [INPUT_PATH]

if not INPUT_FILES:
    logger.error(f"未找到 Y 字头 TIF 文件: {INPUT_PATH}")
    sys.exit(1)

logger.info(f"共 {len(INPUT_FILES)} 个文件待测试: {[os.path.basename(f) for f in INPUT_FILES]}")

DEBUG_BASE = os.path.join(os.path.dirname(__file__), "test_output", "debug_pipeline_hybrid")
os.makedirs(DEBUG_BASE, exist_ok=True)

prefixes = DEFAULT_PREFIXES + ["X"]

from modules.docker_manager import ensure_vlm_ready
logger.info("正在启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 服务启动失败: {msg}")
    sys.exit(1)
logger.info(f"VLM 服务就绪: {msg}")

logger.info("=" * 60)
_log_gpu("启动前")

_total_ok = 0
_total_fail = 0

for _file_idx, INPUT_FILE in enumerate(INPUT_FILES):
    _input_stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    DEBUG_DIR = os.path.join(DEBUG_BASE, _input_stem)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for _f in _glob.glob(os.path.join(DEBUG_DIR, "*.jpg")):
        os.remove(_f)

    logger.info("=" * 60)
    logger.info(f"[{_file_idx+1}/{len(INPUT_FILES)}] {os.path.basename(INPUT_FILE)}")

    try:
        logger.info("Step 1: 加载文件")
        img_array, meta = load_file(INPUT_FILE)
        logger.info(f"  尺寸: {img_array.shape[1]}x{img_array.shape[0]}")

        logger.info("Step 2: 竖线增强")
        enhanced = _enhance_vertical_lines(img_array)

        logger.info("Step 3: 区域检测（红/绿/橙框 + 工厂注意）")
        _log_gpu("区域检测前")
        regions = detect_all_regions(enhanced, prefixes=prefixes)
        _log_gpu("区域检测后(v5)")

        rot_code = regions.get("_metadata", {}).get("rotation")
        if rot_code is not None:
            img_array = cv2.rotate(img_array, rot_code)
            enhanced = cv2.rotate(enhanced, rot_code)
            logger.info(f"  应用旋转: rot_code={rot_code}")

        img_h, img_w = img_array.shape[:2]
        metadata = regions.get("_metadata", {})

        # ── 4. 青框检测 ──
        red_bbox = regions.get("material_code_column")
        cyan_boxes = []
        all_ocr_results = []
        if red_bbox:
            logger.info("Step 4: 青框检测")
            _log_gpu("青框检测前")
            reg_metadata = regions.get("_metadata", {})
            table_search_bbox = reg_metadata.get("table_search_area")
            row_ys = detect_row_ys_for_red_box(
                enhanced, red_bbox, table_search_bbox=table_search_bbox)
            p_chars = "".join(p.upper() for p in (prefixes or ["Y"]))
            red_pattern = (rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1
                           else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b")
            cyan_boxes, cyan_box_data, all_ocr_results = detect_cyan_boxes(
                enhanced, red_bbox, row_ys,
                pattern=red_pattern, prefixes=prefixes,
                return_all_ocr=True,
            )
            _log_gpu("青框检测后")
            logger.info(f"  青框数量: {len(cyan_boxes)}, OCR总识别: {len(all_ocr_results)}")
        else:
            logger.info("Step 4: 跳过（未检测到红框）")

        # ── 03. 综合可视化 ──
        logger.info("生成 03_all_regions")
        combined = img_array.copy()

        region_colors = {
            "material_code_column": ((255, 0, 0), "Red"),
            "bottom_right_number":  ((0, 200, 0), "Green"),
            "top_left_number":      ((255, 165, 0), "Orange"),
        }
        for rname, (color, label) in region_colors.items():
            rb = regions.get(rname)
            if rb and hasattr(rb, "x"):
                cv2.rectangle(combined, (rb.x, rb.y), (rb.x2, rb.y2), color, 3)
                cv2.putText(combined, label, (rb.x + 5, rb.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for cb in cyan_boxes:
            cv2.rectangle(combined, (cb.x, cb.y), (cb.x2, cb.y2), (0, 255, 255), 2)

        Image.fromarray(combined).save(
            os.path.join(DEBUG_DIR, "03_all_regions.jpg"), quality=90)

        # ── 04. OCR 全识别结果可视化 ──
        logger.info("生成 04_ocr_all_detections")
        ocr_vis = img_array.copy()

        if red_bbox:
            cv2.rectangle(ocr_vis, (red_bbox.x, red_bbox.y),
                          (red_bbox.x2, red_bbox.y2), (255, 0, 0), 2)
        for item in all_ocr_results:
            ob = item["abs_bbox"]
            txt = item["text"]
            matched = item["matched"]
            color = (0, 255, 255) if matched else (180, 180, 180)
            cv2.rectangle(ocr_vis, (ob.x, ob.y), (ob.x2, ob.y2), color, 2)
            cv2.putText(ocr_vis, txt, (ob.x + 2, ob.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        gr_bbox = regions.get("bottom_right_number")
        gr_text = metadata.get("bottom_right_text", "")
        if gr_bbox:
            cv2.rectangle(ocr_vis, (gr_bbox.x, gr_bbox.y), (gr_bbox.x2, gr_bbox.y2), (0, 200, 0), 2)
            cv2.putText(ocr_vis, gr_text, (gr_bbox.x + 5, gr_bbox.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        or_bbox = regions.get("top_left_number")
        or_text = metadata.get("top_left_text", "")
        if or_bbox:
            cv2.rectangle(ocr_vis, (or_bbox.x, or_bbox.y), (or_bbox.x2, or_bbox.y2), (255, 165, 0), 2)
            cv2.putText(ocr_vis, or_text, (or_bbox.x + 5, or_bbox.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        Image.fromarray(ocr_vis).save(
            os.path.join(DEBUG_DIR, "04_ocr_all_detections.jpg"), quality=90)

        _total_ok += 1
        logger.info(f"  OK")

    except Exception as e:
        _total_fail += 1
        logger.error(f"  FAIL: {e}", exc_info=True)

_log_gpu("完成")
logger.info("=" * 60)
logger.info(f"测试完成: {_total_ok} 成功, {_total_fail} 失败, 共 {len(INPUT_FILES)} 个文件")
logger.info(f"调试图保存到: {DEBUG_BASE}")
logger.info(f"GPU 显存峰值: {_gpu_peak_mib:.0f} MiB")
