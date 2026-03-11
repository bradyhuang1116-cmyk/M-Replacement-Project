"""诊断脚本：输出原图、检测预览图、详细日志，供第三方分析青色框问题。"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("diagnostic_output/diagnostic.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("diagnostic")

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from config import DEFAULT_PREFIXES, make_pattern
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, draw_regions_debug, _detect_horizontal_lines, _detect_horizontal_lines_adaptive, BBox
from modules.text_replacer import (
    detect_cyan_boxes,
    replace_y_in_region_pixel,
    _get_ocr,
    _parse_ocr_results,
    _determine_uniform_cell_height,
    _find_cell,
    _uniform_row_ys,
)
from config import OCR_LANG_EN

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "diagnostic_output"
OUT_DIR.mkdir(exist_ok=True)

# 指定测试文件
test_file = Path(r"C:\Users\HYMOD\WPSDrive\1435755195\WPS云盘\客户\上海，三菱电机\TEST_undo\YA147C070P3335_0-脱敏.tif")
if not test_file.exists():
    logger.error(f"未找到测试文件: {test_file}")
    sys.exit(1)
logger.info(f"="*80)
logger.info(f"测试文件: {test_file.name}")
logger.info(f"="*80)

# ── 1. 加载原图 ──
img_array, _ = load_file(str(test_file))
logger.info(f"原图尺寸: {img_array.shape}")
cv2.imwrite(str(OUT_DIR / "01_original.png"), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
logger.info(f"已保存: 01_original.png")

# ── 2. 检测所有区域 ──
prefixes = ["X", "B", "Y"]
regions = detect_all_regions(img_array, prefixes=prefixes)

# 如果检测时旋转了图像，同步旋转
rot_code = regions.get("_metadata", {}).get("rotation")
if rot_code is not None:
    img_array = cv2.rotate(img_array, rot_code)
    logger.info(f"旋转后尺寸: {img_array.shape}")
    cv2.imwrite(str(OUT_DIR / "02_rotated.png"), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

# 输出区域信息
logger.info(f"\n{'='*60}")
logger.info(f"检测到的区域:")
for name, bbox in regions.items():
    if name.startswith("_"):
        continue
    logger.info(f"  {name}: {bbox}")
logger.info(f"{'='*60}")

# ── 3. 红框详细分析 ──
red_bbox = regions.get("material_code_column")
metadata = regions.get("_metadata", {})

if red_bbox:
    logger.info(f"\n{'='*60}")
    logger.info(f"红框详细分析")
    logger.info(f"红框坐标: {red_bbox}")
    logger.info(f"{'='*60}")

    # 保存红框裁剪区域
    red_roi = red_bbox.crop(img_array)
    cv2.imwrite(str(OUT_DIR / "03_red_bbox_crop.png"), cv2.cvtColor(red_roi, cv2.COLOR_RGB2BGR))
    logger.info(f"已保存: 03_red_bbox_crop.png (红框裁剪)")

    # ── 3a. 水平线检测 ──
    table_search_bbox = metadata.get("table_search_area")
    if table_search_bbox:
        logger.info(f"\n表格搜索区域: {table_search_bbox}")
        table_roi = table_search_bbox.crop(img_array)
        cv2.imwrite(str(OUT_DIR / "04_table_search_area.png"), cv2.cvtColor(table_roi, cv2.COLOR_RGB2BGR))

        table_gray = cv2.cvtColor(table_roi, cv2.COLOR_RGB2GRAY)
        bbox_y_in_table = red_bbox.y - table_search_bbox.y

        # 不带 min_width_ratio 的原始检测
        all_h_lines_raw = _detect_horizontal_lines(table_gray, min_line_width=18, min_width_ratio=0.0)
        logger.info(f"\n水平线检测（无过滤）: {len(all_h_lines_raw)} 条")
        for i, y in enumerate(all_h_lines_raw):
            logger.info(f"  [{i}] y={y}")

        # 动态双阈值策略检测
        all_h_lines_filtered = _detect_horizontal_lines_adaptive(
            table_gray, bbox_y_in_table, red_bbox.h, min_line_width=18)
        logger.info(f"\n水平线检测（自适应）: {len(all_h_lines_filtered)} 条")
        for i, y in enumerate(all_h_lines_filtered):
            logger.info(f"  [{i}] y={y}")

        # 在表格搜索区域上绘制所有检测到的水平线
        debug_table = table_roi.copy()
        for y in all_h_lines_raw:
            cv2.line(debug_table, (0, y), (table_roi.shape[1], y), (255, 0, 0), 1)  # 红色=原始
        for y in all_h_lines_filtered:
            cv2.line(debug_table, (0, y), (table_roi.shape[1], y), (0, 255, 0), 2)  # 绿色=过滤后
        cv2.imwrite(str(OUT_DIR / "05_horizontal_lines_debug.png"), cv2.cvtColor(debug_table, cv2.COLOR_RGB2BGR))
        logger.info(f"已保存: 05_horizontal_lines_debug.png (红=原始, 绿=过滤后)")

        # 映射到红框内的 row_ys
        bbox_y_in_table = red_bbox.y - table_search_bbox.y
        row_ys_raw = [
            y - bbox_y_in_table
            for y in all_h_lines_raw
            if bbox_y_in_table <= y <= bbox_y_in_table + red_bbox.h
        ]
        row_ys_filtered = [
            y - bbox_y_in_table
            for y in all_h_lines_filtered
            if bbox_y_in_table <= y <= bbox_y_in_table + red_bbox.h
        ]
        logger.info(f"\n红框内 row_ys（无过滤）: {len(row_ys_raw)} 条 → {row_ys_raw}")
        logger.info(f"红框内 row_ys（过滤后）: {len(row_ys_filtered)} 条 → {row_ys_filtered}")

        # 在红框裁剪图上绘制 row_ys
        debug_red = red_roi.copy()
        for y in row_ys_raw:
            cv2.line(debug_red, (0, y), (red_roi.shape[1], y), (255, 0, 0), 1)
        for y in row_ys_filtered:
            cv2.line(debug_red, (0, y), (red_roi.shape[1], y), (0, 255, 0), 2)
        cv2.imwrite(str(OUT_DIR / "06_red_bbox_row_ys.png"), cv2.cvtColor(debug_red, cv2.COLOR_RGB2BGR))
        logger.info(f"已保存: 06_red_bbox_row_ys.png (红=原始row_ys, 绿=过滤后row_ys)")

        # ── 3b. OCR 识别 ──
        row_ys = row_ys_filtered
        logger.info(f"\n{'='*60}")
        logger.info(f"OCR 识别（红框内）")
        logger.info(f"{'='*60}")

        # 缩放参数（复制 replace_y_in_region_pixel 逻辑）
        roi_h, roi_w = red_roi.shape[:2]
        min_dim = min(roi_w, roi_h)
        max_dim = max(roi_w, roi_h)
        if min_dim < 80:
            scale_factor = 5.0
        elif min_dim < 200:
            scale_factor = 3.0
        else:
            scale_factor = 1.0
        if max_dim * scale_factor > 3500:
            scale_factor = max(3500.0 / max_dim, 1.0)

        logger.info(f"红框ROI尺寸: {roi_w}x{roi_h}, scale_factor={scale_factor:.2f}")

        if scale_factor > 1.0:
            new_h = int(roi_h * scale_factor)
            new_w = int(roi_w * scale_factor)
            roi_scaled = cv2.resize(red_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            roi_scaled = red_roi.copy()

        pad_px = 0
        if roi_h < 100 and roi_w < 400:
            pad_px = max(int(min(roi_scaled.shape[:2]) * 0.35), 40)
            padded = np.full(
                (roi_scaled.shape[0] + 2 * pad_px, roi_scaled.shape[1] + 2 * pad_px, 3),
                255, dtype=np.uint8,
            )
            padded[pad_px:pad_px + roi_scaled.shape[0],
                   pad_px:pad_px + roi_scaled.shape[1]] = roi_scaled
            sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            roi_scaled = cv2.filter2D(padded, -1, sharpen)

        logger.info(f"缩放后尺寸: {roi_scaled.shape[1]}x{roi_scaled.shape[0]}, pad_px={pad_px}")
        cv2.imwrite(str(OUT_DIR / "07_red_bbox_scaled.png"), cv2.cvtColor(roi_scaled, cv2.COLOR_RGB2BGR))
        logger.info(f"已保存: 07_red_bbox_scaled.png (OCR输入图)")

        # OCR
        ocr = _get_ocr(OCR_LANG_EN)
        result = ocr.predict(roi_scaled)
        items = _parse_ocr_results(result)

        logger.info(f"\nOCR 识别结果: {len(items)} 项")
        p_chars = "".join(p.upper() for p in prefixes)
        import re
        red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
        pat = re.compile(red_pattern)

        matched_count = 0
        unmatched_items = []
        for i, (poly, text, score) in enumerate(items):
            pts = np.array(poly, dtype=np.float32)
            bx = int(pts[:, 0].min())
            by = int(pts[:, 1].min())
            bw = int(pts[:, 0].max() - bx)
            bh = int(pts[:, 1].max() - by)
            text_nospace = text.replace(" ", "")
            is_match = bool(pat.match(text_nospace))
            status = "✓ MATCH" if is_match else "✗ miss"
            if is_match:
                matched_count += 1
            else:
                unmatched_items.append((text, score))
            logger.info(f"  [{i:2d}] {status} | '{text}' → '{text_nospace}' | score={score:.3f} | bbox=({bx},{by},{bw},{bh})")

        logger.info(f"\n匹配: {matched_count}/{len(items)} 项")
        logger.info(f"未匹配项:")
        for text, score in unmatched_items:
            # 分析为什么没匹配
            text_ns = text.replace(" ", "")
            reason = []
            if len(text_ns) < 9:
                reason.append(f"长度不足({len(text_ns)}<9)")
            elif len(text_ns) > 9:
                reason.append(f"长度过长({len(text_ns)}>9)")
            if text_ns and text_ns[0].upper() not in p_chars:
                reason.append(f"首字母'{text_ns[0]}'不在{p_chars}")
            if not re.match(r'^[A-Z0-9\-]+$', text_ns.upper()):
                reason.append(f"含非法字符")
            logger.info(f"  '{text}' → 原因: {', '.join(reason) if reason else '正则不匹配'}")

        # ── 3c. 统一单元格高度 ──
        if len(row_ys) >= 2:
            row_ys_uniform = _uniform_row_ys(row_ys) if len(row_ys) >= 3 else row_ys
            uniform_h = _determine_uniform_cell_height(row_ys_uniform)
            logger.info(f"\n统一单元格高度: {uniform_h}px")
            logger.info(f"row_ys (uniform后): {row_ys_uniform}")

        # ── 3d. 青色框生成 ──
        cyan_boxes = detect_cyan_boxes(
            img_array, red_bbox, row_ys,
            pattern=red_pattern, prefixes=prefixes,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"青色框生成结果: {len(cyan_boxes)} 个")
        for i, cb in enumerate(cyan_boxes):
            logger.info(f"  [{i}] {cb}")
        logger.info(f"{'='*60}")

        # 在红框裁剪图上绘制青色框
        debug_cyan = red_roi.copy()
        for y in row_ys_filtered:
            cv2.line(debug_cyan, (0, y), (red_roi.shape[1], y), (0, 200, 0), 1)
        for cb in cyan_boxes:
            # 转换为红框内坐标
            cx = cb.x - red_bbox.x
            cy = cb.y - red_bbox.y
            cv2.rectangle(debug_cyan, (cx, cy), (cx + cb.w, cy + cb.h), (0, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / "08_cyan_boxes_on_red.png"), cv2.cvtColor(debug_cyan, cv2.COLOR_RGB2BGR))
        logger.info(f"已保存: 08_cyan_boxes_on_red.png")

    # ── 4. 全图检测预览 ──
    metadata["cyan_boxes"] = cyan_boxes if red_bbox else []
    debug_img = draw_regions_debug(img_array, regions)
    cv2.imwrite(str(OUT_DIR / "09_detection_preview.png"), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    logger.info(f"\n已保存: 09_detection_preview.png (全图检测预览)")

else:
    logger.error("未检测到红框 (material_code_column)")

# ── 5. 输出摘要 JSON ──
summary = {
    "test_file": test_file.name,
    "image_size": list(img_array.shape),
    "prefixes": prefixes,
    "rotation": rot_code,
    "regions": {},
}
for name, bbox in regions.items():
    if name.startswith("_"):
        continue
    summary["regions"][name] = bbox.to_dict() if bbox else None

if red_bbox and table_search_bbox:
    summary["horizontal_lines"] = {
        "raw_count": len(all_h_lines_raw),
        "filtered_count": len(all_h_lines_filtered),
        "row_ys_in_red_raw": row_ys_raw,
        "row_ys_in_red_filtered": row_ys_filtered,
    }
    summary["ocr"] = {
        "total_items": len(items),
        "matched_items": matched_count,
        "pattern": red_pattern,
        "scale_factor": scale_factor,
        "pad_px": pad_px,
    }
    summary["cyan_boxes"] = {
        "count": len(cyan_boxes),
        "uniform_cell_h": uniform_h if len(row_ys) >= 2 else None,
        "boxes": [cb.to_dict() for cb in cyan_boxes],
    }

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info(f"已保存: summary.json")

logger.info(f"\n{'='*80}")
logger.info(f"诊断完成！所有文件已保存到: {OUT_DIR}")
logger.info(f"{'='*80}")
