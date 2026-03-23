"""OCR 文字识别诊断：对指定文件的各搜索区进行 OCR，输出识别框可视化图。"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.getLogger("ppocr").setLevel(logging.WARNING)

OUT_DIR = Path(__file__).parent / "diagnostic_output" / "ocr_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(OUT_DIR / "ocr_diagnostic.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ocr_diag")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import DEFAULT_PREFIXES, OCR_LANG_EN, OCR_LANG_CH
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, BBox
from modules.text_replacer import _get_ocr, _parse_ocr_results, detect_row_ys_for_red_box, detect_cyan_boxes

# ── 配置 ──
TIF_DIR = Path(r"c:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo")

TEST_FILES = [
    
    TIF_DIR / "YA026D941_0d.tif"
]

PREFIXES = ["X", "Y", "Z", "B"]


def draw_ocr_boxes(img: np.ndarray, items: list, title: str = "") -> np.ndarray:
    """在图片上绘制 OCR 识别框和文字。返回标注后的图片副本。"""
    vis = img.copy()

    for i, (poly, text, score) in enumerate(items):
        pts = np.array(poly, dtype=np.int32)
        # 绘制多边形框
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # 文字标注位置
        x_min = int(pts[:, 0].min())
        y_min = int(pts[:, 1].min())

        # 背景色块
        label = f"[{i}] {text} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), (0, 0, 255), -1)
        cv2.putText(vis, label, (x_min + 2, y_min - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 标题
    if title:
        cv2.putText(vis, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    return vis


def run_ocr_on_region(img_array: np.ndarray, bbox: BBox, lang: str = "en") -> list:
    """对指定区域执行 OCR，返回 items 列表 (poly, text, score)。
    poly 坐标为裁剪区域内的局部坐标。"""
    roi = bbox.crop(img_array)
    ocr = _get_ocr(lang)
    result = ocr.predict(roi)
    items = _parse_ocr_results(result)
    return items, roi


def process_one_file(filepath: Path, file_idx: int):
    """处理单个文件：检测区域 → 对各搜索区做 OCR → 输出识别框图。"""
    short_name = filepath.stem
    file_out = OUT_DIR / short_name
    file_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"[{file_idx}] 处理文件: {filepath.name}")
    logger.info(f"{'='*80}")

    # 1. 加载
    img_array, _ = load_file(str(filepath))
    logger.info(f"  图片尺寸: {img_array.shape[1]}x{img_array.shape[0]}")

    # 2. 检测区域
    regions = detect_all_regions(img_array, prefixes=PREFIXES)
    rot_code = regions.get("_metadata", {}).get("rotation")
    if rot_code is not None:
        img_array = cv2.rotate(img_array, rot_code)
        logger.info(f"  旋转后尺寸: {img_array.shape[1]}x{img_array.shape[0]}")

    img_h, img_w = img_array.shape[:2]
    metadata = regions.get("_metadata", {})

    # ── 定义搜索区（与 detect_all_regions 一致）──
    search_areas = {
        "red_search": BBox(0, 0, img_w // 2, img_h),
        "green_search": BBox(
            int(img_w * 5.0 / 8.0),
            int(img_h * 5.0 / 6.0),
            img_w - int(img_w * 5.0 / 8.0),
            img_h - int(img_h * 5.0 / 6.0),
        ),
        "purple_search": BBox(
            0,
            img_h - int(img_h * 1.5 / 6.0),
            int(img_w * 5.0 / 8.0),
            int(img_h * 1.5 / 6.0),
        ),
        "orange_search": BBox(
            0, 0,
            int(img_w * 2.0 / 8.0),
            int(img_h * 2.0 / 12.0) - int(int(img_h * 2.0 / 12.0) / 3),
        ),
    }

    # 已检测到的精确区域框
    detected_regions = {
        "material_code_column (红框)": regions.get("material_code_column"),
        "bottom_right_number (绿框)": regions.get("bottom_right_number"),
        "top_left_number (橙框)": regions.get("top_left_number"),
        "bottom_left_number (紫框)": regions.get("bottom_left_number"),
        "annotations (蓝框)": regions.get("annotations"),
    }

    logger.info(f"  检测到的区域:")
    for name, bbox in detected_regions.items():
        logger.info(f"    {name}: {bbox}")

    # ── 3. 对各搜索区域做 OCR 并输出识别框 ──

    for area_name, area_bbox in search_areas.items():
        logger.info(f"\n  ── OCR: {area_name} ({area_bbox}) ──")

        roi = area_bbox.crop(img_array)

        # 对搜索区做 OCR
        ocr = _get_ocr(OCR_LANG_EN)
        result = ocr.predict(roi)
        items_en = _parse_ocr_results(result)
        logger.info(f"    EN OCR: {len(items_en)} 项")

        for i, (poly, text, score) in enumerate(items_en):
            pts = np.array(poly, dtype=np.float32)
            bx, by = int(pts[:, 0].min()), int(pts[:, 1].min())
            bw, bh = int(pts[:, 0].max() - bx), int(pts[:, 1].max() - by)
            logger.info(f"      [{i:2d}] '{text}' score={score:.3f} bbox=({bx},{by},{bw},{bh})")

        # 绘制识别框
        vis = draw_ocr_boxes(roi, items_en, title=f"{area_name} - EN OCR")
        out_path = file_out / f"{area_name}_en_ocr.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        logger.info(f"    已保存: {out_path.name}")

    # ── 4. 对已检测到的精确区域做 OCR ──

    for region_name, bbox in detected_regions.items():
        if bbox is None:
            continue

        safe_name = region_name.split(" ")[0]
        logger.info(f"\n  ── OCR 精确区域: {region_name} ({bbox}) ──")

        roi = bbox.crop(img_array)
        lang = OCR_LANG_EN
        # 紫框用中文OCR（可能含竖排文字）
        if "bottom_left" in region_name:
            lang = OCR_LANG_CH

        ocr = _get_ocr(lang)
        result = ocr.predict(roi)
        items = _parse_ocr_results(result)
        logger.info(f"    {lang.upper()} OCR: {len(items)} 项")

        for i, (poly, text, score) in enumerate(items):
            pts = np.array(poly, dtype=np.float32)
            bx, by = int(pts[:, 0].min()), int(pts[:, 1].min())
            bw, bh = int(pts[:, 0].max() - bx), int(pts[:, 1].max() - by)
            logger.info(f"      [{i:2d}] '{text}' score={score:.3f} bbox=({bx},{by},{bw},{bh})")

        vis = draw_ocr_boxes(roi, items, title=f"{safe_name} - {lang.upper()} OCR")
        out_path = file_out / f"{safe_name}_precise_ocr.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        logger.info(f"    已保存: {out_path.name}")

    # ── 5. 输出带检测框的区域截图（橙/绿/红+青色框）──

    # 橙框截图（带 OCR 识别框）
    orange_bbox = regions.get("top_left_number")
    if orange_bbox:
        logger.info(f"\n  ── 橙框带框截图 ──")
        orange_roi = orange_bbox.crop(img_array).copy()
        ocr = _get_ocr(OCR_LANG_EN)
        result = ocr.predict(orange_roi)
        items = _parse_ocr_results(result)
        vis = draw_ocr_boxes(orange_roi, items, title="orange")
        # 在截图上标注检测到的文本
        orange_text = metadata.get("top_left_text", "")
        cv2.putText(vis, f"detected: {orange_text}", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)
        out_path = file_out / "orange_bbox_ocr.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        logger.info(f"    橙框 bbox={orange_bbox}, text='{orange_text}'")
        logger.info(f"    已保存: {out_path.name}")

    # 绿框截图（带 OCR 识别框）
    green_bbox = regions.get("bottom_right_number")
    if green_bbox:
        logger.info(f"\n  ── 绿框带框截图 ──")
        green_roi = green_bbox.crop(img_array).copy()
        ocr = _get_ocr(OCR_LANG_EN)
        result = ocr.predict(green_roi)
        items = _parse_ocr_results(result)
        vis = draw_ocr_boxes(green_roi, items, title="green")
        green_text = metadata.get("bottom_right_text", "")
        cv2.putText(vis, f"detected: {green_text}", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)
        out_path = file_out / "green_bbox_ocr.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        logger.info(f"    绿框 bbox={green_bbox}, text='{green_text}'")
        logger.info(f"    已保存: {out_path.name}")

    # ── 6. 红框：行线 + 青色框 OCR + 各青色框带框截图 ──
    red_bbox = regions.get("material_code_column")
    if red_bbox:
        logger.info(f"\n  ── 红框行线 + 青色框 OCR ──")

        table_search_bbox = metadata.get("table_search_area")
        row_ys = detect_row_ys_for_red_box(img_array, red_bbox, table_search_bbox=table_search_bbox)
        logger.info(f"    row_ys: {len(row_ys)} 条")

        p_chars = "".join(p.upper() for p in PREFIXES)
        red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b"
        cyan_boxes, cyan_box_data = detect_cyan_boxes(
            img_array, red_bbox, row_ys,
            pattern=red_pattern, prefixes=PREFIXES,
        )
        logger.info(f"    青色框: {len(cyan_boxes)} 个")

        # 绘制红框区域 + 行线 + 青色框 + OCR 识别框
        red_roi = red_bbox.crop(img_array).copy()
        for y in row_ys:
            cv2.line(red_roi, (0, y), (red_roi.shape[1], y), (0, 200, 0), 1)
        for cb in cyan_boxes:
            cx = cb.x - red_bbox.x
            cy = cb.y - red_bbox.y
            cv2.rectangle(red_roi, (cx, cy), (cx + cb.w, cy + cb.h), (0, 255, 255), 2)

        # 标注青色框内的文字
        if cyan_box_data:
            for cbd in cyan_box_data:
                cb_bbox = cbd["bbox"]
                text = cbd["text"]
                cx = cb_bbox.x - red_bbox.x
                cy = cb_bbox.y - red_bbox.y
                cv2.putText(red_roi, text, (cx + 2, cy + cb_bbox.h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1, cv2.LINE_AA)

        out_path = file_out / "red_bbox_lines_cyan.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(red_roi, cv2.COLOR_RGB2BGR))
        logger.info(f"    已保存: {out_path.name}")

        # 各青色框单独带 OCR 识别框截图
        if cyan_box_data:
            for ci, cbd in enumerate(cyan_box_data):
                cb_bbox = cbd["bbox"]
                cb_roi = cb_bbox.crop(img_array).copy()
                ocr = _get_ocr(OCR_LANG_EN)
                result = ocr.predict(cb_roi)
                items = _parse_ocr_results(result)
                vis = draw_ocr_boxes(cb_roi, items)
                cb_text = cbd.get("text", "")
                cv2.putText(vis, f"[{ci}] {cb_text}", (2, vis.shape[0] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
                out_path = file_out / f"cyan_{ci:02d}_{cb_text}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info(f"    已保存 {len(cyan_box_data)} 个青色框截图")

    logger.info(f"\n  文件 {short_name} 诊断完成 → {file_out}")


# ── 主流程 ──
if __name__ == "__main__":
    logger.info(f"OCR 诊断脚本启动")
    logger.info(f"输出目录: {OUT_DIR}")
    logger.info(f"目标文件: {[f.name for f in TEST_FILES]}")

    for i, f in enumerate(TEST_FILES):
        if not f.exists():
            logger.error(f"文件不存在: {f}")
            continue
        process_one_file(f, i + 1)

    logger.info(f"\n{'='*80}")
    logger.info(f"全部诊断完成！输出目录: {OUT_DIR}")
    logger.info(f"{'='*80}")
