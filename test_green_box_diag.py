"""绿框（bottom_right_number）检测诊断脚本

输出：
  1. 原始图纸（旋转后，如有）
  2. 绿框搜索区截图
  3. 搜索区 OCR 全部结果（标注在图上）
  4. 绿框最终识别结果（bbox + 文字）
  5. 竖线约束可视化
  6. 最终绿框在全图上的位置

用法:
    python test_green_box_diag.py
"""

import os
import sys
import logging
import re

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("green_diag")

import cv2
import numpy as np

from modules.region_detector import (
    BBox, _crop_and_scale, _map_bbox_back,
    _auto_rotate_portrait,
    _locate_bottom_right_number, _ocr_region,
    detect_all_regions, draw_regions_debug,
)
from modules.file_ingestion import load_file
from config import DEFAULT_PREFIXES, make_pattern

# ── 配置 ──
TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA246C928_H.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "green_box")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_img(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  saved: {name} ({img.shape[1]}x{img.shape[0]})")


def main():
    logger.info(f"test image: {TEST_IMAGE}")
    logger.info(f"output dir: {OUTPUT_DIR}")

    if not os.path.isfile(TEST_IMAGE):
        logger.error(f"file not found: {TEST_IMAGE}")
        sys.exit(1)

    # ── Step 1: load + rotate ──
    logger.info("=" * 60)
    logger.info("Step 1: load image + auto rotate")
    logger.info("=" * 60)

    image, _ = load_file(TEST_IMAGE)
    img_h, img_w = image.shape[:2]
    logger.info(f"original size: {img_w}x{img_h}")
    save_img("01_original.jpg", image)

    rot_code = None
    if img_h > img_w:
        logger.info("portrait detected, rotating CCW...")
        image, rot_code = _auto_rotate_portrait(image)
        img_h, img_w = image.shape[:2]
        logger.info(f"rotated size: {img_w}x{img_h}, rot_code={rot_code}")
        save_img("02_rotated.jpg", image)
    else:
        logger.info("landscape, no rotation needed")
        save_img("02_rotated.jpg", image)

    # ── Step 2: green search area (based on image bounds, not frame) ──
    logger.info("=" * 60)
    logger.info("Step 2: green search area (image bounds)")
    logger.info("=" * 60)

    green_x = int(img_w * (5.0 / 8.0))
    green_y = int(img_h * (5.0 / 6.0))
    green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)
    logger.info(f"green search: {green_search}")
    logger.info(f"  x: {green_search.x} ~ {green_search.x2}")
    logger.info(f"  y: {green_search.y} ~ {green_search.y2}")
    logger.info(f"  size: {green_search.w}x{green_search.h}")

    # draw green search area on full image
    search_vis = image.copy()
    cv2.rectangle(search_vis, (green_search.x, green_search.y),
                  (green_search.x2, green_search.y2), (0, 255, 0), 4)
    cv2.putText(search_vis, "green_search", (green_search.x + 10, green_search.y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    save_img("03_green_search_area.jpg", search_vis)

    # crop green search area
    green_crop = image[green_search.y:green_search.y2, green_search.x:green_search.x2]
    save_img("04_green_search_crop.jpg", green_crop)

    # ── Step 3: crop + scale ──
    logger.info("=" * 60)
    logger.info("Step 3: crop + scale for OCR")
    logger.info("=" * 60)

    green_sub, green_scale = _crop_and_scale(image, green_search)
    sub_h, sub_w = green_sub.shape[:2]
    logger.info(f"green_sub: {sub_w}x{sub_h}, scale={green_scale:.3f}")
    save_img("05_green_sub_scaled.jpg", green_sub)

    # ── Step 4: OCR all text in green search area ──
    logger.info("=" * 60)
    logger.info("Step 4: OCR all text in green search area")
    logger.info("=" * 60)

    search_full = BBox(0, 0, sub_w, sub_h)
    ocr_results = _ocr_region(green_sub, search_full)
    logger.info(f"OCR found {len(ocr_results)} text items")

    ocr_vis = green_sub.copy()
    y_re = re.compile(make_pattern())
    for i, (text, conf, poly) in enumerate(ocr_results):
        logger.info(f"  [{i}] text='{text}' conf={conf:.2f} poly={poly}")

        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            # check if matches Y pattern
            text_clean = text.upper().replace(" ", "")
            is_match = bool(y_re.search(text_clean))
            color = (0, 255, 0) if is_match else (255, 165, 0)
            thickness = 3 if is_match else 1

            cv2.rectangle(ocr_vis, (x1, y1), (x2, y2), color, thickness)
            label = f"'{text}' {conf:.1f}"
            cv2.putText(ocr_vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if is_match:
                logger.info(f"    >>> MATCH: '{text_clean}'")

    save_img("06_ocr_all_results.jpg", ocr_vis)

    # ── Step 5: run _locate_bottom_right_number ──
    logger.info("=" * 60)
    logger.info("Step 5: _locate_bottom_right_number()")
    logger.info("=" * 60)

    br_result = _locate_bottom_right_number(green_sub, prefixes=DEFAULT_PREFIXES)

    if br_result:
        br_text, br_bbox_sub = br_result
        logger.info(f"result text: '{br_text}'")
        logger.info(f"result bbox (sub): {br_bbox_sub}")

        # draw on sub image
        result_vis = green_sub.copy()
        cv2.rectangle(result_vis, (br_bbox_sub.x, br_bbox_sub.y),
                      (br_bbox_sub.x2, br_bbox_sub.y2), (0, 255, 0), 3)
        cv2.putText(result_vis, f"'{br_text}'", (br_bbox_sub.x, max(br_bbox_sub.y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        save_img("07_green_result_sub.jpg", result_vis)

        # map back to full image
        br_bbox = _map_bbox_back(br_bbox_sub, green_search, green_scale)
        logger.info(f"result bbox (full): {br_bbox}")

        # draw on full image
        full_result = image.copy()
        cv2.rectangle(full_result, (br_bbox.x, br_bbox.y),
                      (br_bbox.x2, br_bbox.y2), (0, 255, 0), 4)
        cv2.putText(full_result, f"GREEN: '{br_text}'", (br_bbox.x, max(br_bbox.y - 15, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        save_img("08_green_result_full.jpg", full_result)

        # crop the final green box region for inspection
        green_final_crop = image[br_bbox.y:br_bbox.y2, br_bbox.x:br_bbox.x2]
        save_img("09_green_final_crop.jpg", green_final_crop)
    else:
        logger.warning("_locate_bottom_right_number() returned None!")
        save_img("07_green_result_NONE.jpg", green_sub)

    # ── Step 6: vertical line analysis ──
    logger.info("=" * 60)
    logger.info("Step 6: vertical line analysis in green search area")
    logger.info("=" * 60)

    gray_roi = cv2.cvtColor(green_sub, cv2.COLOR_RGB2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_img("10_green_otsu.jpg", thresh_roi)

    # detect vertical lines
    min_vh = max(sub_h // 10, 10)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_mask = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, v_kernel)
    save_img("11_green_vlines_mask.jpg", v_mask)

    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"vertical line contours: {len(v_contours)}")

    vline_vis = green_sub.copy()
    for c in v_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(vline_vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(vline_vis, f"h={bh}", (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        logger.info(f"  vline: x={x}, y={y}, w={bw}, h={bh}")
    save_img("12_green_vlines_annotated.jpg", vline_vis)

    # ── Step 7: full detect_all_regions for comparison ──
    logger.info("=" * 60)
    logger.info("Step 7: full detect_all_regions() for comparison")
    logger.info("=" * 60)

    # reload image for fresh run
    image_fresh, _ = load_file(TEST_IMAGE)
    regions = detect_all_regions(image_fresh, prefixes=DEFAULT_PREFIXES)

    for name, val in regions.items():
        if name == "_metadata":
            logger.info(f"  metadata: {val}")
        elif val is not None:
            logger.info(f"  {name}: {val}")
        else:
            logger.info(f"  {name}: None")

    # draw all regions
    rot_code_full = regions.get("_metadata", {}).get("rotation")
    if rot_code_full is not None:
        image_fresh, _ = load_file(TEST_IMAGE)
        img_h_f, img_w_f = image_fresh.shape[:2]
        if img_h_f > img_w_f:
            image_fresh = cv2.rotate(image_fresh, rot_code_full)
    debug_img = draw_regions_debug(image_fresh, regions)
    save_img("13_all_regions_debug.jpg", debug_img)

    logger.info("=" * 60)
    logger.info("done! check output dir for images.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
