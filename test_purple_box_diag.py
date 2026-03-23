"""紫框（bottom_left_number）检测诊断脚本

输出：
  1. 原始图纸（旋转后，如有）
  2. 紫框搜索区截图
  3. 搜索区旋转90°CW后的图片（竖排→横排）
  4. 旋转后 OCR 全部结果
  5. 紫框最终识别结果（在子图 + 全图上）

用法:
    python test_purple_box_diag.py
"""

import os, sys, logging, re

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("purple_diag")

import cv2
import numpy as np

from modules.region_detector import (
    BBox, _crop_and_scale, _map_bbox_back,
    _auto_rotate_portrait, _locate_bottom_left_number, _ocr_region,
)
from modules.file_ingestion import load_file
from config import DEFAULT_PREFIXES, make_pattern

TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA070A191P7933-1_0-脱敏.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "purple_box")
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

    rot_code = None
    if img_h > img_w:
        logger.info("portrait detected, rotating CCW...")
        image, rot_code = _auto_rotate_portrait(image)
        img_h, img_w = image.shape[:2]
        logger.info(f"rotated size: {img_w}x{img_h}")
    else:
        logger.info("landscape, no rotation needed")

    save_img("01_image.jpg", image)

    # ── Step 2: purple search area ──
    logger.info("=" * 60)
    logger.info("Step 2: purple search area (image bounds)")
    logger.info("=" * 60)

    green_x = int(img_w * (5.0 / 8.0))
    purple_y = img_h - int(img_h * 1.5 / 6.0)
    purple_search = BBox(0, purple_y, green_x, img_h - purple_y)
    logger.info(f"purple search: {purple_search}")
    logger.info(f"  x: {purple_search.x} ~ {purple_search.x2}")
    logger.info(f"  y: {purple_search.y} ~ {purple_search.y2}")
    logger.info(f"  size: {purple_search.w}x{purple_search.h}")

    # draw on full image
    search_vis = image.copy()
    cv2.rectangle(search_vis, (purple_search.x, purple_search.y),
                  (purple_search.x2, purple_search.y2), (128, 0, 128), 4)
    cv2.putText(search_vis, "purple_search", (purple_search.x + 10, purple_search.y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 3)
    save_img("02_purple_search_area.jpg", search_vis)

    # crop
    purple_crop = image[purple_search.y:purple_search.y2, purple_search.x:purple_search.x2]
    save_img("03_purple_search_crop.jpg", purple_crop)

    # ── Step 3: crop + scale ──
    logger.info("=" * 60)
    logger.info("Step 3: crop + scale")
    logger.info("=" * 60)

    purple_sub, purple_scale = _crop_and_scale(image, purple_search)
    sub_h, sub_w = purple_sub.shape[:2]
    logger.info(f"purple_sub: {sub_w}x{sub_h}, scale={purple_scale:.3f}")
    save_img("04_purple_sub_scaled.jpg", purple_sub)

    # ── Step 4: rotate CW90 (vertical text → horizontal) ──
    logger.info("=" * 60)
    logger.info("Step 4: rotate CW90 for OCR")
    logger.info("=" * 60)

    rotated = cv2.rotate(purple_sub, cv2.ROTATE_90_CLOCKWISE)
    rot_h, rot_w = rotated.shape[:2]
    logger.info(f"rotated: {rot_w}x{rot_h}")
    save_img("05_purple_rotated_cw90.jpg", rotated)

    # ── Step 5: OCR all text on rotated image ──
    logger.info("=" * 60)
    logger.info("Step 5: OCR on rotated image")
    logger.info("=" * 60)

    full_bbox = BBox(0, 0, rot_w, rot_h)
    ocr_results = _ocr_region(rotated, full_bbox, lang="en")
    logger.info(f"OCR found {len(ocr_results)} text items")

    y_re = re.compile(make_pattern())
    ocr_vis = rotated.copy()
    for i, (text, conf, poly) in enumerate(ocr_results):
        logger.info(f"  [{i}] text='{text}' conf={conf:.2f}")
        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            text_up = text.upper().strip()
            is_y = text_up and text_up[0] in {p.upper() for p in DEFAULT_PREFIXES}
            color = (0, 255, 0) if is_y else (255, 165, 0)
            thickness = 3 if is_y else 1

            cv2.rectangle(ocr_vis, (x1, y1), (x2, y2), color, thickness)
            label = f"'{text}' {conf:.1f}"
            cv2.putText(ocr_vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if is_y:
                logger.info(f"    >>> Y-PREFIX MATCH: '{text_up}'")

    save_img("06_ocr_rotated_results.jpg", ocr_vis)

    # ── Step 6: run _locate_bottom_left_number ──
    logger.info("=" * 60)
    logger.info("Step 6: _locate_bottom_left_number()")
    logger.info("=" * 60)

    bl_result = _locate_bottom_left_number(purple_sub, prefixes=DEFAULT_PREFIXES)

    if bl_result:
        bl_text, bl_bbox_sub, bl_ocr = bl_result
        logger.info(f"TEXT: '{bl_text}'")
        logger.info(f"BBOX_SUB: x={bl_bbox_sub.x} y={bl_bbox_sub.y} w={bl_bbox_sub.w} h={bl_bbox_sub.h}")

        # draw on sub image
        result_vis = purple_sub.copy()
        cv2.rectangle(result_vis, (bl_bbox_sub.x, bl_bbox_sub.y),
                      (bl_bbox_sub.x2, bl_bbox_sub.y2), (128, 0, 128), 3)
        save_img("07_purple_result_sub.jpg", result_vis)

        # map back to full image
        bl_bbox = _map_bbox_back(bl_bbox_sub, purple_search, purple_scale)
        logger.info(f"BBOX_FULL: x={bl_bbox.x} y={bl_bbox.y} w={bl_bbox.w} h={bl_bbox.h}")

        # draw on full image
        full_result = image.copy()
        cv2.rectangle(full_result, (bl_bbox.x, bl_bbox.y),
                      (bl_bbox.x2, bl_bbox.y2), (128, 0, 128), 4)
        cv2.putText(full_result, f"PURPLE: '{bl_text}'", (bl_bbox.x, max(bl_bbox.y - 15, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 3)
        save_img("08_purple_result_full.jpg", full_result)

        # crop final
        crop = image[bl_bbox.y:bl_bbox.y2, bl_bbox.x:bl_bbox.x2]
        save_img("09_purple_final_crop.jpg", crop)
    else:
        logger.warning("_locate_bottom_left_number() returned None!")
        save_img("07_purple_result_NONE.jpg", purple_sub)

    logger.info("=" * 60)
    logger.info("done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
