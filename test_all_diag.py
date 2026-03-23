"""整体图框诊断脚本

对一张图纸执行全部检测（红/绿/紫/橙/青框），逐步输出每个搜索区和识别结果。

用法:
    python test_all_diag.py
"""

import os, sys, logging, time

os.environ["FLAGS_use_mkldnn"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("all_diag")

import cv2
import numpy as np

from modules.region_detector import (
    BBox, _crop_and_scale, _map_bbox_back, _auto_rotate_portrait,
    _locate_material_code_column, _locate_bottom_right_number,
    _locate_top_left_number, _locate_bottom_left_number,
)
from modules.text_replacer import detect_row_ys_for_red_box, detect_cyan_boxes
from modules.file_ingestion import load_file
from config import DEFAULT_PREFIXES, set_ocr_mode

TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA057C857_0-脱敏.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "all_boxes")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_img(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  saved: {name} ({img.shape[1]}x{img.shape[0]})")


def main():
    set_ocr_mode(device="cpu", model_type="server")
    total_start = time.time()
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

    if img_h > img_w:
        logger.info("portrait detected, rotating CCW...")
        image, _ = _auto_rotate_portrait(image)
        img_h, img_w = image.shape[:2]
        logger.info(f"rotated size: {img_w}x{img_h}")

    save_img("01_image.jpg", image)

    # ── Step 2: compute search areas ──
    logger.info("=" * 60)
    logger.info("Step 2: search areas (image bounds)")
    logger.info("=" * 60)

    red_search = BBox(0, 0, img_w // 2, img_h)
    green_x = int(img_w * (5.0 / 8.0))
    green_y = int(img_h * (5.0 / 6.0))
    green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)
    purple_y = img_h - int(img_h * 1.5 / 6.0)
    purple_search = BBox(0, purple_y, green_x, img_h - purple_y)
    orange_h_full = int(img_h * (2.0 / 12.0))
    orange_search = BBox(0, 0, int(img_w * (2.0 / 8.0)),
                         orange_h_full - int(orange_h_full / 3))

    areas = {"red": red_search, "green": green_search,
             "purple": purple_search, "orange": orange_search}
    for name, bbox in areas.items():
        logger.info(f"  {name}: {bbox}")

    vis = image.copy()
    colors_search = {"red": (255,0,0), "green": (0,255,0),
                     "purple": (128,0,128), "orange": (255,165,0)}
    for name, bbox in areas.items():
        c = colors_search[name]
        cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), c, 4)
        cv2.putText(vis, name, (bbox.x+10, bbox.y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
    save_img("02_search_areas.jpg", vis)

    # ── Step 3: serial detection ──
    logger.info("=" * 60)
    logger.info("Step 3: detection (serial)")
    logger.info("=" * 60)

    detect_start = time.time()
    colors = {"red": (255,0,0), "green": (0,255,0),
              "orange": (255,165,0), "purple": (128,0,128)}

    # RED
    t0 = time.time()
    red_sub, red_scale = _crop_and_scale(image, red_search)
    save_img("03_red_sub.jpg", red_sub)
    mat_result = _locate_material_code_column(red_sub)
    mat_bbox = None
    if mat_result[0] is not None:
        mat_bbox_sub, direction = mat_result
        mat_bbox = _map_bbox_back(mat_bbox_sub, red_search, red_scale)
        logger.info(f"  RED: direction={direction}, bbox={mat_bbox} ({time.time()-t0:.1f}s)")
        save_img("04_red_crop.jpg", image[mat_bbox.y:mat_bbox.y2, mat_bbox.x:mat_bbox.x2])
    else:
        logger.warning(f"  RED: None ({time.time()-t0:.1f}s)")

    # GREEN
    t0 = time.time()
    green_sub, green_scale = _crop_and_scale(image, green_search)
    save_img("05_green_sub.jpg", green_sub)
    br_result = _locate_bottom_right_number(green_sub, prefixes=DEFAULT_PREFIXES)
    br_text, br_bbox = None, None
    if br_result:
        br_text, br_bbox_sub = br_result
        br_bbox = _map_bbox_back(br_bbox_sub, green_search, green_scale)
        logger.info(f"  GREEN: text='{br_text}', bbox={br_bbox} ({time.time()-t0:.1f}s)")
        save_img("06_green_crop.jpg", image[br_bbox.y:br_bbox.y2, br_bbox.x:br_bbox.x2])
    else:
        logger.warning(f"  GREEN: None ({time.time()-t0:.1f}s)")

    # PURPLE
    t0 = time.time()
    purple_sub, purple_scale = _crop_and_scale(image, purple_search)
    save_img("07_purple_sub.jpg", purple_sub)
    bl_result = _locate_bottom_left_number(purple_sub, prefixes=DEFAULT_PREFIXES)
    bl_text, bl_bbox = None, None
    if bl_result:
        bl_text, bl_bbox_sub, _ = bl_result
        bl_bbox = _map_bbox_back(bl_bbox_sub, purple_search, purple_scale)
        logger.info(f"  PURPLE: text='{bl_text}', bbox={bl_bbox} ({time.time()-t0:.1f}s)")
        save_img("08_purple_crop.jpg", image[bl_bbox.y:bl_bbox.y2, bl_bbox.x:bl_bbox.x2])
    else:
        logger.warning(f"  PURPLE: None ({time.time()-t0:.1f}s)")

    # ORANGE
    t0 = time.time()
    orange_sub, orange_scale = _crop_and_scale(image, orange_search)
    save_img("09_orange_sub.jpg", orange_sub)
    tl_result = _locate_top_left_number(orange_sub, None, prefixes=DEFAULT_PREFIXES)
    tl_text, tl_bbox = None, None
    if tl_result:
        tl_text, tl_bbox_sub = tl_result
        tl_bbox = _map_bbox_back(tl_bbox_sub, orange_search, orange_scale)
        logger.info(f"  ORANGE: text='{tl_text}', bbox={tl_bbox} ({time.time()-t0:.1f}s)")
        save_img("10_orange_crop.jpg", image[tl_bbox.y:tl_bbox.y2, tl_bbox.x:tl_bbox.x2])
    else:
        logger.warning(f"  ORANGE: None ({time.time()-t0:.1f}s)")

    # CYAN
    t0 = time.time()
    cyan_boxes = []
    if mat_bbox is not None:
        row_ys = detect_row_ys_for_red_box(image, mat_bbox, table_search_bbox=red_search)
        p_chars = "".join(p.upper() for p in DEFAULT_PREFIXES)
        red_pattern = rf"\b{p_chars}[A-Z0-9\-]{{8}}\b" if len(p_chars) == 1 else rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b"
        cyan_boxes, _ = detect_cyan_boxes(image, mat_bbox, row_ys,
                                       pattern=red_pattern, prefixes=DEFAULT_PREFIXES)
        logger.info(f"  CYAN: {len(cyan_boxes)} boxes ({time.time()-t0:.1f}s)")
    else:
        logger.warning("  CYAN: skipped (red=None)")

    detect_time = time.time() - detect_start
    logger.info(f"  total detection: {detect_time:.1f}s")

    # ── Step 4: combined result ──
    logger.info("=" * 60)
    logger.info("Step 4: all results combined")
    logger.info("=" * 60)

    combined = image.copy()
    results = {"red": mat_bbox, "green": br_bbox,
               "orange": tl_bbox, "purple": bl_bbox}
    texts = {"green": br_text, "orange": tl_text, "purple": bl_text}

    for name, bbox in results.items():
        if bbox is None:
            logger.info(f"  {name}: None")
            continue
        c = colors[name]
        cv2.rectangle(combined, (bbox.x, bbox.y), (bbox.x2, bbox.y2), c, 4)
        label = f"{name}"
        if texts.get(name):
            label += f": '{texts[name]}'"
        cv2.putText(combined, label, (bbox.x, max(bbox.y - 15, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
        logger.info(f"  {name}: {bbox}" + (f" text='{texts.get(name)}'" if texts.get(name) else ""))

    for cb in cyan_boxes:
        cv2.rectangle(combined, (cb.x, cb.y), (cb.x2, cb.y2), (0, 255, 255), 3)
    logger.info(f"  cyan: {len(cyan_boxes)} boxes")

    save_img("11_all_results.jpg", combined)

    if cyan_boxes and mat_bbox:
        cyan_vis = image.copy()
        for cb in cyan_boxes:
            cv2.rectangle(cyan_vis, (cb.x, cb.y), (cb.x2, cb.y2), (0, 255, 255), 3)
        pad = 50
        rx1, ry1 = max(mat_bbox.x-pad, 0), max(mat_bbox.y-pad, 0)
        rx2, ry2 = min(mat_bbox.x2+pad, img_w), min(mat_bbox.y2+pad, img_h)
        save_img("12_cyan_on_red_crop.jpg", cyan_vis[ry1:ry2, rx1:rx2])

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"done! total: {total_time:.1f}s (detection: {detect_time:.1f}s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
