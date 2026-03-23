"""绿框快速诊断 — 只跑绿框检测，跳过全量检测"""
import os, sys, re, logging

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("green_fast")

import cv2
import numpy as np
from modules.region_detector import BBox, _crop_and_scale, _map_bbox_back, _auto_rotate_portrait, _locate_bottom_right_number, _ocr_region
from modules.file_ingestion import load_file
from config import DEFAULT_PREFIXES, make_pattern

TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA128A360-1_C-脱敏.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "green_box")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_img(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  saved: {name} ({img.shape[1]}x{img.shape[0]})")

image, _ = load_file(TEST_IMAGE)
img_h, img_w = image.shape[:2]
logger.info(f"image: {img_w}x{img_h}")

if img_h > img_w:
    image, _ = _auto_rotate_portrait(image)
    img_h, img_w = image.shape[:2]

green_x = int(img_w * (5.0 / 8.0))
green_y = int(img_h * (5.0 / 6.0))
green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)
logger.info(f"green_search: {green_search}")

green_sub, green_scale = _crop_and_scale(image, green_search)
logger.info(f"green_sub: {green_sub.shape[1]}x{green_sub.shape[0]}, scale={green_scale:.3f}")

# OCR all text in green sub for diagnosis
sub_h, sub_w = green_sub.shape[:2]
full_bbox = BBox(0, 0, sub_w, sub_h)
ocr_results = _ocr_region(green_sub, full_bbox)
y_re = re.compile(make_pattern())
log_lines = []
log_lines.append(f"OCR found {len(ocr_results)} items:")
for i, (text, conf, poly) in enumerate(ocr_results):
    text_clean = text.upper().replace(" ", "")
    is_match = bool(y_re.search(text_clean))
    flag = " <<< MATCH" if is_match else ""
    line = f"  [{i}] text='{text}' conf={conf:.2f}{flag}"
    log_lines.append(line)
    if poly:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        log_lines.append(f"       bbox: x={int(min(xs))}, y={int(min(ys))}, w={int(max(xs)-min(xs))}, h={int(max(ys)-min(ys))}")

# Write OCR results to file
with open(os.path.join(OUTPUT_DIR, "ocr_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

br_result = _locate_bottom_right_number(green_sub, prefixes=DEFAULT_PREFIXES)

if br_result:
    br_text, br_bbox_sub = br_result
    logger.info(f"TEXT: '{br_text}'")
    logger.info(f"BBOX_SUB: x={br_bbox_sub.x} y={br_bbox_sub.y} w={br_bbox_sub.w} h={br_bbox_sub.h}")

    result_vis = green_sub.copy()
    cv2.rectangle(result_vis, (br_bbox_sub.x, br_bbox_sub.y), (br_bbox_sub.x2, br_bbox_sub.y2), (0, 255, 0), 3)
    save_img("07_green_result_sub.jpg", result_vis)

    br_bbox = _map_bbox_back(br_bbox_sub, green_search, green_scale)
    logger.info(f"BBOX_FULL: x={br_bbox.x} y={br_bbox.y} w={br_bbox.w} h={br_bbox.h}")

    crop = image[br_bbox.y:br_bbox.y2, br_bbox.x:br_bbox.x2]
    save_img("09_green_final_crop.jpg", crop)
else:
    logger.warning("FAILED: returned None")

logger.info("done")
