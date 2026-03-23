#!/usr/bin/env python3
"""详细诊断单个文件的绿框检测。"""

import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _locate_bottom_right_number,
    _find_bottom_right_cell, _search_y_number, _ocr_region,
    _validate_green_result, _clip_bbox_to_cell,
    BBox, make_pattern,
)
from modules.file_ingestion import load_file
import re

config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"

# 诊断这几个问题文件
TARGETS = [
    "YA097C376$1_D.tif",
    "YA116A216-1_Fh-脱敏.tif",
    "YA128A360-1_C-脱敏.tif",
]

def save_img(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())

def diagnose(fname):
    fpath = os.path.join(TIF_DIR, fname)
    image, _ = load_file(fpath)
    h, w = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"FILE: {fname}  ({w}x{h})")
    print(f"{'='*60}")

    gx = int(w * 5 / 8)
    gy = int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub, scale = _crop_and_scale(image, green_search)
    sh, sw = sub.shape[:2]
    print(f"  绿框搜索区: {green_search}, scaled {sw}x{sh} (scale={scale:.3f})")

    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)

    # Step 1: raw OCR
    print(f"\n  --- RAW OCR on full area ---")
    search_full = BBox(0, 0, sw, sh)
    ocr_results = _ocr_region(sub, search_full)
    for i, (text, conf, bbox) in enumerate(ocr_results):
        print(f"  [{i:2d}] text={text!r:30s} conf={conf:.2f}  bbox={bbox}")

    # Step 2: Y-number search
    y_re = re.compile(make_pattern(None))
    loose_re = re.compile(r"[A-Z][A-Z0-9]*\d{2,}[A-Z]\d{2,}")
    result = _search_y_number(sub, search_full, y_re, loose_re, prefixes=None)
    if result:
        y_text, bbox = result
        valid = _validate_green_result(y_text)
        print(f"\n  _search_y_number: text={y_text!r}, bbox={bbox}, valid={valid}")
    else:
        print(f"\n  _search_y_number: None")

    # Step 3: Cell detection
    cell = _find_bottom_right_cell(gray, sh, sw)
    print(f"\n  _find_bottom_right_cell: {cell}")

    # Step 4: If result found, show clip
    if result and cell:
        y_text, bbox = result
        clipped = _clip_bbox_to_cell(bbox, cell)
        print(f"  _clip_bbox_to_cell: {bbox} → {clipped}")

    print()


for fname in TARGETS:
    diagnose(fname)
