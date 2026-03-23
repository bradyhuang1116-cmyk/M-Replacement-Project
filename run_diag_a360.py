#!/usr/bin/env python3
"""A360 专项诊断"""
import os, sys, re
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _ocr_region, _search_y_number,
    _find_right_vline, _validate_green_result,
    BBox, make_pattern,
)
from modules.file_ingestion import load_file

config.set_ocr_mode("cpu", "server")

fpath = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA128A360-1_C-脱敏.tif"
image, _ = load_file(fpath)
h, w = image.shape[:2]

gx, gy = int(w * 5 / 8), int(h * 5 / 6)
green_search = BBox(gx, gy, w - gx, h - gy)
sub, scale = _crop_and_scale(image, green_search)
sh, sw = sub.shape[:2]
print(f"Sub image: {sw}x{sh}, scale={scale:.3f}")

y_re = re.compile(make_pattern(None))
loose_re = re.compile(r"[A-Z][A-Z0-9]*\d{2,}[A-Z]\d{2,}")

# L1 full area
search_full = BBox(0, 0, sw, sh)
result = _search_y_number(sub, search_full, y_re, loose_re, prefixes=None)
if result:
    y_text, bbox = result
    print(f"\nL1 result: '{y_text}' bbox={bbox}")
    print(f"  valid: {_validate_green_result(y_text)}")

    # 扩展区域
    char_w = bbox.w / max(len(y_text), 1)
    ext_w = min(int(char_w * 2.5), sw - bbox.x - bbox.w)
    ext_bbox = BBox(bbox.x, bbox.y, bbox.w + ext_w, bbox.h)
    print(f"\n  Extended bbox: {ext_bbox} (ext_w={ext_w})")

    # 在扩展区域上 OCR
    print(f"\n  --- OCR on extended bbox ---")
    ocr_results = _ocr_region(sub, ext_bbox)
    for i, (text, conf, poly) in enumerate(ocr_results):
        print(f"  [{i}] text={text!r:30s} conf={conf:.2f}")

    # search_y_number on extended
    re_result = _search_y_number(sub, ext_bbox, y_re, loose_re, prefixes=None)
    if re_result:
        re_text, re_bbox = re_result
        print(f"\n  search_y_number on ext: '{re_text}' bbox={re_bbox}")
        print(f"  valid: {_validate_green_result(re_text)}")
    else:
        print(f"\n  search_y_number on ext: None")

    # 尝试更大的扩展
    ext_w2 = min(int(char_w * 4), sw - bbox.x - bbox.w)
    ext_bbox2 = BBox(bbox.x, bbox.y, bbox.w + ext_w2, bbox.h)
    print(f"\n  Extended bbox (larger): {ext_bbox2} (ext_w={ext_w2})")
    ocr_results2 = _ocr_region(sub, ext_bbox2)
    for i, (text, conf, poly) in enumerate(ocr_results2):
        print(f"  [{i}] text={text!r:30s} conf={conf:.2f}")
    re_result2 = _search_y_number(sub, ext_bbox2, y_re, loose_re, prefixes=None)
    if re_result2:
        re_text2, re_bbox2 = re_result2
        print(f"\n  search_y_number on ext2: '{re_text2}' bbox={re_bbox2}")
    else:
        print(f"\n  search_y_number on ext2: None")
else:
    print("L1: None")
