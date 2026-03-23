#!/usr/bin/env python3
"""快速测试 A216 和 A360"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import _crop_and_scale, _locate_bottom_right_number, BBox
from modules.file_ingestion import load_file

config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
TARGETS = [
    ("YA116A216-1_Fh-脱敏.tif", "YA116A216"),
    ("YA128A360-1_C-脱敏.tif", "YA128A360"),
]

for fname, expected in TARGETS:
    fpath = os.path.join(TIF_DIR, fname)
    image, _ = load_file(fpath)
    h, w = image.shape[:2]
    gx, gy = int(w * 5 / 8), int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub, scale = _crop_and_scale(image, green_search)

    t0 = time.time()
    result = _locate_bottom_right_number(sub, prefixes=None)
    ms = int((time.time() - t0) * 1000)

    if result:
        text, bbox = result
        ok = "OK" if text.startswith(expected) else "WRONG"
        print(f"{fname:<40s} {ok:<6s} {text:<20s} {ms}ms")
    else:
        print(f"{fname:<40s} FAIL   -                    {ms}ms")
