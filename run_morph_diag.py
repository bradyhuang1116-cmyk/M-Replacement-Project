#!/usr/bin/env python3
"""单独生成各图的 morph 图：竖线=红, 横线=蓝, 黑底。"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.region_detector import _crop_and_scale, _binarize_for_lines, BBox
from modules.file_ingestion import load_file

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUT_DIR = os.path.join("diagnostic_output", "morph_only")
os.makedirs(OUT_DIR, exist_ok=True)


def save_img(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def run_one(fname):
    fpath = os.path.join(TIF_DIR, fname)
    base = os.path.splitext(fname)[0]

    image, _ = load_file(fpath)
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]

    # 绿框搜索区域：右下 3/8 宽 × 1/6 高
    gx = int(w * 5 / 8)
    gy = int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub_scaled, _ = _crop_and_scale(image, green_search)

    img_h, img_w = sub_scaled.shape[:2]
    gray = cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2GRAY)
    thresh = _binarize_for_lines(gray)

    # ── 竖线 morph ──
    min_vh = max(img_h // 8, 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # ── 横线 morph ──
    min_hw = max(img_w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # ── morph 图：黑底，竖线=红，横线=蓝 ──
    morph_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    morph_vis[v_morph > 0] = (0, 0, 255)   # 竖线 → 红 (BGR)
    morph_vis[h_morph > 0] = (255, 0, 0)   # 横线 → 蓝 (BGR)
    cv2.putText(morph_vis, f"Red=V(k={min_vh},i2) Blue=H(k={min_hw},i1)",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(morph_vis, base,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    save_img(os.path.join(OUT_DIR, f"{base}_morph.jpg"), morph_vis)
    print(f"  {fname} -> {base}_morph.jpg  ({img_w}x{img_h})")


def main():
    files = sorted([f for f in os.listdir(TIF_DIR)
                    if f.upper().startswith("Y") and f.lower().endswith(".tif")])
    print(f"Found {len(files)} Y-prefixed TIF files\n")

    for fname in files:
        try:
            run_one(fname)
        except Exception as e:
            print(f"  {fname} ERROR: {e}")

    print(f"\nOutput: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
