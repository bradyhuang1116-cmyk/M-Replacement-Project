#!/usr/bin/env python3
"""绿框诊断 v4：输出 OCR识别框 + 竖线/横线 morph 合一图。"""

import os, sys, time, traceback
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _locate_bottom_right_number, _find_cell_boundary,
    _locate_bottom_right_number_core, _binarize_for_lines, BBox,
)
from modules.file_ingestion import load_file

config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUT_DIR = os.path.join("diagnostic_output", "green_diag_v4")
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED = {
    "YA026D941_0d.tif":           "YA026D941",
    "YA036C079$1_B.tif":          "YA036C079",
    "YA050C055_CD.tif":           "YA050C055",
    "YA057C857_0-脱敏.tif":       "YA057C857",
    "YA070A191P7933-1_0-脱敏.tif": "YA070A191P7933",
    "YA097C376$1_D.tif":          "YA097C376",
    "YA116A216-1_Fh-脱敏.tif":    "YA116A216",
    "YA116A226-1_Eg-脱敏.tif":    "YA116A226",
    "YA128A360-1_C-脱敏.tif":     "YA128A360",
    "YA147C070P3335_0-脱敏.tif":  "YA147C070",
    "YA246C928_H.tif":            "YA246C928",
}


def save_img(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def make_diag_image(sub_scaled, y_text, ocr_bbox, clipped_bbox):
    """生成诊断图：三列布局。

    左=原图(带OCR框+最终框)，中=二值化图，右=morph图(竖线红+横线蓝)。
    """
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

    # ── 左图：原图 + OCR框(黄虚线) + 最终框(绿实线) ──
    left = cv2.cvtColor(sub_scaled.copy(), cv2.COLOR_RGB2BGR)
    draw_dashed_rect(left, ocr_bbox, color=(0, 255, 255), thickness=2, dash=10)
    cv2.rectangle(left, (clipped_bbox.x, clipped_bbox.y),
                  (clipped_bbox.x2, clipped_bbox.y2), (0, 255, 0), 2)
    cv2.putText(left, y_text, (clipped_bbox.x, clipped_bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(left, "Yellow dashed=OCR  Green=Final",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(left, "Yellow dashed=OCR  Green=Final",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # ── 中图：二值化图（白线黑底） ──
    mid = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.putText(mid, "Binary (white=foreground)",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # ── 右图：黑底 morph，竖线=红，横线=蓝 ──
    right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    right[v_morph > 0] = (0, 0, 255)   # 竖线 → 红 (BGR)
    right[h_morph > 0] = (255, 0, 0)   # 横线 → 蓝 (BGR)
    cv2.putText(right, f"Red=V(k={min_vh},i2) Blue=H(k={min_hw},i1)",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── 拼接：三列并排 ──
    gap = np.ones((img_h, 4, 3), dtype=np.uint8) * 128
    vis = np.hstack([left, gap, mid, gap, right])
    return vis


def draw_dashed_rect(img, bbox, color, thickness=2, dash=10):
    """画虚线矩形。"""
    pts = [
        ((bbox.x, bbox.y), (bbox.x2, bbox.y)),
        ((bbox.x2, bbox.y), (bbox.x2, bbox.y2)),
        ((bbox.x2, bbox.y2), (bbox.x, bbox.y2)),
        ((bbox.x, bbox.y2), (bbox.x, bbox.y)),
    ]
    for (x1, y1), (x2, y2) in pts:
        dx = x2 - x1
        dy = y2 - y1
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        steps = length // dash
        for i in range(0, steps, 2):
            sx = x1 + dx * i // steps
            sy = y1 + dy * i // steps
            ex = x1 + dx * min(i + 1, steps) // steps
            ey = y1 + dy * min(i + 1, steps) // steps
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def run_one(fname):
    fpath = os.path.join(TIF_DIR, fname)
    base = os.path.splitext(fname)[0]

    t0 = time.time()
    image, _ = load_file(fpath)
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]

    gx = int(w * 5 / 8)
    gy = int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub_scaled, green_scale = _crop_and_scale(image, green_search)

    img_h, img_w = sub_scaled.shape[:2]
    gray = cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2GRAY)

    # 获取原始 OCR bbox（不经过 cell boundary 裁切）
    core_result = _locate_bottom_right_number_core(
        sub_scaled, gray, img_h, img_w, prefixes=None
    )

    # 获取最终结果（经过 cell boundary 裁切）
    final_result = _locate_bottom_right_number(sub_scaled, prefixes=None)

    elapsed = int((time.time() - t0) * 1000)

    if final_result and core_result:
        y_text, clipped_bbox = final_result
        _, ocr_bbox = core_result

        diag = make_diag_image(sub_scaled, y_text, ocr_bbox, clipped_bbox)
        save_img(os.path.join(OUT_DIR, f"{base}.jpg"), diag)
        return y_text, elapsed
    else:
        save_img(os.path.join(OUT_DIR, f"{base}_FAIL.jpg"),
                 cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2BGR))
        return None, elapsed


def main():
    files = sorted([f for f in os.listdir(TIF_DIR)
                    if f.upper().startswith("Y") and f.lower().endswith(".tif")])
    print(f"Found {len(files)} Y-prefixed TIF files\n")

    ok_count = 0
    results = []
    for fname in files:
        exp = EXPECTED.get(fname, "???")
        try:
            text, ms = run_one(fname)
            if text and text.startswith(exp):
                status = "OK"
                ok_count += 1
            elif text:
                status = "WRONG"
            else:
                status = "FAIL"
            line = f"{fname:<45s} {status:<8s} {text or '-':<25s} {ms}ms"
        except Exception as e:
            traceback.print_exc()
            line = f"{fname:<45s} ERROR    {str(e)[:40]}"
        print(line)
        results.append(line)

    print(f"\nSummary: {ok_count}/{len(files)} OK")
    print(f"Output: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
