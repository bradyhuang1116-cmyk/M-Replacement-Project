#!/usr/bin/env python3
"""批量测试绿框检测 v3（竖线裁剪 + 截断扩展 + 简化 cell），server 模式。"""

import os, sys, time, re, traceback
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _locate_bottom_right_number, BBox,
)
from modules.file_ingestion import load_file

# 切换到 server 模式
config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUT_DIR = os.path.join("diagnostic_output", "green_batch_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# 期望结果（Y 编号前缀，用于验证）
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
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())

def run_one(fname):
    fpath = os.path.join(TIF_DIR, fname)
    base = os.path.splitext(fname)[0]

    t0 = time.time()
    image, _ = load_file(fpath)
    h, w = image.shape[:2]

    # 竖立图片需要旋转（与 detect_all_regions 一致）
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]

    gx = int(w * 5 / 8)
    gy = int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub_scaled, green_scale = _crop_and_scale(image, green_search)

    # 执行检测
    result = _locate_bottom_right_number(sub_scaled, prefixes=None)
    elapsed = int((time.time() - t0) * 1000)

    if result:
        y_text, bbox = result
        vis = cv2.cvtColor(sub_scaled.copy(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 255, 0), 2)
        cv2.putText(vis, y_text, (bbox.x, bbox.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        save_img(os.path.join(OUT_DIR, f"{base}.jpg"), vis)
        return y_text, elapsed
    else:
        # 未识别时也保存原图
        save_img(os.path.join(OUT_DIR, f"{base}_FAIL.jpg"),
                 cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2BGR))
        return None, elapsed


def main():
    files = sorted([f for f in os.listdir(TIF_DIR)
                    if f.upper().startswith('Y') and f.lower().endswith('.tif')])
    print(f"Found {len(files)} Y-prefixed TIF files\n")

    results = []
    ok_count = 0
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
            text, ms, status = None, 0, "ERROR"
        print(line)
        results.append(line)

    summary = f"Summary (server v3): {ok_count}/{len(files)} OK"
    print(f"\n{summary}")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
        for line in results:
            f.write(line + "\n")

    print(f"\nOutput: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
