#!/usr/bin/env python3
"""输出各图纸绿框区域的 cell morph 诊断图。

每张图输出：
- 左：原图 + 黄虚线OCR框 + 绿实线最终框 + 青色cell框
- 右：morph图(竖线=红, 横线=蓝) + 黄虚线OCR框 + 绿实线最终框 + 青色cell框
"""

import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _locate_bottom_right_number,
    _locate_bottom_right_number_core, _binarize_for_lines, BBox,
)
from modules.file_ingestion import load_file

config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUT_DIR = os.path.join("diagnostic_output", "cell_morph_diag")
os.makedirs(OUT_DIR, exist_ok=True)


def save_img(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def draw_dashed_rect(img, bbox, color, thickness=2, dash=10):
    pts = [
        ((bbox.x, bbox.y), (bbox.x2, bbox.y)),
        ((bbox.x2, bbox.y), (bbox.x2, bbox.y2)),
        ((bbox.x2, bbox.y2), (bbox.x, bbox.y2)),
        ((bbox.x, bbox.y2), (bbox.x, bbox.y)),
    ]
    for (x1, y1), (x2, y2) in pts:
        dx, dy = x2 - x1, y2 - y1
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        steps = length // dash
        if steps == 0:
            steps = 1
        for i in range(0, steps, 2):
            sx = x1 + dx * i // steps
            sy = y1 + dy * i // steps
            ex = x1 + dx * min(i + 1, steps) // steps
            ey = y1 + dy * min(i + 1, steps) // steps
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def find_cell_from_morph(gray, text_bbox, img_h, img_w):
    """复制 _find_cell_boundary 的局部裁切 + morph 线检测逻辑，
    返回 (cell_bbox, valid_vx, valid_hy, v_morph, h_morph, crop_x1, crop_y1)。"""

    # 局部裁切 + 等比放大
    pad_w = text_bbox.w * 2
    pad_h = text_bbox.h * 2
    crop_x1 = max(0, text_bbox.x - pad_w)
    crop_y1 = max(0, text_bbox.y - pad_h)
    crop_x2 = min(img_w, text_bbox.x2 + pad_w)
    crop_y2 = min(img_h, text_bbox.y2 + pad_h)
    local_gray = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    local_h, local_w = local_gray.shape[:2]

    TARGET_H = 600
    scale = TARGET_H / local_h if local_h > 0 else 1.0
    if scale > 1.0:
        scaled_gray = cv2.resize(local_gray, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
    else:
        scaled_gray = local_gray
        scale = 1.0
    scaled_h, scaled_w = scaled_gray.shape[:2]

    thresh = _binarize_for_lines(scaled_gray)

    # 竖线 morph
    min_vh = max(scaled_h // 8, 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    vlines_raw = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        vlines_raw.append((crop_x1 + int((cx + cw // 2) / scale),
                           crop_y1 + int(cy / scale),
                           crop_y1 + int((cy + ch) / scale)))
    vlines_raw.sort()
    vlines = []
    for x, y1, y2 in vlines_raw:
        if vlines and x - vlines[-1][0] <= 10:
            ox, oy1, oy2 = vlines[-1]
            vlines[-1] = (ox, min(oy1, y1), max(oy2, y2))
        else:
            vlines.append((x, y1, y2))

    # 横线 morph
    min_hw = max(scaled_w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    hlines_raw = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        hlines_raw.append((crop_y1 + int((cy + ch // 2) / scale),
                           crop_x1 + int(cx / scale),
                           crop_x1 + int((cx + cw) / scale)))
    hlines_raw.sort()
    hlines = []
    for y, x1, x2 in hlines_raw:
        if hlines and y - hlines[-1][0] <= 10:
            oy, ox1, ox2 = hlines[-1]
            hlines[-1] = (oy, min(ox1, x1), max(ox2, x2))
        else:
            hlines.append((y, x1, x2))

    text_cx = (text_bbox.x + text_bbox.x2) // 2
    text_cy = (text_bbox.y + text_bbox.y2) // 2

    valid_vx = [x for x, y1, y2 in vlines if y1 <= text_cy <= y2]
    valid_hy = [y for y, x1, x2 in hlines if x1 <= text_cx <= x2]

    cell_left = 0
    for x in valid_vx:
        if x < text_cx:
            cell_left = x
        else:
            break

    cell_right = img_w
    for x in valid_vx:
        if x > text_cx:
            cell_right = x
            break

    cell_top = 0
    for y in valid_hy:
        if y < text_cy:
            cell_top = y
        else:
            break

    cell_bottom = img_h
    for y in valid_hy:
        if y > text_cy:
            cell_bottom = y
            break

    cell = BBox(cell_left, cell_top,
                cell_right - cell_left, cell_bottom - cell_top)

    # 把放大后的 v_morph/h_morph 缩放回局部尺寸，再嵌回全图用于可视化
    if scale > 1.0:
        v_morph_local = cv2.resize(v_morph, (local_w, local_h),
                                   interpolation=cv2.INTER_NEAREST)
        h_morph_local = cv2.resize(h_morph, (local_w, local_h),
                                   interpolation=cv2.INTER_NEAREST)
    else:
        v_morph_local = v_morph
        h_morph_local = h_morph
    v_morph_full = np.zeros((img_h, img_w), dtype=np.uint8)
    v_morph_full[crop_y1:crop_y2, crop_x1:crop_x2] = v_morph_local
    h_morph_full = np.zeros((img_h, img_w), dtype=np.uint8)
    h_morph_full[crop_y1:crop_y2, crop_x1:crop_x2] = h_morph_local

    return cell, valid_vx, valid_hy, v_morph_full, h_morph_full


def run_one(fname):
    fpath = os.path.join(TIF_DIR, fname)
    base = os.path.splitext(fname)[0]

    image, _ = load_file(fpath)
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]

    gx = int(w * 5 / 8)
    gy = int(h * 5 / 6)
    green_search = BBox(gx, gy, w - gx, h - gy)
    sub_scaled, _ = _crop_and_scale(image, green_search)

    img_h, img_w = sub_scaled.shape[:2]
    gray = cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2GRAY)

    # OCR bbox
    core_result = _locate_bottom_right_number_core(
        sub_scaled, gray, img_h, img_w, prefixes=None
    )
    # 最终 bbox（经过 cell boundary）
    final_result = _locate_bottom_right_number(sub_scaled, prefixes=None)

    if not (core_result and final_result):
        print(f"  {fname} FAIL: no detection")
        return

    y_text, ocr_bbox = core_result
    _, final_bbox = final_result

    # 获取 cell 和 morph 数据
    cell, merged_vx, merged_hy, v_morph, h_morph = \
        find_cell_from_morph(gray, ocr_bbox, img_h, img_w)

    ocr_area = ocr_bbox.w * ocr_bbox.h
    cell_area = cell.w * cell.h

    # ── 左图：原图 + OCR框(黄虚线) + 最终框(绿) + cell(青) ──
    left = cv2.cvtColor(sub_scaled.copy(), cv2.COLOR_RGB2BGR)
    draw_dashed_rect(left, ocr_bbox, color=(0, 255, 255), thickness=2, dash=10)
    cv2.rectangle(left, (final_bbox.x, final_bbox.y),
                  (final_bbox.x2, final_bbox.y2), (0, 255, 0), 2)
    cv2.rectangle(left, (cell.x, cell.y),
                  (cell.x2, cell.y2), (255, 255, 0), 2)  # 青色=cell
    cv2.putText(left, f"{y_text}  OCR_area={ocr_area} Cell_area={cell_area}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(left, "Yellow-dash=OCR  Green=Final  Cyan=Cell",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── 右图：morph(竖红横蓝) + 同样的三个框 ──
    right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    right[v_morph > 0] = (0, 0, 255)
    right[h_morph > 0] = (255, 0, 0)
    draw_dashed_rect(right, ocr_bbox, color=(0, 255, 255), thickness=2, dash=10)
    cv2.rectangle(right, (final_bbox.x, final_bbox.y),
                  (final_bbox.x2, final_bbox.y2), (0, 255, 0), 2)
    cv2.rectangle(right, (cell.x, cell.y),
                  (cell.x2, cell.y2), (255, 255, 0), 2)
    cv2.putText(right, f"vx={merged_vx}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(right, f"hy={merged_hy}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 拼接
    gap = np.ones((img_h, 4, 3), dtype=np.uint8) * 128
    vis = np.hstack([left, gap, right])
    save_img(os.path.join(OUT_DIR, f"{base}.jpg"), vis)
    print(f"  {fname}  OCR={ocr_bbox} Cell={cell} Final={final_bbox}  "
          f"OCR_area={ocr_area} Cell_area={cell_area}")


def main():
    files = sorted([f for f in os.listdir(TIF_DIR)
                    if f.upper().startswith("Y") and f.lower().endswith(".tif")])
    print(f"Found {len(files)} files\n")
    for fname in files:
        try:
            run_one(fname)
        except Exception as e:
            print(f"  {fname} ERROR: {e}")
            import traceback; traceback.print_exc()
    print(f"\nOutput: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
