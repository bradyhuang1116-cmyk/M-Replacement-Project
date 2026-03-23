#!/usr/bin/env python3
"""裁切出识别到的单元格区域（+padding），在裁切图上重新OCR+morph，生成单元格边界。

输出：
- 左：裁切后原图 + 新OCR框(黄虚线) + cell(青) + 识别文字
- 右：裁切后morph图(竖红横蓝) + 新OCR框(黄虚线) + cell(青)
"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.region_detector import (
    _crop_and_scale, _locate_bottom_right_number_core,
    _find_cell_boundary, _binarize_for_lines, BBox,
)
from modules.file_ingestion import load_file

config.set_ocr_mode("cpu", "server")

TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUT_DIR = os.path.join("diagnostic_output", "cell_crop")
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


def detect_morph_lines_with_coords(gray):
    """在gray上做morph线检测，返回 (vlines, hlines, v_morph, h_morph)。"""
    h, w = gray.shape[:2]
    thresh = _binarize_for_lines(gray)

    min_vh = max(int(h / 6.5), 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    vlines_raw = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        vlines_raw.append((cx + cw // 2, cy, cy + ch))
    vlines_raw.sort()
    vlines = []
    for x, y1, y2 in vlines_raw:
        if vlines and x - vlines[-1][0] <= 10:
            ox, oy1, oy2 = vlines[-1]
            vlines[-1] = (ox, min(oy1, y1), max(oy2, y2))
        else:
            vlines.append((x, y1, y2))

    min_hw = max(w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    hlines_raw = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        hlines_raw.append((cy + ch // 2, cx, cx + cw))
    hlines_raw.sort()
    hlines = []
    for y, x1, x2 in hlines_raw:
        if hlines and y - hlines[-1][0] <= 10:
            oy, ox1, ox2 = hlines[-1]
            hlines[-1] = (oy, min(ox1, x1), max(ox2, x2))
        else:
            hlines.append((y, x1, x2))

    return vlines, hlines, v_morph, h_morph


def find_cell(ocr_bbox, vlines, hlines, img_w):
    """用morph线生成单元格边界。所有边界均为morph红蓝线。

    规则：
    - 上/左/下：离OCR边界绝对距离最近的morph线（可在OCR内侧或外侧）
    - 右：
      1. 找离OCR右边界绝对距离最近的竖线
      2. 如果在OCR外侧或重合：
         - OCR右侧1/3内没有其他竖线 → 用该外侧竖线
         - OCR右侧1/3内有竖线 → 用右侧1/3内的那根竖线
      3. 如果在OCR内侧，且不是画面最右侧竖线 → 用该竖线
      4. 如果在OCR内侧，且是画面最右侧竖线 → 用OCR右边界左侧第二近的竖线
      5. 单元格宽度 < OCR宽度70% → fallback到OCR右边界
    """
    # ── 左边界 ──
    cell_left = ocr_bbox.x
    best_dist = float('inf')
    for x, y1, y2 in vlines:
        dist = abs(x - ocr_bbox.x)
        if dist < best_dist:
            best_dist = dist
            cell_left = x

    # ── 上边界 ──
    cell_top = ocr_bbox.y
    best_dist = float('inf')
    for y, x1, x2 in hlines:
        dist = abs(y - ocr_bbox.y)
        if dist < best_dist:
            best_dist = dist
            cell_top = y

    # ── 下边界：离OCR下边界绝对距离最近的横线，且长度≥OCR宽度70% ──
    cell_bottom = ocr_bbox.y2
    min_hlen = ocr_bbox.w * 0.7
    hlines_by_bottom_dist = sorted(hlines, key=lambda h: abs(h[0] - ocr_bbox.y2))
    for y, x1, x2 in hlines_by_bottom_dist:
        if (x2 - x1) >= min_hlen:
            cell_bottom = y
            break

    # ── 右边界 ──
    rightmost_vx = max((x for x, y1, y2 in vlines), default=None) if vlines else None
    vlines_by_dist = sorted(vlines, key=lambda v: abs(v[0] - ocr_bbox.x2))

    cell_right = ocr_bbox.x2  # fallback
    if vlines_by_dist:
        nearest_x = vlines_by_dist[0][0]
        if nearest_x >= ocr_bbox.x2:
            # 外侧或重合：检查OCR右侧1/3内是否有其他竖线
            right_third_start = ocr_bbox.x + int(ocr_bbox.w * 2 / 3)
            inner_right = [x for x, y1, y2 in vlines
                           if right_third_start <= x < ocr_bbox.x2]
            if inner_right:
                # 有 → 用右侧1/3内离OCR右边界最近的
                cell_right = max(inner_right)
            else:
                # 没有 → 用外侧竖线
                cell_right = nearest_x
        else:
            # 内侧
            if rightmost_vx is not None and nearest_x != rightmost_vx:
                cell_right = nearest_x
            else:
                # 是画面最右侧竖线 → 第二近
                left_vlines = sorted(
                    [(x, y1, y2) for x, y1, y2 in vlines if x < ocr_bbox.x2],
                    key=lambda v: ocr_bbox.x2 - v[0]
                )
                if len(left_vlines) >= 2:
                    cell_right = left_vlines[1][0]

    # 宽度检查
    cell_w = cell_right - cell_left
    if cell_w < ocr_bbox.w * 0.7:
        cell_right = ocr_bbox.x2

    return BBox(cell_left, cell_top,
                cell_right - cell_left, cell_bottom - cell_top)


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
    sub_scaled, _ = _crop_and_scale(image, BBox(gx, gy, w - gx, h - gy))
    img_h, img_w = sub_scaled.shape[:2]
    gray = cv2.cvtColor(sub_scaled, cv2.COLOR_RGB2GRAY)

    # 第一轮OCR（仅用于确定大致位置）
    core_result = _locate_bottom_right_number_core(
        sub_scaled, gray, img_h, img_w, prefixes=None
    )
    if not core_result:
        print(f"  {fname} FAIL: no detection")
        return

    y_text_orig, ocr_bbox_orig = core_result

    # 用v1获取cell做裁切基准
    _, cell_v1 = _find_cell_boundary(
        gray, ocr_bbox_orig, img_h, img_w, y_text_orig, _return_cell=True
    )

    # 裁切区域 = cell_v1 + 上/左25% padding，下/右保留原图到边
    pad_x = int(cell_v1.w * 0.25)
    pad_y = int(cell_v1.h * 0.25)
    cx1 = max(0, cell_v1.x - pad_x)
    cy1 = max(0, cell_v1.y - pad_y)
    cx2 = img_w
    cy2 = img_h

    cropped_rgb = sub_scaled[cy1:cy2, cx1:cx2].copy()
    cropped_gray = gray[cy1:cy2, cx1:cx2]
    ch, cw = cropped_gray.shape[:2]

    # 在裁切图上重新OCR
    new_core = _locate_bottom_right_number_core(
        cropped_rgb, cropped_gray, ch, cw, prefixes=None
    )
    new_ocr = new_core[1] if new_core else None
    new_text = new_core[0] if new_core else "N/A"

    # morph线检测
    vlines, hlines, v_morph, h_morph = detect_morph_lines_with_coords(cropped_gray)

    # 生成单元格边界
    cell = find_cell(new_ocr, vlines, hlines, cw) if new_ocr else None

    # 生成绿框：OCR面积 > cell面积 → 绿框=cell；否则绿框=OCR，溢出裁到cell
    # 补充：如果OCR右边界在cell内，绿框右边界以OCR右边界为准
    green = None
    if new_ocr and cell:
        ocr_area = new_ocr.w * new_ocr.h
        cell_area = cell.w * cell.h
        if ocr_area > cell_area:
            green = BBox(cell.x, cell.y, cell.w, cell.h)
            # OCR右边界在cell内 → 绿框右边界用OCR
            if new_ocr.x2 < cell.x2:
                green = BBox(green.x, green.y,
                             new_ocr.x2 - green.x, green.h)
        else:
            gx1 = new_ocr.x
            gy1 = new_ocr.y
            gx2 = new_ocr.x2
            gy2 = new_ocr.y2
            if gx1 < cell.x:
                gx1 = cell.x
            if gy1 < cell.y:
                gy1 = cell.y
            if gx2 > cell.x2:
                gx2 = cell.x2
            if gy2 > cell.y2:
                gy2 = cell.y2
            green = BBox(gx1, gy1, gx2 - gx1, gy2 - gy1)

    # 左图：裁切后原图 + OCR框(黄虚线) + cell(青) + 绿框(绿)
    left = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    if new_ocr:
        draw_dashed_rect(left, new_ocr, color=(0, 255, 255), thickness=2, dash=8)
    if cell:
        cv2.rectangle(left, (cell.x, cell.y), (cell.x2, cell.y2), (255, 255, 0), 2)
    if green:
        cv2.rectangle(left, (green.x, green.y), (green.x2, green.y2), (0, 255, 0), 3)
    cv2.putText(left, new_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 右图：morph(竖红横蓝) + OCR框(黄虚线) + cell(青) + 绿框(绿)
    right = np.zeros((ch, cw, 3), dtype=np.uint8)
    right[v_morph > 0] = (0, 0, 255)
    right[h_morph > 0] = (255, 0, 0)
    if new_ocr:
        draw_dashed_rect(right, new_ocr, color=(0, 255, 255), thickness=2, dash=8)
    if cell:
        cv2.rectangle(right, (cell.x, cell.y), (cell.x2, cell.y2), (255, 255, 0), 2)
    if green:
        cv2.rectangle(right, (green.x, green.y), (green.x2, green.y2), (0, 255, 0), 3)

    gap = np.ones((ch, 4, 3), dtype=np.uint8) * 128
    vis = np.hstack([left, gap, right])
    save_img(os.path.join(OUT_DIR, f"{base}.jpg"), vis)
    ocr_area = new_ocr.w * new_ocr.h if new_ocr else 0
    cell_area = cell.w * cell.h if cell else 0
    print(f"  {fname}  crop={cw}x{ch}  text={new_text}  ocr={new_ocr}  cell={cell}  green={green}  ocr_area={ocr_area} cell_area={cell_area}")


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
