"""橙框二次识别升级测试脚本 — C857 + A191P7933 详细诊断版。

针对两个文件输出完整中间过程：
1. 橙框搜索区截图
2. 第一次识别结果截图（OCR bbox 标注）
3. V1 cell 标注图
4. 二次裁切区域截图
5. 二次识别 OCR bbox 标注图
6. 二次识别 morph 线图
7. 单元格 / OCR 识别框 / 最终橙框 对比图
8. 全图标注最终橙框
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import re
import cv2
import numpy as np
from modules.file_ingestion import load_file
from modules.region_detector import (
    detect_all_regions, BBox,
    _crop_and_scale, _locate_top_left_number_core,
    _find_cell_boundary, _detect_morph_lines,
    _find_cell_from_lines, _make_green_bbox, _binarize_for_lines,
    _enhance_vertical_lines, _ocr_region,
)
from config import make_pattern

TIF_DIR = Path(r'C:\Users\Brady Huang\Downloads\TIF_Undo')
OUT_DIR = Path('diagnostic_output/orange_upgrade')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = ['X', 'Y']

# ── 目标文件 ──
TARGET_PATTERNS = ['*C857*', '*C070P3335*']
target_files = []
for pat in TARGET_PATTERNS:
    target_files.extend(
        f for f in sorted(TIF_DIR.glob(pat))
        if f.suffix.lower() in ('.tif', '.tiff', '.pdf', '.png', '.jpg')
    )

if not target_files:
    print("未找到目标文件！")
    sys.exit(1)

print(f"目标文件: {[f.name for f in target_files]}\n")


def save_img(name, img):
    """保存全分辨率诊断图。"""
    path = OUT_DIR / name
    if img.ndim == 2:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                     if img.shape[2] == 3 else img,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'  已保存: {path}')


def draw_bbox(img, bbox, color, label, thickness=3):
    """在图上画 bbox + 标签。"""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    font_scale = max(0.5, min(w, h) / 200)
    cv2.putText(vis, label, (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return vis


def draw_multi_bbox(img, bboxes_colors):
    """在图上画多个 bbox。bboxes_colors = [(bbox, color, label), ...]"""
    vis = img.copy()
    for bbox, color, label in bboxes_colors:
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        font_scale = max(0.5, min(w, h) / 200)
        cv2.putText(vis, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return vis


def diagnose_file(f):
    """对文件进行完整橙框诊断，每步都输出图片。"""
    prefix = f.stem
    print(f'\n{"="*60}')
    print(f'  详细诊断: {f.name}')
    print(f'{"="*60}')

    img, _ = load_file(str(f))
    img_h, img_w = img.shape[:2]
    print(f'  原始尺寸: {img_w}x{img_h}')

    # 纵向旋转
    if img_h > img_w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_h, img_w = img.shape[:2]    
        print(f'  已旋转，尺寸: {img_w}x{img_h}')

    # ── 01. 保存原图 ──
    save_img(f'{prefix}_01_original.jpg', img)

    # ── 02. 橙框搜索区域（与 detect_all_regions 一致）──
    orange_h_full = int(img_h * (2.0 / 12.0))
    orange_h_base = orange_h_full - int(orange_h_full / 3)
    orange_h_expanded = orange_h_base + int(img_h * 0.05)
    orange_search = BBox(0, 0,
                         int(img_w * (2.0 / 8.0)),
                         orange_h_expanded)
    print(f'  搜索区域: {orange_search}')

    orange_sub, orange_scale = _crop_and_scale(img, orange_search)
    print(f'  裁切后: {orange_sub.shape[1]}x{orange_sub.shape[0]}, scale={orange_scale:.3f}')

    # 搜索区截图 + 边界标注
    vis_search = img.copy()
    cv2.rectangle(vis_search,
                  (orange_search.x, orange_search.y),
                  (orange_search.x2, orange_search.y2),
                  (0, 165, 255), 4)
    save_img(f'{prefix}_02a_search_on_full.jpg', vis_search)
    save_img(f'{prefix}_02b_search_crop.jpg', orange_sub)

    # ── 03. 第一次 OCR（原始结果全部输出）──
    sub_h, sub_w = orange_sub.shape[:2]
    gray = cv2.cvtColor(orange_sub, cv2.COLOR_RGB2GRAY)

    # 输出所有 OCR 结果
    all_search = BBox(0, 0, sub_w, sub_h)
    ocr_results = _ocr_region(orange_sub, all_search)
    print(f'\n  ── 第一次 OCR 全部结果 ({len(ocr_results)} 条) ──')
    vis_all_ocr = orange_sub.copy()
    y_re = re.compile(make_pattern(PREFIXES))
    for i, (text, conf, poly) in enumerate(ocr_results):
        text_up = text.upper().strip()
        is_match = bool(y_re.search(text_up.replace(" ", "")))
        marker = " ★匹配" if is_match else ""
        print(f'    [{i}] text="{text}", conf={conf:.2f}{marker}')
        if poly:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)
            color = (0, 255, 0) if is_match else (128, 128, 128)
            cv2.polylines(vis_all_ocr, [pts], True, color, 2)
            cv2.putText(vis_all_ocr, f'[{i}]{text}', (min(xs), max(ys) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    save_img(f'{prefix}_03_pass1_all_ocr.jpg', vis_all_ocr)

    # ── 04. 第一次核心识别 ──
    found = _locate_top_left_number_core(
        orange_sub, gray, sub_h, sub_w, None, PREFIXES
    )
    if found is None:
        print('  第一次识别: 未找到编号！')
        print('  诊断结束（无法继续）')
        return

    y_text_orig, ocr_bbox_orig = found
    print(f'\n  第一次识别结果: text="{y_text_orig}", bbox={ocr_bbox_orig}')

    vis1 = draw_bbox(orange_sub, ocr_bbox_orig, (0, 0, 255),
                     f'Pass1: {y_text_orig or "?"}')
    save_img(f'{prefix}_04_pass1_result.jpg', vis1)

    # ── 05. V1 cell ──
    clipped, cell_v1 = _find_cell_boundary(
        gray, ocr_bbox_orig, sub_h, sub_w,
        y_text_orig if y_text_orig else "UNKNOWN", _return_cell=True
    )
    print(f'  V1 clipped: {clipped}')
    print(f'  V1 cell:    {cell_v1}')

    vis_v1 = draw_multi_bbox(orange_sub, [
        (ocr_bbox_orig, (0, 0, 255), 'OCR bbox'),
        (clipped, (0, 255, 0), 'Clipped'),
        (cell_v1, (255, 0, 0), 'V1 cell'),
    ])
    save_img(f'{prefix}_05_v1_cell.jpg', vis_v1)

    # ── 06. 二次裁切 ──
    pad_x = int(cell_v1.w * 0.25)
    pad_y = int(cell_v1.h * 0.25)
    cx1 = 0
    cy1 = 0
    cx2 = min(sub_w, cell_v1.x2 + pad_x)
    cy2 = min(sub_h, cell_v1.y2 + pad_y)

    # 在搜索区图上标注裁切范围
    vis_crop_area = orange_sub.copy()
    cv2.rectangle(vis_crop_area, (cx1, cy1), (cx2, cy2), (255, 255, 0), 3)
    save_img(f'{prefix}_06a_crop2_area.jpg', vis_crop_area)

    cropped_rgb = orange_sub[cy1:cy2, cx1:cx2].copy()
    cropped_gray = gray[cy1:cy2, cx1:cx2]
    ch, cw = cropped_gray.shape[:2]
    print(f'  二次裁切: offset=({cx1},{cy1}), size={cw}x{ch}')
    save_img(f'{prefix}_06b_crop2.jpg', cropped_rgb)

    # ── 07. 二次 OCR 全部结果 ──
    crop_search = BBox(0, 0, cw, ch)
    ocr_results_2 = _ocr_region(cropped_rgb, crop_search)
    print(f'\n  ── 二次 OCR 全部结果 ({len(ocr_results_2)} 条) ──')
    vis_ocr2 = cropped_rgb.copy()
    for i, (text, conf, poly) in enumerate(ocr_results_2):
        text_up = text.upper().strip()
        is_match = bool(y_re.search(text_up.replace(" ", "")))
        marker = " ★匹配" if is_match else ""
        print(f'    [{i}] text="{text}", conf={conf:.2f}{marker}')
        if poly:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)
            color = (0, 255, 0) if is_match else (128, 128, 128)
            cv2.polylines(vis_ocr2, [pts], True, color, 2)
            cv2.putText(vis_ocr2, f'[{i}]{text}', (min(xs), max(ys) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    save_img(f'{prefix}_07_pass2_all_ocr.jpg', vis_ocr2)

    # ── 08. 二次核心识别 ──
    new_core = _locate_top_left_number_core(
        cropped_rgb, cropped_gray, ch, cw, None, PREFIXES
    )
    if new_core is None:
        print('  二次识别: 未找到，回退到第一轮结果')
        clipped_bbox = _find_cell_boundary(
            gray, ocr_bbox_orig, sub_h, sub_w,
            y_text_orig if y_text_orig else "UNKNOWN"
        )
        final_bbox_sub = clipped_bbox
        final_text = y_text_orig

        final_bbox = BBox(
            int(final_bbox_sub.x / orange_scale) + orange_search.x,
            int(final_bbox_sub.y / orange_scale) + orange_search.y,
            int(final_bbox_sub.w / orange_scale),
            int(final_bbox_sub.h / orange_scale),
        )
        vis_full = draw_bbox(img, final_bbox, (0, 165, 255),
                             f'ORANGE(fallback): {final_text or "?"}')
        save_img(f'{prefix}_13_fullimg_result.jpg', vis_full)
        return

    new_text, new_ocr = new_core
    print(f'  二次识别结果: text="{new_text}", bbox={new_ocr}')

    vis_pass2 = draw_bbox(cropped_rgb, new_ocr, (0, 0, 255),
                          f'Pass2: {new_text or "?"}')
    save_img(f'{prefix}_08_pass2_result.jpg', vis_pass2)

    # ── 09. Morph 线检测 ──
    vlines, hlines = _detect_morph_lines(cropped_gray)
    print(f'\n  Morph 线: {len(vlines)} 条竖线, {len(hlines)} 条横线')
    for i, (x, y1, y2) in enumerate(vlines):
        print(f'    V{i}: x={x}, y={y1}~{y2}, len={y2-y1}')
    for i, (y, x1, x2) in enumerate(hlines):
        print(f'    H{i}: y={y}, x={x1}~{x2}, len={x2-x1}')

    # morph 可视化
    thresh = _binarize_for_lines(cropped_gray)
    morph_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for x, y1, y2 in vlines:
        cv2.line(morph_vis, (x, y1), (x, y2), (0, 0, 255), 2)
    for y, x1, x2 in hlines:
        cv2.line(morph_vis, (x1, y), (x2, y), (255, 0, 0), 2)
    # 在 morph 图上也标注 OCR bbox
    cv2.rectangle(morph_vis, (new_ocr.x, new_ocr.y),
                  (new_ocr.x2, new_ocr.y2), (0, 255, 0), 2)
    save_img(f'{prefix}_09_morph_lines.jpg', morph_vis)

    # ── 10. Cell + OCR bbox + 橙框 对比 ──
    cell = _find_cell_from_lines(new_ocr, vlines, hlines, cw)
    orange_bbox = _make_green_bbox(new_ocr, cell)
    print(f'  Cell:   {cell}')
    print(f'  橙框:   {orange_bbox}')

    vis_final = draw_multi_bbox(cropped_rgb, [
        (cell, (255, 0, 0), 'Cell'),
        (new_ocr, (0, 0, 255), 'OCR'),
        (orange_bbox, (0, 165, 255), 'Orange'),
    ])
    # 图例
    legend_y = 25
    for label, color in [('Blue=Cell', (255, 0, 0)),
                         ('Red=OCR', (0, 0, 255)),
                         ('Orange=Final', (0, 165, 255))]:
        cv2.putText(vis_final, label, (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 25
    save_img(f'{prefix}_10_cell_ocr_orange.jpg', vis_final)

    # ── 11. 全图标注最终橙框 ──
    final_bbox_sub = BBox(orange_bbox.x + cx1, orange_bbox.y + cy1,
                          orange_bbox.w, orange_bbox.h)
    final_bbox = BBox(
        int(final_bbox_sub.x / orange_scale) + orange_search.x,
        int(final_bbox_sub.y / orange_scale) + orange_search.y,
        int(final_bbox_sub.w / orange_scale),
        int(final_bbox_sub.h / orange_scale),
    )
    print(f'  最终橙框(子图): {final_bbox_sub}')
    print(f'  最终橙框(原图): {final_bbox}')

    vis_full = draw_bbox(img, final_bbox, (0, 165, 255),
                         f'ORANGE: {new_text or "?"}')
    save_img(f'{prefix}_11_fullimg_result.jpg', vis_full)

    # ── 12. 同时跑 detect_all_regions 看完整检测结果 ──
    print(f'\n  ── detect_all_regions 完整检测 ──')
    enhanced = _enhance_vertical_lines(img)
    regions = detect_all_regions(enhanced, prefixes=PREFIXES)
    meta = regions.get('_metadata', {})

    orange_region = regions.get('top_left_number')
    orange_text_det = meta.get('top_left_text')
    green_region = regions.get('bottom_right_number')
    green_text_det = meta.get('bottom_right_text')
    red_region = regions.get('material_code_column')

    print(f'  红框: {red_region}')
    print(f'  绿框: {green_region}, text="{green_text_det}"')
    print(f'  橙框: {orange_region}, text="{orange_text_det}"')

    # 全区域标注图
    vis_all = img.copy()
    if red_region:
        vis_all = draw_bbox(vis_all, red_region, (0, 0, 255), 'RED')
    if green_region:
        vis_all = draw_bbox(vis_all, green_region, (0, 255, 0),
                            f'GREEN: {green_text_det or "?"}')
    if orange_region:
        vis_all = draw_bbox(vis_all, orange_region, (0, 165, 255),
                            f'ORANGE: {orange_text_det or "?"}')
    save_img(f'{prefix}_12_all_regions.jpg', vis_all)

    print(f'\n  诊断完成: {f.name}')


# ── 执行诊断 ──
for f in target_files:
    diagnose_file(f)

print("\n全部测试完成。")
