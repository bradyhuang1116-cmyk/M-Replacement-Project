"""方案 A：多尺度膨胀 × OCR 后码级去重

策略：
  1. 对二值化后的搜索区，并行跑 3 个不同膨胀核，各自连通域 → 候选块
  2. 三组候选块全部独立送 VLM OCR（不在块层去重）
  3. OCR 找到的所有 Y 编号，最后按 (code_text, bbox 中心距 < min(w,h)*0.5) 去重
"""
import os
import sys
import logging
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, make_pattern
from modules.file_ingestion import load_file
from modules.region_detector import (
    detect_all_regions, _enhance_vertical_lines, BBox,
    _ocr_region,
)

# ── 输入 ──
_default_dir = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
KEYWORDS = ["P3335", "A226", "C857", "A110-2"]

INPUT_FILES = []
for f in sorted(os.listdir(_default_dir)):
    if not os.path.splitext(f)[1].lower() in (".tif", ".tiff"):
        continue
    for kw in KEYWORDS:
        if kw in f:
            INPUT_FILES.append(os.path.join(_default_dir, f))
            break

if not INPUT_FILES:
    logger.error(f"未找到匹配文件: {KEYWORDS}")
    sys.exit(1)

logger.info(f"测试文件({len(INPUT_FILES)}): {[os.path.basename(f) for f in INPUT_FILES]}")

DEBUG_BASE = os.path.join(os.path.dirname(__file__), "test_output", "factory_note_test_A")
os.makedirs(DEBUG_BASE, exist_ok=True)

prefixes = DEFAULT_PREFIXES
p_chars = "".join(p.upper() for p in prefixes)
p_class = f"[{p_chars}]" if len(p_chars) > 1 else p_chars
y_code_loose = re.compile(rf"{p_class}[A-Z]\d[A-Z0-9]{{3,}}")

# 多尺度膨胀核（宽, 高）
KERNELS = [
    ("k50", (50, 50)),
    ("k60", (60, 60)),
]
MIN_AREA = 40000
FRAME_RATIO = 0.80
BORDER_MARGIN = 40

# ── VLM ──
from modules.docker_manager import ensure_vlm_ready
logger.info("启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 启动失败: {msg}")
    sys.exit(1)
logger.info(f"VLM 就绪: {msg}")


def _connected_blocks(bw, kernel_size, sw, sh):
    """对二值图做指定核膨胀 → 连通域 → 过滤 → 返回 [{x1,y1,x2,y2}, ...]"""
    kw, kh = kernel_size
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(bw, kern, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    blocks = []
    search_total = sw * sh
    for li in range(1, num_labels):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        w_l = int(stats[li, cv2.CC_STAT_WIDTH])
        h_l = int(stats[li, cv2.CC_STAT_HEIGHT])
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        bbox_area = w_l * h_l
        if bbox_area / max(search_total, 1) >= FRAME_RATIO:
            continue
        blocks.append({"x1": x, "y1": y, "x2": x + w_l, "y2": y + h_l})
    blocks.sort(key=lambda b: (b["y1"], b["x1"]))
    return dilated, blocks


def _bbox_center_dist(b1: BBox, b2: BBox) -> float:
    cx1 = b1.x + b1.w / 2
    cy1 = b1.y + b1.h / 2
    cx2 = b2.x + b2.w / 2
    cy2 = b2.y + b2.h / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _dedup_codes(found_codes):
    """码级去重：相同 code 文本 + bbox 中心距 < min(w,h)*0.5 → 视为同一编号。

    保留每组中第一个（按 found_codes 顺序）。
    """
    kept = []
    for fc in found_codes:
        is_dup = False
        for kp in kept:
            if kp["code"] != fc["code"]:
                continue
            ref_w = min(kp["bbox"].w, fc["bbox"].w)
            ref_h = min(kp["bbox"].h, fc["bbox"].h)
            threshold = min(ref_w, ref_h) * 0.5
            dist = _bbox_center_dist(kp["bbox"], fc["bbox"])
            if dist < threshold:
                is_dup = True
                kp.setdefault("dup_sources", []).append(fc.get("kernel", "?"))
                break
        if not is_dup:
            kept.append(fc)
    return kept


def _run_variant(variant, blocks, search_for_ocr, search_roi, fn_left, fn_top, out_dir, do_secondary):
    """对一组候选块跑一遍 OCR + 可选二次裁切，独立去重 + 独立输出。

    variant: 变体名（如 "k50"）—— 用于文件名前缀
    do_secondary: True=含Y的OCR做二次裁切放大；False=仅一次OCR
    """
    found_codes = []
    secondary_idx = 0

    # 画一遍候选块布局
    layout_vis = search_roi.copy()
    for bi, blk in enumerate(blocks):
        cv2.rectangle(layout_vis,
            (blk["x1"], blk["y1"]), (blk["x2"], blk["y2"]),
            (100, 255, 200), 3)
        cv2.putText(layout_vis, f"{variant[0]}{bi}",
            (blk["x1"]+5, blk["y1"]+30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 200), 2)
    Image.fromarray(layout_vis).save(
        os.path.join(out_dir, f"03_layout_{variant}.jpg"), quality=90)

    for ci, cand in enumerate(blocks):
        cx1, cy1, cx2, cy2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 10 or ch < 10:
            continue

        cand_roi = search_for_ocr[cy1:cy2, cx1:cx2].copy()
        Image.fromarray(cand_roi).save(
            os.path.join(out_dir, f"04_{variant}_cand{ci}_crop.jpg"), quality=90)

        ocr_results = _ocr_region(search_for_ocr, BBox(cx1, cy1, cw, ch), engine="vlm")

        ocr_vis = cand_roi.copy()
        for text, conf, poly in ocr_results:
            text_upper = text.upper().replace(" ", "")
            has_y = any(c in text_upper for c in p_chars)
            m = y_code_loose.search(text_upper)

            if poly and len(poly) >= 4:
                pts = np.array([
                    [int(p[0] - cx1), int(p[1] - cy1)]
                    for p in poly
                ], dtype=np.int32)
                if m:
                    color = (0, 255, 0)
                elif has_y:
                    color = (0, 200, 255)
                else:
                    color = (128, 128, 128)
                cv2.polylines(ocr_vis, [pts], True, color, 2)
                cv2.putText(ocr_vis, text.replace(" ", ""),
                    (max(pts[0][0], 0), max(pts[0][1] - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 含 Y 的二次OCR（仅当 do_secondary=True）
            if do_secondary and has_y and poly and len(poly) >= 4:
                poly_x1 = min(pt[0] for pt in poly)
                poly_y1 = min(pt[1] for pt in poly)
                poly_x2 = max(pt[0] for pt in poly)
                poly_y2 = max(pt[1] for pt in poly)
                sh2, sw2 = search_for_ocr.shape[:2]
                pw = max(int(poly_x2 - poly_x1), 1)
                ph = max(int(poly_y2 - poly_y1), 1)
                pad_x = int(pw * 0.20)
                pad_y = int(ph * 0.20)
                crop_x1 = max(int(poly_x1) - pad_x, 0)
                crop_y1 = max(int(poly_y1) - pad_y, 0)
                crop_x2 = min(int(poly_x2) + pad_x, sw2)
                crop_y2 = min(int(poly_y2) + pad_y, sh2)
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                if crop_w > 5 and crop_h > 5:
                    crop_roi = search_for_ocr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    short_edge = min(crop_w, crop_h)
                    if short_edge < 200:
                        zoom = min(4.0, 200.0 / short_edge)
                    else:
                        zoom = 1.0
                    if zoom > 1.01:
                        zoom_w = int(crop_w * zoom)
                        zoom_h = int(crop_h * zoom)
                        crop_zoomed = cv2.resize(crop_roi, (zoom_w, zoom_h),
                                                 interpolation=cv2.INTER_CUBIC)
                    else:
                        zoom_w, zoom_h = crop_w, crop_h
                        crop_zoomed = crop_roi

                    safe_label = re.sub(r'[\\/:*?"<>|,]', '_', text_upper[:20])
                    Image.fromarray(crop_zoomed).save(
                        os.path.join(out_dir,
                            f"06_secondary{secondary_idx}_{variant}_crop_1st_{safe_label}.jpg"),
                        quality=90)

                    ocr2 = _ocr_region(crop_zoomed, BBox(0, 0, zoom_w, zoom_h), engine="vlm")
                    crop_vis = crop_zoomed.copy()
                    for t2, c2, p2 in ocr2:
                        t2_upper = t2.upper().replace(" ", "")
                        m2 = y_code_loose.search(t2_upper)
                        if p2 and len(p2) >= 4:
                            pts2 = np.array([
                                [int(p[0]), int(p[1])]
                                for p in p2
                            ], dtype=np.int32)
                            c2_color = (0, 255, 0) if m2 else (128, 128, 128)
                            cv2.polylines(crop_vis, [pts2], True, c2_color, 2)
                            cv2.putText(crop_vis, t2.replace(" ", ""),
                                (max(pts2[0][0], 0), max(pts2[0][1] - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c2_color, 1)
                        if m2:
                            code2 = m2.group()
                            if p2 and len(p2) >= 4:
                                p2_x1 = min(pt[0] for pt in p2) / zoom
                                p2_y1 = min(pt[1] for pt in p2) / zoom
                                p2_x2 = max(pt[0] for pt in p2) / zoom
                                p2_y2 = max(pt[1] for pt in p2) / zoom
                                abs_x1 = int(crop_x1 + p2_x1)
                                abs_y1 = int(crop_y1 + p2_y1)
                                found_codes.append({
                                    "code": code2,
                                    "bbox": BBox(abs_x1 + fn_left, abs_y1 + fn_top,
                                                 int(p2_x2-p2_x1), int(p2_y2-p2_y1)),
                                    "source": "secondary",
                                    "kernel": variant,
                                })

                    Image.fromarray(crop_vis).save(
                        os.path.join(out_dir,
                            f"06_secondary{secondary_idx}_{variant}_ocr2.jpg"),
                        quality=90)
                    secondary_idx += 1

            if m:
                code = m.group()
                if poly and len(poly) >= 4:
                    poly_x1 = min(pt[0] for pt in poly)
                    poly_y1 = min(pt[1] for pt in poly)
                    poly_x2 = max(pt[0] for pt in poly)
                    poly_y2 = max(pt[1] for pt in poly)
                    poly_w = max(poly_x2 - poly_x1, 1)
                    text_len = max(len(text_upper), 1)
                    char_w = poly_w / text_len
                    code_bx1 = int(poly_x1 + m.start() * char_w)
                    code_bx2 = int(poly_x1 + m.end() * char_w)
                    found_codes.append({
                        "code": code,
                        "bbox": BBox(code_bx1 + fn_left, int(poly_y1) + fn_top,
                                     code_bx2 - code_bx1, int(poly_y2 - poly_y1)),
                        "source": "primary",
                        "kernel": variant,
                    })

        Image.fromarray(ocr_vis).save(
            os.path.join(out_dir, f"05_{variant}_cand{ci}_ocr.jpg"), quality=90)

    # 独立去重
    before_dedup = len(found_codes)
    deduped = _dedup_codes(found_codes)
    logger.info(f"  [{variant}] 去重前 {before_dedup} → 去重后 {len(deduped)}")

    # 独立最终汇总图
    final_vis = search_roi.copy()
    for fc in deduped:
        b = fc["bbox"]
        bx1 = b.x - fn_left
        by1 = b.y - fn_top
        bx2 = b.x2 - fn_left
        by2 = b.y2 - fn_top
        color = (0, 255, 255) if fc.get("source") == "primary" else (0, 255, 0)
        label = f"{fc['code']} [{fc.get('kernel','?')}{'/2nd' if fc.get('source')=='secondary' else ''}]"
        cv2.rectangle(final_vis, (bx1, by1), (bx2, by2), color, 3)
        cv2.putText(final_vis, label, (bx1, by1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    Image.fromarray(final_vis).save(
        os.path.join(out_dir, f"07_final_{variant}.jpg"), quality=90)

    return found_codes, deduped, before_dedup


for file_idx, INPUT_FILE in enumerate(INPUT_FILES):
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(DEBUG_BASE, stem)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"[{file_idx+1}/{len(INPUT_FILES)}] {os.path.basename(INPUT_FILE)}")

    try:
        # Step 1: 加载 + 区域检测
        img_array, meta = load_file(INPUT_FILE)
        enhanced = _enhance_vertical_lines(img_array)
        regions = detect_all_regions(enhanced, prefixes=prefixes)

        rot_code = regions.get("_metadata", {}).get("rotation")
        if rot_code is not None:
            img_array = cv2.rotate(img_array, rot_code)
            enhanced = cv2.rotate(enhanced, rot_code)

        img_h, img_w = img_array.shape[:2]

        red_bbox = regions.get("material_code_column")
        green_bbox = regions.get("bottom_right_number")
        orange_bbox = regions.get("top_left_number")

        logger.info(f"  红框: {red_bbox}")
        logger.info(f"  绿框: {green_bbox}")
        logger.info(f"  橙框: {orange_bbox}")

        # Step 2: 搜索区
        fn_top = 0
        fn_bottom = green_bbox.y if green_bbox else img_h
        fn_left = orange_bbox.x if orange_bbox else 0
        fn_right = img_w
        fn_search = BBox(fn_left, fn_top, fn_right - fn_left, fn_bottom - fn_top)
        logger.info(f"  工厂注意搜索区: {fn_search}")

        search_roi = img_array[fn_top:fn_bottom, fn_left:fn_right].copy()
        Image.fromarray(search_roi).save(os.path.join(out_dir, "01_search_area.jpg"), quality=90)

        # Step 3: 白色遮盖红/绿/橙框
        search_for_ocr = search_roi.copy()
        exclude_bboxes = []
        for name, bbox in [("red", red_bbox), ("green", green_bbox), ("orange", orange_bbox)]:
            if bbox is None:
                continue
            mx1 = max(bbox.x - fn_left, 0)
            my1 = max(bbox.y - fn_top, 0)
            mx2 = min(bbox.x2 - fn_left, fn_right - fn_left)
            my2 = min(bbox.y2 - fn_top, fn_bottom - fn_top)
            if mx2 > mx1 and my2 > my1:
                cv2.rectangle(search_for_ocr, (mx1, my1), (mx2, my2), (255, 255, 255), -1)
                exclude_bboxes.append((bbox.x, bbox.y, bbox.x2, bbox.y2))
        Image.fromarray(search_for_ocr).save(os.path.join(out_dir, "03b_search_masked.jpg"), quality=90)

        # Step 4: 二值化
        sh, sw = search_for_ocr.shape[:2]
        gray_search = cv2.cvtColor(search_for_ocr, cv2.COLOR_RGB2GRAY)
        _, bw_search = cv2.threshold(gray_search, 200, 255, cv2.THRESH_BINARY_INV)
        bw_search[:BORDER_MARGIN, :] = 0
        bw_search[sh-BORDER_MARGIN:, :] = 0
        bw_search[:, :BORDER_MARGIN] = 0
        bw_search[:, sw-BORDER_MARGIN:] = 0
        Image.fromarray(bw_search).save(os.path.join(out_dir, "03c_binary.jpg"), quality=90)

        # Step 5: 多尺度膨胀 → 各自连通域（仅为可视化保留 dilated 图）
        all_blocks_per_kernel = {}
        for kname, ksize in KERNELS:
            dilated, blocks = _connected_blocks(bw_search, ksize, sw, sh)
            Image.fromarray(dilated).save(
                os.path.join(out_dir, f"03d_dilated_{kname}.jpg"), quality=90)
            all_blocks_per_kernel[kname] = blocks
            logger.info(f"  核[{kname} {ksize}]: {len(blocks)} 块")

        # Step 6: 每种变体独立跑一遍 pipeline（do_secondary 配置见下）
        variant_results = {}
        for kname, _ in KERNELS:
            logger.info(f"  ── 变体 {kname} ──")
            variant_results[kname] = _run_variant(
                kname, all_blocks_per_kernel[kname],
                search_for_ocr, search_roi, fn_left, fn_top, out_dir,
                do_secondary=True,
            )

        # Step 7: 每个变体独立写 summary 文件
        for vname, (raw, deduped, raw_count) in variant_results.items():
            with open(os.path.join(out_dir, f"summary_{vname}.txt"), "w", encoding="utf-8") as f:
                f.write(f"file: {os.path.basename(INPUT_FILE)}\n")
                f.write(f"variant: {vname}\n")
                f.write(f"raw_codes: {raw_count}\n")
                f.write(f"deduped: {len(deduped)}\n\n")
                for fc in deduped:
                    b = fc["bbox"]
                    f.write(f"  {fc['code']:20s} bbox=({b.x},{b.y},{b.w},{b.h}) "
                            f"source={fc.get('source')} kernel={fc.get('kernel')}\n")
            logger.info(f"  [{vname}] 最终 {len(deduped)} 个编号 → summary_{vname}.txt")
            for fc in deduped:
                logger.info(f"      {fc['source']}/{fc.get('kernel')}: {fc['code']} @ {fc['bbox']}")

        logger.info(f"  调试图保存: {out_dir}")

    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)

logger.info("=" * 60)
logger.info(f"测试完成，输出: {DEBUG_BASE}")
