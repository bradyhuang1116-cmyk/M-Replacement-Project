"""测试工厂注意区域检测：新搜索区规则 + OCR可视化"""
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

DEBUG_BASE = os.path.join(os.path.dirname(__file__), "test_output", "factory_note_test")
os.makedirs(DEBUG_BASE, exist_ok=True)

prefixes = DEFAULT_PREFIXES
p_chars = "".join(p.upper() for p in prefixes)
p_class = f"[{p_chars}]" if len(p_chars) > 1 else p_chars
y_code_loose = re.compile(rf"{p_class}[A-Z]\d[A-Z0-9]{{3,}}")

# ── VLM ──
from modules.docker_manager import ensure_vlm_ready
logger.info("启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 启动失败: {msg}")
    sys.exit(1)
logger.info(f"VLM 就绪: {msg}")

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

        # Step 2: 新搜索区规则
        # 上=图片上边界, 下=绿框上边界, 左=橙框左边界, 右=图片右边界
        fn_top = 0
        fn_bottom = green_bbox.y if green_bbox else img_h
        fn_left = orange_bbox.x if orange_bbox else 0
        fn_right = img_w
        fn_search = BBox(fn_left, fn_top, fn_right - fn_left, fn_bottom - fn_top)

        logger.info(f"  工厂注意搜索区: {fn_search}")

        # Step 3: 保存搜索区裁切图
        search_roi = img_array[fn_top:fn_bottom, fn_left:fn_right].copy()
        Image.fromarray(search_roi).save(
            os.path.join(out_dir, "01_search_area.jpg"), quality=90)

        # 在搜索区上标注排除区域（红/绿/橙框）
        search_vis = search_roi.copy()
        exclude_bboxes = []
        for name, bbox, color in [
            ("red", red_bbox, (0, 0, 255)),
            ("green", green_bbox, (0, 200, 0)),
            ("orange", orange_bbox, (255, 165, 0)),
        ]:
            if bbox is None:
                continue
            # 映射到搜索区坐标
            rx1 = max(bbox.x - fn_left, 0)
            ry1 = max(bbox.y - fn_top, 0)
            rx2 = min(bbox.x2 - fn_left, fn_right - fn_left)
            ry2 = min(bbox.y2 - fn_top, fn_bottom - fn_top)
            if rx2 > rx1 and ry2 > ry1:
                cv2.rectangle(search_vis, (rx1, ry1), (rx2, ry2), color, 4)
                cv2.putText(search_vis, name, (rx1+5, ry1+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                exclude_bboxes.append((bbox.x, bbox.y, bbox.x2, bbox.y2))
        Image.fromarray(search_vis).save(
            os.path.join(out_dir, "02_search_with_excludes.jpg"), quality=90)

        # Step 4: 白色遮盖红/绿/橙框区域（分割和OCR都在遮盖后的图上进行）
        search_for_ocr = search_roi.copy()
        for eb in exclude_bboxes:
            ex1, ey1, ex2, ey2 = eb
            mx1 = max(ex1 - fn_left, 0)
            my1 = max(ey1 - fn_top, 0)
            mx2 = min(ex2 - fn_left, fn_right - fn_left)
            my2 = min(ey2 - fn_top, fn_bottom - fn_top)
            if mx2 > mx1 and my2 > my1:
                cv2.rectangle(search_for_ocr, (mx1, my1), (mx2, my2), (255, 255, 255), -1)
                logger.info(f"  白色遮盖: ({mx1},{my1})-({mx2},{my2})")
        Image.fromarray(search_for_ocr).save(
            os.path.join(out_dir, "03b_search_masked.jpg"), quality=90)

        # Step 5: 二值化 → 强膨胀 → 连通域 → 块（单一方法，不分类型）
        sh, sw = search_for_ocr.shape[:2]
        gray_search = cv2.cvtColor(search_for_ocr, cv2.COLOR_RGB2GRAY)
        _, bw_search = cv2.threshold(gray_search, 200, 255, cv2.THRESH_BINARY_INV)

        # 边框遮黑
        BORDER_MARGIN = 40
        bw_search[:BORDER_MARGIN, :] = 0
        bw_search[sh-BORDER_MARGIN:, :] = 0
        bw_search[:, :BORDER_MARGIN] = 0
        bw_search[:, sw-BORDER_MARGIN:] = 0
        Image.fromarray(bw_search).save(
            os.path.join(out_dir, "03c_binary.jpg"), quality=90)

        # 膨胀（40x40 让密集文字块完整合并）
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        dilated = cv2.dilate(bw_search, kern, iterations=1)
        Image.fromarray(dilated).save(
            os.path.join(out_dir, "03d_dilated.jpg"), quality=90)

        # 连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        MIN_AREA = 50000
        search_total = sw * sh
        FRAME_RATIO = 0.80  # bbox 面积占搜索区比例 ≥ 此值 → 视为图纸边框
        all_blocks = []
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
                logger.info(f"  剔除边框块({x},{y})-({x+w_l},{y+h_l}): bbox面积比={bbox_area/search_total:.2%}")
                continue
            all_blocks.append({"x1": x, "y1": y, "x2": x + w_l, "y2": y + h_l})

        all_blocks.sort(key=lambda b: (b["y1"], b["x1"]))

        layout_vis = search_roi.copy()
        for bi, blk in enumerate(all_blocks):
            cv2.rectangle(layout_vis,
                (blk["x1"], blk["y1"]), (blk["x2"], blk["y2"]),
                (100, 255, 200), 3)
            cv2.putText(layout_vis, str(bi), (blk["x1"]+5, blk["y1"]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 200), 2)
        Image.fromarray(layout_vis).save(
            os.path.join(out_dir, "03_layout_blocks.jpg"), quality=90)

        logger.info(f"  连通域分块: {len(all_blocks)} 个 (面积阈值={MIN_AREA})")
        for bi, blk in enumerate(all_blocks):
            bw = blk["x2"] - blk["x1"]
            bh = blk["y2"] - blk["y1"]
            logger.info(f"    块[{bi}]: size={bw}x{bh}, "
                        f"pos=({blk['x1']},{blk['y1']})-({blk['x2']},{blk['y2']})")
        Image.fromarray(layout_vis).save(
            os.path.join(out_dir, "03_layout_blocks.jpg"), quality=90)

        # 所有布局块都作为候选（不过滤面积，不过滤重叠——重叠区域已被白色遮盖）
        candidates = all_blocks
        logger.info(f"  候选: {len(candidates)} 个（全部保留）")

        # Step 6: 对每个候选 OCR + 可视化 + 含Y编号的二次OCR
        found_codes = []
        secondary_idx = 0
        for ci, cand in enumerate(candidates):
            cx1, cy1, cx2, cy2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw < 10 or ch < 10:
                continue

            # 裁切候选区域（从遮盖后的搜索区）
            cand_roi = search_for_ocr[cy1:cy2, cx1:cx2].copy()
            Image.fromarray(cand_roi).save(
                os.path.join(out_dir, f"04_cand{ci}_crop.jpg"), quality=90)

            # 第一次OCR（VLM）
            abs_bbox = BBox(fn_left + cx1, fn_top + cy1, cw, ch)
            ocr_results = _ocr_region(search_for_ocr, BBox(cx1, cy1, cw, ch), engine="vlm")

            # 可视化第一次OCR结果
            ocr_vis = cand_roi.copy()
            has_code = False
            for text, conf, poly in ocr_results:
                text_upper = text.upper().replace(" ", "")
                has_y = any(c in text_upper for c in p_chars)
                m = y_code_loose.search(text_upper)

                # poly 在搜索区坐标系（_ocr_region 基于 search_for_ocr）
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

                # 含Y/X字母的OCR框：裁切（20% padding）+ 像素放大 → 二次OCR
                if has_y and poly and len(poly) >= 4:
                    poly_x1 = min(pt[0] for pt in poly)
                    poly_y1 = min(pt[1] for pt in poly)
                    poly_x2 = max(pt[0] for pt in poly)
                    poly_y2 = max(pt[1] for pt in poly)
                    sh, sw = search_for_ocr.shape[:2]
                    pw = max(int(poly_x2 - poly_x1), 1)
                    ph = max(int(poly_y2 - poly_y1), 1)
                    pad_x = int(pw * 0.20)
                    pad_y = int(ph * 0.20)
                    crop_x1 = max(int(poly_x1) - pad_x, 0)
                    crop_y1 = max(int(poly_y1) - pad_y, 0)
                    crop_x2 = min(int(poly_x2) + pad_x, sw)
                    crop_y2 = min(int(poly_y2) + pad_y, sh)
                    crop_w = crop_x2 - crop_x1
                    crop_h = crop_y2 - crop_y1
                    if crop_w > 5 and crop_h > 5:
                        crop_roi = search_for_ocr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        # 放大 (短边 ≥ 200，最大 4×)
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
                                f"06_secondary{secondary_idx}_crop_1st_{safe_label}.jpg"),
                            quality=90)
                        logger.info(f"    含Y框二次OCR: bbox={crop_w}x{crop_h}, zoom={zoom:.2f}x → {zoom_w}x{zoom_h}")

                        # 在放大后的图像上做二次OCR（VLM）
                        ocr2 = _ocr_region(crop_zoomed, BBox(0, 0, zoom_w, zoom_h), engine="vlm")
                        logger.info(f"    候选{ci} 二次OCR ('{text_upper[:30]}'):")

                        crop_vis = crop_zoomed.copy()
                        for t2, c2, p2 in ocr2:
                            t2_upper = t2.upper().replace(" ", "")
                            m2 = y_code_loose.search(t2_upper)
                            logger.info(f"      → '{t2}' (conf={c2:.2f}) {'✓匹配' if m2 else ''}")
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
                                    # p2 在 zoom 图坐标系，需还原到搜索区坐标 → 图片绝对坐标
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
                                    })

                        Image.fromarray(crop_vis).save(
                            os.path.join(out_dir,
                                f"06_secondary{secondary_idx}_ocr2.jpg"),
                            quality=90)
                        secondary_idx += 1

                if m:
                    has_code = True
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
                        })
                    logger.info(f"    候选{ci}: 发现 {code} (conf={conf:.2f})")

            Image.fromarray(ocr_vis).save(
                os.path.join(out_dir, f"05_cand{ci}_ocr.jpg"), quality=90)

            if not has_code:
                logger.info(f"    候选{ci}: 无编号")

        primary = [fc for fc in found_codes if fc.get("source") == "primary"]
        secondary = [fc for fc in found_codes if fc.get("source") == "secondary"]
        logger.info(f"  共发现 {len(found_codes)} 个编号 (第一次={len(primary)}, 二次={len(secondary)})")
        for fc in found_codes:
            logger.info(f"    {fc['source']}: {fc['code']} @ {fc['bbox']}")

        # Step 7: 最终汇总图（在原图搜索区上标注所有发现的编号）
        final_vis = search_roi.copy()
        for fc in found_codes:
            b = fc["bbox"]
            bx1 = b.x - fn_left
            by1 = b.y - fn_top
            bx2 = b.x2 - fn_left
            by2 = b.y2 - fn_top
            color = (0, 255, 255) if fc.get("source") == "primary" else (0, 255, 0)
            label = fc["code"] + (" [2nd]" if fc.get("source") == "secondary" else "")
            cv2.rectangle(final_vis, (bx1, by1), (bx2, by2), color, 3)
            cv2.putText(final_vis, label,
                (bx1, by1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        Image.fromarray(final_vis).save(
            os.path.join(out_dir, "07_final_codes.jpg"), quality=90)

        logger.info(f"  调试图保存: {out_dir}")

    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)

logger.info("=" * 60)
logger.info(f"测试完成，输出: {DEBUG_BASE}")
