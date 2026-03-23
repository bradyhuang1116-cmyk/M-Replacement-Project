"""青框检测改进诊断 — 投影法 + 网格步进分类 + 重叠分片 OCR

核心思路：
  1. 投影法检测红框内所有水平线（含行线+删除线）
  2. 确定单元格高度 → 从第一条线开始网格步进（+cell_h ±3px）
     匹配到的是行线，其余是删除线
  3. 仅用行线做分片裁切（重叠2行）
  4. 用已知删除线坐标直接生成精确 mask → inpaint 修复
  5. 重叠区域 OCR 结果去重

用法:
    python test_cyan_chunk.py
"""

import os
import sys
import time
import logging
from collections import Counter

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cyan_chunk")

# 让 region_detector 输出 DEBUG 日志，用于分析竖线检测
logging.getLogger("modules.region_detector").setLevel(logging.DEBUG)

import cv2
import numpy as np

from modules.region_detector import BBox, _crop_and_scale, _map_bbox_back, _enhance_vertical_lines
from modules.text_replacer import _get_ocr, _parse_ocr_results
from modules.file_ingestion import load_file
from config import OCR_LANG_EN, DEFAULT_PREFIXES

# ── 配置 ──
TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA026D941_0d.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "cyan_chunk")
CHUNK_SIZE = 10      # 每片包含的单元格数
OVERLAP_ROWS = 2     # 重叠行数
GRID_TOLERANCE = 3   # 网格步进容差 (px)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_img(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  保存: {name} ({img.shape[1]}x{img.shape[0]})")


# ── 自适应二值化预处理 ──
def preprocess_for_table(gray):
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.medianBlur(binary, 3)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    return binary


# ── 水平投影法检测所有水平线 ──
def detect_all_horizontal_lines(binary, min_line_ratio=0.8):
    """投影法 — 每行白像素占比超过阈值 → 聚类取中心。"""
    row_sum = np.sum(binary, axis=1) / 255.0
    threshold = min_line_ratio * binary.shape[1]
    line_positions = np.where(row_sum > threshold)[0]

    if len(line_positions) == 0:
        return []

    lines = []
    current = [int(line_positions[0])]
    for y in line_positions[1:]:
        if y - current[-1] < 10:
            current.append(int(y))
        else:
            lines.append(int(np.mean(current)))
            current = [int(y)]
    lines.append(int(np.mean(current)))
    return sorted(lines)


# ── 网格步进：区分行线 vs 删除线 ──
def classify_lines_by_grid(all_lines, cell_height, tolerance=3):
    """从第一条线开始，期望下一条行线在 +cell_height ±tolerance。
    匹配到的是行线，其余是删除线。

    Returns:
        (table_lines, strike_lines) — 两个 sorted list
    """
    if not all_lines:
        return [], []

    table_lines = [all_lines[0]]
    used = {0}
    cur = 0

    while True:
        expected = all_lines[cur] + cell_height
        best_idx = None
        best_dist = float('inf')
        for j in range(cur + 1, len(all_lines)):
            d = abs(all_lines[j] - expected)
            if d < best_dist:
                best_idx = j
                best_dist = d
            # 超过期望位置太远，不用继续找
            if all_lines[j] > expected + tolerance:
                break

        if best_idx is not None and best_dist <= tolerance:
            table_lines.append(all_lines[best_idx])
            used.add(best_idx)
            cur = best_idx
        else:
            break

    strike_lines = [all_lines[i] for i in range(len(all_lines)) if i not in used]
    return sorted(table_lines), sorted(strike_lines)


# ── 从已知删除线坐标生成精确 mask ──
def make_strike_mask(chunk_gray, strike_ys_local, band_half=6):
    """在已知删除线 y 坐标附近提取水平墨迹像素作为 mask。"""
    h, w = chunk_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not strike_ys_local:
        return mask

    _, thresh = cv2.threshold(chunk_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kw = max(int(w * 0.25), 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))

    for sy in strike_ys_local:
        y_top = max(0, sy - band_half)
        y_bot = min(h, sy + band_half + 1)
        band = thresh[y_top:y_bot, :]
        # 形态学 open 只保留水平连续像素（删除线墨迹）
        h_only = cv2.morphologyEx(band, cv2.MORPH_OPEN, h_kernel)
        mask[y_top:y_bot, :] = h_only

    return mask


def main():
    logger.info(f"测试图片: {TEST_IMAGE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    if not os.path.isfile(TEST_IMAGE):
        logger.error(f"文件不存在: {TEST_IMAGE}")
        sys.exit(1)

    # ── Step 1: 加载图片 + 检测红框 ──
    logger.info("=" * 60)
    logger.info("Step 1: 加载 + 红框检测")
    logger.info("=" * 60)

    image, _ = load_file(TEST_IMAGE)
    img_h, img_w = image.shape[:2]
    logger.info(f"图片: {img_w}x{img_h}")

    # 竖线增强预处理
    enhanced = _enhance_vertical_lines(image)
    logger.info(f"竖线增强预处理完成")

    from modules.region_detector import _locate_material_code_column
    frame = BBox(0, 0, img_w, img_h)
    red_search = BBox(frame.x, frame.y, frame.w // 2, frame.h)
    logger.info(f"红框搜索区: {red_search}")
    red_sub, red_scale = _crop_and_scale(enhanced, red_search)
    logger.info(f"红框搜索区缩放: scale={red_scale:.4f}, sub_size={red_sub.shape[1]}x{red_sub.shape[0]}")

    # 保存红框搜索区截图
    save_img("step1_red_search_area.jpg", red_sub)

    # 对红框搜索区做 OCR 并输出识别框
    logger.info("对红框搜索区表头做 OCR...")
    header_h = max(int(red_sub.shape[0] * 0.2), min(100, red_sub.shape[0]))
    header_roi = red_sub[:header_h, :]
    ocr = _get_ocr(OCR_LANG_EN)
    result = ocr.predict(header_roi)
    header_items = _parse_ocr_results(result)
    logger.info(f"  表头区域 OCR: {len(header_items)} 项")

    vis_header = header_roi.copy()
    for i, (poly, text, score) in enumerate(header_items):
        if poly is None:
            continue
        pts = np.array(poly, dtype=np.int32)
        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
        cv2.polylines(vis_header, [pts], True, (0, 0, 255), 2)
        label = f"[{i}] {text} ({score:.2f})"
        cv2.putText(vis_header, label, (x_min, max(y_min - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        logger.info(f"    [{i:2d}] '{text}' score={score:.3f} bbox=({x_min},{y_min},{x_max-x_min},{y_max-y_min})")
    save_img("step1_header_ocr.jpg", vis_header)

    mat_result = _locate_material_code_column(red_sub)
    if mat_result[0] is None:
        logger.error("未检测到 MATERIAL CODE 列")
        sys.exit(1)

    mat_bbox_sub, direction = mat_result
    logger.info(f"红框(子图内): {mat_bbox_sub}, direction={direction}")
    mat_bbox = _map_bbox_back(mat_bbox_sub, red_search, red_scale)
    logger.info(f"红框(原图): {mat_bbox}")

    # 保存红框区域截图 + 在搜索区上标注红框位置
    vis_search = red_sub.copy()
    rx = int(mat_bbox_sub.x)
    ry = int(mat_bbox_sub.y)
    rw = int(mat_bbox_sub.w)
    rh = int(mat_bbox_sub.h)
    cv2.rectangle(vis_search, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 3)
    cv2.putText(vis_search, "MATERIAL CODE column", (rx + 5, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    save_img("step1_red_detected.jpg", vis_search)

    # ── 竖线检测诊断：在红框附近区域独立检测竖线并可视化 ──
    from modules.region_detector import _detect_vertical_lines, _detect_vlines_by_projection
    logger.info("=" * 40)
    logger.info("竖线检测诊断")
    logger.info("=" * 40)

    # 在红框子图中，取表头附近区域检测竖线
    sub_gray = cv2.cvtColor(red_sub, cv2.COLOR_RGB2GRAY)
    sub_h, sub_w = sub_gray.shape[:2]

    # 模拟 _trace_vertical_table 的窄带检测
    kw_poly = None
    # 从 header OCR 中找 MATERIAL CODE 关键词位置
    for i, (poly, text, score) in enumerate(header_items):
        if poly is None:
            continue
        text_up = text.upper().replace(" ", "")
        if "MATERIAL" in text_up or "CODE" in text_up:
            kw_poly = poly
            logger.info(f"  关键词 '{text}' poly={poly}")

    if kw_poly is not None:
        kw_cx = sum(p[0] for p in kw_poly) / len(kw_poly)
        kw_cy = sum(p[1] for p in kw_poly) / len(kw_poly)
        logger.info(f"  kw_cx={kw_cx:.0f}, kw_cy={kw_cy:.0f}")

        # 窄带竖线检测（复制 _trace_vertical_table 步骤②逻辑）
        x_search_half = max(400, int(sub_w * 0.15))
        x_lo = max(0, int(kw_cx - x_search_half))
        x_hi = min(sub_w, int(kw_cx + x_search_half))
        band_half = max(100, int(sub_h * 0.05))
        band_top = max(0, int(kw_cy - band_half))
        band_bot = min(sub_h, int(kw_cy + band_half))
        band = sub_gray[band_top:band_bot, x_lo:x_hi]
        band_h = band.shape[0]

        v_min_h_band = max(int(band_h * 0.3), 15)
        v_lines_band, _, _ = _detect_vertical_lines(band, min_line_height=v_min_h_band)
        v_lines_band_abs = [vx + x_lo for vx in v_lines_band]
        logger.info(f"  窄带竖线 (x=[{x_lo},{x_hi}], y=[{band_top},{band_bot}], min_h={v_min_h_band}): {v_lines_band_abs}")

        # 可视化窄带 + 竖线
        vis_band = cv2.cvtColor(band, cv2.COLOR_GRAY2RGB)
        for vx in v_lines_band:
            cv2.line(vis_band, (vx, 0), (vx, band_h), (0, 0, 255), 2)
        save_img("step1_vlines_band.jpg", vis_band)

        # 在搜索区全图上标注竖线位置
        vis_vlines = red_sub.copy()
        # 标注窄带区域
        cv2.rectangle(vis_vlines, (x_lo, band_top), (x_hi, band_bot), (255, 255, 0), 2)
        for vx in v_lines_band_abs:
            cv2.line(vis_vlines, (vx, 0), (vx, sub_h), (0, 0, 255), 1)
            cv2.putText(vis_vlines, f"x={vx}", (vx + 3, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # 标红框位置
        cv2.rectangle(vis_vlines, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        save_img("step1_vlines_overview.jpg", vis_vlines)

        # 扩展区域竖线检测
        ext_top = max(0, int(kw_cy - 50))
        ext_bot = min(sub_h, int(kw_cy + int(sub_h * 0.3)))
        ext_band = sub_gray[ext_top:ext_bot, x_lo:x_hi]
        ext_h = ext_band.shape[0]
        v_min_h_ext = max(int(ext_h * 0.1), 20)
        v_lines_ext, _, _ = _detect_vertical_lines(ext_band, min_line_height=v_min_h_ext)
        v_lines_ext_abs = [vx + x_lo for vx in v_lines_ext]
        logger.info(f"  扩展竖线 (x=[{x_lo},{x_hi}], y=[{ext_top},{ext_bot}], min_h={v_min_h_ext}): {v_lines_ext_abs}")

        # 投影法竖线
        proj_band = sub_gray[ext_top:ext_bot, x_lo:x_hi]
        proj_lines = _detect_vlines_by_projection(proj_band)
        proj_lines_abs = [px + x_lo for px in proj_lines]
        logger.info(f"  投影法竖线 (x=[{x_lo},{x_hi}], y=[{ext_top},{ext_bot}]): {proj_lines_abs}")

        # 找到 kw_cx 落在哪两条竖线之间
        all_vl = sorted(set(v_lines_band_abs + v_lines_ext_abs + proj_lines_abs))
        logger.info(f"  所有竖线(合并): {all_vl}")
        for i in range(len(all_vl) - 1):
            if all_vl[i] <= kw_cx <= all_vl[i + 1]:
                logger.info(f"  kw_cx={kw_cx:.0f} 落在 [{all_vl[i]}, {all_vl[i+1]}] 之间, 列宽={all_vl[i+1]-all_vl[i]}px")
                break

    # ── Step 2: 投影法检测所有线 → 网格分类 → 分片 ──
    logger.info("=" * 60)
    logger.info("Step 2: 投影法 + 网格分类 + 分片")
    logger.info("=" * 60)

    box_roi = enhanced[mat_bbox.y:mat_bbox.y2, mat_bbox.x:mat_bbox.x2]
    box_gray = cv2.cvtColor(box_roi, cv2.COLOR_RGB2GRAY)
    box_h, box_w = box_gray.shape[:2]
    logger.info(f"红框区域: {box_w}x{box_h}")

    # 对红框全区域做 OCR 并输出识别框图
    logger.info("对红框全区域做 OCR（原图）...")
    t0 = time.perf_counter()
    ocr = _get_ocr(OCR_LANG_EN)
    result_full = ocr.predict(box_roi)
    items_full = _parse_ocr_results(result_full)
    ocr_full_time = time.perf_counter() - t0
    logger.info(f"  红框全区域 OCR: {len(items_full)} 项, 耗时 {ocr_full_time:.2f}s")

    vis_red_ocr = box_roi.copy()
    for i, (poly, text, score) in enumerate(items_full):
        if poly is None:
            continue
        pts = np.array(poly, dtype=np.int32)
        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
        cv2.polylines(vis_red_ocr, [pts], True, (0, 0, 255), 1)
        label = f"[{i}]{text[:12]}({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(vis_red_ocr, (x_min, y_min - th - 4), (x_min + tw + 2, y_min), (0, 0, 255), -1)
        cv2.putText(vis_red_ocr, label, (x_min + 1, y_min - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        logger.info(f"    [{i:2d}] '{text}' score={score:.3f} bbox=({x_min},{y_min},{x_max-x_min},{y_max-y_min})")
    save_img("step2_red_roi_ocr.jpg", vis_red_ocr)

    binary = preprocess_for_table(box_gray)
    save_img("step2_binary.jpg", binary)

    all_lines = detect_all_horizontal_lines(binary, min_line_ratio=0.8)
    logger.info(f"投影法检测到水平线: {len(all_lines)} 条")

    if len(all_lines) < 2:
        logger.error("检测到的水平线不足")
        sys.exit(1)

    # 确定单元格高度：取所有间距中 > 50px 的众数（四舍五入到5px）
    gaps = [all_lines[i + 1] - all_lines[i] for i in range(len(all_lines) - 1)]
    large_gaps = [g for g in gaps if g > 50]
    if not large_gaps:
        logger.error("无有效大间距")
        sys.exit(1)
    rounded = [round(g / 5) * 5 for g in large_gaps]
    cell_height = Counter(rounded).most_common(1)[0][0]
    logger.info(f"单元格高度: {cell_height}px (众数, 来自 {len(large_gaps)} 个大间距)")

    # 网格步进分类
    table_lines, strike_lines = classify_lines_by_grid(
        all_lines, cell_height, tolerance=GRID_TOLERANCE
    )
    logger.info(f"行线: {len(table_lines)} 条, 删除线: {len(strike_lines)} 条")

    # 可视化：绿=行线，红=删除线
    vis_classify = cv2.cvtColor(box_gray, cv2.COLOR_GRAY2RGB)
    for y in table_lines:
        cv2.line(vis_classify, (0, y), (box_w, y), (0, 255, 0), 1)
    for y in strike_lines:
        cv2.line(vis_classify, (0, y), (box_w, y), (255, 0, 0), 2)
    save_img("step2_classified_lines.jpg", vis_classify)

    if len(strike_lines) > 0:
        logger.info(f"删除线 y 坐标: {strike_lines}")

    # 分片（仅用行线，带重叠）
    chunk_step = max(CHUNK_SIZE - OVERLAP_ROWS, 1)
    chunks = []  # (chunk_top, chunk_bot, [行线列表])
    n_tl = len(table_lines)
    for i in range(0, n_tl, chunk_step):
        end_idx = i + CHUNK_SIZE
        chunk_top = table_lines[i]
        if end_idx < n_tl:
            chunk_bot = table_lines[end_idx]
        else:
            chunk_bot = min(table_lines[-1] + cell_height, box_h)
        lines_in_chunk = table_lines[i:min(end_idx, n_tl)]
        chunks.append((chunk_top, chunk_bot, lines_in_chunk))
        if end_idx >= n_tl:
            break

    logger.info(f"分片: {len(chunks)} 片 (每片{CHUNK_SIZE}行, 重叠{OVERLAP_ROWS}行, 步进{chunk_step})")
    for ci, (yt, yb, lns) in enumerate(chunks):
        logger.info(f"  片[{ci}] y=[{yt},{yb}] h={yb - yt}px 行线={len(lns)}")

    # ── 每片：删除线 mask + inpaint + 可视化 ──
    chunks_with_strike = set()
    chunk_strike_masks = {}
    total_strike_count = 0

    for ci, (chunk_top, chunk_bot, chunk_tbl_lines) in enumerate(chunks):
        chunk_roi = box_roi[chunk_top:chunk_bot, :]
        ch_h, ch_w = chunk_roi.shape[:2]

        # 原始分片截图（黄线=行线）
        vis = chunk_roi.copy()
        for ly in chunk_tbl_lines:
            local_y = ly - chunk_top
            if 0 <= local_y < ch_h:
                cv2.line(vis, (0, local_y), (ch_w, local_y), (255, 255, 0), 1)
        cv2.putText(vis, f"chunk {ci}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        save_img(f"chunk_{ci:02d}_raw.jpg", vis)

        if ch_h < 10 or ch_w < 10:
            continue

        # 该 chunk 内的删除线（转为 chunk 本地坐标）
        local_strikes = [sy - chunk_top for sy in strike_lines
                         if chunk_top < sy < chunk_bot]

        if not local_strikes:
            continue

        # 用已知坐标生成精确 mask
        chunk_gray = cv2.cvtColor(chunk_roi, cv2.COLOR_RGB2GRAY)
        strike_mask = make_strike_mask(chunk_gray, local_strikes, band_half=6)
        n_pixels = int(np.sum(strike_mask > 0))

        if n_pixels == 0:
            logger.info(f"  片[{ci}] 删除线坐标 {local_strikes} 但 mask 为空（可能太细）")
            continue

        total_strike_count += len(local_strikes)
        chunks_with_strike.add(ci)
        chunk_strike_masks[ci] = strike_mask

        # 可视化：红色标注删除线
        vis_strike = chunk_roi.copy()
        vis_strike[strike_mask > 0] = [255, 0, 0]
        # 也标注删除线 y 位置（蓝色虚线）
        for sy in local_strikes:
            if 0 <= sy < ch_h:
                cv2.line(vis_strike, (0, sy), (ch_w, sy), (0, 0, 255), 1)
        save_img(f"chunk_{ci:02d}_strike.jpg", vis_strike)

        # inpaint 修复
        chunk_bgr = cv2.cvtColor(chunk_roi, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(chunk_bgr, strike_mask,
                                     inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        vis_cleaned = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        save_img(f"chunk_{ci:02d}_cleaned.jpg", vis_cleaned)

        logger.info(f"  片[{ci}] 删除线: {len(local_strikes)} 条 @ {local_strikes}, "
                     f"mask={n_pixels}px")

    logger.info(f"删除线统计: {total_strike_count} 条, 涉及 {len(chunks_with_strike)} 个分片")

    # 总览图
    pad = 20
    y1 = max(0, mat_bbox.y - pad)
    y2 = min(img_h, mat_bbox.y2 + pad)
    x1 = max(0, mat_bbox.x - pad)
    x2 = min(img_w, mat_bbox.x2 + pad)
    vis_full = image[y1:y2, x1:x2].copy()
    # 分片切割线（红色粗线）
    for ci, (chunk_top, chunk_bot, _) in enumerate(chunks):
        for cy in [chunk_top, chunk_bot]:
            local_y = mat_bbox.y + cy - y1
            if 0 <= local_y < vis_full.shape[0]:
                cv2.line(vis_full, (0, local_y), (vis_full.shape[1], local_y), (255, 0, 0), 2)
    # 行线（黄色细线）
    for ly in table_lines:
        local_y = mat_bbox.y + ly - y1
        if 0 <= local_y < vis_full.shape[0]:
            cv2.line(vis_full, (0, local_y), (vis_full.shape[1], local_y), (255, 255, 0), 1)
    # 删除线（蓝色）
    for sy in strike_lines:
        local_y = mat_bbox.y + sy - y1
        if 0 <= local_y < vis_full.shape[0]:
            cv2.line(vis_full, (0, local_y), (vis_full.shape[1], local_y), (0, 100, 255), 2)
    save_img("overview_chunks.jpg", vis_full)

    logger.info("分片截图已保存，开始 OCR...")

    # ── Step 3: 分片 OCR + 青框生成（带去重）──
    logger.info("=" * 60)
    logger.info("Step 3: 分片 OCR + 青框生成")
    logger.info("=" * 60)

    prefixes = DEFAULT_PREFIXES
    prefix_set = {p.upper() for p in prefixes}
    ocr = _get_ocr(OCR_LANG_EN)

    cyan_boxes = []
    matched_cell_ys = set()
    total_ocr_time = 0

    for ci, (chunk_top, chunk_bot, chunk_tbl_lines) in enumerate(chunks):
        chunk_roi = box_roi[chunk_top:chunk_bot, :]
        ch, cw = chunk_roi.shape[:2]
        if ch < 3 or cw < 3:
            continue

        # ① 删除线处理
        has_strike = ci in chunks_with_strike
        if has_strike:
            chunk_bgr = cv2.cvtColor(chunk_roi, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(chunk_bgr, chunk_strike_masks[ci],
                                         inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            chunk_for_ocr = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            logger.info(f"  片[{ci}] 使用 cleaned 版本 OCR")
        else:
            chunk_for_ocr = chunk_roi

        # ② 缩放
        min_dim = min(cw, ch)
        max_dim = max(cw, ch)
        if min_dim < 80:
            sf = 5.0
        elif min_dim < 200:
            sf = 3.0
        else:
            sf = 1.0
        if max_dim * sf > 3500:
            sf = max(3500.0 / max_dim, 1.0)

        if sf > 1.0:
            scaled = cv2.resize(chunk_for_ocr, (int(cw * sf), int(ch * sf)),
                                interpolation=cv2.INTER_CUBIC)
        else:
            scaled = chunk_for_ocr
            sf = 1.0

        # ③ OCR
        t0 = time.perf_counter()
        result = ocr.predict(scaled)
        ocr_elapsed = time.perf_counter() - t0
        total_ocr_time += ocr_elapsed

        items = _parse_ocr_results(result)
        logger.info(f"  片[{ci}] y=[{chunk_top},{chunk_bot}] size={cw}x{ch} sf={sf:.1f} "
                     f"strike={'Y' if has_strike else 'N'} OCR: {len(items)} 项, {ocr_elapsed:.2f}s")

        # ④ 可视化 + 坐标映射
        vis_chunk = chunk_for_ocr.copy() if has_strike else chunk_roi.copy()
        for ly in chunk_tbl_lines:
            local_y = ly - chunk_top
            if 0 <= local_y < vis_chunk.shape[0]:
                cv2.line(vis_chunk, (0, local_y), (vis_chunk.shape[1], local_y),
                         (255, 255, 0), 1)

        for poly, text, score in items:
            if poly is None or len(poly) < 4:
                continue
            text_ns = text.replace(" ", "").strip()
            if not text_ns:
                continue

            ys_poly = [pt[1] / sf for pt in poly]
            xs_poly = [pt[0] / sf for pt in poly]
            text_cy = sum(ys_poly) / len(ys_poly)
            abs_y = chunk_top + text_cy

            x_min, y_min = int(min(xs_poly)), int(min(ys_poly))
            x_max, y_max = int(max(xs_poly)), int(max(ys_poly))
            is_match = text_ns[0].upper() in prefix_set
            color = (0, 255, 0) if is_match else (180, 180, 180)
            cv2.rectangle(vis_chunk, (x_min, y_min), (x_max, y_max), color, 1)
            label = f"{'*' if is_match else ''}{text_ns[:15]}"
            cv2.putText(vis_chunk, label, (x_min, max(y_min - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # 详细日志：每个识别项的文字、置信度、bbox
            match_tag = " [MATCH]" if is_match else ""
            logger.info(f"    [{text_ns[:20]:20s}] score={score:.3f} "
                         f"local=({x_min},{y_min},{x_max-x_min},{y_max-y_min}) "
                         f"abs_y={abs_y:.0f}{match_tag}")

            if not is_match:
                continue

            # ⑤ 匹配单元格（用行线 y 去重）
            for tl in table_lines:
                cell_bot = tl + cell_height
                if tl <= abs_y < cell_bot:
                    if tl not in matched_cell_ys:
                        matched_cell_ys.add(tl)
                        cyan_boxes.append(BBox(
                            mat_bbox.x, mat_bbox.y + tl,
                            mat_bbox.w, cell_height
                        ))
                        logger.info(f"    青框: line_y={tl}, text='{text_ns}'")
                    break

        save_img(f"chunk_{ci:02d}_ocr.jpg", vis_chunk)

        # 单独保存带详细 OCR 识别框的截图（红色多边形框 + 文字标注）
        vis_ocr_detail = chunk_for_ocr.copy() if has_strike else chunk_roi.copy()
        for idx, (poly, text, score) in enumerate(items):
            if poly is None or len(poly) < 4:
                continue
            text_ns = text.replace(" ", "").strip()
            if not text_ns:
                continue
            # 映射回原始尺度
            pts_orig = np.array([[pt[0] / sf, pt[1] / sf] for pt in poly], dtype=np.int32)
            is_match = text_ns[0].upper() in prefix_set
            box_color = (0, 200, 0) if is_match else (0, 0, 255)
            cv2.polylines(vis_ocr_detail, [pts_orig], True, box_color, 2)
            lx = int(pts_orig[:, 0].min())
            ly = int(pts_orig[:, 1].min())
            lbl = f"[{idx}] {text_ns[:15]} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(vis_ocr_detail, (lx, ly - th - 4), (lx + tw + 2, ly), box_color, -1)
            cv2.putText(vis_ocr_detail, lbl, (lx + 1, ly - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        save_img(f"chunk_{ci:02d}_ocr_boxes.jpg", vis_ocr_detail)

    # ── 汇总 ──
    logger.info(f"\n分片 OCR 总耗时: {total_ocr_time:.2f}s")
    logger.info(f"行线: {len(table_lines)}, 删除线: {len(strike_lines)}")
    logger.info(f"含删除线的分片: {len(chunks_with_strike)}")
    logger.info(f"青框数量: {len(cyan_boxes)}")

    # ── Step 4: 最终青框结果 ──
    logger.info("=" * 60)
    logger.info("Step 4: 最终青框结果")
    logger.info("=" * 60)

    vis_result = image[y1:y2, x1:x2].copy()
    cv2.rectangle(vis_result,
                  (mat_bbox.x - x1, mat_bbox.y - y1),
                  (mat_bbox.x2 - x1, mat_bbox.y2 - y1), (255, 0, 0), 2)
    for cb in cyan_boxes:
        cv2.rectangle(vis_result,
                      (cb.x - x1, cb.y - y1),
                      (cb.x2 - x1, cb.y2 - y1), (0, 255, 255), 2)
    save_img("result_cyan_boxes.jpg", vis_result)

    logger.info("完成!")


if __name__ == "__main__":
    main()
