"""文本替换引擎 — OCR 识别 + 像素级替换（支持多前缀）"""

import os
import re
import logging
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Y_PATTERN, FONT_PATH, OCR_LANG_EN, OCR_LANG_CH, DEFAULT_REGIONS, make_pattern, DEFAULT_PREFIXES
from modules.region_detector import BBox, _pct_to_px, _detect_horizontal_lines, _ocr_region

logger = logging.getLogger(__name__)

# ── OCR 引擎（延迟初始化，避免重复创建）──────────────────────

_ocr_cache = {}


def _get_ocr(lang: str = "en"):
    """获取 PaddleOCR v5 实例（按语言缓存）。"""
    if lang not in _ocr_cache:
        from paddleocr import PaddleOCR
        _ocr_cache[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_cache[lang]


# ── OCR 结果解析 ──────────────────────────────────────────────


def _parse_ocr_results(result):
    """
    从 PaddleOCR 返回值中提取 (polygon, text, confidence) 列表。
    兼容 PaddleOCR 不同版本的返回结构。
    """
    items = []
    if not result:
        return items

    for res in result:
        polys = None
        texts = None
        scores = None

        if isinstance(res, dict):
            polys = res.get("dt_polys", res.get("boxes"))
            texts = res.get("rec_texts", res.get("texts"))
            scores = res.get("rec_scores", res.get("scores"))
        elif hasattr(res, "dt_polys"):
            polys = res.dt_polys
            texts = res.rec_texts
            scores = getattr(res, "rec_scores", None)

        if polys is None or texts is None:
            continue

        if scores is None:
            scores = [1.0] * len(texts)

        for poly, text, score in zip(polys, texts, scores):
            items.append((poly, text, score))

    return items


# ── 最小像素替换 ──────────────────────────────────────────────


def _detect_row_boundaries(roi_gray: np.ndarray, scale_factor: float = 1.0) -> list[int]:
    """检测区域内的水平网格线，返回 y 坐标排序列表（原始尺寸坐标）。

    如果 scale_factor > 1，先放大图像再检测（放大后网格线更清晰），
    然后将坐标映射回原始尺寸。
    """
    if scale_factor > 1.0:
        sh, sw = roi_gray.shape[:2]
        scaled = cv2.resize(
            roi_gray,
            (int(sw * scale_factor), int(sh * scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        scaled = roi_gray

    _, thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)
    # 使用较小的核宽度（30% 列宽），以检测更短的水平线
    kernel_w = max(int(scaled.shape[1] * 0.3), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    y_coords = sorted(
        set(y + h // 2 for c in contours for (_, y, _, h) in [cv2.boundingRect(c)])
    )

    # 映射回原始坐标
    if scale_factor > 1.0:
        y_coords = [int(y / scale_factor) for y in y_coords]

    filtered = []
    for y in y_coords:
        if not filtered or y - filtered[-1] > 5:
            filtered.append(y)
    return filtered


def _uniform_row_ys(row_ys: list[int]) -> list[int]:
    """将不均匀的水平线列表转换为均匀间距的网格。

    计算所有间距的众数作为统一单元格高度，
    从第一条线开始按此高度重新生成网格。
    """
    if len(row_ys) < 3:
        return row_ys

    gaps = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]

    # 按 ±3px 聚类，取成员最多的组的均值作为单格高度
    valid_gaps = sorted(g for g in gaps if g >= 8)
    if not valid_gaps:
        return row_ys
    groups = []
    for g in valid_gaps:
        if not groups or g - groups[-1][-1] > 3:
            groups.append([g])
        else:
            groups[-1].append(g)
    largest_group = max(groups, key=len)
    cell_h = int(sum(largest_group) / len(largest_group))

    if cell_h < 5:
        return row_ys

    # 从第一条线开始，按 cell_h 生成均匀网格
    uniform = []
    y = row_ys[0]
    last_y = row_ys[-1]
    while y <= last_y + cell_h:
        uniform.append(y)
        y += cell_h

    return uniform


def _find_cell(text_cy: int, row_ys: list[int]) -> tuple[int, int]:
    """根据文字中心 y 坐标找到所在的单元格边界。

    返回 (cell_top, cell_bot)。如果找不到匹配的单元格，
    返回以 text_cy 为中心的默认范围。
    """
    if not row_ys or len(row_ys) < 2:
        return max(text_cy - 8, 0), text_cy + 8

    for i in range(len(row_ys) - 1):
        if row_ys[i] <= text_cy <= row_ys[i + 1]:
            return row_ys[i], row_ys[i + 1]

    # 文字在最后一条线以下
    if text_cy > row_ys[-1]:
        # 用最后两条线的间距作为估算
        last_gap = row_ys[-1] - row_ys[-2] if len(row_ys) >= 2 else 17
        return row_ys[-1], row_ys[-1] + last_gap

    # 文字在第一条线以上
    if text_cy < row_ys[0]:
        first_gap = row_ys[1] - row_ys[0] if len(row_ys) >= 2 else 17
        return row_ys[0] - first_gap, row_ys[0]

    return max(text_cy - 8, 0), text_cy + 8


def _find_cell_column_bounds(
    roi_gray: np.ndarray,
    text_x: int, text_w: int, text_y: int, text_h: int,
    margin: int = 3,
) -> tuple[int, int]:
    """检测文字所在单元格的左右竖线边界。

    在 ROI 灰度图中用形态学检测竖线，找 text 中心两侧最近的竖线。
    返回 (left, right) — ROI 内 x 坐标，已含 margin 内缩。
    """
    roi_h, roi_w = roi_gray.shape[:2]
    text_cx = text_x + text_w // 2

    # 竖线检测
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_h = max(text_h, 15)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取竖线 x 坐标（取轮廓中心），按 y 范围过滤（与文字行重叠）
    vlines = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        vx = x + bw // 2
        # 竖线需要与文字 y 范围有一定重叠
        if y < text_y + text_h and y + bh > text_y:
            vlines.append(vx)
    vlines.sort()

    # 找 text_cx 两侧最近的竖线
    left_vlines = [vx for vx in vlines if vx < text_x + 2]
    right_vlines = [vx for vx in vlines if vx > text_x + text_w - 2]

    if left_vlines:
        cell_left = left_vlines[-1] + margin
    else:
        cell_left = max(text_x - margin, 0)

    if right_vlines:
        cell_right = right_vlines[0] - margin
    else:
        cell_right = min(text_x + text_w + margin, roi_w)

    # 安全检查：如果检测到的列宽过窄（竖线误检），回退到 OCR 宽度
    if cell_right - cell_left < text_w * 0.5:
        cell_left = max(text_x - margin, 0)
        cell_right = min(text_x + text_w + margin, roi_w)

    return cell_left, cell_right


def _find_y_prefix_bbox(poly, text: str):
    """
    估算 Y 字符在文本框中的像素位置。

    poly: 四点多边形 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: OCR 识别的文本

    返回 (y_char_bbox, full_bbox)
        y_char_bbox: Y 字符的像素矩形 (x, y, w, h)
        full_bbox:   整个文本框的像素矩形
    """
    pts = np.array(poly, dtype=np.float32)
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())
    x_max = int(pts[:, 0].max())
    y_max = int(pts[:, 1].max())
    full_w = x_max - x_min
    full_h = y_max - y_min

    if len(text) == 0:
        return None, (x_min, y_min, full_w, full_h)

    # 估算 Y 字符宽度（按字符均分）
    char_w = full_w / len(text)
    # Y 在文本中的位置（只处理开头的 Y）
    y_char_bbox = (x_min, y_min, int(char_w), full_h)
    full_bbox = (x_min, y_min, full_w, full_h)

    return y_char_bbox, full_bbox


def _render_text_distributed(
    text: str, target_w: int, target_h: int, font_path: str = None
) -> Image.Image:
    """渲染文本为 RGBA 图像，紧凑排列后整体缩放到目标尺寸。

    参数:
        text: 要渲染的文本
        target_w: 目标图像宽度（像素）
        target_h: 目标图像高度（像素）
        font_path: 字体路径

    返回 target_w x target_h 的 RGBA 图像。
    """
    if font_path is None:
        font_path = FONT_PATH
    if not text:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))

    # 用较大字号渲染以保证质量，后续缩放到目标尺寸
    render_size = max(target_h * 2, 32)
    try:
        font = ImageFont.truetype(font_path, render_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)
    bb = draw_dummy.textbbox((0, 0), text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]

    if text_w <= 0 or text_h <= 0:
        return Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))

    # 在刚好够大的画布上紧凑渲染
    img = Image.new("RGBA", (text_w, text_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bb[0], -bb[1]), text, fill="black", font=font)

    # 等比例缩放：取较小的比例，确保不超出目标区域且不变形
    scale_h = target_h / text_h
    scale_w = target_w / text_w
    scale = min(scale_h, scale_w)

    new_w = max(int(text_w * scale), 1)
    new_h = max(int(text_h * scale), 1)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 居中放置在 target_w × target_h 画布上
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = target_h - new_h  # 底部对齐
    canvas.paste(img, (paste_x, paste_y), img)
    return canvas


def replace_y_in_region_pixel(
    image: np.ndarray, bbox: BBox, lang: str = OCR_LANG_EN,
    use_grid_alignment: bool = False,
    row_ys: list[int] = None,
    pattern: str = None,
    prefixes: list[str] = None,
) -> tuple[np.ndarray, list]:
    """
    在指定区域内 OCR 识别并像素级替换（前缀加H）。

    策略：
    1. 裁剪区域 → 自适应放大 → OCR 获取所有文本及位置
    2. 检测行边界（水平网格线）用于限制替换范围
    3. 过滤出 Y 开头的编号
    4. 白色填充原文本区域（严格限制在单元格内）
    5. 渲染新文本，按比例缩放并居中贴回

    Args:
        pattern: Y编号匹配正则（None则使用全局Y_PATTERN）

    返回 (修改后的全图, [(old_text, new_text), ...])
    """
    roi = bbox.crop(image)

    # 自适应放大：基于最小维度决定缩放比例
    roi_h, roi_w = roi.shape[:2]
    min_dim = min(roi_w, roi_h)
    max_dim = max(roi_w, roi_h)

    if min_dim < 80:
        scale_factor = 5.0
    elif min_dim < 200:
        scale_factor = 3.0
    else:
        scale_factor = 1.0

    # 限制最大维度不超过 OCR 上限（避免被内部 resize 降质）
    if max_dim * scale_factor > 3500:
        scale_factor = max(3500.0 / max_dim, 1.0)

    if scale_factor > 1.0:
        new_h = int(roi.shape[0] * scale_factor)
        new_w = int(roi.shape[1] * scale_factor)
        roi_scaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        roi_scaled = roi

    # 小区域（面积有限）加白色边距 + 锐化，帮助 OCR 检测到文字
    pad_px = 0
    if roi_h < 100 and roi_w < 400:
        pad_px = max(int(min(roi_scaled.shape[:2]) * 0.35), 40)
        padded = np.full(
            (roi_scaled.shape[0] + 2 * pad_px, roi_scaled.shape[1] + 2 * pad_px, 3),
            255, dtype=np.uint8,
        )
        padded[pad_px:pad_px + roi_scaled.shape[0],
               pad_px:pad_px + roi_scaled.shape[1]] = roi_scaled
        # 锐化提高边缘清晰度
        sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        roi_scaled = cv2.filter2D(padded, -1, sharpen)

    # 条件性检测行边界（仅网格对齐模式）
    if use_grid_alignment:
        if row_ys is None:
            # 回退：在当前 ROI 上检测（窄列可能不可靠）
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            row_ys = _detect_row_boundaries(roi_gray, scale_factor=scale_factor)
        row_ys = _uniform_row_ys(row_ys)
        cell_h = (row_ys[1] - row_ys[0]) if len(row_ys) >= 2 else 0
        logger.info(f"  行边界: {len(row_ys)} 条线, cell_h={cell_h}px")
    else:
        row_ys = []

    ocr = _get_ocr(lang)
    result = ocr.predict(roi_scaled)
    items = _parse_ocr_results(result)

    # 诊断日志：输出OCR识别的文字样本
    logger.info(f"  检测到文本: {len(items)} 项")
    for i, (poly, text, score) in enumerate(items[:15]):  # 显示前15项
        logger.info(f"    [{i}] '{text}' (score={score:.2f})")

    modified = image.copy()
    replacements = []
    prefixes = prefixes or DEFAULT_PREFIXES
    pattern = re.compile(pattern if pattern else make_pattern(prefixes))

    # ── 第一遍：收集所有 Y 匹配项的坐标信息 ──
    matched_items = []
    for poly, text, score in items:
        # 去掉空格后再匹配（OCR可能在编号中插入空格）
        text_nospace = text.replace(" ", "")
        match = pattern.match(text_nospace)
        if not match:
            continue

        y_bbox, full_bbox = _find_y_prefix_bbox(poly, text)
        if y_bbox is None:
            continue

        # 将缩放后的坐标映射回原尺寸（减去 padding 偏移）
        fx, fy, fw, fh = full_bbox
        fx = int((fx - pad_px) / scale_factor)
        fy = int((fy - pad_px) / scale_factor)
        fw = max(int(fw / scale_factor), 1)
        fh = max(int(fh / scale_factor), 1)
        matched_items.append((text_nospace, fx, fy, fw, fh))

    matched_items.sort(key=lambda m: m[2])  # 按 fy 排序

    # ── 第二遍：执行替换 ──
    # 预计算 ROI 灰度图（用于竖线检测，避免重复转换）
    _roi_gray_cache = None

    FILL_MARGIN = 2  # 填充/渲染内缩像素，避开网格线

    for text_nospace, fx, fy, fw, fh in matched_items:
        old_text = text_nospace
        new_text = "H" + old_text

        if use_grid_alignment and len(row_ys) >= 2:
            # 网格对齐模式：用单元格边界定位（行方向）
            text_cy = fy + fh // 2
            cell_top, cell_bot = _find_cell(text_cy, row_ys)
            cell_h = cell_bot - cell_top
            fill_top = cell_top
            fill_h = cell_h
            if fill_h < 4:
                fill_top = fy
                fill_h = fh

            # 列方向：用竖线检测确定单元格左右边界
            if _roi_gray_cache is None:
                _roi_gray_cache = cv2.cvtColor(bbox.crop(image), cv2.COLOR_RGB2GRAY)
            cell_left, cell_right = _find_cell_column_bounds(
                _roi_gray_cache, fx, fw, fy, fh, margin=3)
            safe_x = cell_left
            safe_w = cell_right - cell_left
        else:
            # 直接模式：用 OCR bbox + 适度扩展（容纳多1字符）
            fill_top = fy
            fill_h = fh
            extra = max(fw // max(len(old_text), 1), 4)
            safe_x = max(fx - 1, 0)
            safe_w = fw + extra

        # 转换为全图坐标
        gx = bbox.x + safe_x
        gy_fill = bbox.y + fill_top

        # 白色填充：内缩 FILL_MARGIN 避开网格线
        cv2.rectangle(
            modified,
            (gx + FILL_MARGIN, gy_fill + FILL_MARGIN),
            (gx + safe_w - FILL_MARGIN, gy_fill + fill_h - FILL_MARGIN),
            (255, 255, 255),
            -1,
        )

        # 渲染替换后的完整文本（空格已去除）
        remaining = text_nospace[len(old_text):]
        render_text = new_text + remaining
        render_w = max(safe_w - 2 * FILL_MARGIN, 6)
        render_h = max(fill_h - 2 * FILL_MARGIN, 6)
        text_img = _render_text_distributed(render_text, render_w, render_h)

        # 贴回：对齐内缩后的区域
        paste_x = gx + FILL_MARGIN
        paste_y = gy_fill + FILL_MARGIN

        pil_modified = Image.fromarray(modified)
        pil_modified.paste(text_img, (paste_x, paste_y), text_img)
        modified = np.array(pil_modified)

        replacements.append((old_text, new_text))
        logger.info(f"  替换: {old_text} → {new_text} (safe_w={safe_w}, fill_h={fill_h})")

    return modified, replacements


# ── 公共 OCR 辅助 + 保存后验证 ────────────────────────────────


def _ocr_region_with_scaling(
    image: np.ndarray, bbox: BBox, lang: str = OCR_LANG_EN
) -> list:
    """对指定区域进行自适应放大 OCR，返回 [(poly, text, score), ...]。

    复用 replace_y_in_region_pixel() 中的放大 + padding + 锐化逻辑。
    """
    roi = bbox.crop(image)
    roi_h, roi_w = roi.shape[:2]
    min_dim = min(roi_w, roi_h)
    max_dim = max(roi_w, roi_h)

    if min_dim < 80:
        sf = 5.0
    elif min_dim < 200:
        sf = 3.0
    else:
        sf = 1.0
    if max_dim * sf > 3500:
        sf = max(3500.0 / max_dim, 1.0)

    if sf > 1.0:
        roi_scaled = cv2.resize(
            roi, (int(roi_w * sf), int(roi_h * sf)),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        roi_scaled = roi

    if roi_h < 100 and roi_w < 400:
        pad = max(int(min(roi_scaled.shape[:2]) * 0.35), 40)
        padded = np.full(
            (roi_scaled.shape[0] + 2 * pad, roi_scaled.shape[1] + 2 * pad, 3),
            255, dtype=np.uint8,
        )
        padded[pad:pad + roi_scaled.shape[0],
               pad:pad + roi_scaled.shape[1]] = roi_scaled
        sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        roi_scaled = cv2.filter2D(padded, -1, sharpen)

    ocr = _get_ocr(lang)
    result = ocr.predict(roi_scaled)
    return _parse_ocr_results(result)


def verify_output(
    output_path: str, regions: dict, expected_total: int,
) -> dict:
    """对保存后的输出文件进行 OCR 验证。

    加载输出图片 → 对每个有替换的区域 OCR → 统计 HY 数量 → 对比。
    """
    img = np.array(Image.open(output_path).convert("RGB"))

    hy_pattern = re.compile(r"H[A-Z][A-Z0-9][A-Z0-9\-]{2,}")
    total_hy = 0

    for region_name, bbox in regions.items():
        if region_name.startswith("_") or bbox is None:
            continue
        items = _ocr_region_with_scaling(img, bbox)
        hy_count = sum(1 for _, text, _ in items if hy_pattern.search(text))
        total_hy += hy_count
        logger.info(f"  验证 {region_name}: 检测到 {hy_count} 个 HY")

    match = total_hy >= expected_total
    logger.info(
        f"  验证总结: 替换了 {expected_total} 个, "
        f"回检到 {total_hy} 个 HY "
        f"({'通过' if match else '不匹配'})"
    )
    return {"expected": expected_total, "found_hy": total_hy, "match": match}


# ── 统一替换入口 ───────────────────────────────────────────────


def replace_in_all_regions(
    image: np.ndarray, regions: dict[str, BBox | None],
    filename: str | None = None,
    prefixes: list[str] = None,
) -> tuple[np.ndarray, list]:
    """
    对所有检测到的区域执行文本替换。

    处理顺序固定：material_code_column → bottom_right_number → top_left_number → annotations
    top_left_number 直接复用 bottom_right_number 的替换结果（内容相同）。

    返回 (修改后全图, 所有替换记录)
    """

    modified = image.copy()
    all_replacements = []

    # 固定处理顺序，确保 bottom_right 在 top_left 之前
    ordered = [
        "material_code_column", "bottom_right_number",
        "top_left_number",
    ]

    # ── 预处理：从 metadata 获取检测时的文本 ──
    metadata = regions.get("_metadata", {})
    green_text = metadata.get("bottom_right_text")   # 检测时OCR
    orange_text = metadata.get("top_left_text")       # 检测时OCR
    red_text = None  # 红框替换后从 repls 中提取

    green_bbox = regions.get("bottom_right_number")
    orange_bbox = regions.get("top_left_number")
    green_orange_result = None  # 四方投票结果，绿框处理时设置，橙框复用

    prefixes = prefixes or DEFAULT_PREFIXES
    prefixes_upper = [p.upper() for p in prefixes]
    prefix_pattern = make_pattern(prefixes)

    # 从文件名提取编号
    def _extract_from_filename(fname):
        if not fname:
            return None
        base = os.path.splitext(os.path.basename(fname))[0].upper()
        m = re.search(prefix_pattern, base)
        return m.group() if m else None

    filename_y = _extract_from_filename(filename)

    # 文本验证：9位、无空格、首字母在 prefixes 中
    def _valid_prefix(s):
        if not s:
            return None
        s = s.upper().replace(" ", "")
        if len(s) != 9:
            return None
        if s[0] in prefixes_upper and re.match(r'^[A-Z][A-Z0-9]{8}$', s):
            return s
        return None

    p_chars = "".join(prefixes_upper)

    for region_name in ordered:
        bbox = regions.get(region_name)
        if bbox is None:
            logger.info(f"跳过区域 {region_name}: 未检测到")
            continue

        logger.info(f"处理区域: {region_name} ({bbox})")

        if region_name == "material_code_column":
            # 用宽表格区域检测水平线（比窄列更可靠）
            metadata = regions.get("_metadata", {})
            table_search_bbox = metadata.get("table_search_area")

            if table_search_bbox:
                # 新路径：用检测阶段传来的宽搜索区域
                table_gray = cv2.cvtColor(table_search_bbox.crop(image), cv2.COLOR_RGB2GRAY)
                all_h_lines = _detect_horizontal_lines(table_gray, min_line_width=18)
                bbox_y_in_table = bbox.y - table_search_bbox.y
            else:
                # 回退：用百分比区域
                left_table_pct = DEFAULT_REGIONS["left_table"]
                search = _pct_to_px(image, left_table_pct)
                table_gray = cv2.cvtColor(search.crop(image), cv2.COLOR_RGB2GRAY)
                all_h_lines = _detect_horizontal_lines(table_gray, min_line_width=18)
                bbox_y_in_table = bbox.y - search.y

            # 过滤到 bbox 的 y 范围，转为 bbox 内相对坐标
            row_ys = [
                y - bbox_y_in_table
                for y in all_h_lines
                if bbox_y_in_table <= y <= bbox_y_in_table + bbox.h
            ]
            logger.info(f"  宽表格检测: {len(all_h_lines)} 条线, 列内 {len(row_ys)} 条")
            # 红框使用宽松匹配：Y + 8位字母数字或连字符
            # 红框使用宽松匹配：前缀 + 8位字母数字或连字符
            red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
            modified, repls = replace_y_in_region_pixel(
                modified, bbox, lang=OCR_LANG_EN,
                use_grid_alignment=True, row_ys=row_ys,
                pattern=red_pattern, prefixes=prefixes,
            )

            # 从红框替换结果中提取 red_text
            if repls:
                red_text = repls[0][0]  # old_y
                logger.info(f"  红框提取文本: '{red_text}'")

        elif region_name in ("bottom_right_number", "top_left_number"):
            # 首次遇到绿/橙框时执行五方校验
            if region_name == "bottom_right_number":
                # 四方投票（绿框、橙框、红框、文件名）
                g = _valid_prefix(green_text)
                o = _valid_prefix(orange_text)
                r = _valid_prefix(red_text)
                f = _valid_prefix(filename_y)
                logger.info(f"  四方校验: green={g}, orange={o}, red={r}, filename={f}")

                candidates = [x for x in [g, o, r, f] if x]
                source_y = None

                if not candidates:
                    logger.warning("  四方校验：无有效候选文本，跳过绿/橙框替换")
                elif len(set(candidates)) == 1:
                    source_y = candidates[0]
                    logger.info(f"  四方校验：全部一致 → '{source_y}'")
                else:
                    counts = Counter(candidates)
                    top_text, top_count = counts.most_common(1)[0]
                    if top_count >= 2:
                        source_y = top_text
                        logger.info(f"  四方校验：多数一致({top_count}/{len(candidates)}) → '{source_y}'")
                    else:
                        source_y = g or o or r or f
                        logger.info(f"  四方校验：全不同，优先绿/橙/红/文件名 → '{source_y}'")

                # 保存结果供橙框复用
                if source_y:
                    new_y = "H" + source_y
                    green_orange_result = (source_y, new_y)
                    logger.info(f"  三方投票结果: '{source_y}' → '{new_y}'")
                else:
                    green_orange_result = None

            if green_orange_result:
                old_y, new_y = green_orange_result
                M = 3  # 内缩像素，避开框线
                if region_name == "bottom_right_number":
                    # 绿框：右侧缩5%（避开竖线）+ 全方向内缩M
                    pad_r = int(bbox.w * 0.05)
                    draw_x = bbox.x + M
                    draw_y = bbox.y + M
                    draw_w = max(bbox.w - pad_r - 2 * M, 1)
                    draw_h = max(bbox.h - 2 * M, 1)
                    cv2.rectangle(
                        modified, (draw_x, draw_y),
                        (draw_x + draw_w, draw_y + draw_h),
                        (255, 255, 255), -1,
                    )
                    text_img = _render_text_distributed(new_y, draw_w, draw_h)
                    pil_modified = Image.fromarray(modified)
                    pil_modified.paste(text_img, (draw_x, draw_y), text_img)
                    modified = np.array(pil_modified)
                else:
                    # 橙框：整个bbox + 全方向内缩M
                    draw_x = bbox.x + M
                    draw_y = bbox.y + M
                    draw_w = max(bbox.w - 2 * M, 1)
                    draw_h = max(bbox.h - 2 * M, 1)
                    cv2.rectangle(
                        modified, (draw_x, draw_y),
                        (draw_x + draw_w, draw_y + draw_h),
                        (255, 255, 255), -1,
                    )
                    text_img = _render_text_distributed(new_y, draw_w, draw_h)
                    pil_modified = Image.fromarray(modified)
                    pil_modified.paste(text_img, (draw_x, draw_y), text_img)
                    modified = np.array(pil_modified)
                repls = [(old_y, new_y)]
                logger.info(f"  {region_name}: 替换 {old_y} → {new_y}")
            else:
                repls = []
                logger.warning(f"  {region_name}: 无有效文本，跳过替换")

        all_replacements.extend(repls)
        logger.info(f"  区域 {region_name}: {len(repls)} 处替换")

    return modified, all_replacements
