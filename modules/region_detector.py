"""动态区域检测 — 关键词锚定 + CV 结构检测 + 局部 OCR 确认"""

import re
import logging
import difflib

import cv2
import numpy as np

from config import (
    DEFAULT_REGIONS, KEYWORD_ANCHORS,
    Y_PATTERN, OCR_LANG_EN, OCR_LANG_CH, FUZZY_DIGIT_MAP,
    make_pattern, DEFAULT_PREFIXES,
)

logger = logging.getLogger(__name__)

# 延迟导入缓存（避免循环依赖，同时避免每次调用都 import）
_cached_get_ocr = None
_cached_parse = None


def _get_ocr_funcs():
    """延迟导入并缓存 OCR 函数，避免循环依赖和重复 import。"""
    global _cached_get_ocr, _cached_parse
    if _cached_get_ocr is None:
        from modules.text_replacer import _get_ocr, _parse_ocr_results
        _cached_get_ocr = _get_ocr
        _cached_parse = _parse_ocr_results
    return _cached_get_ocr, _cached_parse


# ══════════════════════════════════════════════════════════════════
#  BBox 基础类
# ══════════════════════════════════════════════════════════════════

class BBox:
    """边界框（像素坐标）"""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y2, self.x : self.x2]

    def contains_point(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def contains(self, other: "BBox") -> bool:
        return (self.x <= other.x and self.y <= other.y
                and self.x2 >= other.x2 and self.y2 >= other.y2)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: dict) -> "BBox":
        return cls(d["x"], d["y"], d["w"], d["h"])

    def __repr__(self):
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


# ══════════════════════════════════════════════════════════════════
#  基础工具函数
# ══════════════════════════════════════════════════════════════════

def _pct_to_px(image: np.ndarray, region_pct: dict) -> BBox:
    """百分比坐标 → 像素坐标"""
    h, w = image.shape[:2]
    x = int(region_pct["x_min"] * w)
    y = int(region_pct["y_min"] * h)
    bw = int((region_pct["x_max"] - region_pct["x_min"]) * w)
    bh = int((region_pct["y_max"] - region_pct["y_min"]) * h)
    return BBox(x, y, bw, bh)


def _detect_vertical_lines(roi_gray: np.ndarray, min_line_height: int = 100):
    """检测 ROI 中的垂直线，返回 (x坐标列表, y_min, y_max)。"""
    denoised = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_height))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_data = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        line_data.append([x + bw // 2, y, y + bh, bh])
    line_data.sort(key=lambda d: d[0])

    merged = []
    for xc, y1, y2, h in line_data:
        if not merged or xc - merged[-1][0] > 8:
            merged.append([xc, y1, y2, h])
        else:
            prev = merged[-1]
            prev[1] = min(prev[1], y1)
            prev[2] = max(prev[2], y2)
            prev[3] = prev[2] - prev[1]

    x_coords = [m[0] for m in merged]

    if merged:
        max_h = max(m[3] for m in merged)
        tall = [m for m in merged if m[3] > max_h * 0.5]
        y_min = min(m[1] for m in tall)
        y_max = max(m[2] for m in tall)
    else:
        y_min = 0
        y_max = roi_gray.shape[0]

    return x_coords, y_min, y_max


def _detect_horizontal_lines(roi_gray: np.ndarray, min_line_width: int = 30,
                             min_width_ratio: float = 0.0):
    """检测 ROI 中的水平线，返回 y 坐标排序列表。

    Args:
        min_line_width: 形态学核宽度（像素）
        min_width_ratio: 线条最小宽度占 ROI 宽度的比例 (0~1)，
                         用于过滤删除线等短横线。0 表示不过滤。
    """
    denoised = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_w = roi_gray.shape[1]
    min_w = int(roi_w * min_width_ratio) if min_width_ratio > 0 else 0

    y_coords = sorted(set(
        y + bh // 2
        for c in contours
        for (x, y, bw, bh) in [cv2.boundingRect(c)]
        if bw >= min_w
    ))

    filtered = []
    for y in y_coords:
        if not filtered or y - filtered[-1] > 8:
            filtered.append(y)
    return filtered


def _detect_horizontal_lines_adaptive(
    roi_gray: np.ndarray,
    bbox_y_offset: int,
    bbox_h: int,
    min_line_width: int = 18,
    min_required_lines: int = 6,
) -> list[int]:
    """动态双阈值策略检测水平线，确保红框内有足够的 row_ys。

    第一优先级: min_width_ratio=0.3
    回退: 若红框内 row_ys < min_required_lines, 降至 min_width_ratio=0.20
    再回退: 若仍不足, 使用更大形态学核 + min_width_ratio=0.15

    Args:
        roi_gray: 宽表格搜索区域的灰度图
        bbox_y_offset: 红框 y 相对于 roi_gray 的偏移 (bbox.y - table_search_bbox.y)
        bbox_h: 红框高度
        min_line_width: 形态学核宽度
        min_required_lines: 红框内所需最少水平线数

    Returns:
        宽表格区域内的所有水平线 y 坐标列表 (全局坐标, 未映射)
    """
    thresholds = [
        (0.30, min_line_width),       # 第一优先级
        (0.20, min_line_width),       # 第二优先级
        (0.15, max(min_line_width, 20)),  # 兜底: 更大核 + 更低阈值
    ]

    for ratio, kernel_w in thresholds:
        all_h_lines = _detect_horizontal_lines(
            roi_gray, min_line_width=kernel_w, min_width_ratio=ratio)

        # 统计映射到红框内的有效线数
        row_ys_count = sum(
            1 for y in all_h_lines
            if bbox_y_offset <= y <= bbox_y_offset + bbox_h
        )

        logger.info(
            f"  水平线检测 (ratio={ratio}, kernel={kernel_w}): "
            f"全表格 {len(all_h_lines)} 条, 红框内 {row_ys_count} 条"
        )

        if row_ys_count >= min_required_lines:
            return all_h_lines

    # 所有阈值都不够, 返回最宽松的结果
    logger.warning(
        f"  水平线检测: 所有阈值均未达到 {min_required_lines} 条, "
        f"使用最宽松结果 ({len(all_h_lines)} 条)"
    )
    return all_h_lines


def _detect_vlines_by_projection(
    roi_gray: np.ndarray, min_peak_ratio: float = 0.3, max_peak_width: int = 10
) -> list[int]:
    """用垂直投影法检测竖线（对断裂/模糊线更鲁棒）。

    对每列统计黑色像素占比，找窄且高的峰——即竖线位置。
    与形态学方法互补：形态学要求连续 min_h 像素，投影法不要求连续。
    """
    denoised = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height = thresh.shape[0]
    if height == 0:
        return []

    # 每列黑色像素占比 [0, 1]
    profile = np.sum(thresh, axis=0).astype(float) / (255.0 * height)

    # 找连续列 > min_peak_ratio 的区段，保留宽度 ≤ max_peak_width 的窄峰
    above = profile > min_peak_ratio
    peaks = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            width = i - start
            if width <= max_peak_width:
                peaks.append(start + width // 2)
        else:
            i += 1

    return peaks


# ══════════════════════════════════════════════════════════════════
#  OCR 工具（局部使用，仅在确认阶段调用）
# ══════════════════════════════════════════════════════════════════

def _ocr_region(image_rgb: np.ndarray, bbox: BBox, lang: str = OCR_LANG_EN) -> list:
    """对指定区域做 OCR，返回 [(text, confidence, poly), ...]。

    自动放大小区域 + 加白色 padding + 锐化。
    """
    _get_ocr, _parse_ocr_results = _get_ocr_funcs()

    roi = bbox.crop(image_rgb)
    roi_h, roi_w = roi.shape[:2]

    # 自适应缩放
    min_dim = min(roi_w, roi_h)
    max_dim = max(roi_w, roi_h)
    if min_dim < 80:
        scale = 5.0
    elif min_dim < 200:
        scale = 3.0
    else:
        scale = 1.0
    if max_dim * scale > 3500:
        scale = 3500.0 / max_dim

    if scale != 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 小区域：加 padding + 锐化
    pad_px = 0
    if roi_h < 100 and roi_w < 400:
        pad_px = max(int(min(roi.shape[:2]) * 0.35), 40)
        padded = np.full(
            (roi.shape[0] + 2 * pad_px, roi.shape[1] + 2 * pad_px, 3),
            255, dtype=np.uint8,
        )
        padded[pad_px:pad_px + roi.shape[0], pad_px:pad_px + roi.shape[1]] = roi
        sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        roi = cv2.filter2D(padded, -1, sharpen)

    ocr = _get_ocr(lang)
    result = ocr.predict(roi)
    items = _parse_ocr_results(result)

    out = []
    for poly, text, conf in items:
        # 映射坐标回原图（仅用于需要位置的场景）
        if poly is not None and len(poly) >= 4:
            mapped_poly = []
            for pt in poly:
                px = (pt[0] - pad_px) / scale + bbox.x
                py = (pt[1] - pad_px) / scale + bbox.y
                mapped_poly.append([px, py])
            out.append((text, conf, mapped_poly))
        else:
            out.append((text, conf, None))
    return out


# ══════════════════════════════════════════════════════════════════
#  关键词模糊匹配
# ══════════════════════════════════════════════════════════════════

def _fuzzy_find_keyword(
    ocr_results: list,
    keywords: list[str],
    threshold: float = 0.70,
) -> dict | None:
    """在 OCR 结果中模糊匹配关键词。

    ocr_results: [(text, confidence, poly), ...]

    返回 {"text": str, "keyword": str, "score": float, "poly": list, "confidence": float}
    或 None
    """
    best = None

    for text, conf, poly in ocr_results:
        text_upper = text.upper().strip()
        text_nospace = text_upper.replace(" ", "")

        for kw in keywords:
            kw_upper = kw.upper()
            kw_nospace = kw_upper.replace(" ", "")

            # 精确子串匹配
            if kw_upper in text_upper or kw_nospace in text_nospace:
                score = 1.0
            else:
                # 模糊匹配
                score = difflib.SequenceMatcher(None, text_nospace, kw_nospace).ratio()

            if score >= threshold:
                if best is None or score > best["score"]:
                    best = {
                        "text": text, "keyword": kw,
                        "score": score, "poly": poly, "confidence": conf,
                    }

    return best


# ══════════════════════════════════════════════════════════════════
#  Phase A: 纯 CV 结构检测 — 找到所有表格区域
# ══════════════════════════════════════════════════════════════════

def _detect_drawing_frame(image: np.ndarray) -> BBox:
    """检测图纸外边框（实际画框线）。

    用形态学检测跨越图纸大部分宽度/高度的长线条，
    取最外侧的水平线和垂直线组成外边框。
    检测失败则回退到图片边界（2% 内缩）。
    """
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 多阈值检测：先尝试高阈值，失败后尝试低阈值
    for threshold in [150, 130, 100]:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # 检测长水平线（>50% 图片宽度）
        h_len = max(int(img_w * 0.5), 100)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_ys = sorted(set(y + bh // 2 for c in contours for (_, y, _, bh) in [cv2.boundingRect(c)]))

        # 检测长垂直线（>50% 图片高度）
        v_len = max(int(img_h * 0.5), 100)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_xs = sorted(set(x + bw // 2 for c in contours for (x, _, bw, _) in [cv2.boundingRect(c)]))

        if len(h_ys) >= 2 and len(v_xs) >= 2:
            frame = BBox(v_xs[0], h_ys[0], v_xs[-1] - v_xs[0], h_ys[-1] - h_ys[0])
            logger.info(f"检测到图纸边界(阈值={threshold}): {frame}")
            return frame

    # 第二轮：降低长度要求到40%
    logger.debug("第一轮检测失败，尝试40%长度")
    for threshold in [150, 130, 100]:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        h_len = max(int(img_w * 0.4), 100)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_ys = sorted(set(y + bh // 2 for c in contours for (_, y, _, bh) in [cv2.boundingRect(c)]))

        v_len = max(int(img_h * 0.4), 100)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_xs = sorted(set(x + bw // 2 for c in contours for (x, _, bw, _) in [cv2.boundingRect(c)]))

        if len(h_ys) >= 2 and len(v_xs) >= 2:
            frame = BBox(v_xs[0], h_ys[0], v_xs[-1] - v_xs[0], h_ys[-1] - h_ys[0])
            # 验证边框合理性：应该接近图像边缘且覆盖大部分图像
            if (frame.x < img_w * 0.15 and frame.y < img_h * 0.15 and
                frame.w > img_w * 0.7 and frame.h > img_h * 0.7):
                logger.info(f"检测到图纸边界(阈值={threshold},40%): {frame}")
                return frame

    # 回退：图片边界 5% 内缩（增加内缩比例以去除留白）
    mx, my = int(img_w * 0.05), int(img_h * 0.05)
    frame = BBox(mx, my, img_w - 2 * mx, img_h - 2 * my)
    logger.info(f"未检测到边框线，使用图片边界: {frame}")
    return frame


# ══════════════════════════════════════════════════════════════════
#  Phase B-1: 定位 MATERIAL CODE 列
# ══════════════════════════════════════════════════════════════════

def _validate_material_code_column(image: np.ndarray, col_bbox: BBox) -> bool:
    """验证列内是否包含MATERIAL CODE文字，确认找对了位置"""
    # 向上扩展200px包含表头区域
    extended_y = max(0, col_bbox.y - 200)
    extended_h = col_bbox.h + (col_bbox.y - extended_y)
    extended_bbox = BBox(col_bbox.x, extended_y, col_bbox.w, extended_h)

    # 先英文OCR，再中文OCR，合并结果
    ocr_results_en = _ocr_region(image, extended_bbox)
    ocr_results_ch = _ocr_region(image, extended_bbox, lang=OCR_LANG_CH)
    ocr_results = ocr_results_en + ocr_results_ch
    logger.info(f"  验证区域OCR: en={len(ocr_results_en)}项, ch={len(ocr_results_ch)}项")
    for i, (text, conf, poly) in enumerate(ocr_results[:10]):
        logger.info(f"    [{i}] '{text}'")

    # 第一轮：精确匹配
    for text, conf, poly in ocr_results:
        text_up = text.upper().replace(" ", "")
        if "MATERIALCODE" in text_up or "MATERIAL" in text_up or "零部件图号" in text:
            logger.info(f"  列内容验证通过(精确): 找到'{text}'")
            return True

    # 第二轮：模糊匹配
    keywords = ["MATERIAL CODE", "MATERUL CODE", "零部件图号", "DEF"]
    match = _fuzzy_find_keyword(ocr_results, keywords, threshold=0.60)
    if match:
        logger.info(f"  列内容验证通过(模糊): 找到'{match['text']}' 匹配'{match['keyword']}' (score={match['score']:.2f})")
        return True

    logger.warning(f"  列内容验证失败: 未找到MATERIAL CODE相关文字")
    return False


def _locate_material_code_column(
    image: np.ndarray,
    table_regions: list[BBox],
) -> tuple[BBox | None, BBox | None, str | None]:
    """在表格区域中找到 MATERIAL CODE 列。

    策略：
    1. 对每个表格区域的表头行做局部 OCR
    2. 模糊匹配 "MATERIAL CODE"
    3. 用网格线定位具体列，追踪到表格结束

    返回 (column_bbox, table_search_bbox, direction) 或 (None, None, None)
    """
    kw_config = KEYWORD_ANCHORS.get("material_code", {})
    keywords = kw_config.get("keywords", ["MATERIAL CODE"])
    fuzzy_thresh = kw_config.get("fuzzy_threshold", 0.70)

    img_h, img_w = image.shape[:2]

    for idx, table_bbox in enumerate(table_regions):
        logger.debug(f"  尝试表格区域[{idx}]: {table_bbox}")
        # 对表头区域做 OCR（取表格顶部 ~20% 或至少 100px）
        header_h = max(int(table_bbox.h * 0.2), min(100, table_bbox.h))
        header_bbox = BBox(table_bbox.x, table_bbox.y, table_bbox.w, header_h)

        ocr_results = _ocr_region(image, header_bbox)
        match = _fuzzy_find_keyword(ocr_results, keywords, fuzzy_thresh)
        if match is None:
            # 英文OCR未找到，尝试中文OCR（识别"零部件图号"等中文关键词）
            ocr_results_ch = _ocr_region(image, header_bbox, lang=OCR_LANG_CH)
            match = _fuzzy_find_keyword(ocr_results_ch, keywords, fuzzy_thresh)
            if match is not None:
                ocr_results = ocr_results_ch  # 后续邻居搜索也用中文结果
            else:
                continue

        logger.info(
            f"找到 MATERIAL CODE 关键词: '{match['text']}' "
            f"(匹配 '{match['keyword']}', score={match['score']:.2f})"
        )

        # 在同一批 OCR 结果中搜索相邻列关键词（DEF / 品群）
        # MATERIAL CODE 关键词的 x 中心（table ROI 坐标）
        kw_poly = match.get("poly")
        mat_cx = (
            sum(p[0] for p in kw_poly) / len(kw_poly) - table_bbox.x
            if kw_poly else table_bbox.w / 2
        )

        neighbors = {}
        def_candidates = []
        shin_candidates = []
        for text, conf, poly in ocr_results:
            if poly is None:
                continue
            text_up = text.upper().strip()
            cx = sum(p[0] for p in poly) / len(poly) - table_bbox.x
            # DEF 列（MATERIAL CODE 左邻）
            if text_up in ("DEF", "DEF.") or (
                len(text_up) <= 5
                and difflib.SequenceMatcher(None, text_up, "DEF").ratio() > 0.7
            ):
                def_candidates.append((cx, text))
            # 品/群 列（MATERIAL CODE 右邻）
            if "品" in text or "群" in text:
                shin_candidates.append((cx, text))

        # DEF：选在 MATERIAL CODE 左边、且最近的
        left_defs = [(cx, t) for cx, t in def_candidates if cx < mat_cx]
        if left_defs:
            best_def = max(left_defs, key=lambda d: d[0])  # 最靠右的（最近）
            neighbors["def_cx"] = best_def[0]
            logger.info(f"  DEF 邻列: '{best_def[1]}' @ x={best_def[0]:.0f}")
        elif def_candidates:
            # 没有在左边的，取最近的
            best_def = min(def_candidates, key=lambda d: abs(d[0] - mat_cx))
            neighbors["def_cx"] = best_def[0]
            logger.info(f"  DEF 邻列(回退): '{best_def[1]}' @ x={best_def[0]:.0f}")

        # 品/群：选在 MATERIAL CODE 右边、且最近的
        right_shins = [(cx, t) for cx, t in shin_candidates if cx > mat_cx]
        if right_shins:
            best_shin = min(right_shins, key=lambda s: s[0])  # 最靠左的（最近）
            neighbors["shingun_cx"] = best_shin[0]
            logger.info(f"  品/群 邻列: '{best_shin[1]}' @ x={best_shin[0]:.0f}")
        elif shin_candidates:
            best_shin = min(shin_candidates, key=lambda s: abs(s[0] - mat_cx))
            neighbors["shingun_cx"] = best_shin[0]
            logger.info(f"  品/群 邻列(回退): '{best_shin[1]}' @ x={best_shin[0]:.0f}")

        # 找到关键词了，现在在这个表格区域中定位具体列
        table_roi_gray = cv2.cvtColor(table_bbox.crop(image), cv2.COLOR_RGB2GRAY)

        # 所有图纸均为竖向布局，直接调用竖向追踪
        direction = "vertical"
        col_bbox = _trace_vertical_table(
            image, table_bbox, match, table_roi_gray, neighbors
        )

        if col_bbox is not None:
            if not _validate_material_code_column(image, col_bbox):
                logger.warning("列内容验证失败，继续尝试下一个区域")
                continue
            return col_bbox, table_bbox, direction

    return None, None, None


def _get_vline_y_extent(
    table_gray: np.ndarray, x_pos: int, kw_cy: float, gap_threshold: int = 80
) -> tuple[int | None, int | None]:
    """检测 x_pos 处竖线的 y 范围（包含 kw_cy，并链接断裂片段）。"""
    roi_h, roi_w = table_gray.shape[:2]
    x_lo = max(0, x_pos - 5)
    x_hi = min(roi_w, x_pos + 6)
    strip = table_gray[:, x_lo:x_hi]
    denoised = cv2.GaussianBlur(strip, (3, 3), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 收集所有片段，按 y 排序
    segments = []
    for c in contours:
        _, cy, _, ch = cv2.boundingRect(c)
        segments.append((cy, cy + ch))
    segments.sort()

    # 找包含 kw_cy 的主片段
    primary = None
    for y_start, y_end in segments:
        if y_start <= kw_cy + 10 and y_end >= kw_cy - 10:
            primary = (y_start, y_end)
            break

    if primary is None:
        return None, None

    # 向下/向上链接：间距 ≤ gap_threshold 的相邻片段合并
    y_min, y_max = primary
    changed = True
    while changed:
        changed = False
        for y_start, y_end in segments:
            if y_start > y_max and y_start - y_max <= gap_threshold:
                y_max = y_end
                changed = True
            if y_end < y_min and y_min - y_end <= gap_threshold:
                y_min = y_start
                changed = True

    return y_min, y_max


def _trace_vertical_table(
    image: np.ndarray,
    table_bbox: BBox,
    keyword_match: dict,
    table_gray: np.ndarray,
    neighbors: dict | None = None,
) -> BBox | None:
    """追踪竖向 material code 表格（列式布局）。

    策略：
    1. 在关键词 y 附近窄带中检测 v_lines（避免混入图纸内容区线条）
    2. 用 DEF / 品群 锚点 + v_lines 确定列宽（满足其一即可）
    3. 用列边界竖线的 y 范围确定表格高度（竖线只存在于表格范围内）
    4. 在列区域内检测 h_lines 确定数据行边界
    """
    roi_h, roi_w = table_gray.shape[:2]
    neighbors = neighbors or {}

    # ── ① 获取关键词在 table ROI 中的位置 ──
    if keyword_match.get("poly"):
        poly = keyword_match["poly"]
        kw_cx = sum(p[0] for p in poly) / len(poly) - table_bbox.x
        kw_cy = sum(p[1] for p in poly) / len(poly) - table_bbox.y
    else:
        kw_cx = roi_w / 2
        kw_cy = roi_h * 0.1

    # ── ② 粗定位：窄带 v_lines + 表格 y 范围 ──
    # 先用窄带找到大致的列区域，然后用竖线确定表格 y 范围

    # 水平裁剪：只搜索 kw_cx 附近区域，避免 ROI 过大时拾取远处结构线
    x_search_half = max(400, int(roi_w * 0.15))
    x_lo = max(0, int(kw_cx - x_search_half))
    x_hi = min(roi_w, int(kw_cx + x_search_half))

    # 第一轮：关键词附近窄带（高阈值，检测强竖线）
    band_half = max(100, int(roi_h * 0.05))
    band_top = max(0, int(kw_cy - band_half))
    band_bot = min(roi_h, int(kw_cy + band_half))
    band = table_gray[band_top:band_bot, x_lo:x_hi]

    band_h = band.shape[0]
    v_min_h_band = max(int(band_h * 0.3), 15)
    v_lines_band, _, _ = _detect_vertical_lines(band, min_line_height=v_min_h_band)
    v_lines_band = [vx + x_lo for vx in v_lines_band]

    logger.debug(f"窄带 v_lines (x=[{x_lo},{x_hi}], y=[{band_top},{band_bot}]): {v_lines_band}")

    # 第二轮：如果窄带不足，向下扩展搜索（表格体在关键词下方）
    if len(v_lines_band) < 2:
        ext_top = max(0, int(kw_cy - 50))
        ext_bot = min(roi_h, int(kw_cy + int(roi_h * 0.3)))
        ext_band = table_gray[ext_top:ext_bot, x_lo:x_hi]
        ext_h = ext_band.shape[0]
        v_min_h_ext = max(int(ext_h * 0.1), 20)
        v_lines_band, _, _ = _detect_vertical_lines(ext_band, min_line_height=v_min_h_ext)
        v_lines_band = [vx + x_lo for vx in v_lines_band]
        logger.debug(
            f"扩展搜索 v_lines (x=[{x_lo},{x_hi}], y=[{ext_top},{ext_bot}]): "
            f"{v_lines_band}"
        )

    # 验证：检测到的 v_lines 是否在 kw_cx 附近（排除来自其他结构的线）
    if len(v_lines_band) >= 2:
        vmin, vmax = min(v_lines_band), max(v_lines_band)
        if kw_cx < vmin - 200 or kw_cx > vmax + 200:
            logger.debug(
                f"v_lines [{vmin},{vmax}] 距 kw_cx={kw_cx:.0f} 太远，丢弃"
            )
            v_lines_band = []

    # 第三轮：垂直投影法（对断裂/模糊线更鲁棒，不要求连续像素）
    if len(v_lines_band) < 2:
        ext_top = max(0, int(kw_cy - 50))
        ext_bot = min(roi_h, int(kw_cy + int(roi_h * 0.3)))
        proj_band = table_gray[ext_top:ext_bot, x_lo:x_hi]
        proj_lines = _detect_vlines_by_projection(proj_band)
        proj_lines = [px + x_lo for px in proj_lines]
        logger.debug(
            f"投影法 v_lines (x=[{x_lo},{x_hi}], y=[{ext_top},{ext_bot}]): "
            f"{proj_lines}"
        )
        if len(proj_lines) >= 2:
            pmin, pmax = min(proj_lines), max(proj_lines)
            if kw_cx >= pmin - 200 and kw_cx <= pmax + 200:
                v_lines_band = proj_lines
            else:
                logger.debug(
                    f"投影 v_lines [{pmin},{pmax}] 距 kw_cx={kw_cx:.0f} 太远"
                )

    # 第四轮回退：v_lines 不可用时，用关键词+锚点位置估算列边界
    use_kw_fallback = False
    if len(v_lines_band) < 2:
        kw_poly = keyword_match.get("poly")
        if kw_poly:
            kw_left = min(p[0] for p in kw_poly) - table_bbox.x
            kw_right = max(p[0] for p in kw_poly) - table_bbox.x
            kw_width = kw_right - kw_left
            margin = max(int(kw_width * 0.05), 5)  # 列宽≈文字宽度,仅+5%边距
            col_left = int(kw_left - margin)
            col_right = int(kw_right + margin)

            # 用 DEF 锚点约束左边界（仅防止侵入 DEF 列，不扩大列宽）
            def_cx = neighbors.get("def_cx")
            if def_cx is not None and def_cx < kw_left:
                separator = int((def_cx + kw_left) / 2)
                if col_left < separator:
                    col_left = separator

            col_w = col_right - col_left
            table_y_min = max(0, int(kw_cy) - 20)
            table_y_max = roi_h
            use_kw_fallback = True
            logger.info(
                f"V_lines 回退: 用关键词估算列=[{col_left},{col_right}], w={col_w}, "
                f"kw=[{kw_left:.0f},{kw_right:.0f}], def_cx={def_cx}"
            )
        else:
            logger.warning("窄带中垂直线不足且无关键词多边形，竖向追踪失败")
            return None

    if not use_kw_fallback:
        # ── v_lines 正常路径 ──

        # 粗定位 col_left / col_right（kw_cx 所在的 v_line 间隔）
        rough_col_left = None
        rough_col_right = None
        for i in range(len(v_lines_band) - 1):
            if v_lines_band[i] <= kw_cx <= v_lines_band[i + 1]:
                rough_col_left = v_lines_band[i]
                rough_col_right = v_lines_band[i + 1]
                break
        if rough_col_left is None:
            rough_col_left = min(v_lines_band, key=lambda vx: abs(vx - kw_cx))
            idx = v_lines_band.index(rough_col_left)
            rough_col_right = v_lines_band[min(idx + 1, len(v_lines_band) - 1)]

        # ── ③ 表格 y 范围：用两侧竖线 y 范围的并集确定 ──
        left_ymin, left_ymax = _get_vline_y_extent(table_gray, rough_col_left, kw_cy)
        right_ymin, right_ymax = _get_vline_y_extent(table_gray, rough_col_right, kw_cy)

        logger.debug(
            f"竖线 y 范围: 左@{rough_col_left}=[{left_ymin},{left_ymax}], "
            f"右@{rough_col_right}=[{right_ymin},{right_ymax}]"
        )

        table_y_min, table_y_max = 0, roi_h
        if left_ymin is not None and right_ymin is not None:
            table_y_min = min(left_ymin, right_ymin)
            table_y_max = max(left_ymax, right_ymax)
            logger.info(
                f"表格 y 范围(并集): [{table_y_min},{table_y_max}] "
                f"(左={left_ymax - left_ymin}, 右={right_ymax - right_ymin})"
            )
        elif left_ymin is not None:
            table_y_min, table_y_max = left_ymin, left_ymax
            logger.info(f"表格 y 范围(仅左): [{table_y_min},{table_y_max}]")
        elif right_ymin is not None:
            table_y_min, table_y_max = right_ymin, right_ymax
            logger.info(f"表格 y 范围(仅右): [{table_y_min},{table_y_max}]")
        else:
            logger.warning("未检测到列边界竖线 y 范围，使用全高")

        # ── ④ 精定位 v_lines：在表格 y 范围内重新检测 ──
        rough_col_w = rough_col_right - rough_col_left
        x_margin = max(200, rough_col_w)
        x_lo_fine = max(0, rough_col_left - x_margin)
        x_hi_fine = min(roi_w, rough_col_right + x_margin)
        table_region = table_gray[table_y_min:table_y_max, x_lo_fine:x_hi_fine]
        table_h = table_y_max - table_y_min
        v_min_h_full = max(int(table_h * 0.15), 15)
        v_lines, _, _ = _detect_vertical_lines(table_region, min_line_height=v_min_h_full)
        v_lines = [vx + x_lo_fine for vx in v_lines]

        logger.info(
            f"精定位 v_lines ({len(v_lines)}条, x搜索范围=[{x_lo_fine},{x_hi_fine}]): "
            f"{v_lines}"
        )

        if len(v_lines) < 2:
            v_lines = v_lines_band

        # ── ⑤ 列宽：kw_cx 优先定位 + 锚点验证 ──
        col_left, col_right = None, None
        def_cx = neighbors.get("def_cx")
        shin_cx = neighbors.get("shingun_cx")

        for i in range(len(v_lines) - 1):
            if v_lines[i] <= kw_cx <= v_lines[i + 1]:
                col_left = v_lines[i]
                col_right = v_lines[i + 1]
                break
        logger.info(f"kw_cx={kw_cx:.0f} → 初始列=[{col_left}, {col_right}]")

        if col_left is not None and col_right is not None:
            # ── DEF 锚点验证：如果 DEF 在列内，尝试用 v_line 分离 ──
            if def_cx is not None and col_left <= def_cx <= col_right:
                # DEF 在列内 → 列可能包含了 DEF 列，寻找 DEF 与 kw_cx 之间的分隔线
                better_left = None
                for vx in sorted(v_lines):
                    if def_cx < vx < kw_cx:
                        better_left = vx
                        break
                if better_left is not None:
                    col_left = better_left
                    logger.info(
                        f"DEF@{def_cx:.0f}在列内, 用分隔线修正 col_left={col_left}"
                    )
                else:
                    logger.info(f"DEF@{def_cx:.0f}在列内(同列), 列=[{col_left}, {col_right}]")
            elif def_cx is not None:
                logger.info(f"DEF@{def_cx:.0f}在列外, 列=[{col_left}, {col_right}]")

            # ── 品/群 锚点验证：如果品群在列内，缩窄右边界 ──
            if shin_cx is not None and col_left <= shin_cx <= col_right:
                for vx in v_lines:
                    if kw_cx < vx < shin_cx:
                        col_right = vx
                        break
                logger.info(f"品群@{shin_cx:.0f}在列内, 修正后=[{col_left}, {col_right}]")
            elif shin_cx is not None:
                logger.info(f"品群@{shin_cx:.0f}在右侧, 列=[{col_left}, {col_right}]")

        if col_left is None or col_right is None:
            col_left, col_right = None, None
            best_dist = float("inf")
            for i in range(len(v_lines) - 1):
                mid = (v_lines[i] + v_lines[i + 1]) / 2
                d = abs(mid - kw_cx)
                if d < best_dist:
                    best_dist = d
                    col_left = v_lines[i]
                    col_right = v_lines[i + 1]
            logger.info(f"关键词回退: col=[{col_left}, {col_right}]")

        if col_left is None or col_right is None:
            logger.warning("无法确定列边界")
            return None

        col_w = col_right - col_left
        logger.info(f"列宽确定: x=[{col_left},{col_right}], w={col_w}")

        MAX_REASONABLE_COL_WIDTH = 600
        if col_w > MAX_REASONABLE_COL_WIDTH:
            logger.warning(
                f"列宽 {col_w}px 超过合理范围 ({MAX_REASONABLE_COL_WIDTH}px)，"
                f"精定位结果不可靠"
            )
            # 精定位的 v_lines 不可靠，回退到步骤②的 v_lines_band
            if len(v_lines_band) >= 2:
                col_left, col_right = None, None
                for i in range(len(v_lines_band) - 1):
                    if v_lines_band[i] <= kw_cx <= v_lines_band[i + 1]:
                        col_left = v_lines_band[i]
                        col_right = v_lines_band[i + 1]
                        break
                if col_left is not None:
                    col_w = col_right - col_left
                    if col_w <= MAX_REASONABLE_COL_WIDTH:
                        logger.info(
                            f"回退到步骤②结果: x=[{col_left},{col_right}], w={col_w}"
                        )
                    else:
                        logger.warning("步骤②结果也超宽，放弃")
                        return None
                else:
                    logger.warning("步骤② v_lines 中找不到包含 kw_cx 的区间")
                    return None
            else:
                return None

    # ── ⑥ H_LINES：在表格 y 范围 + 列 x 范围内检测 ──
    region_x_lo = max(0, col_left - 5)
    region_x_hi = min(roi_w, col_right + 5)
    h_min_w = max(int(col_w * 0.3), 10)

    # 初始范围内的 h_lines
    region_y_lo = max(0, table_y_min)
    region_y_hi = min(roi_h, table_y_max)
    col_region = table_gray[region_y_lo:region_y_hi, region_x_lo:region_x_hi]
    h_lines_local = _detect_horizontal_lines(col_region, min_line_width=h_min_w)
    h_lines = [hy + region_y_lo for hy in h_lines_local]

    # 向下延伸搜索：竖线可能在水平线交叉处断裂，但 h_lines 可能还在继续
    # 计算行高，然后逐步向下搜索更多 h_lines
    if len(h_lines) >= 2:
        row_heights = [h_lines[i + 1] - h_lines[i] for i in range(len(h_lines) - 1)]
        avg_row_h = sum(row_heights) / len(row_heights)
        extend_step = max(int(avg_row_h * 3), 100)  # 每次搜索 3 行高度
        search_top = table_y_max
        search_limit = min(roi_h, table_y_max + extend_step * 3)  # 最多延伸 ~9 行
        while search_top < search_limit:
            search_bot = min(roi_h, search_top + extend_step)
            ext_region = table_gray[search_top:search_bot, region_x_lo:region_x_hi]
            ext_local = _detect_horizontal_lines(ext_region, min_line_width=h_min_w)
            ext_lines = [hy + search_top for hy in ext_local]
            if not ext_lines:
                break  # 没有更多 h_lines → 表格结束
            h_lines.extend(ext_lines)
            search_top = ext_lines[-1] + 1  # 从最后一条 h_line 后继续

    logger.debug(f"列区域 h_lines ({len(h_lines)}条): {h_lines}")

    # ── ⑦ 数据区边界 ──
    data_y_top = None
    for hy in h_lines:
        if hy >= kw_cy - 5:
            data_y_top = hy
            break
    if data_y_top is None and h_lines:
        data_y_top = h_lines[0]
    if data_y_top is None:
        data_y_top = table_y_min
        logger.warning("未找到水平线，使用竖线上边界")

    if h_lines:
        data_y_bot = h_lines[-1]
    else:
        data_y_bot = table_y_max

    result = BBox(
        table_bbox.x + col_left,
        table_bbox.y + data_y_top,
        col_w,
        data_y_bot - data_y_top,
    )
    logger.info(f"竖向追踪结果: {result}")

    # 高度验证回退：如果红框太矮，用更大的gap重试
    MIN_REASONABLE_HEIGHT = 500
    if result.h < MIN_REASONABLE_HEIGHT:
        logger.warning(
            f"红框高度 {result.h}px 太矮（<{MIN_REASONABLE_HEIGHT}px），"
            f"用 gap_threshold=120 重试"
        )

        # 重新计算竖线y范围（更大的gap）
        left_y_min, left_y_max = _get_vline_y_extent(
            table_gray, col_left, kw_cy, gap_threshold=120
        )
        right_y_min, right_y_max = _get_vline_y_extent(
            table_gray, col_right, kw_cy, gap_threshold=120
        )

        if left_y_min and right_y_min:
            retry_y_min = min(left_y_min, right_y_min)
        else:
            retry_y_min = left_y_min or right_y_min or 0

        if left_y_max and right_y_max:
            retry_y_max = max(left_y_max, right_y_max)
        else:
            retry_y_max = left_y_max or right_y_max or roi_h

        # 简化：直接使用扩展后的y范围，但需要找到第一条横线作为上边界
        # 重新检测 h_lines
        retry_region_y_lo = max(0, retry_y_min)
        retry_region_y_hi = min(roi_h, retry_y_max)
        retry_col_region = table_gray[retry_region_y_lo:retry_region_y_hi, region_x_lo:region_x_hi]
        retry_h_lines_local = _detect_horizontal_lines(retry_col_region, min_line_width=h_min_w)
        retry_h_lines = [hy + retry_region_y_lo for hy in retry_h_lines_local]

        # 找到第一条横线作为数据区上边界
        retry_data_y_top = None
        for hy in retry_h_lines:
            if hy >= kw_cy - 5:
                retry_data_y_top = hy
                break
        if retry_data_y_top is None and retry_h_lines:
            retry_data_y_top = retry_h_lines[0]
        if retry_data_y_top is None:
            retry_data_y_top = retry_y_min

        retry_data_y_bot = retry_h_lines[-1] if retry_h_lines else retry_y_max

        result = BBox(
            table_bbox.x + col_left,
            table_bbox.y + retry_data_y_top,
            col_w,
            retry_data_y_bot - retry_data_y_top,
        )
        logger.info(f"重试后红框: {result}")

    return result


# ══════════════════════════════════════════════════════════════════
#  Phase B-2: 定位右下角编号栏
# ══════════════════════════════════════════════════════════════════

def _fuzzy_fix_y_text(raw_text: str, prefixes: list[str] = None) -> str | None:
    """对去空格后的OCR文本尝试符号→数字模糊回填，返回修复后的9位编号或None。

    仅处理非字母数字的符号字符（如 / ! | 等），字母不参与回填。
    """
    y_re = re.compile(make_pattern(prefixes))
    result = []
    for ch in raw_text:
        if ch == ' ':
            continue
        if re.match(r'[A-Z0-9]', ch):
            result.append(ch)
        elif ch in FUZZY_DIGIT_MAP:
            result.append(FUZZY_DIGIT_MAP[ch])
        else:
            # 未知符号，无法回填
            return None
    fixed = ''.join(result)
    if len(fixed) == 9 and y_re.match(fixed):
        return fixed
    return None


def _search_y_number(
    image: np.ndarray,
    search: BBox,
    y_re,
    loose_re,
    material_code_bbox: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在指定搜索区域内找 Y 编号，返回 (文本, BBox)。"""
    ocr_results = _ocr_region(image, search)

    best_match = None
    for text, conf, poly in ocr_results:
        text_clean = text.upper().replace(" ", "")
        m = y_re.search(text_clean)
        if not m:
            # 模糊回填：符号→数字
            fuzzy_text = _fuzzy_fix_y_text(text.upper().replace(" ", ""), prefixes=prefixes)
            if fuzzy_text:
                m = y_re.search(fuzzy_text)
                if m:
                    text_clean = fuzzy_text
                    logger.info(f"  模糊回填: '{text.upper().replace(' ', '')}' → '{fuzzy_text}'")
        if not m:
            m_loose = loose_re.search(text_clean)
            if not m_loose:
                continue
            if len(text_clean) < 9:
                continue
        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            text_bbox = BBox(
                int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys)),
            )
            if material_code_bbox and material_code_bbox.contains(text_bbox):
                continue
            area = text_bbox.w * text_bbox.h
            # 保存正则匹配到的文本（优先Y_PATTERN的精确匹配）
            matched_group = m.group() if m else text_clean
            if best_match is None or area > best_match[1]:
                best_match = (text_bbox, area, matched_group)

    if not best_match:
        # ── 重试：疑似Y编号（7-8位）裁剪放大重新OCR ──
        p_chars = "".join(p.upper() for p in (prefixes or DEFAULT_PREFIXES))
        partial_re = re.compile(rf'^[{p_chars}][A-Z0-9]{{6,7}}$')
        for text, conf, poly in ocr_results:
            text_clean = text.upper().replace(" ", "")
            if not partial_re.match(text_clean):
                continue
            if poly is None:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            tb = BBox(int(min(xs)), int(min(ys)),
                      int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            if material_code_bbox and material_code_bbox.contains(tb):
                continue
            logger.info(f"  疑似Y编号 '{text_clean}'({len(text_clean)}位), 重试OCR...")
            # 左右各扩展一个字符宽度，覆盖可能被截断的末位
            char_w = tb.w / max(len(text_clean), 1)
            retry_bbox = BBox(
                max(0, tb.x - int(char_w)),
                max(0, tb.y - int(tb.h * 0.1)),
                tb.w + int(char_w * 2),
                tb.h + int(tb.h * 0.2),
            )
            # 裁剪 → 放大3x → 二值化 → 重新OCR
            retry_roi = retry_bbox.crop(image)
            retry_roi = cv2.resize(retry_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray_retry = cv2.cvtColor(retry_roi, cv2.COLOR_RGB2GRAY)
            _, binarized = cv2.threshold(gray_retry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            retry_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
            _get_ocr, _parse_ocr_results = _get_ocr_funcs()
            ocr_engine = _get_ocr(OCR_LANG_EN)
            retry_result = ocr_engine.predict(retry_rgb)
            retry_items = _parse_ocr_results(retry_result)
            for _, rt, rc in retry_items:
                rt_clean = rt.upper().replace(" ", "")
                m2 = y_re.search(rt_clean)
                if m2:
                    best_match = (tb, tb.w * tb.h, m2.group())
                    logger.info(f"  重试OCR成功: '{rt_clean}' → '{m2.group()}'")
                    break
            if best_match:
                break

        # 重试也失败 → 接受疑似结果，扩展bbox宽度到9位估算
        if not best_match:
            for text, conf, poly in ocr_results:
                text_clean = text.upper().replace(" ", "")
                if not partial_re.match(text_clean):
                    continue
                if poly is None:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                tb = BBox(int(min(xs)), int(min(ys)),
                          int(max(xs) - min(xs)), int(max(ys) - min(ys)))
                if material_code_bbox and material_code_bbox.contains(tb):
                    continue
                char_w = tb.w / len(text_clean)
                estimated_w = int(char_w * 9)
                expanded_tb = BBox(tb.x, tb.y, estimated_w, tb.h)
                best_match = (expanded_tb, expanded_tb.w * expanded_tb.h, text_clean)
                logger.info(f"  接受疑似Y编号: '{text_clean}', bbox扩展 {tb.w}→{estimated_w}px")
                break

    if not best_match:
        return None

    text_bbox, _, matched_text = best_match
    logger.info(f"右下角编号: '{matched_text}' at {text_bbox}")

    # 直接使用文字 bbox（不加边距）
    result = text_bbox

    # Y 字符补偿（OCR 漏掉开头 Y）
    if not y_re.search(matched_text):
        char_w = text_bbox.w / max(len(matched_text), 1)
        ext = int(char_w)
        result = BBox(result.x - ext, result.y, result.w + ext, result.h)
        logger.info(f"  Y字符补偿: 左扩展 {ext}px → {result}")

    return (matched_text, result)


def _locate_bottom_right_number(
    image: np.ndarray,
    material_code_bbox: BBox | None = None,
    search: BBox | None = None,
    frame: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在右下角区域找到编号栏。

    策略：在 search 区域 OCR 找编号，文字宽度+5%，上下找横线。
    返回 (文本, BBox) 或 None。
    """
    img_h, img_w = image.shape[:2]
    y_re = re.compile(make_pattern(prefixes))
    loose_re = re.compile(r"[A-Z][A-Z0-9]*\d{2,}[A-Z]\d{2,}")

    # 回退：未传入 search/frame 时自行检测
    if frame is None:
        frame = _detect_drawing_frame(image)
    if search is None:
        search = BBox(
            frame.x + int(frame.w * 0.65),
            frame.y + int(frame.h * 0.85),
            int(frame.w * 0.35),
            int(frame.h * 0.15),
        )
    logger.info(f"搜索区域: {search}")

    # ── OCR 找 Y 编号 ──
    result = _search_y_number(image, search, y_re, loose_re, material_code_bbox, prefixes=prefixes)

    # 仍未找到9位 → 扩展搜索区右边界10%重试（仅绿框）
    if result is None or len(result[0]) < 9:
        expanded_w = min(search.w + int(search.w * 0.10), img_w - search.x)
        expanded_search = BBox(search.x, search.y, expanded_w, search.h)
        logger.info(f"  绿框搜索区扩展10%重试: {search} → {expanded_search}")
        result2 = _search_y_number(image, expanded_search, y_re, loose_re, material_code_bbox, prefixes=prefixes)
        if result2 and (result is None or len(result2[0]) > len(result[0])):
            result = result2

    if result is None:
        return None

    y_text, text_bbox = result
    logger.info(f"找到右下角编号: '{y_text}' at {text_bbox}")

    # ── 宽度：文字宽度+5%，居中 ──
    final_w = int(text_bbox.w * 1.05)
    pad_w = (final_w - text_bbox.w) // 2
    final_x = text_bbox.x - pad_w

    # ── 上下边界：紧贴文字 +5% padding ──
    top_y = text_bbox.y - int(text_bbox.h * 0.05)
    bot_y = text_bbox.y + text_bbox.h + int(text_bbox.h * 0.05)

    final_bbox = BBox(final_x, top_y, final_w, bot_y - top_y)

    # ── 竖线约束：绿框不得超过两侧第一条竖线 ──
    # 检测两类竖线：text_bbox外的 + text_bbox内的单元格分隔线
    gray_roi = cv2.cvtColor(search.crop(image), cv2.COLOR_RGB2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    min_vh = max(int(text_bbox.h * 0.5), 10)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_mask = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, v_kernel)
    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 只保留与文字y范围重叠的竖线，同时记录高度
    ty_local_top = text_bbox.y - search.y - 20
    ty_local_bot = text_bbox.y - search.y + text_bbox.h + 20
    vline_info = []  # (global_x, height)
    for c in v_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if y < ty_local_bot and y + bh > ty_local_top:
            vline_info.append((search.x + x + bw // 2, bh))
    vline_info.sort(key=lambda v: v[0])

    text_right = text_bbox.x + text_bbox.w

    # 右侧约束候选：
    # 1) text_bbox 右边界之外的竖线（任何高度）
    # 2) text_bbox 内部但高度>=text_bbox.h*0.8的竖线（贯穿的单元格分隔线，非字符笔画）
    right_candidates = []
    for vx, vh in vline_info:
        if vx > text_right:
            right_candidates.append(vx)
        elif vx > text_bbox.x + text_bbox.w * 0.5 and vh >= text_bbox.h * 0.8:
            # 内部的贯穿竖线（在文字后半部分，高度接近文字高度）
            right_candidates.append(vx)
            logger.info(f"  检测到内部单元格分隔线: x={vx}, h={vh}")

    if right_candidates:
        right_limit = min(right_candidates)
        if final_bbox.x + final_bbox.w > right_limit:
            final_bbox = BBox(final_bbox.x, final_bbox.y, right_limit - final_bbox.x, final_bbox.h)
            logger.info(f"  绿框右边界受竖线约束: right_limit={right_limit}")

    # 左侧约束：找 text_bbox 左侧的最近竖线
    left_vlines = [vx for vx, _ in vline_info if vx < text_bbox.x]
    if left_vlines:
        left_limit = left_vlines[-1]
        if final_bbox.x < left_limit:
            old_right = final_bbox.x + final_bbox.w
            final_bbox = BBox(left_limit, final_bbox.y, old_right - left_limit, final_bbox.h)
            logger.info(f"  绿框左边界受竖线约束: left_limit={left_limit}")

    logger.info(f"绿框结果: {text_bbox} → {final_bbox}")
    return (y_text, final_bbox)




# ══════════════════════════════════════════════════════════════════
#  Phase B-4: 定位左下角竖排编号
# ══════════════════════════════════════════════════════════════════

def _locate_bottom_left_number(
    image: np.ndarray,
    material_code_bbox: BBox | None = None,
    search: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox, list] | None:
    """在左下角区域找到竖排 Y 编号。

    返回 (文本, 原图坐标BBox, 旋转后OCR结果) 或 None。
    """
    if search is None:
        return None

    img_h, img_w = image.shape[:2]
    # 截取搜索区 ROI
    roi = image[search.y:search.y + search.h, search.x:search.x + search.w]
    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]

    # 右旋90°：竖排文字变横排
    rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    # OCR 扫描旋转后图像
    rot_h, rot_w = rotated.shape[:2]
    full_bbox = BBox(0, 0, rot_w, rot_h)
    ocr_results = _ocr_region(rotated, full_bbox, lang="en")

    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        text_up = text.upper().strip()
        prefixes_upper = {p.upper() for p in (prefixes or DEFAULT_PREFIXES)}
        if not text_up or text_up[0] not in prefixes_upper:
            continue
        # 首字母匹配 → 命中
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        rx = int(min(xs))
        ry = int(min(ys))
        rw = int(max(xs) - min(xs))
        rh = int(max(ys) - min(ys))

        # 映射回原图 ROI 坐标（右旋90°CW 逆映射）
        # rotated(rx, ry) → orig(ry, roi_h - rx - rw)
        # rotated(rw, rh) → orig(rh, rw)
        orig_x = ry
        orig_y = roi_h - rx - rw
        orig_w = rh
        orig_h = rw

        # 全局坐标
        global_bbox = BBox(search.x + orig_x, search.y + orig_y, orig_w, orig_h)
        logger.info(f"  紫框找到竖排 Y 编号: '{text_up}' "
                    f"旋转后BBox({rx},{ry},{rw},{rh}) → 原图{global_bbox}")
        return (text_up, global_bbox, ocr_results)

    logger.info("  紫框未找到 Y 编号")
    return None


# ══════════════════════════════════════════════════════════════════
#  Phase B-3: 定位左上角编号栏
# ══════════════════════════════════════════════════════════════════

def _locate_top_left_number(
    image: np.ndarray,
    material_code_bbox: BBox | None = None,
    search: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在左上角区域找到编号。

    策略：
    1. OCR 左上角区域，直接搜索独立的编号文字
    2. 若编号与 DWG NO 合体识别，用字符比例从合体文本中估算编号位置
    3. 兜底搜索所有 OCR 文本
    返回 (文本, BBox) 或 None。
    """
    img_h, img_w = image.shape[:2]
    y_re = re.compile(make_pattern(prefixes))

    # 搜索区域：使用传入的 search，或回退到默认
    if search is None:
        search = BBox(0, 0, int(img_w * 0.35), int(img_h * 0.20))
    ocr_results = _ocr_region(image, search)

    # 生成最终 bbox：宽度+5%居中，上下紧贴文字+5% padding
    def _make_final_bbox(text_bbox: BBox, text_len: int = 9) -> BBox:
        # 如果识别到的文本不足9位，按字符比例扩展到9位宽度
        if text_len < 9 and text_len > 0:
            char_w = text_bbox.w / text_len
            expanded_w = int(char_w * 9)
            logger.info(f"  橙框bbox扩展: {text_len}位→9位, {text_bbox.w}→{expanded_w}px")
            text_bbox = BBox(text_bbox.x, text_bbox.y, expanded_w, text_bbox.h)

        return text_bbox

    def _clip_by_vlines(final_bbox: BBox) -> BBox:
        """用搜索区内竖线约束橙框右边界（不超过文字右侧最近竖线）"""
        if search is None:
            return final_bbox
        roi = image[search.y:search.y + search.h, search.x:search.x + search.w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        min_v_h = max(final_bbox.h, 20)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_h))
        v_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fb_right = final_bbox.x + final_bbox.w
        # 文字中心x，用于排除文字左侧的竖线
        text_cx = final_bbox.x + final_bbox.w // 2
        right_limit = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            vx = search.x + x + w // 2
            # 只看文字右半部分之后的竖线
            if vx > text_cx and vx < fb_right:
                if right_limit is None or vx < right_limit:
                    right_limit = vx

        if right_limit is not None:
            new_w = right_limit - final_bbox.x
            if new_w < final_bbox.w:
                logger.info(f"  橙框竖线约束: right_limit={right_limit}, w:{final_bbox.w}→{new_w}")
                return BBox(final_bbox.x, final_bbox.y, new_w, final_bbox.h)
        return final_bbox

    # ── 策略 1：找独立的 Y 编号文字 ──
    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        text_up = text.upper().strip()
        # 跳过 DWG NO 合体文本（策略 2 处理）
        if any(text_up.startswith(prefix) for prefix in ("DWG", "OWG", "図")):
            continue
        m = y_re.search(text_up)
        if not m:
            m = y_re.search(text_up.replace(" ", ""))
        if not m:
            fuzzy_text = _fuzzy_fix_y_text(text_up.replace(" ", ""))
            if fuzzy_text:
                m = y_re.search(fuzzy_text)
                if m:
                    logger.info(f"  策略1模糊回填: '{text_up.replace(' ', '')}' → '{fuzzy_text}'")
        if m:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = BBox(int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            # 排除在 material_code 区域内的
            if material_code_bbox and material_code_bbox.contains(bbox):
                continue
            matched_text = m.group()
            # OCR文本可能含前缀/后缀/空格，按字符比例裁剪bbox到匹配范围
            raw_for_ratio = text  # 原始OCR文本（含空格，polygon对应此文本）
            if len(raw_for_ratio) > len(matched_text) and bbox.w > 0:
                char_w = bbox.w / len(raw_for_ratio)
                # 在去空格文本中找匹配位置
                no_sp = raw_for_ratio.upper().replace(" ", "")
                m_start = no_sp.find(matched_text)
                if m_start >= 0:
                    # 映射回原始文本中的字符位置
                    orig_start = 0
                    count = 0
                    for i, ch in enumerate(raw_for_ratio.upper()):
                        if ch != ' ':
                            if count == m_start:
                                orig_start = i
                                break
                            count += 1
                    new_x = bbox.x + int(char_w * orig_start)
                    new_w = int(char_w * len(matched_text))
                    logger.info(f"  橙框bbox裁剪: OCR='{raw_for_ratio}'({len(raw_for_ratio)}字符)→{len(matched_text)}字符, "
                                f"x:{bbox.x}→{new_x}, w:{bbox.w}→{new_w}")
                    bbox = BBox(new_x, bbox.y, new_w, bbox.h)
            # 竖排文字（h/w > 3）：直接使用 OCR bbox
            if bbox.w > 0 and bbox.h > bbox.w * 3:
                logger.info(f"  找到竖排 Y 编号: '{text}' at {bbox}")
                return (matched_text, bbox)
            logger.info(f"  找到独立 Y 编号: '{text}' at {bbox}")
            final_bbox = _make_final_bbox(bbox, len(matched_text))
            logger.info(f"  橙框结果: {bbox} → {final_bbox}")
            return (matched_text, _clip_by_vlines(final_bbox))

    # 诊断日志：策略1 未命中时输出 OCR 内容
    if not any(y_re.search(t.upper().replace(" ", "")) for t, _, _ in ocr_results):
        logger.info(f"  橙框 OCR 未发现任何 Y 编号 (共 {len(ocr_results)} 条)")
    for t, c, _ in ocr_results:
        logger.debug(f"    OCR: '{t}' (conf={c:.2f})")

    # ── 策略 2：从 DWG NO 合体文本中提取编号位置 ──
    dwg_keywords = ["DWG NO", "DWG NO.", "DWGNO", "DWG", "図番", "図面番号"]
    dwg_match = _fuzzy_find_keyword(ocr_results, dwg_keywords, threshold=0.60)

    if dwg_match and dwg_match.get("poly"):
        logger.info(
            f"  找到 DWG NO 关键词: '{dwg_match['text']}' "
            f"(score={dwg_match['score']:.2f})"
        )
        text_up = dwg_match["text"].upper()

        # OCR 可能在编号中插入空格（如 "YA026 D 941"），去除后再匹配
        # 先找 DWG NO 前缀的结束位置
        prefix_end = 0
        for prefix in ["DWG NO.", "DWG NO", "DWGNO", "DWG"]:
            if text_up.startswith(prefix):
                prefix_end = len(prefix)
                break
        while prefix_end < len(text_up) and text_up[prefix_end] in (" ", "."):
            prefix_end += 1

        # 编号部分（去空格后匹配 Y_PATTERN）
        num_part = text_up[prefix_end:]
        num_part_clean = num_part.replace(" ", "")
        m = y_re.search(num_part_clean)
        if not m:
            fuzzy_text = _fuzzy_fix_y_text(num_part_clean)
            if fuzzy_text:
                m = y_re.search(fuzzy_text)
                if m:
                    num_part_clean = fuzzy_text
                    logger.info(f"  策略2模糊回填: '{num_part.replace(' ', '')}' → '{fuzzy_text}'")
        if m:
            # 映射回原始文本中的字符位置
            clean_i, orig_i = 0, 0
            while clean_i < m.start() and orig_i < len(num_part):
                if num_part[orig_i] == " ":
                    orig_i += 1
                else:
                    clean_i += 1
                    orig_i += 1
            match_start_orig = prefix_end + orig_i

            while clean_i < m.end() and orig_i < len(num_part):
                if num_part[orig_i] == " ":
                    orig_i += 1
                else:
                    clean_i += 1
                    orig_i += 1
            match_end_orig = prefix_end + orig_i

            # 用字符位置比例估算编号在 polygon 中的 x 范围
            poly = dwg_match["poly"]
            poly_left = min(p[0] for p in poly)
            poly_right = max(p[0] for p in poly)
            poly_top = min(p[1] for p in poly)
            poly_bot = max(p[1] for p in poly)
            poly_w = poly_right - poly_left

            total_chars = max(len(text_up), 1)
            num_x_left = poly_left + poly_w * (match_start_orig / total_chars)
            num_x_right = poly_left + poly_w * (match_end_orig / total_chars)

            bbox = BBox(
                int(num_x_left), int(poly_top),
                max(int(num_x_right - num_x_left), 1),
                max(int(poly_bot - poly_top), 1),
            )
            matched_text = m.group()
            logger.info(
                f"  从 DWG NO 合体文本提取编号: '{matched_text}' at {bbox}"
            )
            final_bbox = _make_final_bbox(bbox, len(matched_text))
            logger.info(f"  橙框结果: {bbox} → {final_bbox}")
            return (matched_text, _clip_by_vlines(final_bbox))

    # ── 策略 3：兜底 — 在所有 OCR 文本中搜索 Y 编号 ──
    y_re_relaxed = re.compile(r"Y[A-Z0-9][A-Z0-9\-]{2,}")
    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        text_up = text.upper().strip()
        m = y_re.search(text_up)
        used_original = True
        if not m:
            text_clean = text_up.replace(" ", "")
            m = y_re_relaxed.search(text_clean)
            used_original = False
        if not m:
            fuzzy_text = _fuzzy_fix_y_text(text_up.replace(" ", ""))
            if fuzzy_text:
                m = y_re.search(fuzzy_text)
                if m:
                    text_clean = fuzzy_text
                    used_original = False
                    logger.info(f"  策略3模糊回填: '{text_up.replace(' ', '')}' → '{fuzzy_text}'")
        if m:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = BBox(int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            if material_code_bbox and material_code_bbox.contains(bbox):
                continue
            # 编号不在文本开头 → 用字符比例截取
            source_text = text_up if used_original else text_clean
            match_start = m.start()
            match_end = m.end()
            total_chars = max(len(source_text), 1)
            poly_left = min(p[0] for p in poly)
            poly_right = max(p[0] for p in poly)
            poly_w = poly_right - poly_left
            if match_start > 0 and poly_w > 0:
                num_x_left = poly_left + poly_w * (match_start / total_chars)
                num_x_right = poly_left + poly_w * (match_end / total_chars)
                poly_top = min(p[1] for p in poly)
                poly_bot = max(p[1] for p in poly)
                bbox = BBox(int(num_x_left), int(poly_top),
                            max(int(num_x_right - num_x_left), 1),
                            max(int(poly_bot - poly_top), 1))
            matched_text = m.group()
            logger.info(f"  策略3 兜底找到 Y 编号: '{text}' -> '{matched_text}' at {bbox}")
            final_bbox = _make_final_bbox(bbox, len(matched_text))
            logger.info(f"  橙框结果: {bbox} → {final_bbox}")
            return (matched_text, _clip_by_vlines(final_bbox))

    # ── 回退：DWG NO 找到但 Y 编号匹配失败 → 用 DWG NO 位置估算编号位置 ──
    if dwg_match and dwg_match.get("poly"):
        poly = dwg_match["poly"]
        poly_left = min(p[0] for p in poly)
        poly_right = max(p[0] for p in poly)
        poly_top = min(p[1] for p in poly)
        poly_bot = max(p[1] for p in poly)
        poly_h = max(int(poly_bot - poly_top), 1)
        poly_w = max(int(poly_right - poly_left), 1)
        dwg_text_len = max(len(dwg_match["text"].replace(" ", "")), 1)
        char_w = poly_w / dwg_text_len
        est_num_w = int(char_w * 9)
        gap = int(char_w * 0.5)
        est_bbox = BBox(int(poly_right + gap), int(poly_top), est_num_w, poly_h)
        logger.info(f"  DWG NO右侧估算橙框位置: {est_bbox}")
        final_bbox = _make_final_bbox(est_bbox, 9)
        return (None, _clip_by_vlines(final_bbox))

    logger.info("  左上角未找到 Y 编号")
    return None


# ══════════════════════════════════════════════════════════════════
#  旧版后备检测（关键词检测失败时使用）
# ══════════════════════════════════════════════════════════════════


def _fallback_detect_bottom_right_number(
    image: np.ndarray, region_pct: dict = None
) -> BBox | None:
    """旧版：百分比区域 + 最大单元格。"""
    if region_pct is None:
        region_pct = DEFAULT_REGIONS["bottom_right_title"]

    search = _pct_to_px(image, region_pct)
    roi = search.crop(image)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    roi_h, roi_w = roi.shape[:2]

    h_kernel_w = max(int(roi_w * 0.3), 20)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_ys = sorted(set(y + bh // 2 for c in h_contours for (_, y, _, bh) in [cv2.boundingRect(c)]))
    h_filtered = []
    for y in h_ys:
        if not h_filtered or y - h_filtered[-1] > 5:
            h_filtered.append(y)

    v_kernel_h = max(int(roi_h * 0.1), 8)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_xs = sorted(set(x + bw // 2 for c in v_contours for (x, _, bw, _) in [cv2.boundingRect(c)]))
    v_filtered = []
    for x in v_xs:
        if not v_filtered or x - v_filtered[-1] > 5:
            v_filtered.append(x)

    if len(h_filtered) < 2 or len(v_filtered) < 2:
        y_start = int(roi_h * 0.7)
        return BBox(search.x, search.y + y_start, roi_w, roi_h - y_start)

    candidates = []
    for ri in range(len(h_filtered) - 1):
        row_top = h_filtered[ri]
        row_bot = h_filtered[ri + 1]
        row_h = row_bot - row_top
        if row_h < 5:
            continue
        for ci in range(len(v_filtered) - 1):
            cell_left = v_filtered[ci]
            cell_right = v_filtered[ci + 1]
            cell_w = cell_right - cell_left
            if cell_w < 10:
                continue
            if (row_bot > roi_h * 0.4 and cell_w > roi_w * 0.15
                    and 8 < row_h < roi_h * 0.5):
                score = (row_bot / roi_h) * 0.6 + (cell_w / roi_w) * 0.4
                candidates.append((score, cell_left, row_top, cell_w, row_h))

    if candidates:
        candidates.sort(reverse=True)
        _, cx, cy, cw, ch = candidates[0]
        margin = 4
        return BBox(
            search.x + cx + margin, search.y + cy + margin,
            max(cw - 2 * margin, 1), max(ch - 2 * margin, 1),
        )

    y_start = int(roi_h * 0.7)
    return BBox(search.x, search.y + y_start, roi_w, roi_h - y_start)


def _auto_rotate_portrait(image: np.ndarray, prefixes: list[str] = None) -> tuple[np.ndarray, int]:
    """纵向图纸自动选择正确旋转方向。

    策略：分别尝试 CCW90 和 CW90，在右下角搜索区域做 OCR，
    选择能找到横排编号的方向（横排 = w > h）。

    返回 (旋转后图像, 旋转代码) 或 (原图, -1) 如果不需要旋转。
    """
    img_h, img_w = image.shape[:2]
    logger.info(f"纵向图纸 ({img_w}x{img_h})，自动选择旋转方向...")

    y_re = re.compile(make_pattern(prefixes))

    for rot_name, rot_code in [
        ("CCW90", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("CW90", cv2.ROTATE_90_CLOCKWISE),
    ]:
        rotated = cv2.rotate(image, rot_code)
        rh, rw = rotated.shape[:2]
        # 在右下角区域快速 OCR 找 Y 编号
        search = BBox(int(rw * 0.60), int(rh * 0.80),
                       rw - int(rw * 0.60), rh - int(rh * 0.80))
        results = _ocr_region(rotated, search)
        for text, _, poly in results:
            text_clean = text.upper().replace(" ", "")
            if y_re.search(text_clean) and poly:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bw = int(max(xs) - min(xs))
                bh = int(max(ys) - min(ys))
                if bw > bh:  # 横排文字 → 方向正确
                    logger.info(f"  选择 {rot_name}（右下角 Y 编号为横排 {bw}x{bh}）")
                    return rotated, rot_code

    # 两个方向都没找到横排 Y 编号 → 默认 CCW90
    logger.info("  未找到横排 Y 编号，默认 CCW90")
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.ROTATE_90_COUNTERCLOCKWISE


# ══════════════════════════════════════════════════════════════════
#  统一检测入口
# ══════════════════════════════════════════════════════════════════

def detect_all_regions(
    image: np.ndarray, region_config: dict = None, prefixes: list[str] = None
) -> dict[str, BBox | None]:
    """
    检测图纸中所有目标区域。

    策略：先用 CV 结构检测定位表格区域，再用局部 OCR 确认关键词。
    失败时回退到百分比检测。

    返回 dict: {区域名: BBox 或 None, "_metadata": {...}}
    """
    img_h, img_w = image.shape[:2]
    if region_config is None:
        region_config = DEFAULT_REGIONS

    # 纵向图纸自动旋转（用户确认：图纸宽一定 > 高）
    rot_code = None
    if img_h > img_w:
        image, rot_code = _auto_rotate_portrait(image, prefixes=prefixes)
        img_h, img_w = image.shape[:2]

    metadata = {
        "method": "keyword",
        "table_direction": None,
        "keyword_positions": {},
        "rotation": rot_code
    }

    # ── Phase A: 图纸边框检测 + 统一裁切搜索区域 ──
    logger.info("Phase A: 检测图纸边框...")
    frame = _detect_drawing_frame(image)
    logger.info(f"  图纸边框: {frame}")

    # 红框搜索区域: frame左边界 ~ frame中轴线
    red_x = frame.x
    red_w = frame.w // 2
    red_search = BBox(red_x, frame.y, red_w, int(frame.h * (5.5 / 6.0)))
    table_regions = [red_search]
    logger.info(f"  红框搜索区域: {red_search}")

    # 绿框搜索区域: 左 5/8 ~ 右边界, 上 5/6 ~ 图片下边界
    green_x = frame.x + int(frame.w * (5.0 / 8.0))
    green_y = frame.y + int(frame.h * (5.0 / 6.0))
    green_search = BBox(
        green_x, green_y,
        frame.x + frame.w - green_x,
        img_h - green_y,
    )
    logger.info(f"  绿框搜索区域: {green_search}")

    # 紫框搜索区域: 左边界~绿框左边界, 从下往上1.5/6~下边界
    purple_y = frame.y + frame.h - int(frame.h * 1.5 / 6.0)
    purple_search = BBox(
        frame.x, purple_y,
        green_x - frame.x,
        frame.y + frame.h - purple_y,
    )
    logger.info(f"  紫框搜索区域: {purple_search}")

    # 橙框搜索区域: 图片左边界(x=0) ~ 右 2/8, 图片顶部(y=0) ~ frame上边界 + frame高 2/12
    # 下边界向上缩 1/3
    # 注意：DWG NO 可能在 frame 上方，所以从图片顶部开始搜索
    orange_h_full = frame.y + int(frame.h * (2.0 / 12.0))
    orange_search = BBox(0, 0,
                         frame.x + int(frame.w * (2.0 / 8.0)),
                         orange_h_full - int(orange_h_full / 3))
    logger.info(f"  橙框搜索区域: {orange_search}")

    # ── Phase B-1: 定位 MATERIAL CODE ──
    logger.info("Phase B-1: 定位 MATERIAL CODE...")
    mat_bbox, table_search_bbox, direction = _locate_material_code_column(image, table_regions)

    if mat_bbox is None:
        logger.info("  关键词检测失败，使用红框搜索区域（左半）作为搜索范围")
        metadata["method"] = "left_half"
        table_search_bbox = red_search
    else:
        metadata["table_direction"] = direction

    # ── Phase B-2: 定位右下角编号 ──
    logger.info("Phase B-2: 定位右下角编号...")
    br_result = _locate_bottom_right_number(image, mat_bbox, search=green_search, frame=frame, prefixes=prefixes)

    if br_result:
        br_text, br_bbox = br_result
        metadata["bottom_right_text"] = br_text
    else:
        br_bbox = None
        logger.info("  新检测失败，回退到百分比检测")
        br_bbox = _fallback_detect_bottom_right_number(image, region_config.get("bottom_right_title"))

    # ── Phase B-3: 定位左上角编号 ──
    logger.info("Phase B-3: 定位左上角编号...")
    tl_result = _locate_top_left_number(image, mat_bbox, search=orange_search, prefixes=prefixes)

    if tl_result:
        tl_text, tl_bbox = tl_result
        metadata["top_left_text"] = tl_text
    else:
        tl_bbox = None

    # ── Phase B-4: 定位左下角竖排编号 ──
    logger.info("Phase B-4: 定位左下角竖排编号...")
    bl_result = _locate_bottom_left_number(image, mat_bbox, search=purple_search, prefixes=prefixes)

    if bl_result:
        bl_text, bl_bbox, bl_ocr = bl_result
        metadata["bottom_left_text"] = bl_text
        metadata["purple_ocr_results"] = bl_ocr
    else:
        bl_bbox = None

    # ── 注释区域（百分比，不变）──
    ann_pct = region_config.get("annotations", DEFAULT_REGIONS["annotations"])
    ann_bbox = _pct_to_px(image, ann_pct)

    # ── 组装结果 ──
    metadata["table_search_area"] = table_search_bbox
    metadata["purple_search"] = purple_search  # 替换时需要
    metadata["search_areas"] = {
        "red_search": red_search,
        "green_search": green_search,
        "purple_search": purple_search,
        "orange_search": orange_search,
    }

    result = {
        "material_code_column": mat_bbox,
        "bottom_right_number": br_bbox,
        "top_left_number": tl_bbox,
        "bottom_left_number": bl_bbox,
        "annotations": ann_bbox,
        "_metadata": metadata,
    }

    for name, bbox in result.items():
        if name.startswith("_"):
            continue
        if bbox:
            logger.info(f"  {name}: {bbox}")
        else:
            logger.info(f"  {name}: 未检测到")

    # 纵向图纸：不进行坐标逆映射，保持旋转后图像的坐标系
    # 这样后续替换操作可以直接在旋转后的图像上进行
    if rot_code is not None:
        logger.info("保持旋转后图像的坐标系（不映射回原始方向）")

    return result


def draw_regions_debug(image: np.ndarray, regions: dict) -> np.ndarray:
    """在图像上画出检测到的区域边界（调试用）"""
    colors = {
        "material_code_column": (255, 0, 0),
        "bottom_right_number": (0, 255, 0),
        "top_left_number": (255, 165, 0),
        "bottom_left_number": (128, 0, 128),
        "annotations": (0, 0, 255),
    }
    debug_img = image.copy()

    # 画区域框
    for name, bbox in regions.items():
        if name.startswith("_") or bbox is None:
            continue
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(debug_img, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 3)
        cv2.putText(
            debug_img, name, (bbox.x, bbox.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    # 显示检测元信息
    meta = regions.get("_metadata", {})
    method = meta.get("method", "?")
    direction = meta.get("table_direction", "?")
    label = f"Method: {method} | Dir: {direction}"
    cv2.putText(
        debug_img, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3,
    )
    cv2.putText(
        debug_img, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1,
    )

    # 绘制青色框（红框内单元格级替换画布）
    cyan_boxes = meta.get("cyan_boxes", [])
    for cb in cyan_boxes:
        cv2.rectangle(debug_img, (cb.x, cb.y), (cb.x2, cb.y2), (0, 255, 255), 2)

    return debug_img
