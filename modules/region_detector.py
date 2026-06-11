"""动态区域检测 — 关键词锚定 + CV 结构检测 + 局部 OCR 确认"""

import re
import logging
import difflib

import cv2
import numpy as np

from config import (
    DEFAULT_REGIONS, KEYWORD_ANCHORS,
    Y_PATTERN, OCR_LANG_EN, OCR_LANG_CH, FUZZY_DIGIT_MAP,
    make_pattern, DEFAULT_PREFIXES, OCR_MODE,
)

logger = logging.getLogger(__name__)

# ── region_detector OCR 引擎 ──────────────────────────────────
# 红框用 v5（结构定位），绿框/橙框用 VLM
import threading as _threading

_v5_cache: dict = {}
_v5_lock = _threading.Lock()


def _get_ocr_v5(lang: str = "en"):
    with _v5_lock:
        if lang not in _v5_cache:
            from paddleocr import PaddleOCR
            _v5_cache[lang] = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _v5_cache[lang]


def _get_ocr_vlm(lang: str = "en"):
    from modules.vlm_ocr_engine import get_vlm_engine
    return get_vlm_engine()


def _parse_ocr_results_common(result):
    items = []
    if not result:
        return items
    for res in result:
        polys, texts, scores = None, None, None
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


def _get_ocr_funcs(engine: str = "v5"):
    if engine == "vlm":
        return _get_ocr_vlm, _parse_ocr_results_common
    return _get_ocr_v5, _parse_ocr_results_common


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
    if roi_gray.size == 0 or roi_gray.shape[0] < 2 or roi_gray.shape[1] < 2:
        return []

    # 超大区域降采样后检测，坐标映射回原尺寸
    MAX_EDGE = 2100
    rh, rw = roi_gray.shape[:2]
    ds_scale = 1.0
    if max(rh, rw) > MAX_EDGE:
        ds_scale = MAX_EDGE / max(rh, rw)
        roi_gray = cv2.resize(roi_gray, (int(rw * ds_scale), int(rh * ds_scale)),
                              interpolation=cv2.INTER_AREA)
        min_line_width = max(int(min_line_width * ds_scale), 3)

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

    # 映射回原尺寸
    if ds_scale != 1.0:
        filtered = [int(y / ds_scale) for y in filtered]
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
#  裁切 + 缩放工具
# ══════════════════════════════════════════════════════════════════

MAX_SEARCH_EDGE = 2100  # 搜索区裁切后最长边上限


def _crop_and_scale(image: np.ndarray, bbox: BBox, max_edge: int = MAX_SEARCH_EDGE):
    """裁切搜索区并按需缩放。

    返回 (sub_image, scale_factor)。scale_factor=1.0 表示无需缩放。
    """
    sub = bbox.crop(image)
    h, w = sub.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        sub = cv2.resize(sub, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        logger.debug(f"搜索区缩放: {w}x{h} → {sub.shape[1]}x{sub.shape[0]} (scale={scale:.3f})")
        return sub, scale
    return sub, 1.0


def _map_bbox_back(bbox: BBox, crop_bbox: BBox, scale: float) -> BBox:
    """将子图内的 BBox 映射回原图坐标。"""
    return BBox(
        int(bbox.x / scale) + crop_bbox.x,
        int(bbox.y / scale) + crop_bbox.y,
        int(bbox.w / scale),
        int(bbox.h / scale),
    )


# ══════════════════════════════════════════════════════════════════
#  OCR 工具（局部使用，仅在确认阶段调用）
# ══════════════════════════════════════════════════════════════════

def _ocr_region(image_rgb: np.ndarray, bbox: BBox, lang: str = OCR_LANG_EN,
                engine: str = "v5") -> list:
    """对指定区域做 OCR，返回 [(text, confidence, poly), ...]。

    engine: "v5" = 本地 OCR 引擎 v5（红框定位用），"vlm" = VLM（绿框/橙框用）。
    自动放大小区域 + 加白色 padding + 锐化。
    """
    _get_ocr, _parse_ocr_results = _get_ocr_funcs(engine)

    roi = bbox.crop(image_rgb)
    roi_h, roi_w = roi.shape[:2]

    # 自适应缩放
    min_dim = min(roi_w, roi_h)
    max_dim = max(roi_w, roi_h)

    MAX_OCR_EDGE = 2100  # 大区域缩小上限

    if min_dim < 80:
        scale = 5.0
    elif min_dim < 200:
        scale = 3.0
    elif max_dim > MAX_OCR_EDGE:
        scale = MAX_OCR_EDGE / max_dim
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
    for _attempt in range(3):
        try:
            result = ocr.predict(roi)
            break
        except (AssertionError, RuntimeError) as e:
            if _attempt < 2:
                logger.warning(f"OCR predict 重试 ({_attempt+1}/3): {e}")
                continue
            logger.error(f"OCR predict 3次均失败: {e}")
            result = []
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

def _merge_adjacent_short_texts(ocr_results: list, gap_thresh: float = 100, max_char_len: int = 2) -> list:
    """把相邻的短文本（<=max_char_len字符）合并为候选词。

    同时处理水平相邻（cy接近、cx递增）和垂直相邻（cx接近、cy递增）。
    gap_thresh: 两个文本框边缘间距上限（像素）。
    返回合并后的额外候选列表，不修改原始列表。
    """
    items = []
    for text, conf, poly in ocr_results:
        if poly is None or len(text.strip()) == 0 or len(text.strip()) > max_char_len:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        items.append({
            "text": text.strip(), "conf": conf, "poly": poly,
            "cx": cx, "cy": cy,
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        })

    if len(items) < 2:
        return []

    def _build_groups(sort_key, main_axis, cross_axis, cross_thresh, main_gap_fn):
        items_sorted = sorted(range(len(items)), key=lambda i: sort_key(items[i]))
        used = [False] * len(items)
        groups = []
        for ii in range(len(items_sorted)):
            i = items_sorted[ii]
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for jj in range(ii + 1, len(items_sorted)):
                j = items_sorted[jj]
                if used[j]:
                    continue
                if abs(cross_axis(items[j]) - cross_axis(items[i])) > cross_thresh:
                    continue
                last = group[-1]
                gap = main_gap_fn(items[last], items[j])
                if gap < gap_thresh:
                    group.append(j)
                    used[j] = True
            if len(group) >= 2:
                group.sort(key=lambda idx: main_axis(items[idx]))
                groups.append(group)
        return groups

    char_h = np.median([it["y_max"] - it["y_min"] for it in items]) if items else 30
    cross_thresh = max(char_h * 0.8, 20)

    h_groups = _build_groups(
        sort_key=lambda it: (it["cy"], it["cx"]),
        main_axis=lambda it: it["cx"],
        cross_axis=lambda it: it["cy"],
        cross_thresh=cross_thresh,
        main_gap_fn=lambda a, b: b["x_min"] - a["x_max"],
    )

    v_groups = _build_groups(
        sort_key=lambda it: (it["cx"], it["cy"]),
        main_axis=lambda it: it["cy"],
        cross_axis=lambda it: it["cx"],
        cross_thresh=cross_thresh,
        main_gap_fn=lambda a, b: b["y_min"] - a["y_max"],
    )

    merged = []
    seen = set()
    for group in h_groups + v_groups:
        key = tuple(sorted(group))
        if key in seen:
            continue
        seen.add(key)
        for window in range(2, min(4, len(group) + 1)):
            for start in range(len(group) - window + 1):
                sub_idxs = group[start:start + window]
                sub_texts = [items[idx]["text"] for idx in sub_idxs]
                sub_confs = [items[idx]["conf"] for idx in sub_idxs]
                sub_pts = []
                for idx in sub_idxs:
                    sub_pts.extend(items[idx]["poly"])
                sxs = [p[0] for p in sub_pts]
                sys_ = [p[1] for p in sub_pts]
                sub_poly = [[min(sxs), min(sys_)], [max(sxs), min(sys_)],
                            [max(sxs), max(sys_)], [min(sxs), max(sys_)]]
                merged.append(("".join(sub_texts), sum(sub_confs) / len(sub_confs), sub_poly))
    return merged


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

    merged_extras = _merge_adjacent_short_texts(ocr_results)
    candidates = list(ocr_results) + merged_extras

    for text, conf, poly in candidates:
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
    超大图内部降采样后检测，坐标映射回原图。
    检测失败则回退到图片边界（2% 内缩）。
    """
    img_h, img_w = image.shape[:2]

    # 超大图降采样（仅用于边框检测）
    MAX_FRAME_EDGE = 2100
    frame_scale = 1.0
    long_edge = max(img_h, img_w)
    if long_edge > MAX_FRAME_EDGE:
        frame_scale = MAX_FRAME_EDGE / long_edge
        work_img = cv2.resize(image, (int(img_w * frame_scale), int(img_h * frame_scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        work_img = image

    wh, ww = work_img.shape[:2]
    gray = cv2.cvtColor(work_img, cv2.COLOR_RGB2GRAY)

    def _map_frame_back(f: BBox) -> BBox:
        """将降采样坐标映射回原图。"""
        if frame_scale == 1.0:
            return f
        inv = 1.0 / frame_scale
        return BBox(int(f.x * inv), int(f.y * inv), int(f.w * inv), int(f.h * inv))

    # 多阈值检测：先尝试高阈值，失败后尝试低阈值
    for threshold in [150, 130, 100]:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # 检测长水平线（>50% 图片宽度）
        h_len = max(int(ww * 0.5), 100)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_ys = sorted(set(y + bh // 2 for c in contours for (_, y, _, bh) in [cv2.boundingRect(c)]))

        # 检测长垂直线（>50% 图片高度）
        v_len = max(int(wh * 0.5), 100)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_xs = sorted(set(x + bw // 2 for c in contours for (x, _, bw, _) in [cv2.boundingRect(c)]))

        if len(h_ys) >= 2 and len(v_xs) >= 2:
            frame = BBox(v_xs[0], h_ys[0], v_xs[-1] - v_xs[0], h_ys[-1] - h_ys[0])
            logger.info(f"检测到图纸边界(阈值={threshold}): {frame}")
            return _map_frame_back(frame)

    # 第二轮：降低长度要求到40%
    logger.debug("第一轮检测失败，尝试40%长度")
    for threshold in [150, 130, 100]:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        h_len = max(int(ww * 0.4), 100)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_ys = sorted(set(y + bh // 2 for c in contours for (_, y, _, bh) in [cv2.boundingRect(c)]))

        v_len = max(int(wh * 0.4), 100)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_xs = sorted(set(x + bw // 2 for c in contours for (x, _, bw, _) in [cv2.boundingRect(c)]))

        if len(h_ys) >= 2 and len(v_xs) >= 2:
            frame = BBox(v_xs[0], h_ys[0], v_xs[-1] - v_xs[0], h_ys[-1] - h_ys[0])
            # 验证边框合理性：应该接近图像边缘且覆盖大部分图像
            if (frame.x < ww * 0.15 and frame.y < wh * 0.15 and
                frame.w > ww * 0.7 and frame.h > wh * 0.7):
                logger.info(f"检测到图纸边界(阈值={threshold},40%): {frame}")
                return _map_frame_back(frame)

    # 回退：图片边界 5% 内缩
    mx, my = int(img_w * 0.05), int(img_h * 0.05)
    frame = BBox(mx, my, img_w - 2 * mx, img_h - 2 * my)
    logger.info(f"未检测到边框线，使用图片边界: {frame}")
    return frame


# ══════════════════════════════════════════════════════════════════
#  Phase B-1: 定位 MATERIAL CODE 列
# ══════════════════════════════════════════════════════════════════

def _validate_material_code_column(
    image: np.ndarray,
    col_bbox: BBox,
    anchor_match: dict | None = None,
) -> bool:
    """验证列内是否包含MATERIAL CODE文字，确认找对了位置。

    anchor_match: 定位阶段的关键词命中结果（含 text/keyword/score/poly）。
      若是精确命中（score>=0.95）且 poly 落在扩展验证区域的水平范围内，
      则直接通过——避免对窄列做二次 OCR 时小字漏检（如 PDF 渲染的"代号"）。
    """
    # 向上扩展200px包含表头区域
    extended_y = max(0, col_bbox.y - 200)
    extended_h = col_bbox.h + (col_bbox.y - extended_y)
    extended_bbox = BBox(col_bbox.x, extended_y, col_bbox.w, extended_h)

    # 旁路：定位阶段已是精确命中且关键词位置就在扩展区域内 → 直接信任
    if anchor_match and anchor_match.get("score", 0) >= 0.95:
        poly = anchor_match.get("poly")
        if poly:
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            if (extended_bbox.x <= cx <= extended_bbox.x + extended_bbox.w
                    and extended_bbox.y <= cy <= extended_bbox.y + extended_bbox.h):
                logger.info(
                    f"  列内容验证旁路: 定位精确命中 '{anchor_match['text']}' "
                    f"(score={anchor_match['score']:.2f})"
                )
                return True

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
        if "MATERIALCODE" in text_up or "MATERIAL" in text_up or "零部件图号" in text or "代号" in text:
            logger.info(f"  列内容验证通过(精确): 找到'{text}'")
            return True

    # 第二轮：模糊匹配
    keywords = ["MATERIAL CODE", "MATERUL CODE", "零部件图号", "DEF", "代号"]
    match = _fuzzy_find_keyword(ocr_results, keywords, threshold=0.60)
    if match:
        logger.info(f"  列内容验证通过(模糊): 找到'{match['text']}' 匹配'{match['keyword']}' (score={match['score']:.2f})")
        return True

    logger.warning(f"  列内容验证失败: 未找到MATERIAL CODE相关文字")
    return False


def _fix_dwg_left_boundary(image: np.ndarray, col_bbox: BBox) -> BBox:
    """检查红框内 OCR 结果是否包含 DWG 前缀，若有则修正左边界。

    当红框列宽定位偏大、把左侧 DWG 子列也包进来时，OCR 可能识别出：
    - 合并文字 "DWGZ212C0008"：用字符比例估算编号起始 x
    - 独立文字 "DWG"：用 DWG 识别框的右边界作为新左边界
    """
    ocr_results = _ocr_region(image, col_bbox)
    logger.debug(f"  DWG修正检查: col_bbox={col_bbox}, OCR结果={len(ocr_results)}项")
    for text, conf, poly in ocr_results:
        logger.debug(f"    '{text}' conf={conf:.3f}")

    for text, conf, poly in ocr_results:
        if poly is None or len(poly) < 4:
            continue
        text_up = text.upper().replace(" ", "")

        # 检查是否以 DWG 开头（DWG、DWGNO 等）
        dwg_prefix = None
        for prefix in ("DWGNO", "DWG"):
            if text_up.startswith(prefix):
                dwg_prefix = prefix
                break
        if dwg_prefix is None:
            continue

        poly_xs = [p[0] for p in poly]
        poly_left = min(poly_xs)
        poly_right = max(poly_xs)
        poly_w = poly_right - poly_left
        if poly_w <= 0:
            continue

        total_chars = len(text_up)
        prefix_chars = len(dwg_prefix)

        if total_chars <= prefix_chars:
            # DWG 被单独识别（如 "DWG"、"DWG NO"）→ 用其右边界作为新左边界
            new_x = int(poly_right)
        else:
            # DWG 和编号合并（如 "DWGZ212C0008"）→ 用字符比例估算
            new_x = int(poly_left + poly_w * (prefix_chars / total_chars))

        new_w = col_bbox.x + col_bbox.w - new_x

        if new_x > col_bbox.x and new_w > 0:
            logger.info(
                f"  DWG左边界修正: 识别到'{text}', DWG前缀='{dwg_prefix}'"
                f" → 左边界从 x={col_bbox.x} 移到 x={new_x} (缩窄{new_x - col_bbox.x}px)"
            )
            return BBox(new_x, col_bbox.y, new_w, col_bbox.h)
        break

    return col_bbox


def _locate_material_code_column(
    sub_image: np.ndarray,
) -> tuple[BBox | None, str | None]:
    """在子图中找到 MATERIAL CODE 列。

    子图为红框搜索区的裁切（可能已缩放）。
    策略：
    1. 对表头行做局部 OCR
    2. 模糊匹配 "MATERIAL CODE"
    3. 用网格线定位具体列，追踪到表格结束

    返回 (column_bbox, direction) 或 (None, None)，坐标为子图内坐标。
    """
    kw_config = KEYWORD_ANCHORS.get("material_code", {})
    keywords = kw_config.get("keywords", ["MATERIAL CODE"])
    fuzzy_thresh = kw_config.get("fuzzy_threshold", 0.70)

    img_h, img_w = sub_image.shape[:2]
    table_bbox = BBox(0, 0, img_w, img_h)

    # 对表头区域做 OCR（取顶部 ~20% 或至少 100px）
    header_h = max(int(img_h * 0.2), min(100, img_h))
    header_bbox = BBox(0, 0, img_w, header_h)

    ocr_results = _ocr_region(sub_image, header_bbox)
    match = _fuzzy_find_keyword(ocr_results, keywords, fuzzy_thresh)
    if match is None:
        # 英文OCR未找到，尝试中文OCR（识别"零部件图号"等中文关键词）
        ocr_results_ch = _ocr_region(sub_image, header_bbox, lang=OCR_LANG_CH)
        match = _fuzzy_find_keyword(ocr_results_ch, keywords, fuzzy_thresh)
        if match is not None:
            ocr_results = ocr_results_ch  # 后续邻居搜索也用中文结果
        else:
            return None, None

    logger.info(
        f"找到 MATERIAL CODE 关键词: '{match['text']}' "
        f"(匹配 '{match['keyword']}', score={match['score']:.2f})"
    )

    # 在同一批 OCR 结果中搜索相邻列关键词（DEF / 品群）
    # MATERIAL CODE 关键词的 x 中心（子图坐标）
    kw_poly = match.get("poly")
    mat_cx = (
        sum(p[0] for p in kw_poly) / len(kw_poly)
        if kw_poly else img_w / 2
    )

    neighbors = {}
    def_candidates = []
    shin_candidates = []
    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        text_up = text.upper().strip()
        cx = sum(p[0] for p in poly) / len(poly)
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

    # 在子图中定位具体列
    table_roi_gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)

    # 所有图纸均为竖向布局，直接调用竖向追踪
    direction = "vertical"
    col_bbox = _trace_vertical_table(
        table_bbox, match, table_roi_gray, neighbors
    )

    if col_bbox is not None:
        if not _validate_material_code_column(sub_image, col_bbox, anchor_match=match):
            logger.warning("列内容验证失败")
            return None, None
        col_bbox = _fix_dwg_left_boundary(sub_image, col_bbox)
        return col_bbox, direction

    return None, None


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
    """对去空格后的OCR文本尝试符号→数字模糊回填，返回修复后的编号或None。

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
    if len(fixed) >= 5 and y_re.match(fixed):
        return fixed
    if len(fixed) >= 5 and fixed[0] == 'T':
        fixed_t = 'Y' + fixed[1:]
        if y_re.match(fixed_t):
            return fixed_t
    return None


def _search_y_number(
    image: np.ndarray,
    search: BBox,
    y_re,
    material_code_bbox: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在指定搜索区域内找 Y 编号，返回 (文本, BBox)。"""
    ocr_results = _ocr_region(image, search, engine="vlm")

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
                best_match = (text_bbox, area, matched_group, text_clean, m)

    if not best_match:
        return None

    text_bbox, _, matched_text, full_text, match_obj = best_match

    # ── 根据匹配位置缩紧 bbox（排除页码等多余内容）──
    full_len = len(full_text)
    match_len = len(matched_text)
    if match_obj and full_len > match_len and text_bbox.w > 0:
        char_w = text_bbox.w / full_len
        match_start = match_obj.start()
        tight_x = text_bbox.x + int(match_start * char_w)
        tight_w = int(match_len * char_w)
        # 小幅扩展（约 0.3 个字符宽度），避免裁切过紧
        pad = max(int(char_w * 0.3), 3)
        tight_bbox = BBox(
            max(0, tight_x - pad), text_bbox.y,
            min(tight_w + 2 * pad, text_bbox.x + text_bbox.w - tight_x + pad),
            text_bbox.h,
        )
        logger.info(f"右下角编号: '{matched_text}' at {text_bbox} "
                    f"→ tight {tight_bbox} (full='{full_text}', "
                    f"match@{match_start}:{match_start+match_len})")
        return (matched_text, tight_bbox)

    logger.info(f"右下角编号: '{matched_text}' at {text_bbox}")
    return (matched_text, text_bbox)


# ── 绿框辅助函数 ────────────────────────────────────────────


def _validate_green_result(text: str) -> bool:
    """验证绿框 OCR 结果：≥9 位字符，末尾非符号。"""
    t = text.strip()
    if len(t) < 9:
        return False
    if not t[-1].isalnum():
        return False
    return True


def _fix_digit_letter_confusion(text: str) -> str:
    """修复 OCR 在数字位置误读为形似字母的情况。

    Y 编号格式: Y + A + 3位数字 + 1位字母 + 3-4位数字 [+ P + 4位数字]
    例: YA128A360, YA070A191P7933
    在已知为数字的位置，将 I→1, O→0, S→5, B→8 等替换。
    """
    if len(text) < 5 or not text[0:1].isalpha():
        return text
    # 字母→数字映射（仅在数字位置使用）
    letter_to_digit = {'I': '1', 'O': '0', 'S': '5', 'B': '8', 'Z': '2',
                       'G': '6', 'T': '7', 'Q': '0', 'D': '0'}
    result = list(text)
    # 位置 0: 前缀字母 (Y) — 不改
    # 位置 1: 字母 (A) — 不改
    # 位置 2-4: 3 位数字 — 可修复
    # 位置 5: 字母 (A/C/D) — 不改
    # 位置 6-8+: 数字 — 可修复
    # 如果有 P 后缀: P 后面全是数字
    # 通用策略：在偏移 2-4 和 6+ 的非字母组中修复
    if len(result) >= 6:
        # 修复位置 2,3,4（应为数字）
        for i in range(2, min(5, len(result))):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
        # 修复位置 6+（跳过字母位置 5）
        for i in range(6, len(result)):
            ch = result[i]
            if ch == 'P' and i < len(result) - 1:
                # P 后缀开始，后面全是数字
                for j in range(i + 1, len(result)):
                    if result[j] in letter_to_digit:
                        result[j] = letter_to_digit[result[j]]
                break
            if ch in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
    fixed = ''.join(result)
    if fixed != text:
        logger.info(f"  数字/字母混淆修复: '{text}' → '{fixed}'")
    return fixed


def _find_right_vline(
    gray: np.ndarray, text_bbox: BBox, img_w: int,
) -> int | None:
    """在 OCR bbox 右侧附近搜索竖线（用于裁剪跨单元格 bbox）。

    返回竖线 x 坐标，或 None。
    """
    char_w = max(text_bbox.w // 9, 10)  # 按 9 字符估算单字符宽度
    # 搜索范围：bbox 右边界左侧 1 字符 到 右边界右侧 0.5 字符
    search_x1 = max(0, text_bbox.x2 - char_w)
    search_x2 = min(img_w, text_bbox.x2 + char_w // 2)
    if search_x2 <= search_x1:
        return None

    # 裁切搜索区域（只取文本所在的 y 范围）
    y1 = max(0, text_bbox.y)
    y2 = min(gray.shape[0], text_bbox.y2)
    if y2 <= y1:
        return None

    roi = gray[y1:y2, search_x1:search_x2]
    binary = _binarize_for_lines(roi)

    # 列投影：统计每列的黑像素数
    col_proj = np.sum(binary > 0, axis=0)
    threshold = (y2 - y1) * 0.5  # 超过行高 50% 视为竖线

    # 找投影峰值
    best_x = None
    best_val = threshold
    for i, val in enumerate(col_proj):
        if val > best_val:
            best_val = val
            best_x = search_x1 + i

    if best_x is not None:
        logger.info(f"  绿框vline: 在 x={best_x} 找到竖线 (投影={best_val:.0f})")
    return best_x


def _trim_text_by_vline(
    text: str, text_bbox: BBox, vline_x: int,
) -> tuple[str, BBox]:
    """按竖线位置裁剪文本和 bbox。"""
    if len(text) == 0:
        return text, text_bbox
    char_w = text_bbox.w / len(text)
    chars_to_keep = int((vline_x - text_bbox.x) / char_w)  # floor，竖线左侧的完整字符数
    chars_to_keep = max(1, min(chars_to_keep, len(text)))
    trimmed = text[:chars_to_keep]
    new_w = max(vline_x - text_bbox.x - 2, 1)
    new_bbox = BBox(text_bbox.x, text_bbox.y, new_w, text_bbox.h)
    logger.info(f"  绿框trim: '{text}' → '{trimmed}', bbox w: {text_bbox.w} → {new_w}")
    return trimmed, new_bbox


def _detect_long_lines(gray: np.ndarray, img_h: int, img_w: int):
    """从 morph 图中检测所有竖线，按长度分为"长线"和"短线"两组；检测横线。

    长短分类方法：将所有竖线按长度排序，找最大落差点分组。
    落差以上的为"长线"，落差以下的为"短线"。

    返回 (long_vlines_by_x, hlines_by_y):
      long_vlines_by_x: [(x, height), ...] 仅长线，按 x 从右到左排序
      hlines_by_y: [(y, width), ...]  按 y 从上到下排序
    """
    thresh = _binarize_for_lines(gray)

    # 竖线检测（形态学）
    min_vh = max(img_h // 8, 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vline_raw = []
    for c in v_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        vline_raw.append([x + bw // 2, bh])
    vline_raw.sort(key=lambda d: d[0])
    # 合并相邻 (x 间距 < 10)
    merged_v = []
    for xc, h in vline_raw:
        if not merged_v or xc - merged_v[-1][0] > 10:
            merged_v.append([xc, h])
        else:
            merged_v[-1][1] = max(merged_v[-1][1], h)

    # 按长度从大到小排序，区分长线/短线
    # 策略：将所有高度从小到大排列，从底部向上找第一个超过中位数的空隙
    # 这样能正确把密集的短线群和上方的长线群分开
    sorted_by_h = sorted(merged_v, key=lambda v: -v[1])
    long_vlines = []
    if len(sorted_by_h) >= 2:
        all_heights = sorted(set(v[1] for v in sorted_by_h))
        if len(all_heights) >= 2:
            # 计算所有相邻空隙
            gaps = [(all_heights[i + 1] - all_heights[i], all_heights[i])
                    for i in range(len(all_heights) - 1)]
            median_gap = sorted(g[0] for g in gaps)[len(gaps) // 2]
            # 从底部向上，找第一个显著超过中位数的空隙（> 中位数*3 且 > 30）
            cut_value = 0
            for gap_val, lower_h in gaps:
                if gap_val > max(median_gap * 3, 30):
                    cut_value = lower_h
                    logger.info(f"  竖线分组: 空隙={gap_val} at h={lower_h}~"
                                f"{lower_h + gap_val}, 中位数空隙={median_gap}")
                    break
            if cut_value > 0:
                long_vlines = [(xc, h) for xc, h in sorted_by_h if h > cut_value]
                short_count = len(sorted_by_h) - len(long_vlines)
                logger.info(f"  竖线分组: 长线{len(long_vlines)}根(h>{cut_value}), "
                            f"短线{short_count}根(h<={cut_value})")
            else:
                long_vlines = [(xc, h) for xc, h in sorted_by_h]
                logger.info(f"  竖线分组: 无显著空隙，全部{len(long_vlines)}根")
        else:
            long_vlines = [(xc, h) for xc, h in sorted_by_h]
    elif len(sorted_by_h) == 1:
        long_vlines = [(sorted_by_h[0][0], sorted_by_h[0][1])]

    # 按 x 从右到左排序
    long_vlines.sort(key=lambda v: -v[0])

    # 横线检测（形态学，带宽度信息）
    min_hw = max(img_w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hline_raw = []
    for c in h_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        hline_raw.append([y + bh // 2, bw])
    hline_raw.sort(key=lambda d: d[0])
    # 合并相邻 (y 间距 < 10)
    merged_h = []
    for yc, w in hline_raw:
        if not merged_h or yc - merged_h[-1][0] > 10:
            merged_h.append([yc, w])
        else:
            merged_h[-1][1] = max(merged_h[-1][1], w)

    # 横线也做长短分类（同样的 gap 策略）
    long_hlines = []
    if len(merged_h) >= 2:
        all_widths = sorted(set(h[1] for h in merged_h))
        if len(all_widths) >= 2:
            gaps_h = [(all_widths[i + 1] - all_widths[i], all_widths[i])
                      for i in range(len(all_widths) - 1)]
            median_gap_h = sorted(g[0] for g in gaps_h)[len(gaps_h) // 2]
            cut_value_h = 0
            for gap_val, lower_w in gaps_h:
                if gap_val > max(median_gap_h * 3, 30):
                    cut_value_h = lower_w
                    logger.info(f"  横线分组: 空隙={gap_val} at w={lower_w}~"
                                f"{lower_w + gap_val}, 中位数空隙={median_gap_h}")
                    break
            if cut_value_h > 0:
                long_hlines = [(yc, w) for yc, w in merged_h if w > cut_value_h]
                short_h_count = len(merged_h) - len(long_hlines)
                logger.info(f"  横线分组: 长线{len(long_hlines)}根(w>{cut_value_h}), "
                            f"短线{short_h_count}根(w<={cut_value_h})")
            else:
                # 无显著空隙 → 不做横向裁切（用户指示：横线识别不到就不处理）
                logger.info(f"  横线分组: 无显著空隙({len(merged_h)}根)，跳过横线裁切")
                long_hlines = []
        else:
            long_hlines = [(yc, w) for yc, w in merged_h]
    elif len(merged_h) == 1:
        long_hlines = [(merged_h[0][0], merged_h[0][1])]

    # 按 y 从上到下排序
    long_hlines.sort(key=lambda h: h[0])

    return long_vlines, long_hlines


def _crop_by_long_lines(img_h, img_w, long_vlines_rtl, hlines_by_y,
                        v_rank, h_rank):
    """按长线从右往左排名裁切：返回裁切后的 BBox。

    v_rank: 从右往左第 v_rank 根长竖线，其左侧全部裁掉。
    h_rank: 从下往上第 h_rank 根横线，其上方全部裁掉。
            横线不足则不做纵向裁切(crop_y=0)。
    """
    if v_rank <= len(long_vlines_rtl):
        crop_x = long_vlines_rtl[v_rank - 1][0]
    else:
        crop_x = 0

    # 从下往上第 h_rank 根横线
    if h_rank <= len(hlines_by_y):
        crop_y = hlines_by_y[-h_rank][0]
    else:
        crop_y = 0  # 横线不足，不做纵向裁切

    crop_w = img_w - crop_x
    crop_h = img_h - crop_y
    if crop_w < 20 or crop_h < 20:
        return None
    return BBox(crop_x, crop_y, crop_w, crop_h)


def _preprocess_green_ocr(sub_image: np.ndarray) -> np.ndarray:
    """dilation + closing 增强绿框子图的文字笔画。"""
    gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    enhanced = cv2.bitwise_not(closed)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _enhance_gray_for_lines(gray: np.ndarray) -> np.ndarray:
    """增强灰度图以使淡细分隔线更易检测。

    适用于所有线检测场景（竖线/横线 morph、列投影、右侧竖线搜索等）。
    不适用于 OCR 文字识别预处理。

    Pipeline: CLAHE局部对比度 → 锐化 → 形态学黑帽提取细暗线 → 混合加深
    """
    # 1. CLAHE 局部对比度增强
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. 锐化（unsharp mask）— 增强线条边缘
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 2.0, blur, -1.0, 0)

    # 3. 形态学黑帽 — 提取浅底上的细暗结构
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(sharpened, cv2.MORPH_BLACKHAT, bh_kernel)

    # 4. 混合 — 黑帽 3x 权重加深线条区域
    result = np.clip(
        sharpened.astype(np.int16) - blackhat.astype(np.int16) * 3,
        0, 255
    ).astype(np.uint8)
    return result


def _binarize_for_lines(gray: np.ndarray) -> np.ndarray:
    """灰度图 → 增强 → 去噪 → 反色二值化。线检测的统一入口。

    使用 Otsu + 自适应阈值的并集：
      - Otsu 捕获高对比度的粗线（全局阈值）
      - 自适应阈值捕获局部低对比度的淡线（局部阈值）
      - 后续 morph open 会过滤掉并集引入的小噪点
    """
    enhanced = _enhance_gray_for_lines(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    # Otsu — 高对比度特征
    _, thresh_otsu = cv2.threshold(denoised, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 自适应阈值 — 局部低对比度淡线（blockSize=25 覆盖线宽邻域，C=10 容忍淡线）
    thresh_adapt = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    # 并集：任一方法检出即保留
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)
    return thresh


def _find_cell_boundary(gray: np.ndarray, text_bbox: BBox,
                        img_h: int, img_w: int,
                        y_text: str = "",
                        _return_cell: bool = False) -> BBox:
    """用 morph 横竖线定位单元格，决定绿框边界。

    铁律：绿框的每条边只能是 OCR 识别框的边 或 morph 线位置，无其他可能。

    步骤：
    1. 裁切 text_bbox 周围局部区域，在局部做 morph 线检测
    2. 筛选经过文字中心的线，找最近的四条 → cell
    3. OCR 面积 > cell 面积 → 绿框 = cell（四边都是 morph 线）
    4. OCR 面积 ≤ cell 面积 → 绿框 = OCR，但每条边若超出 cell 则裁到 morph 线
    """
    # ── 裁切局部区域：text_bbox 周围扩展 2 倍尺寸，等比放大到标准高度 ──
    pad_w = text_bbox.w * 2
    pad_h = text_bbox.h * 2
    crop_x1 = max(0, text_bbox.x - pad_w)
    crop_y1 = max(0, text_bbox.y - pad_h)
    crop_x2 = min(img_w, text_bbox.x2 + pad_w)
    crop_y2 = min(img_h, text_bbox.y2 + pad_h)
    local_gray = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    local_h, local_w = local_gray.shape[:2]

    # 等比放大：将局部区域放大到标准高度 TARGET_H
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

    # ── 形态学竖线检测（保留 x, y_min, y_max）──
    min_vh = max(scaled_h // 8, 12)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    # 每条竖线: (x_center, y_min, y_max) — 缩放回局部坐标再转原图坐标
    vlines_raw = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        vlines_raw.append((crop_x1 + int((cx + cw // 2) / scale),
                           crop_y1 + int(cy / scale),
                           crop_y1 + int((cy + ch) / scale)))
    vlines_raw.sort()

    # 合并相近的竖线（x差≤10）
    vlines = []  # [(x, y_min, y_max), ...]
    for x, y1, y2 in vlines_raw:
        if vlines and x - vlines[-1][0] <= 10:
            ox, oy1, oy2 = vlines[-1]
            vlines[-1] = (ox, min(oy1, y1), max(oy2, y2))
        else:
            vlines.append((x, y1, y2))

    # ── 形态学横线检测（保留 y, x_min, x_max）──
    min_hw = max(scaled_w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    # 每条横线: (y_center, x_min, x_max) — 缩放回局部坐标再转原图坐标
    hlines_raw = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        hlines_raw.append((crop_y1 + int((cy + ch // 2) / scale),
                           crop_x1 + int(cx / scale),
                           crop_x1 + int((cx + cw) / scale)))
    hlines_raw.sort()

    # 合并相近的横线（y差≤10）
    hlines = []  # [(y, x_min, x_max), ...]
    for y, x1, x2 in hlines_raw:
        if hlines and y - hlines[-1][0] <= 10:
            oy, ox1, ox2 = hlines[-1]
            hlines[-1] = (oy, min(ox1, x1), max(ox2, x2))
        else:
            hlines.append((y, x1, x2))

    logger.info(f"  morph线: {len(vlines)}条竖线, {len(hlines)}条横线")

    text_cx = (text_bbox.x + text_bbox.x2) // 2
    text_cy = (text_bbox.y + text_bbox.y2) // 2

    # ── 筛选经过文字中心的线 ──
    # 竖线：y范围必须覆盖 text_cy（线经过文字所在的水平带）
    valid_vx = [x for x, y1, y2 in vlines if y1 <= text_cy <= y2]
    # 横线：x范围必须覆盖 text_cx（线经过文字所在的垂直带）
    valid_hy = [y for y, x1, x2 in hlines if x1 <= text_cx <= x2]

    logger.info(f"  经过文字中心的线: vx={valid_vx}, hy={valid_hy}")

    # 左边界：text_cx 左侧最近的竖线
    cell_left = 0
    for x in valid_vx:
        if x < text_cx:
            cell_left = x
        else:
            break

    # 右边界：text_cx 右侧最近的竖线
    cell_right = img_w
    for x in valid_vx:
        if x > text_cx:
            cell_right = x
            break

    # 上边界：text_cy 上方最近的横线
    cell_top = 0
    for y in valid_hy:
        if y < text_cy:
            cell_top = y
        else:
            break

    # 下边界：text_cy 下方最近的横线
    cell_bottom = img_h
    for y in valid_hy:
        if y > text_cy:
            cell_bottom = y
            break

    cell = BBox(cell_left, cell_top,
                cell_right - cell_left, cell_bottom - cell_top)

    logger.info(f"  v1 Cell={cell}, OCR={text_bbox}")

    if _return_cell:
        return cell, cell
    return cell


def _detect_morph_lines(gray: np.ndarray, v_ratio: float = 6.5,
                        enhance_lines: bool = False):
    """在灰度图上做 morph 线检测（裁切图坐标）。

    返回 (vlines, hlines)：
    - vlines: [(x, y_min, y_max), ...]
    - hlines: [(y, x_min, x_max), ...]
    v_ratio: 竖线最小长度 = h / v_ratio，值越大阈值越低。
    enhance_lines: True 时先用 morph，再用投影法补充检测淡线。
    """
    h, w = gray.shape[:2]
    thresh = _binarize_for_lines(gray)

    min_vh = max(int(h / v_ratio), 12)
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

    if enhance_lines:
        existing_vx = {x for x, _, _ in vlines}
        col_proj = np.mean(gray < 200, axis=0)
        mean_density = np.mean(col_proj)
        threshold = max(mean_density * 3, 0.15)
        for x in range(1, w - 1):
            if col_proj[x] > threshold:
                if not any(abs(x - ex) <= 10 for ex in existing_vx):
                    vlines.append((x, 0, h))
                    existing_vx.add(x)
                    logger.info(f"    投影法补充竖线: x={x} (density={col_proj[x]:.3f})")
        vlines.sort()

        existing_hy = {y for y, _, _ in hlines}
        row_proj = np.mean(gray < 200, axis=1)
        mean_h_density = np.mean(row_proj)
        h_threshold = max(mean_h_density * 3, 0.15)
        for y in range(1, h - 1):
            if row_proj[y] > h_threshold:
                if not any(abs(y - ey) <= 10 for ey in existing_hy):
                    hlines.append((y, 0, w))
                    existing_hy.add(y)
                    logger.info(f"    投影法补充横线: y={y} (density={row_proj[y]:.3f})")
        hlines.sort()

    return vlines, hlines


def _find_cell_from_lines(text_bbox: BBox, vlines: list, hlines: list,
                          img_w: int) -> BBox:
    """用 morph 线生成单元格边界。

    严格规则：
    - 每条边界只能来自 morph 线或 OCR bbox 边界
    - 上边界：文字中心上方、最近的横线；无则用 OCR 上边界
    - 下边界：文字中心下方、最近的横线；无则用 OCR 下边界
    - 左边界：文字中心左侧、最近的竖线（在 OCR 左边界 ±10% 内）；无则用 OCR 左边界
    - 右边界：文字中心右侧、最近的竖线；无则用 OCR 右边界
    - 必须保证 top < bottom, left < right
    """
    text_cx = (text_bbox.x + text_bbox.x2) // 2
    text_cy = (text_bbox.y + text_bbox.y2) // 2

    # ── 上边界：text_cy 上方最近的横线 ──
    above_hlines = [(y, x1, x2) for y, x1, x2 in hlines if y <= text_cy]
    if above_hlines:
        cell_top = max(above_hlines, key=lambda h: h[0])[0]  # 最近的（y最大）
    else:
        cell_top = text_bbox.y

    # ── 下边界：text_cy 下方最近的横线 ──
    below_hlines = [(y, x1, x2) for y, x1, x2 in hlines if y >= text_cy]
    if below_hlines:
        cell_bottom = min(below_hlines, key=lambda h: h[0])[0]  # 最近的（y最小）
    else:
        cell_bottom = text_bbox.y2

    # ── 左边界：text_cx 左侧、OCR 左边界 ±10% 内最近的竖线 ──
    margin = text_bbox.w * 0.1
    left_candidates = [(x, y1, y2) for x, y1, y2 in vlines
                       if x <= text_cx and (text_bbox.x - margin) <= x <= (text_bbox.x + margin)]
    if left_candidates:
        cell_left = min(left_candidates, key=lambda v: abs(v[0] - text_bbox.x))[0]
    else:
        cell_left = text_bbox.x

    # ── 右边界：text_cx 右侧最近的竖线 ──
    right_candidates = [(x, y1, y2) for x, y1, y2 in vlines if x >= text_cx]
    if right_candidates:
        cell_right = min(right_candidates, key=lambda v: v[0])[0]  # 最近的（x最小）
    else:
        cell_right = text_bbox.x2

    # ── 安全检查：保证 cell 包含 OCR bbox ──
    if cell_top > text_bbox.y:
        cell_top = text_bbox.y
    if cell_bottom < text_bbox.y2:
        cell_bottom = text_bbox.y2
    if cell_left > text_bbox.x:
        cell_left = text_bbox.x
    if cell_right < text_bbox.x2:
        cell_right = text_bbox.x2

    cell_w = cell_right - cell_left
    cell_h = cell_bottom - cell_top

    logger.info(f"  cell_from_lines: top={cell_top}, bottom={cell_bottom}, "
                f"left={cell_left}, right={cell_right}, w={cell_w}, h={cell_h}")

    return BBox(cell_left, cell_top, cell_w, cell_h)


def _make_orange_bbox(ocr_bbox: BBox, cell: BBox) -> BBox:
    """根据 OCR bbox 和 cell bbox 生成橙框，保留 OCR 右边界不裁剪。"""
    ocr_area = ocr_bbox.w * ocr_bbox.h
    cell_area = cell.w * cell.h

    logger.info(f"  橙框: Cell={cell} (area={cell_area}), OCR={ocr_bbox} (area={ocr_area})")

    if ocr_area > cell_area:
        result = BBox(cell.x, cell.y, cell.w, cell.h)
        if ocr_bbox.x2 < cell.x2:
            result = BBox(result.x, result.y,
                          ocr_bbox.x2 - result.x, result.h)
        logger.info(f"  橙框=Cell(保留OCR右边界): {result}")
    else:
        rx1 = ocr_bbox.x
        ry1 = ocr_bbox.y
        rx2 = ocr_bbox.x2
        ry2 = ocr_bbox.y2

        if rx1 < cell.x:
            rx1 = cell.x
        if ry1 < cell.y:
            ry1 = cell.y
        if ry2 > cell.y2:
            ry2 = cell.y2

        result = BBox(rx1, ry1, rx2 - rx1, ry2 - ry1)
        logger.info(f"  橙框=OCR(右边界不裁剪): {result}")

    return result


def _make_green_bbox(ocr_bbox: BBox, cell: BBox) -> BBox:
    """根据 OCR bbox 和 cell bbox 生成绿框。

    规则：
    - OCR 面积 > cell 面积 → 绿框 = cell；若 OCR 右边界在 cell 内则绿框右边界用 OCR
    - OCR 面积 ≤ cell 面积 → 绿框 = OCR，溢出边裁到 cell
    """
    ocr_area = ocr_bbox.w * ocr_bbox.h
    cell_area = cell.w * cell.h

    logger.info(f"  Cell={cell} (area={cell_area}), OCR={ocr_bbox} (area={ocr_area})")

    if ocr_area > cell_area:
        result = BBox(cell.x, cell.y, cell.w, cell.h)
        if ocr_bbox.x2 < cell.x2:
            result = BBox(result.x, result.y,
                          ocr_bbox.x2 - result.x, result.h)
            logger.info(f"  绿框=Cell(OCR右在内): {result}")
        else:
            logger.info(f"  绿框=Cell: OCR面积{ocr_area} > Cell面积{cell_area}")
    else:
        rx1 = ocr_bbox.x
        ry1 = ocr_bbox.y
        rx2 = ocr_bbox.x2
        ry2 = ocr_bbox.y2

        if rx1 < cell.x:
            rx1 = cell.x
        if ry1 < cell.y:
            ry1 = cell.y
        if rx2 > cell.x2:
            rx2 = cell.x2
        if ry2 > cell.y2:
            ry2 = cell.y2

        result = BBox(rx1, ry1, rx2 - rx1, ry2 - ry1)
        if result.w != ocr_bbox.w or result.h != ocr_bbox.h:
            logger.info(f"  绿框=OCR(溢出裁剪): OCR={ocr_bbox} → {result}")
        else:
            logger.info(f"  绿框=OCR: {ocr_bbox}")

    return result


def _locate_bottom_right_number(
    sub_image: np.ndarray,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在子图（绿框搜索区裁切）中找到编号栏。

    流程：
      1. 第一轮 OCR → 大致定位
      2. v1 cell → 裁切基准
      3. 二次裁切（上/左 25% padding，下/右保留到图边）
      4. 裁切图上重新 OCR + morph 线检测 + cell + 绿框生成
      5. 绿框坐标映射回子图坐标系

    返回 (文本, BBox) 或 None，坐标为子图内坐标。
    """
    img_h, img_w = sub_image.shape[:2]
    gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)

    # ── 第一轮 OCR：大致定位 ──
    found = _locate_bottom_right_number_core(sub_image, gray, img_h, img_w, prefixes)
    if found is None:
        return None

    y_text_orig, ocr_bbox_orig = found

    # ── 用 v1 获取 cell 作为裁切基准 ──
    _, cell_v1 = _find_cell_boundary(
        gray, ocr_bbox_orig, img_h, img_w, y_text_orig, _return_cell=True
    )

    # ── 二次裁切：上/左 25% padding，下/右保留到图边 ──
    pad_x = int(cell_v1.w * 0.25)
    pad_y = int(cell_v1.h * 0.25)
    cx1 = max(0, cell_v1.x - pad_x)
    cy1 = max(0, cell_v1.y - pad_y)
    cx2 = img_w
    cy2 = img_h

    cropped_rgb = sub_image[cy1:cy2, cx1:cx2].copy()
    cropped_gray = gray[cy1:cy2, cx1:cx2]
    ch, cw = cropped_gray.shape[:2]

    logger.info(f"  二次裁切: offset=({cx1},{cy1}), size={cw}x{ch}")

    # ── 裁切图上重新 OCR ──
    new_core = _locate_bottom_right_number_core(
        cropped_rgb, cropped_gray, ch, cw, prefixes
    )
    if new_core is None:
        # 二次 OCR 失败，回退到第一轮结果 + v1 绿框
        logger.info("  二次OCR失败, 回退到第一轮结果")
        clipped_bbox = _find_cell_boundary(
            gray, ocr_bbox_orig, img_h, img_w, y_text_orig
        )
        return (y_text_orig, clipped_bbox)

    new_text, new_ocr = new_core

    # ── 裁切图上 morph 线检测 ──
    vlines, hlines = _detect_morph_lines(cropped_gray)
    logger.info(f"  裁切图morph线: {len(vlines)}条竖线, {len(hlines)}条横线")

    # ── 用 morph 线生成 cell 边界 ──
    cell = _find_cell_from_lines(new_ocr, vlines, hlines, cw)

    # ── 生成绿框 ──
    green = _make_green_bbox(new_ocr, cell)

    # ── 坐标映射回子图坐标系 ──
    result_bbox = BBox(green.x + cx1, green.y + cy1, green.w, green.h)
    logger.info(f"  最终绿框(子图坐标): {result_bbox}, text='{new_text}'")

    return (new_text, result_bbox)


def _locate_bottom_right_number_core(
    sub_image: np.ndarray,
    gray: np.ndarray,
    img_h: int, img_w: int,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """核心编号检测逻辑（不含 bbox 裁切）。"""
    y_re = re.compile(make_pattern(prefixes))
    search = BBox(0, 0, img_w, img_h)

    # ── Layer 1: 全区域 OCR ──
    result = _search_y_number(sub_image, search, y_re, prefixes=prefixes)
    if result:
        y_text, text_bbox = result

        # ── 1a: 右侧竖线裁剪（去除跨单元格的页码：A216H, C928+2/2）──
        vline_x = _find_right_vline(gray, text_bbox, img_w)
        if vline_x is not None and vline_x < text_bbox.x2 - 5:
            trimmed_text, trimmed_bbox = _trim_text_by_vline(
                y_text, text_bbox, vline_x
            )
            if _validate_green_result(trimmed_text):
                logger.info(f"找到右下角编号 (L1a竖线裁剪): "
                            f"'{y_text}' → '{trimmed_text}'")
                return (trimmed_text, trimmed_bbox)

        # ── 1b: 直接验证通过 → 返回 OCR bbox ──
        if _validate_green_result(y_text):
            logger.info(f"找到右下角编号 (L1全区域): '{y_text}' at {text_bbox}")
            return (y_text, text_bbox)

        # ── 1c: 截断修复（如 A360: 8 字符 → 扩展右侧重新 OCR）──
        stripped = y_text.strip()
        if len(stripped) >= 7:
            char_w = text_bbox.w / max(len(y_text), 1)
            # 水平扩展：向右 2.5 字符宽度
            ext_r = min(int(char_w * 2.5), img_w - text_bbox.x2)
            # 垂直扩展：上下各扩展 50%
            ext_v = int(text_bbox.h * 0.5)
            ext_bbox = BBox(
                text_bbox.x,
                max(0, text_bbox.y - ext_v),
                text_bbox.w + ext_r,
                text_bbox.h + ext_v * 2,
            )
            # 确保不超出图像
            if ext_bbox.x2 > img_w:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                img_w - ext_bbox.x, ext_bbox.h)
            if ext_bbox.y2 > img_h:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                ext_bbox.w, img_h - ext_bbox.y)
            if ext_r > 0:
                logger.info(f"  绿框L1c: '{y_text}' 截断，扩展 +{ext_r}px右 "
                            f"+{ext_v}px上下 重新 OCR")
                re_result = _search_y_number(
                    sub_image, ext_bbox, y_re, prefixes=prefixes
                )
                if re_result:
                    re_text, re_bbox = re_result
                    # OCR 可能将数字误读为形似字母（1→I, 0→O, 5→S, 8→B）
                    re_text = _fix_digit_letter_confusion(re_text)
                    if _validate_green_result(re_text):
                        logger.info(f"找到右下角编号 (L1c扩展修复): "
                                    f"'{y_text}' → '{re_text}'")
                        return (re_text, re_bbox)

        logger.info(f"  绿框L1: '{y_text}' 未通过验证 (len={len(stripped)})")

    # ── Layer 2/3: 按长线逐步裁切 ──
    # 检测长竖线和长横线
    long_vlines, long_hlines = _detect_long_lines(gray, img_h, img_w)
    logger.info(f"  绿框长线: V={len(long_vlines)}条, H={len(long_hlines)}条")
    if long_vlines:
        logger.info(f"    长竖线(x): {[v[0] for v in long_vlines]}")
    if long_hlines:
        logger.info(f"    长横线(y): {[h[0] for h in long_hlines]}")

    # L2/L3 辅助：搜索 + 截断扩展修复
    def _search_and_extend(crop, layer_name, v_rank, h_rank):
        """在裁切区域 OCR，若截断则扩展右侧重试。返回 (text, bbox) 或 None。"""
        result = _search_y_number(sub_image, crop, y_re, prefixes=prefixes)
        if not result:
            return None
        y_text, text_bbox = result
        y_text = _fix_digit_letter_confusion(y_text)
        if _validate_green_result(y_text):
            logger.info(f"找到右下角编号 ({layer_name}长线裁切): '{y_text}' at {text_bbox}")
            return (y_text, text_bbox)
        # 截断修复：≥7 字符但未通过验证 → 扩展右侧重新 OCR
        stripped = y_text.strip()
        if len(stripped) >= 7:
            char_w = text_bbox.w / max(len(y_text), 1)
            ext_r = min(int(char_w * 2.5), img_w - text_bbox.x2)
            ext_v = int(text_bbox.h * 0.5)
            ext_bbox = BBox(
                text_bbox.x,
                max(0, text_bbox.y - ext_v),
                text_bbox.w + ext_r,
                text_bbox.h + ext_v * 2,
            )
            if ext_bbox.x2 > img_w:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                img_w - ext_bbox.x, ext_bbox.h)
            if ext_bbox.y2 > img_h:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                ext_bbox.w, img_h - ext_bbox.y)
            if ext_r > 0:
                logger.info(f"  绿框{layer_name}: '{y_text}' 截断，扩展 +{ext_r}px右 重新OCR")
                re_result = _search_y_number(
                    sub_image, ext_bbox, y_re, prefixes=prefixes
                )
                if re_result:
                    re_text, re_bbox = re_result
                    re_text = _fix_digit_letter_confusion(re_text)
                    if _validate_green_result(re_text):
                        logger.info(f"找到右下角编号 ({layer_name}扩展修复): "
                                    f"'{y_text}' → '{re_text}'")
                        return (re_text, re_bbox)
        logger.info(f"  绿框{layer_name}: '{y_text}' 未通过验证")
        return None

    # L2: 右数第3根长竖线左侧 + 下数第3根长横线上方 裁掉
    crop2 = _crop_by_long_lines(img_h, img_w, long_vlines, long_hlines, 3, 3)
    if crop2 is not None:
        logger.info(f"  绿框L2: 长线裁切(v3,h3) → {crop2}")
        found = _search_and_extend(crop2, "L2", 3, 3)
        if found:
            return found

    # L3: 右数第2根长竖线左侧 + 下数第2根长横线上方 裁掉
    crop3 = _crop_by_long_lines(img_h, img_w, long_vlines, long_hlines, 2, 2)
    if crop3 is not None and (crop2 is None or crop3 != crop2):
        logger.info(f"  绿框L3: 长线裁切(v2,h2) → {crop3}")
        found = _search_and_extend(crop3, "L3", 2, 2)
        if found:
            return found

    # ── Layer 4: 预处理增强 + 全区域 OCR（处理大字号/低对比度编号）──
    preprocessed = _preprocess_green_ocr(sub_image)
    result = _search_y_number(preprocessed, search, y_re, prefixes=prefixes)
    if result:
        y_text, text_bbox = result
        y_text = _fix_digit_letter_confusion(y_text)
        if _validate_green_result(y_text):
            logger.info(f"找到右下角编号 (L4预处理): '{y_text}' at {text_bbox}")
            return (y_text, text_bbox)
        # 截断扩展
        stripped = y_text.strip()
        if len(stripped) >= 7:
            char_w = text_bbox.w / max(len(y_text), 1)
            ext_r = min(int(char_w * 2.5), img_w - text_bbox.x2)
            ext_v = int(text_bbox.h * 0.5)
            ext_bbox = BBox(
                text_bbox.x,
                max(0, text_bbox.y - ext_v),
                text_bbox.w + ext_r,
                text_bbox.h + ext_v * 2,
            )
            if ext_bbox.x2 > img_w:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                img_w - ext_bbox.x, ext_bbox.h)
            if ext_bbox.y2 > img_h:
                ext_bbox = BBox(ext_bbox.x, ext_bbox.y,
                                ext_bbox.w, img_h - ext_bbox.y)
            if ext_r > 0:
                logger.info(f"  绿框L4: '{y_text}' 截断，扩展 +{ext_r}px右 重新OCR")
                re_result = _search_y_number(
                    preprocessed, ext_bbox, y_re, prefixes=prefixes
                )
                if re_result:
                    re_text, re_bbox = re_result
                    re_text = _fix_digit_letter_confusion(re_text)
                    if _validate_green_result(re_text):
                        logger.info(f"找到右下角编号 (L4扩展修复): "
                                    f"'{y_text}' → '{re_text}'")
                        return (re_text, re_bbox)
        logger.info(f"  绿框L4: '{y_text}' 未通过验证")

    logger.info("  绿框: 所有层均未找到有效编号")
    return None




# ══════════════════════════════════════════════════════════════════
#  Phase B-3: 定位左上角编号栏
# ══════════════════════════════════════════════════════════════════

def _enhance_vertical_lines(image: np.ndarray) -> np.ndarray:
    """增强图像中的竖线，使 OCR 能正确按竖线分段识别文字。

    对细竖线进行形态学增强：检测竖线 → 加粗 → 叠加回原图。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 二值化（反转：线条为白色）
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 竖线形态学提取：用高窄核开运算
    vert_kernel_h = max(h // 4, 15)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_h))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    # 加粗竖线（膨胀）
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    vert_lines_thick = cv2.dilate(vert_lines, dilate_kernel, iterations=1)

    # 叠加回原图：竖线位置变黑
    result = image.copy()
    result[vert_lines_thick > 0] = 0

    return result


def _locate_top_left_number_core(
    sub_image: np.ndarray,
    gray: np.ndarray,
    img_h: int, img_w: int,
    material_code_bbox: BBox | None = None,
    prefixes: list[str] = None,
    vlines: list = None,
) -> tuple[str, BBox] | None:
    """在子图中用多策略 OCR 搜索编号（橙框核心逻辑）。

    策略：
    1. OCR 子图，直接搜索独立的编号文字
    2. 若编号与 DWG NO / 图号 合体识别，用局部重OCR从合体文本中精确定位编号
    3. 兜底搜索所有 OCR 文本
    返回 (文本, BBox) 或 None，坐标为子图内坐标。
    """
    y_re = re.compile(make_pattern(prefixes))

    search = BBox(0, 0, img_w, img_h)
    ocr_results = _ocr_region(sub_image, search, engine="vlm")

    # ── 策略 1：找独立的 Y 编号文字 ──
    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        text_up = text.upper().strip()
        # 跳过 DWG NO / 图号 合体文本（策略 2 处理）
        if any(text_up.startswith(prefix) for prefix in ("DWG", "OWG", "図", "图")):
            continue
        # 优先用去空格文本匹配（避免空格截断编号），取更长结果
        text_no_sp = text_up.replace(" ", "")
        m_orig = y_re.search(text_up)
        m_nosp = y_re.search(text_no_sp)
        # 取匹配更长的结果；相同则优先去空格版（完整编号）
        if m_orig and m_nosp:
            m = m_nosp if len(m_nosp.group()) >= len(m_orig.group()) else m_orig
        else:
            m = m_nosp or m_orig
        if not m:
            fuzzy_text = _fuzzy_fix_y_text(text_no_sp)
            if fuzzy_text:
                m = y_re.search(fuzzy_text)
                if m:
                    logger.info(f"  策略1模糊回填: '{text_no_sp}' → '{fuzzy_text}'")
        if m:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = BBox(int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            # 排除在 material_code 区域内的
            if material_code_bbox and material_code_bbox.contains(bbox):
                continue
            matched_text = m.group()
            # OCR文本包含多余字符 → 从左侧竖线逐步截取重新OCR
            if len(text.strip()) > len(matched_text) and bbox.w > 0 and vlines:
                sorted_vx = sorted(
                    [x for x, _, _ in vlines if bbox.x < x < bbox.x2],
                )
                for vi, vx in enumerate(sorted_vx):
                    crop_bbox = BBox(vx, max(bbox.y - 5, 0),
                                     bbox.x2 - vx, min(bbox.h + 10, img_h - max(bbox.y - 5, 0)))
                    logger.info(f"  合框裁切第{vi+1}次: 从竖线x={vx}截取, crop={crop_bbox}")
                    local_results = _ocr_region(sub_image, crop_bbox, engine="vlm")
                    for lt, lc, lp in local_results:
                        lt_up = lt.upper().replace(" ", "")
                        lm = y_re.search(lt_up)
                        if not lm and _fuzzy_fix_y_text(lt_up):
                            lm = y_re.search(_fuzzy_fix_y_text(lt_up))
                        if lm and lp and len(lt.strip()) == len(lm.group()):
                            lxs = [p[0] for p in lp]
                            lys = [p[1] for p in lp]
                            bbox = BBox(int(min(lxs)), int(min(lys)),
                                        int(max(lxs) - min(lxs)), int(max(lys) - min(lys)))
                            matched_text = lm.group()
                            logger.info(f"  竖线裁切OCR成功: '{matched_text}' at {bbox}")
                            return (matched_text, bbox)
                logger.info(f"  竖线裁切均未得到纯编号，使用原始bbox")
            # 竖排文字（h/w > 3）：直接使用 OCR bbox
            if bbox.w > 0 and bbox.h > bbox.w * 3:
                logger.info(f"  找到竖排 Y 编号: '{text}' at {bbox}")
                return (matched_text, bbox)
            logger.info(f"  找到独立 Y 编号: '{matched_text}' at {bbox}")
            return (matched_text, bbox)

    # 诊断日志：策略1 未命中时输出 OCR 内容
    if not any(y_re.search(t.upper().replace(" ", "")) for t, _, _ in ocr_results):
        logger.info(f"  橙框 OCR 未发现任何 Y 编号 (共 {len(ocr_results)} 条)")
    for t, c, _ in ocr_results:
        logger.info(f"    橙框OCR: '{t}' (conf={c:.2f})")

    # ── 策略 2：从 DWG NO / 图号 合体文本中提取编号位置 ──
    dwg_keywords = ["DWG NO", "DWG NO.", "DWGNO", "DWG", "図番", "図面番号", "图号"]
    dwg_match = _fuzzy_find_keyword(ocr_results, dwg_keywords, threshold=0.60)

    # 英文 OCR 未找到关键词 → 用中文 OCR 回退
    if dwg_match is None:
        ocr_results_ch = _ocr_region(sub_image, search, lang=OCR_LANG_CH, engine="vlm")
        dwg_match = _fuzzy_find_keyword(ocr_results_ch, dwg_keywords, threshold=0.60)
        if dwg_match:
            logger.info(f"  中文OCR回退找到关键词: '{dwg_match['text']}' (score={dwg_match['score']:.2f})")

    if dwg_match and dwg_match.get("poly"):
        logger.info(
            f"  找到 DWG NO 关键词: '{dwg_match['text']}' "
            f"(score={dwg_match['score']:.2f})"
        )
        text_up = dwg_match["text"].upper()

        # OCR 可能在编号中插入空格（如 "YA026 D 941"），去除后再匹配
        # 先找 DWG NO / 图号 前缀的结束位置
        prefix_end = 0
        for prefix in ["DWG NO.", "DWG NO", "DWGNO", "DWG", "図面番号", "図番", "图号"]:
            if text_up.startswith(prefix) or dwg_match["text"].startswith(prefix):
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

            # DWG NO 和编号之间有竖线分隔，字符等比例估算不可靠。
            # 在子图中对编号区域重新做局部 OCR 定位：
            #   先用字符比例粗定位编号首字母 x，然后从该位置到 polygon 右边界截取子图，
            #   对子图做 OCR 精确定位编号 bbox。
            poly = dwg_match["poly"]
            poly_left = min(p[0] for p in poly)
            poly_right = max(p[0] for p in poly)
            poly_top = min(p[1] for p in poly)
            poly_bot = max(p[1] for p in poly)
            poly_w = poly_right - poly_left
            poly_h = poly_bot - poly_top

            total_chars = max(len(text_up), 1)
            # 粗略估算编号首字母 x 位置，左移 1 字符宽度作为搜索起点
            char_w_est = poly_w / total_chars
            search_x = max(int(poly_left + poly_w * (match_start_orig / total_chars) - char_w_est), int(poly_left))

            # 截取从编号首字母到 polygon 右边界的子图做精确 OCR
            num_sub_x = search_x
            num_sub_w = int(poly_right) - num_sub_x
            num_sub_y = max(int(poly_top) - 5, 0)
            num_sub_h = int(poly_h) + 10
            if num_sub_w > 10 and num_sub_h > 5:
                num_sub_bbox = BBox(num_sub_x, num_sub_y, num_sub_w, num_sub_h)
                local_results = _ocr_region(sub_image, num_sub_bbox, engine="vlm")
                # 在局部 OCR 结果中找编号
                local_found = False
                for lt, lc, lp in local_results:
                    lt_up = lt.upper().replace(" ", "")
                    lm = y_re.search(lt_up)
                    if not lm and _fuzzy_fix_y_text(lt_up):
                        lm = y_re.search(_fuzzy_fix_y_text(lt_up))
                    if lm and lp:
                        lxs = [p[0] for p in lp]
                        lys = [p[1] for p in lp]
                        bbox = BBox(int(min(lxs)), int(min(lys)),
                                    int(max(lxs) - min(lxs)), int(max(lys) - min(lys)))
                        matched_text = lm.group()
                        logger.info(f"  DWG NO 合体 → 局部重OCR定位: '{matched_text}' at {bbox}")
                        local_found = True
                        return (matched_text, bbox)

            # 局部 OCR 未命中，回退：从编号首字母估算位置到 polygon 右边界
            num_x_left = poly_left + poly_w * (match_start_orig / total_chars)
            bbox = BBox(
                int(num_x_left), int(poly_top),
                max(int(poly_right - num_x_left), 1),
                max(int(poly_h), 1),
            )
            matched_text = m.group()
            logger.info(
                f"  从 DWG NO 合体文本提取编号: '{matched_text}' at {bbox}"
            )
            return (matched_text, bbox)

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
                poly_top = min(p[1] for p in poly)
                poly_bot = max(p[1] for p in poly)
                bbox = BBox(int(num_x_left), int(poly_top),
                            max(int(poly_right - num_x_left), 1),
                            max(int(poly_bot - poly_top), 1))
            matched_text = m.group()
            logger.info(f"  策略3 兜底找到 Y 编号: '{text}' -> '{matched_text}' at {bbox}")
            return (matched_text, bbox)

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
        return (None, est_bbox)

    logger.info("  左上角未找到 Y 编号")
    return None


def _locate_top_left_number(
    sub_image: np.ndarray,
    material_code_bbox: BBox | None = None,
    prefixes: list[str] = None,
) -> tuple[str, BBox] | None:
    """在子图（橙框搜索区裁切）中找到编号栏。

    流程：
      1. OCR 搜索区 → 找到 Y 编号 OCR bbox
      2. morph 线检测
      3. cell 边界（最近的线围成单元格）
      4. 比较 cell 和 OCR bbox → 最终框

    返回 (文本, BBox) 或 None，坐标为子图内坐标。
    """
    img_h, img_w = sub_image.shape[:2]
    gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)

    vlines, hlines = _detect_morph_lines(gray, v_ratio=6.5, enhance_lines=True)
    logger.info(f"  橙框morph线: {len(vlines)}条竖线, {len(hlines)}条横线")

    found = _locate_top_left_number_core(
        sub_image, gray, img_h, img_w, material_code_bbox, prefixes,
        vlines=vlines,
    )
    if found is None:
        return None

    y_text, ocr_bbox = found

    cell = _find_cell_from_lines(ocr_bbox, vlines, hlines, img_w)
    orange = _make_orange_bbox(ocr_bbox, cell)
    logger.info(f"  最终橙框(子图坐标): {orange}, text='{y_text}'")

    return (y_text, orange)


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
    """纵向图纸始终向左旋转90°（CCW）至横向。"""
    img_h, img_w = image.shape[:2]
    logger.info(f"纵向图纸 ({img_w}x{img_h})，向左旋转90°...")
    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated, cv2.ROTATE_90_COUNTERCLOCKWISE


# ══════════════════════════════════════════════════════════════════
#  统一检测入口
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  工厂注意部分（Factory Note）检测
# ══════════════════════════════════════════════════════════════════

_layout_model = None


def _get_layout_model():
    global _layout_model
    if _layout_model is None:
        from paddlex import create_model
        _layout_model = create_model("PP-DocLayoutV3")
    return _layout_model


def _detect_factory_note_candidates(image_pil, exclude_bboxes):
    """用 PP-DocLayoutV3 检测布局块，排除红/绿/橙框区域，返回工厂注意候选区域。"""
    from PIL import Image
    img_w, img_h = image_pil.size

    model = _get_layout_model()
    result = list(model.predict(np.array(image_pil), batch_size=1))
    if not result:
        return []

    res = result[0]
    all_blocks = []
    if hasattr(res, "boxes"):
        for item in res.boxes:
            coord = item["coordinate"]
            all_blocks.append({
                "x1": int(coord[0]), "y1": int(coord[1]),
                "x2": int(coord[2]), "y2": int(coord[3]),
                "label": item.get("label", ""),
                "score": float(item.get("score", 0)),
            })
    elif isinstance(res, dict) and "boxes" in res:
        for item in res["boxes"]:
            coord = item["coordinate"]
            all_blocks.append({
                "x1": int(coord[0]), "y1": int(coord[1]),
                "x2": int(coord[2]), "y2": int(coord[3]),
                "label": item.get("label", ""),
                "score": float(item.get("score", 0)),
            })

    logger.info(f"Factory Note: PP-DocLayoutV3 检测到 {len(all_blocks)} 个布局块")

    candidates = []
    for blk in all_blocks:
        bw = blk["x2"] - blk["x1"]
        bh = blk["y2"] - blk["y1"]
        if bw * bh < 200 * 50:
            continue

        overlaps = False
        for eb in exclude_bboxes:
            ex1, ey1, ex2, ey2 = eb
            overlap_x = max(0, min(blk["x2"], ex2) - max(blk["x1"], ex1))
            overlap_y = max(0, min(blk["y2"], ey2) - max(blk["y1"], ey1))
            if bw > 0 and bh > 0 and (overlap_x * overlap_y) / (bw * bh) > 0.5:
                overlaps = True
                break
        if not overlaps:
            candidates.append(blk)

    logger.info(f"Factory Note: {len(candidates)} 个候选区域（排除红/绿/橙框后）")
    return candidates


def _detect_factory_note_codes(image_rgb, candidates, prefixes=None):
    """对每个工厂注意候选区域做 OCR，提取编号列表。

    使用通用 _ocr_region（支持 v5 和 VLM），一次 OCR 同时获取文本+坐标。
    返回: [{"code": str, "bbox": BBox, "confidence": float}, ...]
    """
    prefixes = prefixes or DEFAULT_PREFIXES
    p_chars = "".join(p.upper() for p in prefixes)
    p_class = f"[{p_chars}]" if len(p_chars) > 1 else p_chars
    y_code_loose = re.compile(rf"{p_class}[A-Z]\d[A-Z0-9]{{3,}}")
    y_code_strict = re.compile(rf"{p_class}[A-Z]\d{{2,3}}[A-Z]\d{{2,4}}")
    found_codes = []

    from PIL import Image as PILImage
    if isinstance(image_rgb, PILImage.Image):
        image_rgb = np.array(image_rgb)

    for i, cand in enumerate(candidates):
        x1, y1 = cand["x1"], cand["y1"]
        x2, y2 = cand["x2"], cand["y2"]
        cw, ch = x2 - x1, y2 - y1
        if cw < 10 or ch < 10:
            continue

        cand_bbox = BBox(x1, y1, cw, ch)
        ocr_results = _ocr_region(image_rgb, cand_bbox)

        has_code = False
        for text, conf, poly in ocr_results:
            text_upper = text.upper().replace(" ", "")
            m = y_code_loose.search(text_upper)
            if not m:
                continue
            has_code = True
            code = m.group()
            confidence = 1.0 if y_code_strict.match(code) else 0.7

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
                    "bbox": BBox(code_bx1, int(poly_y1), code_bx2 - code_bx1, int(poly_y2 - poly_y1)),
                    "confidence": confidence,
                })
            else:
                found_codes.append({
                    "code": code,
                    "bbox": cand_bbox,
                    "confidence": confidence,
                })

        if not has_code:
            logger.info(f"  Factory Note 候选 {i}: OCR 无编号")
        else:
            logger.info(f"  Factory Note 候选 {i}: 发现编号")

    logger.info(f"Factory Note: 共发现 {len(found_codes)} 个编号")
    return found_codes


def detect_all_regions(
    image: np.ndarray, region_config: dict = None, prefixes: list[str] = None
) -> dict[str, BBox | None]:
    """
    检测图纸中所有目标区域。

    策略：裁切3个搜索区子图，3路并行OCR检测（红/绿/橙）。
    各搜索区裁切后，若最长边超过 2100px 则缩放后再做 OCR/CV。

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

    # 红框搜索区域: 图片8%起点 ~ 画框2.8/8, 从图片顶部到95%高度
    red_x = int(img_w * 0.08)
    red_w = frame.x + int(frame.w * (2.8 / 8.0)) - red_x
    red_h = int(img_h * 0.95)
    red_search = BBox(red_x, 0, red_w, red_h)
    logger.info(f"  红框搜索区域: {red_search}")

    # 绿框搜索区域: 左 5/8 ~ 右边界, 上 5/6 ~ 图片下边界
    green_x = int(img_w * (5.0 / 8.0))
    green_y = int(img_h * (5.0 / 6.0))
    green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)
    logger.info(f"  绿框搜索区域: {green_search}")

    # 橙框搜索区域: 依赖红框结果，Phase B 后动态计算

    # ── Phase B: 裁切子图 + 4路并行 OCR 检测 ──
    # 每个线程独立创建 OCR 引擎实例（线程本地缓存），避免共享 predictor
    from concurrent.futures import ThreadPoolExecutor
    logger.info("Phase B: 裁切子图并行检测（3线程）...")

    red_sub, red_scale = _crop_and_scale(image, red_search)
    green_sub, green_scale = _crop_and_scale(image, green_search)
    logger.info(f"  红框裁切: {red_sub.shape[1]}x{red_sub.shape[0]} (scale={red_scale:.3f})")
    logger.info(f"  绿框裁切: {green_sub.shape[1]}x{green_sub.shape[0]} (scale={green_scale:.3f})")

    def _detect_red():
        return _locate_material_code_column(red_sub)

    def _detect_green():
        return _locate_bottom_right_number(green_sub, prefixes=prefixes)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_red = executor.submit(_detect_red)
        f_green = executor.submit(_detect_green)

        mat_result = f_red.result()
        br_result = f_green.result()

    # ── 汇总红框/绿框结果 ──
    table_search_bbox = red_search
    if mat_result[0] is not None:
        mat_bbox_sub, direction = mat_result
        mat_bbox = _map_bbox_back(mat_bbox_sub, red_search, red_scale)
        metadata["table_direction"] = direction
    else:
        mat_bbox = None
        logger.info("  红框: 关键词检测失败")
        metadata["method"] = "left_half"

    if br_result:
        br_text, br_bbox_sub = br_result
        br_bbox = _map_bbox_back(br_bbox_sub, green_search, green_scale)
        metadata["bottom_right_text"] = br_text
    else:
        br_bbox = None
        br_bbox = _fallback_detect_bottom_right_number(image, region_config.get("bottom_right_title"))

    # ── 橙框搜索区域: 基于红框结果动态计算 ──
    # 右边界 = 红框左边界, 下边界 = 红框垂直中线
    tl_bbox = None
    if mat_bbox is not None:
        orange_right = mat_bbox.x
        orange_bottom = mat_bbox.y
        orange_search = BBox(0, 0, orange_right, orange_bottom)
        logger.info(f"  橙框搜索区域: {orange_search} (基于红框 x={mat_bbox.x}, 上边界y={orange_bottom})")

        orange_sub, orange_scale = _crop_and_scale(image, orange_search)
        logger.info(f"  橙框裁切: {orange_sub.shape[1]}x{orange_sub.shape[0]} (scale={orange_scale:.3f})")

        tl_result = _locate_top_left_number(orange_sub, None, prefixes=prefixes)
        if tl_result:
            tl_text, tl_bbox_sub = tl_result
            tl_bbox = _map_bbox_back(tl_bbox_sub, orange_search, orange_scale)
            metadata["top_left_text"] = tl_text
    else:
        orange_search = BBox(0, 0, int(img_w * (2.0 / 8.0)), int(img_h * (2.0 / 12.0)))
        logger.info(f"  橙框搜索区域(fallback): {orange_search} (红框未检测到)")

        orange_sub, orange_scale = _crop_and_scale(image, orange_search)
        tl_result = _locate_top_left_number(orange_sub, None, prefixes=prefixes)
        if tl_result:
            tl_text, tl_bbox_sub = tl_result
            tl_bbox = _map_bbox_back(tl_bbox_sub, orange_search, orange_scale)
            metadata["top_left_text"] = tl_text

    # ── 组装结果 ──
    metadata["table_search_area"] = table_search_bbox
    metadata["search_areas"] = {
        "red_search": red_search,
        "green_search": green_search,
        "orange_search": orange_search,
    }
    metadata["crop_scales"] = {
        "red": red_scale,
        "green": green_scale,
        "orange": orange_scale,
    }

    result = {
        "material_code_column": mat_bbox,
        "bottom_right_number": br_bbox,
        "top_left_number": tl_bbox,
        "_metadata": metadata,
    }

    for name, bbox in result.items():
        if name.startswith("_"):
            continue
        if bbox:
            logger.info(f"  {name}: {bbox}")
        else:
            logger.info(f"  {name}: 未检测到")

    # ── Phase C: 工厂注意部分检测 ──
    # 主路径：纯像素 V6（V5 粗扫 → VLM 精定位）；失败回退到 PP-DocLayoutV3。
    logger.info("Phase C: 工厂注意部分检测 ...")
    fn_codes: list = []
    fn_source = "v6"
    try:
        from modules.factory_note_pixel import detect_factory_note_codes_v6
        fn_codes = detect_factory_note_codes_v6(image, result, prefixes=prefixes)
    except Exception as e:
        logger.warning(f"Factory Note v6 失败，回退 PP-DocLayoutV3: {e}", exc_info=True)
        fn_source = "pp_doclayout_v3"
        from PIL import Image as PILImage
        image_pil = PILImage.fromarray(image)
        exclude_bboxes = []
        for name in ("material_code_column", "bottom_right_number", "top_left_number"):
            b = result.get(name)
            if b is not None:
                exclude_bboxes.append((b.x, b.y, b.x2, b.y2))
        try:
            fn_candidates = _detect_factory_note_candidates(image_pil, exclude_bboxes)
            fn_codes = _detect_factory_note_codes(image_pil, fn_candidates, prefixes)
            metadata["factory_note_candidates"] = fn_candidates
            metadata["factory_note_candidate_count"] = len(fn_candidates)
        except Exception as e2:
            logger.warning(f"Factory Note PP-DocLayoutV3 fallback 也失败: {e2}")
            fn_codes = []

    result["factory_note_codes"] = fn_codes
    metadata["factory_note_source"] = fn_source
    logger.info(f"Factory Note: {len(fn_codes)} 个编号 (source={fn_source})")

    # 纵向图纸：不进行坐标逆映射，保持旋转后图像的坐标系
    if rot_code is not None:
        logger.info("保持旋转后图像的坐标系（不映射回原始方向）")

    return result


def draw_regions_debug(image: np.ndarray, regions: dict) -> np.ndarray:
    """在图像上画出检测到的区域边界（调试用）"""
    colors = {
        "material_code_column": (255, 0, 0),
        "bottom_right_number": (0, 255, 0),
        "top_left_number": (255, 165, 0),
    }
    debug_img = image.copy()

    # 画区域框
    for name, bbox in regions.items():
        if name.startswith("_") or name == "factory_note_codes" or bbox is None:
            continue
        color = colors.get(name, (128, 128, 128))
        bboxes = bbox if isinstance(bbox, list) else [bbox]
        for b in bboxes:
            if b is None:
                continue
            cv2.rectangle(debug_img, (b.x, b.y), (b.x2, b.y2), color, 3)
            cv2.putText(
                debug_img, name, (b.x, b.y - 10),
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

    # 绘制蓝色框（工厂注意部分编号）
    fn_codes = regions.get("factory_note_codes", [])
    for fc in fn_codes:
        b = fc["bbox"]
        cv2.rectangle(debug_img, (b.x, b.y), (b.x2, b.y2), (255, 0, 0), 3)
        cv2.putText(
            debug_img, fc.get("code", "factory_note"), (b.x, b.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
        )

    return debug_img
