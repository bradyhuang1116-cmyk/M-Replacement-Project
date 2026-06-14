"""工厂注意区域 Y 编号检测 — 纯像素 V6（OCR 粗扫 + VLM 精定位）。

整套流程：
  1. 用 detect_all_regions 已得到的红/绿/橙框，推导工厂注意搜索 ROI；
  2. erase_drawing_border 擦图纸外框线 → find_bottom_span_y 性能裁切；
  3. cluster_table_zone 用横/竖线聚类得到主表 zone + exclusion_zone；
  4. detect_tables（V3 strict）+ process_variant（h_factor=8）得文字块候选；
  5. V5 粗扫每个 crop 是否包含 Y 编号；命中才进 VLM 精定位 _vlm_annotate_single；
  6. 把 VLM 命中的每个 polygon 换算回原图坐标，输出 [{code, bbox, confidence}]。

对外接口：detect_factory_note_codes_v6(image_rgb, regions, prefixes) -> list[dict]
内部跨文件汇总：_y_box_records 列表；调 flush_y_boxes_csv(path) 落盘。
"""

import os
import re
import csv
import logging

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, Y_PATTERN
from modules.region_detector import (
    BBox,
    _get_ocr_v5,
    _parse_ocr_results_common,
)

logger = logging.getLogger(__name__)


# ── VLM/V5 引擎 lazy 句柄 ──────────────────────────────────────
_vlm_engine = None
_v5_engine = None


def _get_vlm():
    global _vlm_engine
    if _vlm_engine is None:
        from modules.vlm_ocr_engine import get_vlm_engine
        _vlm_engine = get_vlm_engine()
    return _vlm_engine


def _get_v5():
    global _v5_engine
    if _v5_engine is None:
        _v5_engine = _get_ocr_v5("en")
    return _v5_engine


# ── Y 编号正则（含 V→Y 误识兜底）────────────────────────────────
_FN_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_Y_RE = re.compile(Y_PATTERN)
_V_RE = re.compile(r"V(?=[A-Z0-9]*\d)[A-Z0-9]{6,}")


def _safe_filename_tag(text: str, max_len: int = 32) -> str:
    cleaned = _FN_SAFE_RE.sub("_", (text or "").strip())[:max_len].strip("_.")
    return cleaned or "hit"


def _find_y_token(text: str) -> str | None:
    """在 OCR 文本里搜符合 Y_PATTERN 的 token；首字母为 V 时回填为 Y。"""
    if not text:
        return None
    up = text.upper()
    m = _Y_RE.search(up)
    if m:
        return m.group()
    m = _V_RE.search(up)
    if m:
        return "Y" + m.group()[1:]
    return None


# ── 基础几何 ───────────────────────────────────────────────────
def _poly_bbox(poly):
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _column_runs(bw):
    col_sum = (bw > 0).sum(axis=0)
    runs = []
    in_run = False
    rs = 0
    for c in range(len(col_sum)):
        if col_sum[c] > 0 and not in_run:
            rs = c; in_run = True
        elif col_sum[c] == 0 and in_run:
            runs.append((rs, c)); in_run = False
    if in_run:
        runs.append((rs, len(col_sum)))
    return runs


# ── V5 粗扫 ───────────────────────────────────────────────────
def _v5_run(pil_crop):
    np_img = np.array(pil_crop)
    try:
        result = _get_v5().predict(np_img)
    except Exception as e:
        logger.warning(f"  V5 OCR 异常: {e}")
        return np_img, []
    items = _parse_ocr_results_common(result)
    return np_img, items


def _v5_filter_y(items):
    texts = [t for _, t, _ in items]
    hits: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if not t:
            continue
        tok = _find_y_token(t)
        if not tok:
            continue
        if tok not in seen:
            seen.add(tok)
            hits.append(tok)
    return bool(hits), hits, texts


# ── VLM 精定位：在 polygon 包围盒内把红框收紧到只覆盖 Y token ──
def _localize_y_box(crop_np, poly, target_tok):
    if not target_tok:
        return None
    x1, y1, x2, y2 = _poly_bbox(poly)
    H, W = crop_np.shape[:2]
    char_h = max(8, y2 - y1)
    pad = max(int(char_h * 0.6), 12)
    x1 = max(0, x1 - pad); y1 = max(0, y1)
    x2 = min(W, x2 + pad); y2 = min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 6:
        return None

    sub = crop_np[y1:y2, x1:x2]
    vlm = _get_vlm()

    _norm_table = str.maketrans({'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8'})

    def _norm(s: str) -> str:
        return (s or "").upper().replace(" ", "").translate(_norm_table)

    tok_candidates = [target_tok.upper()]
    short = target_tok.upper()[:-1]
    if len(short) >= 7 and _Y_RE.fullmatch(short):
        tok_candidates.append(short)

    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY) if sub.ndim == 3 else sub
    _, bwm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    runs = _column_runs(bwm)
    step = max(4, (x2 - x1) // 200)
    grid = list(range(0, x2 - x1, step))
    starts_set = sorted({r[0] for r in runs} | set(grid) | {0})
    ends_set = sorted({r[1] for r in runs} | set(grid) | {x2 - x1})

    def _fallback_box(left_col: int, right_col: int, tok_used: str):
        left_col = max(0, min(left_col, x2 - x1))
        right_col = max(left_col + 1, min(right_col, x2 - x1))
        ax1 = x1 + left_col
        ax2 = x1 + right_col
        return [(ax1, y1), (ax2, y1), (ax2, y2), (ax1, y2)], tok_used

    def _snap_to_runs(left_col: int, right_col: int):
        overlapping = [(rs, re_) for rs, re_ in runs
                       if re_ > left_col and rs < right_col]
        if not overlapping:
            return left_col, right_col
        char_w = max(8, (right_col - left_col) // 12)
        import config as _cfg
        if _cfg.VLM_PROVIDER != "paddleocr_api":
            # 本地模式：保持原逻辑（取所有重叠笔画 min/max + 外扩一个字宽），
            # 本地框本就紧凑，不做云端那套聚类收紧
            new_left = min(rs for rs, _ in overlapping)
            new_right = max(re_ for _, re_ in overlapping)
            new_left = max(new_left, left_col - char_w)
            new_right = min(new_right, right_col + char_w)
            return new_left, new_right
        # 云端模式：按字间隙聚类，只保留覆盖范围最大的连续簇，
        # 剔除两端孤立的相邻字符/标点笔画，避免框过宽
        overlapping.sort()
        gap_thresh = max(int(char_w * 1.2), 14)
        clusters = [[overlapping[0]]]
        for rs, re_ in overlapping[1:]:
            if rs - clusters[-1][-1][1] > gap_thresh:
                clusters.append([(rs, re_)])
            else:
                clusters[-1].append((rs, re_))
        best = max(clusters, key=lambda cl: cl[-1][1] - cl[0][0])
        new_left = min(rs for rs, _ in best)
        new_right = max(re_ for _, re_ in best)
        new_left = max(new_left, left_col)
        new_right = min(new_right, right_col)
        return new_left, new_right

    def _run_for_tok(tok_up: str):
        alt_tok = "V" + tok_up[1:] if tok_up.startswith("Y") else None
        tok_n = _norm(tok_up)
        alt_n = _norm(alt_tok) if alt_tok else None

        def _is_tok_exact(text_norm: str) -> bool:
            return text_norm == tok_n or (alt_n is not None and text_norm == alt_n)

        def _contains_tok(text_norm):
            if not text_norm:
                return False
            return tok_n in text_norm or (alt_n is not None and alt_n in text_norm)

        UPSCALE = 2.5
        try:
            sub_up = cv2.resize(sub, None, fx=UPSCALE, fy=UPSCALE,
                                interpolation=cv2.INTER_CUBIC)
            res_up = vlm.predict(sub_up)
            its_up = _parse_ocr_results_common(res_up)
        except Exception:
            its_up = []
        for poly2, text2, _sc in its_up:
            if poly2 is None or len(poly2) == 0:
                continue
            if not _is_tok_exact(_norm(text2)):
                continue
            ux1, uy1, ux2, uy2 = _poly_bbox(poly2)
            if ux2 - ux1 < max(len(tok_up) * 6, 24) or uy2 - uy1 < 4:
                continue
            sx1 = max(0, int(ux1 / UPSCALE))
            sx2 = min(sub.shape[1], int(ux2 / UPSCALE))
            return 'exact', sx1, sx2, tok_up

        if len(starts_set) < 2 or len(ends_set) < 2:
            return 'fail', 0, x2 - x1, ''

        min_w = max(len(tok_up) * 6, 30)
        _cache: dict = {}

        def _vlm_text_norm(left_col, right_col):
            if right_col - left_col < min_w:
                return None
            key = (left_col, right_col)
            if key in _cache:
                return _cache[key]
            s = sub[:, left_col:right_col]
            try:
                res = vlm.predict(s)
                its = _parse_ocr_results_common(res)
            except Exception:
                its = []
            text = _norm("".join((t or "") for _, t, _ in its))
            _cache[key] = text
            return text

        if not _contains_tok(_vlm_text_norm(starts_set[0], ends_set[-1])):
            return 'fail', 0, x2 - x1, ''

        full_right = ends_set[-1]
        lo, hi = 0, len(starts_set) - 1
        left_idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if _contains_tok(_vlm_text_norm(starts_set[mid], full_right)):
                left_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        left_col = starts_set[left_idx]

        valid_ends = [e for e in ends_set if e > left_col]
        if not valid_ends:
            return 'contains', left_col, x2 - x1, ''
        lo, hi = 0, len(valid_ends) - 1
        right_idx = len(valid_ends) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _contains_tok(_vlm_text_norm(left_col, valid_ends[mid])):
                right_idx = mid
                hi = mid - 1
            else:
                lo = mid + 1
        right_col = valid_ends[right_idx]

        def _shrink_left(cur_left, cur_right):
            candidates = [c for c in starts_set if cur_left <= c < cur_right]
            best = cur_left
            for c in candidates:
                txt = _vlm_text_norm(c, cur_right)
                if txt is None:
                    continue
                if _is_tok_exact(txt):
                    best = c
                elif not _contains_tok(txt):
                    break
            return best

        def _shrink_right(cur_left, cur_right):
            candidates = sorted([c for c in ends_set if cur_left < c <= cur_right],
                                reverse=True)
            best = cur_right
            for c in candidates:
                txt = _vlm_text_norm(cur_left, c)
                if txt is None:
                    continue
                if _is_tok_exact(txt):
                    best = c
                elif not _contains_tok(txt):
                    break
            return best

        tight_left = _shrink_left(left_col, right_col)
        tight_right = _shrink_right(tight_left, right_col)
        final_text = _vlm_text_norm(tight_left, tight_right) or ''
        if _is_tok_exact(final_text) and tight_right - tight_left >= min_w:
            return 'exact', tight_left, tight_right, tok_up

        try:
            sub_narrow = sub[:, left_col:right_col]
            sub_narrow_up = cv2.resize(sub_narrow, None, fx=UPSCALE, fy=UPSCALE,
                                       interpolation=cv2.INTER_CUBIC)
            res_n = vlm.predict(sub_narrow_up)
            its_n = _parse_ocr_results_common(res_n)
        except Exception:
            its_n = []
        for poly2, text2, _sc in its_n:
            if poly2 is None or len(poly2) == 0:
                continue
            if not _is_tok_exact(_norm(text2)):
                continue
            ux1, uy1, ux2, uy2 = _poly_bbox(poly2)
            if ux2 - ux1 < max(len(tok_up) * 6, 24) or uy2 - uy1 < 4:
                continue
            sx1 = max(0, int(ux1 / UPSCALE)) + left_col
            sx2 = min(sub.shape[1], int(ux2 / UPSCALE) + left_col)
            return 'phase4', sx1, sx2, tok_up

        return 'contains', left_col, right_col, ''

    results = []
    status_priority = {'exact': 0, 'phase4': 1, 'contains': 2, 'fail': 3}
    for idx, tok_cand in enumerate(tok_candidates):
        status, lc, rc, tk = _run_for_tok(tok_cand)
        if idx == 0:
            results.append((status_priority[status], rc - lc, status, lc, rc, tk or tok_cand))
            if status == 'exact':
                break
        else:
            results.append((status_priority[status], rc - lc, status, lc, rc, tk or tok_cand))

    results.sort(key=lambda r: (r[0], r[1]))
    _, _, status, lc, rc, tok_used = results[0]

    lc_snap, rc_snap = _snap_to_runs(lc, rc)
    return _fallback_box(lc_snap, rc_snap, tok_used)


def _vlm_annotate_single(pil_crop):
    """VLM 二次识别：返回 [(sub_poly, token), ...]，sub_poly 在 crop 局部坐标系。"""
    np_img = np.array(pil_crop)
    try:
        result = _get_vlm().predict(np_img)
    except Exception as e:
        logger.warning(f"  VLM predict 异常: {e}")
        return []
    items = _parse_ocr_results_common(result)

    matched = []
    for poly, text, _ in items:
        if poly is None or len(poly) == 0:
            continue
        tok = _find_y_token(text)
        if not tok:
            continue
        refined = _localize_y_box(np_img, poly, tok)
        if refined is not None:
            sub_poly, sub_tok = refined
            matched.append((sub_poly, sub_tok))
            continue
        logger.info(f"  VLM 二次定位极端兜底：用 VLM 原 polygon 画框，tok={tok}, text='{text}'")
        matched.append(([(int(p[0]), int(p[1])) for p in poly], tok))
    return matched


# ── V4 保留: 擦除搜索区图纸外框 ─────────────────────────────────
def erase_drawing_border(bw, h_ratio=0.95, v_ratio=0.95,
                         edge_band=0.05, dilate_px=6):
    h, w = bw.shape
    h_min_len = int(w * h_ratio)
    v_min_len = int(h * v_ratio)

    border_mask = np.zeros_like(bw)
    line_segs = []

    if h_min_len >= 10 and w >= h_min_len:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
        h_long = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
        num_h, _, stats_h, _ = cv2.connectedComponentsWithStats(h_long, connectivity=8)
        keep_idx_h = np.zeros(num_h, dtype=bool)
        edge_top = h * edge_band
        edge_bot = h * (1 - edge_band)
        for i in range(1, num_h):
            x, y, cw, ch, area = stats_h[i]
            yc = y + ch / 2.0
            if cw < h_min_len:
                continue
            if yc < edge_top or yc > edge_bot:
                keep_idx_h[i] = True
                line_segs.append({
                    "type": "h", "bbox": (int(x), int(y), int(x + cw), int(y + ch)),
                    "len": int(cw),
                })
        if keep_idx_h.any():
            labels_h = cv2.connectedComponents(h_long, connectivity=8)[1]
            border_mask = np.maximum(border_mask, (keep_idx_h[labels_h] * 255).astype(np.uint8))

    if v_min_len >= 10 and h >= v_min_len:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min_len))
        v_long = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
        num_v, _, stats_v, _ = cv2.connectedComponentsWithStats(v_long, connectivity=8)
        keep_idx_v = np.zeros(num_v, dtype=bool)
        edge_left = w * edge_band
        edge_right = w * (1 - edge_band)
        for i in range(1, num_v):
            x, y, cw, ch, area = stats_v[i]
            xc = x + cw / 2.0
            if ch < v_min_len:
                continue
            if xc < edge_left or xc > edge_right:
                keep_idx_v[i] = True
                line_segs.append({
                    "type": "v", "bbox": (int(x), int(y), int(x + cw), int(y + ch)),
                    "len": int(ch),
                })
        if keep_idx_v.any():
            labels_v = cv2.connectedComponents(v_long, connectivity=8)[1]
            border_mask = np.maximum(border_mask, (keep_idx_v[labels_v] * 255).astype(np.uint8))

    if dilate_px > 0 and border_mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        border_mask = cv2.dilate(border_mask, k, iterations=1)

    if not border_mask.any():
        return bw.copy(), [], border_mask

    bw_out = bw.copy()
    bw_out[border_mask > 0] = 0
    return bw_out, line_segs, border_mask


# ── V6 表格 zone 检测 ─────────────────────────────────────────
def cluster_table_zone(bw, mc_y1,
                       min_h_len_ratio=0.05, h_xrange_tol_ratio=0.02,
                       min_v_len_ratio=0.03, v_yrange_tol_ratio=0.03,
                       right_gap_ratio=1.8, buffer=20):
    sh, sw = bw.shape
    sx1, sy1 = 0, max(0, int(mc_y1))
    sx2, sy2 = sw, sh
    if sx2 - sx1 < 40 or sy2 - sy1 < 40:
        return None, {"error": f"search_rect too small: ({sx1},{sy1},{sx2},{sy2})"}

    roi = bw[sy1:sy2, sx1:sx2]
    rh, rw = roi.shape

    h_min_len = max(int(rw * min_h_len_ratio), 40)
    v_min_len = max(int(rh * min_v_len_ratio), 40)
    x_tol = max(int(rw * h_xrange_tol_ratio), 12)
    y_tol = max(int(rh * v_yrange_tol_ratio), 12)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel, iterations=1)
    num_h, _, stats_h, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    h_segs = []
    for i in range(1, num_h):
        x, y, w, h, _ = stats_h[i]
        if w < h_min_len:
            continue
        h_segs.append({
            "y": int(y + h // 2),
            "x1": int(x),
            "x2": int(x + w),
            "len": int(w),
        })

    if len(h_segs) < 2:
        return None, {"error": f"only {len(h_segs)} h_segs (need ≥2)",
                       "h_segs_total": int(num_h - 1)}

    best_h_set = []
    best_h_med_len = -1
    for anchor in h_segs:
        sub = [s for s in h_segs
               if abs(s["x1"] - anchor["x1"]) <= x_tol
               and abs(s["x2"] - anchor["x2"]) <= x_tol]
        if len(sub) < 2:
            continue
        lens = sorted(s["len"] for s in sub)
        med_len = lens[len(lens) // 2]
        if med_len > best_h_med_len:
            best_h_med_len = med_len
            best_h_set = sub

    if len(best_h_set) < 2:
        return None, {"error": f"no h aligned-end set (size≥2) found",
                       "h_segs_total": int(num_h - 1),
                       "h_segs_long": len(h_segs)}

    y_bottom_roi = max(s["y"] for s in best_h_set)
    h_x1_med = int(np.median([s["x1"] for s in best_h_set]))
    h_x2_med = int(np.median([s["x2"] for s in best_h_set]))

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min_len))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel, iterations=1)
    num_v, _, stats_v, _ = cv2.connectedComponentsWithStats(v_lines, connectivity=8)
    v_segs = []
    for i in range(1, num_v):
        x, y, w, h, _ = stats_v[i]
        if h < v_min_len:
            continue
        v_segs.append({
            "x": int(x + w // 2),
            "y1": int(y),
            "y2": int(y + h),
            "len": int(h),
        })

    best_v_set = []
    best_v_med_len = -1
    for anchor in v_segs:
        sub = [s for s in v_segs
               if abs(s["y1"] - anchor["y1"]) <= y_tol
               and abs(s["y2"] - anchor["y2"]) <= y_tol]
        if len(sub) < 2:
            continue
        lens = sorted(s["len"] for s in sub)
        med_len = lens[len(lens) // 2]
        if med_len > best_v_med_len:
            best_v_med_len = med_len
            best_v_set = sub

    x_tol_right = max(int(rw * 0.02), 20)

    v_near_h_right = [v["x"] for v in best_v_set
                      if abs(v["x"] - h_x2_med) <= x_tol_right] if best_v_set else []

    if v_near_h_right:
        x_right_roi = max(v_near_h_right)
    else:
        x_right_roi = h_x2_med

    zone_left = 0
    zone_top = max(0, sy1 - buffer)
    zone_bottom = min(sh, y_bottom_roi + sy1)
    zone_right = min(sw, x_right_roi + sx1)

    info = {
        "search_rect": (sx1, sy1, sx2, sy2),
        "mc_y1": int(mc_y1),
        "h_segs_total": int(num_h - 1),
        "h_segs_long": len(h_segs),
        "h_aligned_set_size": len(best_h_set),
        "h_x_median": (int(h_x1_med + sx1), int(h_x2_med + sx1)),
        "h_y_bottom_chosen": int(y_bottom_roi + sy1),
        "v_segs_total": int(num_v - 1),
        "v_segs_long": len(v_segs),
        "v_equal_height_set_size": len(best_v_set),
        "x_right_chosen": int(x_right_roi + sx1),
        "exclusion_zone": (0, 0, int(zone_right), int(zone_bottom)),
    }
    return (int(zone_left), int(zone_top),
            int(zone_right), int(zone_bottom)), info


# ── 表格抽取 ──────────────────────────────────────────────────
def extract_table_lines(bw, min_line_len_ratio=0.015):
    h, w = bw.shape
    min_len = max(int(max(h, w) * min_line_len_ratio), 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return h_lines, v_lines


def find_bottom_span_y(bw, len_ratio=0.9, y_floor_ratio=2.0 / 3.0):
    h, w = bw.shape
    h_lines, _ = extract_table_lines(bw)
    num, _, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    min_len = int(w * len_ratio)
    y_floor = h * y_floor_ratio
    best_yc = -1.0
    best_y_top = None
    for i in range(1, num):
        _, cy, cw, ch, _ = stats[i]
        if cw < min_len:
            continue
        yc = cy + ch / 2.0
        if yc <= y_floor:
            continue
        if yc > best_yc:
            best_yc = yc
            best_y_top = int(cy)
    return best_y_top


def detect_tables(bw, strict_level="strict", zone=None):
    h, w = bw.shape
    h_lines, v_lines = extract_table_lines(bw)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines_d = cv2.dilate(h_lines, dilate_k, iterations=1)
    v_lines_d = cv2.dilate(v_lines, dilate_k, iterations=1)

    skel = cv2.bitwise_or(h_lines_d, v_lines_d)
    cross = cv2.bitwise_and(h_lines_d, v_lines_d)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    skel_closed = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, close_k, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(skel_closed, connectivity=8)

    tables = []
    rejected_by_zone = 0
    table_mask = np.zeros_like(bw)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if cw < 30 or ch < 30:
            continue
        x1, y1, x2, y2 = x, y, x + cw, y + ch

        roi_h = h_lines_d[y1:y2, x1:x2]
        roi_v = v_lines_d[y1:y2, x1:x2]
        roi_cross = cross[y1:y2, x1:x2]

        row_white = (roi_h > 0).sum(axis=1)
        h_line_rows = row_white > (cw * 0.5)
        h_count = 0
        prev = False
        for v in h_line_rows:
            if v and not prev:
                h_count += 1
            prev = v

        col_white = (roi_v > 0).sum(axis=0)
        v_line_cols = col_white > (ch * 0.5)
        v_count = 0
        prev = False
        for v in v_line_cols:
            if v and not prev:
                v_count += 1
            prev = v

        num_cross, _, _, _ = cv2.connectedComponentsWithStats(roi_cross, connectivity=8)
        cross_count = max(num_cross - 1, 0)

        if strict_level == "strict":
            ok = (h_count >= 3 and v_count >= 3 and cross_count >= 4)
        else:
            ok = False

        if not ok:
            continue

        if zone is not None:
            zx1, zy1, zx2, zy2 = zone
            if not (x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2):
                rejected_by_zone += 1
                continue

        tables.append([int(x1), int(y1), int(x2), int(y2)])
        cv2.rectangle(table_mask, (x1, y1), (x2, y2), 255, -1)

    return tables, table_mask, rejected_by_zone


# ── CC 分类 ───────────────────────────────────────────────────
def classify_ccs(stats, img_h, img_w):
    classes = []
    for i in range(len(stats)):
        x, y, w, h, area = stats[i]
        if i == 0:
            classes.append("bg")
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if w > img_w * 0.7 or h > img_h * 0.7:
            classes.append("huge")
            continue
        if area < 8:
            classes.append("noise")
            continue
        bbox_area = w * h
        fill = area / max(bbox_area, 1)
        if aspect > 8 and fill < 0.15:
            classes.append("line")
            continue
        if area > 5000 and max(w, h) > 100:
            classes.append("block")
            continue
        classes.append("text")
    return classes


def build_text_mask(labels, text_indices):
    if not text_indices:
        return np.zeros_like(labels, dtype=np.uint8)
    text_idx_arr = np.array(sorted(text_indices), dtype=np.int32)
    mask = np.isin(labels, text_idx_arr).astype(np.uint8) * 255
    return mask


def rlsa_horizontal(bin_img, gap):
    out = bin_img.copy()
    h, w = out.shape
    for y in range(h):
        row = out[y]
        idx = np.where(row > 0)[0]
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        close = np.where(gaps <= gap)[0]
        for i in close:
            out[y, idx[i]:idx[i + 1] + 1] = 255
    return out


def rlsa_vertical(bin_img, gap):
    out = bin_img.copy()
    h, w = out.shape
    for x in range(w):
        col = out[:, x]
        idx = np.where(col > 0)[0]
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        close = np.where(gaps <= gap)[0]
        for i in close:
            out[idx[i]:idx[i + 1] + 1, x] = 255
    return out


# ── 文字块过滤 ─────────────────────────────────────────────────
def remove_nested(boxes, contain_thr=0.9):
    n = len(boxes)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        ax1, ay1, ax2, ay2 = boxes[i][:4]
        a_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        for j in range(n):
            if i == j or not keep[j]:
                continue
            bx1, by1, bx2, by2 = boxes[j][:4]
            b_area = max((bx2 - bx1) * (by2 - by1), 1)
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / b_area >= contain_thr and a_area > b_area:
                keep[j] = False
    return [b for k, b in zip(keep, boxes) if k]


def remove_boxes_inside_tables(boxes, tables, overlap_ratio=0.5):
    out = []
    for b in boxes:
        bx1, by1, bx2, by2 = b[:4]
        b_area = max((bx2 - bx1) * (by2 - by1), 1)
        absorbed = False
        for tb in tables:
            tx1, ty1, tx2, ty2 = tb[:4]
            ix1, iy1 = max(bx1, tx1), max(by1, ty1)
            ix2, iy2 = min(bx2, tx2), min(by2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / b_area > overlap_ratio:
                absorbed = True
                break
        if not absorbed:
            out.append(b)
    return out


def merge_overlapping(boxes, iou_thr=0.1):
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax1, ay1, ax2, ay2 = boxes[i][:4]
                bx1, by1, bx2, by2 = boxes[j][:4]
                ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                a_area = (ax2 - ax1) * (ay2 - ay1)
                b_area = (bx2 - bx1) * (by2 - by1)
                iou = inter / max(a_area + b_area - inter, 1)
                if iou > iou_thr:
                    new = [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]
                    new.append((new[2] - new[0]) * (new[3] - new[1]))
                    boxes[i] = new
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def absorb_lines(text_boxes, line_bboxes, dist_thr):
    def bbox_dist(a, b):
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        dx = max(0, max(ax1 - bx2, bx1 - ax2))
        dy = max(0, max(ay1 - by2, by1 - ay2))
        return np.hypot(dx, dy)
    text_boxes = [list(b) for b in text_boxes]
    unabsorbed = []
    for lb in line_bboxes:
        best, best_d = -1, 1e18
        for i, tb in enumerate(text_boxes):
            d = bbox_dist(lb, tb)
            if d < best_d:
                best, best_d = i, d
        if best != -1 and best_d <= dist_thr:
            tb = text_boxes[best]
            new = [min(tb[0], lb[0]), min(tb[1], lb[1]),
                   max(tb[2], lb[2]), max(tb[3], lb[3])]
            new.append((new[2] - new[0]) * (new[3] - new[1]))
            text_boxes[best] = new
        else:
            unabsorbed.append(lb)
    return text_boxes, unabsorbed


def tighten_to_content(bw, box):
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(bw.shape[1], x2); y2 = min(bw.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    sub = bw[y1:y2, x1:x2]
    rows = (sub > 0).any(axis=1)
    cols = (sub > 0).any(axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return [x1 + int(c0), y1 + int(r0), x1 + int(c1) + 1, y1 + int(r1) + 1]


def filter_edge_small_boxes(boxes, sh, sw, edge_margin=100, max_area=500):
    out = []
    dropped = 0
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        near_edge = (cx < edge_margin or cx > sw - edge_margin or
                     cy < edge_margin or cy > sh - edge_margin)
        if near_edge and area < max_area:
            dropped += 1
            continue
        out.append(b)
    return out, dropped


# ── 单变体处理 ─────────────────────────────────────────────────
def process_variant(bw, tables, table_mask, sh, sw, h_factor, v_factor=1.5):
    bw_no_table = bw.copy()
    bw_no_table[table_mask > 0] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_no_table, connectivity=8)
    classes = classify_ccs(stats, sh, sw)
    text_idx = [i for i in range(1, num) if classes[i] == "text"]
    line_idx = [i for i in range(1, num) if classes[i] == "line"]
    block_idx = [i for i in range(1, num) if classes[i] == "block"]

    if text_idx:
        text_heights = [stats[i, 3] for i in text_idx]
        median_h = int(np.median(text_heights))
    else:
        median_h = 20
    h_gap = max(int(median_h * h_factor), 12)
    v_gap = max(int(median_h * v_factor), 8)

    text_mask = build_text_mask(labels, text_idx)
    text_h = rlsa_horizontal(text_mask, h_gap)
    text_v = rlsa_vertical(text_mask, v_gap)
    text_smear = cv2.bitwise_or(text_h, text_v)

    num2, _, stats2, _ = cv2.connectedComponentsWithStats(text_smear, connectivity=8)
    text_boxes = []
    for i in range(1, num2):
        x, y, w, h, area = stats2[i]
        if area < 200:
            continue
        text_boxes.append([int(x), int(y), int(x + w), int(y + h), int(area)])

    block_boxes = []
    for i in block_idx:
        x, y, w, h, area = stats[i]
        block_boxes.append([int(x), int(y), int(x + w), int(y + h), int(area)])

    long_thr = min(sw, sh) * 0.3
    line_bboxes = []
    for i in line_idx:
        x, y, w_cc, h_cc, _ = stats[i]
        if max(w_cc, h_cc) > long_thr:
            continue
        line_bboxes.append([int(x), int(y), int(x + w_cc), int(y + h_cc)])

    all_candidates = text_boxes + block_boxes
    merged = merge_overlapping(all_candidates, iou_thr=0.1)

    line_dist = max(median_h * 2, 30)
    merged_with_lines, unabsorbed_lines = absorb_lines(merged, line_bboxes, line_dist)
    for lb in unabsorbed_lines:
        merged_with_lines.append([lb[0], lb[1], lb[2], lb[3],
                                  (lb[2] - lb[0]) * (lb[3] - lb[1])])

    merged_no_table = remove_boxes_inside_tables(merged_with_lines, tables, overlap_ratio=0.5)

    edge_filtered, _ = filter_edge_small_boxes(
        merged_no_table, sh, sw, edge_margin=100, max_area=500)

    final = remove_nested(edge_filtered, contain_thr=0.9)

    tight = []
    for b in final:
        t = tighten_to_content(bw_no_table, b)
        if t is None:
            continue
        bx1, by1, bx2, by2 = t[:4]
        bw_box = bx2 - bx1
        bh_box = by2 - by1
        area_box = bw_box * bh_box
        if min(bw_box, bh_box) < 30:
            continue
        if area_box < 2000:
            continue
        tight.append(t + [area_box])

    return tight


# ── y_boxes.csv 跨文件汇总 ─────────────────────────────────────
# 字段：source_file, token, x1, y1, x2, y2
# 仅用于人工核对/排查问题；正常流水线不读取。
_y_box_records: list[dict] = []


def clear_y_box_records() -> None:
    _y_box_records.clear()


def record_y_box(source_file: str, token: str, bbox) -> None:
    """记录一个被替换的 Y 编号框（cyan / green / orange / factory_note 通用）。

    bbox 支持 BBox 对象或 (x, y, w, h) 四元组。
    """
    if hasattr(bbox, "x") and hasattr(bbox, "y") and hasattr(bbox, "w") and hasattr(bbox, "h"):
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    else:
        x, y, w, h = bbox
    _y_box_records.append({
        "source_file": source_file or "",
        "token": token or "",
        "x1": int(x), "y1": int(y),
        "x2": int(x + w), "y2": int(y + h),
    })


def flush_y_boxes_csv(out_path: str) -> int:
    """把累计的 Y 框写入 CSV；返回记录数。空列表也会写出仅含表头的 CSV。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "source_file", "token", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(_y_box_records)
    n = len(_y_box_records)
    logger.info(f"Y 编号框坐标汇总: {out_path} ({n} 条)")
    return n


# ── 主入口 ────────────────────────────────────────────────────
STRICT_LEVELS = ["strict"]
H_FACTORS = [8]
TABLE_Y_BUFFER = 30


def detect_factory_note_codes_v6(
    image_rgb: np.ndarray,
    regions: dict,
    prefixes: list[str] = None,
    source_file: str = "",
) -> list[dict]:
    """工厂注意区域 Y 编号检测（纯像素 V6）。

    Args:
        image_rgb: 完整图纸 RGB (H, W, 3)。
        regions: detect_all_regions 已得到的字典，需含
                 material_code_column / bottom_right_number / top_left_number。
        prefixes: Y 编号首字母（默认 ["Y", "X"]）。当前 V6 不直接用 prefixes
                 做正则（沿用 config.Y_PATTERN），保留参数为接口对齐。
        source_file: 来源文件名，写入 y_boxes.csv 用。

    Returns:
        list[{"code": str, "bbox": BBox, "confidence": float}]
    """
    _ = prefixes  # 接口对齐占位
    img_h, img_w = image_rgb.shape[:2]

    red_bbox = regions.get("material_code_column")
    green_bbox = regions.get("bottom_right_number")
    orange_bbox = regions.get("top_left_number")

    fn_top = 0
    fn_bottom = green_bbox.y if green_bbox is not None else img_h
    fn_left = orange_bbox.x if orange_bbox is not None else 0
    fn_right = img_w
    if fn_bottom <= fn_top or fn_right <= fn_left:
        logger.warning(f"  Factory Note v6: search_roi 退化, "
                       f"top={fn_top} bottom={fn_bottom} left={fn_left} right={fn_right}")
        return []

    search_roi = image_rgb[fn_top:fn_bottom, fn_left:fn_right].copy()
    sh, sw = search_roi.shape[:2]

    def _to_roi(bbox):
        if bbox is None:
            return None
        x1 = max(bbox.x - fn_left, 0)
        y1 = max(bbox.y - fn_top, 0)
        x2 = min(bbox.x2 - fn_left, sw)
        y2 = min(bbox.y2 - fn_top, sh)
        if x2 <= x1 or y2 <= y1:
            return None
        return (int(x1), int(y1), int(x2), int(y2))

    ref_boxes = {
        "material_code": _to_roi(red_bbox),
        "bottom_right_number": _to_roi(green_bbox),
        "top_left_number": _to_roi(orange_bbox),
    }

    gray = cv2.cvtColor(search_roi, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bw_no_border, _, _ = erase_drawing_border(bw)
    bw = bw_no_border

    # 性能裁切：底部贯穿线以下整片丢掉
    y_cut = find_bottom_span_y(bw)
    if y_cut is not None and y_cut > 0:
        mc_roi_pre = ref_boxes.get("material_code")
        if mc_roi_pre is not None and mc_roi_pre[1] >= y_cut:
            pass
        else:
            search_roi = search_roi[:y_cut, :]
            bw = bw[:y_cut, :]
            sh, sw = bw.shape

            def _clip_ref(rb):
                if rb is None:
                    return None
                rx1, ry1, rx2, ry2 = rb
                ry2 = min(ry2, y_cut)
                if ry2 <= ry1:
                    return None
                return (rx1, ry1, rx2, ry2)
            ref_boxes = {k: _clip_ref(v) for k, v in ref_boxes.items()}

    table_zone = None
    mc_roi = ref_boxes["material_code"]
    if mc_roi is not None:
        _, mc_y1_roi, _, _ = mc_roi
        zone_out, _ = cluster_table_zone(
            bw, mc_y1=mc_y1_roi, buffer=TABLE_Y_BUFFER)
        if zone_out is not None:
            table_zone = zone_out

    exclusion_zone = None
    if table_zone is not None:
        _, _, tz_right, tz_bottom = table_zone
        exclusion_zone = (0, 0, int(tz_right), int(tz_bottom))

    # 把 exclusion_zone 在 bw 上整片涂白（255=ink→0），后续 OCR 不在该区域内出框
    bw_for_ocr = bw
    if exclusion_zone is not None:
        bw_for_ocr = bw.copy()
        ex1, ey1, ex2, ey2 = exclusion_zone
        pad = 3
        bh, bw_w = bw_for_ocr.shape
        ex1 = max(0, ex1 - pad)
        ey1 = max(0, ey1 - pad)
        ex2 = min(bw_w, ex2 + pad)
        ey2 = min(bh, ey2 + pad)
        bw_for_ocr[ey1:ey2, ex1:ex2] = 0

    found_codes: list[dict] = []

    for strict in STRICT_LEVELS:
        tables, table_mask, _ = detect_tables(bw, strict_level=strict, zone=table_zone)

        for h_factor in H_FACTORS:
            tight = process_variant(bw_for_ocr, tables, table_mask, sh, sw, h_factor)

            def _inside_ex(box):
                if exclusion_zone is None:
                    return False
                bx1, by1, bx2, by2 = box[:4]
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                return (exclusion_zone[0] <= cx <= exclusion_zone[2]
                        and exclusion_zone[1] <= cy <= exclusion_zone[3])

            crop_targets = []
            for b in tight:
                crop_targets.append(("text", b[:4]))
            for tb in tables:
                if _inside_ex(tb):
                    continue
                crop_targets.append(("table", tb[:4]))

            seen_global_polys: list[tuple[tuple[int, int, int, int], str]] = []
            v5_total = 0
            v5_kept = 0

            for idx, (kind, box) in enumerate(crop_targets):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(sw, x2); y2 = min(sh, y2)
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                crop = search_roi[y1:y2, x1:x2]
                pil_crop = Image.fromarray(crop)

                _, items = _v5_run(pil_crop)
                v5_total += 1

                v5_pass, v5_hits, _ = _v5_filter_y(items)
                if not v5_pass:
                    continue

                # polygon 级去重：相同 token + IoU≥0.5 视为同一物理位置
                crop_y_polys: list[tuple[tuple[int, int, int, int], str]] = []
                for poly, text, _sc in items:
                    if poly is None or len(poly) == 0:
                        continue
                    tok = _find_y_token(text)
                    if not tok:
                        continue
                    px1, py1, px2, py2 = _poly_bbox(poly)
                    gbox = (px1 + x1, py1 + y1, px2 + x1, py2 + y1)
                    crop_y_polys.append((gbox, tok))
                new_polys: list[tuple[tuple[int, int, int, int], str]] = []
                for gbox, tok in crop_y_polys:
                    is_dup = False
                    for sbox, stok in seen_global_polys:
                        if stok != tok:
                            continue
                        if _bbox_iou(gbox, sbox) >= 0.5:
                            is_dup = True
                            break
                    if not is_dup:
                        new_polys.append((gbox, tok))
                if crop_y_polys and not new_polys:
                    continue
                seen_global_polys.extend(new_polys)
                v5_kept += 1

                vlm_matched = _vlm_annotate_single(pil_crop)
                for _poly, _tk in vlm_matched:
                    xs = [p[0] for p in _poly]
                    ys = [p[1] for p in _poly]
                    bx1 = int(min(xs)) + x1 + fn_left
                    by1 = int(min(ys)) + y1 + fn_top
                    bx2 = int(max(xs)) + x1 + fn_left
                    by2 = int(max(ys)) + y1 + fn_top
                    w_px = max(bx2 - bx1, 1)
                    h_px = max(by2 - by1, 1)
                    found_codes.append({
                        "code": _tk,
                        "bbox": BBox(bx1, by1, w_px, h_px),
                        "confidence": 1.0,
                    })

            logger.info(
                f"  Factory Note v6 [{strict}|h={h_factor}]: "
                f"tables={len(tables)} boxes={len(tight)} "
                f"v5_run={v5_total}/{len(crop_targets)} "
                f"v5_y_hit={v5_kept} vlm_drawn={len(found_codes)}")

    return found_codes
