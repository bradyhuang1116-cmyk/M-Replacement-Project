"""工厂注意区域 纯像素分块 V6 — 基于 V3 strict，加位置 prior 和小修。

完全抛弃 V5。基线沿用 V3：
  - 形态学表格抽取 min_line_len_ratio=0.015
  - close_size=15
  - strict 判据: h_count >= 3 and v_count >= 3 and cross_count >= 4
  - h_count/v_count 用 50% 贯穿率（V3 原方式，不用 V5 的 CC 计数）

V4 保留两项:
  - erase_drawing_border: 擦除图纸外框
  - 长线丢弃: line CC max(w, h) > min(sw, sh)*0.3 直接丢弃

V6 新增:
  - 位置 prior（矩形 zone, 横/竖线聚类）:
    搜索矩形 = (0, 0, sw, MC.y2 + buffer)
    Step1: 在矩形内抽 h_lines (min_len=40), 按长度过滤 (≥ max_len × 0.6),
           按 y 找"间距相近"的最长连续段 → 表格的行集合
    Step2: 在矩形内抽 v_lines, 同样长度过滤 + 间距聚类 → 表格的列集合
    Step3: 行/列 bbox 取并集 + 外扩 buffer = 表格 zone
           表格 CC 必须完全落在 zone 内才保留
    聚类失败 / 无 material_code → 退回 V4 strict 全图行为
  - 边缘小点过滤: box 中心距搜索区任一边 < 100px 且面积 < 500 → 丢弃
  - 大框套小框放宽: B 的 ≥ 90% 在 A 内即视为嵌套（不要求 100% 包含）

变体: strict_level="strict" × h_factor ∈ {8, 15} = 2 个变体
"""
import os
import sys
import json
import csv
import gc
import logging
import shutil
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, Y_PATTERN
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, _enhance_vertical_lines

_default_dir = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
KEYWORDS = ["P3335", "A226", "C857", "A110-2"]

INPUT_FILES = []
for f in sorted(os.listdir(_default_dir)):
    if os.path.splitext(f)[1].lower() not in (".tif", ".tiff"):
        continue
    for kw in KEYWORDS:
        if kw in f:
            INPUT_FILES.append(os.path.join(_default_dir, f))
            break

if not INPUT_FILES:
    logger.error(f"未找到匹配文件: {KEYWORDS}")
    sys.exit(1)

logger.info(f"测试文件({len(INPUT_FILES)}): {[os.path.basename(f) for f in INPUT_FILES]}")

DEBUG_BASE = os.path.join(os.path.dirname(__file__), "test_output", "factory_note_test_pixel_v6")
# 每次运行前清空输出目录，避免上次的过期文件混淆当前结果
if os.path.isdir(DEBUG_BASE):
    shutil.rmtree(DEBUG_BASE)
os.makedirs(DEBUG_BASE, exist_ok=True)
prefixes = DEFAULT_PREFIXES

from modules.docker_manager import ensure_vlm_ready
logger.info("启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 失败: {msg}")
    sys.exit(1)

from modules.vlm_ocr_engine import get_vlm_engine
from modules.region_detector import _get_ocr_v5, _parse_ocr_results_common
_vlm = get_vlm_engine()
_v5 = _get_ocr_v5("en")
_FN_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename_tag(text: str, max_len: int = 32) -> str:
    """把 OCR 文本压成文件名安全片段。"""
    cleaned = _FN_SAFE_RE.sub("_", (text or "").strip())[:max_len].strip("_.")
    return cleaned or "hit"


_Y_RE = re.compile(Y_PATTERN)              # 全局 Y_PATTERN（≥6 位需含数字、无 \b）
_V_RE = re.compile(r"V(?=[A-Z0-9]*\d)[A-Z0-9]{6,}")  # 同形态、首字母 V：V5 把 Y 误识为 V 时回填


def _find_y_token(text: str) -> str | None:
    """在 OCR 文本里搜符合 Y_PATTERN 的 token。
      - 优先全局 Y_PATTERN（Y/X 开头 + 6 位含数字）
      - 兜底：相同形态但首字母为 V（V5 把 Y 误识为 V），命中后回填为 Y
    """
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


def _v5_run(pil_crop):
    """V5 OCR 一次。返回 (np_img, items)；items 形如 [(poly, text, score), ...]。"""
    np_img = np.array(pil_crop)
    try:
        result = _v5.predict(np_img)
    except Exception as e:
        logger.warning(f"  V5 OCR 异常: {e}")
        return np_img, []
    items = _parse_ocr_results_common(result)
    return np_img, items


def _v5_filter_y(items):
    """V5 粗扫：polygon 文本必须包含 Y 编号（_find_y_token 命中，含 V→Y 误识修正）才算命中。
    返回 (是否命中, 命中 Y 编号 token 列表, 所有文本)。"""
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


def _v5_annotate_all(np_img, items):
    """在 np_img 上画出 V5 所有 polygon（绿框） + 标文本。"""
    annotated = np_img.copy()
    for poly, text, _ in items:
        if poly is None or len(poly) == 0:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        x_min = int(np.min(pts[:, 0, 0]))
        y_min = int(np.min(pts[:, 0, 1]))
        label = (text or "").strip()
        if label:
            cv2.putText(annotated, label, (x_min, max(y_min - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return annotated


def _poly_bbox(poly):
    """polygon → (x_min, y_min, x_max, y_max)。"""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a, b):
    """两个 (x1,y1,x2,y2) 的 IoU。空相交返回 0。"""
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
    """对二值列方向做投影，返回 [(c_start, c_end_exclusive), ...] 的"墨水列簇"。"""
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


def _localize_y_box(crop_np, poly, target_tok):
    """在 polygon 包围盒内精确定位 target_tok 这几个字符的红框。

    全程只用 VLM 二次 OCR（按目标要求）。流程：

    1. tok 候选生成：target_tok 本身；若 target_tok 长度 > 7 且 tok[:-1] 仍匹配
       Y_PATTERN，则把 tok[:-1] 也加入候选（修 VLM 把相邻字符当 tok 末位的误读）。
    2. 对每个 tok 候选跑 Strategy A/B/Phase4，返回各自的最紧窗口（含失败标志）。
    3. 选窗口更窄、且 final_text == tok 的；都失败时退到 _fallback_box。
    4. 最终对窗口做"列簇边界对齐"：左/右 snap 到最近 _column_runs 的 start/end，
       避免切在墨水中间（修 text_010 左边 Y 只露一半）。

    返回 (sub_poly[4 顶点], 实际使用的 tok)；可能返回 tok 的短版本。"""
    if not target_tok:
        return None
    x1, y1, x2, y2 = _poly_bbox(poly)
    H, W = crop_np.shape[:2]
    # 按 1 个字符宽外扩搜索区域：VLM 给的 polygon 常把首字符切半（如 Y 只露半边），
    # 这里给左/右各加一格 padding，让后续列簇 snap 有机会向外抓回完整 ink。
    char_h = max(8, y2 - y1)
    pad = max(int(char_h * 0.6), 12)
    x1 = max(0, x1 - pad); y1 = max(0, y1)
    x2 = min(W, x2 + pad); y2 = min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 6:
        return None

    sub = crop_np[y1:y2, x1:x2]

    _norm_table = str.maketrans({'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8'})

    def _norm(s: str) -> str:
        return (s or "").upper().replace(" ", "").translate(_norm_table)

    # tok 候选：原 tok + 末位回退（仅当回退后仍是合法 Y_PATTERN）
    tok_candidates = [target_tok.upper()]
    short = target_tok.upper()[:-1]
    if len(short) >= 7 and _Y_RE.fullmatch(short):
        tok_candidates.append(short)

    # 共享：列簇 + 网格切点
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
        """把 [left_col, right_col] 收紧到该窗口内"所有列簇组成的 ink 区间"。

        - 收集所有与 [left_col, right_col] 有交集的 runs；
        - left → 最左 run 的 start（向左外扩到墨水开端，避免切在 Y 半腰）；
        - right → 最右 run 的 end（向右外扩到墨水结尾）；
        - 若 [left_col, right_col] 窗口内没有任何 run（极端），保持原值不动。
        - 安全网：每边最多外扩 char_w（≈ 一格字符宽），避免吞掉相邻字符。"""
        overlapping = [(rs, re_) for rs, re_ in runs
                       if re_ > left_col and rs < right_col]
        if not overlapping:
            return left_col, right_col
        new_left = min(rs for rs, _ in overlapping)
        new_right = max(re_ for _, re_ in overlapping)
        char_w = max(8, (right_col - left_col) // 12)  # ≈ 1 char 宽
        new_left = max(new_left, left_col - char_w)
        new_right = min(new_right, right_col + char_w)
        return new_left, new_right

    def _run_for_tok(tok_up: str):
        """对一个 tok 候选跑全套定位。返回 (status, left_col, right_col, final_text)。
        status: 'exact' / 'phase4' / 'contains' / 'fail'
        坐标在 sub 坐标系。"""
        alt_tok = "V" + tok_up[1:] if tok_up.startswith("Y") else None
        tok_n = _norm(tok_up)
        alt_n = _norm(alt_tok) if alt_tok else None

        def _is_tok_exact(text_norm: str) -> bool:
            return text_norm == tok_n or (alt_n is not None and text_norm == alt_n)

        def _contains_tok(text_norm):
            if not text_norm:
                return False
            return tok_n in text_norm or (alt_n is not None and alt_n in text_norm)

        # ── 策略 A：upscale + VLM 重检测，挑文本恰好 == tok 的 polygon ──
        UPSCALE = 2.5
        try:
            sub_up = cv2.resize(sub, None, fx=UPSCALE, fy=UPSCALE,
                                interpolation=cv2.INTER_CUBIC)
            res_up = _vlm.predict(sub_up)
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

        # ── 策略 B 准备 ──
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
                res = _vlm.predict(s)
                its = _parse_ocr_results_common(res)
            except Exception:
                its = []
            text = _norm("".join((t or "") for _, t, _ in its))
            _cache[key] = text
            return text

        if not _contains_tok(_vlm_text_norm(starts_set[0], ends_set[-1])):
            return 'fail', 0, x2 - x1, ''

        # Phase 1
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

        # Phase 2
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

        # Phase 3
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

        # Phase 4: 窄窗 upscale 再切
        try:
            sub_narrow = sub[:, left_col:right_col]
            sub_narrow_up = cv2.resize(sub_narrow, None, fx=UPSCALE, fy=UPSCALE,
                                       interpolation=cv2.INTER_CUBIC)
            res_n = _vlm.predict(sub_narrow_up)
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

    # ── 对每个 tok 候选跑一次，挑最好的 ──
    # 策略：优先原 target_tok（最长、最忠实于上游识别）；只有当原 tok 无法 exact
    # 命中（说明 VLM 把相邻字符误读进 tok）时，才用 tok[:-1] 兜底。
    # 这样：text_010(YA057C800 exact) 用原 tok；text_046(Y3357998 上游误读，VLM
    # 重检测会拿不到 exact) 自动滑到 rollback。
    results = []  # [(priority, width, status, left, right, tok_used)]
    status_priority = {'exact': 0, 'phase4': 1, 'contains': 2, 'fail': 3}
    original_status = None
    for idx, tok_cand in enumerate(tok_candidates):
        status, lc, rc, tk = _run_for_tok(tok_cand)
        if idx == 0:
            original_status = status
            results.append((status_priority[status], rc - lc, status, lc, rc, tk or tok_cand))
            if status == 'exact':
                # 原 tok 已 exact，无需再跑 rollback
                break
        else:
            # rollback 只在原 tok 未 exact 时考虑；偏好 rollback 也 exact 的
            results.append((status_priority[status], rc - lc, status, lc, rc, tk or tok_cand))

    # 优先级：精确 > phase4 > contains > fail；同级选窗口更窄的
    results.sort(key=lambda r: (r[0], r[1]))
    _, _, status, lc, rc, tok_used = results[0]

    # 列簇边界对齐（修像素级溢出，如左边 Y 只露一半）
    lc_snap, rc_snap = _snap_to_runs(lc, rc)
    return _fallback_box(lc_snap, rc_snap, tok_used)


def _vlm_annotate_single(pil_crop):
    """VLM 二次识别（V5 已确认 crop 含 Y 编号）：
      1. 在整张 pil_crop 上跑 VLM，得到所有 polygon+text；
      2. 对文本含 Y 的 polygon，调 _localize_y_box 用列簇 + 二分 VLM 把红框收紧到只覆盖 Y；
      3. 定位失败的 polygon → 退回 VLM 原 polygon 画框（必出框，不留空）。
    返回 (标注后 numpy 图, [(sub_poly, token), ...])。"""
    np_img = np.array(pil_crop)
    try:
        result = _vlm.predict(np_img)
    except Exception as e:
        logger.warning(f"  VLM predict 异常: {e}")
        return np_img.copy(), []
    items = _parse_ocr_results_common(result)

    matched = []
    for poly, text, _ in items:
        if poly is None or len(poly) == 0:
            continue
        tok = _find_y_token(text)
        if not tok:
            continue
        # 即使 VLM 文本恰好 == tok，也跑 _localize_y_box，保证列簇 snap 生效
        # （否则 VLM 原 polygon 常切到 Y 半边或越界到下一字符）
        refined = _localize_y_box(np_img, poly, tok)
        if refined is not None:
            sub_poly, sub_tok = refined
            matched.append((sub_poly, sub_tok))
            continue
        # _localize_y_box 兜底已尽力，仍 None 说明 polygon 不可用（太小/退化）；
        # 此时退回 VLM 原 polygon —— 至少有一个框（必出框）。
        logger.info(f"  VLM 二次定位极端兜底：用 VLM 原 polygon 画框，tok={tok}, text='{text}'")
        matched.append(([(int(p[0]), int(p[1])) for p in poly], tok))

    annotated = np_img.copy()
    for sub_poly, tok in matched:
        pts = np.array(sub_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
        x_min = int(np.min(pts[:, 0, 0]))
        y_min = int(np.min(pts[:, 0, 1]))
        cv2.putText(annotated, tok, (x_min, max(y_min - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return annotated, matched


# ────────────────────────────────────────────────────────────
# V4 保留: 擦除搜索区图纸外框
# ────────────────────────────────────────────────────────────
def erase_drawing_border(bw, h_ratio=0.95, v_ratio=0.95,
                         edge_band=0.05, dilate_px=6):
    """只擦图纸最外画框的"长线像素"，不擦 CC（避免与表格相连时连带误擦）。

    判定：
      - 横线像素位于 y < h*edge_band 或 y > h*(1-edge_band)，且
        所在横线长度 ≥ w * h_ratio
      - 竖线像素位于 x < w*edge_band 或 x > w*(1-edge_band)，且
        所在竖线长度 ≥ h * v_ratio
      - 满足条件的线 mask 用 dilate_px 膨胀后从 bw 中擦掉

    这样：
      - 表格的横线即使很长，也不会贴在 search_roi 顶/底 5% 内（因为表格上下都有
        其它内容），不会被擦
      - 表格的竖线即使很长，也不会贴在 search_roi 左/右 5% 内
      - 即使大画框 CC 与表格相连，只有"画框那部分线像素"被抹，表格主体保留
    """
    h, w = bw.shape
    h_min_len = int(w * h_ratio)
    v_min_len = int(h * v_ratio)

    border_mask = np.zeros_like(bw)
    line_segs = []  # 用于诊断

    # 横线
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

    # 竖线
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


# ────────────────────────────────────────────────────────────
# V6 表格 zone 检测：横/竖线聚类
# ────────────────────────────────────────────────────────────
def _longest_similar_run(items, key, gap_lo=0.5, gap_hi=2.0):
    """按 key 排序后，找"相邻间距相近"的最长连续段。

    items: list of records; key: callable returning sort key (numeric)
    返回 items 的子列表（已按 key 排序），长度 >= 1。
    """
    if len(items) < 2:
        return list(items)
    items = sorted(items, key=key)
    keys = [key(it) for it in items]
    gaps = [keys[i + 1] - keys[i] for i in range(len(keys) - 1)]
    if not gaps:
        return items
    med_gap = sorted(gaps)[len(gaps) // 2]
    if med_gap <= 0:
        return items
    lo, hi = gap_lo * med_gap, gap_hi * med_gap
    runs = [[0]]
    for i, g in enumerate(gaps):
        if lo <= g <= hi:
            runs[-1].append(i + 1)
        else:
            runs.append([i + 1])
    best = max(runs, key=len)
    return [items[k] for k in best]


def _length_cluster(segs, length_idx=3, tol=0.3, min_n=3):
    """按长度做密度聚类，返回长度最相似的最大子集。

    segs: list of records; length_idx: 长度在 record 中的位置
    tol: 相对容差 (默认 ±30%)
    min_n: 子集最小元素数
    """
    if len(segs) < min_n:
        return list(segs)
    lengths = sorted([s[length_idx] for s in segs])
    # 每个 anchor 长度 L 看落在 [L*(1-tol), L*(1+tol)] 的元素数
    best_L = None
    best_count = 0
    best_set = []
    for L in lengths:
        lo, hi = L * (1 - tol), L * (1 + tol)
        sub = [s for s in segs if lo <= s[length_idx] <= hi]
        if len(sub) > best_count:
            best_count = len(sub)
            best_L = L
            best_set = sub
    return best_set


def _write_cluster_debug(debug_out, bw, sx1, sy1, sx2, sy2,
                         h_lines, v_lines, cluster_h, cluster_v, zone,
                         exclusion_zone=None):
    """渲染 cluster 调试图：成功/失败路径共用。

    h_lines / v_lines 是 search_rect 内 ROI 大小的二值图；
    cluster_h / cluster_v 是 ROI 坐标系下的线段列表；
    zone 是 bw 坐标系的 (zx1,zy1,zx2,zy2)，失败路径传 None；
    exclusion_zone=(0,0,zone_right,zone_bottom)：OCR 排除大区域，半透明紫色填充。
    """
    if debug_out is None or not SAVE_DEBUG_ARTIFACTS:
        return
    dbg = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    if h_lines is not None and h_lines.size > 0:
        h_full = np.zeros_like(bw)
        h_full[sy1:sy2, sx1:sx2] = h_lines
        dbg[h_full > 0] = (180, 60, 60)
    if v_lines is not None and v_lines.size > 0:
        v_full = np.zeros_like(bw)
        v_full[sy1:sy2, sx1:sx2] = v_lines
        dbg[v_full > 0] = (60, 60, 180)
    for s in cluster_h:
        yc, xl, xr, _ = s
        cv2.line(dbg, (xl + sx1, yc + sy1), (xr + sx1, yc + sy1), (255, 120, 120), 2)
    for s in cluster_v:
        xc, yt, yb, _ = s
        cv2.line(dbg, (xc + sx1, yt + sy1), (xc + sx1, yb + sy1), (120, 120, 255), 2)
    cv2.rectangle(dbg, (sx1, sy1), (sx2, sy2), (140, 140, 140), 2)
    _draw_exclusion_overlay(dbg, exclusion_zone)
    if zone is not None:
        zx1, zy1, zx2, zy2 = zone
        cv2.rectangle(dbg, (zx1, zy1), (zx2, zy2), (0, 220, 220), 4)
    Image.fromarray(dbg).save(debug_out, quality=88)


def cluster_table_zone(bw, mc_y1,
                       min_h_len_ratio=0.05, h_xrange_tol_ratio=0.02,
                       min_v_len_ratio=0.03, v_yrange_tol_ratio=0.03,
                       right_gap_ratio=1.8, buffer=20, debug_out=None):
    """按以下规则定位工厂注意表格 zone：

      zone.left   = 0
      zone.top    = mc_y1（MC 上边界）
      zone.bottom = "两端对齐 + 长度近似"的横线集合中最下面那条的 y
      zone.right  = "等高竖线"集合中"右侧空隙巨大"的那条竖线的 x

    横线判定（行分隔线）：
      - 在 (0, mc_y1, sw, sh) 内抽足够长的横线
      - 对每根横线作 anchor，找 x1, x2 都落在 anchor.x1±tol / anchor.x2±tol 的子集
      - 取最大子集 → 同一张表格的行分隔线
      - 最下面那条 y 即 zone.bottom

    竖线判定（右边界）：
      - 在 (0, mc_y1, sw, sh) 内抽足够长的竖线
      - 对每根竖线作 anchor，找 y1, y2 都落在 anchor.y1±tol / anchor.y2±tol 的子集
      - 取最大子集 → 等高竖线（同一张表格的列分隔线）
      - 按 x 排序，找"右侧到下一条等高竖线（或到 sw）距离 ≥ right_gap_ratio×列间距中位数"的最右那条
        → 该竖线的 x 即 zone.right

    输入：
      bw: 二值图
      mc_y1: MC 上边界（bw 坐标）
    输出：
      ((x1, y1, x2, y2), info) 或 (None, error_dict)。坐标都在 bw 坐标系。
    """
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

    # ── 横线 ───────────────────────────────────────────
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel, iterations=1)
    num_h, _, stats_h, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    h_segs = []  # list of dict
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
        _write_cluster_debug(debug_out, bw, sx1, sy1, sx2, sy2,
                             h_lines, None, [], [], None)
        return None, {"error": f"only {len(h_segs)} h_segs (need ≥2)",
                       "h_segs_total": int(num_h - 1)}

    # 两端对齐聚类：对每根横线作 anchor，找 x1/x2 都接近 anchor 的子集；
    # 选 length(median) 最大的子集（要求 size ≥ 2）。
    # 理由：MC 列内部行数多但长度短；工厂注意主表行数少但长度接近 search 宽度。
    best_h_set = []
    best_h_med_len = -1
    h_candidates_log = []
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
        h_candidates_log.append({
            "anchor_x1": anchor["x1"], "anchor_x2": anchor["x2"],
            "size": len(sub), "med_len": int(med_len),
        })

    if len(best_h_set) < 2:
        _write_cluster_debug(debug_out, bw, sx1, sy1, sx2, sy2,
                             h_lines, None, [], [], None)
        return None, {"error": f"no h aligned-end set (size≥2) found",
                       "h_segs_total": int(num_h - 1),
                       "h_segs_long": len(h_segs)}

    y_bottom_roi = max(s["y"] for s in best_h_set)
    h_x1_med = int(np.median([s["x1"] for s in best_h_set]))
    h_x2_med = int(np.median([s["x2"] for s in best_h_set]))

    # ── 竖线 ───────────────────────────────────────────
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

    # 等高竖线聚类（长度优先：选 median length 最大且 size ≥ 2 的子集）
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

    # ── 右边界选取 ─────────────────────────────────────
    # 新规则：右边界 X 必须"落在表格内的横线上"——即 X 与 best_h_set 横线右端 median
    # (h_x2_med) 接近（差 ≤ x_tol_right）。优先选 best_v_set（等高竖线）中满足该约束
    # 且最右的那条；若没有任何等高竖线落在横线右端附近，则直接用 h_x2_med（横线右端）。
    x_tol_right = max(int(rw * 0.02), 20)
    x_right_roi = None
    right_gap_detail = None

    v_xs_sorted_all = sorted({v["x"] for v in best_v_set}) if best_v_set else []
    v_near_h_right = [v["x"] for v in best_v_set
                      if abs(v["x"] - h_x2_med) <= x_tol_right] if best_v_set else []

    if v_near_h_right:
        x_right_roi = max(v_near_h_right)
        right_gap_detail = {
            "strategy": "v_eq_near_h_x2",
            "h_x2_med_roi": int(h_x2_med),
            "x_tol": int(x_tol_right),
            "v_xs_all": [int(x) for x in v_xs_sorted_all],
            "v_near_h_right": [int(x) for x in sorted(v_near_h_right)],
            "picked_x_roi": int(x_right_roi),
        }
    else:
        x_right_roi = h_x2_med
        right_gap_detail = {
            "strategy": "fallback_h_x2_med",
            "h_x2_med_roi": int(h_x2_med),
            "x_tol": int(x_tol_right),
            "v_xs_all": [int(x) for x in v_xs_sorted_all],
            "v_near_h_right": [],
            "picked_x_roi": int(h_x2_med),
        }

    # ── 全图坐标 ──────────────────────────────
    # 严格落在线上：zone.bottom = 横线 y, zone.right = 竖线 x；
    # buffer 只对 zone.top 起作用（往上预留一点保护 MC 顶部线条）。
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
        "right_gap_detail": right_gap_detail,
        "x_right_chosen": int(x_right_roi + sx1),
    }

    # 重新组装 cluster_h / cluster_v 给调试图用（兼容旧 helper 元组格式）
    cluster_h = [(s["y"], s["x1"], s["x2"], s["len"]) for s in best_h_set]
    cluster_v = [(s["x"], s["y1"], s["y2"], s["len"]) for s in best_v_set]
    _excl = (0, 0, int(zone_right), int(zone_bottom))
    _write_cluster_debug(debug_out, bw, sx1, sy1, sx2, sy2,
                         h_lines, v_lines, cluster_h, cluster_v,
                         (int(zone_left), int(zone_top),
                          int(zone_right), int(zone_bottom)),
                         exclusion_zone=_excl)

    info["exclusion_zone"] = _excl
    return (int(zone_left), int(zone_top),
            int(zone_right), int(zone_bottom)), info


# ────────────────────────────────────────────────────────────
# 多表格全图扫描
# ────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _segment_in_zone(x1, y1, x2, y2, zones, tol=4):
    """线段几何中点 ± tol 是否落在任一 zone 内（用于排除已知主表内的线）。"""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    for zx1, zy1, zx2, zy2 in zones:
        if zx1 - tol <= cx <= zx2 + tol and zy1 - tol <= cy <= zy2 + tol:
            return True
    return False


def _extract_lines_hough(bw, axis, min_len,
                         max_thickness=2, gap=10,
                         merge_dpos=4, merge_iou=0.5):
    """用 HoughLinesP 提取轴对齐的线段，并对同一物理线在多次响应中合并。

    morphology+connectedComponents 在"多条平行细线挤在一起 + 它们之间的间隙也连成 255"
    时会把整片当作一个 24px 高的大连通域（钢丝表 y=302 顶边就是这样被埋没的）。
    Hough 直接抽线段，再按 (Δpos ≤ merge_dpos 且 x/y 范围 IoU ≥ merge_iou) 合并，
    每条物理线最终输出一条记录。

    返回 list of dict：
      axis='h' → {y, x1, x2, len}
      axis='v' → {x, y1, y2, len}
    """
    lines = cv2.HoughLinesP(bw, rho=1, theta=np.pi / 180,
                            threshold=max(int(min_len * 0.4), 50),
                            minLineLength=min_len,
                            maxLineGap=gap)
    if lines is None:
        return []
    raw = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx, dy = x2 - x1, y2 - y1
        length = int(np.hypot(dx, dy))
        if axis == 'h' and abs(dy) <= max_thickness and length >= min_len:
            raw.append([int(min(y1, y2)), int(min(x1, x2)), int(max(x1, x2))])
        elif axis == 'v' and abs(dx) <= max_thickness and length >= min_len:
            raw.append([int(min(x1, x2)), int(min(y1, y2)), int(max(y1, y2))])
    if not raw:
        return []
    raw.sort()
    # 迭代合并：|Δpos| ≤ merge_dpos 且 范围 IoU ≥ merge_iou
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(raw)
        for i in range(len(raw)):
            if used[i]:
                continue
            cur = raw[i][:]
            used[i] = True
            for j in range(i + 1, len(raw)):
                if used[j]:
                    continue
                if abs(raw[j][0] - cur[0]) > merge_dpos:
                    continue
                a1, a2 = cur[1], cur[2]
                b1, b2 = raw[j][1], raw[j][2]
                inter = max(0, min(a2, b2) - max(a1, b1))
                union = max(a2, b2) - min(a1, b1)
                if union <= 0 or inter / union < merge_iou:
                    continue
                cur[0] = (cur[0] + raw[j][0]) // 2
                cur[1] = min(cur[1], raw[j][1])
                cur[2] = max(cur[2], raw[j][2])
                used[j] = True
                changed = True
            out.append(cur)
        raw = out
    res = []
    for pos, p1, p2 in raw:
        if axis == 'h':
            res.append({"y": int(pos), "x1": int(p1), "x2": int(p2),
                        "len": int(p2 - p1)})
        else:
            res.append({"x": int(pos), "y1": int(p1), "y2": int(p2),
                        "len": int(p2 - p1)})
    return res


def scan_all_tables(bw, mask_zones=None,
                    min_h_len_ratio=0.04,
                    min_v_len_ratio=0.04,
                    x_tol_ratio=0.015,
                    y_tol_ratio=0.015,
                    end_match_ratio=0.025,
                    min_area_ratio=0.003,
                    iou_dedupe_thr=0.35,
                    min_side_ratio=0.04,
                    v_cover_ratio=0.6,
                    debug_out=None):
    """全图扫描候选表格。

    候选定义：存在一组"两端对齐 + 等长的横线"作为表格的多条行分隔线，
    再在全部竖线中找到两根边竖线（一根 x≈h.x1_med，一根 x≈h.x2_med，
    且各自 y 范围基本覆盖 h-cluster 的 y 范围）→ 候选表格 zone。

    关键改动 vs 旧版：不要求左右边竖线"等高"——大表内部分栏会让左右外框竖线
    被打断成不等长的多段，等高约束会把这种表全部漏检。

    算法：
      1. 全图提长横线、长竖线 (min_*_len_ratio 控制最短长度比例)
      2. mask_zones 内的线段排除（避免重复检出主表）
      3. 横线两端对齐聚类：anchor 的 x1/x2 ± x_tol 内的子集，size ≥ 2
      4. 对每个 h-cluster，在 v_segs 中找：
          left_v   = x 接近 h.x1_med，y 范围 ⊇ [h.y_min, h.y_max] 的最近一根
          right_v  = x 接近 h.x2_med，y 范围 ⊇ [h.y_min, h.y_max] 的最近一根
         覆盖判定用 v_cover_ratio（默认 60%，避免被打断的竖线全砍掉）
      5. 候选 zone = (left_v.x, min(left_v.y1, right_v.y1, h.y_min),
                       right_v.x, max(left_v.y2, right_v.y2, h.y_max))
      6. 过滤：面积 / 边长比 / 与 mask_zones 的 IoU
      7. 候选互相去重：IoU > iou_dedupe_thr → 保留面积大者

    输入：
      bw: 二值图 (search_roi 大小)
      mask_zones: 已知主表 zones 列表（bw 坐标系），扫描时跳过其内部
    输出：
      (candidates, info)
      candidates: list of (x1, y1, x2, y2) bw 坐标系矩形
      info: 各阶段统计 dict（含每个候选的 metadata）
    """
    H, W = bw.shape
    if mask_zones is None:
        mask_zones = []

    h_min_len = max(int(W * min_h_len_ratio), 40)
    v_min_len = max(int(H * min_v_len_ratio), 40)
    x_tol = max(int(W * x_tol_ratio), 10)
    y_tol = max(int(H * y_tol_ratio), 10)
    end_match = max(int(W * end_match_ratio), 15)
    min_area = max(int(W * H * min_area_ratio), 50 * 50)
    min_side_w = int(W * min_side_ratio)
    min_side_h = int(H * min_side_ratio)

    # 横线（Hough + 合并；解决 morphology+CC 把"挨在一起的多条平行线"误并成
    # 一个高块的问题，例如 P3335 钢丝表顶边 y=302 在 morphology 下被合并进
    # 跨整张图宽的 cc#6 而漏检）
    h_lines_raw = _extract_lines_hough(bw, axis='h', min_len=h_min_len)
    h_segs = []
    for ln in h_lines_raw:
        if _segment_in_zone(ln["x1"], ln["y"], ln["x2"], ln["y"], mask_zones):
            continue
        h_segs.append(ln)

    # 竖线
    v_lines_raw = _extract_lines_hough(bw, axis='v', min_len=v_min_len)
    v_segs = []
    for ln in v_lines_raw:
        if _segment_in_zone(ln["x"], ln["y1"], ln["x"], ln["y2"], mask_zones):
            continue
        v_segs.append(ln)

    # 同 x 拼接：表格的"外框竖线"常被表内横线在交点处打断成多段。
    # 把同一列附近 (Δx ≤ x_merge_tol) 且**相邻段 y 间隙较小** (≤ y_gap_max) 的
    # 多条短竖线合并成一条"虚拟长竖线"，避免不同表格的同 x 竖线被错合。
    # 原 v_segs 保留不变（其他逻辑仍用真实段）。
    def _merge_v_by_x(segs, x_merge_tol, y_gap_max):
        if not segs:
            return []
        items = sorted(segs, key=lambda s: (s["x"], s["y1"]))
        # 先按 x 切分桶
        buckets = []
        cur = [items[0]]
        for s in items[1:]:
            ref_x = int(np.median([c["x"] for c in cur]))
            if abs(s["x"] - ref_x) <= x_merge_tol:
                cur.append(s)
            else:
                buckets.append(cur)
                cur = [s]
        buckets.append(cur)
        merged = []
        for bucket in buckets:
            bucket = sorted(bucket, key=lambda s: s["y1"])
            chain = [bucket[0]]
            for s in bucket[1:]:
                prev_y2 = max(c["y2"] for c in chain)
                if s["y1"] - prev_y2 <= y_gap_max:
                    chain.append(s)
                else:
                    x_mean = int(np.mean([c["x"] for c in chain]))
                    y1 = min(c["y1"] for c in chain)
                    y2 = max(c["y2"] for c in chain)
                    cov = sum(c["len"] for c in chain)
                    merged.append({"x": x_mean, "y1": y1, "y2": y2,
                                    "len": int(y2 - y1), "coverage": int(cov),
                                    "parts": len(chain)})
                    chain = [s]
            if chain:
                x_mean = int(np.mean([c["x"] for c in chain]))
                y1 = min(c["y1"] for c in chain)
                y2 = max(c["y2"] for c in chain)
                cov = sum(c["len"] for c in chain)
                merged.append({"x": x_mean, "y1": y1, "y2": y2,
                                "len": int(y2 - y1), "coverage": int(cov),
                                "parts": len(chain)})
        return merged

    v_merge_y_gap = max(int(v_min_len * 0.6), int(H * 0.04))
    v_merged = _merge_v_by_x(v_segs,
                              x_merge_tol=max(int(W * 0.008), 8),
                              y_gap_max=v_merge_y_gap)

    # 横线两端对齐聚类（保留所有 size ≥ 2 的不同聚类）
    h_clusters = []
    seen_h_keys = set()
    for anchor in h_segs:
        sub = [s for s in h_segs
               if abs(s["x1"] - anchor["x1"]) <= x_tol
               and abs(s["x2"] - anchor["x2"]) <= x_tol]
        if len(sub) < 2:
            continue
        x1_med = int(np.median([s["x1"] for s in sub]))
        x2_med = int(np.median([s["x2"] for s in sub]))
        key = (x1_med // max(x_tol, 1), x2_med // max(x_tol, 1), len(sub))
        if key in seen_h_keys:
            continue
        seen_h_keys.add(key)
        ys = sorted(s["y"] for s in sub)
        h_clusters.append({
            "members": sub,
            "x1_med": x1_med, "x2_med": x2_med,
            "y_min": ys[0], "y_max": ys[-1],
            "size": len(sub),
        })

    # 对每个 h-cluster 找左右边竖线
    # 先在 cluster.members 内提取"行间距一致的最长 y-run"（典型表格各行等距，
    # 离群 horizontal line 会带巨大 y_gap），用 run 的 y_min/y_max 做覆盖检查，
    # 避免被一两根远处同 x 长横线把覆盖范围拉得太大。
    def _longest_uniform_y_run(members, gap_lo=0.5, gap_hi=2.0):
        if len(members) < 2:
            return list(members)
        items = sorted(members, key=lambda s: s["y"])
        ys = [s["y"] for s in items]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if not gaps:
            return items
        med_gap = sorted(gaps)[len(gaps) // 2]
        if med_gap <= 0:
            return items
        lo, hi = gap_lo * med_gap, gap_hi * med_gap
        runs = [[0]]
        for i, g in enumerate(gaps):
            if lo <= g <= hi:
                runs[-1].append(i + 1)
            else:
                runs.append([i + 1])
        best = max(runs, key=len)
        return [items[k] for k in best]

    candidates = []
    for hc in h_clusters:
        x1, x2 = hc["x1_med"], hc["x2_med"]
        run = _longest_uniform_y_run(hc["members"])
        if len(run) < 2:
            continue
        run_ys = sorted(s["y"] for s in run)
        y_min, y_max = run_ys[0], run_ys[-1]
        h_span = max(1, y_max - y_min)
        if x2 - x1 < min_side_w or h_span < min_side_h:
            continue

        # 左边竖线：x 接近 x1；用 v_merged（同 x 合并后的虚拟竖线）做覆盖判定。
        # coverage 是真实线段总长（多段累加），不是外接长度，避免"两段中间巨大空隙"
        # 也被算成覆盖。
        left_candidates = []
        for v in v_merged:
            if abs(v["x"] - x1) > end_match:
                continue
            ext_cov = max(0, min(v["y2"], y_max) - max(v["y1"], y_min))
            if ext_cov <= 0:
                continue
            # 真实覆盖率：min(总线段长 / span, 外接覆盖 / span)
            real_cov = min(v["coverage"], ext_cov)
            if real_cov / h_span >= v_cover_ratio:
                left_candidates.append((abs(v["x"] - x1), v))
        if not left_candidates:
            continue
        left_v = min(left_candidates, key=lambda t: t[0])[1]

        # 右边竖线
        right_candidates = []
        for v in v_merged:
            if abs(v["x"] - x2) > end_match:
                continue
            ext_cov = max(0, min(v["y2"], y_max) - max(v["y1"], y_min))
            if ext_cov <= 0:
                continue
            real_cov = min(v["coverage"], ext_cov)
            if real_cov / h_span >= v_cover_ratio:
                right_candidates.append((abs(v["x"] - x2), v))
        if not right_candidates:
            continue
        right_v = min(right_candidates, key=lambda t: t[0])[1]

        zx1 = left_v["x"]
        zx2 = right_v["x"]
        if zx2 - zx1 < min_side_w:
            continue
        # 边竖线允许在 h-cluster 范围之外的小延伸（行线顶/底常常略向外突出），
        # 但禁止整条图纸外框 vertical 把候选高度拉到全图。
        # 上限：h_span 的 15% 或 30px（取大者）。
        max_extend_y = max(int(h_span * 0.15), 30)
        zy1 = max(min(left_v["y1"], right_v["y1"], y_min), y_min - max_extend_y)
        zy2 = min(max(left_v["y2"], right_v["y2"], y_max), y_max + max_extend_y)
        if zy2 - zy1 < min_side_h:
            continue
        # 防御性丢弃：候选高度接近整张图纸高度 → 必然是把图纸外框/区段贯穿性
        # vertical 选作了边竖线，不是真表格。
        if (zy2 - zy1) > H * 0.70:
            continue
        area = (zx2 - zx1) * (zy2 - zy1)
        if area < min_area:
            continue
        candidates.append({
            "zone": (int(zx1), int(zy1), int(zx2), int(zy2)),
            "area": int(area),
            "h_cluster_size": hc["size"],
            "h_run_size": len(run),
            "left_v_len": int(left_v["len"]),
            "left_v_parts": int(left_v.get("parts", 1)),
            "right_v_len": int(right_v["len"]),
            "right_v_parts": int(right_v.get("parts", 1)),
            "h_x1_med": x1, "h_x2_med": x2,
            "h_y_min": y_min, "h_y_max": y_max,
        })

    # 与 mask_zones 重复的去掉
    for mz in mask_zones:
        candidates = [c for c in candidates if _iou(c["zone"], mz) < iou_dedupe_thr]

    # 候选互相去重：保留面积大的
    candidates.sort(key=lambda c: -c["area"])
    kept = []
    for c in candidates:
        if all(_iou(c["zone"], k["zone"]) < iou_dedupe_thr for k in kept):
            kept.append(c)

    info = {
        "h_segs_total": len(h_segs),
        "v_segs_total": len(v_segs),
        "h_clusters_total": len(h_clusters),
        "candidates_kept": len(kept),
        "min_h_len": int(h_min_len),
        "min_v_len": int(v_min_len),
        "x_tol": int(x_tol), "y_tol": int(y_tol),
        "end_match": int(end_match),
        "min_area": int(min_area),
        "v_cover_ratio": float(v_cover_ratio),
        "kept_detail": [
            {"zone": list(c["zone"]), "area": c["area"],
             "h_cluster_size": c["h_cluster_size"],
             "left_v_len": c["left_v_len"],
             "left_v_parts": c.get("left_v_parts", 1),
             "right_v_len": c["right_v_len"],
             "right_v_parts": c.get("right_v_parts", 1)}
            for c in kept
        ],
    }

    if debug_out is not None and SAVE_DEBUG_ARTIFACTS:
        dbg = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        # Hough 输出：直接画 h_segs / v_segs 表示提取到的线
        for h in h_segs:
            cv2.line(dbg, (h["x1"], h["y"]), (h["x2"], h["y"]), (180, 60, 60), 2)
        for v in v_segs:
            cv2.line(dbg, (v["x"], v["y1"]), (v["x"], v["y2"]), (60, 60, 180), 2)
        for mz in mask_zones:
            cv2.rectangle(dbg, (mz[0], mz[1]), (mz[2], mz[3]), (90, 90, 90), 2)
        palette = [(0, 200, 0), (255, 140, 0), (200, 0, 200),
                   (0, 180, 200), (200, 200, 0), (160, 60, 0)]
        for idx, c in enumerate(kept):
            color = palette[idx % len(palette)]
            zx1, zy1, zx2, zy2 = c["zone"]
            cv2.rectangle(dbg, (zx1, zy1), (zx2, zy2), color, 4)
            tag = f"T{idx} {zx2-zx1}x{zy2-zy1}"
            cv2.putText(dbg, tag, (zx1 + 6, max(zy1 + 28, 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        Image.fromarray(dbg).save(debug_out, quality=88)

    zones_only = [c["zone"] for c in kept]
    return zones_only, info


# ────────────────────────────────────────────────────────────
# V3 表格检测 + V6 矩形 zone 过滤
# ────────────────────────────────────────────────────────────
def extract_table_lines(bw, min_line_len_ratio=0.015):
    h, w = bw.shape
    min_len = max(int(max(h, w) * min_line_len_ratio), 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return h_lines, v_lines


def find_bottom_span_y(bw, len_ratio=0.9, y_floor_ratio=2.0 / 3.0):
    """找下方 1/3 区域内"长度 ≥ 宽×len_ratio"的最底部那根横线的 y_top。

    实现说明：原始 bw 里这种贯穿长线通常和上下文字/表格粘连成巨大连通块，
    无法直接用 morph open(w*0.9,1) 提取。先走 extract_table_lines 把横线
    "瘦化提取"出来（min_len=max(h,w)*0.015），再在 CC 中筛宽度 ≥ w*len_ratio。

    返回 None 表示没找到；找到时返回该长横线的 y 顶部坐标（bw 坐标系，int）。
    """
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


def detect_tables(bw, strict_level="strict", zone=None, debug_dir=None, debug_tag=None):
    """V3 strict 判据 + V6 位置 prior（表格 bbox 必须完全落在 zone 矩形内）。

    zone: (x1, y1, x2, y2) 矩形，None 时退化为不过滤（V4 strict 行为）。
    debug_dir / debug_tag: 提供时把 morph 横线/竖线/交点/合并骨架落盘成
        04a_morph_h_{tag}.jpg / 04b_morph_v_{tag}.jpg /
        04c_morph_cross_{tag}.jpg / 04d_morph_skel_{tag}.jpg
    """
    h, w = bw.shape
    h_lines, v_lines = extract_table_lines(bw)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines_d = cv2.dilate(h_lines, dilate_k, iterations=1)
    v_lines_d = cv2.dilate(v_lines, dilate_k, iterations=1)

    skel = cv2.bitwise_or(h_lines_d, v_lines_d)
    cross = cv2.bitwise_and(h_lines_d, v_lines_d)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    skel_closed = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, close_k, iterations=2)

    if debug_dir is not None and SAVE_DEBUG_ARTIFACTS:
        tag = debug_tag or strict_level
        os.makedirs(debug_dir, exist_ok=True)
        # 04a 在横线 morph 图上标记"贯穿底部超长横线"：长度 ≥ w*0.9 且
        # y_center 落在下方 1/3 (y > h*2/3) 的所有 CC 中，y 最大的那一根。
        h_vis = cv2.cvtColor(h_lines_d, cv2.COLOR_GRAY2RGB)
        num_h, _, stats_h, _ = cv2.connectedComponentsWithStats(h_lines_d, connectivity=8)
        long_thresh = w * 0.9
        y_floor = h * 2.0 / 3.0
        target = None
        for i in range(1, num_h):
            cx, cy, cw, ch, _ = stats_h[i]
            yc = cy + ch / 2.0
            if cw >= long_thresh and yc > y_floor:
                if target is None or yc > target[5]:
                    target = (int(cx), int(cy), int(cw), int(ch), int(cx + cw), float(yc))
        if target is not None:
            x1, y1, cw, ch, x2, _ = target
            cv2.rectangle(h_vis, (x1, y1), (x2, y1 + ch), (255, 0, 0), 3)
            cv2.putText(h_vis, f"bottom-span len={cw}",
                        (x1 + 6, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        Image.fromarray(h_vis).save(
            os.path.join(debug_dir, f"04a_morph_h_{tag}.jpg"), quality=88)
        Image.fromarray(v_lines_d).save(
            os.path.join(debug_dir, f"04b_morph_v_{tag}.jpg"), quality=88)
        Image.fromarray(cross).save(
            os.path.join(debug_dir, f"04c_morph_cross_{tag}.jpg"), quality=88)
        Image.fromarray(skel_closed).save(
            os.path.join(debug_dir, f"04d_morph_skel_{tag}.jpg"), quality=88)

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

        # H 线条数（50% 贯穿率，V3 原方式）
        row_white = (roi_h > 0).sum(axis=1)
        h_line_rows = row_white > (cw * 0.5)
        h_count = 0
        prev = False
        for v in h_line_rows:
            if v and not prev:
                h_count += 1
            prev = v

        # V 线条数
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

        # V6 位置 prior: 表格 bbox 必须完全落在 zone 矩形内
        if zone is not None:
            zx1, zy1, zx2, zy2 = zone
            if not (x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2):
                rejected_by_zone += 1
                continue

        tables.append([int(x1), int(y1), int(x2), int(y2)])
        cv2.rectangle(table_mask, (x1, y1), (x2, y2), 255, -1)

    return tables, table_mask, h_lines_d, v_lines_d, skel_closed, rejected_by_zone


# ────────────────────────────────────────────────────────────
# CC 分类（V3 一致）
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# V6 修改: remove_nested 放宽到 90% 包含
# ────────────────────────────────────────────────────────────
def remove_nested(boxes, contain_thr=0.9):
    """若 B 的 ≥ contain_thr 面积被 A 包含且 A > B → 删 B。"""
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


# ────────────────────────────────────────────────────────────
# V6 新增: 边缘小点过滤
# ────────────────────────────────────────────────────────────
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


def _draw_dashed_rect(vis, zone, color=(255, 220, 0), thickness=4, dash=20, gap=20):
    """画虚线矩形 zone=(x1,y1,x2,y2)。"""
    x1, y1, x2, y2 = zone
    for x0 in range(x1, x2, dash + gap):
        cv2.line(vis, (x0, y1), (min(x0 + dash, x2), y1), color, thickness)
        cv2.line(vis, (x0, y2), (min(x0 + dash, x2), y2), color, thickness)
    for y0 in range(y1, y2, dash + gap):
        cv2.line(vis, (x1, y0), (x1, min(y0 + dash, y2)), color, thickness)
        cv2.line(vis, (x2, y0), (x2, min(y0 + dash, y2)), color, thickness)


def _draw_exclusion_overlay(vis, exclusion_zone, fill=(180, 60, 200), alpha=0.18,
                            border=(180, 0, 200)):
    """对 exclusion_zone=(0,0,zx2,zy2) 画半透明紫色填充 + 紫色虚线边界，
    表示 OCR 阶段应该排除的左上大区域。"""
    if exclusion_zone is None:
        return
    x1, y1, x2, y2 = exclusion_zone
    if x2 <= x1 or y2 <= y1:
        return
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    _draw_dashed_rect(vis, exclusion_zone, color=border, thickness=3, dash=16, gap=12)


def draw_reference_overlay(img_rgb, ref_boxes, zone=None, exclusion_zone=None):
    """只画 zone 黄虚线矩形 + 可选 exclusion_zone 紫色填充。"""
    vis = img_rgb.copy()
    _draw_exclusion_overlay(vis, exclusion_zone)
    if zone is not None:
        _draw_dashed_rect(vis, zone, color=(255, 220, 0), thickness=4)
        zx1, zy1, zx2, zy2 = zone
        cv2.putText(vis, f"zone=({zx1},{zy1})-({zx2},{zy2})",
                    (max(zx1, 20), max(zy1 - 12, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 3)
    return vis


def draw_boxes_typed(img_rgb, boxes, tables, zone=None, ref_boxes=None,
                     exclusion_zone=None):
    vis = img_rgb.copy()
    _draw_exclusion_overlay(vis, exclusion_zone)
    if zone is not None:
        _draw_dashed_rect(vis, zone, color=(255, 220, 0), thickness=4)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 4)
        tag = f"{i}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        ty = y1 - 6 if y1 - 6 > th + 4 else y1 + th + 8
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 12, ty + 6), (0, 200, 0), -1)
        cv2.putText(vis, tag, (x1 + 6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    for i, tb in enumerate(tables):
        x1, y1, x2, y2 = tb[:4]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (220, 0, 0), 5)
        tag = f"T{i}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        ty = y1 - 6 if y1 - 6 > th + 4 else y1 + th + 8
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 12, ty + 6), (220, 0, 0), -1)
        cv2.putText(vis, tag, (x1 + 6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return vis


# ────────────────────────────────────────────────────────────
# 单变体处理
# ────────────────────────────────────────────────────────────
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

    # V4 长线丢弃
    long_thr = min(sw, sh) * 0.3
    line_bboxes = []
    long_lines_dropped = 0
    for i in line_idx:
        x, y, w_cc, h_cc, _ = stats[i]
        if max(w_cc, h_cc) > long_thr:
            long_lines_dropped += 1
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

    # V6: 边缘小点过滤
    edge_filtered, edge_dropped = filter_edge_small_boxes(
        merged_no_table, sh, sw, edge_margin=100, max_area=500)

    # V6: remove_nested 放宽到 90% 包含
    final = remove_nested(edge_filtered, contain_thr=0.9)

    tight = []
    line_thin_dropped = 0
    dot_small_dropped = 0
    for b in final:
        t = tighten_to_content(bw_no_table, b)
        if t is None:
            continue
        bx1, by1, bx2, by2 = t[:4]
        bw_box = bx2 - bx1
        bh_box = by2 - by1
        area_box = bw_box * bh_box
        # 横/竖细线过滤：绝对最薄边 < 3px → 一定是纯线条而不是文字块
        if min(bw_box, bh_box) < 30:
            line_thin_dropped += 1
            continue
        # 小点过滤：面积 < 60 → 单像素噪声、断笔残留
        if area_box < 2000:
            dot_small_dropped += 1
            continue
        tight.append(t + [area_box])

    return tight, {"median_h": median_h, "h_gap": h_gap, "v_gap": v_gap,
                   "text_cc": len(text_idx), "line_cc": len(line_idx),
                   "block_cc": len(block_idx),
                   "long_lines_dropped": long_lines_dropped,
                   "edge_dropped": edge_dropped,
                   "line_thin_dropped": line_thin_dropped,
                   "dot_small_dropped": dot_small_dropped}


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────
STRICT_LEVELS = ["strict"]
H_FACTORS = [8]
TABLE_Y_BUFFER = 30

# 是否落盘"人眼调试图"（中间二值/morph/cluster/V5 标注/最终大图/_v5_crops 等）。
# False 时仅保留 _y_crops/*.jpg 和 y_boxes.csv —— 这两个是流水线对外结果。
# 注意：_y_crops/ 的路径写在 y_boxes.csv 的 crop_path 列里，不能跳过。
SAVE_DEBUG_ARTIFACTS = False

# 全局收集：Y 编号最终框（原图坐标系），跨文件汇总到一张 y_boxes.csv
# 字段：source_file, token, x1, y1, x2, y2, crop_path
# 用途：人工核对/手动重画框时的位置参考；识别全部正确时不会被读取。
_y_box_records: list[dict] = []

for file_idx, INPUT_FILE in enumerate(INPUT_FILES):
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    # 最终可视化按变体分类：DEBUG_BASE/{strict}_h{h_factor}/{stem}.jpg
    # 中间调试产物（二值图、cluster 调试图、v6_sweep.json 等）放 _debug/{stem}/
    # 用户日常只看顶层变体文件夹，_debug 目录仅用于排查问题。
    out_dir = os.path.join(DEBUG_BASE, "_debug", stem)
    if SAVE_DEBUG_ARTIFACTS:
        os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"[{file_idx+1}/{len(INPUT_FILES)}] {os.path.basename(INPUT_FILE)}")

    try:
        img_array, _ = load_file(INPUT_FILE)
        enhanced = _enhance_vertical_lines(img_array)
        regions = detect_all_regions(enhanced, prefixes=prefixes)

        rot_code = regions.get("_metadata", {}).get("rotation")
        if rot_code is not None:
            img_array = cv2.rotate(img_array, rot_code)

        img_h, img_w = img_array.shape[:2]
        red_bbox = regions.get("material_code_column")
        green_bbox = regions.get("bottom_right_number")
        orange_bbox = regions.get("top_left_number")

        # ── 红框（代号/material_code）检测过程图 ──
        # 00a：detect_all_regions 用来做 OCR 找 "代号/MATERIAL CODE" 的搜索区截图
        # 00b：原图上叠加搜索区（黄）+ 最终红框（红实线）/ 失败标记
        search_areas = regions.get("_metadata", {}).get("search_areas", {}) or {}
        red_search = search_areas.get("red_search")
        if red_search is not None:
            rsx1 = max(0, int(red_search.x))
            rsy1 = max(0, int(red_search.y))
            rsx2 = min(img_w, int(red_search.x2))
            rsy2 = min(img_h, int(red_search.y2))
            if SAVE_DEBUG_ARTIFACTS:
                if rsx2 > rsx1 and rsy2 > rsy1:
                    rs_crop = img_array[rsy1:rsy2, rsx1:rsx2].copy()
                    Image.fromarray(rs_crop).save(
                        os.path.join(out_dir, "00a_red_search.jpg"), quality=88)

                res_vis = img_array.copy()
                cv2.rectangle(res_vis, (rsx1, rsy1), (rsx2 - 1, rsy2 - 1),
                              (255, 200, 0), 4)  # 黄：搜索区
                if red_bbox is not None:
                    cv2.rectangle(res_vis,
                                  (int(red_bbox.x), int(red_bbox.y)),
                                  (int(red_bbox.x2) - 1, int(red_bbox.y2) - 1),
                                  (255, 0, 0), 6)  # 红：最终红框
                    cv2.putText(res_vis, "MATERIAL_CODE",
                                (int(red_bbox.x), max(0, int(red_bbox.y) - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                else:
                    cv2.putText(res_vis, "RED BOX: NOT DETECTED",
                                (rsx1, max(0, rsy1 - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
                Image.fromarray(res_vis).save(
                    os.path.join(out_dir, "00b_red_result.jpg"), quality=88)

        fn_top = 0
        fn_bottom = green_bbox.y if green_bbox else img_h
        fn_left = orange_bbox.x if orange_bbox else 0
        fn_right = img_w
        search_roi = img_array[fn_top:fn_bottom, fn_left:fn_right].copy()
        sh, sw = search_roi.shape[:2]

        # 参考框（搜索区坐标系）
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
        logger.info(f"  ref_boxes (ROI 坐标): MC={ref_boxes['material_code']}, "
                    f"BR={ref_boxes['bottom_right_number']}, TL={ref_boxes['top_left_number']}")

        # 不再涂白任何参考框 —— 保留全部像素证据给线聚类算法；
        # 参考框（MC/BR/TL）在 box 过滤阶段单独处理。
        gray = cv2.cvtColor(search_roi, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if SAVE_DEBUG_ARTIFACTS:
            Image.fromarray(search_roi).save(os.path.join(out_dir, "01_search.jpg"), quality=88)
            Image.fromarray(bw).save(os.path.join(out_dir, "02a_binary_raw.jpg"), quality=88)

        bw_no_border, border_segs, border_mask = erase_drawing_border(bw)
        logger.info(f"  擦除外框线段: {len(border_segs)} 条 (mask px={int((border_mask>0).sum())})")
        for e in border_segs:
            logger.info(f"    {e['type']}-line: bbox={e['bbox']} len={e['len']}")
        if SAVE_DEBUG_ARTIFACTS:
            Image.fromarray(bw_no_border).save(os.path.join(out_dir, "02b_binary_no_border.jpg"), quality=88)

        # 诊断图：被擦像素在 search_roi 上用红色叠加，让肉眼判断是否误擦了表格。
        if border_mask.any() and SAVE_DEBUG_ARTIFACTS:
            erase_dbg = search_roi.copy()
            red_overlay = erase_dbg.copy()
            red_overlay[border_mask > 0] = (255, 0, 0)
            erase_dbg = cv2.addWeighted(erase_dbg, 0.5, red_overlay, 0.5, 0)
            for e in border_segs:
                x1, y1, x2, y2 = e["bbox"]
                cv2.rectangle(erase_dbg, (x1, y1), (x2, y2), (255, 220, 0), 2)
            Image.fromarray(erase_dbg).save(
                os.path.join(out_dir, "02b_erase_diagnostic.jpg"), quality=88)

        bw = bw_no_border

        # 性能裁切：找下方 1/3 区域内"长度≥宽×0.9"的最底部超长横线，
        # 将 search_roi / bw 一起裁到该线以上，缩小所有下游 OCR/VLM 输入。
        y_cut = find_bottom_span_y(bw)
        if y_cut is not None and y_cut > 0:
            mc_roi_pre = ref_boxes.get("material_code")
            if mc_roi_pre is not None and mc_roi_pre[1] >= y_cut:
                logger.info(f"  bottom-span y={y_cut} 落在 MC 上方，跳过裁切")
            else:
                old_sh = sh
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
                logger.info(f"  bottom-span y={y_cut}，search_roi/bw 由 h={old_sh} 裁到 h={sh}")
                if SAVE_DEBUG_ARTIFACTS:
                    Image.fromarray(bw).save(
                        os.path.join(out_dir, "02d_binary_cut.jpg"), quality=88)

        # V6 位置 prior：
        #   zone.left = 0, zone.top = MC.y1
        #   zone.bottom = 两端对齐+等长横线集合的最下面那条 y
        #   zone.right  = 等高竖线集合中"右侧空隙巨大"的那条 x
        table_zone = None
        cluster_info = None
        cluster_status = None
        mc_roi = ref_boxes["material_code"]
        if mc_roi is not None:
            _, mc_y1_roi, _, _ = mc_roi
            zone_out, info = cluster_table_zone(
                bw, mc_y1=mc_y1_roi, buffer=TABLE_Y_BUFFER,
                debug_out=os.path.join(out_dir, "02c_cluster_debug.jpg"))
            if zone_out is not None:
                table_zone = zone_out
                cluster_info = info
                cluster_status = "ok"
                logger.info(f"  cluster zone={table_zone} "
                            f"(h_aligned={info['h_aligned_set_size']} "
                            f"y_bot={info['h_y_bottom_chosen']}, "
                            f"v_eq={info['v_equal_height_set_size']} "
                            f"x_right={info['x_right_chosen']})")
            else:
                cluster_status = f"fallback: {info}"
                cluster_info = info
                logger.info(f"  cluster 失败 ({info}) → 表格全图检测（V4 strict 行为）")
        else:
            cluster_status = "no_material_code"
            logger.info(f"  无 material_code → 表格全图检测（V4 strict 行为）")

        # OCR 排除大区域：基于 table_zone 推导
        #   exclusion_zone = (0, 0, table_zone.right, table_zone.bottom)
        # 含义：左/上至截图边界，右/下至工厂注意表格的右/下边界。
        # 该区域包含标题栏 + 工厂注意表格 + 它们左上方的所有内容，
        # 下游 OCR 阶段应完全忽略此区域内的文字。
        exclusion_zone = None
        if table_zone is not None:
            _, _, tz_right, tz_bottom = table_zone
            exclusion_zone = (0, 0, int(tz_right), int(tz_bottom))

        # 多表格全图扫描已禁用：只检测左上角主表，不再输出其他候选。
        extra_table_zones = []
        scan_info = None

        # 在 OCR / 文字块生成之前，把 exclusion_zone (= 紫色覆盖的"主表+左上区域")
        # 在 bw 中整片涂成白色（bw 用 THRESH_BINARY_INV → 255=ink，0=background；
        # 所以"涂白" = 把像素置 0）。这样后续 process_variant 在该区域内不会再
        # 产生任何文字块/零件 OCR 候选框。detect_tables 仍用原始 bw 跑（它只负责
        # 在该 zone 内画出主表红框，不影响后续 OCR）。
        bw_for_ocr = bw
        if exclusion_zone is not None:
            bw_for_ocr = bw.copy()
            ex1, ey1, ex2, ey2 = exclusion_zone
            # 向四周外扩 3px，确保边界上残留的细线像素也被清掉
            pad = 3
            bh, bw_w = bw_for_ocr.shape
            ex1 = max(0, ex1 - pad)
            ey1 = max(0, ey1 - pad)
            ex2 = min(bw_w, ex2 + pad)
            ey2 = min(bh, ey2 + pad)
            bw_for_ocr[ey1:ey2, ex1:ex2] = 0
            if SAVE_DEBUG_ARTIFACTS:
                Image.fromarray(bw_for_ocr).save(
                    os.path.join(out_dir, "02e_binary_ocr_input.jpg"), quality=88)

        # 多表格候选用循环蓝色系画到主可视化图上（区别于主表红色框 / 黄色 zone）
        extra_palette = [(0, 120, 255), (255, 120, 0), (180, 0, 200),
                         (0, 180, 200), (200, 200, 0), (160, 60, 0)]

        def _overlay_extra_tables(img):
            for idx, ez in enumerate(extra_table_zones):
                color = extra_palette[idx % len(extra_palette)]
                ex1, ey1, ex2, ey2 = ez
                cv2.rectangle(img, (ex1, ey1), (ex2, ey2), color, 4)
                tag = f"T{idx}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.rectangle(img, (ex1, ey1 - th - 10), (ex1 + tw + 12, ey1), color, -1)
                cv2.putText(img, tag, (ex1 + 6, ey1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        variants_summary = []
        for strict in STRICT_LEVELS:
            tables, table_mask, h_lines, v_lines, skel, rej_zone = detect_tables(
                bw, strict_level=strict, zone=table_zone,
                debug_dir=out_dir, debug_tag=strict)
            logger.info(f"  [{strict}] 表格数: {len(tables)} (zone外被排除: {rej_zone})")

            if SAVE_DEBUG_ARTIFACTS:
                tab_vis = draw_reference_overlay(search_roi, ref_boxes,
                                                 zone=table_zone,
                                                 exclusion_zone=exclusion_zone)
                for tb in tables:
                    cv2.rectangle(tab_vis, (tb[0], tb[1]), (tb[2], tb[3]), (220, 0, 0), 5)
                _overlay_extra_tables(tab_vis)
                Image.fromarray(tab_vis).save(
                    os.path.join(out_dir, f"03_tables_{strict}.jpg"), quality=88)

            for h_factor in H_FACTORS:
                tight, stat = process_variant(bw_for_ocr, tables, table_mask, sh, sw, h_factor)

                vis = draw_boxes_typed(search_roi, tight, tables,
                                       zone=table_zone, ref_boxes=ref_boxes,
                                       exclusion_zone=exclusion_zone)
                _overlay_extra_tables(vis)
                tag = f"{strict}_h{h_factor}"
                # 最终可视化按变体目录归类：DEBUG_BASE/{strict}_h{h_factor}/{stem}.jpg
                variant_dir = os.path.join(DEBUG_BASE, tag)
                os.makedirs(variant_dir, exist_ok=True)
                final_path = os.path.join(variant_dir, f"{stem}.jpg")
                if SAVE_DEBUG_ARTIFACTS:
                    Image.fromarray(vis).save(final_path, quality=88)
                fname = os.path.relpath(final_path, DEBUG_BASE)

                # ── V5 粗扫筛 Y → VLM 二次识别画准确框 ──
                # tight + 不在紫区（工厂注意主表）内的 tables 作为候选；
                # V5 只看单段 polygon 文本是否匹配 Y_PATTERN，命中才进 VLM；
                # VLM 也只为单段命中 Y_PATTERN 的 polygon 画红框 + 标 Y 编号；
                # 仅最终加框的裁切图落盘。
                ex = exclusion_zone
                def _inside_ex(box):
                    if ex is None:
                        return False
                    bx1, by1, bx2, by2 = box[:4]
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    return ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3]

                crop_targets = []
                for b in tight:
                    crop_targets.append(("text", b[:4]))
                for tb in tables:
                    if _inside_ex(tb):
                        continue
                    crop_targets.append(("table", tb[:4]))

                v5_crops_dir = os.path.join(variant_dir, "_v5_crops", stem)
                y_crops_dir = os.path.join(variant_dir, "_y_crops", stem)
                y_hits = []
                v5_kept = 0
                v5_total = 0
                # 跨 crop 的 polygon 级去重：(global_bbox, token) 列表。
                # 同 token + IoU≥0.5 视为同一物理位置（如 857 text_015/016）。
                seen_global_polys: list[tuple[tuple[int, int, int, int], str]] = []
                if crop_targets:
                    if SAVE_DEBUG_ARTIFACTS:
                        os.makedirs(v5_crops_dir, exist_ok=True)
                    for idx, (kind, box) in enumerate(crop_targets):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(sw, x2); y2 = min(sh, y2)
                        if x2 - x1 < 6 or y2 - y1 < 6:
                            continue
                        crop = search_roi[y1:y2, x1:x2]
                        pil_crop = Image.fromarray(crop)

                        # V5 跑一次，产出全部 polygon 标注
                        np_img, items = _v5_run(pil_crop)
                        v5_total += 1
                        v5_crop_path = os.path.join(
                            v5_crops_dir, f"{kind}_{idx:03d}.jpg")
                        if SAVE_DEBUG_ARTIFACTS:
                            v5_annotated = _v5_annotate_all(np_img, items)
                            Image.fromarray(v5_annotated).save(v5_crop_path, quality=92)

                        # 命中 Y 才进入 VLM 二次识别
                        v5_pass, v5_hits, v5_texts = _v5_filter_y(items)
                        if not v5_pass:
                            continue

                        # polygon 级去重：把命中 Y 的每个 polygon 映射到 search_roi
                        # 全图坐标系，与已收纳的 (bbox, token) 比 IoU。
                        # 全部都是重复 → 整个 crop 跳过 VLM 与落盘。
                        crop_y_polys: list[tuple[tuple[int, int, int, int], str]] = []
                        for poly, text, _ in items:
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
                            logger.info(
                                f"    [{strict}|h={h_factor}] {kind}_{idx:03d} "
                                f"polygon 全部与已收 Y 重叠(IoU≥0.5)，跳过 VLM")
                            continue
                        seen_global_polys.extend(new_polys)
                        v5_kept += 1

                        if not os.path.isdir(y_crops_dir):
                            os.makedirs(y_crops_dir, exist_ok=True)
                        annotated, vlm_matched = _vlm_annotate_single(pil_crop)
                        hit_tag = "_".join(_safe_filename_tag(h) for h in v5_hits[:2])
                        crop_path = os.path.join(
                            y_crops_dir,
                            f"{kind}_{idx:03d}_{hit_tag}.jpg")
                        Image.fromarray(annotated).save(crop_path, quality=92)

                        # 记录每个 Y 编号的最终框（原图坐标系）到全局表
                        # 换算：poly(crop 局部) + (x1, y1)(crop→search_roi) + (fn_left, fn_top)(roi→原图)
                        crop_rel = os.path.relpath(crop_path, DEBUG_BASE)
                        for _poly, _tk in vlm_matched:
                            xs = [p[0] for p in _poly]
                            ys = [p[1] for p in _poly]
                            bx1 = min(xs) + x1 + fn_left
                            by1 = min(ys) + y1 + fn_top
                            bx2 = max(xs) + x1 + fn_left
                            by2 = max(ys) + y1 + fn_top
                            _y_box_records.append({
                                "source_file": os.path.basename(INPUT_FILE),
                                "token": _tk,
                                "x1": int(bx1), "y1": int(by1),
                                "x2": int(bx2), "y2": int(by2),
                                "crop_path": crop_rel,
                            })

                        y_hits.append({
                            "kind": kind,
                            "box_roi": [x1, y1, x2, y2],
                            "v5_hits": v5_hits,
                            "v5_texts": v5_texts,
                            "v5_crop": os.path.relpath(v5_crop_path, DEBUG_BASE),
                            "vlm_polys": [
                                {"poly": p, "token": tk}
                                for p, tk in vlm_matched
                            ],
                            "crop": os.path.relpath(crop_path, DEBUG_BASE),
                        })

                logger.info(f"    [{strict}|h={h_factor}] tables={len(tables)} boxes={len(tight)} "
                            f"(text_cc={stat['text_cc']} line={stat['line_cc']} block={stat['block_cc']} "
                            f"long_drop={stat['long_lines_dropped']} edge_drop={stat['edge_dropped']} "
                            f"thin_drop={stat['line_thin_dropped']} dot_drop={stat['dot_small_dropped']} "
                            f"median_h={stat['median_h']} h_gap={stat['h_gap']}) "
                            f"v5_run={v5_total}/{len(crop_targets)} v5_y_hit={v5_kept} vlm_drawn={len(y_hits)}")

                variants_summary.append({
                    "strict": strict,
                    "h_factor": h_factor,
                    "tables": len(tables),
                    "text_block_count": len(tight),
                    "total_boxes": len(tight) + len(tables),
                    "stat": stat,
                    "file": fname,
                    "v5_kept": v5_kept,
                    "y_hits": y_hits,
                })

        summary = {
            "file": os.path.basename(INPUT_FILE),
            "search_area": {"w": sw, "h": sh},
            "table_zone": table_zone,
            "exclusion_zone": exclusion_zone,
            "extra_table_zones": extra_table_zones,
            "scan_info": scan_info,
            "cluster_status": cluster_status,
            "cluster_info": cluster_info,
            "border_lines_erased": len(border_segs),
            "border_lines_detail": border_segs,
            "tables_rejected_by_zone": rej_zone,
            "variants": variants_summary,
        }

        def _to_jsonable(o):
            import numpy as _np
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
            if isinstance(o, tuple):
                return list(o)
            raise TypeError(f"Object of type {o.__class__.__name__} not JSON serializable")
        if SAVE_DEBUG_ARTIFACTS:
            with open(os.path.join(out_dir, "v6_sweep.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=_to_jsonable)
            logger.info(f"  输出: {out_dir}")

    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)

    # 处理完一张图后主动释放所有大型 ndarray 引用 + 触发 GC。
    # 大头：img_array/enhanced（原图 RGB，~80MB 一张）、search_roi/bw/
    # bw_no_border/bw_for_ocr/table_mask/h_lines/v_lines/skel 以及 V5/VLM
    # 在调用过程中产生的 crop/annotated 临时数组。不显式删的话，这些循环 local
    # 变量要等到下一轮对应变量被赋值才会被覆盖，期间 RSS 会叠加。
    _g = globals()
    for _v in ("img_array", "enhanced", "search_roi", "bw", "bw_no_border",
               "bw_for_ocr", "table_mask", "h_lines", "v_lines", "skel",
               "regions", "tight", "tables", "crop_targets",
               "seen_global_polys", "summary", "variants_summary"):
        _g.pop(_v, None)
    gc.collect()

logger.info("=" * 60)
# 汇总所有文件的 Y 编号框（原图坐标系）到 y_boxes.csv
# 仅在识别/替换出现问题时供人工核对、按坐标手动重画框使用；正常流程不读取。
_csv_path = os.path.join(DEBUG_BASE, "y_boxes.csv")
with open(_csv_path, "w", encoding="utf-8", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=[
        "source_file", "token", "x1", "y1", "x2", "y2", "crop_path"])
    _w.writeheader()
    _w.writerows(_y_box_records)
logger.info(f"Y 编号框坐标汇总: {_csv_path} ({len(_y_box_records)} 条)")
logger.info(f"测试完成: {DEBUG_BASE}")
