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
import threading

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_PREFIXES, make_pattern
from modules.region_detector import (
    BBox,
    _get_ocr_v5,
    _parse_ocr_results_common,
    _classify_corner_ori,
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
_Y_RE = re.compile(make_pattern(DEFAULT_PREFIXES))
_V_RE = re.compile(r"V(?=[A-Z0-9]*\d)[A-Z0-9]{6,}")

# ── _localize_y_box 增强开关（工厂注意框选精修）─────────────────────
# ① 溢出兜底时用 VLM 重认修正幻读的 code（如 ')' 幻读成 '1' → X06AX-791），
#    再用正确 code 重跑二分定位，修 X06AX-791 溢出整行(宽694→295)。
# ③ 上下墨迹贴合：用 VLM 同一编号分词的 y 范围收紧上下边界，修框上下不贴字
#    （如 YA046D753-02 上边界框到横线，高69→44）。剔窄笔画(②)未并入。
FN_LOCALIZE_REFINE = True
_SAME_ID_TBL = str.maketrans({'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8'})


def _same_id(a: str, b: str) -> bool:
    """两编号是否'基本同一'：OCR混淆归一(O→0等)+去非字母数字后 相等，
    或一个是另一个去尾≤2字符（容忍 X06AX-791 vs X06AX-79 的幻读1）。"""
    def _n(s):
        return re.sub(r'[^A-Z0-9]', '', (s or '').upper()).translate(_SAME_ID_TBL)
    a, b = _n(a), _n(b)
    if not a or not b:
        return False
    return (a == b or (a.startswith(b) and len(a)-len(b) <= 2)
            or (b.startswith(a) and len(b)-len(a) <= 2))


def _vlm_same_poly(crop_np, poly, target_tok):
    """VLM 放大重认 crop，找与 target_tok '基本同一'的分词，返回
    (vtok, sub_x1, sub_x2, crop_top, crop_bot) 或 None。坐标为 crop 绝对坐标。"""
    x1, y1, x2, y2 = _poly_bbox(poly)
    H, W = crop_np.shape[:2]
    ch = max(8, y2 - y1); pad = max(int(ch * 0.6), 12)
    sx1 = max(0, x1 - pad); sy1 = max(0, y1); sx2 = min(W, x2 + pad); sy2 = min(H, y2)
    sub = crop_np[sy1:sy2, sx1:sx2]
    UP = 2.5
    try:
        its = _parse_ocr_results_common(
            _get_vlm().predict(cv2.resize(sub, None, fx=UP, fy=UP,
                                          interpolation=cv2.INTER_CUBIC)))
    except Exception:  # noqa: BLE001
        return None
    for p2, t2, _sc in its:
        if p2 is None or len(p2) == 0:
            continue
        vt = _find_y_token(t2 or "")
        if vt and _same_id(target_tok, vt):
            bx1, by1_, bx2, by2_ = _poly_bbox(p2)
            return (vt, sx1 + int(bx1 / UP), sx1 + int(bx2 / UP),
                    sy1 + int(by1_ / UP), sy1 + int(by2_ / UP))
    return None


def _fit_updown(crop_np, abs_l, abs_r, vTop, vBot):
    """在编号列 [abs_l,abs_r] 内、VLM poly 的 y 范围 [vTop,vBot] 内按墨迹收紧上下边界。"""
    g2 = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY) if crop_np.ndim == 3 else crop_np
    _, bw2 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vTc = max(0, vTop); vBc = min(crop_np.shape[0], vBot)
    if vBc <= vTc or abs_r <= abs_l:
        return vTc, vBc
    win = bw2[vTc:vBc, abs_l:abs_r]
    rh = (win > 0).any(axis=1); ys = np.where(rh)[0]
    if len(ys):
        return vTc + int(ys[0]), vTc + int(ys[-1]) + 1
    return vTc, vBc


def _safe_filename_tag(text: str, max_len: int = 32) -> str:
    cleaned = _FN_SAFE_RE.sub("_", (text or "").strip())[:max_len].strip("_.")
    return cleaned or "hit"


def _find_y_token(text: str) -> str | None:
    """在 OCR 文本里搜符合 Y_PATTERN 的 token；首字母为 V 时回填为 Y。

    左边界 (?<![A-Z0-9])：编号首字母(Y/X/B/H)前不得紧贴字母或数字，即编号必须
    是一个字母数字连续段的开头。否则会从元件型号内部误截——如 OCR 识别为整段的
    'MC74HC4046AF'(首字母 M，型号)会被从内部 H 截出假编号 'HC4046AF'。真编号
    前面是分隔符(如 '1.021.YA057C800' 的 '.'、空格、行首)则不受影响，正常匹配。

    含连字符编号 (STRUCT 分支)：X45AT-03 / X44HT-02 / X55GA-21 这类"首字母+数字段
    +字母段+连字符+数字段"的编号，连字符打断了连续段，首字母后连续仅 5 位，不满足
    基础 pattern 的 {6,}，会整列漏检。STRUCT 分支 [P][0-9]{1,3}[A-Z]{1,3}-[0-9]{1,4}
    专门纳入这类；因要求完整"数字段+字母段+连字符+数字段"结构，不会误吞 H35 这种
    行号短序号，也不误中 HCPL-7840-300(H 后是纯字母段 CPL，无数字段)。
    """
    if not text:
        return None
    up = text.upper()
    chars = "".join(p.upper() for p in DEFAULT_PREFIXES)
    cls = chars if len(chars) == 1 else f"[{chars}]"
    LB = r"(?<![A-Z0-9])"
    STRUCT = rf"{cls}[0-9]{{1,3}}[A-Z]{{1,3}}-[0-9]{{1,4}}"
    # 结构式在前(纳入 X45AT-03 类)，基础 6+ 在后(保 YX304B657A-01→YX304B657A)
    m = re.search(rf"{LB}(?:{STRUCT}|{make_pattern(DEFAULT_PREFIXES)})", up)
    if m:
        return m.group()
    m = re.search(LB + r"V(?=[A-Z0-9]*\d)[A-Z0-9]{6,}", up)
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


# ── ROI 旋转坐标变换（工厂注意竖排文字处理）────────────────────
def _rot_point_fwd(x, y, W0, H0, rot):
    """把原始 ROI 坐标 (x,y) 变换到旋转后坐标系。W0/H0=原始宽/高。"""
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return H0 - 1 - y, x
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return y, W0 - 1 - x
    return x, y


def _rot_point_inv(x, y, W0, H0, rot):
    """把旋转后坐标 (x,y) 逆变换回原始 ROI 坐标。W0/H0=原始宽/高。"""
    if rot == cv2.ROTATE_90_CLOCKWISE:
        # fwd: (x,y)->(H0-1-y, x); inv:
        return y, H0 - 1 - x
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
        # fwd: (x,y)->(y, W0-1-x); inv:
        return W0 - 1 - y, x
    return x, y


def _rot_bbox(box, W0, H0, rot, inv=False):
    """对 bbox(x1,y1,x2,y2) 做旋转/逆旋转，用两对角点变换后重新 min/max。"""
    if rot is None or box is None:
        return box
    x1, y1, x2, y2 = box
    f = _rot_point_inv if inv else _rot_point_fwd
    pts = [f(x1, y1, W0, H0, rot), f(x2, y1, W0, H0, rot),
           f(x2, y2, W0, H0, rot), f(x1, y2, W0, H0, rot)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


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
    # 贯穿线剔除:sub 条带里若混进横贯全宽的实线(单元格边框线),会在 runs 里形成
    # 一个跨度≈全宽的墨迹段,污染下游 _snap_to_runs 使框被拉到两端(991 曾 433→505)。
    # 字符笔画最宽也仅一个字,能横贯全宽的必是线 → 剔除跨度≥80% 全宽的段。
    _sub_w = x2 - x1
    if any((re_ - rs) >= _sub_w * 0.8 for rs, re_ in runs):
        runs = [(rs, re_) for rs, re_ in runs if (re_ - rs) < _sub_w * 0.8]
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
    result_poly, result_tok = _fallback_box(lc_snap, rc_snap, tok_used)

    if not FN_LOCALIZE_REFINE:
        return result_poly, result_tok

    # ── 精修 ①③（crop 局部坐标）──
    rxs = [p[0] for p in result_poly]; rys = [p[1] for p in result_poly]
    box_l, box_r = min(rxs), max(rxs)
    box_t, box_b = min(rys), max(rys)
    code_out = result_tok
    _ox1, _oy1, _ox2, _oy2 = _poly_bbox(poly)   # 原始 poly(未加pad)宽, 判溢出基准
    poly_w = _ox2 - _ox1

    # ① 溢出兜底(框宽≈全候选宽=二分失败) → VLM 重认修正 code, 用正确 code 重跑二分
    if (box_r - box_l) >= poly_w:
        same = _vlm_same_poly(crop_np, poly, tok_used)
        if same:
            vtok, _vL, _vR, _vT, _vB = same
            if not _same_id(vtok, tok_used) or vtok.upper() != tok_used.upper():
                # 用修正 code 重跑本函数的二分(递归一层, 关精修避免死循环)
                _save = None
                try:
                    globals()["FN_LOCALIZE_REFINE"] = False
                    re_res = _localize_y_box(crop_np, poly, vtok)
                finally:
                    globals()["FN_LOCALIZE_REFINE"] = True
                if re_res is not None:
                    rp, rt = re_res
                    rxs2 = [p[0] for p in rp]
                    if (max(rxs2) - min(rxs2)) < poly_w:   # 修正后收紧成功
                        rys2 = [p[1] for p in rp]
                        box_l, box_r = min(rxs2), max(rxs2)
                        box_t, box_b = min(rys2), max(rys2)
                        code_out = rt

    # ③ 上下墨迹贴合：用 VLM 同一编号分词的 y 范围收紧上下(仅更紧时采用)
    same2 = _vlm_same_poly(crop_np, poly, code_out or tok_used)
    if same2:
        _vt, _vL, _vR, vTop, vBot = same2
        nt, nb = _fit_updown(crop_np, box_l, box_r, vTop, vBot)
        if nb > nt and (nb - nt) <= (box_b - box_t):
            box_t, box_b = nt, nb

    # 下边界往下扩 3px（不超出 crop 高度）
    box_b = min(crop_np.shape[0], box_b + 3)

    return [(box_l, box_t), (box_r, box_t), (box_r, box_b), (box_l, box_b)], code_out


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


# ── 自动矫正: 小角度倾斜校正(deskew) ─────────────────────────────
def _fn_auto_deskew(roi_rgb, bw, ref_boxes):
    """自动矫正搜索区的小角度倾斜。零写死角度——每张图自扫。

    倾斜会让表格横边框斜穿多行,按行统计的连续横线长度不达标 → detect_tables
    的 h_count 判据失败 → 表格检不出 → 编号识别崩坏(如 994 原 tables=0/h943)。
    这里在 -1.5~+1.5° 范围扫描,取「横线响应最强」的角度(横线越平直=图越正),
    对 search_roi / bw / ref_boxes 同步旋正。正的图峰值在 0° 附近,基本不动。

    Args:
        roi_rgb: 搜索区 RGB。
        bw: 搜索区二值图(线为白)。
        ref_boxes: {name: (x1,y1,x2,y2) | None} 材料/绿/橙框在 ROI 坐标。
    Returns:
        (roi_r, bw_r, ref_boxes_r, angle_deg)。angle 为 0 时原样返回。
    """
    def _resp(a):
        h, w = bw.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        hl, _ = extract_table_lines(r)
        rw = (hl > 0).sum(axis=1)
        return float(np.sort(rw)[-10:].mean())  # 前10强行均值, 抗单条噪声

    coarse = np.arange(-1.5, 1.51, 0.2)
    best_a = max(coarse, key=_resp)
    fine = np.arange(best_a - 0.2, best_a + 0.201, 0.05)
    ang = float(max(fine, key=_resp))
    if abs(ang) < 0.05:
        return roi_rgb, bw, ref_boxes, 0.0

    h, w = bw.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
    bw_r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    roi_r = cv2.warpAffine(roi_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

    def _rot_box(b):
        if b is None:
            return None
        x1, y1, x2, y2 = b
        pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], np.float64)
        pr = (M @ np.c_[pts, np.ones(4)].T).T
        nx1 = max(0, int(pr[:, 0].min())); ny1 = max(0, int(pr[:, 1].min()))
        nx2 = min(w, int(pr[:, 0].max())); ny2 = min(h, int(pr[:, 1].max()))
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return (nx1, ny1, nx2, ny2)

    ref_r = {k: _rot_box(v) for k, v in ref_boxes.items()}
    return roi_r, bw_r, ref_r, ang


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
                       right_gap_ratio=1.8, buffer=20, enable_probe=True,
                       mc_y2=None):
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

    # 底边 y 优先采用红框底边(mc_y2)：红框底边在窄列内确定，不受横穿全宽的
    # 删除线影响；而 best_h_set 取"全宽最长对齐组"，删除线常被误选成底边
    # (如 986：删除线 [0-3844] 胜出，y_bottom 与右沿都被带偏)。红框底边已
    # 在上游 _trace_vertical_table 算定且经竖线伴随校正，这里直接复用更可靠。
    if mc_y2 is not None:
        mc_y2_roi = int(mc_y2) - sy1
        if 0 < mc_y2_roi <= rh:
            logger.info(
                f"  底边采用红框底边 mc_y2: y_bottom_roi {y_bottom_roi} → {mc_y2_roi}")
            y_bottom_roi = mc_y2_roi

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

    # ── 端点延伸探测：纳入右侧紧贴的相邻表 ──
    # 这类相邻表的底边与主表底边在同一条 y 线上，相互之间仅有细小断点/间隙
    # （实测 <20px，远小于字符宽）。connectedComponents 在断点处把贯通底边
    # 拆成多段，端点 x 各不相同，故 best_h_set 只收主表那段、相邻表被漏掉。
    # 从右沿出发逐段向右接力：下一段左端点落在 当前右沿+PROBE 内即视为同一条
    # 底边的延续，吃进其右端点。y 用"跟随式"——参考 y 随每次接力更新为上一段
    # 的 y，而非固定全局底边 y；这样扫描件横线轻微扭曲(同一底边 y 沿 x 渐变，
    # 如 987)时仍能逐段跟随接上。表格群结束后右侧无横线段，自然终止不过界。
    PROBE = 40       # 探测步长(px)，略大于实测最大断点(~20px)，留余量
    Y_STEP = 18      # 相邻段间 y 容差：跟随扭曲，逐段累积漂移
    if enable_probe:
        right_ext = x_right_roi
        ref_y = y_bottom_roi
        changed = True
        while changed:
            changed = False
            cand = None
            for s in h_segs:
                if (s["x2"] > right_ext
                        and s["x1"] <= right_ext + PROBE
                        and abs(s["y"] - ref_y) <= Y_STEP):
                    if cand is None or s["x2"] > cand["x2"]:
                        cand = s
            if cand is not None:
                right_ext = cand["x2"]
                ref_y = cand["y"]   # 跟随：参考 y 更新为本段 y
                changed = True
        if right_ext > x_right_roi:
            logger.info(
                f"  端点延伸探测: 右边界 {x_right_roi} → {right_ext} "
                f"(跟随式接入相邻表，起始底边y={y_bottom_roi})")
            x_right_roi = right_ext

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
_y_box_lock = threading.Lock()


def clear_y_box_records() -> None:
    with _y_box_lock:
        _y_box_records.clear()


def record_y_box(source_file: str, token: str, bbox) -> None:
    """记录一个被替换的 Y 编号框（cyan / green / orange / factory_note 通用）。

    bbox 支持 BBox 对象或 (x, y, w, h) 四元组。
    """
    if hasattr(bbox, "x") and hasattr(bbox, "y") and hasattr(bbox, "w") and hasattr(bbox, "h"):
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    else:
        x, y, w, h = bbox
    with _y_box_lock:
        _y_box_records.append({
            "source_file": source_file or "",
            "token": token or "",
            "x1": int(x), "y1": int(y),
            "x2": int(x + w), "y2": int(y + h),
        })


def flush_y_boxes_csv(out_path: str) -> int:
    """把累计的 Y 框写入 CSV；返回记录数。空列表也会写出仅含表头的 CSV。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with _y_box_lock:
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
FN_ORIENT_VOTE_N = 5  # 方向投票抽样数：取最大的 N 个候选投票
FN_ORIENT_VOTE_BY_TEXT = True  # True=按 v5 文字框数取前N投票（避开图形块误判）；False=按面积

# ── found_codes 框规整（宽度归一 + 首/尾行横线定上下边界）────────
FN_NORM_DEV = 0.15       # 框宽偏离同结构组中位 >此比例才归一
FN_NORM_MIN_GROUP = 3    # 同结构组样本 <此数不归一（中位不可靠）
FN_COL_X_TOL = 120       # 列聚类：框中心 x 间距 ≤此值视为同列
FN_HLINE_PROBE = 70      # 首/尾行找横线的纵向探测距离(px)


def _fn_structure(code: str) -> str:
    """把编号抽象成结构串：字母→L 数字→D 其它(符号)→S。用于同类分组。"""
    return ''.join('L' if c.isalpha() else 'D' if c.isdigit() else 'S'
                   for c in code)


def _fn_local_hline(gray, b, edge_y, probe=FN_HLINE_PROBE, min_w_ratio=0.4):
    """从 edge_y 往下 probe 范围、在框宽内做 morph 横线检测，返回最靠近
    edge_y 的横线 y(整图坐标)；无则 None。"""
    x1, x2 = b.x, b.x2
    y1 = edge_y
    y2 = min(gray.shape[0], edge_y + probe)
    if y2 - y1 < 3 or x2 - x1 < 4:
        return None
    roi = gray[y1:y2, x1:x2]
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kw = max(int((x2 - x1) * min_w_ratio), 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    hm = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk, iterations=1)
    ys = [yy for yy in range(hm.shape[0])
          if hm[yy].sum() > 255 * (x2 - x1) * 0.3]
    if not ys:
        return None
    return y1 + min(ys)


def _cluster_by_proximity(items: list) -> list:
    """把结构串已相同的一批编号，按纵向位置链式聚类成组。
    只有"位置接近(纵向间隙≤2×框高) + 字号接近(框高差≤30%) + 字符数相同"的
    相邻编号才归入同簇。用于宽度归一分组——避免把表外/远处的同名编号误当同组
    （归一本为表内防溢出，表外编号字号/位置不同，一刀切同结构分组会误伤）。
    """
    if not items:
        return []
    items = sorted(items, key=lambda f: f["bbox"].y)
    clusters = [[items[0]]]
    for f in items[1:]:
        prev = clusters[-1][-1]
        b, pb = f["bbox"], prev["bbox"]
        vgap = max(0, (b.y - pb.y2) if b.y > pb.y2 else (pb.y - b.y2))
        h_ref = max(pb.h, 1)
        near = vgap <= 2 * h_ref
        same_size = abs(b.h - pb.h) / h_ref <= 0.30
        same_len = len(f["code"]) == len(prev["code"])
        if near and same_size and same_len:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    return clusters


def _normalize_found_codes(found_codes: list, image_rgb: np.ndarray) -> None:
    """就地规整 found_codes 的 bbox（工厂注意专用，不影响红/绿/橙框）：

    1) 宽度归一：按编号结构串(L/D/S)分组，组内样本≥FN_NORM_MIN_GROUP 时取框宽
       中位为通用宽，偏离>FN_NORM_DEV 的框以中心为锚拉回该宽（修右边界吃竖线等）。
    2) 首/尾行横线定界：按框中心 x 聚类分列，列内按 y 排序：
       - 首行：从 OCR 框上边界往下找最近横线，有则用作新上边界；
       - 尾行：从 OCR 框下边界往下找最近横线，有则用作新下边界。
    """
    if not found_codes:
        return
    import statistics as _st
    from collections import defaultdict

    # 只对横排框(w>=h)做规整；竖排编号(h>w，如竖写的 YE309C848A)的宽/高语义与
    # 横排相反，宽度归一和首/尾行横线定界会误伤（曾把竖排框上边界压掉切字），故
    # 竖排框原样保留、不参与规整。
    found_codes = [f for f in found_codes if f["bbox"].w >= f["bbox"].h]
    if not found_codes:
        return

    # 1) 宽度归一
    groups = defaultdict(list)
    for f in found_codes:
        groups[_fn_structure(f["code"])].append(f)
    for _pat, items in groups.items():
        # 结构串相同的再按位置/字号/字符数链式聚类，只对同簇(表内同组)归一，
        # 避免表外/远处的同名编号被误拉到同一宽度(w=1234→620 那类误伤)。
        for cluster in _cluster_by_proximity(items):
            if len(cluster) < FN_NORM_MIN_GROUP:
                continue
            med_w = int(_st.median([f["bbox"].w for f in cluster]))
            if med_w <= 0:
                continue
            for f in cluster:
                b = f["bbox"]
                if abs(b.w - med_w) / med_w > FN_NORM_DEV:
                    cx = b.x + b.w / 2.0
                    new_x = int(round(cx - med_w / 2.0))
                    logger.info(f"  框宽归一: {f['code']} w={b.w}→{med_w} "
                                f"(同簇中位)")
                    f["bbox"] = BBox(new_x, b.y, med_w, b.h)

    # 2) 首/尾行横线定界
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    items = sorted(found_codes, key=lambda f: f["bbox"].x + f["bbox"].w / 2.0)
    cols = []
    for f in items:
        cx = f["bbox"].x + f["bbox"].w / 2.0
        if cols and cx - (cols[-1][-1]["bbox"].x
                          + cols[-1][-1]["bbox"].w / 2.0) <= FN_COL_X_TOL:
            cols[-1].append(f)
        else:
            cols.append([f])
    for col in cols:
        if not col:
            continue
        col.sort(key=lambda f: f["bbox"].y)
        first, last = col[0], col[-1]
        b = first["bbox"]
        hy = _fn_local_hline(gray, b, edge_y=b.y)
        if hy is not None and b.y < hy < b.y2:
            logger.info(f"  首行上边界对齐横线: {first['code']} y={b.y}→{hy}")
            first["bbox"] = BBox(b.x, hy, b.w, b.y2 - hy)
        b = last["bbox"]
        hy = _fn_local_hline(gray, b, edge_y=b.y2)
        if hy is not None and hy > b.y2:
            logger.info(f"  尾行下边界对齐横线: {last['code']} y2={b.y2}→{hy}")
            last["bbox"] = BBox(b.x, b.y, b.w, hy - b.y)


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

    fn_top = orange_bbox.y if orange_bbox is not None else 0
    fn_bottom = green_bbox.y if green_bbox is not None else img_h
    fn_left = orange_bbox.x if orange_bbox is not None else 0
    fn_right = green_bbox.x2 if green_bbox is not None else img_w
    # 绿框右边界按图号 OCR 框画时，其右侧可能仍有工厂注意内容被漏在搜索区外：
    # 搜索区右界向右扩「搜索区宽 5%」。仅在有绿框且右边界=OCR 时生效。
    if green_bbox is not None and (regions.get("_metadata", {}) or {}).get("green_right_is_ocr"):
        _ext = int((fn_right - fn_left) * 0.05)
        fn_right = min(fn_right + _ext, img_w)
        logger.info(f"  Factory Note v6: 绿框右边界=OCR，搜索区右界+{_ext}px(搜索区宽5%)")
    if fn_bottom <= fn_top or fn_right <= fn_left:
        logger.warning(f"  Factory Note v6: search_roi 退化, "
                       f"top={fn_top} bottom={fn_bottom} left={fn_left} right={fn_right}")
        return []

    search_roi = image_rgb[fn_top:fn_bottom, fn_left:fn_right].copy()
    sh, sw = search_roi.shape[:2]

    # ── 竖排判定：整片 ROI 跑一次方向分类，判 90/270 即旋转校正 ──
    # 旋转后中间管线（擦边框/表格/V5/VLM）参数完全不变，最后 Y 坐标逆映射回原图。
    # 开关：FN_ROTATE=1 启用；默认关闭（验证不旋转裁切阶段）。
    fn_rot = None
    roi_W0, roi_H0 = sw, sh
    if os.environ.get("FN_ROTATE") == "1":
        try:
            _ang, _sc = _classify_corner_ori(search_roi)
        except Exception as e:  # noqa: BLE001
            _ang, _sc = 0, 0.0
            logger.warning(f"  Factory Note v6: ROI 方向分类失败: {e}")
        if _ang == 270:
            fn_rot = cv2.ROTATE_90_CLOCKWISE
        elif _ang == 90:
            fn_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
    if fn_rot is not None:
        logger.info(f"  Factory Note v6: ROI 竖排(方向={_ang}/{_sc:.2f})，"
                    f"旋转校正后检测")
        search_roi = cv2.rotate(search_roi, fn_rot)
        sh, sw = search_roi.shape[:2]

    def _to_roi(bbox):
        if bbox is None:
            return None
        x1 = max(bbox.x - fn_left, 0)
        y1 = max(bbox.y - fn_top, 0)
        x2 = min(bbox.x2 - fn_left, roi_W0)
        y2 = min(bbox.y2 - fn_top, roi_H0)
        if x2 <= x1 or y2 <= y1:
            return None
        box = (int(x1), int(y1), int(x2), int(y2))
        if fn_rot is not None:
            box = _rot_bbox(box, roi_W0, roi_H0, fn_rot)
        return box

    ref_boxes = {
        "material_code": _to_roi(red_bbox),
        "bottom_right_number": _to_roi(green_bbox),
        "top_left_number": _to_roi(orange_bbox),
    }

    gray = cv2.cvtColor(search_roi, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bw_no_border, _, _ = erase_drawing_border(bw)
    bw = bw_no_border

    # 自动矫正: 小角度倾斜校正(见 _fn_auto_deskew)。search_roi/bw/ref_boxes 同步旋正,
    # 修复倾斜导致表格横线按行统计不达标 → 表格漏检 → 编号识别崩坏(如 994)。
    search_roi, bw, ref_boxes, _fn_ang = _fn_auto_deskew(search_roi, bw, ref_boxes)
    sh, sw = bw.shape
    if abs(_fn_ang) >= 0.05:
        logger.info(f"  Factory Note v6: 自动矫正 {_fn_ang:+.2f}deg")

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
        _, mc_y1_roi, _, mc_y2_roi = mc_roi
        zone_out, _ = cluster_table_zone(
            bw, mc_y1=mc_y1_roi, buffer=TABLE_Y_BUFFER, mc_y2=mc_y2_roi)
        if zone_out is not None:
            table_zone = zone_out

    exclusion_zone = None
    if table_zone is not None:
        _, _, tz_right, tz_bottom = table_zone
        exclusion_zone = (0, 0, int(tz_right), int(tz_bottom))

    # 把 exclusion_zone + 橙框(左上角图号)涂白排除：
    #   1) bw_for_ocr(二值图)涂 0 → 候选不在这些区域出框；
    #   2) search_roi(RGB)物理涂白 255 → OCR 即便被相邻候选 crop 覆盖也读不到图号。
    # 物理涂白与材料/红框替换一致：按 bbox 矩形填 255，内缩 FILL_MARGIN 避免吃边框线。
    # 抹实线: 抹掉搜索区内足够长的实线(横≥宽50% / 竖≥高50%),防止长边框/贯穿线
    # 干扰候选出框与 OCR。用连通域包围盒长度衡量(而非 MORPH_OPEN 的连续像素长),
    # 故斜线/被交叉打断的长线也能整条抹除。仅动 bw,不影响 search_roi(RGB 仍供 OCR)。
    _bh, _bw = bw.shape
    _brd = np.zeros_like(bw)
    _hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)), 1)
    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(_hl, connectivity=8)
    _ix = [i for i in range(1, _n) if _st[i, 2] >= _bw * 0.5]
    if _ix:
        _brd = cv2.bitwise_or(_brd, (np.isin(_lab, _ix).astype('uint8') * 255))
    _vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)), 1)
    _n, _lab, _st, _ = cv2.connectedComponentsWithStats(_vl, connectivity=8)
    _ix = [i for i in range(1, _n) if _st[i, 3] >= _bh * 0.5]
    if _ix:
        _brd = cv2.bitwise_or(_brd, (np.isin(_lab, _ix).astype('uint8') * 255))
    _brd = cv2.dilate(_brd, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    bw = bw.copy()
    bw[_brd > 0] = 0

    bw_for_ocr = bw.copy()
    bh, bw_w = bw_for_ocr.shape
    pad = 3
    FILL_MARGIN = 2

    def _white_out(zone):
        if zone is None:
            return
        zx1, zy1, zx2, zy2 = zone
        # 二值图：阻止候选出框
        bx1 = max(0, zx1 - pad); by1 = max(0, zy1 - pad)
        bx2 = min(bw_w, zx2 + pad); by2 = min(bh, zy2 + pad)
        if bx2 > bx1 and by2 > by1:
            bw_for_ocr[by1:by2, bx1:bx2] = 0
        # RGB 物理涂白：阻止 OCR 读到内容（内缩 FILL_MARGIN，与红框替换一致）
        rx1 = max(0, zx1 + FILL_MARGIN); ry1 = max(0, zy1 + FILL_MARGIN)
        rx2 = min(sw, zx2 - FILL_MARGIN); ry2 = min(sh, zy2 - FILL_MARGIN)
        if rx2 > rx1 and ry2 > ry1:
            search_roi[ry1:ry2, rx1:rx2] = 255

    _white_out(exclusion_zone)
    _white_out(ref_boxes.get("top_left_number"))

    found_codes: list[dict] = []

    for strict in STRICT_LEVELS:
        tables, table_mask, _ = detect_tables(bw, strict_level=strict, zone=table_zone)

        for h_factor in H_FACTORS:
            # 新流程: 先把已检出表格(蓝框)的整个矩形区从 bw_for_ocr 遮盖掉,
            # 再让 process_variant 出文字候选(绿框),避免绿框套住蓝框导致重复识别。
            for _tb in tables:
                _x1, _y1, _x2, _y2 = [int(v) for v in _tb[:4]]
                bw_for_ocr[max(0, _y1):_y2, max(0, _x1):_x2] = 0
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

            # ── 抽样方向投票：在候选里按面积取最大的 FN_ORIENT_VOTE_N 个，
            #    各自丢进方向分类器(非OCR)，竖排(90/270)占比 ≥1/3 则整批旋转。
            #    只旋转小图，候选检测不变；结果坐标再逆旋转回 ROI。
            #    分类器 angle 语义：需逆时针转多少度才正立 → 90 用 CCW、270 用 CW。
            patch_rot = None

            def _clip_box(box):
                bx1, by1, bx2, by2 = [int(v) for v in box[:4]]
                bx1 = max(0, bx1); by1 = max(0, by1)
                bx2 = min(sw, bx2); by2 = min(sh, by2)
                return bx1, by1, bx2, by2

            vote_pool = []
            for _kind, _box in crop_targets:
                cb = _clip_box(_box)
                if cb[2] - cb[0] < 6 or cb[3] - cb[1] < 6:
                    continue
                vote_pool.append(cb)
            # 未旋转 crop 的 v5 结果缓存，键=clip 后 box；主循环在不旋转时复用
            _v5_cache: dict[tuple[int, int, int, int], list] = {}
            if FN_ORIENT_VOTE_BY_TEXT:
                # 按 v5 文字框数取前 N（图形块框数少→不进投票，避免误翻）。
                # 此处对每候选跑一次 v5 并缓存；若最终判定不旋转，主循环直接
                # 复用缓存（正向图零额外开销）；判定要旋转时主循环对旋转后
                # crop 重跑（方向已变，本躲不掉）。
                _scored = []
                for cb in vote_pool:
                    try:
                        _, _it = _v5_run(Image.fromarray(
                            search_roi[cb[1]:cb[3], cb[0]:cb[2]]))
                        _v5_cache[cb] = _it
                        _nb = sum(1 for _p, _t, _s in _it
                                  if _p is not None and (_t or "").strip())
                    except Exception:  # noqa: BLE001
                        _nb = 0
                    _scored.append((_nb, (cb[2] - cb[0]) * (cb[3] - cb[1]), cb))
                _scored.sort(key=lambda z: (z[0], z[1]), reverse=True)
                vote_pool = [z[2] for z in _scored[:FN_ORIENT_VOTE_N]]
            else:
                vote_pool.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                               reverse=True)
                vote_pool = vote_pool[:FN_ORIENT_VOTE_N]

            n_total = 0
            n_vert = 0
            vote_cw = 0
            vote_ccw = 0
            for bx1, by1, bx2, by2 in vote_pool:
                try:
                    _a, _s = _classify_corner_ori(search_roi[by1:by2, bx1:bx2])
                except Exception:  # noqa: BLE001
                    _a, _s = 0, 0.0
                n_total += 1
                if _a == 90:
                    n_vert += 1; vote_ccw += 1
                elif _a == 270:
                    n_vert += 1; vote_cw += 1
            if n_total > 0 and n_vert / n_total >= 1.0 / 3.0:
                patch_rot = (cv2.ROTATE_90_CLOCKWISE if vote_cw >= vote_ccw
                             else cv2.ROTATE_90_COUNTERCLOCKWISE)
                logger.info(
                    f"  Factory Note v6 [{strict}|h={h_factor}]: "
                    f"竖排候选 {n_vert}/{n_total} (≥1/3)，整批旋转"
                    f"{'CW' if patch_rot == cv2.ROTATE_90_CLOCKWISE else 'CCW'} 后 OCR")

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
                crop_W0, crop_H0 = x2 - x1, y2 - y1
                # 抽样投票判为竖排 → 旋转该小图后再 OCR；poly 坐标稍后逆旋转回 crop
                crop_ocr = cv2.rotate(crop, patch_rot) if patch_rot is not None else crop
                pil_crop = Image.fromarray(crop_ocr)

                def _poly_to_roi_bbox(px1, py1, px2, py2):
                    # rotated-crop 坐标 → (逆 patch_rot) → crop 局部 → 加偏移到 ROI
                    cb = _rot_bbox((px1, py1, px2, py2),
                                   crop_W0, crop_H0, patch_rot, inv=True)
                    if cb is None:
                        cb = (px1, py1, px2, py2)
                    return (cb[0] + x1, cb[1] + y1, cb[2] + x1, cb[3] + y1)

                # 不旋转时复用投票段对同一(未旋转)crop 的 v5 结果，避免重复 OCR
                _ck = (x1, y1, x2, y2)
                if patch_rot is None and _ck in _v5_cache:
                    items = _v5_cache[_ck]
                else:
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
                    gbox = _poly_to_roi_bbox(px1, py1, px2, py2)
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
                    # rotated-crop poly → (逆 patch_rot + 偏移) → ROI 坐标系
                    rx1, ry1, rx2, ry2 = _poly_to_roi_bbox(
                        int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                    # 逆旋转回原始 ROI 坐标系（fn_rot 整片竖排校正）
                    rb = _rot_bbox((rx1, ry1, rx2, ry2),
                                   roi_W0, roi_H0, fn_rot, inv=True)
                    ox1, oy1, ox2, oy2 = rb if rb is not None else (rx1, ry1, rx2, ry2)
                    # 映射回整图坐标
                    bx1 = ox1 + fn_left
                    by1 = oy1 + fn_top
                    bx2 = ox2 + fn_left
                    by2 = oy2 + fn_top
                    w_px = max(bx2 - bx1, 1)
                    h_px = max(by2 - by1, 1)
                    found_codes.append({
                        "code": _tk,
                        "bbox": BBox(bx1, by1, w_px, h_px),
                        "confidence": 1.0,
                        # patch_rot：检测时把竖排 crop 转正供 OCR 的旋转码。
                        # 替换端渲染横排文字后按其逆旋转贴回，匹配原图竖排方向。
                        "orientation": patch_rot,
                    })

            logger.info(
                f"  Factory Note v6 [{strict}|h={h_factor}]: "
                f"tables={len(tables)} boxes={len(tight)} "
                f"v5_run={v5_total}/{len(crop_targets)} "
                f"v5_y_hit={v5_kept} vlm_drawn={len(found_codes)}")

    # 框规整：宽度归一 + 首/尾行横线定上下边界（工厂注意专用）
    _normalize_found_codes(found_codes, image_rgb)

    return found_codes
