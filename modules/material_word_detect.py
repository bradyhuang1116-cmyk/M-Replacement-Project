"""材料列词级(字符级 return_word_box)检测。

主策略: 词级 OCR 逐格框选 + snap 贴合 + 删除线 inpaint + 首/末行补格。
兜底: 原 detect_cyan_boxes(其自带双策略)。任一失败分支都回退, 保证不劣于原策略。
不含 deskew —— 旋转/弯曲由上游整体矫正(modules.dewarp)处理, 此处只管词级识别。

云模式(VLM_PROVIDER=='paddleocr_api')无本地 GPU, 直接强制走兜底, 不构造本地 PaddleOCR。
"""
import logging

import numpy as np
import cv2

from modules.factory_note_pixel import _find_y_token
from modules.text_replacer import (
    _preprocess_for_table, _detect_all_hlines_projection,
    _filter_strikes_by_text_overlap, _classify_lines_by_grid,
)
from modules.region_detector import BBox

logger = logging.getLogger(__name__)

_WORD_OCR = None


def _word_ocr_available():
    """云模式(paddleocr_api)无本地 GPU → 不可用, 强制兜底。"""
    try:
        from config import VLM_PROVIDER
    except Exception:  # noqa: BLE001
        return True
    return VLM_PROVIDER != "paddleocr_api"


def _get_word_ocr():
    """lazy 单例; return_word_box=True 是独立配置, 无法复用 _get_ocr_v5。"""
    global _WORD_OCR
    if _WORD_OCR is None:
        from paddleocr import PaddleOCR
        _WORD_OCR = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                              use_doc_unwarping=False, use_textline_orientation=False,
                              return_word_box=True)
    return _WORD_OCR


def _snap_word_bbox(cell_rgb, bx1, by1, bx2, by2):
    """源码 _snap_to_runs 本地分支: 取重叠笔画 min/max, 外扩≤1字宽。"""
    sub = cell_rgb[by1:by2, :]
    g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY) if sub.ndim == 3 else sub
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs = (bw > 0).sum(axis=0)
    runs = []
    in_r = False
    rs = 0
    for c in range(len(cs)):
        if cs[c] > 0 and not in_r:
            rs = c
            in_r = True
        elif cs[c] == 0 and in_r:
            runs.append((rs, c))
            in_r = False
    if in_r:
        runs.append((rs, len(cs)))
    ov = [(rs, re_) for rs, re_ in runs if re_ > bx1 and rs < bx2]
    if not ov:
        return bx1, bx2
    char_w = max(8, (bx2 - bx1) // 12)
    nl = max(min(rs for rs, _ in ov), bx1 - char_w)
    nr = min(max(re_ for _, re_ in ov), bx2 + char_w)
    return nl, nr


def make_word_detect_cyan(orig_detect_cyan):
    """返回 detect_cyan_boxes 的替身: 词级为主, orig 兜底。签名与源码一致。"""

    def _word_detect_cyan(image, bbox, row_ys, pattern=None, prefixes=None):
        from config import DEFAULT_PREFIXES
        prefixes = prefixes or DEFAULT_PREFIXES
        FILL_MARGIN = 2

        def _fallback(reason):
            logger.info(f"[词级材料列] 兜底原策略: {reason}")
            return orig_detect_cyan(image, bbox, row_ys, pattern=pattern, prefixes=prefixes)

        if not _word_ocr_available():
            return _fallback("云模式无本地词级OCR")

        try:
            box_roi = image[bbox.y:bbox.y2, bbox.x:bbox.x2]
            box_gray = cv2.cvtColor(box_roi, cv2.COLOR_RGB2GRAY)
            box_h, box_w = box_gray.shape[:2]
            binary = _preprocess_for_table(box_gray)
            all_lines = _detect_all_hlines_projection(binary, min_line_ratio=0.8)
            if len(all_lines) < 2:
                return _fallback("投影法行线不足")
            all_lines, text_strikes = _filter_strikes_by_text_overlap(binary, all_lines)

            gaps = [all_lines[i + 1] - all_lines[i] for i in range(len(all_lines) - 1)]
            large_gaps = [g for g in gaps if g > 50]
            if not large_gaps:
                return _fallback("无有效行间隙")
            _med0 = sorted(large_gaps)[len(large_gaps) // 2]
            base_gaps = [g for g in large_gaps if 0.6 * _med0 <= g <= 1.4 * _med0] or large_gaps
            base_s = sorted(base_gaps)
            cell_height = base_s[len(base_s) // 2]
            _mean = sum(base_gaps) / len(base_gaps)
            _std = (sum((g - _mean) ** 2 for g in base_gaps) / len(base_gaps)) ** 0.5
            cv = _std / _mean if _mean else 0.0
            tol = 3 if cv < 0.05 else max(3, int(cell_height * min(cv, 0.2)))
            table_lines, strike_lines = _classify_lines_by_grid(all_lines, cell_height, tolerance=tol)
            if text_strikes:
                strike_lines = sorted(set(strike_lines) | set(text_strikes))
            logger.info(f"[词级材料列] 行线{len(table_lines)} 删除线{len(strike_lines)} cell_h={cell_height}")

            # 删除线 inpaint (整列 mask → inpaint → 逐格词级OCR)
            if strike_lines:
                combined_mask = np.zeros((box_h, box_w), dtype=np.uint8)
                _, thresh_full = cv2.threshold(box_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kw = max(int(box_w * 0.25), 15)
                h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
                for sy in strike_lines:
                    ytop = max(0, sy - 6)
                    ybot = min(box_h, sy + 7)
                    combined_mask[ytop:ybot, :] = cv2.morphologyEx(
                        thresh_full[ytop:ybot, :], cv2.MORPH_OPEN, h_kernel)
                if np.any(combined_mask):
                    box_bgr = cv2.cvtColor(box_roi, cv2.COLOR_RGB2BGR)
                    box_roi = cv2.cvtColor(
                        cv2.inpaint(box_bgr, combined_mask, 2, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                    logger.info(f"[词级材料列] inpaint删除线 {len(strike_lines)}条")

            x, y_img = bbox.x, bbox.y
            cyan_boxes, cyan_box_data = [], []
            ocr_w = _get_word_ocr()
            # 补首/末格: 第一条行线上方、最后一条行线到列底, 若够一格高则补格。
            #   首行上边界常是表格顶边(未检为row_y) → 漏首行(如首行YE205B042);
            #   末行下边界常是表格底边(未检为row_y) → 漏末行(如X39HA-143)。
            bounds = list(row_ys)
            min_cell = max(15, int(cell_height * 0.5))
            if bounds and box_h - bounds[-1] >= min_cell:
                bounds.append(box_h)
            if bounds and bounds[0] >= min_cell:
                bounds.insert(0, 0)
            for i in range(len(bounds) - 1):
                cy1, cy2 = bounds[i], bounds[i + 1]
                if cy2 - cy1 < 15:
                    continue
                cell = box_roi[cy1:cy2, :]
                res = ocr_w.predict(cell)
                if not res or not res[0]:
                    continue
                r0 = res[0]
                tw = r0.get("text_word", [[]])[0] if r0.get("text_word") else []
                tr_reg = r0.get("text_word_region", [[]])[0] if r0.get("text_word_region") else []
                for wd, reg in zip(tw, tr_reg):
                    tok = _find_y_token(wd)
                    if not tok:
                        continue
                    pts = [(int(p[0]), int(p[1])) for p in reg]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    lx, ly, rx, ry = min(xs), min(ys), max(xs), max(ys)
                    nl, nr = _snap_word_bbox(cell, lx, ly, rx, ry)
                    gw, gh = nr - nl, ry - ly
                    has_s = any(cy1 + cell_height * 0.15 <= sy <= cy2 - cell_height * 0.15
                                for sy in strike_lines)
                    pre_mx = max(int(round(gw * 0.025)), FILL_MARGIN)
                    pre_my_top = max(int(round(gh * 0.045)), FILL_MARGIN)
                    pre_my_bot = max(int(round(gh * 0.035)), FILL_MARGIN)
                    cb = BBox(x + nl - pre_mx, y_img + cy1 + ly - pre_my_top,
                              gw + 2 * pre_mx, gh + pre_my_top + pre_my_bot)
                    item = {"bbox": cb, "text": tok, "pattern": pattern}
                    if has_s:
                        item["has_strikethrough"] = True
                    cyan_boxes.append(cb)
                    cyan_box_data.append(item)

            # 词级零命中 → 兜底(可能词级不适配此表, 原策略更稳)
            if not cyan_boxes:
                return _fallback("词级零命中")
            logger.info(f"[词级材料列] 命中 {len(cyan_boxes)} 个编号")
            return cyan_boxes, cyan_box_data
        except Exception as e:  # noqa: BLE001
            return _fallback(f"异常 {e!r}")

    return _word_detect_cyan
