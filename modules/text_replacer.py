"""文本替换引擎 — OCR 识别 + 像素级替换（支持多前缀）"""

import os
import re
import logging
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Y_PATTERN, FONT_PATH, OCR_LANG_EN, OCR_LANG_CH, DEFAULT_REGIONS, make_pattern, DEFAULT_PREFIXES, OCR_MODE
from modules.region_detector import BBox, _pct_to_px, _detect_horizontal_lines, _detect_horizontal_lines_adaptive, _ocr_region
from modules.factory_note_pixel import record_y_box

logger = logging.getLogger(__name__)

# ── OCR 引擎（VLM 单例，线程安全）──────────────────────────────

from modules.vlm_ocr_engine import get_vlm_engine, clear_vlm_engine
import gc


def clear_ocr_cache():
    """释放 OCR 引擎缓存（v5 + VLM）。"""
    from modules.region_detector import _v5_cache
    _v5_cache.clear()
    gc.collect()
    logger.info("OCR 缓存已清除（v5 + VLM）")


def _get_ocr(lang: str = "en"):
    """返回 VLM OCR 引擎（lang 参数保留兼容，VLM 本身多语言）。"""
    return get_vlm_engine()


# ── OCR 结果解析 ──────────────────────────────────────────────


def _parse_ocr_results(result):
    """
    从 OCR 引擎返回值中提取 (polygon, text, confidence) 列表。
    兼容不同版本的返回结构。
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


# ── 统一单元格高度确定 ─────────────────────────────────────────


def _determine_uniform_cell_height(row_ys: list[int]) -> int | None:
    """从 row_ys 的前5对相邻间距中投票确定统一单元格高度。

    纯粹基于横线位置计算，不依赖 OCR 结果。

    Returns:
        统一单元格高度(px)，失败时返回 None
    """
    if len(row_ys) < 2:
        return None

    # 计算相邻间距（最多取前5对）
    gaps = [row_ys[i + 1] - row_ys[i] for i in range(min(len(row_ys) - 1, 5))]
    gaps = [g for g in gaps if g >= 30]  # 过滤双线/表头边界噪声（单元格至少30px高）

    if not gaps:
        return None

    if len(gaps) == 1:
        return gaps[0]

    # ±10px 容差分组，多数投票
    groups = {}
    for h in gaps:
        placed = False
        for key in groups:
            if abs(h - key) <= 10:
                groups[key].append(h)
                placed = True
                break
        if not placed:
            groups[h] = [h]

    best_group = max(groups.values(), key=len)
    result = int(np.mean(best_group))

    logger.info(f"  单元格高度采样: {gaps} → 统一高度={result}px")
    return result


def _filter_strikethrough_lines(row_ys: list[int]) -> list[int]:
    """过滤行线列表中由删除线造成的假行线。

    删除线被误检为行线时，表现为：正常间距(~65px)中突然出现
    一组极小间距(如 41+8+16=65)。

    策略：确定主间距(uniform_cell_h)，扫描行线列表，
    遇到间距 < uniform_cell_h × 0.6 时，说明当前行线是删除线产生的假行线，
    跳过它和后续所有间距过小的行线，直到找到下一条距离上一真实行线
    约 uniform_cell_h 的行线。
    """
    if len(row_ys) < 4:
        return row_ys

    uniform_h = _determine_uniform_cell_height(row_ys)
    if not uniform_h or uniform_h < 10:
        return row_ys

    min_gap = uniform_h * 0.7
    filtered = [row_ys[0]]

    i = 1
    while i < len(row_ys):
        gap = row_ys[i] - filtered[-1]
        if gap >= min_gap:
            # 正常间距，保留
            filtered.append(row_ys[i])
            i += 1
        else:
            # 间距过小 → 删除线假行线
            # 从 filtered[-1]（真实行线）往后扫描，跳过所有假行线，
            # 直到找到一条距离 filtered[-1] 在 [uniform_h - tolerance, ...] 范围的行线
            anchor = filtered[-1]
            j = i
            while j < len(row_ys):
                dist = row_ys[j] - anchor
                if dist >= min_gap:
                    # 这条行线距离足够远，是下一条真实行线
                    filtered.append(row_ys[j])
                    j += 1
                    break
                j += 1
            i = j

    removed = len(row_ys) - len(filtered)
    if removed > 0:
        logger.info(f"  删除线行线过滤: {len(row_ys)}条 → {len(filtered)}条 (移除{removed}条假行线)")
    return filtered


def detect_row_ys_for_red_box(
    image: np.ndarray, bbox: BBox, table_search_bbox: BBox = None,
) -> list[int]:
    """检测红框内的行分界线 y 坐标（相对于 bbox 内部）。

    策略：
    1. 先在宽表格区域检测（跨表的长横线）
    2. 补充在红框列内直接检测（列内短横线）
    3. 合并去重，取数量更多、间距更合理的结果
    """
    from modules.region_detector import _detect_horizontal_lines_adaptive

    wide_row_ys = []
    if table_search_bbox:
        table_gray = cv2.cvtColor(table_search_bbox.crop(image), cv2.COLOR_RGB2GRAY)
        bbox_y_in_table = bbox.y - table_search_bbox.y
        all_h = _detect_horizontal_lines_adaptive(
            table_gray, bbox_y_in_table, bbox.h, min_line_width=18)
        wide_row_ys = sorted([
            y - bbox_y_in_table for y in all_h
            if bbox_y_in_table <= y <= bbox_y_in_table + bbox.h
        ])

    # 直接在红框列内检测短横线（投影法）
    red_gray = cv2.cvtColor(bbox.crop(image), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(red_gray, 180, 255, cv2.THRESH_BINARY_INV)
    # 形态学开运算：核宽 = 列宽的 1/3（只保留较长的横线）
    kernel_w = max(bbox.w // 3, 10)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    proj = np.sum(h_lines, axis=1)
    threshold_px = bbox.w * 0.3
    ys_raw = np.where(proj > threshold_px)[0]
    # 合并相邻像素
    narrow_row_ys = []
    for y in ys_raw:
        if not narrow_row_ys or y - narrow_row_ys[-1] > 5:
            narrow_row_ys.append(int(y))

    logger.info(
        f"  行检测: 宽表格={len(wide_row_ys)}条, 红框直接={len(narrow_row_ys)}条"
    )

    # 选择更好的结果：行数更多且间距合理的一方
    # 如果红框直接检测的行数明显多于宽表格（>1.5倍），优先用红框检测
    if len(narrow_row_ys) > len(wide_row_ys) * 1.5 and len(narrow_row_ys) >= 6:
        logger.info(f"  → 使用红框直接检测结果 ({len(narrow_row_ys)}条)")
        return _filter_strikethrough_lines(narrow_row_ys)

    # 否则合并两组，去重（±8px 视为同一条线）
    if wide_row_ys and narrow_row_ys:
        merged = sorted(set(wide_row_ys + narrow_row_ys))
        deduped = [merged[0]]
        for y in merged[1:]:
            if y - deduped[-1] > 8:
                deduped.append(y)
        if len(deduped) > len(wide_row_ys) * 1.2:
            logger.info(f"  → 使用合并结果 ({len(deduped)}条)")
            return _filter_strikethrough_lines(deduped)

    # 默认使用宽表格结果
    result = wide_row_ys if wide_row_ys else narrow_row_ys
    logger.info(f"  → 使用{'宽表格' if wide_row_ys else '红框直接'}结果 ({len(result)}条)")

    # ── 过滤删除线产生的假行线 ──────────────────────────────────────
    # 删除线是单元格中间的水平线，会被误检为行分界线，
    # 表现为：正常间距(~65px) 突然出现一组小间距(41+8+16=65)。
    # 策略：先确定 uniform_cell_h，然后合并间距过小的连续行线。
    result = _filter_strikethrough_lines(result)
    return result


# ── Step 2: 表格线保护 + 删除线检测 ─────────────────────────────


def _detect_strikethrough_mask(cell_gray: np.ndarray) -> np.ndarray:
    """检测单元格内的删除线（strikethrough），返回二值掩膜。

    删除线特征：水平线条，位于单元格中间区域（15%~85%高度），
    比表格边框线细（<=4px），横跨单元格。

    返回 uint8 掩膜，255=删除线像素，0=其他。
    """
    cell_h, cell_w = cell_gray.shape[:2]
    if cell_h < 10 or cell_w < 10:
        return np.zeros_like(cell_gray)

    # Otsu 二值化
    _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学开运算提取水平线（核宽 = 单元格宽的 40%，只保留足够长的线）
    kernel_w = max(int(cell_w * 0.4), 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 找到所有水平线轮廓
    contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(cell_gray)

    # 中间区域范围（排除顶部/底部的表格边框线）
    y_min_zone = int(cell_h * 0.15)
    y_max_zone = int(cell_h * 0.85)

    strike_mask = np.zeros_like(cell_gray)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        cy = y + bh // 2
        # 删除线条件：在中间区域 且 线条不太粗（<=4px）
        if y_min_zone <= cy <= y_max_zone and bh <= 4:
            cv2.drawContours(strike_mask, [c], -1, 255, -1)

    # 上下膨胀 1px，覆盖删除线边缘
    if np.any(strike_mask):
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        strike_mask = cv2.dilate(strike_mask, dilate_k, iterations=1)

    return strike_mask


def _extract_line_mask(roi_gray: np.ndarray, h_kernel_w: int = 30, v_kernel_h: int = 15) -> np.ndarray:
    """提取水平+垂直线条的二值掩膜，用于替换后恢复线条像素。

    返回 uint8 掩膜，255=线条像素，0=非线条。
    """
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 垂直线
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # 合并 + 膨胀1px覆盖抗锯齿边缘
    combined = cv2.bitwise_or(h_mask, v_mask)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.dilate(combined, dilate_kernel, iterations=1)

    return combined


def _restore_protected_pixels(modified: np.ndarray, original: np.ndarray,
                              line_mask: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    """将线条区域的像素从原图恢复到修改后的图上。

    line_mask 是 ROI 尺寸的掩膜，offset_x/offset_y 是 ROI 在全图中的偏移。
    """
    mask_h, mask_w = line_mask.shape[:2]
    y1, y2 = offset_y, offset_y + mask_h
    x1, x2 = offset_x, offset_x + mask_w

    # 边界安全检查
    img_h, img_w = modified.shape[:2]
    if y2 > img_h or x2 > img_w:
        mask_h = min(mask_h, img_h - offset_y)
        mask_w = min(mask_w, img_w - offset_x)
        y2 = offset_y + mask_h
        x2 = offset_x + mask_w
        line_mask = line_mask[:mask_h, :mask_w]

    roi_mod = modified[y1:y2, x1:x2]
    roi_orig = original[y1:y2, x1:x2]
    mask_bool = line_mask > 0

    if len(roi_mod.shape) == 3:
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)
        roi_mod[mask_3ch] = roi_orig[mask_3ch]
    else:
        roi_mod[mask_bool] = roi_orig[mask_bool]

    return modified


# ── Step 1: 像素级文字擦除 ──────────────────────────────────────


def _create_text_mask(roi_gray: np.ndarray, cell_x: int, cell_y: int,
                      cell_w: int, cell_h: int,
                      line_mask: np.ndarray = None,
                      shrink_px: int = 2) -> np.ndarray:
    """创建文字区域的精细掩膜，仅标记文字像素。

    返回与 roi_gray 同尺寸的掩膜（255=需擦除的文字像素）。
    失败时回退为实心矩形。
    """
    h, w = roi_gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 安全裁剪 cell 区域
    cy1, cy2 = max(cell_y, 0), min(cell_y + cell_h, h)
    cx1, cx2 = max(cell_x, 0), min(cell_x + cell_w, w)
    if cy2 <= cy1 or cx2 <= cx1:
        # 区域无效，回退为实心矩形
        cv2.rectangle(mask, (cx1, cy1), (cx2, cy2), 255, -1)
        return mask

    cell_roi = roi_gray[cy1:cy2, cx1:cx2]

    # 自适应阈值提取前景（文字+线条）
    cell_mask = cv2.adaptiveThreshold(
        cell_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=5
    )

    # 减去线条掩膜（如果有）
    if line_mask is not None:
        line_sub = line_mask[cy1:cy2, cx1:cx2]
        cell_mask = cv2.subtract(cell_mask, line_sub)

    # 形态学闭合填补字符间隙
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 腐蚀收缩，避免触碰相邻线条
    if shrink_px > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px * 2 + 1, shrink_px * 2 + 1))
        cell_mask = cv2.erode(cell_mask, erode_kernel, iterations=1)

    # 过滤小连通域（噪声）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cell_mask, connectivity=8)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 5:
            cell_mask[labels == label_id] = 0

    # 检查是否提取到文字像素
    if cv2.countNonZero(cell_mask) == 0:
        # 回退：实心矩形
        cv2.rectangle(mask, (cx1, cy1), (cx2, cy2), 255, -1)
        return mask

    mask[cy1:cy2, cx1:cx2] = cell_mask
    return mask


def _erase_text_pixels(image: np.ndarray, text_mask: np.ndarray,
                       offset_x: int, offset_y: int,
                       method: str = "white") -> np.ndarray:
    """根据文字掩膜擦除图像中的文字像素。

    method: "white" 直接置白, "inpaint" 用周围像素填充
    """
    mask_h, mask_w = text_mask.shape[:2]
    y1, y2 = offset_y, offset_y + mask_h
    x1, x2 = offset_x, offset_x + mask_w

    img_h, img_w = image.shape[:2]
    y2 = min(y2, img_h)
    x2 = min(x2, img_w)
    actual_mask = text_mask[:y2 - y1, :x2 - x1]

    if method == "inpaint":
        roi = image[y1:y2, x1:x2].copy()
        inpainted = cv2.inpaint(roi, actual_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        image[y1:y2, x1:x2] = inpainted
    else:
        # white fill on masked pixels only
        mask_bool = actual_mask > 0
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                image[y1:y2, x1:x2, c][mask_bool] = 255
        else:
            image[y1:y2, x1:x2][mask_bool] = 255

    return image


# ── Step 3: 原图字体特征采样 ────────────────────────────────────


def _estimate_stroke_thickness(char_img_gray: np.ndarray) -> float:
    """通过距离变换估算笔画粗细。"""
    _, binary = cv2.threshold(char_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    non_zero = dist[dist > 0]
    if len(non_zero) == 0:
        return 1.5  # 默认值
    return float(np.median(non_zero)) * 2


def _sample_font_features(roi_gray: np.ndarray,
                          matched_items: list,
                          sample_count: int = 2) -> dict | None:
    """从已匹配的 OCR 文本中采样字体特征。

    返回 {"char_height", "stroke_thickness", "grayscale_value", "char_spacing"} 或 None。
    """
    # 挑选合适的采样项（h > 8, w > 5）
    candidates = [(t, x, y, w, h) for t, x, y, w, h in matched_items if h > 8 and w > 5]
    if not candidates:
        return None

    candidates = candidates[:sample_count]

    heights = []
    strokes = []
    grayscales = []
    spacings = []

    for text, x, y, w, h in candidates:
        # 安全裁剪
        roi_h, roi_w = roi_gray.shape[:2]
        cy1 = max(y, 0)
        cy2 = min(y + h, roi_h)
        cx1 = max(x, 0)
        cx2 = min(x + w, roi_w)
        if cy2 <= cy1 or cx2 <= cx1:
            continue

        char_roi = roi_gray[cy1:cy2, cx1:cx2]

        # 字符高度
        heights.append(h)

        # 笔画粗细
        strokes.append(_estimate_stroke_thickness(char_roi))

        # 文字灰度（提取暗像素）
        _, binary = cv2.threshold(char_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_pixels = char_roi[binary > 0]
        if len(text_pixels) > 0:
            grayscales.append(float(np.mean(text_pixels)))

        # 字符间距（通过连通域分析）
        if len(text) > 1:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels > 2:  # 背景 + 至少2个字符
                # 取连通域中心 x 坐标，排序后计算间距
                cx_list = sorted([centroids[i][0] for i in range(1, num_labels)
                                  if stats[i, cv2.CC_STAT_AREA] > 3])
                if len(cx_list) >= 2:
                    gaps = [cx_list[j + 1] - cx_list[j] for j in range(len(cx_list) - 1)]
                    spacings.append(float(np.mean(gaps)))

    if not heights:
        return None

    features = {
        "char_height": int(np.mean(heights)),
        "stroke_thickness": float(np.mean(strokes)) if strokes else 1.5,
        "grayscale_value": int(np.mean(grayscales)) if grayscales else 40,
        "char_spacing": float(np.mean(spacings)) if spacings else None,
    }

    logger.info(f"  字体特征采样: height={features['char_height']}, "
                f"stroke={features['stroke_thickness']:.1f}, "
                f"gray={features['grayscale_value']}, "
                f"spacing={features['char_spacing']}")
    return features


# ── Step 4: 双维度约束渲染辅助 ──────────────────────────────────

_font_size_cache = {}


def _find_font_size_for_height(font_path: str, target_height_px: int, text: str = "H") -> int:
    """二分查找匹配目标像素高度的字号。"""
    cache_key = (font_path, target_height_px)
    if cache_key in _font_size_cache:
        return _font_size_cache[cache_key]

    lo, hi = 6, 200
    best_size = 16

    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except (IOError, OSError):
            break
        bb = draw.textbbox((0, 0), text, font=font)
        rendered_h = bb[3] - bb[1]

        if rendered_h <= target_height_px:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    _font_size_cache[cache_key] = best_size
    return best_size


# ── Step 5: 后处理灰度一致性 ────────────────────────────────────


def _check_and_adjust_grayscale(modified: np.ndarray,
                                text_rect: tuple,
                                target_grayscale: int,
                                threshold: float = 15.0) -> np.ndarray:
    """检查并调整新渲染文字的灰度值，使其与原图文字一致。

    text_rect: (x, y, w, h) 全图坐标
    """
    gx, gy, gw, gh = text_rect
    img_h, img_w = modified.shape[:2]

    y1, y2 = max(gy, 0), min(gy + gh, img_h)
    x1, x2 = max(gx, 0), min(gx + gw, img_w)
    if y2 <= y1 or x2 <= x1:
        return modified

    roi = modified[y1:y2, x1:x2]

    # 转灰度
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi.copy()

    # 提取文字像素（非白色）
    text_mask = gray < 200
    if not np.any(text_mask):
        return modified

    current_mean = float(np.mean(gray[text_mask]))

    deviation = abs(current_mean - target_grayscale)
    if deviation <= threshold:
        return modified

    # 调整灰度
    if current_mean < 1:
        return modified

    ratio = target_grayscale / current_mean

    if len(roi.shape) == 3:
        for c in range(3):
            channel = roi[:, :, c].astype(np.float32)
            text_pixels = text_mask
            channel[text_pixels] = np.clip(channel[text_pixels] * ratio, 0, 255)
            roi[:, :, c] = channel.astype(np.uint8)
    else:
        roi_f = roi.astype(np.float32)
        roi_f[text_mask] = np.clip(roi_f[text_mask] * ratio, 0, 255)
        modified[y1:y2, x1:x2] = roi_f.astype(np.uint8)

    return modified


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
    text: str, target_w: int, target_h: int, font_path: str = None,
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

    text_color = (0, 0, 0, 255)

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
    draw.text((-bb[0], -bb[1]), text, fill=text_color, font=font)

    # 等比例缩放
    scale_h = target_h / text_h
    scale_w = target_w / text_w
    scale = min(scale_h, scale_w)

    new_w = max(int(text_w * scale), 1)
    new_h = max(int(text_h * scale), 1)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 居中放置在 target_w × target_h 画布上
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2  # 垂直居中
    canvas.paste(img, (paste_x, paste_y), img)
    return canvas


def replace_y_in_region_pixel(
    image: np.ndarray, bbox: BBox, lang: str = OCR_LANG_EN,
    use_grid_alignment: bool = False,
    row_ys: list[int] = None,
    pattern: str = None,
    prefixes: list[str] = None,
    cyan_box_data: list[dict] = None,
    original: np.ndarray = None,
) -> tuple[np.ndarray, list]:
    """
    在指定区域内 OCR 识别并像素级替换（前缀加H）。

    三种模式：
    - 数据驱动模式 (cyan_box_data 不为空): 使用预检测的 OCR 结果直接替换
    - 网格对齐模式 (use_grid_alignment=True): 逐格扫描红框内单元格
    - 直接模式: 整体 OCR 后按文字 bbox 替换

    original: 原始未修改图像（数据驱动模式下用于删除线恢复）。
              传入时跳过内部copy，直接操作 image（调用者须保证 image 已是副本）。

    返回 (修改后的全图, [(old_text, new_text), ...], cyan_boxes)
    """
    # 数据驱动模式 + 提供 original → 跳过copy（调用者已copy）
    if original is not None and use_grid_alignment and cyan_box_data:
        modified = image
        orig_ref = original
    else:
        modified = image.copy()
        orig_ref = image
    replacements = []
    cyan_boxes = []
    prefixes = prefixes or DEFAULT_PREFIXES
    pat = re.compile(pattern if pattern else make_pattern(prefixes))
    FILL_MARGIN = 2

    if use_grid_alignment and cyan_box_data:
        # ── 数据驱动模式：批量操作，减少 numpy↔PIL 转换 ──
        text_paste_data = []    # (text_img, paste_x, paste_y)
        strike_restore_data = []  # (mask, gx, gy_fill)

        for item in cyan_box_data:
            cell_bbox = item["bbox"]
            old_text = item["text"]
            has_strike = item["has_strikethrough"]
            gx, gy_fill = cell_bbox.x, cell_bbox.y
            safe_w, fill_h = cell_bbox.w, cell_bbox.h
            new_text = "H" + old_text

            # 删除线：从原始图检测掩膜
            if has_strike:
                cell_gray = cv2.cvtColor(
                    orig_ref[gy_fill:gy_fill + fill_h, gx:gx + safe_w],
                    cv2.COLOR_RGB2GRAY,
                )
                strike_mask = _detect_strikethrough_mask(cell_gray)
                if strike_mask is not None and np.any(strike_mask):
                    strike_restore_data.append((strike_mask, gx, gy_fill))

            # 白填充 (numpy)
            cv2.rectangle(modified,
                (gx + FILL_MARGIN, gy_fill + FILL_MARGIN),
                (gx + safe_w - FILL_MARGIN, gy_fill + fill_h - FILL_MARGIN),
                (255, 255, 255), -1)

            # 渲染新文字（收集，稍后批量粘贴）
            render_w = max(safe_w - 2 * FILL_MARGIN, 6)
            render_h = max(fill_h - 2 * FILL_MARGIN, 6)
            text_img = _render_text_distributed(new_text, render_w, render_h)
            text_paste_data.append((text_img, gx + FILL_MARGIN, gy_fill + FILL_MARGIN))

            cyan_boxes.append(cell_bbox)
            replacements.append((old_text, new_text))
            logger.info(f"  替换(预检测): {old_text} → {new_text}")

        # 先恢复删除线（在白底之上）
        for strike_mask, gx, gy_fill in strike_restore_data:
            _restore_protected_pixels(modified, orig_ref, strike_mask, gx, gy_fill)
            logger.info(f"  删除线已恢复: ({gx}, {gy_fill})")

        # 再批量粘贴文字（在删除线之上，文字不被遮挡）
        if text_paste_data:
            pil_modified = Image.fromarray(modified)
            for text_img, px, py in text_paste_data:
                pil_modified.paste(text_img, (px, py), text_img)
            modified = np.array(pil_modified)

    elif use_grid_alignment and row_ys and len(row_ys) >= 2:
        # ── 网格对齐模式：逐格扫描 ──
        if row_ys is None:
            roi_gray = cv2.cvtColor(bbox.crop(image), cv2.COLOR_RGB2GRAY)
            row_ys = _detect_row_boundaries(roi_gray, scale_factor=1.0)

        uniform_cell_h = _determine_uniform_cell_height(row_ys)
        if not uniform_cell_h:
            logger.warning("  无法确定单元格高度, 跳过网格替换")
            return modified, replacements, cyan_boxes

        # ── 填充大间隙：检测线缺失时按 uniform_cell_h 补行 ──
        filled_row_ys = []
        gap_threshold = uniform_cell_h * 1.8  # 超过1.8倍视为缺失
        for i, y in enumerate(row_ys):
            filled_row_ys.append(y)
            next_y = row_ys[i + 1] if i + 1 < len(row_ys) else bbox.h
            gap = next_y - y
            if gap > gap_threshold:
                # 在间隙中按 uniform_cell_h 均匀插入行
                n_fill = round(gap / uniform_cell_h) - 1
                if n_fill > 0:
                    step = gap / (n_fill + 1)
                    for k in range(1, n_fill + 1):
                        filled_row_ys.append(int(y + step * k))
                    logger.info(f"  间隙填充: y={y}→{next_y} (gap={gap}), 插入 {n_fill} 行")
        filled_row_ys = sorted(set(filled_row_ys))

        logger.info(f"  网格替换: 原始{len(row_ys)}条线 → 填充后{len(filled_row_ys)}条, uniform_h={uniform_cell_h}px")
        ocr = _get_ocr(lang)

        for cell_top in filled_row_ys:
            cell_bottom = cell_top + uniform_cell_h
            if cell_bottom > bbox.h:
                break

            # 裁剪单元格
            cell_roi = image[
                bbox.y + cell_top: bbox.y + cell_bottom,
                bbox.x: bbox.x2,
            ]

            # 独立 OCR 该单元格（内部自动检测并去除删除线）
            items, has_strikethrough = _ocr_cell(cell_roi, ocr)

            # 红框内只需首字母匹配前缀即可（T→Y 模糊修正）
            matched_text = None
            prefix_set = {p.upper() for p in prefixes}
            for _poly, text, _score in items:
                text_ns = text.replace(" ", "").strip()
                if not text_ns:
                    continue
                first = text_ns[0].upper()
                if first in prefix_set:
                    matched_text = text_ns
                    break
                if first == 'T' and 'Y' in prefix_set and len(text_ns) >= 5:
                    matched_text = 'Y' + text_ns[1:]
                    logger.info(f"  红框T→Y修正: '{text_ns}' → '{matched_text}'")
                    break

            if not matched_text:
                continue

            old_text = matched_text
            new_text = "H" + old_text

            # 填充区域 = 整个单元格（全图坐标）
            gx = bbox.x
            gy_fill = bbox.y + cell_top
            safe_w = bbox.w
            fill_h = uniform_cell_h

            # 若有删除线，先保存删除线掩膜（在白填充前）
            strike_mask = None
            if has_strikethrough:
                cell_gray = cv2.cvtColor(
                    image[gy_fill:gy_fill + fill_h, gx:gx + safe_w],
                    cv2.COLOR_RGB2GRAY,
                )
                strike_mask = _detect_strikethrough_mask(cell_gray)

            # 白色填充
            cv2.rectangle(
                modified,
                (gx + FILL_MARGIN, gy_fill + FILL_MARGIN),
                (gx + safe_w - FILL_MARGIN, gy_fill + fill_h - FILL_MARGIN),
                (255, 255, 255),
                -1,
            )

            # 先恢复删除线（在白底之上）
            if strike_mask is not None and np.any(strike_mask):
                _restore_protected_pixels(modified, image, strike_mask, gx, gy_fill)
                logger.info(f"  删除线已恢复: cell_top={cell_top}")

            # 再渲染新文字（在删除线之上，文字不被遮挡）
            render_w = max(safe_w - 2 * FILL_MARGIN, 6)
            render_h = max(fill_h - 2 * FILL_MARGIN, 6)
            text_img = _render_text_distributed(new_text, render_w, render_h)

            paste_x = gx + FILL_MARGIN
            paste_y = gy_fill + FILL_MARGIN
            pil_modified = Image.fromarray(modified)
            pil_modified.paste(text_img, (paste_x, paste_y), text_img)
            modified = np.array(pil_modified)

            cyan_boxes.append(BBox(bbox.x, gy_fill, bbox.w, fill_h))
            replacements.append((old_text, new_text))
            logger.info(f"  替换: {old_text} → {new_text} (cell_top={cell_top})")

    else:
        # ── 直接模式：整体 OCR（蓝框 annotations 等使用）──
        roi = bbox.crop(image)
        roi_h, roi_w = roi.shape[:2]
        min_dim = min(roi_w, roi_h)
        max_dim = max(roi_w, roi_h)

        if min_dim < 80:
            scale_factor = 5.0
        elif min_dim < 200:
            scale_factor = 3.0
        else:
            scale_factor = 1.0
        if max_dim * scale_factor > 3500:
            scale_factor = max(3500.0 / max_dim, 1.0)

        if scale_factor > 1.0:
            roi_scaled = cv2.resize(roi, (int(roi_w * scale_factor), int(roi_h * scale_factor)),
                                    interpolation=cv2.INTER_CUBIC)
        else:
            roi_scaled = roi

        pad_px = 0
        if roi_h < 100 and roi_w < 400:
            pad_px = max(int(min(roi_scaled.shape[:2]) * 0.35), 40)
            padded = np.full(
                (roi_scaled.shape[0] + 2 * pad_px, roi_scaled.shape[1] + 2 * pad_px, 3),
                255, dtype=np.uint8,
            )
            padded[pad_px:pad_px + roi_scaled.shape[0],
                   pad_px:pad_px + roi_scaled.shape[1]] = roi_scaled
            sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            roi_scaled = cv2.filter2D(padded, -1, sharpen)

        ocr = _get_ocr(lang)
        result = ocr.predict(roi_scaled)
        items = _parse_ocr_results(result)

        logger.info(f"  检测到文本: {len(items)} 项")
        for i, (_poly, text, score) in enumerate(items[:15]):
            logger.info(f"    [{i}] '{text}' (score={score:.2f})")

        for poly, text, score in items:
            text_nospace = text.replace(" ", "")
            if not pat.match(text_nospace):
                continue

            y_bbox, full_bbox = _find_y_prefix_bbox(poly, text)
            if y_bbox is None:
                continue

            fx, fy, fw, fh = full_bbox
            fx = int((fx - pad_px) / scale_factor)
            fy = int((fy - pad_px) / scale_factor)
            fw = max(int(fw / scale_factor), 1)
            fh = max(int(fh / scale_factor), 1)

            old_text = text_nospace
            new_text = "H" + old_text
            extra = max(fw // max(len(old_text), 1), 4)
            safe_x = max(fx - 1, 0)
            safe_w = fw + extra

            gx = bbox.x + safe_x
            gy_fill = bbox.y + fy

            cv2.rectangle(
                modified,
                (gx + FILL_MARGIN, gy_fill + FILL_MARGIN),
                (gx + safe_w - FILL_MARGIN, gy_fill + fh - FILL_MARGIN),
                (255, 255, 255),
                -1,
            )

            render_w = max(safe_w - 2 * FILL_MARGIN, 6)
            render_h = max(fh - 2 * FILL_MARGIN, 6)
            text_img = _render_text_distributed(new_text, render_w, render_h)

            paste_x = gx + FILL_MARGIN
            paste_y = gy_fill + FILL_MARGIN
            pil_modified = Image.fromarray(modified)
            pil_modified.paste(text_img, (paste_x, paste_y), text_img)
            modified = np.array(pil_modified)

            replacements.append((old_text, new_text))
            logger.info(f"  替换: {old_text} → {new_text} (safe_w={safe_w}, fill_h={fh})")

    return modified, replacements, cyan_boxes


def _ocr_cell(cell_roi: np.ndarray, ocr, remove_strikethrough: bool = True) -> tuple[list, bool]:
    """对单个单元格进行缩放+padding+锐化+OCR，返回 (识别结果, 是否有删除线)。"""
    cell_h, cell_w = cell_roi.shape[:2]
    if cell_h < 3 or cell_w < 3:
        return [], False

    has_strikethrough = False

    # 删除线检测与去除（在缩放前，原始分辨率上检测更准确）
    if remove_strikethrough:
        cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_RGB2GRAY)
        strike_mask = _detect_strikethrough_mask(cell_gray)
        if np.any(strike_mask):
            has_strikethrough = True
            # 在副本上用白色填充删除线区域，再送入 OCR
            cell_roi = cell_roi.copy()
            cell_roi[strike_mask > 0] = 255
            logger.info(f"    删除线检测: 发现并去除")

    min_dim = min(cell_w, cell_h)
    max_dim = max(cell_w, cell_h)
    if min_dim < 80:
        sf = 5.0
    elif min_dim < 200:
        sf = 3.0
    else:
        sf = 1.0
    if max_dim * sf > 3500:
        sf = max(3500.0 / max_dim, 1.0)

    if sf > 1.0:
        scaled = cv2.resize(cell_roi, (int(cell_w * sf), int(cell_h * sf)),
                            interpolation=cv2.INTER_CUBIC)
    else:
        scaled = cell_roi

    # 小区域加 padding + 锐化
    sh, sw = scaled.shape[:2]
    if cell_h < 100:
        pad = max(int(min(sh, sw) * 0.35), 40)
        padded = np.full((sh + 2 * pad, sw + 2 * pad, 3), 255, dtype=np.uint8)
        padded[pad:pad + sh, pad:pad + sw] = scaled
        sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        scaled = cv2.filter2D(padded, -1, sharpen)

    result = ocr.predict(scaled)
    return _parse_ocr_results(result), has_strikethrough


# ── 投影法 + 网格分类 辅助函数 ──────────────────────────────────


def _preprocess_for_table(gray: np.ndarray) -> np.ndarray:
    """自适应二值化 + 去噪 + 形态学标准化线厚。"""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.medianBlur(binary, 3)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def _detect_all_hlines_projection(binary: np.ndarray, min_line_ratio: float = 0.8) -> list[int]:
    """水平投影法 — 每行白像素占比超过阈值 → 聚类取中心。"""
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


def _classify_lines_by_grid(all_lines: list[int], cell_height: int,
                            tolerance: int = 3) -> tuple[list[int], list[int]]:
    """网格步进分类：从第一条线开始，期望下一条行线在 +cell_height ±tolerance。
    匹配到的是行线，其余是删除线。

    Returns: (table_lines, strike_lines)
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


def _make_strike_mask(chunk_gray: np.ndarray, strike_ys_local: list[int],
                      band_half: int = 6) -> np.ndarray:
    """在已知删除线 y 坐标附近提取水平墨迹像素作为 inpaint mask。"""
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
        h_only = cv2.morphologyEx(band, cv2.MORPH_OPEN, h_kernel)
        mask[y_top:y_bot, :] = h_only

    return mask


def detect_cyan_boxes(
    image: np.ndarray, bbox: BBox, row_ys: list[int],
    pattern: str = None, prefixes: list[str] = None,
    return_all_ocr: bool = False,
    debug_dir: str = None,
) -> list:
    """分片 OCR 扫描红框内单元格，匹配首字母则生成青色框。

    流程：
    1. 投影法检测红框内所有水平线
    2. 网格步进区分行线 vs 删除线
    3. 按行线分片（每10格一片，重叠2格）
    4. 删除线 inpaint 修复后整片 OCR
    5. 结果按 y 坐标匹配单元格，首字母匹配则生成青框

    返回 [BBox, ...] — 每个青色框对应一个匹配编号的单元格。
    """
    CHUNK_SIZE = 10
    OVERLAP_ROWS = 2
    GRID_TOLERANCE = 3

    # ── 投影法检测所有水平线 ──
    box_roi = image[bbox.y:bbox.y2, bbox.x:bbox.x2]
    box_gray = cv2.cvtColor(box_roi, cv2.COLOR_RGB2GRAY)
    box_h, box_w = box_gray.shape[:2]

    binary = _preprocess_for_table(box_gray)
    all_lines = _detect_all_hlines_projection(binary, min_line_ratio=0.8)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(box_roi).save(os.path.join(debug_dir, "cyan_00_red_roi.jpg"), quality=90)
        lines_vis = cv2.cvtColor(box_gray, cv2.COLOR_GRAY2RGB)
        for ly in all_lines:
            cv2.line(lines_vis, (0, ly), (box_w, ly), (255, 0, 0), 2)
        Image.fromarray(lines_vis).save(os.path.join(debug_dir, "cyan_01_all_hlines.jpg"), quality=90)

    if len(all_lines) < 2:
        logger.warning("  投影法检测行线不足，跳过青框生成")
        return [], []

    # ── 确定单元格高度（>50px 间距的众数）──
    gaps = [all_lines[i + 1] - all_lines[i] for i in range(len(all_lines) - 1)]
    large_gaps = [g for g in gaps if g > 50]
    if not large_gaps:
        logger.warning("  无有效大间距，跳过青框生成")
        return [], []
    rounded = [round(g / 5) * 5 for g in large_gaps]
    cell_height = Counter(rounded).most_common(1)[0][0]

    # ── 网格步进分类 ──
    table_lines, strike_lines = _classify_lines_by_grid(
        all_lines, cell_height, tolerance=GRID_TOLERANCE
    )
    logger.info(f"  投影法: {len(all_lines)} 条线 → 行线 {len(table_lines)}, "
                f"删除线 {len(strike_lines)}, cell_h={cell_height}px")

    if debug_dir:
        grid_vis = cv2.cvtColor(box_gray, cv2.COLOR_GRAY2RGB)
        for ly in table_lines:
            cv2.line(grid_vis, (0, ly), (box_w, ly), (0, 200, 0), 2)
        for ly in strike_lines:
            cv2.line(grid_vis, (0, ly), (box_w, ly), (0, 0, 255), 2)
        Image.fromarray(grid_vis).save(os.path.join(debug_dir, "cyan_02_grid_classify.jpg"), quality=90)

    # ── 分片（仅用行线，带重叠）──
    chunk_step = max(CHUNK_SIZE - OVERLAP_ROWS, 1)
    chunks = []
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

    logger.info(f"  分片: {len(chunks)} 片 (每片{CHUNK_SIZE}行, 重叠{OVERLAP_ROWS}行)")

    # ── 每片：删除线 mask + inpaint ──
    chunk_strike_masks = {}
    for ci, (chunk_top, chunk_bot, _chunk_tbl) in enumerate(chunks):
        local_strikes = [sy - chunk_top for sy in strike_lines
                         if chunk_top < sy < chunk_bot]
        if not local_strikes:
            continue
        chunk_roi = box_roi[chunk_top:chunk_bot, :]
        chunk_gray = cv2.cvtColor(chunk_roi, cv2.COLOR_RGB2GRAY)
        strike_mask = _make_strike_mask(chunk_gray, local_strikes, band_half=6)
        if np.any(strike_mask):
            chunk_strike_masks[ci] = strike_mask
            logger.info(f"    片[{ci}] 删除线: {len(local_strikes)} 条 @ {local_strikes}")

    # ── 分片 OCR + 青框生成 ──
    prefixes = prefixes or DEFAULT_PREFIXES
    prefix_set = {p.upper() for p in prefixes}
    ocr = _get_ocr(OCR_LANG_EN)

    cyan_boxes = []
    cyan_box_data = []
    all_ocr_results = []
    matched_cell_ys = set()

    for ci, (chunk_top, chunk_bot, chunk_tbl_lines) in enumerate(chunks):
        chunk_roi = box_roi[chunk_top:chunk_bot, :]
        ch, cw = chunk_roi.shape[:2]
        if ch < 3 or cw < 3:
            continue

        # 删除线 inpaint
        if ci in chunk_strike_masks:
            chunk_bgr = cv2.cvtColor(chunk_roi, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(chunk_bgr, chunk_strike_masks[ci],
                                         inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            chunk_for_ocr = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        else:
            chunk_for_ocr = chunk_roi

        # 缩放
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

        # OCR
        result = ocr.predict(scaled)
        items = _parse_ocr_results(result)

        if debug_dir:
            Image.fromarray(chunk_roi).save(
                os.path.join(debug_dir, f"cyan_10_chunk{ci}_roi.jpg"), quality=90)
            Image.fromarray(scaled).save(
                os.path.join(debug_dir, f"cyan_11_chunk{ci}_scaled.jpg"), quality=90)
            ocr_vis = scaled.copy()
            for poly, text, score in items:
                if poly is None or len(poly) < 4:
                    continue
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(ocr_vis, [pts], True, (0, 255, 0), 2)
                cv2.putText(ocr_vis, text.replace(" ", ""),
                            (int(pts[0][0]), int(pts[0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            Image.fromarray(ocr_vis).save(
                os.path.join(debug_dir, f"cyan_12_chunk{ci}_ocr_result.jpg"), quality=90)

        # 坐标映射 + 匹配
        for poly, text, score in items:
            if poly is None or len(poly) < 4:
                continue
            text_ns = text.replace(" ", "").strip()
            if not text_ns:
                continue

            if return_all_ocr:
                xs = [pt[0] / sf for pt in poly]
                ys = [pt[1] / sf for pt in poly]
                all_ocr_results.append({
                    "abs_bbox": BBox(
                        bbox.x + int(min(xs)),
                        bbox.y + chunk_top + int(min(ys)),
                        int(max(xs) - min(xs)),
                        int(max(ys) - min(ys)),
                    ),
                    "text": text_ns,
                    "score": score,
                    "matched": text_ns[0].upper() in prefix_set,
                })

            first = text_ns[0].upper()
            if first not in prefix_set:
                if first == 'T' and 'Y' in prefix_set and len(text_ns) >= 5:
                    text_ns = 'Y' + text_ns[1:]
                    logger.info(f"  青框T→Y修正: '{text}' → '{text_ns}'")
                else:
                    continue

            ys_poly = [pt[1] / sf for pt in poly]
            text_cy = sum(ys_poly) / len(ys_poly)
            abs_y = chunk_top + text_cy

            for tl in table_lines:
                cell_bot = tl + cell_height
                if tl <= abs_y < cell_bot:
                    if tl not in matched_cell_ys:
                        matched_cell_ys.add(tl)
                        cyan_boxes.append(BBox(bbox.x, bbox.y + tl,
                                               bbox.w, cell_height))
                        cell_has_strike = any(tl < sy < tl + cell_height
                                              for sy in strike_lines)
                        cyan_box_data.append({
                            "bbox": BBox(bbox.x, bbox.y + tl,
                                         bbox.w, cell_height),
                            "text": text_ns,
                            "has_strikethrough": cell_has_strike,
                            "cell_top": tl,
                        })
                        logger.info(f"  青框: cell_top={tl}, 匹配='{text_ns}'")
                    break

    logger.info(f"  分片OCR: {len(chunks)} 片, {len(cyan_boxes)} 个青框, "
                f"删除线 {len(strike_lines)} 条")
    if return_all_ocr:
        return cyan_boxes, cyan_box_data, all_ocr_results
    return cyan_boxes, cyan_box_data


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
        if isinstance(bbox, list) or not hasattr(bbox, "crop"):
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

    green_bbox = regions.get("bottom_right_number")
    orange_bbox = regions.get("top_left_number")
    green_orange_result = None  # 三方投票结果，绿框处理时设置，橙框复用

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

    # 文本验证：首字母+至少4位、无空格、首字母在 prefixes 中
    def _valid_prefix(s):
        if not s:
            return None
        s = s.upper().replace(" ", "")
        if len(s) < 5:
            return None
        if s[0] in prefixes_upper and re.match(r'^[A-Z][A-Z0-9]{4,}$', s):
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
            metadata = regions.get("_metadata", {})
            cyan_box_data = metadata.get("cyan_box_data")  # 从检测阶段获取

            if cyan_box_data:
                # 有预检测数据，直接使用，不需要重新检测行线和 OCR
                red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
                modified, repls, cyan_boxes = replace_y_in_region_pixel(
                    modified, bbox, lang=OCR_LANG_EN,
                    use_grid_alignment=True,
                    pattern=red_pattern, prefixes=prefixes,
                    cyan_box_data=cyan_box_data,
                    original=image,
                )
            else:
                # 无预检测数据（CLI模式等），走原有 OCR 逻辑
                table_search_bbox = metadata.get("table_search_area")
                row_ys = detect_row_ys_for_red_box(
                    modified, bbox, table_search_bbox=table_search_bbox)
                # 红框使用宽松匹配：前缀 + 8位字母数字或连字符
                red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
                modified, repls, cyan_boxes = replace_y_in_region_pixel(
                    modified, bbox, lang=OCR_LANG_EN,
                    use_grid_alignment=True, row_ys=row_ys,
                    pattern=red_pattern, prefixes=prefixes,
                )

            # 存储青色框供 debug 绘图使用
            regions.setdefault("_metadata", {})["cyan_boxes"] = cyan_boxes

            # CSV 记录：每个被替换的青色（红框单元格）框
            for cb, (old_t, _new_t) in zip(cyan_boxes, repls):
                record_y_box(filename or "", old_t, cb)

        elif region_name in ("bottom_right_number", "top_left_number"):
            # 首次遇到绿/橙框时执行五方校验
            if region_name == "bottom_right_number":
                # 三方投票（绿框、橙框、文件名）— 红框不参与
                g = _valid_prefix(green_text)
                o = _valid_prefix(orange_text)
                f = _valid_prefix(filename_y)
                logger.info(f"  三方校验: green={g}, orange={o}, filename={f}")

                candidates = [x for x in [g, o, f] if x]
                source_y = None

                if not candidates:
                    logger.warning("  三方校验：无有效候选文本，跳过绿/橙框替换")
                elif len(set(candidates)) == 1:
                    source_y = candidates[0]
                    logger.info(f"  三方校验：全部一致 → '{source_y}'")
                else:
                    counts = Counter(candidates)
                    top_text, top_count = counts.most_common(1)[0]
                    if top_count >= 2:
                        source_y = top_text
                        logger.info(f"  三方校验：多数一致({top_count}/{len(candidates)}) → '{source_y}'")
                    else:
                        source_y = f or g or o
                        logger.info(f"  三方校验：全不同，优先文件名/绿/橙 → '{source_y}'")

                # 保存结果供橙框复用
                if source_y:
                    new_y = "H" + source_y
                    green_orange_result = (source_y, new_y)
                    logger.info(f"  三方投票结果: '{source_y}' → '{new_y}'")
                else:
                    green_orange_result = None

            if green_orange_result:
                old_y, new_y = green_orange_result
                draw_x = bbox.x
                draw_y = bbox.y
                draw_w = max(bbox.w, 1)
                draw_h = max(bbox.h, 1)
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
                # CSV 记录：绿/橙框
                record_y_box(filename or "", old_y, bbox)
            else:
                repls = []
                logger.warning(f"  {region_name}: 无有效文本，跳过替换")

        all_replacements.extend(repls)
        logger.info(f"  区域 {region_name}: {len(repls)} 处替换")

    # ── 工厂注意部分替换 ──
    fn_codes = regions.get("factory_note_codes", [])
    if fn_codes:
        logger.info(f"处理工厂注意部分: {len(fn_codes)} 个编号")

        replaced_regions = []
        red_bbox = regions.get("material_code_column")
        if red_bbox:
            replaced_regions.append(red_bbox)
        if green_bbox:
            replaced_regions.append(green_bbox)
        if orange_bbox:
            replaced_regions.append(orange_bbox)

        def _bbox_overlap(a, b):
            ox = max(0, min(a.x2, b.x2) - max(a.x, b.x))
            oy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
            return ox * oy

        for fc in fn_codes:
            code = fc["code"]
            bbox = fc["bbox"]
            fn_area = max(bbox.w * bbox.h, 1)
            skip = False
            for rb in replaced_regions:
                overlap = _bbox_overlap(bbox, rb)
                if overlap / fn_area > 0.3:
                    logger.info(f"  Factory Note: 跳过 {code}，与已替换区域重叠 {overlap/fn_area:.0%}")
                    skip = True
                    break
            if skip:
                continue

            new_text = "H" + code
            draw_x, draw_y = bbox.x, bbox.y
            draw_w, draw_h = max(bbox.w, 1), max(bbox.h, 1)
            cv2.rectangle(
                modified, (draw_x, draw_y),
                (draw_x + draw_w, draw_y + draw_h),
                (255, 255, 255), -1,
            )
            text_img = _render_text_distributed(new_text, draw_w, draw_h)
            pil_modified = Image.fromarray(modified)
            pil_modified.paste(text_img, (draw_x, draw_y), text_img)
            modified = np.array(pil_modified)
            all_replacements.append((code, new_text))
            logger.info(f"  Factory Note: {code} → {new_text}")
            # CSV 记录：工厂注意编号框
            record_y_box(filename or "", code, bbox)

    return modified, all_replacements
