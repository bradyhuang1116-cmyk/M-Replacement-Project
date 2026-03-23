"""删除线检测诊断 — 可视化 _detect_strikethrough_mask 的效果

不需要 OCR 模型，只做图像处理。
输出每个单元格的：
  1. 原图
  2. Otsu 二值化
  3. 形态学水平线提取
  4. 检测到的删除线掩膜（红色叠加）
  5. 移除删除线后的图像

用法:
    python test_strikethrough_diag.py
"""

import os
import sys
import logging

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
logger = logging.getLogger("strike_diag")

import cv2
import numpy as np

from modules.region_detector import BBox, _crop_and_scale, _map_bbox_back
from modules.text_replacer import (
    _determine_uniform_cell_height,
    detect_row_ys_for_red_box,
    _detect_strikethrough_mask,
)
from modules.file_ingestion import load_file

# ── 配置 ──
TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA070A191P7933-1_0-脱敏.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "strikethrough")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_img(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  保存: {name} ({img.shape[1]}x{img.shape[0]})")


def detect_strikethrough_verbose(cell_gray: np.ndarray, cell_idx: int):
    """详细版删除线检测 — 保存中间步骤的可视化图像。"""
    cell_h, cell_w = cell_gray.shape[:2]
    if cell_h < 10 or cell_w < 10:
        return np.zeros_like(cell_gray), {}

    # 1. 保存原始灰度图
    save_img(f"cell_{cell_idx:02d}_a_gray.jpg", cell_gray)

    # 2. Otsu 二值化
    _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_img(f"cell_{cell_idx:02d}_b_otsu.jpg", thresh)

    # 3. 形态学水平线提取（多种核宽对比）
    for ratio in [0.2, 0.3, 0.4]:
        kernel_w = max(int(cell_w * ratio), 15)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
        save_img(f"cell_{cell_idx:02d}_c_hline_{int(ratio*100)}.jpg", h_mask)

    # 4. 使用当前算法的核宽（40%）
    kernel_w = max(int(cell_w * 0.4), 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_min_zone = int(cell_h * 0.15)
    y_max_zone = int(cell_h * 0.85)

    # 5. 逐轮廓分析
    info = {
        "cell_h": cell_h, "cell_w": cell_w,
        "y_zone": (y_min_zone, y_max_zone),
        "contours_total": len(contours),
        "contours_detail": [],
        "detected_strikes": 0,
    }

    vis = cv2.cvtColor(cell_gray, cv2.COLOR_GRAY2RGB)
    # 画中间区域边界（蓝色虚线）
    cv2.line(vis, (0, y_min_zone), (cell_w, y_min_zone), (0, 0, 255), 1)
    cv2.line(vis, (0, y_max_zone), (cell_w, y_max_zone), (0, 0, 255), 1)

    strike_mask = np.zeros_like(cell_gray)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        cy = y + bh // 2
        in_zone = y_min_zone <= cy <= y_max_zone
        thin = bh <= 4
        is_strike = in_zone and thin

        detail = {
            "bbox": (x, y, bw, bh), "cy": cy,
            "in_zone": in_zone, "thin(<=4)": thin,
            "is_strike": is_strike,
        }
        info["contours_detail"].append(detail)

        # 画轮廓：绿色=删除线，红色=排除（不在区域），黄色=排除（太粗）
        if is_strike:
            color = (0, 255, 0)  # 绿色 = 检测为删除线
            cv2.drawContours(strike_mask, [c], -1, 255, -1)
            info["detected_strikes"] += 1
        elif not in_zone:
            color = (255, 0, 0)  # 红色 = 不在中间区域
        else:
            color = (255, 255, 0)  # 黄色 = 太粗

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)
        label = f"h={bh} cy={cy} {'HIT' if is_strike else 'MISS'}"
        cv2.putText(vis, label, (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    save_img(f"cell_{cell_idx:02d}_d_analysis.jpg", vis)

    # 6. 膨胀后的掩膜
    if np.any(strike_mask):
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        strike_mask = cv2.dilate(strike_mask, dilate_k, iterations=1)

    # 7. 删除线掩膜叠加到原图（红色半透明）
    overlay = cv2.cvtColor(cell_gray, cv2.COLOR_GRAY2RGB)
    overlay[strike_mask > 0] = [255, 0, 0]
    save_img(f"cell_{cell_idx:02d}_e_mask_overlay.jpg", overlay)

    # 8. 移除删除线后
    cleaned = cell_gray.copy()
    cleaned[strike_mask > 0] = 255
    save_img(f"cell_{cell_idx:02d}_f_cleaned.jpg", cleaned)

    # 也用核心函数检测做对比
    core_mask = _detect_strikethrough_mask(cell_gray)
    core_has = np.any(core_mask)
    info["core_detected"] = bool(core_has)

    return strike_mask, info


def main():
    logger.info(f"测试图片: {TEST_IMAGE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    if not os.path.isfile(TEST_IMAGE):
        logger.error(f"文件不存在: {TEST_IMAGE}")
        sys.exit(1)

    # ── Step 1: 加载图片 + 检测红框 + 行线 ──
    logger.info("=" * 60)
    logger.info("Step 1: 加载 + 红框 + 行线")
    logger.info("=" * 60)

    image, _ = load_file(TEST_IMAGE)
    img_h, img_w = image.shape[:2]
    logger.info(f"图片: {img_w}x{img_h}")

    from modules.region_detector import _locate_material_code_column
    frame = BBox(0, 0, img_w, img_h)
    red_search = BBox(frame.x, frame.y, frame.w // 2, frame.h)
    red_sub, red_scale = _crop_and_scale(image, red_search)

    mat_result = _locate_material_code_column(red_sub)
    if mat_result[0] is None:
        logger.error("未检测到 MATERIAL CODE 列")
        sys.exit(1)

    mat_bbox_sub, direction = mat_result
    mat_bbox = _map_bbox_back(mat_bbox_sub, red_search, red_scale)
    logger.info(f"红框: {mat_bbox}")

    row_ys = detect_row_ys_for_red_box(image, mat_bbox, table_search_bbox=red_search)
    logger.info(f"行线: {len(row_ys)} 条")

    # ── Step 2: 单元格高度 + 填充 ──
    uniform_cell_h = _determine_uniform_cell_height(row_ys)
    if not uniform_cell_h:
        logger.error("无法确定单元格高度")
        sys.exit(1)
    logger.info(f"单元格高度: {uniform_cell_h}px")

    filled_row_ys = []
    gap_threshold = uniform_cell_h * 1.8
    for i, y in enumerate(row_ys):
        filled_row_ys.append(y)
        next_y = row_ys[i + 1] if i + 1 < len(row_ys) else mat_bbox.h
        gap = next_y - y
        if gap > gap_threshold:
            n_fill = round(gap / uniform_cell_h) - 1
            if n_fill > 0:
                step = gap / (n_fill + 1)
                for k in range(1, n_fill + 1):
                    filled_row_ys.append(int(y + step * k))
    filled_row_ys = sorted(set(filled_row_ys))
    logger.info(f"填充后行线: {len(filled_row_ys)} 条")

    # ── Step 3: 逐单元格删除线诊断 ──
    logger.info("=" * 60)
    logger.info("Step 3: 逐单元格删除线检测诊断")
    logger.info("=" * 60)

    summary = []
    for ci, cell_top in enumerate(filled_row_ys):
        cell_bot = cell_top + uniform_cell_h
        if cell_bot > mat_bbox.h:
            cell_bot = mat_bbox.h
        if cell_bot - cell_top < 5:
            continue

        cell_roi = image[
            mat_bbox.y + cell_top: mat_bbox.y + cell_bot,
            mat_bbox.x: mat_bbox.x2,
        ]
        cell_gray = cv2.cvtColor(cell_roi, cv2.COLOR_RGB2GRAY)

        logger.info(f"  Cell[{ci}] y=[{cell_top},{cell_bot}] size={cell_roi.shape[1]}x{cell_roi.shape[0]}")

        strike_mask, info = detect_strikethrough_verbose(cell_gray, ci)
        has_strike = np.any(strike_mask)

        status = "有删除线" if has_strike else "无删除线"
        logger.info(f"    → {status} | 轮廓数: {info.get('contours_total', 0)} | "
                     f"检测到: {info.get('detected_strikes', 0)} | "
                     f"核心函数: {info.get('core_detected', False)}")

        if info.get("contours_detail"):
            for d in info["contours_detail"]:
                logger.info(f"      轮廓 bbox={d['bbox']} cy={d['cy']} "
                            f"in_zone={d['in_zone']} thin={d['thin(<=4)']} → {d['is_strike']}")

        summary.append({
            "cell_idx": ci, "cell_top": cell_top,
            "has_strike": has_strike, "info": info,
        })

    # ── 汇总 ──
    logger.info("=" * 60)
    logger.info("汇总")
    logger.info("=" * 60)
    total = len(summary)
    strikes = sum(1 for s in summary if s["has_strike"])
    logger.info(f"总单元格: {total}")
    logger.info(f"检测到删除线: {strikes}")
    logger.info(f"未检测到: {total - strikes}")

    # 列出所有有删除线但被漏掉的可疑情况（有轮廓但被过滤）
    suspicious = []
    for s in summary:
        info = s["info"]
        if not s["has_strike"] and info.get("contours_total", 0) > 0:
            # 有轮廓但没被判定为删除线
            for d in info.get("contours_detail", []):
                if not d["is_strike"]:
                    suspicious.append((s["cell_idx"], d))

    if suspicious:
        logger.info(f"\n可疑漏检（有轮廓但被过滤）: {len(suspicious)} 个")
        for ci, d in suspicious:
            reason = []
            if not d["in_zone"]:
                reason.append("不在中间区域")
            if not d["thin(<=4)"]:
                reason.append(f"太粗(h={d['bbox'][3]})")
            logger.info(f"  Cell[{ci}] bbox={d['bbox']} cy={d['cy']} 原因: {', '.join(reason)}")

    logger.info("完成! 查看输出目录中的图片了解详情。")


if __name__ == "__main__":
    main()
