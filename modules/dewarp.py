"""几何矫正: deskew(竖线刚体旋转) + dewarp(竖线平滑弯曲矫正)。

收到图后对整图(或区域 ROI)矫正, 矫正图作为后续全流程的输入与交付物。
- deskew: 竖线倾斜角中位数 → 刚体旋转(直线仍是直线, 全图安全)。
- dewarp: 每条长竖线拟合平滑二次曲线 → 逐行水平位移场(含外推限幅)。
  仅取长竖线做锚点(排除图形区短竖线), 降低把标题栏/绿橙框掰弯的风险。

只依赖 region_detector._binarize_for_lines + factory_note_pixel.extract_table_lines。
"""
import numpy as np
import cv2

from modules.region_detector import _binarize_for_lines
from modules.factory_note_pixel import extract_table_lines

# 竖线锚点最小跨度比例(占图高)
DESKEW_MIN_SPAN = 0.12   # deskew 用: 中等长度即可参与角度投票
DEWARP_MIN_SPAN = 0.25   # dewarp 用: 只取长竖线(表格边框/图框), 排除图形区短竖线


def _v_tilt_median(v_bin, r_h, min_span_ratio=DESKEW_MIN_SPAN):
    """竖线平均倾斜角(偏离竖直, deg) + 参与投票的竖线数。"""
    contours, _ = cv2.findContours(v_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angs = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 30:
            continue
        if pts[:, 1].max() - pts[:, 1].min() < r_h * min_span_ratio:
            continue
        [vx, vy, _, _] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        a = np.degrees(np.arctan2(vy, vx))
        if 75 < abs(a) < 105:
            angs.append(float(a - 90 if a > 0 else a + 90))
    return (float(np.median(angs)) if angs else 0.0), len(angs)


def _fit_v_curves(v_bin, r_h, r_w, y_bin=15, min_span_ratio=DEWARP_MIN_SPAN):
    """每条长竖线 → 分箱中位数 → deg-2 平滑曲线, 全高求值(范围外端点常量保持)。
    返回 [(X_k目标, x_curve[r_h], ymin, ymax, idx, max_dev)], 按 X_k 升序。"""
    contours, _ = cv2.findContours(v_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for i, cnt in enumerate(contours):
        pts = cnt.reshape(-1, 2)
        if len(pts) < 30:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        if ys.max() - ys.min() < r_h * min_span_ratio:   # 仅长竖线做锚点
            continue
        yb_c, xb_c = [], []
        for yb in range(int(ys.min()), int(ys.max()) + 1, y_bin):
            m = (ys >= yb) & (ys < yb + y_bin)
            if m.sum() < 2:
                continue
            yb_c.append(yb + y_bin / 2)
            xb_c.append(np.median(xs[m]))
        if len(yb_c) < 5:
            continue
        yb_c = np.array(yb_c)
        xb_c = np.array(xb_c)
        coeffs = np.polyfit(yb_c, xb_c, 2)
        X_k = float(np.mean(xb_c))
        ymin, ymax = int(yb_c.min()), int(yb_c.max())
        x_curve = np.polyval(coeffs, np.clip(np.arange(r_h), ymin, ymax)).astype(np.float64)
        max_dev = float(np.abs(x_curve[ymin:ymax] - X_k).max())
        lines.append((X_k, x_curve, ymin, ymax, i, max_dev))
    lines.sort(key=lambda L: L[0])
    return lines


def correct_region(roi_rgb, logger=None):
    """对整图(或区域 ROI)做 deskew + dewarp 矫正。
    返回 (corrected_rgb, info)。info: deskew_ang / n_lines / dx_range。
    竖线不足(<2)时只做 deskew(或原样返回)。"""
    def log(m):
        if logger:
            logger(m)

    # 3 通道 RGB 守卫(下游 cvtColor RGB2GRAY / borderValue 3元组都要求 3 通道)
    if roi_rgb.ndim == 2:
        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_GRAY2RGB)
    elif roi_rgb.shape[2] == 4:
        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGBA2RGB)

    r_h, r_w = roi_rgb.shape[:2]

    bw = _binarize_for_lines(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY))
    _, v_lines = extract_table_lines(bw)

    # ── Step 0: deskew 刚体旋转 ──
    deskew_ang, n_v = _v_tilt_median(v_lines, r_h)
    log(f"[correct] deskew {n_v}条竖线 中位数={deskew_ang:.3f}°")
    if abs(deskew_ang) > 0.05:
        M = cv2.getRotationMatrix2D((r_w / 2, r_h / 2), deskew_ang, 1.0)
        roi_rgb = cv2.warpAffine(roi_rgb, M, (r_w, r_h),
                                 flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        bw = _binarize_for_lines(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY))
        _, v_lines = extract_table_lines(bw)

    # ── Step 1: 长竖线平滑曲线 ──
    lines = _fit_v_curves(v_lines, r_h, r_w)
    log(f"[correct] dewarp 长竖线{len(lines)}条")

    if len(lines) < 2:
        return roi_rgb, {"deskew_ang": deskew_ang, "n_lines": len(lines), "dx_range": (0.0, 0.0)}

    # ── Step 2: 逐行分段线性(含外推限幅) → map_x ──
    X_targets = np.array([L[0] for L in lines])
    curves = np.array([L[1] for L in lines])
    K = len(lines)
    xd = np.arange(r_w, dtype=np.float64)
    map_x = np.empty((r_h, r_w), dtype=np.float32)
    for y in range(r_h):
        a = curves[:, y]
        t = X_targets
        d = t - a
        dq = np.interp(xd, t, d)
        # 外推: 端点斜率线性延伸, 但限幅到锚点实测位移范围(防远处无限增长掰弯图形)
        d_lo, d_hi = float(d.min()), float(d.max())
        sl_l = (d[1] - d[0]) / (t[1] - t[0]) if t[1] != t[0] else 0.0
        left = xd < t[0]
        dq[left] = d[0] + sl_l * (xd[left] - t[0])
        sl_r = (d[-1] - d[-2]) / (t[-1] - t[-2]) if t[-1] != t[-2] else 0.0
        right = xd > t[-1]
        dq[right] = d[-1] + sl_r * (xd[right] - t[-1])
        np.clip(dq, d_lo, d_hi, out=dq)
        map_x[y, :] = (xd - dq).astype(np.float32)

    dx_full = np.tile(xd.astype(np.float32), (r_h, 1)) - map_x
    map_y = np.tile(np.arange(r_h, dtype=np.float32)[:, None], (1, r_w))
    corrected = cv2.remap(roi_rgb, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    log(f"[correct] 位移场 dx∈[{dx_full.min():.1f},{dx_full.max():.1f}]px")
    return corrected, {"deskew_ang": deskew_ang, "n_lines": K,
                       "dx_range": (float(dx_full.min()), float(dx_full.max()))}
