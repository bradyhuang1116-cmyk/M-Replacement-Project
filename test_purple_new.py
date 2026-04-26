"""紫框检测诊断脚本（含边界约束）。

流程：
1. 紫框搜索区裁切 → 旋转90°CW（竖排→横排）
2. OCR (server det)：获取粗略文本行位置
3. 正则筛选含前缀编号的行 → 字符比例粗估编号起始位置
4. 从粗估位置二次裁切（排除 CJK 字符）→ 放大至最长边2000px
5. 形态学修复 → OCR → 字符比例定位前三字符 bbox
6. 边界约束：检查前三字符左右是否有相邻字符，若有则收缩 bbox
7. 映射回全图坐标，输出诊断图
"""
import os, sys, re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import cv2
import numpy as np
from modules.file_ingestion import load_file
from modules.region_detector import (
    BBox, _crop_and_scale, _ocr_purple, _map_bbox_back,
)
from config import DEFAULT_PREFIXES, OCR_MODE

TIF_DIR = Path(r'C:\Users\Brady Huang\Downloads\TIF_Undo')
OUT_DIR = Path('diagnostic_output/purple_diag')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = ['X', 'Y']

y_files = sorted(f for f in TIF_DIR.glob('*A216*')
                 if f.suffix.lower() in ('.tif', '.tiff', '.pdf', '.png', '.jpg'))
print(f"找到 {len(y_files)} 个 A216 测试文件\n")


def detect_lines_morph(gray):
    """检测横竖线并返回彩色标注图像（竖线=红色，横线=蓝色）。"""
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 检测竖线（降低阈值以适应紫框区域）
    min_vh = max(int(h / 30), 50)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # 检测横线
    min_hw = max(w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 分析竖线
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vlines = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        vlines.append((cx + cw // 2, cy, cy + ch, ch))
    vlines.sort(key=lambda v: v[3], reverse=True)

    # 分析横线
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hlines = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        hlines.append((cy + ch // 2, cx, cx + cw))
    hlines.sort()

    print(f'  竖线检测: min_vh={min_vh}, 检测到{len(vlines)}条竖线')
    if len(vlines) >= 2:
        print(f'    最长竖线: x={vlines[0][0]}, y={vlines[0][1]}-{vlines[0][2]}, 长度={vlines[0][3]}')
        print(f'    第二长竖线: x={vlines[1][0]}, y={vlines[1][1]}-{vlines[1][2]}, 长度={vlines[1][3]}')
    print(f'  横线检测: min_hw={min_hw}, 检测到{len(hlines)}条横线')

    # 创建彩色图：BGR格式，竖线=红色(0,0,255)，横线=蓝色(255,0,0)
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, 0] = h_morph  # B通道=横线（蓝色）
    result[:, :, 2] = v_morph  # R通道=竖线（红色）
    return result, vlines, hlines


def detect_lines_and_crop(gray, rotated, prefix, code_re):
    """检测横竖线，裁切两条最长竖线外的区域，按横线分割成小块并OCR。"""
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 检测竖线
    min_vh = max(int(h / 30), 50)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vh))
    v_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # 检测横线
    min_hw = max(w // 8, 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_hw, 1))
    h_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 保存morph图
    morph_vis = np.zeros((h, w, 3), dtype=np.uint8)
    morph_vis[:, :, 0] = h_morph
    morph_vis[:, :, 2] = v_morph
    save_img(f'{prefix}_rotated_morph.jpg', morph_vis)

    # 分析竖线
    v_contours, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vlines = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        vlines.append((cx + cw // 2, cy, cy + ch, ch))
    vlines.sort(key=lambda v: v[3], reverse=True)

    print(f'  竖线检测: 检测到{len(vlines)}条竖线')
    if len(vlines) >= 2:
        print(f'    最长竖线: x={vlines[0][0]}, 长度={vlines[0][3]}')
        print(f'    第二长竖线: x={vlines[1][0]}, 长度={vlines[1][3]}')

    # 裁切两条最长竖线之间的区域
    if len(vlines) >= 2:
        x1 = min(vlines[0][0], vlines[1][0])
        x2 = max(vlines[0][0], vlines[1][0])
        cropped = rotated[:, x1:x2]
        print(f'  裁切区域: x={x1}-{x2}, 宽度={x2-x1}')
        save_img(f'{prefix}_cropped_between_vlines.jpg', cropped)
    else:
        print('  竖线不足2条，跳过裁切')
        return vlines, []

    # 分析横线（全图范围）
    h_contours, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hlines = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        y_center = cy + ch // 2
        hlines.append((y_center, cx, cx + cw))
    hlines.sort()
    print(f'  横线检测: 检测到{len(hlines)}条横线')

    # 按横线分割裁切区域成小块并OCR
    found_codes = []
    if len(hlines) > 0:
        found_codes = process_chunks_by_hlines(cropped, hlines, x1, prefix, code_re)

    return vlines, hlines, found_codes


def process_chunks_by_hlines(cropped, hlines, x_offset, prefix, code_re):
    """按横线分割图像为小块，对每个小块OCR识别编号。返回找到的所有结果。"""
    h, w = cropped.shape[:2]

    # 构建分割点（横线的y坐标，保持最小间隔）
    split_points = [0]
    MIN_INTERVAL = 60  # 最小间隔60像素，避免分割过细
    for y, _, _ in hlines:
        if y > 0 and y < h and y > split_points[-1] + MIN_INTERVAL:
            split_points.append(y)
    split_points.append(h)

    print(f'  分割为{len(split_points)-1}个小块')

    # 对每个小块进行处理，收集结果
    found_codes = []
    for i in range(len(split_points) - 1):
        y1 = split_points[i]
        y2 = split_points[i + 1]
        chunk = cropped[y1:y2, :]

        print(f'\n  ── 小块[{i}]: y={y1}-{y2}, 高度={y2-y1} ──')
        save_img(f'{prefix}_chunk_{i}.jpg', chunk)

        # OCR识别，传入偏移量
        results = ocr_and_locate_code(chunk, i, prefix, code_re, y_offset=y1, x_offset=x_offset)
        if results:
            found_codes.extend(results)

    return found_codes


def ocr_and_locate_code(chunk, chunk_idx, prefix, code_re, y_offset=0, x_offset=0):
    """对小块进行OCR识别并定位编号。返回找到的结果列表。"""
    # 放大
    TARGET_LONG_EDGE = 2000
    ch, cw = chunk.shape[:2]
    long_edge = max(cw, ch)
    scale = TARGET_LONG_EDGE / long_edge if long_edge > 0 else 1
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    enlarged = cv2.resize(chunk, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 拉伸
    STRETCH_X = 2
    STRETCH_Y = 1.2
    stretched_w = int(new_w * STRETCH_X)
    stretched_h = int(new_h * STRETCH_Y)
    enlarged = cv2.resize(enlarged, (stretched_w, stretched_h), interpolation=cv2.INTER_CUBIC)

    # 对比度增强
    if len(enlarged.shape) == 3:
        lab = cv2.cvtColor(enlarged, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enlarged = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2RGB)

    save_img(f'{prefix}_chunk_{chunk_idx}_enlarged.jpg', enlarged)

    # 在enlarged图上OCR（用于可视化原始识别）
    full_bbox_enlarged = BBox(0, 0, enlarged.shape[1], enlarged.shape[0])
    ocr_results_enlarged = _ocr_purple(enlarged, full_bbox_enlarged, lang="cn")

    # 绘制enlarged上的OCR结果
    ocr_vis_enlarged = enlarged.copy()
    for text, conf, poly in ocr_results_enlarged:
        if poly is not None:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(ocr_vis_enlarged, [pts], True, (0, 255, 0), 2)
            cv2.putText(ocr_vis_enlarged, f'{text[:10]}', (pts[0][0], pts[0][1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_img(f'{prefix}_chunk_{chunk_idx}_ocr_vis.jpg', ocr_vis_enlarged)

    # 形态学修复
    repaired = morph_repair_image(enlarged)
    save_img(f'{prefix}_chunk_{chunk_idx}_repaired.jpg', repaired)

    # 在repaired图上OCR（用于精确定位）
    full_bbox = BBox(0, 0, repaired.shape[1], repaired.shape[0])
    ocr_results = _ocr_purple(repaired, full_bbox, lang="cn")

    # 为chunk 9添加详细调试
    if chunk_idx == 9:
        debug_vis = enlarged.copy()
        print(f'    === Chunk 9 调试信息 ===')
        print(f'    chunk尺寸: {ch}x{cw}')
        print(f'    enlarged尺寸: {enlarged.shape[0]}x{enlarged.shape[1]}')
        print(f'    scale={scale:.3f}, STRETCH_X={STRETCH_X}, STRETCH_Y={STRETCH_Y}')
        print(f'    x_offset={x_offset}, y_offset={y_offset}')

        for i, (text, conf, poly) in enumerate(ocr_results):
            if poly is not None:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(debug_vis, [pts], True, (255, 0, 0), 2)
                cv2.putText(debug_vis, f'{i}:{text[:8]}', (pts[0][0], pts[0][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        save_img(f'{prefix}_chunk_{chunk_idx}_debug_ocr.jpg', debug_vis)

    # 遍历OCR结果，找到所有匹配的Y/X编号
    found_results = []
    for idx, (text, conf, poly) in enumerate(ocr_results):
        if poly is None:
            continue
        text_up = text.upper().strip()
        if not text_up:
            continue

        # 检查是否匹配Y或X
        match = code_re.search(text_up)
        if not match:
            print(f'    OCR[{idx}]: "{text}" conf={conf:.2f} (无匹配)')
            continue

        # 找到匹配，使用精确定位
        print(f'    OCR[{idx}]: "{text}" → match="{match.group(0)}" conf={conf:.2f}')

        # 调用精确定位函数
        debug_prefix_param = f'{prefix}_chunk_{chunk_idx}' if chunk_idx == 9 else None
        result = locate_prefix_by_ocr_with_bounds(repaired, enlarged, code_re, debug_prefix=debug_prefix_param)
        if result:
            matched_code, prefix_char, code_bbox, raw_bbox, cc_info = result
            print(f'    找到编号: "{matched_code}"')

            # 将enlarged坐标转换回chunk坐标，再转换到原图坐标
            cx, cy, cw, ch = code_bbox
            chunk_x = int(cx / STRETCH_X / scale)
            chunk_y = int(cy / STRETCH_Y / scale)
            chunk_w = int(cw / STRETCH_X / scale)
            chunk_h = int(ch / STRETCH_Y / scale)
            orig_x = x_offset + chunk_x
            orig_y = y_offset + chunk_y

            if chunk_idx == 9:
                print(f'    code_bbox在enlarged上: ({cx},{cy},{cw},{ch})')
                print(f'    转换到chunk: ({chunk_x},{chunk_y},{chunk_w},{chunk_h})')
                print(f'    转换到原图: ({orig_x},{orig_y},{chunk_w},{chunk_h})')

                # 在enlarged图上画出定位框
                debug_vis2 = enlarged.copy()
                cv2.rectangle(debug_vis2, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 3)
                save_img(f'{prefix}_chunk_{chunk_idx}_debug_bbox.jpg', debug_vis2)

            found_results.append((matched_code, orig_x, orig_y, chunk_w, chunk_h, orig_x))
            break  # 找到一个后退出，避免重复处理同一个

    if not found_results:
        print(f'    未找到编号')

    return found_results


def save_img(name, img):
    path = OUT_DIR / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'    保存: {path}')


def morph_repair_image(image):
    """简单阈值二值化，避免形态学膨胀导致的字符粘连。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 方式1：使用 Otsu 自动阈值（推荐）
    #_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 方式2：手动指定阈值（取消注释使用，调整值 0-255）
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return binary


def get_col_projection(image, ry, rh):
    """对图像 [ry, ry+rh] 行做垂直投影，返回每列黑色像素计数数组。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    y1 = max(0, ry)
    y2 = min(h, ry + rh)
    if y1 >= y2:
        return np.zeros(w, dtype=int)

    roi = gray[y1:y2, :]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.sum(bw > 0, axis=0)


def find_clean_edge(col_sum, start_x, direction, max_scan=200, top_n=1):
    """从 start_x 附近找最佳分割位置。

    策略：在搜索范围内按 col_sum 值排序，返回 top_n 个候选。
    1. 优先找 col_sum==0 的列（完全空白），取最近的
    2. 如果找不到，找投影值最低的列
    top_n>1 时返回列表，top_n==1 时返回单个值。
    """
    w = len(col_sum)

    # 收集搜索范围内所有候选：(col_sum值, 距离, x坐标)
    candidates = []
    for dist in range(max_scan):
        for x in [start_x + dist * direction, start_x - dist * direction]:
            if 0 <= x < w:
                candidates.append((col_sum[x], dist, x))

    if not candidates:
        return [start_x] * top_n if top_n > 1 else start_x

    # 排序：先按 col_sum 升序（越少黑色越好），再按距离升序（越近越好）
    candidates.sort(key=lambda c: (c[0], c[1]))

    # 去重 x 坐标，保留最优的
    seen = set()
    unique = []
    for _, _, x in candidates:
        if x not in seen:
            seen.add(x)
            unique.append(x)
            if len(unique) >= top_n:
                break

    # 补齐不足的
    while len(unique) < top_n:
        unique.append(unique[-1] if unique else start_x)

    if top_n == 1:
        return unique[0]
    return unique


def find_cc_at_boundary(cc_boxes, boundary_x, ry, rh, side='left'):
    """检查 boundary_x 是否落在某个 CC 框内部（切到了字符）。
    未使用，保留备用。
    """
    y_center = ry + rh / 2
    y_margin = rh * 0.8
    min_area = 200

    for (bx, by, bw, bh) in cc_boxes:
        if bw * bh < min_area:
            continue
        by_center = by + bh / 2
        if abs(by_center - y_center) > y_margin:
            continue
        margin = 3
        if bx + margin < boundary_x < bx + bw - margin:
            return (bx, by, bw, bh)
    return None


def match_cc_to_chars(cc_boxes, rx, ry, rw, rh):
    """将 OCR 行 y 范围内的 CC 框按 x 排序，返回有序列表。

    每个 CC 框对应文本中的一个字符（按从左到右顺序）。
    过滤噪点（面积太小的 CC）。
    """
    y_center = ry + rh / 2
    y_margin = rh * 0.4
    min_area = 50

    # 收集行内的 CC 框
    line_ccs = []
    for (bx, by, bw, bh) in cc_boxes:
        if bw * bh < min_area or bw < 5 or bh < 5:
            continue
        by_center = by + bh / 2
        if abs(by_center - y_center) > y_margin:
            continue
        # x 范围大致在 OCR 行范围内（左右各扩展 50%）
        line_left = rx - int(rw * 0.1)
        line_right = rx + rw + int(rw * 0.1)
        if bx + bw < line_left or bx > line_right:
            continue
        line_ccs.append((bx, by, bw, bh))

    # 按 x 排序
    line_ccs.sort(key=lambda b: b[0])
    return line_ccs


def _is_cjk(ch):
    """判断字符是否为 CJK（中日韩）字符。"""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or      # CJK 统一汉字
            0x3040 <= cp <= 0x309F or      # 平假名
            0x30A0 <= cp <= 0x30FF or      # 片假名
            0x3400 <= cp <= 0x4DBF or      # CJK 扩展 A
            0xF900 <= cp <= 0xFAFF or      # CJK 兼容
            0xFF00 <= cp <= 0xFFEF)        # 全角字符


def calc_char_positions(text, rx, rw):
    """根据字符类型（CJK≈2倍宽）计算每个字符的起止 x 坐标。

    返回 list of (x_start, x_end)，长度 == len(text)。
    """
    if not text:
        return []
    # CJK 字符宽度权重 2，其他 1
    weights = [2.0 if _is_cjk(ch) else 1.0 for ch in text]
    total_w = sum(weights)
    px_per_unit = rw / total_w if total_w > 0 else 0

    positions = []
    x = float(rx)
    for w in weights:
        x_end = x + w * px_per_unit
        positions.append((int(round(x)), int(round(x_end))))
        x = x_end
    return positions


def locate_prefix_by_ocr_with_bounds(image, enlarged_image, code_re, debug_prefix=None):
    """OCR 定位前三字符 bbox，并根据相邻字符约束左右边界。

    image: 修复后图像（用于OCR和CC检测）
    enlarged_image: 放大图（修复前，用于边界投影检测）

    返回 (matched_code, prefix_char, code_bbox, raw_bbox, cc_info) 或 None。
        code_bbox: (x, y, w, h) 约束后的前三字符 bbox
        raw_bbox:  (x, y, w, h) 约束前的前三字符 bbox（用于对比）
        cc_info:   dict with 'line_ccs', 'text_up', 'char_positions', 'rx', 'ry', 'rw', 'rh'
    """
    # ── 连通域分析（在 repaired 图上，已经是干净的二值图） ──
    if len(image.shape) == 3:
        gray_cc = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_cc = image.copy()
    _, bw_cc = cv2.threshold(gray_cc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 轻度闭运算：连通同一字符内的断裂笔画（如"表"的横竖撇捺）
    # 使用更小的核避免字符粘连
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_cc = cv2.morphologyEx(bw_cc, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    num_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(bw_cc, connectivity=8)
    img_h_cc, img_w_cc = image.shape[:2]
    cc_boxes = []
    for i in range(1, num_labels):
        sx, sy, sw, sh, area = stats_cc[i]
        if area < 30 or sw < 3 or sh < 3:
            continue
        if sw > img_w_cc * 0.5 or sh > img_h_cc * 0.5:
            continue
        cc_boxes.append((sx, sy, sw, sh))
    full_bbox = BBox(0, 0, image.shape[1], image.shape[0])
    ocr_results = _ocr_purple(image, full_bbox, lang="cn")

    # 调试：保存OCR结果
    if debug_prefix:
        debug_vis = image.copy()
        for i, (text, conf, poly) in enumerate(ocr_results):
            if poly is not None:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(debug_vis, [pts], True, (0, 0, 255), 2)
                cv2.putText(debug_vis, f'{i}:{text[:8]}', (pts[0][0], pts[0][1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_img(f'{debug_prefix}_locate_ocr.jpg', debug_vis)

    best = None
    best_conf = 0

    for j, (text, conf, poly) in enumerate(ocr_results):
        if poly is None:
            continue
        text_up = text.upper().strip()
        if not text_up:
            continue

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        rx, ry = int(min(xs)), int(min(ys))
        rw, rh = int(max(xs) - min(xs)), int(max(ys) - min(ys))

        m = code_re.search(text_up)
        if not m:
            print(f'    OCR[{j}]: "{text_up}" conf={conf:.2f} (无匹配)')
            continue

        matched = m.group(0)
        prefix_char = m.group(1)
        n_chars = 3  # 用前三字符定位
        n_chars_box = 1  # 但仅框选首字母
        char_positions = calc_char_positions(text_up, rx, rw)
        # 等效 char_w（用于 max_scan 等计算）
        char_w = rw / len(text_up) if len(text_up) > 0 else rw

        # 原始前三字符 bbox（用于定位）
        code_start_idx = m.start()
        code_end_idx = m.start() + n_chars - 1
        raw_code_x = char_positions[code_start_idx][0]
        raw_code_w = char_positions[code_end_idx][1] - raw_code_x

        # 首字母bbox（最终框选）
        first_char_x = char_positions[code_start_idx][0]
        first_char_w = char_positions[code_start_idx][1] - first_char_x

        print(f'    前三字符定位: "{matched[:n_chars]}" '
              f'start_idx={code_start_idx} end_idx={code_end_idx} '
              f'bbox=({raw_code_x},{ry},{raw_code_w},{rh})')
        print(f'    首字母框选: "{prefix_char}" bbox=({first_char_x},{ry},{first_char_w},{rh})')

        # 调试：绘制前三字符和首字母框
        if debug_prefix:
            debug_bbox_vis = image.copy()
            # 绘制整行OCR框（蓝色）
            cv2.rectangle(debug_bbox_vis, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
            cv2.putText(debug_bbox_vis, 'OCR', (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # 绘制前三字符定位框（黄色）
            cv2.rectangle(debug_bbox_vis, (raw_code_x, ry), (raw_code_x+raw_code_w, ry+rh), (0, 255, 255), 2)
            cv2.putText(debug_bbox_vis, '3chars', (raw_code_x, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # 绘制首字母框（绿色）
            cv2.rectangle(debug_bbox_vis, (first_char_x, ry), (first_char_x+first_char_w, ry+rh), (0, 255, 0), 3)
            cv2.putText(debug_bbox_vis, '1st', (first_char_x, ry+rh+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            save_img(f'{debug_prefix}_locate_boxes.jpg', debug_bbox_vis)

        # ── 边界约束：用放大图（修复前）做投影，避免修复后字符膨胀导致间隙消失 ──
        col_sum = get_col_projection(enlarged_image, ry, rh)
        max_scan = max(int(char_w * 1.5), 30)

        # 首字母 bbox（用于最终框选）
        code_x_left = first_char_x
        code_x_right = first_char_x + first_char_w

        # 邻居检查（基于首字母）
        left_char_idx = m.start() - 1
        right_char_idx = m.start() + 1  # 首字母右侧就是第二个字符
        has_left_neighbor = left_char_idx >= 0
        has_right_neighbor = right_char_idx < len(text_up)

        # 左边界：仅在有左邻字符时调整，从估算位置找最近空白间隙
        left_boundary = code_x_left
        if has_left_neighbor:
            left_boundary = find_clean_edge(col_sum, code_x_left, -1, max_scan)
            if left_boundary != code_x_left:
                print(f'    左边界清理: {code_x_left} → {left_boundary} '
                      f'(左邻 "{text_up[left_char_idx]}", 找最近空白间隙)')
            code_x_left = left_boundary

        # 右边界：仅在有右邻字符时调整，从估算位置找最近空白间隙
        right_boundary = code_x_right
        if has_right_neighbor:
            right_boundary = find_clean_edge(col_sum, code_x_right, +1, max_scan)
            if right_boundary != code_x_right:
                print(f'    右边界清理: {code_x_right} → {right_boundary} '
                      f'(右邻 "{text_up[right_char_idx]}", 找最近空白间隙)')
            code_x_right = right_boundary

        final_code_x = code_x_left
        final_code_w = max(code_x_right - code_x_left, 4)

        # ── 宽度保护：约束后宽度不得低于原始宽度的 80% ──
        MIN_WIDTH_RATIO = 0.80
        min_w = int(first_char_w * MIN_WIDTH_RATIO)
        if final_code_w < min_w:
            print(f'    宽度过窄: {final_code_w} < {min_w} (原始{first_char_w}×80%)')
            left_shrink = code_x_left - first_char_x
            right_shrink = (first_char_x + first_char_w) - code_x_right
            w_img = len(col_sum)

            if left_shrink >= right_shrink:
                # 左边界收缩更多，向左扩展
                # 从当前左边界向左，按 col_sum 升序排列候选列
                candidates = []
                for x in range(code_x_left - 1, max(code_x_left - max_scan, 0) - 1, -1):
                    if 0 <= x < w_img:
                        candidates.append((col_sum[x], x))
                candidates.sort()  # 按黑色像素升序
                for _, cx in candidates:
                    test_w = code_x_right - cx
                    if test_w >= min_w:
                        code_x_left = cx
                        print(f'    左边界扩展 → {cx} (col_sum={col_sum[cx]}, 宽度={test_w})')
                        break
            else:
                # 右边界收缩更多，向右扩展
                # 从当前右边界向右，按 col_sum 升序排列候选列
                candidates = []
                for x in range(code_x_right + 1, min(code_x_right + max_scan, w_img)):
                    candidates.append((col_sum[x], x))
                candidates.sort()  # 按黑色像素升序
                for _, cx in candidates:
                    test_w = cx - code_x_left
                    if test_w >= min_w:
                        code_x_right = cx
                        print(f'    右边界扩展 → {cx} (col_sum={col_sum[cx]}, 宽度={test_w})')
                        break

            final_code_x = code_x_left
            final_code_w = max(code_x_right - code_x_left, 4)

        # ── 连通域校验（仅诊断，不修改边界） ──
        cc1 = find_cc_at_boundary(cc_boxes, final_code_x, ry, rh, side='left')
        cc3 = find_cc_at_boundary(cc_boxes, final_code_x + final_code_w, ry, rh, side='right')

        if cc1:
            print(f'    CC左边界切入字符: x={cc1[0]}, w={cc1[2]} → 应snap到左边缘={cc1[0]}')
        else:
            print(f'    CC左边界: 未切入字符（target_x={final_code_x}）')
        if cc3:
            cc3_right = cc3[0] + cc3[2]
            print(f'    CC右边界切入字符: x={cc3[0]}, w={cc3[2]} → 应snap到右边缘={cc3_right}')
        else:
            print(f'    CC右边界: 未切入字符（target_x={final_code_x + final_code_w}）')

        print(f'    OCR[{j}]: "{text_up}" → match="{matched}" prefix="{prefix_char}" '
              f'conf={conf:.2f} char_w={char_w:.1f}')
        print(f'      原始bbox: ({raw_code_x},{ry},{raw_code_w},{rh})')
        print(f'      约束后bbox: ({final_code_x},{ry},{final_code_w},{rh})')
        print(f'      左边界: {left_boundary}  右边界: {right_boundary}')

        if best is None or len(matched) > len(best[0]) or (
                len(matched) == len(best[0]) and conf > best_conf):
            best = (matched, prefix_char,
                    (final_code_x, ry, final_code_w, rh),
                    (raw_code_x, ry, raw_code_w, rh),
                    {'line_ccs': match_cc_to_chars(cc_boxes, rx, ry, rw, rh),
                     'text_up': text_up,
                     'char_positions': char_positions,
                     'code_start_idx': code_start_idx,
                     'rx': rx, 'ry': ry, 'rw': rw, 'rh': rh})
            best_conf = conf

    return best


def diagnose_purple(f):
    prefix = f.stem
    print(f'\n{"="*60}')
    print(f'  紫框诊断（边界约束）: {f.name}')
    print(f'{"="*60}')

    img, _ = load_file(str(f))
    img_h, img_w = img.shape[:2]

    if img_h > img_w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_h, img_w = img.shape[:2]
        print(f'  已旋转，尺寸: {img_w}x{img_h}')

    # ── 1. 紫框搜索区域 ──
    green_x = int(img_w * (5.0 / 8.0))
    purple_y = img_h - int(img_h * 1.5 / 6.0)
    purple_search = BBox(0, purple_y, green_x, img_h - purple_y)

    purple_sub, purple_scale = _crop_and_scale(img, purple_search)
    roi_h, roi_w = purple_sub.shape[:2]

    if purple_sub.size == 0:
        print('  搜索区为空，跳过')
        return

    # 保存紫框搜索区
    save_img(f'{prefix}_purple_search.jpg', purple_sub)

    # ── 2. 旋转 90° CW ──
    rotated = cv2.rotate(purple_sub, cv2.ROTATE_90_CLOCKWISE)
    rot_h, rot_w = rotated.shape[:2]

    # 保存旋转后图像及其形态学线检测结果
    save_img(f'{prefix}_rotated.jpg', rotated)

    # 转灰度并检测横竖线
    if len(rotated.shape) == 3:
        gray_rot = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    else:
        gray_rot = rotated.copy()

    # 构建正则表达式（只捕获Y或X，后面必须跟字母数字组合且含数字）
    prefixes_upper = {p.upper() for p in PREFIXES}
    prefix_pattern = '|'.join(re.escape(p) for p in sorted(prefixes_upper, key=len, reverse=True))
    # 匹配Y或X，后面跟任意字母数字组合（至少1个字符）
    code_re = re.compile(rf'(?<![A-Z0-9])({prefix_pattern})(?=[A-Z0-9-])')


    # 检测横竖线并获取线条信息
    vlines, hlines, found_codes = detect_lines_and_crop(gray_rot, rotated, prefix, code_re)

    if not found_codes:
        print('  未找到匹配')
        return

    # chunk中已完成精确定位，直接构建final_candidates
    # 格式: (code, crx, cry, crw, crh, rrx, rry, rrw, rrh, cc_info, sx, sy, sc, cx1, cy1)
    final_candidates = []
    for matched_code, rx, ry, rw, rh, est_code_x in found_codes:
        # 使用相同的坐标作为约束后和原始坐标
        final_candidates.append((matched_code,
                                 rx, ry, rw, rh,  # 约束后坐标
                                 rx, ry, rw, rh,  # 原始坐标（相同）
                                 None, 1.0, 1.0, 1.0, 0, 0))  # 其他参数

    # ── 6. 逆映射 + 输出最终标注图 ──
    vis_full = img.copy()
    colors = [(255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 128, 255)]

    for i, (code, crx, cry, crw, crh, rrx, rry, rrw, rrh,
            cc_info, sx, sy_s, sc, cx1, cy1) in enumerate(final_candidates):
        color = colors[i % len(colors)]

        # 约束后 bbox 逆映射
        code_sub = BBox(cry, roi_h - crx - crw, crh, crw)
        code_full = _map_bbox_back(code_sub, purple_search, purple_scale)

        # 原始 bbox 逆映射（红色虚线对比）
        raw_sub = BBox(rry, roi_h - rrx - rrw, rrh, rrw)
        raw_full = _map_bbox_back(raw_sub, purple_search, purple_scale)

        print(f'  [{i}] "{code}" 约束后: {code_full}  原始: {raw_full}')

        # 红色：原始 bbox
        cv2.rectangle(vis_full, (raw_full.x, raw_full.y),
                      (raw_full.x2, raw_full.y2), (0, 0, 255), 2)
        # 绿色：约束后 bbox
        cv2.rectangle(vis_full, (code_full.x, code_full.y),
                      (code_full.x2, code_full.y2), (0, 255, 0), 3)

        # CC 框逆映射到全图并绘制（仅字母/数字，跳过 CJK/标点）
        if cc_info:
            line_ccs = cc_info['line_ccs']
            text_up = cc_info['text_up']
            char_positions = cc_info['char_positions']

            # 构建 CJK/标点 区域
            cjk_zones = []
            for ci, ch in enumerate(text_up):
                if (_is_cjk(ch) or not ch.isalnum()) and ci < len(char_positions):
                    cjk_zones.append(char_positions[ci])

            # 构建字母/数字字符列表
            latin_chars = []
            for ci, ch in enumerate(text_up):
                if ch.isalnum() and not _is_cjk(ch) and ci < len(char_positions):
                    lcx = (char_positions[ci][0] + char_positions[ci][1]) / 2
                    latin_chars.append((ci, ch, lcx))

            for (bx, by, bw, bh) in line_ccs:
                cc_cx = bx + bw / 2
                # 跳过 CJK 区域
                in_cjk = any(zx1 <= cc_cx <= zx2 for (zx1, zx2) in cjk_zones)
                if in_cjk:
                    continue

                # CC 框坐标：enlarged → rotated
                rot_x = int(bx / sx / sc) + cx1
                rot_y = int(by / sy_s / sc) + cy1
                rot_w = max(int(bw / sx / sc), 1)
                rot_h_cc = max(int(bh / sy_s / sc), 1)

                # rotated → purple_sub (逆 90° CW 旋转)
                sub_bbox = BBox(rot_y, roi_h - rot_x - rot_w, rot_h_cc, rot_w)
                full_cc = _map_bbox_back(sub_bbox, purple_search, purple_scale)

                # 找最近的 Latin 字符标签
                best_match = None
                best_dist = float('inf')
                for (ci, ch, lcx) in latin_chars:
                    d = abs(cc_cx - lcx)
                    if d < best_dist:
                        best_dist = d
                        best_match = ch

                cv2.rectangle(vis_full, (full_cc.x, full_cc.y),
                              (full_cc.x2, full_cc.y2), (255, 200, 0), 2)
                if best_match:
                    cv2.putText(vis_full, best_match,
                                (full_cc.x, full_cc.y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    save_img(f'{prefix}_result.jpg', vis_full)


# ── 执行诊断 ──
for f in y_files:
    try:
        diagnose_purple(f)
    except Exception as e:
        import traceback
        print(f'  处理失败: {e}')
        traceback.print_exc()
    print()

print("紫框诊断测试完成。")
