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

PREFIXES = ['X', 'Y', 'Z', 'B', 'P']

y_files = sorted(f for f in TIF_DIR.glob('*A216*')
                 if f.suffix.lower() in ('.tif', '.tiff', '.pdf', '.png', '.jpg'))
print(f"找到 {len(y_files)} 个 A216 测试文件\n")


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


def locate_prefix_by_ocr_with_bounds(image, enlarged_image, code_re):
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
        n_chars = min(3, len(matched))
        char_positions = calc_char_positions(text_up, rx, rw)
        # 等效 char_w（用于 max_scan 等计算）
        char_w = rw / len(text_up) if len(text_up) > 0 else rw

        # 原始前三字符 bbox（用 CJK 感知位置）
        code_start_idx = m.start()
        code_end_idx = m.start() + n_chars - 1
        raw_code_x = char_positions[code_start_idx][0]
        raw_code_w = char_positions[code_end_idx][1] - raw_code_x

        print(f'    前三字符识别框: "{matched[:n_chars]}" '
              f'start_idx={code_start_idx} end_idx={code_end_idx} '
              f'bbox=({raw_code_x},{ry},{raw_code_w},{rh})')

        # ── 边界约束：用放大图（修复前）做投影，避免修复后字符膨胀导致间隙消失 ──
        col_sum = get_col_projection(enlarged_image, ry, rh)
        max_scan = max(int(char_w * 1.5), 30)
    
        # 原始前三字符 bbox
        code_x_left = raw_code_x
        code_x_right = raw_code_x + raw_code_w

        # 邻居检查
        left_char_idx = m.start() - 1
        right_char_idx = m.start() + n_chars
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

        # ── 宽度保护：约束后宽度不得低于原始宽度的 70% ──
        MIN_WIDTH_RATIO = 0.80
        min_w = int(raw_code_w * MIN_WIDTH_RATIO)
        if final_code_w < min_w:
            print(f'    宽度过窄: {final_code_w} < {min_w} (原始{raw_code_w}×70%)')
            left_shrink = code_x_left - raw_code_x
            right_shrink = (raw_code_x + raw_code_w) - code_x_right
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

    # ── 2. 旋转 90° CW ──
    rotated = cv2.rotate(purple_sub, cv2.ROTATE_90_CLOCKWISE)
    rot_h, rot_w = rotated.shape[:2]

    # ── 3. 第一遍 OCR ──
    full_bbox = BBox(0, 0, rot_w, rot_h)
    ocr_results = _ocr_purple(rotated, full_bbox, lang="cn")

    # ── 4. 正则筛选粗候选 ──
    prefixes_upper = {p.upper() for p in PREFIXES}
    prefix_pattern = '|'.join(re.escape(p) for p in sorted(prefixes_upper, key=len, reverse=True))
    code_re = re.compile(rf'(?<![A-Z0-9])({prefix_pattern})(?=[A-Z0-9]*\d)[A-Z0-9]*\d')
    rough_candidates = []

    for text, conf, poly in ocr_results:
        if poly is None:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        rx, ry = int(min(xs)), int(min(ys))
        rw, rh = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        if rw < rh:
            continue
        text_up = text.upper().strip()
        if not text_up:
            continue
        mid_match = code_re.search(text_up)
        if not mid_match:
            continue

        matched_code = mid_match.group(0)
        char_w_est = rw / len(text_up) if len(text_up) > 0 else rw
        est_code_x = rx + int(mid_match.start() * char_w_est)

        print(f'  粗候选[{len(rough_candidates)}]: "{matched_code}" '
              f'(原文: "{text_up}") conf={conf:.2f}')
        rough_candidates.append((matched_code, rx, ry, rw, rh, est_code_x))

    if not rough_candidates:
        print('  未找到匹配')
        return

    # ── 5. 二次裁切 → 放大 → 形态学修复 → OCR精确定位（含边界约束） ──
    TARGET_LONG_EDGE = 2000
    final_candidates = []

    for idx, (code_p1, rx, ry, rw, rh, est_code_x) in enumerate(rough_candidates):
        print(f'\n  ── 候选[{idx}]: "{code_p1}" ──')

        margin_x = max(int(rh * 0.8), 16)
        margin_y = max(int(rh * 0.5), 12)
        crop_x1 = max(0, est_code_x - margin_x)
        crop_y1 = max(0, ry - margin_y)
        crop_x2 = min(rot_w, rx + rw + margin_x)
        crop_y2 = min(rot_h, ry + rh + margin_y)

        crop = rotated[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            final_candidates.append((code_p1, est_code_x, ry, int(min(3, len(code_p1)) * rh * 0.6), rh,
                                     est_code_x, ry, int(min(3, len(code_p1)) * rh * 0.6), rh))
            continue

        crop_h, crop_w = crop.shape[:2]
        save_img(f'{prefix}_crop_{idx}.jpg', crop)

        # 放大至最长边 2000px
        long_edge = max(crop_w, crop_h)
        scale = TARGET_LONG_EDGE / long_edge if long_edge > 0 else 1
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        enlarged = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 水平拉伸 + 垂直拉伸（使字符间隙更宽，提高边界检测精度）
        STRETCH_X = 1.2
        STRETCH_Y = 1.2
        stretched_w = int(new_w * STRETCH_X)
        stretched_h = int(new_h * STRETCH_Y)
        enlarged = cv2.resize(enlarged, (stretched_w, stretched_h), interpolation=cv2.INTER_CUBIC)

        # 对比度增强（CLAHE）
        if len(enlarged.shape) == 3:
            lab = cv2.cvtColor(enlarged, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            enlarged = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enlarged = clahe.apply(enlarged)

        save_img(f'{prefix}_enlarged_{idx}.jpg', enlarged)

        # 形态学修复
        repaired = morph_repair_image(enlarged)
        save_img(f'{prefix}_repaired_{idx}.jpg', repaired)

        # OCR 精确定位（含边界约束）
        result = locate_prefix_by_ocr_with_bounds(
            repaired, enlarged, code_re)

        if result:
            matched_code, prefix_char, code_bbox, raw_bbox, cc_info = result
            cx, cy, cw, ch = code_bbox
            rawx, rawy, raww, rawh = raw_bbox

            # 绘制 CC 连通域边界图（在 repaired 图上，与 CC 检测同图）
            if cc_info and cc_info.get('line_ccs'):
                cc_vis = repaired.copy()
                line_ccs = cc_info['line_ccs']
                text_up = cc_info['text_up']
                char_positions = cc_info['char_positions']
                ocr_rx, ocr_ry, ocr_rw, ocr_rh = cc_info['rx'], cc_info['ry'], cc_info['rw'], cc_info['rh']

                # 构建 CJK/标点 区域
                cjk_zones = []
                for ci, char in enumerate(text_up):
                    if (_is_cjk(char) or not char.isalnum()) and ci < len(char_positions):
                        cjk_zones.append(char_positions[ci])

                # 构建字母/数字字符列表（仅前三字符）
                code_start_idx = cc_info.get('code_start_idx', 0)
                latin_chars = []
                for ci, char in enumerate(text_up):
                    # 只处理编号部分的前三个字符
                    if ci >= code_start_idx and char.isalnum() and not _is_cjk(char):
                        latin_chars.append((ci, char))
                        if len(latin_chars) >= 3:
                            break

                print(f'    Latin字符: {[ch for ci, ch in latin_chars]}')

                # 过滤CC框：根据宽高比和面积特征筛选Latin字符
                img_w = repaired.shape[1]
                img_h = repaired.shape[0]

                print(f'    过滤前line_ccs数量: {len(line_ccs)}')

                valid_ccs = []
                for (bx, by, bw, bh) in line_ccs:
                    # 跳过左右边界噪点
                    if bx < 10 or bx + bw > img_w - 10:
                        continue

                    # Latin字符特征：宽高比通常在0.3-1.5之间，面积适中
                    aspect_ratio = bw / bh if bh > 0 else 0
                    area = bw * bh

                    # 过滤掉明显的汉字（宽高比接近1且面积较大）和噪点（面积太小）
                    if area < 1000 or area > 50000:
                        continue
                    if aspect_ratio < 0.2 or aspect_ratio > 2.0:
                        continue

                    cc_cx = bx + bw / 2
                    valid_ccs.append((bx, by, bw, bh, cc_cx))

                # 按x坐标排序，取前3个
                valid_ccs.sort(key=lambda c: c[0])
                valid_ccs = valid_ccs[:3]

                print(f'    过滤后valid_ccs数量: {len(valid_ccs)}')

                # 简单顺序匹配：第i个CC框对应第i个Latin字符
                matched_pairs = []
                for i in range(min(len(valid_ccs), len(latin_chars))):
                    bx, by, bw, bh, cc_cx = valid_ccs[i]
                    ci, char = latin_chars[i]
                    matched_pairs.append((i, char, (bx, by, bw, bh, cc_cx)))

                # 按顺序匹配：第 i 个 CC 框对应第 i 个 Latin 字符
                for cc_idx_label, (cc_idx, char, (bx, by, bw, bh, cc_cx)) in enumerate(matched_pairs):
                    print(f'    CC框[{cc_idx_label}] x={bx} cx={int(cc_cx)} w={bw} → 匹配字符="{char}"')

                    cv2.rectangle(cc_vis, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
                    label = f'{cc_idx_label}:{char}'
                    cv2.putText(cc_vis, label, (bx, by - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # 绘制原始前三字符框（黄色虚线）
                for x in range(rawx, rawx + raww, 10):
                    cv2.line(cc_vis, (x, rawy), (min(x+5, rawx+raww), rawy), (0, 255, 255), 2)
                    cv2.line(cc_vis, (x, rawy+rawh), (min(x+5, rawx+raww), rawy+rawh), (0, 255, 255), 2)
                for y in range(rawy, rawy + rawh, 10):
                    cv2.line(cc_vis, (rawx, y), (rawx, min(y+5, rawy+rawh)), (0, 255, 255), 2)
                    cv2.line(cc_vis, (rawx+raww, y), (rawx+raww, min(y+5, rawy+rawh)), (0, 255, 255), 2)

                # 绘制约束后的前三字符框（红色）
                cv2.rectangle(cc_vis, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 3)

                # 添加时间戳验证（右上角）
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                cv2.putText(cc_vis, timestamp, (cc_vis.shape[1] - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                save_img(f'{prefix}_cc_bounds_{idx}.jpg', cc_vis)

            # 约束后 bbox → 旋转图坐标（x/w 需除以拉伸系数）
            final_code_rx = int(cx / STRETCH_X / scale) + crop_x1
            final_code_ry = int(cy / STRETCH_Y / scale) + crop_y1
            final_code_rw = int(cw / STRETCH_X / scale)
            final_code_rh = int(ch / STRETCH_Y / scale)

            # 原始 bbox → 旋转图坐标（对比用）
            raw_code_rx = int(rawx / STRETCH_X / scale) + crop_x1
            raw_code_ry = int(rawy / STRETCH_Y / scale) + crop_y1
            raw_code_rw = int(raww / STRETCH_X / scale)
            raw_code_rh = int(rawh / STRETCH_Y / scale)

            print(f'    约束后rot: ({final_code_rx},{final_code_ry},{final_code_rw},{final_code_rh})')
            print(f'    原始rot:   ({raw_code_rx},{raw_code_ry},{raw_code_rw},{raw_code_rh})')

            final_candidates.append((matched_code,
                                     final_code_rx, final_code_ry, final_code_rw, final_code_rh,
                                     raw_code_rx, raw_code_ry, raw_code_rw, raw_code_rh,
                                     cc_info, STRETCH_X, STRETCH_Y, scale, crop_x1, crop_y1))
        else:
            print(f'    OCR定位失败，回退粗估')
            final_candidates.append((code_p1,
                                     est_code_x, ry, int(len(code_p1) * rh * 0.6), rh,
                                     est_code_x, ry, int(len(code_p1) * rh * 0.6), rh,
                                     None, STRETCH_X, STRETCH_Y, scale, crop_x1, crop_y1))

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
