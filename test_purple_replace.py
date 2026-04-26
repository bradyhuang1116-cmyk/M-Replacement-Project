"""紫框替换测试脚本：在检测到的前三字符位置替换为 H+前三字符（竖排）。

使用 test_purple_diag.py 的检测流程获取 bbox，然后在原图上进行替换。
"""
import os, sys, re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from modules.file_ingestion import load_file
from modules.region_detector import (
    BBox, _crop_and_scale, _ocr_purple, _map_bbox_back,
)
from config import OCR_MODE

TIF_DIR = Path(r'C:\Users\Brady Huang\Downloads\TIF_Undo')
OUT_DIR = Path('diagnostic_output/purple_diag')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = ['X', 'Y', 'Z', 'B', 'P']
TARGET_LONG_EDGE = 2000

# 字体路径
FONT_PATH = str(Path(__file__).parent / 'fonts' / 'JosefinSans-Light-5.ttf')


def morph_repair_image(image):
    """形态学修复断裂文字笔画。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    repaired = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h_close, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    dist = cv2.distanceTransform(repaired, cv2.DIST_L2, 5)
    mean_stroke = dist[repaired > 0].mean() if np.any(repaired > 0) else 1.0
    extra_radius = max(int(round(mean_stroke * 0.1)), 1)
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * extra_radius + 1, 2 * extra_radius + 1))
    repaired = cv2.dilate(repaired, kernel_expand, iterations=1)

    repaired_inv = cv2.bitwise_not(repaired)
    if len(image.shape) == 3:
        return cv2.cvtColor(repaired_inv, cv2.COLOR_GRAY2RGB)
    return repaired_inv


def locate_prefix_by_ocr(image, code_re):
    """OCR 定位前三字符 bbox。返回 (matched_code, prefix_char, code_bbox) 或 None。"""
    saved_mt = OCR_MODE["model_type"]
    OCR_MODE["model_type"] = "server"
    try:
        full_bbox = BBox(0, 0, image.shape[1], image.shape[0])
        ocr_results = _ocr_purple(image, full_bbox, lang="en")
    finally:
        OCR_MODE["model_type"] = saved_mt

    best = None
    best_conf = 0

    for text, conf, poly in ocr_results:
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
            continue

        matched = m.group(0)
        prefix_char = m.group(1)

        char_w = rw / len(text_up) if len(text_up) > 0 else rw
        code_x = rx + int(m.start() * char_w)
        n_chars = min(3, len(matched))
        code_w = int(n_chars * char_w)

        if best is None or len(matched) > len(best[0]) or (
                len(matched) == len(best[0]) and conf > best_conf):
            best = (matched, prefix_char, (code_x, ry, code_w, rh))
            best_conf = conf

    return best


def render_text_vertical(text, box_w, box_h):
    """渲染横排文本，旋转90°CCW 变竖排，匹配 box_w x box_h。

    返回 RGBA Image，尺寸 box_w x box_h。
    """
    # 横排渲染：宽=box_h, 高=box_w（旋转后宽高互换）
    h_w = max(box_h, 4)
    h_h = max(box_w, 4)

    render_size = max(h_h * 2, 32)
    font = None
    if FONT_PATH:
        try:
            font = ImageFont.truetype(FONT_PATH, render_size)
        except (IOError, OSError):
            pass
    if font is None:
        font = ImageFont.load_default()

    # 测量文本尺寸
    dummy = Image.new("RGBA", (1, 1))
    draw_d = ImageDraw.Draw(dummy)
    bb = draw_d.textbbox((0, 0), text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]

    if text_w <= 0 or text_h <= 0:
        return Image.new("RGBA", (box_w, box_h), (255, 255, 255, 0))

    # 紧凑渲染
    img = Image.new("RGBA", (text_w, text_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bb[0], -bb[1]), text, fill=(0, 0, 0, 255), font=font)

    # 缩放到横排目标尺寸
    img = img.resize((h_w, h_h), Image.LANCZOS)

    # 旋转90°CCW → 竖排
    img_v = img.rotate(90, expand=True)

    # 确保尺寸精确匹配
    if img_v.size != (box_w, box_h):
        img_v = img_v.resize((box_w, box_h), Image.LANCZOS)

    return img_v


def detect_and_replace(f):
    stem = f.stem
    print(f'\n{"="*60}')
    print(f'  紫框替换测试: {f.name}')
    print(f'{"="*60}')

    img, _ = load_file(str(f))
    img_h, img_w = img.shape[:2]

    if img_h > img_w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_h, img_w = img.shape[:2]

    # 1. 搜索区域
    green_x = int(img_w * (5.0 / 8.0))
    purple_y = img_h - int(img_h * 1.5 / 6.0)
    purple_search = BBox(0, purple_y, green_x, img_h - purple_y)

    purple_sub, purple_scale = _crop_and_scale(img, purple_search)
    roi_h, roi_w = purple_sub.shape[:2]
    if purple_sub.size == 0:
        print('  搜索区为空')
        return

    # 2. 旋转
    rotated = cv2.rotate(purple_sub, cv2.ROTATE_90_CLOCKWISE)
    rot_h, rot_w = rotated.shape[:2]

    # 3. 第一遍 OCR
    saved_mt = OCR_MODE["model_type"]
    OCR_MODE["model_type"] = "server"
    try:
        ocr_results = _ocr_purple(rotated, BBox(0, 0, rot_w, rot_h), lang="en")
    finally:
        OCR_MODE["model_type"] = saved_mt

    # 4. 正则筛选
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

        char_w_est = rw / len(text_up) if len(text_up) > 0 else rw
        est_code_x = rx + int(mid_match.start() * char_w_est)
        rough_candidates.append((mid_match.group(0), rx, ry, rw, rh, est_code_x))

    if not rough_candidates:
        print('  未找到匹配')
        return

    # 5. 二次裁切 + 放大 + 修复 + OCR 精确定位 + 替换
    modified = img.copy()

    for idx, (code_p1, rx, ry, rw, rh, est_code_x) in enumerate(rough_candidates):
        margin_x = max(int(rh * 0.3), 4)
        margin_y = max(int(rh * 0.15), 2)
        crop_x1 = max(0, est_code_x - margin_x)
        crop_y1 = max(0, ry - margin_y)
        crop_x2 = min(rot_w, rx + rw + margin_x)
        crop_y2 = min(rot_h, ry + rh + margin_y)

        crop = rotated[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        long_edge = max(crop_w, crop_h)
        scale = TARGET_LONG_EDGE / long_edge if long_edge > 0 else 1
        enlarged = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        repaired = morph_repair_image(enlarged)
        result = locate_prefix_by_ocr(repaired, code_re)

        if not result:
            print(f'  [{idx}] OCR定位失败，跳过')
            continue

        matched_code, prefix_char, code_bbox = result
        cx, cy, cw, ch = code_bbox

        # 映射回旋转图坐标
        crx = int(cx / scale) + crop_x1
        cry = int(cy / scale) + crop_y1
        crw = int(cw / scale)
        crh = int(ch / scale)

        # 逆映射到子图坐标
        code_sub = BBox(cry, roi_h - crx - crw, crh, crw)
        # 映射到全图坐标
        code_full = _map_bbox_back(code_sub, purple_search, purple_scale)

        bx, by, bw, bh = code_full.x, code_full.y, code_full.w, code_full.h
        first3 = matched_code[:3]
        new_text = "H" + first3
        print(f'  [{idx}] "{matched_code}" → 替换 "{first3}" 为 "{new_text}" '
              f'at ({bx},{by},{bw},{bh})')

        # 渲染竖排文本
        # 注意：经旋转逆映射后 bw 对应竖排文字上下方向，bh 对应左右方向
        shrink_tb = 0.15  # 上下各缩 5%
        shrink_x = int(bw * shrink_tb)
        inner_w = max(bw - 2 * shrink_x, 4)
        inner_h = bh
        text_img = render_text_vertical(new_text, inner_w, inner_h)

        # 白色覆盖原区域（上下各缩5%）+ 居中贴入新文本
        cv2.rectangle(modified, (bx + shrink_x, by), (bx + bw - shrink_x, by + bh), (255, 255, 255), -1)
        paste_x = bx + shrink_x
        paste_y = by
        pil_img = Image.fromarray(modified)
        pil_img.paste(text_img, (paste_x, paste_y), text_img)
        modified = np.array(pil_img)

    out_path = OUT_DIR / f'{stem}_replaced.jpg'
    cv2.imwrite(str(out_path), modified, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'  保存: {out_path}')


# ── 执行 ──
y_files = sorted(f for f in TIF_DIR.glob('*C055*')
                 if f.suffix.lower() in ('.tif', '.tiff', '.pdf', '.png', '.jpg'))
print(f"找到 {len(y_files)} 个测试文件\n")

for f in y_files:
    try:
        detect_and_replace(f)
    except Exception as e:
        import traceback
        print(f'  处理失败: {e}')
        traceback.print_exc()
    print()

print("替换测试完成。")
