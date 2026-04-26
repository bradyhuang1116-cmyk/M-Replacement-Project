import cv2
import numpy as np
import glob
from paddleocr import PaddleOCR

# 初始化OCR
ocr = PaddleOCR(
    text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_name='PP-OCRv5_server_rec',
    use_textline_orientation=False
)

# 查找crop文件（原始裁切图）
pattern = "diagnostic_output/purple_diag/*_crop_4.jpg"
files = glob.glob(pattern)

if not files:
    print(f"未找到文件: {pattern}")
    exit(1)

img_path = files[0]
print(f"读取文件: {img_path}")

# 使用numpy读取避免中文路径问题
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"无法读取图像: {img_path}")
    exit(1)

print(f"原始图像尺寸: {img.shape}")

# 放大图像（参考test_purple_diag.py的逻辑）
TARGET_LONG_EDGE = 2000
long_edge = max(img.shape[0], img.shape[1])
scale = TARGET_LONG_EDGE / long_edge if long_edge > 0 else 1
new_w = int(img.shape[1] * scale)
new_h = int(img.shape[0] * scale)
enlarged = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# 水平拉伸 + 垂直拉伸
STRETCH_X = 2.0
STRETCH_Y = 1.2
stretched_w = int(new_w * STRETCH_X)
stretched_h = int(new_h * STRETCH_Y)
enlarged = cv2.resize(enlarged, (stretched_w, stretched_h), interpolation=cv2.INTER_CUBIC)

print(f"放大后图像尺寸: {enlarged.shape}")

# 使用放大后的图像进行CC检测
img_h, img_w = enlarged.shape

# 二值化
_, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 连通域检测
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# 收集所有CC框，过滤掉明显的噪点和汉字
# 收集所有CC框
img_h, img_w = enlarged.shape
valid_boxes = []

print(f"\n检测到 {num_labels-1} 个连通域")

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    # 过滤条件
    if area < 2000 or area > 100000:
        continue

    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.1 or aspect_ratio > 3.0:
        continue

    if x < 10 or x + w > img_w - 10:
        continue

    valid_boxes.append((x, y, w, h, area))

print(f"检测到 {num_labels-1} 个连通域，过滤后保留 {len(valid_boxes)} 个")

# 按x坐标排序
valid_boxes.sort(key=lambda b: b[0])

# 对每个CC框进行OCR识别
print("\n识别结果:")
recognized_chars = []

for i, (x, y, w, h, area) in enumerate(valid_boxes):
    # CC框往外扩大10%
    expand = 0.1
    expand_w = int(w * expand)
    expand_h = int(h * expand)

    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h)
    x2 = min(img_w, x + w + expand_w)
    y2 = min(img_h, y + h + expand_h)

    # 裁剪单个字符
    char_img = enlarged[y1:y2, x1:x2]

    # 添加白色边框padding
    pad_w = int(char_img.shape[1] * 0.5)
    pad_h = int(char_img.shape[0] * 0.5)
    char_img = cv2.copyMakeBorder(char_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)

    # 转换为3通道图像
    if len(char_img.shape) == 2:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)

    # OCR识别
    result = ocr.ocr(char_img)

    if result and len(result) > 0:
        result_dict = result[0]
        rec_texts = result_dict.get('rec_texts', [])
        rec_scores = result_dict.get('rec_scores', [])

        if rec_texts and len(rec_texts) > 0:
            text = rec_texts[0]
            conf = rec_scores[0] if rec_scores else 0.0
            clean_text = ''.join(c for c in text if c.isalnum())

            # 检查是否包含中文
            def is_cjk(ch):
                cp = ord(ch)
                return (0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x309F or
                        0x30A0 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or
                        0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF)

            has_cjk = any(is_cjk(c) for c in text)

            if clean_text and not has_cjk:  # 只保留识别成功且无中文的
                recognized_chars.append((i, x, y, w, h, clean_text, conf))
                print(f"CC[{i}]: x={x} y={y} 识别=\"{clean_text}\" conf={conf:.2f}")

                # 保存裁切图
                char_output_path = f"diagnostic_output/char_{i}_{clean_text}.jpg"
                cv2.imencode('.jpg', char_img)[1].tofile(char_output_path)
            else:
                print(f"CC[{i}]: x={x} y={y} 无有效字符，移除")
        else:
            print(f"CC[{i}]: x={x} y={y} 识别失败，移除")
    else:
        print(f"CC[{i}]: x={x} y={y} OCR失败，移除")

# 过滤：只保留X或Y所在行的CC框
print(f"\n识别成功 {len(recognized_chars)} 个字符")

if recognized_chars:
    # 找到包含X或Y的CC框
    target_chars = [c for c in recognized_chars if c[5].upper().startswith('X') or c[5].upper().startswith('Y')]

    if target_chars:
        # 获取目标行的y坐标范围（更严格）
        target_y = target_chars[0][2]  # y坐标
        target_h = target_chars[0][4]  # 高度
        y_min = target_y - target_h * 0.2
        y_max = target_y + target_h * 1.2

        print(f"找到目标字符: {[c[5] for c in target_chars]}")
        print(f"目标行y范围: {y_min:.0f} - {y_max:.0f}")

        # 过滤：只保留在目标行范围内的CC框
        filtered_chars = []
        for char in recognized_chars:
            i, x, y, w, h, text, conf = char
            if y_min <= y <= y_max:
                filtered_chars.append(char)
                print(f"  保留: CC[{i}] \"{text}\" y={y}")
            else:
                print(f"  移除: CC[{i}] \"{text}\" y={y} (不在目标行)")

        recognized_chars = filtered_chars
        print(f"\n过滤后保留 {len(recognized_chars)} 个字符")
    else:
        print("未找到X或Y开头的字符")
else:
    print("没有识别成功的字符")

# 绘制可视化图像（使用enlarged图）
vis = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

# 添加标记确认使用enlarged图
cv2.putText(vis, f"ENLARGED {vis.shape}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

for i, x, y, w, h, text, conf in recognized_chars:
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f'{i}:{text}'
    cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 保存输出图像
output_path = "diagnostic_output/test_cc_simple_output.jpg"
cv2.imencode('.jpg', vis)[1].tofile(output_path)
print(f"\n输出图像已保存: {output_path}")
