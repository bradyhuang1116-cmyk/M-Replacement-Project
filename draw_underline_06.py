"""在找到的 YA169B585G06 中 '06' 位置下画横线"""
import fitz
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUT_DIR / 'P131062A101-1_l脱敏_underline06_v2.pdf')

doc = fitz.open(pdf_path)
page = doc[0]
orig_rotation = page.rotation
print(f"页面: rect={page.rect}, rotation={orig_rotation}")

# 临时去旋转，在统一坐标空间操作
page.set_rotation(0)

# 从 rawdict 定位 YA169B585G06 中末尾 '06' 的精确字符 bbox
blocks = page.get_text("rawdict")["blocks"]

for b in blocks:
    if "lines" not in b:
        continue
    for line in b["lines"]:
        for span in line["spans"]:
            chars = span.get("chars", [])
            if not chars:
                continue
            span_text = "".join(c["c"] for c in chars)
            if "YA169B585G06" not in span_text:
                continue

            print(f"找到span: '{span_text}' dir={line['dir']} bbox={span['bbox']}")

            # 定位 YA169B585G06 在 span 中的起止
            idx = span_text.index("YA169B585G06")
            code_chars = chars[idx:idx + 12]  # 12 chars for YA169B585G06
            suffix_chars = chars[idx + 10:idx + 12]  # 最后2个字符 '06'

            # 整串 bbox
            full_x0 = min(c["bbox"][0] for c in code_chars)
            full_y0 = min(c["bbox"][1] for c in code_chars)
            full_x1 = max(c["bbox"][2] for c in code_chars)
            full_y1 = max(c["bbox"][3] for c in code_chars)
            print(f"  YA169B585G06 bbox: ({full_x0:.1f}, {full_y0:.1f}, {full_x1:.1f}, {full_y1:.1f})")

            # '06' 部分 bbox
            s_x0 = min(c["bbox"][0] for c in suffix_chars)
            s_y0 = min(c["bbox"][1] for c in suffix_chars)
            s_x1 = max(c["bbox"][2] for c in suffix_chars)
            s_y1 = max(c["bbox"][3] for c in suffix_chars)
            print(f"  末尾'06' bbox: ({s_x0:.1f}, {s_y0:.1f}, {s_x1:.1f}, {s_y1:.1f})")
            print(f"  字符详情:")
            for c in suffix_chars:
                print(f"    '{c['c']}' bbox={[round(x,1) for x in c['bbox']]}")

            dx, dy = line["dir"]
            shape = page.new_shape()

            # 竖排文本 dir=(0, -1): 文字从上到下排列
            # 在整个 YA169B585G06 右侧画一条竖线(去旋转后)
            # 恢复旋转90度后，视觉上就是整串文字下方的横线
            margin = 2  # 间距
            line_x = full_x1 + margin  # 整串文字右侧
            line_y_start = full_y0
            line_y_end = full_y1
            print(f"  画线(去旋转坐标): ({line_x:.1f}, {line_y_start:.1f}) -> ({line_x:.1f}, {line_y_end:.1f})")

            shape.draw_line(fitz.Point(line_x, line_y_start), fitz.Point(line_x, line_y_end))
            shape.finish(color=(1, 0, 0), width=1.5)  # 红色，1.5pt粗
            shape.commit()
            print("  红色下划线已绘制")

# 恢复旋转
page.set_rotation(orig_rotation)
doc.save(output_path)
doc.close()
print(f"\n保存至: {output_path}")

# 截图验证
doc_out = fitz.open(output_path)
page_out = doc_out[0]
pix = page_out.get_pixmap(dpi=150)
pix.save(str(OUT_DIR / 'underline_06_full.png'))
print(f"全页截图: {pix.width}x{pix.height}")

# 局部截图
from PIL import Image
page_out.set_rotation(0)
pix_norot = page_out.get_pixmap(dpi=150)
img = Image.frombytes("RGB", [pix_norot.width, pix_norot.height], pix_norot.samples)
scale = 150 / 72
# 裁切 YA169B585G06 附近区域
crop_x0 = max(int(full_x0 * scale) - 30, 0)
crop_y0 = max(int(full_y0 * scale) - 30, 0)
crop_x1 = min(int(full_x1 * scale) + 50, pix_norot.width)
crop_y1 = min(int(full_y1 * scale) + 30, pix_norot.height)
crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
crop.save(str(OUT_DIR / 'underline_06_detail.png'))
print(f"局部截图: underline_06_detail.png ({crop.width}x{crop.height})")

doc_out.close()
print("完成")
