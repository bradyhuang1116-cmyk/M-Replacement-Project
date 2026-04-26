import fitz
import sys
import io
import re
from pathlib import Path
from PIL import Image

FONT_PATH = str(Path(__file__).parent / 'fonts' / 'JosefinSans-Light-5.ttf')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output\pdf_replace')
OUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUT_DIR / 'P131062A101-1_l脱敏_replaced25.pdf')

doc = fitz.open(pdf_path)
page = doc[0]
orig_rotation = page.rotation
print(f"页面: rect={page.rect}, rotation={orig_rotation}")

# 临时去掉旋转，让所有坐标在同一空间
page.set_rotation(0)
print(f"临时去旋转: rect={page.rect}")

code_re = re.compile(r'(?<![A-Z0-9])(Y|X)(?=[A-Z0-9-])')
blocks = page.get_text("rawdict")["blocks"]

replacements = []

for b in blocks:
    if "lines" not in b:
        continue
    for l in b["lines"]:
        direction = l["dir"]
        for s in l["spans"]:
            chars = s.get("chars", [])
            if not chars:
                continue
            txt = "".join(c["c"] for c in chars)
            m = code_re.search(txt)
            if not m:
                continue

            code_start = m.start()
            code_end = code_start + 1
            while code_end < len(chars) and re.match(r'[A-Z0-9a-z-]', chars[code_end]["c"]):
                code_end += 1

            code_text = txt[code_start:code_end]
            new_code_text = "H" + code_text

            # 跳过竖排文本（紫框区域不再替换）
            if abs(direction[0]) <= 0.5:
                continue

            code_chars = chars[code_start:code_end]
            x0 = min(c["bbox"][0] for c in code_chars)
            y0 = min(c["bbox"][1] for c in code_chars)
            x1 = max(c["bbox"][2] for c in code_chars)
            y1 = max(c["bbox"][3] for c in code_chars)

            char_bboxes = [c["bbox"] for c in code_chars]
            replacements.append({
                "code_text": code_text,
                "new_text": new_code_text,
                "code_bbox": (x0, y0, x1, y1),
                "char_bboxes": char_bboxes,
                "dir": direction,
                "size": s["size"],
                "span_text": txt,
            })

print(f"共找到 {len(replacements)} 个编号")

# === Step 1: 逐字符白色覆盖（保留字符间隙中的"原为"等图形）===
shape = page.new_shape()
for r in replacements:
    for cb in r["char_bboxes"]:
        shape.draw_rect(fitz.Rect(cb))
shape.finish(color=None, fill=(1, 1, 1))
shape.commit()

# === Step 2: 写入新文本 ===
for r in replacements:
    rect = fitz.Rect(r["code_bbox"])
    fontsize = r["size"] - 3

    # 水平文本：用 textbox 约束
    rc = page.insert_textbox(
        rect, r["new_text"],
        fontname="josefin", fontfile=FONT_PATH,
        fontsize=fontsize, color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_LEFT,
    )
    if rc < 0:
        rc = page.insert_textbox(
            rect, r["new_text"],
            fontname="josefin", fontfile=FONT_PATH,
            fontsize=0, color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )
    print(f"  H '{r['code_text']}' -> '{r['new_text']}' size={fontsize:.1f} rc={rc:.1f}")

# 恢复旋转
page.set_rotation(orig_rotation)
print(f"恢复旋转: rotation={page.rotation}")

doc.save(output_path)
doc.close()
print(f"保存至: {output_path}")

# === 截图 ===
doc_out = fitz.open(output_path)
page_out = doc_out[0]

# 全页
pix_full = page_out.get_pixmap(dpi=150)
pix_full.save(str(OUT_DIR / 'full_page.png'))
print(f"全页: {pix_full.width}x{pix_full.height}")

# 局部截图：去掉旋转后用rawdict坐标直接裁
page_out.set_rotation(0)
scale = 150 / 72
pix_norot = page_out.get_pixmap(dpi=150)
img = Image.frombytes("RGB", [pix_norot.width, pix_norot.height], pix_norot.samples)

for i, r in enumerate(replacements[:8]):
    fb = r["code_bbox"]
    x0 = max(int(fb[0] * scale) - 150, 0)
    y0 = max(int(fb[1] * scale) - 15, 0)
    x1 = min(int(fb[2] * scale) + 15, pix_norot.width)
    y1 = min(int(fb[3] * scale) + 15, pix_norot.height)
    crop = img.crop((x0, y0, x1, y1))
    crop.save(str(OUT_DIR / f'replace_{i}.png'))
    print(f"  replace_{i}.png: '{r['code_text']}' -> '{r['new_text']}'")

print("完成")
doc_out.close()
    