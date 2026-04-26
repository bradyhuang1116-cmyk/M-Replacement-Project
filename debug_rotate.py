import fitz
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

code_re = re.compile(r'(?<![A-Z0-9])(Y|X)(?=[A-Z0-9-])')

# 原始状态
print("=== rotation=0 ===")
for b in page.get_text("dict")["blocks"]:
    if "lines" not in b:
        continue
    for l in b["lines"]:
        for s in l["spans"]:
            if code_re.search(s["text"]) and abs(l["dir"][0]) <= 0.5:
                print(f"  竖直: '{s['text']}' dir={l['dir']}")

# 旋转90
page.set_rotation(90)
print("\n=== rotation=90 ===")
v_count = 0
h_count = 0
for b in page.get_text("dict")["blocks"]:
    if "lines" not in b:
        continue
    for l in b["lines"]:
        for s in l["spans"]:
            if code_re.search(s["text"]):
                d = l["dir"]
                is_h = abs(d[0]) > 0.5
                if not is_h:
                    v_count += 1
                else:
                    h_count += 1
                    # 检查是不是之前的竖直文本（现在变水平了）
                    if "G06" in s["text"] or "G08" in s["text"] or "B035" in s["text"] or "C620" in s["text"]:
                        print(f"  前竖直→水平: '{s['text']}' dir={d} origin={s['origin']}")

print(f"  竖直: {v_count}, 水平: {h_count}")

# 试试旋转矩阵方式
page.set_rotation(0)
# 用derotation_matrix
print(f"\n页面rotation属性: {page.rotation}")

doc.close()
