import fitz
import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output\pdf_replace')
OUT_DIR.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)
page = doc[0]

code_re = re.compile(r'(?<![A-Z0-9])(Y|X)(?=[A-Z0-9-])')

# 分析所有Y/X匹配的方向
blocks = page.get_text("dict")["blocks"]
h_matches = []  # 水平
v_matches = []  # 竖直

for b in blocks:
    if "lines" not in b:
        continue
    for l in b["lines"]:
        direction = l["dir"]
        for s in l["spans"]:
            txt = s["text"]
            for m in code_re.finditer(txt):
                info = {
                    "text": txt,
                    "pos": m.start(),
                    "origin": s["origin"],
                    "bbox": s["bbox"],
                    "dir": direction,
                    "font": s["font"],
                    "size": s["size"],
                    "flags": s["flags"],
                }
                if abs(direction[0]) > 0.5:  # 水平
                    h_matches.append(info)
                else:  # 竖直
                    v_matches.append(info)

print(f"水平匹配: {len(h_matches)}")
print(f"竖直匹配: {len(v_matches)}")

print("\n=== 竖直文本详情 ===")
for i, v in enumerate(v_matches):
    print(f"  [{i}] text='{v['text']}' dir={v['dir']} origin={v['origin']} bbox={v['bbox']} font={v['font']} size={v['size']}")

print("\n=== 水平文本前5个 ===")
for i, h in enumerate(h_matches[:5]):
    print(f"  [{i}] text='{h['text']}' dir={h['dir']} origin={h['origin']} bbox={h['bbox']} font={h['font']} size={h['size']}")

# 紫框区域的文本（Y>1500）
purple = [m for m in h_matches if m['origin'][1] > 1500]
green = [m for m in h_matches if m['origin'][1] <= 1500]
print(f"\n紫框区域（水平，Y>1500）: {len(purple)}")
print(f"绿框区域（水平，Y<=1500）: {len(green)}")

for p in purple:
    print(f"  text='{p['text']}' origin={p['origin']}")

doc.close()
