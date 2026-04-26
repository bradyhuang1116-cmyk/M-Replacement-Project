import fitz
import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 精确测试 rotate=90 vs rotate=270
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output\pdf_replace')

doc = fitz.open()
page = doc.new_page(width=400, height=400)
shape = page.new_shape()

# 目标: 文字"ABC"竖排，A在底部(y=300)，C在顶部(y=200)
# 这模拟dir=(0,-1): origin在底部，文字向上

# rotate=90: point=(50, 300)
shape.draw_rect(fitz.Rect(40, 200, 60, 310))
shape.finish(color=(1, 0, 0), width=0.5)
page.insert_text(fitz.Point(50, 300), "ABC", fontname="helv", fontsize=20, color=(1, 0, 0), rotate=90)

# rotate=270: point=(150, 200)
shape.draw_rect(fitz.Rect(140, 200, 160, 310))
shape.finish(color=(0, 0, 1), width=0.5)
page.insert_text(fitz.Point(150, 200), "ABC", fontname="helv", fontsize=20, color=(0, 0, 1), rotate=270)

# rotate=270: point=(250, 300)  (底部)
shape.draw_rect(fitz.Rect(240, 200, 260, 310))
shape.finish(color=(0, 0.5, 0), width=0.5)
page.insert_text(fitz.Point(250, 300), "ABC", fontname="helv", fontsize=20, color=(0, 0.5, 0), rotate=270)

shape.commit()

# 标注
page.insert_text(fitz.Point(30, 170), "rot90 pt(50,300)", fontsize=8, color=(1,0,0))
page.insert_text(fitz.Point(130, 170), "rot270 pt(150,200)", fontsize=8, color=(0,0,1))
page.insert_text(fitz.Point(220, 170), "rot270 pt(250,300)", fontsize=8, color=(0,0.5,0))

pix = page.get_pixmap(dpi=150)
pix.save(str(OUT_DIR / 'rotate_compare.png'))
doc.close()
print("已保存旋转对比图")
