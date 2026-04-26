import fitz
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output\pdf_replace')
output_path = str(OUT_DIR / 'P131062A101-1_l脱敏_replaced20.pdf')

doc = fitz.open(output_path)
page = doc[0]

# 正确的绿框截图（顶部区域）
clip = fitz.Rect(250, 150, 330, 500)
pix = page.get_pixmap(clip=clip, dpi=200)
pix.save(str(OUT_DIR / 'green_top_after20.png'))

# 紫框下半部分特写
clip2 = fitz.Rect(570, 1520, 640, 1660)
pix2 = page.get_pixmap(clip=clip2, dpi=300)
pix2.save(str(OUT_DIR / 'purple_detail_after20.png'))

# 整个紫框区域
clip3 = fitz.Rect(80, 1520, 900, 1660)
pix3 = page.get_pixmap(clip=clip3, dpi=200)
pix3.save(str(OUT_DIR / 'purple_wide_after20.png'))

doc.close()
print("done")
