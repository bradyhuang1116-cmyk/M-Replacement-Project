import fitz
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

page.clean_contents()
content = page.read_contents().decode("latin-1")

# Find text rendering mode (Tr operator) near Y/X codes
# Look for patterns: number Tr ... (Y) or (X...)
# Tr values: 0=fill, 1=stroke, 2=fill+stroke, 3=invisible

# Find all Tr settings and the text that follows
tr_re = re.compile(r'(\d+)\s+Tr')
tr_matches = list(tr_re.finditer(content))
print(f"找到 {len(tr_matches)} 个Tr设置")

# Count by Tr value
from collections import Counter
tr_counts = Counter(m.group(1) for m in tr_matches)
print(f"Tr值分布: {dict(tr_counts)}")

# Check what Tr mode is active for the purple area text
# Purple area is near Y=1652 (from text API)
# Find the content around purple area matches
print("\n=== 紫框区域文本的Tr模式 ===")
# Find YA169B585G06 pattern in the content (purple area)
purple_re = re.compile(r'YA169B585G')
for m in purple_re.finditer(content):
    start = max(0, m.start()-300)
    ctx = content[start:m.start()]
    # Find last Tr before this text
    trs = list(tr_re.finditer(ctx))
    if trs:
        last_tr = trs[-1]
        print(f"  紫框文本 pos={m.start()} Tr={last_tr.group(1)}")
    # Show context
    ctx_short = content[max(0,m.start()-100):m.end()+50].replace('\n', '\\n')
    print(f"  context: {ctx_short}")

print("\n=== 绿框区域文本的Tr模式 ===")
# Green area - first YA169B585 (without G suffix) around Y=170
green_re = re.compile(r'\(Y\).*?A169B585')
for m in green_re.finditer(content[:200000]):  # first portion
    start = max(0, m.start()-300)
    ctx = content[start:m.start()]
    trs = list(tr_re.finditer(ctx))
    if trs:
        last_tr = trs[-1]
        print(f"  绿框文本 pos={m.start()} Tr={last_tr.group(1)}")
    ctx_short = content[max(0,m.start()-100):m.end()+50].replace('\n', '\\n')
    print(f"  context: {ctx_short}")

# Check if there are image objects on the page
print("\n=== 页面图片对象 ===")
images = page.get_images()
print(f"找到 {len(images)} 个图片")
for img in images[:5]:
    print(f"  xref={img[0]} width={img[2]} height={img[3]}")

# Check page structure
print(f"\n页面尺寸: {page.rect}")
print(f"MediaBox: {page.mediabox}")

doc.close()
