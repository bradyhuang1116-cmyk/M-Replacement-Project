import fitz
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

# Check ALL images including inline ones
print("=== 所有图片（包括inline） ===")
images = page.get_images(full=True)
for img in images:
    print(f"  xref={img[0]} smask={img[1]} width={img[2]} height={img[3]} bpc={img[4]} cs={img[5]} name={img[7]}")

# Check for Form XObjects that might contain the background
print("\n=== XObject资源 ===")
xref = page.xref
# Get page resources
res = page.get_text("rawdict")
print(f"页面blocks数: {len(res['blocks'])}")

# Check block types
img_blocks = [b for b in res['blocks'] if b['type'] == 1]  # image blocks
text_blocks = [b for b in res['blocks'] if b['type'] == 0]  # text blocks
print(f"图片blocks: {len(img_blocks)}")
print(f"文本blocks: {len(text_blocks)}")

for ib in img_blocks[:5]:
    print(f"  图片block: bbox={ib['bbox']} size={ib.get('width','?')}x{ib.get('height','?')}")

# Look at the content stream for image drawing operations
page.clean_contents()
content = page.read_contents().decode("latin-1")

# Find image references: /ImageName Do
import re
img_re = re.compile(r'/([\w]+)\s+Do')
img_ops = list(img_re.finditer(content))
print(f"\n=== 图片绘制操作 ({len(img_ops)}个) ===")
for m in img_ops[:20]:
    name = m.group(1)
    # Find the CTM (cm) before this operation
    start = max(0, m.start()-200)
    ctx = content[start:m.start()]
    cm_matches = list(re.finditer(r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+cm', ctx))
    if cm_matches:
        last_cm = cm_matches[-1]
        print(f"  /{name} Do  CTM: {last_cm.group(0)}")
    else:
        print(f"  /{name} Do  (no CTM found)")

doc.close()
