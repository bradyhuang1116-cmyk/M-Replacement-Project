import fitz
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

# 1. 先看text extraction找到的所有Y/X codes
code_re = re.compile(r'(?<![A-Z0-9])(Y|X)(?=[A-Z0-9-])')
blocks = page.get_text("dict")["blocks"]
api_matches = []
for b in blocks:
    if "lines" not in b:
        continue
    for l in b["lines"]:
        for s in l["spans"]:
            txt = s["text"]
            for m in code_re.finditer(txt):
                api_matches.append((txt, m.start(), s["origin"], s.get("dir", (1,0))))

print(f"=== Text API找到 {len(api_matches)} 个Y/X匹配 ===")
for i, (txt, pos, origin, d) in enumerate(api_matches):
    print(f"  [{i}] text='{txt}' pos={pos} origin={origin} dir={d}")

# 2. 看content stream的原始内容
page.clean_contents()
raw = page.read_contents()
content = raw.decode("latin-1")

print(f"\n=== Content stream长度: {len(content)} bytes ===")

# 3. 找所有文本操作符: (text)Tj, [(...)offset(...)]TJ, <hex>Tj
# 找parenthesized strings
paren_re = re.compile(r'\(([^)]*)\)')
paren_matches = list(paren_re.finditer(content))
paren_with_yx = []
for m in paren_matches:
    txt = m.group(1)
    if code_re.search(txt):
        paren_with_yx.append((m.start(), txt))

print(f"\n=== 括号文本中找到 {len(paren_with_yx)} 个Y/X匹配 ===")
for pos, txt in paren_with_yx[:30]:
    print(f"  pos={pos} text='{txt}'")

# 4. 找hex strings
hex_re = re.compile(r'<([0-9A-Fa-f]+)>')
hex_matches = list(hex_re.finditer(content))
hex_with_yx = []
for m in hex_matches:
    hex_str = m.group(1)
    try:
        decoded = bytes.fromhex(hex_str).decode("latin-1")
        if code_re.search(decoded):
            hex_with_yx.append((m.start(), hex_str, decoded))
    except:
        pass

print(f"\n=== Hex文本中找到 {len(hex_with_yx)} 个Y/X匹配 ===")
for pos, h, d in hex_with_yx[:30]:
    print(f"  pos={pos} hex='{h}' decoded='{d}'")

# 5. 分析content stream中Tj/TJ操作符周围的内容
# 找所有Tj和TJ出现的位置
tj_re = re.compile(r'(Tj|TJ)')
tj_positions = list(tj_re.finditer(content))
print(f"\n=== 找到 {len(tj_positions)} 个Tj/TJ操作符 ===")

# 6. 查看一些包含Y/X的上下文
print("\n=== 在content stream中搜索Y和X字符的上下文 ===")
# 搜索单字符文本 (Y) 或 (X)
single_re = re.compile(r'\(([YXyx])\)')
single_matches = list(single_re.finditer(content))
print(f"找到 {len(single_matches)} 个单字符Y/X文本")
for m in single_matches[:20]:
    start = max(0, m.start()-80)
    end = min(len(content), m.end()+80)
    ctx = content[start:end].replace('\n', '\\n')
    print(f"  pos={m.start()} char='{m.group(1)}' context: ...{ctx}...")

# 7. 查看font encoding - 检查是否有ToUnicode映射
print("\n=== 页面字体信息 ===")
fonts = page.get_fonts()
for f in fonts[:20]:
    print(f"  xref={f[0]} name={f[3]} encoding={f[4]}")

doc.close()
