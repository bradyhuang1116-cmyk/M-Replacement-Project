"""在PDF中查找 YA169B585G06 字符串，利用PDF文本特性"""
import fitz
import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

target = "06"
full_target = "YA169B585G06"

doc = fitz.open(pdf_path)
page = doc[0]
print(f"页面: rect={page.rect}, rotation={page.rotation}")

# === 方法1: 简单文本搜索 ===
print("\n=== 方法1: page.search_for() ===")
for query in [full_target, "YA169B585G", "G06", "06", "169B585"]:
    results = page.search_for(query)
    print(f"  搜索 '{query}': 找到 {len(results)} 个")
    for r in results:
        print(f"    位置: {r}")

# === 方法2: get_text 各种模式 ===
print("\n=== 方法2: get_text('text') 全文 ===")
text = page.get_text("text")
# 搜索目标字符串
for query in [full_target, "YA169B585G", "B585G06", "585G06", "G06"]:
    idx = text.find(query)
    if idx >= 0:
        context = text[max(0, idx-20):idx+len(query)+20]
        print(f"  找到 '{query}' 在位置 {idx}, 上下文: '{context}'")
    else:
        print(f"  未找到 '{query}'")

# === 方法3: rawdict 逐span查找 ===
print("\n=== 方法3: rawdict 逐span查找 ===")
page.set_rotation(0)
blocks = page.get_text("rawdict")["blocks"]
found_spans = []
for bi, b in enumerate(blocks):
    if "lines" not in b:
        continue
    for li, line in enumerate(b["lines"]):
        for si, span in enumerate(line["spans"]):
            chars = span.get("chars", [])
            if not chars:
                continue
            span_text = "".join(c["c"] for c in chars)
            # 查找包含目标的span
            if any(q in span_text for q in ["06", "G06", "585", "169"]):
                print(f"  block[{bi}] line[{li}] span[{si}]: text='{span_text}' "
                      f"dir={line['dir']} size={span['size']:.1f} "
                      f"bbox={[round(x,1) for x in span['bbox']]}")
                # 显示每个字符及bbox
                if "06" in span_text or "G06" in span_text:
                    found_spans.append(span_text)
                    for ci, c in enumerate(chars):
                        print(f"    char[{ci}]: '{c['c']}' bbox={[round(x,1) for x in c['bbox']]}")

# === 方法4: get_text("words") 查找 ===
print("\n=== 方法4: get_text('words') ===")
words = page.get_text("words")
for w in words:
    text_w = w[4]
    if any(q in text_w for q in ["06", "G06", "585", "169", "YA169"]):
        print(f"  word: '{text_w}' bbox=({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}, {w[3]:.1f}) block={w[5]} line={w[6]}")

# === 方法5: 查找所有包含 "06" 的文本 ===
print("\n=== 方法5: 所有包含'06'的span ===")
blocks2 = page.get_text("rawdict")["blocks"]
for bi, b in enumerate(blocks2):
    if "lines" not in b:
        continue
    for li, line in enumerate(b["lines"]):
        for si, span in enumerate(line["spans"]):
            chars = span.get("chars", [])
            if not chars:
                continue
            span_text = "".join(c["c"] for c in chars)
            if "06" in span_text:
                print(f"  '{span_text}' dir={line['dir']} bbox={[round(x,1) for x in span['bbox']]}")

# === 方法6: 检查PDF底层内容流 ===
print("\n=== 方法6: 页面xref和内容流信息 ===")
xref = page.xref
print(f"  页面xref: {xref}")
# 获取页面内容流文本
contents = page.get_contents()
print(f"  内容流xref列表: {contents}")
for cx in contents:
    stream = doc.xref_stream(cx)
    if stream:
        stream_text = stream.decode('latin-1', errors='replace')
        # 在内容流中搜索06相关
        for pattern in ["06", "G06", "585G06"]:
            positions = [m.start() for m in re.finditer(re.escape(pattern), stream_text)]
            if positions:
                print(f"  内容流xref={cx}: 找到'{pattern}' 在 {len(positions)} 处")
                for pos in positions[:5]:
                    context = stream_text[max(0, pos-50):pos+len(pattern)+50]
                    # 清理不可打印字符
                    context_clean = ''.join(c if c.isprintable() else '.' for c in context)
                    print(f"    pos={pos}: ...{context_clean}...")

doc.close()
print("\n完成")
