"""检查下划线是否溢出：对比红线范围与文字实际范围"""
import fitz
import sys
import io
from pathlib import Path
from PIL import Image, ImageDraw

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r'C:\Users\Brady Huang\Downloads\TIF_Undo\P131062A101-1_l脱敏.pdf'
OUT_DIR = Path(r'C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project\diagnostic_output')

doc = fitz.open(pdf_path)
page = doc[0]
page.set_rotation(0)

blocks = page.get_text("rawdict")["blocks"]
scale = 150 / 72

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

            idx = span_text.index("YA169B585G06")
            code_chars = chars[idx:idx + 12]

            print("=== 整个span字符及bbox ===")
            for i, c in enumerate(chars):
                marker = " <-- YA169B585G06" if idx <= i < idx + 12 else ""
                print(f"  [{i:2d}] '{c['c']}' y0={c['bbox'][1]:.1f} y1={c['bbox'][3]:.1f}{marker}")

            # YA169B585G06 的精确范围
            code_y0 = min(c["bbox"][1] for c in code_chars)
            code_y1 = max(c["bbox"][3] for c in code_chars)
            code_x0 = min(c["bbox"][0] for c in code_chars)
            code_x1 = max(c["bbox"][2] for c in code_chars)

            # 红线实际绘制位置
            line_x = code_x1 + 2  # margin=2
            line_y_start = code_y0
            line_y_end = code_y1

            print(f"\n=== 溢出分析 ===")
            print(f"YA169B585G06 bbox: x({code_x0:.1f}-{code_x1:.1f}) y({code_y0:.1f}-{code_y1:.1f})")
            print(f"红线位置: x={line_x:.1f}, y({line_y_start:.1f}-{line_y_end:.1f})")

            # 检查红线是否覆盖到相邻文字
            # 上方: 'Y'(idx=5) 上面是 ' '(idx=4), bbox y
            if idx > 0:
                prev_char = chars[idx - 1]
                gap = code_y1 - prev_char["bbox"][1]  # 红线顶端 vs 前一个字符底端
                print(f"上方邻居: '{prev_char['c']}' y=({prev_char['bbox'][1]:.1f}-{prev_char['bbox'][3]:.1f})")
                print(f"  红线顶端({code_y1:.1f}) vs 上方字符底端({prev_char['bbox'][1]:.1f}), 间距={prev_char['bbox'][1] - code_y1:.1f}pt")
                if code_y1 > prev_char["bbox"][1]:
                    print(f"  ⚠ 溢出! 红线向上超出 {code_y1 - prev_char['bbox'][1]:.1f}pt 进入上方字符区域")
                else:
                    print(f"  ✓ 未溢出")

            # 下方: '6'(idx+11) 下面没有更多字符(idx+12超出)
            if idx + 12 < len(chars):
                next_char = chars[idx + 12]
                print(f"下方邻居: '{next_char['c']}' y=({next_char['bbox'][1]:.1f}-{next_char['bbox'][3]:.1f})")
                print(f"  红线底端({code_y0:.1f}) vs 下方字符顶端({next_char['bbox'][3]:.1f}), 间距={code_y0 - next_char['bbox'][3]:.1f}pt")
            else:
                print(f"下方: 无更多字符 (已是span末尾)")
                print(f"  ✓ 未溢出")

            # 生成带标注的诊断图
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)

            # 裁切范围：包含 span 全部 + 周围
            span_y0 = min(c["bbox"][1] for c in chars)
            span_y1 = max(c["bbox"][3] for c in chars)
            crop_x0 = max(int(code_x0 * scale) - 60, 0)
            crop_y0 = max(int(span_y0 * scale) - 20, 0)
            crop_x1 = min(int(code_x1 * scale) + 80, pix.width)
            crop_y1 = min(int(span_y1 * scale) + 20, pix.height)

            # 在全图上画标注
            # 蓝色框: YA169B585G06 范围
            draw.rectangle([
                int(code_x0 * scale), int(code_y0 * scale),
                int(code_x1 * scale), int(code_y1 * scale)
            ], outline="blue", width=2)

            # 绿色框: 前缀 " 06  " 范围
            prefix_chars = chars[:idx]
            if prefix_chars:
                py0 = min(c["bbox"][1] for c in prefix_chars)
                py1 = max(c["bbox"][3] for c in prefix_chars)
                px0 = min(c["bbox"][0] for c in prefix_chars)
                px1 = max(c["bbox"][2] for c in prefix_chars)
                draw.rectangle([
                    int(px0 * scale), int(py0 * scale),
                    int(px1 * scale), int(py1 * scale)
                ], outline="green", width=2)

            # 红线位置标注
            draw.line([
                int(line_x * scale), int(line_y_start * scale),
                int(line_x * scale), int(line_y_end * scale)
            ], fill="red", width=3)

            crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            crop.save(str(OUT_DIR / 'overflow_check.png'))
            print(f"\n诊断图已保存: overflow_check.png")
            print("  蓝框=YA169B585G06范围, 绿框=前缀' 06  '范围, 红线=下划线位置")

doc.close()
print("\n完成")
