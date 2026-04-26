"""矢量 PDF 直接文本替换 — 基于 rawdict 字符级精准替换，不需要 OCR"""

import re
import os
import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from config import DEFAULT_PREFIXES, NEW_PREFIX, PDF_FONT_PATH

logger = logging.getLogger(__name__)


def is_vector_pdf(file_path: str, min_text_len: int = 50) -> bool:
    """检查 PDF 是否包含可提取的矢量文本（排除垃圾OCR文本）"""
    if fitz is None:
        return False
    try:
        doc = fitz.open(file_path)
        page = doc[0]
        text = page.get_text().strip()
        doc.close()

        if len(text) < min_text_len:
            return False

        printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
        ratio = printable / len(text) if text else 0
        return ratio >= 0.3
    except Exception:
        return False


def replace_text_in_pdf(
    file_path: str, output_path: str, prefixes: list[str] = None
) -> dict:
    """
    在矢量 PDF 中替换匹配编号，使用 rawdict 字符级精准白填充 + textbox 重写。

    流程：
      1. rawdict 提取每个 span 的字符级 bbox
      2. 正则匹配编号（如 YA026D941）
      3. 逐字符白色矩形覆盖
      4. insert_textbox 写入新文本（H前缀 + 原编号）

    返回 dict:
      - replacements: [(old_text, new_text, page_num), ...]
      - total: 替换总数
    """
    if fitz is None:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

    prefixes = prefixes or DEFAULT_PREFIXES
    prefixes_upper = [p.upper() for p in prefixes]
    p_chars = "".join(prefixes_upper)

    # 构建正则：匹配前缀开头 + 至少4位字母数字
    if len(p_chars) == 1:
        code_re = re.compile(rf'(?<![A-Z0-9]){p_chars}(?=[A-Z0-9-])')
    else:
        code_re = re.compile(rf'(?<![A-Z0-9])[{p_chars}](?=[A-Z0-9-])')

    doc = fitz.open(file_path)
    all_replacements = []

    for page_num, page in enumerate(doc):
        orig_rotation = page.rotation
        page.set_rotation(0)

        blocks = page.get_text("rawdict")["blocks"]
        replacements = []

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                direction = line["dir"]
                # 跳过竖排文本
                if abs(direction[0]) <= 0.5:
                    continue

                for span in line["spans"]:
                    chars = span.get("chars", [])
                    if not chars:
                        continue
                    txt = "".join(c["c"] for c in chars)
                    m = code_re.search(txt)
                    if not m:
                        continue

                    code_start = m.start()
                    code_end = code_start + 1
                    while code_end < len(chars) and re.match(
                        r'[A-Z0-9a-z-]', chars[code_end]["c"]
                    ):
                        code_end += 1

                    code_text = txt[code_start:code_end]
                    if len(code_text) < 7:
                        continue

                    new_text = NEW_PREFIX + code_text
                    code_chars = chars[code_start:code_end]

                    x0 = min(c["bbox"][0] for c in code_chars)
                    y0 = min(c["bbox"][1] for c in code_chars)
                    x1 = max(c["bbox"][2] for c in code_chars)
                    y1 = max(c["bbox"][3] for c in code_chars)

                    replacements.append({
                        "code_text": code_text,
                        "new_text": new_text,
                        "code_bbox": (x0, y0, x1, y1),
                        "char_bboxes": [c["bbox"] for c in code_chars],
                        "size": span["size"],
                    })

        if not replacements:
            page.set_rotation(orig_rotation)
            continue

        # Step 1: 逐字符白色覆盖
        shape = page.new_shape()
        for r in replacements:
            for cb in r["char_bboxes"]:
                shape.draw_rect(fitz.Rect(cb))
        shape.finish(color=None, fill=(1, 1, 1))
        shape.commit()

        # Step 2: 写入新文本
        for r in replacements:
            rect = fitz.Rect(r["code_bbox"])
            fontsize = r["size"] - 3

            rc = page.insert_textbox(
                rect, r["new_text"],
                fontname="josefin", fontfile=PDF_FONT_PATH,
                fontsize=fontsize, color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT,
            )
            if rc < 0:
                page.insert_textbox(
                    rect, r["new_text"],
                    fontname="josefin", fontfile=PDF_FONT_PATH,
                    fontsize=0, color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                )

            all_replacements.append(
                (r["code_text"], r["new_text"], page_num)
            )
            logger.info(
                f"  PDF替换: '{r['code_text']}' → '{r['new_text']}' "
                f"page={page_num} size={fontsize:.1f}"
            )

        page.set_rotation(orig_rotation)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info(f"PDF替换完成: {len(all_replacements)} 处, 保存至 {output_path}")
    return {
        "replacements": all_replacements,
        "total": len(all_replacements),
    }
