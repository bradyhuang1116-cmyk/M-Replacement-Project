"""矢量 PDF 直接文本替换 — 不需要 OCR，速度极快"""

import re
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from config import Y_PATTERN


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

        # 检查是否为垃圾OCR文本（大量乱码/重复字符）
        # 统计ASCII可打印字符比例
        printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
        ratio = printable / len(text) if text else 0

        # 如果可打印字符比例 <30%，认为是垃圾OCR
        return ratio >= 0.3
    except Exception:
        return False


def replace_text_in_pdf(file_path: str, output_path: str, pattern: str = None) -> dict:
    """
    在矢量 PDF 中直接替换 Y 开头编号为 HY 开头。

    返回 dict:
      - replacements: [(old_text, new_text, page_num), ...]
      - total: 替换总数
    """
    if fitz is None:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

    if pattern is None:
        pattern = Y_PATTERN

    doc = fitz.open(file_path)
    all_replacements = []

    for page_num, page in enumerate(doc):
        # 获取所有文本及其位置
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:  # 跳过图像块
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    # 查找 Y 开头编号
                    matches = list(re.finditer(pattern, text))
                    if not matches:
                        continue

                    # 获取 span 的位置和字体信息
                    rect = fitz.Rect(span["bbox"])
                    font_size = span["size"]
                    font_name = span["font"]

                    # 构造替换文本
                    new_text = text
                    for m in reversed(matches):
                        old = m.group()
                        replacement = "HY" + old[1:]
                        new_text = new_text[:m.start()] + replacement + new_text[m.end():]
                        all_replacements.append((old, replacement, page_num))

                    if new_text != text:
                        # 用 redaction 遮盖原文
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        page.apply_redactions()

                        # 在原位置插入新文本
                        # 尝试使用原字体，若不可用则用 helvetica
                        try:
                            page.insert_text(
                                rect.tl + fitz.Point(0, font_size * 0.85),
                                new_text,
                                fontsize=font_size,
                                fontname="helv",
                            )
                        except Exception:
                            page.insert_text(
                                rect.tl + fitz.Point(0, font_size * 0.85),
                                new_text,
                                fontsize=font_size,
                            )

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    doc.close()

    return {
        "replacements": all_replacements,
        "total": len(all_replacements),
    }
