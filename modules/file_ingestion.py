"""文件加载模块 — 统一 PDF/TIF/JPG 输入为 RGB numpy 数组"""

import os
import logging
import math
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def load_file(file_path: str, dpi: int = 300):
    """
    加载图纸文件，返回 (image_rgb: np.ndarray, metadata: dict)

    metadata 包含:
      - source_path: 原始路径
      - format: 文件格式
      - is_vector_pdf: 是否矢量PDF
      - pages: 页数（PDF/TIFF多页时）
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
        raise ValueError(f"不支持的文件格式: {ext}")

    if ext == ".pdf":
        return _load_pdf(file_path, dpi)
    elif ext in (".tif", ".tiff"):
        return _load_tiff(file_path)
    else:
        return _load_image(file_path)


def _load_pdf(file_path: str, dpi: int):
    if fitz is None:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

    doc = fitz.open(file_path)
    page = doc[0]  # 处理第一页

    # 检测是否有矢量文本
    text = page.get_text().strip()
    is_vector = len(text) > 50

    # 渲染为图像
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    # 确保 RGB
    if pix.n == 4:
        img_array = img_array[:, :, :3]

    metadata = {
        "source_path": file_path,
        "format": "pdf",
        "is_vector_pdf": is_vector,
        "pages": len(doc),
        "dpi": dpi,
    }
    doc.close()
    return img_array, metadata


def _load_tiff(file_path: str):
    # 工程图纸可能很大，提升PIL尺寸限制
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(file_path).convert("RGB")
    n_frames = getattr(img, "n_frames", 1)
    img_array = np.array(img)

    metadata = {
        "source_path": file_path,
        "format": "tiff",
        "is_vector_pdf": False,
        "pages": n_frames,
    }
    return img_array, metadata


def _load_image(file_path: str):
    img = Image.open(file_path).convert("RGB")
    img_array = np.array(img)

    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    metadata = {
        "source_path": file_path,
        "format": ext,
        "is_vector_pdf": False,
        "pages": 1,
    }
    return img_array, metadata


def convert_pdf_to_tif(pdf_path: str, output_dir: str, dpi: int = 600, max_dim: int = 7000) -> str:
    """将PDF转换为TIF文件（高DPI），提升OCR质量。

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        dpi: 渲染DPI（默认600）
        max_dim: 最大边长限制（默认15000px）

    Returns:
        转换后的TIF文件路径
    """
    import fitz  # PyMuPDF
    import logging

    logger = logging.getLogger(__name__)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    tif_path = os.path.join(output_dir, f"{basename}_converted.tif")

    doc = fitz.open(pdf_path)
    page = doc[0]

    # 计算缩放比例，限制最大尺寸
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    rect = page.rect
    target_w = int(rect.width * zoom)
    target_h = int(rect.height * zoom)

    # 如果超过最大尺寸，降低zoom
    if max(target_w, target_h) > max_dim:
        scale = max_dim / max(target_w, target_h)
        zoom *= scale
        mat = fitz.Matrix(zoom, zoom)
        logger.info(f"PDF尺寸过大，降低DPI: {dpi} → {int(zoom * 72)}")

    pix = page.get_pixmap(matrix=mat, alpha=False)

    # PyMuPDF不支持TIF，用PIL保存
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    Image.fromarray(img_array).save(tif_path, compression="tiff_deflate")
    doc.close()

    logger.info(f"PDF转TIF完成: {tif_path} ({pix.width}x{pix.height})")
    return tif_path


def compress_tif(file_path: str, max_kb: int = 1300) -> str:
    """如果 TIF 文件超过 max_kb，按比例缩小尺寸直到文件大小合规。

    原地覆盖文件，返回路径。
    """
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb <= max_kb:
        return file_path

    Image.MAX_IMAGE_PIXELS = None
    orig_img = Image.open(file_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # 迭代缩放：每轮根据实际文件大小重新计算 scale
    img = orig_img
    w, h = orig_w, orig_h
    for _ in range(5):
        scale = math.sqrt(max_kb / size_kb) * 0.95  # 留 5% 余量
        w = int(w * scale)
        h = int(h * scale)
        img = orig_img.resize((w, h), Image.Resampling.LANCZOS)
        img.save(file_path, compression="tiff_deflate")
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb <= max_kb:
            break

    logger.info(f"TIF压缩: {orig_w}x{orig_h} → {w}x{h} ({size_kb:.0f}KB)")
    return file_path
