"""保存时把 TEXT_COMMITTED 的框落到 PIL 图像上：白底矩形 + 黑色居中文字。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .font_config import find_font


@dataclass
class CommittedBox:
    x: int
    y: int
    w: int
    h: int
    text: str


def _fit_font(text: str, box_w: int, box_h: int,
              font_path: str | None) -> ImageFont.ImageFont:
    """选一个能塞进 box_w/box_h 的最大字号。"""
    # 字高上限：box_h 的 70%；下限 8
    max_size = max(int(box_h * 0.7), 8)
    min_size = 8
    if font_path is None:
        return ImageFont.load_default()

    # 二分找最大可放下的字号
    lo, hi = min_size, max_size
    best = ImageFont.truetype(font_path, min_size)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = f.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= box_w * 0.92 and th <= box_h * 0.85:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_committed_boxes(image_path: str | Path,
                            boxes: list[CommittedBox],
                            output_path: str | Path) -> Path:
    """读 image_path、把所有 boxes 渲染为白底黑字、保存到 output_path。

    Returns
    -------
    output_path 的 Path 对象。
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as src:
        img = src.convert("RGB") if src.mode != "RGB" else src.copy()

    draw = ImageDraw.Draw(img)
    font_path = find_font()

    for b in boxes:
        if b.w <= 0 or b.h <= 0:
            continue
        draw.rectangle(
            [b.x, b.y, b.x + b.w, b.y + b.h],
            fill=(255, 255, 255),
        )
        text = b.text or ""
        if not text:
            continue
        font = _fit_font(text, b.w, b.h, font_path)
        tb = font.getbbox(text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        cx = b.x + (b.w - tw) // 2 - tb[0]
        cy = b.y + (b.h - th) // 2 - tb[1]
        draw.text((cx, cy), text, fill=(0, 0, 0), font=font)

    suffix = output_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        img.save(output_path, compression="tiff_lzw")
    else:
        img.save(output_path)
    return output_path
