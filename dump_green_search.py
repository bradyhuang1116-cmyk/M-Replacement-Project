"""批量输出 TIF_Undo 文件夹中所有 Y 开头文件的绿框搜索区截图。"""

import os
import sys
import glob
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dump_green")

import cv2
import numpy as np

from modules.file_ingestion import load_file
from modules.region_detector import BBox, _auto_rotate_portrait

# ── 配置 ──
TIF_DIR = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output", "green_search_all")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 找到所有 Y 开头的文件
    patterns = [os.path.join(TIF_DIR, "Y*.*")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    logger.info(f"找到 {len(files)} 个 Y 开头文件")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        stem = os.path.splitext(fname)[0]
        logger.info(f"[{i+1}/{len(files)}] {fname}")

        try:
            image, _ = load_file(filepath)
        except Exception as e:
            logger.error(f"  加载失败: {e}")
            continue

        img_h, img_w = image.shape[:2]

        # 竖图旋转
        if img_h > img_w:
            image, _ = _auto_rotate_portrait(image)
            img_h, img_w = image.shape[:2]

        # 绿框搜索区
        green_x = int(img_w * (5.0 / 8.0))
        green_y = int(img_h * (5.0 / 6.0))
        green_crop = image[green_y:img_h, green_x:img_w]

        # 保存
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_green.jpg")
        if len(green_crop.shape) == 3 and green_crop.shape[2] == 3:
            cv2.imwrite(out_path, cv2.cvtColor(green_crop, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(out_path, green_crop)

        logger.info(f"  saved: {stem}_green.jpg ({green_crop.shape[1]}x{green_crop.shape[0]})")

    logger.info(f"完成！共 {len(files)} 个文件，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
