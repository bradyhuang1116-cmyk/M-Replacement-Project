"""批量处理流水线"""

import os
import logging
import traceback
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import SUPPORTED_EXTENSIONS
from modules.file_ingestion import load_file
from modules.pdf_vector_handler import is_vector_pdf, replace_text_in_pdf
from modules.region_detector import detect_all_regions, _enhance_vertical_lines
from modules.text_replacer import replace_in_all_regions

logger = logging.getLogger(__name__)


def _scan_files(input_dir: str) -> list[str]:
    """扫描输入目录中所有支持的文件"""
    files = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(input_dir, fname))
    return files


def process_single_file(
    file_path: str,
    output_dir: str,
    region_config: dict = None,
    generate_debug: bool = False,
    regions_override: dict = None,
    prefixes: list[str] = None,
) -> dict:
    """
    处理单个图纸文件。

    Args:
        regions_override: 预检测的区域 dict（Phase 1 生成），跳过重新检测

    返回 dict:
      - file: 文件路径
      - status: "success" / "error"
      - method: "vector" / "ocr"
      - replacements: 替换列表
      - output_path: 输出文件路径
      - error: 错误信息（如有）
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower()
    orig_ext = ext  # 保存原始扩展名

    # ── 矢量 PDF 快速路径 ──
    if ext == ".pdf" and is_vector_pdf(file_path):
        vector_dir = os.path.join(output_dir, "vector")
        os.makedirs(vector_dir, exist_ok=True)
        output_path = os.path.join(vector_dir, "H" + basename + ".pdf")
        result = replace_text_in_pdf(file_path, output_path, prefixes=prefixes)
        return {
            "file": file_path,
            "status": "success",
            "method": "vector",
            "replacements": result["replacements"],
            "total": result["total"],
            "output_path": output_path,
        }

    # ── 扫描版PDF预处理 ──
    if ext == ".pdf":
        from modules.file_ingestion import convert_pdf_to_tif
        logger.info("检测到扫描版PDF，转换为TIF (600 DPI)")
        file_path = convert_pdf_to_tif(file_path, output_dir, dpi=600)
        ext = ".tif"

    # ── OCR 图像路径 ──
    img_array, metadata = load_file(file_path)
    logger.info(
        f"加载: {os.path.basename(file_path)} "
        f"({img_array.shape[1]}x{img_array.shape[0]}, {metadata['format']})"
    )

    # 区域检测
    if regions_override:
        regions = regions_override
        logger.info(f"使用预检测区域: {[k for k in regions if not k.startswith('_')]}")
    else:
        enhanced = _enhance_vertical_lines(img_array)
        regions = detect_all_regions(enhanced, region_config, prefixes=prefixes)

    # 如果检测时旋转了图像，将 img_array 也旋转（后续操作都在旋转后的图像上）
    rot_code = regions.get("_metadata", {}).get("rotation")
    if rot_code is not None:
        img_array = cv2.rotate(img_array, rot_code)
        logger.info(f"应用旋转到图像: rot_code={rot_code}")

    # ── 文件名首字母 → 输出抑制规则 ──
    # 内部依赖（如橙框搜索区以红框为锚）照常计算；这里只把不应进入最终输出
    # 的 region 设为 None，下游 cyan 生成 / text_replacer / debug 绘图 / y_boxes
    # 看到 None 一律跳过。
    #   Y         → 全输出
    #   B         → 不输出红框
    #   P/G/O/J   → 不输出绿框 + 橙框
    #   其它字母  → 仅输出工厂注意（红/绿/橙都不输出）
    first_letter = basename[0].upper() if basename else ''
    if first_letter == 'Y':
        suppress: set[str] = set()
    elif first_letter == 'B':
        suppress = {"material_code_column"}
    elif first_letter in ('P', 'G', 'O', 'J'):
        suppress = {"bottom_right_number", "top_left_number"}
    else:
        suppress = {"material_code_column", "bottom_right_number", "top_left_number"}
    for key in suppress:
        if regions.get(key) is not None:
            logger.info(f"  跳框规则 (首字母={first_letter or '?'}): 抑制 {key}")
            regions[key] = None

    detected = {k: v for k, v in regions.items() if v is not None}
    logger.info(f"检测到 {len(detected)} 个区域: {list(detected.keys())}")

    # 生成 cyan_box_data（红框预检测），替换阶段直接使用，不再重新 OCR
    red_bbox = regions.get("material_code_column")
    reg_metadata = regions.setdefault("_metadata", {})
    if red_bbox and not reg_metadata.get("cyan_box_data"):
        from modules.text_replacer import detect_cyan_boxes, detect_row_ys_for_red_box
        use_img = enhanced if 'enhanced' in locals() else img_array
        table_search_bbox = reg_metadata.get("table_search_area")
        row_ys = detect_row_ys_for_red_box(
            use_img, red_bbox, table_search_bbox=table_search_bbox)
        p_chars = "".join(p.upper() for p in (prefixes or ["Y"]))
        red_pattern = (rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1
                       else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b")
        cyan_boxes, cyan_box_data = detect_cyan_boxes(
            use_img, red_bbox, row_ys,
            pattern=red_pattern, prefixes=prefixes,
        )
        reg_metadata["cyan_boxes"] = cyan_boxes
        reg_metadata["cyan_box_data"] = cyan_box_data
        logger.info(f"生成 {len(cyan_boxes)} 个青色框（预检测）")

    # 调试：保存区域检测图
    if generate_debug:
        from modules.region_detector import draw_regions_debug

        debug_img = draw_regions_debug(img_array, regions)
        debug_path = os.path.join(output_dir, basename + "_debug_regions.jpg")
        Image.fromarray(debug_img).save(debug_path, quality=90)

    # 文本替换（使用预检测的 cyan_box_data，不重新 OCR）
    modified, replacements = replace_in_all_regions(img_array, regions, filename=basename, prefixes=prefixes)

    # 保存（输出格式与原始输入一致）
    ocr_dir = os.path.join(output_dir, "ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    if orig_ext in ('.tif', '.tiff'):
        out_ext = '.tif'
    elif orig_ext == '.pdf':
        out_ext = '.tif'
    else:
        out_ext = orig_ext
    output_path = os.path.join(ocr_dir, "H" + basename + "-R" + out_ext)
    if out_ext in ('.tif', '.tiff'):
        Image.fromarray(modified).save(output_path, compression="tiff_lzw")
    else:
        Image.fromarray(modified).save(output_path, quality=95)

    # 保存后验证
    from modules.text_replacer import verify_output
    verify_result = verify_output(output_path, regions, len(replacements))

    return {
        "file": file_path,
        "status": "success",
        "method": "ocr",
        "replacements": replacements,
        "total": len(replacements),
        "output_path": output_path,
        "regions_detected": list(detected.keys()),
        "verify": verify_result,
    }


def process_batch(
    input_dir: str,
    output_dir: str,
    region_config: dict = None,
    generate_debug: bool = False,
    prefixes: list[str] = None,
) -> list[dict]:
    """
    批量处理目录中所有图纸。

    返回处理结果列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = _scan_files(input_dir)
    if not files:
        logger.warning(f"输入目录没有找到支持的文件: {input_dir}")
        return []

    logger.info(f"共找到 {len(files)} 个文件待处理")
    results = []

    from modules.factory_note_pixel import clear_y_box_records, flush_y_boxes_csv
    clear_y_box_records()

    for file_path in tqdm(files, desc="处理图纸", unit="张"):
        try:
            result = process_single_file(
                file_path, output_dir, region_config, generate_debug,
                prefixes=prefixes,
            )
            results.append(result)
            logger.info(
                f"  ✓ {os.path.basename(file_path)}: "
                f"{result['total']} 处替换 ({result['method']})"
            )
        except Exception as e:
            logger.error(f"  ✗ {os.path.basename(file_path)}: {e}")
            results.append({
                "file": file_path,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # 统计
    success = sum(1 for r in results if r["status"] == "success")
    total_repls = sum(r.get("total", 0) for r in results if r["status"] == "success")
    logger.info(
        f"\n处理完成: {success}/{len(results)} 成功, 共 {total_repls} 处替换"
    )

    # Y 编号框坐标汇总（先于报告生成，避免报告异常时丢 CSV）
    try:
        flush_y_boxes_csv(os.path.join(output_dir, "y_boxes.csv"))
    except Exception as e:
        logger.warning(f"y_boxes.csv 写入失败（不影响结果）: {e}")

    # 生成报告
    _save_report(results, output_dir)

    return results


def _save_report(results: list[dict], output_dir: str):
    """生成简单的文本处理报告"""
    report_path = os.path.join(output_dir, "processing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"图纸批量处理报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")

        success = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]

        f.write(f"总计: {len(results)} 个文件\n")
        f.write(f"成功: {len(success)}\n")
        f.write(f"失败: {len(errors)}\n")
        total_repls = sum(r.get("total", 0) for r in success)
        f.write(f"总替换数: {total_repls}\n\n")

        if success:
            f.write("成功列表:\n")
            for r in success:
                f.write(
                    f"  {os.path.basename(r['file'])} "
                    f"[{r['method']}] {r['total']} 处替换\n"
                )
                for repl in r.get("replacements", []):
                    # OCR: (old, new); 矢量PDF: (old, new, page)
                    if not isinstance(repl, (tuple, list)) or len(repl) < 2:
                        continue
                    old, new = repl[0], repl[1]
                    if isinstance(old, str):
                        f.write(f"    {old} → {new}\n")

        if errors:
            f.write(f"\n失败列表:\n")
            for r in errors:
                f.write(f"  {os.path.basename(r['file'])}: {r.get('error', 'unknown')}\n")

    logger.info(f"报告已保存: {report_path}")
