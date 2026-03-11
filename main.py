"""图纸批量 Y→HY 替换系统 — CLI 入口"""

import os
import sys
import json
import argparse
import logging

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.getLogger("ppocr").setLevel(logging.WARNING)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_detect(args):
    """Phase 1: 检测区域并保存供确认"""
    from modules.file_ingestion import load_file
    from modules.region_detector import detect_all_regions, draw_regions_debug, BBox
    from PIL import Image

    img_array, metadata = load_file(args.file)
    print(f"加载: {img_array.shape[1]}x{img_array.shape[0]}, {metadata['format']}")

    os.makedirs(args.output_dir, exist_ok=True)

    regions = detect_all_regions(img_array)

    # 保存 debug 可视化图
    debug_img = draw_regions_debug(img_array, regions)
    basename = os.path.splitext(os.path.basename(args.file))[0]
    debug_path = os.path.join(args.output_dir, f"{basename}_detect.jpg")
    Image.fromarray(debug_img).save(debug_path, quality=90)

    # 保存机器可读的区域 JSON
    regions_json = {}
    for name, bbox in regions.items():
        if name.startswith("_"):
            continue
        if bbox:
            regions_json[name] = bbox.to_dict()
    json_path = os.path.join(args.output_dir, f"{basename}_detect.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(regions_json, f, indent=2)

    # 保存 metadata
    meta = regions.get("_metadata", {})
    meta_json = {
        "method": meta.get("method", "unknown"),
        "table_direction": meta.get("table_direction"),
    }
    if meta.get("table_search_area"):
        meta_json["table_search_area"] = meta["table_search_area"].to_dict()
    meta_path = os.path.join(args.output_dir, f"{basename}_detect_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    # 打印摘要
    print(f"\n检测方法: {meta.get('method', 'unknown')}")
    print(f"表格方向: {meta.get('table_direction', 'N/A')}")
    for name, bbox in regions.items():
        if name.startswith("_"):
            continue
        if bbox:
            print(f"  {name}: {bbox}")
        else:
            print(f"  {name}: 未检测到")
    print(f"\n调试图: {debug_path}")
    print(f"区域JSON: {json_path}")


def cmd_single(args):
    """处理单个文件"""
    from modules.batch_processor import process_single_file
    from modules.region_detector import BBox

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载预检测区域（如有）
    regions_override = None
    if hasattr(args, 'regions') and args.regions:
        with open(args.regions, "r", encoding="utf-8") as f:
            rj = json.load(f)
        regions_override = {}
        for name, d in rj.items():
            regions_override[name] = BBox.from_dict(d)
        # 加载 metadata
        meta_path = args.regions.replace("_detect.json", "_detect_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metadata = {"method": meta.get("method"), "table_direction": meta.get("table_direction")}
            if meta.get("table_search_area"):
                metadata["table_search_area"] = BBox.from_dict(meta["table_search_area"])
            regions_override["_metadata"] = metadata

    result = process_single_file(
        args.file,
        args.output_dir,
        generate_debug=args.debug,
        regions_override=regions_override,
    )
    if result["status"] == "success":
        print(f"\n处理成功!")
        print(f"  方式: {result['method']}")
        print(f"  替换数: {result['total']}")
        print(f"  输出: {result['output_path']}")
        for old, new in result.get("replacements", []):
            if isinstance(old, str):
                print(f"    {old} → {new}")
        verify = result.get("verify")
        if verify:
            status = "通过" if verify["match"] else "不匹配"
            print(f"  验证: 回检到 {verify['found_hy']}/{verify['expected']} 个 HY ({status})")
    else:
        print(f"\n处理失败: {result.get('error', 'unknown')}")
        sys.exit(1)


def cmd_batch(args):
    """批量处理目录"""
    from modules.batch_processor import process_batch
    from modules.verification import generate_html_report

    results = process_batch(
        args.input_dir,
        args.output_dir,
        generate_debug=args.debug,
    )

    if args.verify:
        generate_html_report(results, args.output_dir)

    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"\n完成: {success}/{total} 成功")


def cmd_debug_regions(args):
    """调试：可视化区域检测结果（旧版兼容）"""
    from modules.file_ingestion import load_file
    from modules.region_detector import detect_all_regions, draw_regions_debug
    from PIL import Image

    img_array, metadata = load_file(args.file)
    print(f"加载: {img_array.shape[1]}x{img_array.shape[0]}, {metadata['format']}")

    regions = detect_all_regions(img_array)
    for name, bbox in regions.items():
        if name.startswith("_"):
            continue
        if bbox:
            print(f"  {name}: {bbox}")
        else:
            print(f"  {name}: 未检测到")

    debug_img = draw_regions_debug(img_array, regions)
    out_path = args.output or "debug_regions.jpg"
    Image.fromarray(debug_img).save(out_path, quality=90)
    print(f"\n调试图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="图纸批量 Y→HY 替换系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Phase 1: 检测区域（生成调试图供确认）
  python main.py detect drawing.tif -o output/

  # Phase 2: 用确认后的区域执行替换
  python main.py single drawing.tif -o output/ --regions output/drawing_detect.json

  # 直接处理（跳过确认）
  python main.py single drawing.pdf -o output/

  # 批量处理目录
  python main.py batch input_drawings/ output/ --verify
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # ── detect (Phase 1) ──
    sp_detect = subparsers.add_parser("detect", help="检测区域（Phase 1）")
    sp_detect.add_argument("file", help="输入文件路径 (PDF/TIF/JPG)")
    sp_detect.add_argument("-o", "--output-dir", default="output", help="输出目录")

    # ── single (Phase 2 or direct) ──
    sp_single = subparsers.add_parser("single", help="处理单个文件")
    sp_single.add_argument("file", help="输入文件路径 (PDF/TIF/JPG)")
    sp_single.add_argument("-o", "--output-dir", default="output", help="输出目录")
    sp_single.add_argument("--debug", action="store_true", help="生成区域检测调试图")
    sp_single.add_argument("--regions", help="预检测的区域JSON文件路径（跳过重新检测）")

    # ── batch ──
    sp_batch = subparsers.add_parser("batch", help="批量处理目录")
    sp_batch.add_argument("input_dir", help="输入目录")
    sp_batch.add_argument("output_dir", help="输出目录")
    sp_batch.add_argument("--verify", action="store_true", help="生成HTML验证报告")
    sp_batch.add_argument("--debug", action="store_true", help="生成区域检测调试图")

    # ── debug-regions (旧版兼容) ──
    sp_debug = subparsers.add_parser("debug-regions", help="调试区域检测")
    sp_debug.add_argument("file", help="输入文件路径")
    sp_debug.add_argument("-o", "--output", help="输出调试图路径")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "debug-regions":
        cmd_debug_regions(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
