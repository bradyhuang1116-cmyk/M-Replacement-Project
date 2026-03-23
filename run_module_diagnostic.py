"""各模块分步诊断脚本 — 计时 + 截图输出

用法:
    python run_module_diagnostic.py

测试图片: TIF_Undo/YA070A191P7933-1_0-脱敏.tif
输出目录: diagnostic_output/
"""

import os
import sys
import time
import logging

# ── 环境 ──
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnostic")

import cv2
import numpy as np

# ── 配置 ──
TEST_IMAGE = r"C:\Users\huang\Downloads\Downloads\mitsu\TIF_Undo\YA070A191P7933-1_0-脱敏.tif"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_img(name: str, img: np.ndarray):
    """保存截图到 OUTPUT_DIR，BGR/RGB 均可。"""
    path = os.path.join(OUTPUT_DIR, name)
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
    logger.info(f"  已保存: {name} ({img.shape[1]}x{img.shape[0]})")


def draw_bbox_on(img, bbox, color=(0, 255, 0), thickness=3, label=None):
    """在 RGB 图片副本上画框+标注，返回副本。"""
    vis = img.copy()
    cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, thickness)
    if label:
        cv2.putText(vis, label, (bbox.x, max(bbox.y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return vis


# ══════════════════════════════════════════════════════════════════
#  Step 1: file_ingestion — 加载图片
# ══════════════════════════════════════════════════════════════════
def step1_load():
    logger.info("=" * 60)
    logger.info("Step 1: file_ingestion.load_file()")
    logger.info("=" * 60)
    from modules.file_ingestion import load_file

    t0 = time.perf_counter()
    image, meta = load_file(TEST_IMAGE)
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")
    logger.info(f"  图片尺寸: {image.shape[1]}x{image.shape[0]} (WxH)")
    logger.info(f"  元数据: {meta}")

    # 缩略图
    thumb_h = 800
    scale = thumb_h / image.shape[0]
    thumb = cv2.resize(image, (int(image.shape[1] * scale), thumb_h))
    save_img("01_loaded_thumbnail.jpg", thumb)

    return image, meta


# ══════════════════════════════════════════════════════════════════
#  Step 2: 各搜索区域裁切 + 截图（跳过边框检测，直接用图片尺寸）
# ══════════════════════════════════════════════════════════════════
def step2_search_areas(image):
    logger.info("=" * 60)
    logger.info("Step 2: 搜索区域定义 + 裁切截图")
    logger.info("=" * 60)
    from modules.region_detector import BBox, _crop_and_scale

    img_h, img_w = image.shape[:2]

    # frame = 整张图片（跳过边框检测）
    frame = BBox(0, 0, img_w, img_h)
    logger.info(f"  frame (=图片): {frame}")

    # 红框搜索区域: frame左边界 ~ frame中轴线，高度=frame高
    red_search = BBox(frame.x, frame.y, frame.w // 2, frame.h)
    # 绿框搜索区域: 左 5/8 ~ 右边界, 上 5/6 ~ 图片下边界
    green_x = frame.x + int(frame.w * (5.0 / 8.0))
    green_y = frame.y + int(frame.h * (5.0 / 6.0))
    green_search = BBox(green_x, green_y, frame.x + frame.w - green_x, img_h - green_y)
    # 紫框搜索区域: 左边界~绿框左边界, 从下往上1.5/6~下边界
    purple_y = frame.y + frame.h - int(frame.h * 1.5 / 6.0)
    purple_search = BBox(frame.x, purple_y, green_x - frame.x, frame.y + frame.h - purple_y)
    # 橙框搜索区域: 图片左边界 ~ 右 2/8, 图片顶部 ~ frame高 2/12 再缩 1/3
    orange_h_full = frame.y + int(frame.h * (2.0 / 12.0))
    orange_search = BBox(0, 0, frame.x + int(frame.w * (2.0 / 8.0)),
                         orange_h_full - int(orange_h_full / 3))

    areas = {
        "red_search":    (red_search,    (255, 0, 0)),
        "green_search":  (green_search,  (0, 255, 0)),
        "purple_search": (purple_search, (128, 0, 128)),
        "orange_search": (orange_search, (255, 165, 0)),
    }

    # 总览图：所有搜索区域画在原图上
    vis = image.copy()
    for name, (bbox, color) in areas.items():
        cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 5)
        cv2.putText(vis, name, (bbox.x + 10, bbox.y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    thumb_h = 800
    scale_t = thumb_h / vis.shape[0]
    thumb = cv2.resize(vis, (int(vis.shape[1] * scale_t), thumb_h))
    save_img("02_all_search_areas.jpg", thumb)

    # 每个搜索区域的裁切 + 缩放后截图
    subs = {}
    for name, (bbox, color) in areas.items():
        logger.info(f"  {name}: {bbox}")
        sub, sc = _crop_and_scale(image, bbox)
        logger.info(f"    裁切: {bbox.w}x{bbox.h} → {sub.shape[1]}x{sub.shape[0]} (scale={sc:.3f})")
        save_img(f"02_{name}_crop.jpg", sub)
        subs[name] = (bbox, sub, sc)

    return subs, areas


# ══════════════════════════════════════════════════════════════════
#  Step 3: Phase B-1 红框 — MATERIAL CODE 列检测
# ══════════════════════════════════════════════════════════════════
def step3_red_detect(image, subs):
    logger.info("=" * 60)
    logger.info("Step 3: Phase B-1 _locate_material_code_column()")
    logger.info("=" * 60)
    from modules.region_detector import _locate_material_code_column, _map_bbox_back

    red_search, red_sub, red_scale = subs["red_search"]

    t0 = time.perf_counter()
    mat_result = _locate_material_code_column(red_sub)
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")

    mat_bbox = None
    if mat_result[0] is not None:
        mat_bbox_sub, direction = mat_result
        mat_bbox = _map_bbox_back(mat_bbox_sub, red_search, red_scale)
        logger.info(f"  子图结果: {mat_bbox_sub}")
        logger.info(f"  原图坐标: {mat_bbox}")
        logger.info(f"  表格方向: {direction}")

        # 子图上画红框
        vis_sub = draw_bbox_on(red_sub, mat_bbox_sub, (255, 0, 0), 3, "MATERIAL CODE")
        save_img("03_red_result_sub.jpg", vis_sub)

        # 原图上画红框
        vis = draw_bbox_on(image, mat_bbox, (255, 0, 0), 4, "MATERIAL CODE")
        thumb_h = 800
        sc = thumb_h / vis.shape[0]
        save_img("03_red_result_full.jpg", cv2.resize(vis, (int(vis.shape[1] * sc), thumb_h)))
    else:
        logger.warning("  未检测到 MATERIAL CODE 列")

    return mat_bbox


# ══════════════════════════════════════════════════════════════════
#  Step 4: Phase B-2 绿框 — 右下角编号
# ══════════════════════════════════════════════════════════════════
def step4_green_detect(image, subs):
    logger.info("=" * 60)
    logger.info("Step 4: Phase B-2 _locate_bottom_right_number()")
    logger.info("=" * 60)
    from modules.region_detector import _locate_bottom_right_number, _map_bbox_back

    green_search, green_sub, green_scale = subs["green_search"]

    t0 = time.perf_counter()
    br_result = _locate_bottom_right_number(green_sub)
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")

    br_bbox = None
    if br_result:
        br_text, br_bbox_sub = br_result
        br_bbox = _map_bbox_back(br_bbox_sub, green_search, green_scale)
        logger.info(f"  编号: '{br_text}'")
        logger.info(f"  子图结果: {br_bbox_sub}")
        logger.info(f"  原图坐标: {br_bbox}")

        vis_sub = draw_bbox_on(green_sub, br_bbox_sub, (0, 255, 0), 3, br_text)
        save_img("04_green_result_sub.jpg", vis_sub)

        vis = draw_bbox_on(image, br_bbox, (0, 255, 0), 4, br_text)
        thumb_h = 800
        sc = thumb_h / vis.shape[0]
        save_img("04_green_result_full.jpg", cv2.resize(vis, (int(vis.shape[1] * sc), thumb_h)))
    else:
        logger.warning("  未检测到右下角编号")

    return br_bbox


# ══════════════════════════════════════════════════════════════════
#  Step 5: Phase B-3 橙框 — 左上角编号
# ══════════════════════════════════════════════════════════════════
def step5_orange_detect(image, subs, mat_bbox):
    logger.info("=" * 60)
    logger.info("Step 5: Phase B-3 _locate_top_left_number()")
    logger.info("=" * 60)
    from modules.region_detector import _locate_top_left_number, _map_bbox_back, BBox

    orange_search, orange_sub, orange_scale = subs["orange_search"]

    # 将 mat_bbox 转换为橙框子图坐标
    mat_in_orange = None
    if mat_bbox is not None:
        mx = int((mat_bbox.x - orange_search.x) * orange_scale)
        my = int((mat_bbox.y - orange_search.y) * orange_scale)
        mw = int(mat_bbox.w * orange_scale)
        mh = int(mat_bbox.h * orange_scale)
        oh, ow = orange_sub.shape[:2]
        if mx < ow and my < oh and mx + mw > 0 and my + mh > 0:
            mat_in_orange = BBox(max(mx, 0), max(my, 0), mw, mh)

    t0 = time.perf_counter()
    tl_result = _locate_top_left_number(orange_sub, mat_in_orange)
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")

    tl_bbox = None
    if tl_result:
        tl_text, tl_bbox_sub = tl_result
        tl_bbox = _map_bbox_back(tl_bbox_sub, orange_search, orange_scale)
        logger.info(f"  编号: '{tl_text}'")
        logger.info(f"  子图结果: {tl_bbox_sub}")
        logger.info(f"  原图坐标: {tl_bbox}")

        vis_sub = draw_bbox_on(orange_sub, tl_bbox_sub, (255, 165, 0), 3, str(tl_text))
        save_img("05_orange_result_sub.jpg", vis_sub)

        vis = draw_bbox_on(image, tl_bbox, (255, 165, 0), 4, str(tl_text))
        thumb_h = 800
        sc = thumb_h / vis.shape[0]
        save_img("05_orange_result_full.jpg", cv2.resize(vis, (int(vis.shape[1] * sc), thumb_h)))
    else:
        logger.warning("  未检测到左上角编号")

    return tl_bbox


# ══════════════════════════════════════════════════════════════════
#  Step 6: Phase B-4 紫框 — 左下角竖排编号
# ══════════════════════════════════════════════════════════════════
def step6_purple_detect(image, subs):
    logger.info("=" * 60)
    logger.info("Step 6: Phase B-4 _locate_bottom_left_number()")
    logger.info("=" * 60)
    from modules.region_detector import _locate_bottom_left_number, _map_bbox_back

    purple_search, purple_sub, purple_scale = subs["purple_search"]

    t0 = time.perf_counter()
    bl_result = _locate_bottom_left_number(purple_sub)
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")

    bl_bbox = None
    if bl_result:
        bl_text, bl_bbox_sub, _ocr = bl_result
        bl_bbox = _map_bbox_back(bl_bbox_sub, purple_search, purple_scale)
        logger.info(f"  编号: '{bl_text}'")
        logger.info(f"  子图结果: {bl_bbox_sub}")
        logger.info(f"  原图坐标: {bl_bbox}")

        vis_sub = draw_bbox_on(purple_sub, bl_bbox_sub, (128, 0, 128), 3, bl_text)
        save_img("06_purple_result_sub.jpg", vis_sub)

        vis = draw_bbox_on(image, bl_bbox, (128, 0, 128), 4, bl_text)
        thumb_h = 800
        sc = thumb_h / vis.shape[0]
        save_img("06_purple_result_full.jpg", cv2.resize(vis, (int(vis.shape[1] * sc), thumb_h)))
    else:
        logger.warning("  未检测到左下角竖排编号")

    return bl_bbox


# ══════════════════════════════════════════════════════════════════
#  Step 7: 行检测 + 青框检测（text_replacer）
# ══════════════════════════════════════════════════════════════════
def step7_cyan_detect(image, mat_bbox, subs):
    logger.info("=" * 60)
    logger.info("Step 7: detect_row_ys_for_red_box() + detect_cyan_boxes()")
    logger.info("=" * 60)
    from modules.text_replacer import detect_row_ys_for_red_box, detect_cyan_boxes

    if mat_bbox is None:
        logger.warning("  无红框，跳过行检测和青框")
        return None, None

    red_search_bbox = subs["red_search"][0]

    # 行检测
    t0 = time.perf_counter()
    row_ys = detect_row_ys_for_red_box(image, mat_bbox, table_search_bbox=red_search_bbox)
    t1 = time.perf_counter()
    logger.info(f"  行检测耗时: {t1 - t0:.3f}s")
    logger.info(f"  检测到 {len(row_ys)} 条行线")
    if row_ys:
        logger.info(f"  前10条: {row_ys[:10]}")

    # 画行线
    vis = image.copy()
    cv2.rectangle(vis, (mat_bbox.x, mat_bbox.y), (mat_bbox.x2, mat_bbox.y2), (255, 0, 0), 2)
    for y in row_ys:
        abs_y = mat_bbox.y + y
        cv2.line(vis, (mat_bbox.x, abs_y), (mat_bbox.x2, abs_y), (255, 255, 0), 1)
    # 裁切红框附近区域保存
    pad = 50
    crop_y1 = max(0, mat_bbox.y - pad)
    crop_y2 = min(vis.shape[0], mat_bbox.y2 + pad)
    crop_x1 = max(0, mat_bbox.x - pad)
    crop_x2 = min(vis.shape[1], mat_bbox.x2 + pad)
    save_img("07_row_lines.jpg", vis[crop_y1:crop_y2, crop_x1:crop_x2])

    # 青框检测
    t2 = time.perf_counter()
    cyan_boxes, _ = detect_cyan_boxes(image, mat_bbox, row_ys)
    t3 = time.perf_counter()
    logger.info(f"  青框检测耗时: {t3 - t2:.3f}s")
    logger.info(f"  检测到 {len(cyan_boxes)} 个青框")

    # 画青框
    vis2 = image.copy()
    cv2.rectangle(vis2, (mat_bbox.x, mat_bbox.y), (mat_bbox.x2, mat_bbox.y2), (255, 0, 0), 2)
    for i, cb in enumerate(cyan_boxes):
        cv2.rectangle(vis2, (cb.x, cb.y), (cb.x2, cb.y2), (0, 255, 255), 2)
        cv2.putText(vis2, str(i), (cb.x + 5, cb.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    save_img("07_cyan_boxes.jpg", vis2[crop_y1:crop_y2, crop_x1:crop_x2])

    return row_ys, cyan_boxes


# ══════════════════════════════════════════════════════════════════
#  Step 8: 替换（text_replacer — 单区域测试）
# ══════════════════════════════════════════════════════════════════
def step8_replace(image, mat_bbox, row_ys):
    logger.info("=" * 60)
    logger.info("Step 8: replace_y_in_region_pixel() (红框)")
    logger.info("=" * 60)
    from modules.text_replacer import replace_y_in_region_pixel

    if mat_bbox is None or row_ys is None:
        logger.warning("  无红框/行线，跳过替换测试")
        return

    t0 = time.perf_counter()
    modified, replacements, cyan = replace_y_in_region_pixel(
        image, mat_bbox, use_grid_alignment=True, row_ys=row_ys,
    )
    elapsed = time.perf_counter() - t0

    logger.info(f"  耗时: {elapsed:.3f}s")
    logger.info(f"  替换数: {len(replacements)}")
    for old, new in replacements[:10]:
        logger.info(f"    {old} → {new}")

    # 裁切红框附近区域对比
    pad = 50
    y1 = max(0, mat_bbox.y - pad)
    y2 = min(image.shape[0], mat_bbox.y2 + pad)
    x1 = max(0, mat_bbox.x - pad)
    x2 = min(image.shape[1], mat_bbox.x2 + pad)
    save_img("08_before_replace.jpg", image[y1:y2, x1:x2])
    save_img("08_after_replace.jpg", modified[y1:y2, x1:x2])


# ══════════════════════════════════════════════════════════════════
#  Step 9: 总览（所有检测结果画在一张图上）
# ══════════════════════════════════════════════════════════════════
def step9_summary(image, mat_bbox, br_bbox, tl_bbox, bl_bbox):
    logger.info("=" * 60)
    logger.info("Step 9: 总览")
    logger.info("=" * 60)

    vis = image.copy()
    results = [
        ("material_code (红)", mat_bbox, (255, 0, 0)),
        ("bottom_right (绿)", br_bbox, (0, 255, 0)),
        ("top_left (橙)", tl_bbox, (255, 165, 0)),
        ("bottom_left (紫)", bl_bbox, (128, 0, 128)),
    ]
    for label, bbox, color in results:
        if bbox:
            cv2.rectangle(vis, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 4)
            cv2.putText(vis, label, (bbox.x, max(bbox.y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            logger.info(f"  {label}: {bbox}")
        else:
            logger.info(f"  {label}: 未检测到")

    thumb_h = 800
    sc = thumb_h / vis.shape[0]
    save_img("09_summary.jpg", cv2.resize(vis, (int(vis.shape[1] * sc), thumb_h)))


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════
def main():
    total_t0 = time.perf_counter()
    logger.info(f"测试图片: {TEST_IMAGE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    if not os.path.isfile(TEST_IMAGE):
        logger.error(f"测试图片不存在: {TEST_IMAGE}")
        sys.exit(1)

    # Step 1: 加载
    image, meta = step1_load()

    # Step 2: 搜索区域（直接用图片尺寸，不检测边框）
    subs, areas = step2_search_areas(image)

    # Step 3-6: 各 Phase 检测
    mat_bbox = step3_red_detect(image, subs)
    br_bbox = step4_green_detect(image, subs)
    tl_bbox = step5_orange_detect(image, subs, mat_bbox)
    bl_bbox = step6_purple_detect(image, subs)

    # Step 7: 行检测 + 青框
    row_ys, cyan_boxes = step7_cyan_detect(image, mat_bbox, subs)

    # Step 8: 替换测试
    step8_replace(image, mat_bbox, row_ys)

    # Step 9: 总览
    step9_summary(image, mat_bbox, br_bbox, tl_bbox, bl_bbox)

    total_elapsed = time.perf_counter() - total_t0
    logger.info("=" * 60)
    logger.info(f"全部完成! 总耗时: {total_elapsed:.1f}s")
    logger.info(f"截图输出: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
