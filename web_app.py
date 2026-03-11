"""图纸批量编号替换系统 — Gradio Web 界面（优化版：预览/替换对比 + 图片缩放）"""

import os
import sys
import shutil
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import gradio as gr
import cv2
from PIL import Image
import numpy as np

from config import SUPPORTED_EXTENSIONS, DEFAULT_PREFIXES
from modules.batch_processor import process_single_file
from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, draw_regions_debug, _detect_horizontal_lines, _detect_horizontal_lines_adaptive, BBox
from modules.text_replacer import detect_cyan_boxes, replace_in_all_regions, detect_row_ys_for_red_box

BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "output"
RESULT_DIR.mkdir(exist_ok=True)

# 检测阶段缓存：{filename: (img_array, regions)}
_detect_cache: dict[str, tuple[np.ndarray, dict]] = {}


def warmup_ui():
    """Web界面预热函数，使用真实TIF文件"""
    try:
        tif_dir = BASE_DIR / "TIF_Undo"
        tif_files = list(tif_dir.glob("*.tif"))
        if not tif_files:
            return "预热失败：未找到TIF文件"

        logger.info(f"使用 {tif_files[0].name} 预热模型...")
        from modules.text_replacer import _get_ocr
        _get_ocr("en")
        _get_ocr("ch")

        temp_out = BASE_DIR / "output" / "warmup_temp"
        temp_out.mkdir(exist_ok=True, parents=True)
        process_single_file(str(tif_files[0]), str(temp_out), generate_debug=False)
        shutil.rmtree(temp_out, ignore_errors=True)
        return "预热完成！"
    except Exception as e:
        return f"预热失败：{e}"


# ── Step 1: 检测区域预览 ──

def detect_regions(files, prefixes):
    """检测所有上传文件的区域，返回带彩色框的预览图。"""
    global _detect_cache
    _detect_cache.clear()

    if not files:
        gr.Warning("请先上传文件")
        return [], gr.update(interactive=False), "等待上传文件..."

    if not prefixes:
        prefixes = DEFAULT_PREFIXES

    preview_images = []

    for i, f in enumerate(files):
        fname = os.path.basename(f)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        logger.info(f"检测区域 ({i+1}/{len(files)}): {fname}")

        try:
            img_array, _ = load_file(f)
            regions = detect_all_regions(img_array, prefixes=prefixes)

            # 如果检测时旋转了图像，同步旋转 img_array
            rot_code = regions.get("_metadata", {}).get("rotation")
            if rot_code is not None:
                img_array = cv2.rotate(img_array, rot_code)

            # 生成青色框（红框内OCR识别匹配编号位置）
            red_bbox = regions.get("material_code_column")
            metadata = regions.get("_metadata", {})
            if red_bbox:
                table_search_bbox = metadata.get("table_search_area")
                row_ys = detect_row_ys_for_red_box(
                    img_array, red_bbox, table_search_bbox=table_search_bbox)
                p_chars = "".join(p.upper() for p in prefixes)
                red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
                cyan_boxes = detect_cyan_boxes(
                    img_array, red_bbox, row_ys,
                    pattern=red_pattern, prefixes=prefixes,
                )
                metadata["cyan_boxes"] = cyan_boxes
                logger.info(f"  生成 {len(cyan_boxes)} 个青色框")

            # 缓存检测结果供替换阶段使用
            _detect_cache[fname] = (img_array, regions)

            # 彩色框全图（仅全图，不裁剪搜索区域）
            debug_img = draw_regions_debug(img_array, regions)
            preview_images.append((
                Image.fromarray(debug_img),
                f"{fname} - 检测预览"
            ))

            logger.info(f"  检测完成: {fname}")

        except Exception as e:
            logger.error(f"  检测失败 {fname}: {e}", exc_info=True)

    has_results = len(preview_images) > 0
    status = f"检测完成：{len(preview_images)} 张图纸" if has_results else "未检测到有效图纸"
    return preview_images, gr.update(interactive=has_results), status


# ── Step 2: 确认并替换（使用检测阶段缓存的结果） ──

def run_replace(files, prefixes):
    """使用检测阶段缓存的 img_array 和 regions，直接执行替换。"""
    global _detect_cache

    if not files:
        gr.Warning("请先上传文件")
        return [], "替换失败：无文件"

    if not prefixes:
        prefixes = DEFAULT_PREFIXES

    out_dir = RESULT_DIR / "latest"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    result_images = []
    total_replacements = 0

    for i, f in enumerate(files):
        fname = os.path.basename(f)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        logger.info(f"替换 ({i+1}/{len(files)}): {fname}")

        try:
            cached = _detect_cache.get(fname)
            if cached:
                # 使用检测阶段缓存的结果，直接替换
                img_array, regions = cached
                basename = os.path.splitext(fname)[0]
                modified, replacements = replace_in_all_regions(
                    img_array, regions, filename=basename, prefixes=prefixes,
                )
                output_path = str(out_dir / (basename + "_modified.jpg"))
                Image.fromarray(modified).save(output_path, quality=95)
                count = len(replacements)
            else:
                # 无缓存，回退到完整流程
                logger.warning(f"  无检测缓存，回退到完整流程: {fname}")
                result = process_single_file(
                    f, str(out_dir), generate_debug=False, prefixes=prefixes,
                )
                output_path = result.get("output_path", "")
                replacements = result.get("replacements", [])
                count = result.get("total", 0)

            if output_path and os.path.exists(output_path):
                mod_img = Image.open(output_path).convert("RGB")
                total_replacements += count
                result_images.append((
                    mod_img,
                    f"{fname} ({count} 处替换)"
                ))

            logger.info(f"  OK {fname}: {count} 处替换")

        except Exception as e:
            logger.error(f"  FAIL {fname}: {e}", exc_info=True)

    _detect_cache.clear()
    status = f"替换完成：{len(result_images)} 张图纸，共 {total_replacements} 处替换"
    return result_images, status


# ── 构建界面 ──

with gr.Blocks(
    title="图纸编号替换系统",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
    ),
    css="""
    .main-title { text-align: center; margin-bottom: 0.2em; }
    .gradio-container { max-width: 2000px !important; }
    footer { display: none !important; }

    /* 图片缩放容器 */
    .zoomable-img {
        overflow: hidden;
        cursor: grab;
        max-height: 75vh;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #f8f8f8;
    }
    .zoomable-img:active { cursor: grabbing; }
    .zoomable-img img {
        transform-origin: 0 0;
        max-width: none !important;
        width: 100%;
        user-select: none;
        -webkit-user-drag: none;
    }

    /* 状态栏 */
    .status-bar {
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        text-align: center;
    }

    /* Tab样式优化 */
    .tab-nav button { font-size: 15px !important; font-weight: 600 !important; }
    """
) as demo:

    gr.Markdown("# 图纸编号替换系统", elem_classes="main-title")

    with gr.Row():
        # ── 左侧控制面板 ──
        with gr.Column(scale=1, min_width=280):
            file_input = gr.File(
                file_count="multiple",
                label="1. 上传图纸文件",
                file_types=[".tif", ".tiff", ".pdf", ".jpg", ".jpeg", ".png"],
                height=150,
            )
            prefix_input = gr.CheckboxGroup(
                choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                value=["X", "Y", "Z", "B"],
                label="2. 选择检测首字母",
                info="编号首字母匹配规则，替换为 H+原文",
            )
            detect_btn = gr.Button("3. 检测预览", variant="primary", size="lg")
            replace_btn = gr.Button(
                "4. 确认替换", variant="stop", size="lg",
                interactive=False,
            )
            status_text = gr.Textbox(
                label="状态",
                value="等待上传文件...",
                interactive=False,
                lines=1,
            )
            with gr.Accordion("高级选项", open=False):
                warmup_btn = gr.Button("预热模型", variant="secondary", size="sm")
                warmup_btn.click(fn=warmup_ui, inputs=[], outputs=[status_text])

        # ── 右侧图片查看区 ──
        with gr.Column(scale=4):
            with gr.Tabs() as tabs:
                with gr.Tab("检测预览", id="tab_preview"):
                    preview_gallery = gr.Gallery(
                        label="检测预览（各区域标注）",
                        columns=1,
                        height=780,
                        object_fit="contain",
                        preview=True,
                    )
                with gr.Tab("替换结果", id="tab_result"):
                    result_gallery = gr.Gallery(
                        label="替换结果",
                        columns=1,
                        height=780,
                        object_fit="contain",
                        preview=True,
                    )

    # ── 图片滚轮缩放 JS ──
    zoom_js = """
    () => {
        function addZoom() {
            document.querySelectorAll('.preview img, .grid-container img').forEach(img => {
                if (img.dataset.zoomBound) return;
                img.dataset.zoomBound = 'true';
                let scale = 1, tx = 0, ty = 0;
                let dragging = false, startX = 0, startY = 0, startTx = 0, startTy = 0;
                img.style.transformOrigin = '0 0';
                const container = img.parentElement;
                container.style.overflow = 'hidden';
                container.style.position = 'relative';

                function applyTransform() {
                    img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
                }

                // 滚轮缩放（以鼠标位置为中心）
                img.addEventListener('wheel', (e) => {
                    e.preventDefault();
                    const rect = container.getBoundingClientRect();
                    const mx = e.clientX - rect.left;
                    const my = e.clientY - rect.top;
                    const oldScale = scale;
                    const delta = e.deltaY > 0 ? -0.15 : 0.15;
                    scale = Math.max(0.2, Math.min(8, scale + delta));
                    // 缩放时保持鼠标指向的图片位置不变
                    tx = mx - (mx - tx) * (scale / oldScale);
                    ty = my - (my - ty) * (scale / oldScale);
                    applyTransform();
                }, { passive: false });

                // 鼠标拖拽平移
                img.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    dragging = true;
                    startX = e.clientX; startY = e.clientY;
                    startTx = tx; startTy = ty;
                    img.style.cursor = 'grabbing';
                });
                window.addEventListener('mousemove', (e) => {
                    if (!dragging) return;
                    tx = startTx + (e.clientX - startX);
                    ty = startTy + (e.clientY - startY);
                    applyTransform();
                });
                window.addEventListener('mouseup', () => {
                    if (!dragging) return;
                    dragging = false;
                    img.style.cursor = 'grab';
                });

                // 双击还原
                img.addEventListener('dblclick', () => {
                    scale = 1; tx = 0; ty = 0;
                    applyTransform();
                });

                img.style.cursor = 'grab';
            });
        }
        addZoom();
        const observer = new MutationObserver(() => setTimeout(addZoom, 200));
        observer.observe(document.body, { childList: true, subtree: true });
    }
    """

    # ── 事件绑定 ──

    # Step 1: 检测区域预览 → 写入预览 gallery
    detect_btn.click(
        fn=detect_regions,
        inputs=[file_input, prefix_input],
        outputs=[preview_gallery, replace_btn, status_text],
    )

    # Step 2: 确认替换 → 写入结果 gallery + 自动切换到结果 Tab
    replace_btn.click(
        fn=run_replace,
        inputs=[file_input, prefix_input],
        outputs=[result_gallery, status_text],
    ).then(
        fn=lambda: gr.update(selected="tab_result"),
        outputs=[tabs],
    )

    # 页面加载时注入缩放 JS
    demo.load(fn=None, js=zoom_js)


if __name__ == "__main__":
    logger.info("启动 Web 服务: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
