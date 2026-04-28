"""图纸批量编号替换系统 — Gradio Web 界面（优化版：预览/替换对比 + 图片缩放）"""

import os
import sys
import shutil
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

# 注意：不设置 FLAGS_use_mkldnn / FLAGS_enable_pir_api
# 这两个 CPU 加速 flag 一旦设置，会导致 GPU 静态推理引擎卡死（GPU stream nullptr），
# 且 Paddle 内部状态在进程生命周期内不可逆，无法在运行时清除。
# 不设置它们 CPU 模式仍可正常工作，仅损失少量 MKLDNN 加速。

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

from config import SUPPORTED_EXTENSIONS, DEFAULT_PREFIXES, set_ocr_mode
from modules.file_ingestion import load_file
from modules.pdf_vector_handler import is_vector_pdf, replace_text_in_pdf
from modules.region_detector import detect_all_regions, _enhance_vertical_lines
from modules.text_replacer import detect_cyan_boxes, replace_in_all_regions, detect_row_ys_for_red_box, clear_ocr_cache

SHRINK_RATIO = 0.025

def _shrink_canvas(img: Image.Image) -> Image.Image:
    """将图片内容缩小 SHRINK_RATIO，保持画布尺寸不变，周围填白。"""
    w, h = img.size
    new_w, new_h = int(w * (1 - SHRINK_RATIO)), int(h * (1 - SHRINK_RATIO))
    shrunk = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(shrunk, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "output"
RESULT_DIR.mkdir(exist_ok=True)

def load_model_ui(device_choice):
    """初始化混合模式：红框用 v5 定位，绿框/橙框+替换用 VLM。"""
    try:
        device = "gpu" if device_choice == "GPU" else "cpu"
        set_ocr_mode(device=device)
        clear_ocr_cache()

        status_parts = []

        from modules.region_detector import _get_ocr_v5
        _get_ocr_v5("en")
        status_parts.append("PaddleOCR v5 (红框定位): OK")
        logger.info("PaddleOCR v5 预热完成")

        from modules.vlm_ocr_engine import get_vlm_engine
        from config import VLLM_BASE_URL
        engine = get_vlm_engine()
        if engine.health_check():
            status_parts.append(f"VLM (绿/橙框+替换): OK ({VLLM_BASE_URL})")
        else:
            status_parts.append(f"VLM (绿/橙框+替换): 未响应 ({VLLM_BASE_URL})")

        return "混合模式 — " + " | ".join(status_parts)
    except Exception as e:
        logger.error(f"模型初始化失败: {e}", exc_info=True)
        return f"模型初始化失败: {e}"


# ── 一键替换：检测 + 替换合并执行 ──

def run_direct_replace(files, prefixes):
    """检测区域并立即执行替换，返回结果图片。"""
    if not files:
        gr.Warning("请先上传文件")
        return [], "替换失败：无文件"

    if not prefixes:
        prefixes = DEFAULT_PREFIXES

    clear_ocr_cache()

    out_dir = RESULT_DIR / "latest"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir_vector = out_dir / "vector"
    out_dir_ocr = out_dir / "ocr"
    out_dir_vector.mkdir(parents=True)
    out_dir_ocr.mkdir(parents=True)

    result_images = []
    total_replacements = 0

    for i, f in enumerate(files):
        fname = os.path.basename(f)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        logger.info(f"处理 ({i+1}/{len(files)}): {fname}")

        try:
            # 矢量 PDF：直接用 pdf_vector_handler 替换
            if ext == ".pdf" and is_vector_pdf(f):
                logger.info(f"  矢量PDF: {fname}")
                basename = os.path.splitext(fname)[0]
                output_path = str(out_dir_vector / ("H" + basename + ".pdf"))
                result = replace_text_in_pdf(f, output_path, prefixes=prefixes)
                count = result["total"]
                total_replacements += count
                import fitz
                doc = fitz.open(output_path)
                pix = doc[0].get_pixmap(dpi=150)
                mod_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                result_images.append((mod_img, f"{fname} ({count} 处替换, 矢量PDF)"))
                logger.info(f"  OK {fname}: {count} 处替换")
                continue

            # OCR 模式：检测 → 替换
            img_array, _ = load_file(f)
            enhanced = _enhance_vertical_lines(img_array)
            regions = detect_all_regions(enhanced, prefixes=prefixes)

            rot_code = regions.get("_metadata", {}).get("rotation")
            if rot_code is not None:
                img_array = cv2.rotate(img_array, rot_code)
                enhanced = cv2.rotate(enhanced, rot_code)

            red_bbox = regions.get("material_code_column")
            metadata = regions.get("_metadata", {})
            if red_bbox:
                table_search_bbox = metadata.get("table_search_area")
                row_ys = detect_row_ys_for_red_box(
                    enhanced, red_bbox, table_search_bbox=table_search_bbox)
                p_chars = "".join(p.upper() for p in prefixes)
                red_pattern = rf"\b[{p_chars}][A-Z0-9\-]{{8}}\b" if len(p_chars) > 1 else rf"\b{p_chars}[A-Z0-9\-]{{8}}\b"
                cyan_boxes, cyan_box_data = detect_cyan_boxes(
                    enhanced, red_bbox, row_ys,
                    pattern=red_pattern, prefixes=prefixes,
                )
                metadata["cyan_boxes"] = cyan_boxes
                metadata["cyan_box_data"] = cyan_box_data

            basename = os.path.splitext(fname)[0]
            modified, replacements = replace_in_all_regions(
                img_array, regions, filename=basename, prefixes=prefixes,
            )
            count = len(replacements)
            mod_img = _shrink_canvas(Image.fromarray(modified))

            if ext in ('.pdf', '.tif', '.tiff'):
                out_ext = '.tif'
            else:
                out_ext = ext
            output_path = str(out_dir_ocr / ("H" + basename + "-R" + out_ext))
            if out_ext == '.tif':
                mod_img.save(output_path, compression="tiff_lzw")
            else:
                mod_img.save(output_path, quality=95)

            total_replacements += count
            result_images.append((mod_img, f"{fname} ({count} 处替换)"))
            logger.info(f"  OK {fname}: {count} 处替换")

        except Exception as e:
            logger.error(f"  FAIL {fname}: {e}", exc_info=True)

    status = f"替换完成：{len(result_images)} 张图纸，共 {total_replacements} 处替换"
    return result_images, status


# ── 构建界面 ──

_APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
)
_APP_CSS = """
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

with gr.Blocks(title="图纸编号替换系统") as demo:

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
                value=["X", "Y"],
                label="2. 选择检测首字母",
                info="编号首字母匹配规则，替换为 H+原文",
            )
            replace_btn = gr.Button(
                "3. 开始替换", variant="primary", size="lg",
            )
            status_text = gr.Textbox(
                label="状态",
                value="等待上传文件...",
                interactive=False,
                lines=1,
            )
            with gr.Accordion("模型设置（混合模式：红框v5 + 绿/橙框VLM）", open=True):
                device_radio = gr.Radio(
                    choices=["CPU", "GPU"],
                    value="GPU",
                    label="运算设备",
                )
                load_model_btn = gr.Button("加载模型", variant="secondary", size="sm")
                load_model_btn.click(
                    fn=load_model_ui,
                    inputs=[device_radio],
                    outputs=[status_text],
                )

        # ── 右侧图片查看区 ──
        with gr.Column(scale=4):
            result_gallery = gr.Gallery(
                label="替换结果",
                columns=1,
                height=780,
                object_fit="contain",
                format="jpeg",
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

    replace_btn.click(
        fn=run_direct_replace,
        inputs=[file_input, prefix_input],
        outputs=[result_gallery, status_text],
    )

    # 页面加载时注入缩放 JS
    demo.load(fn=None, js=zoom_js)


if __name__ == "__main__":
    logger.info("启动 Web 服务: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=_APP_THEME, css=_APP_CSS)
