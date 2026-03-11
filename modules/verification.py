"""验证模块 — 生成对比图和 HTML 报告"""

import os
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from modules.file_ingestion import load_file


def generate_comparison_image(
    original_path: str, modified_path: str, max_width: int = 2000
) -> Image.Image:
    """生成原图与修改后图的并排对比图"""
    orig = Image.open(original_path).convert("RGB")
    mod = Image.open(modified_path).convert("RGB")

    # 统一高度
    h = max(orig.height, mod.height)
    ratio_orig = h / orig.height
    ratio_mod = h / mod.height
    orig = orig.resize((int(orig.width * ratio_orig), h), Image.Resampling.LANCZOS)
    mod = mod.resize((int(mod.width * ratio_mod), h), Image.Resampling.LANCZOS)

    # 拼接
    gap = 10
    total_w = orig.width + gap + mod.width
    comp = Image.new("RGB", (total_w, h + 40), "white")
    comp.paste(orig, (0, 40))
    comp.paste(mod, (orig.width + gap, 40))

    # 标签
    draw = ImageDraw.Draw(comp)
    draw.text((10, 5), "ORIGINAL", fill="red")
    draw.text((orig.width + gap + 10, 5), "MODIFIED", fill="green")

    # 缩放到最大宽度
    if total_w > max_width:
        ratio = max_width / total_w
        comp = comp.resize(
            (max_width, int(comp.height * ratio)), Image.Resampling.LANCZOS
        )

    return comp


def _img_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_html_report(results: list[dict], output_dir: str):
    """生成 HTML 验证报告"""
    report_path = os.path.join(output_dir, "verification_report.html")

    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    total_repls = sum(r.get("total", 0) for r in success)

    rows_html = []
    for r in results:
        fname = os.path.basename(r["file"])
        status_color = "green" if r["status"] == "success" else "red"
        status_text = r["status"].upper()

        if r["status"] == "success" and r.get("output_path"):
            # 缩略图
            try:
                thumb = Image.open(r["output_path"]).convert("RGB")
                thumb.thumbnail((400, 400))
                thumb_b64 = _img_to_base64(thumb)
                img_tag = f'<img src="data:image/jpeg;base64,{thumb_b64}" style="max-width:400px;cursor:pointer" onclick="window.open(\'{os.path.basename(r["output_path"])}\')">'
            except Exception:
                img_tag = "(预览不可用)"

            repls_text = "<br>".join(
                f"{old} → {new}"
                for old, new in r.get("replacements", [])
                if isinstance(old, str)
            ) or "(无替换)"

            rows_html.append(f"""
            <tr>
                <td>{fname}</td>
                <td style="color:{status_color}">{status_text}</td>
                <td>{r.get('method', '-')}</td>
                <td>{r.get('total', 0)}</td>
                <td style="font-size:12px">{repls_text}</td>
                <td>{img_tag}</td>
            </tr>""")
        else:
            rows_html.append(f"""
            <tr>
                <td>{fname}</td>
                <td style="color:{status_color}">{status_text}</td>
                <td colspan="4">{r.get('error', '-')}</td>
            </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>图纸处理验证报告</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #4CAF50; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .summary {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
</style>
</head><body>
<h1>图纸批量处理 - 验证报告</h1>
<div class="summary">
  <b>总计:</b> {len(results)} 个文件 |
  <b>成功:</b> {len(success)} |
  <b>失败:</b> {len(errors)} |
  <b>总替换数:</b> {total_repls}
</div>
<table>
<tr><th>文件名</th><th>状态</th><th>方式</th><th>替换数</th><th>替换详情</th><th>预览</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"验证报告已生成: {report_path}")
