"""V6 合入主代码后的烟雾测试。

直接调 detect_all_regions（应走 Phase C v6 路径），验证：
  1. regions["factory_note_codes"] 字段存在且非空（4 个已知文件至少各 1 个 Y）；
  2. metadata["factory_note_source"] == "v6"（没 fallback 到 PP-DocLayoutV3）；
  3. 不抛异常。
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

from modules.file_ingestion import load_file
from modules.region_detector import detect_all_regions, _enhance_vertical_lines
from modules.factory_note_pixel import flush_y_boxes_csv, clear_y_box_records
from modules.docker_manager import ensure_vlm_ready

INPUT_DIR = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
KEYWORDS = ["P3335", "A226", "C857", "A110-2"]

files = []
for f in sorted(os.listdir(INPUT_DIR)):
    if os.path.splitext(f)[1].lower() not in (".tif", ".tiff"):
        continue
    for kw in KEYWORDS:
        if kw in f:
            files.append(os.path.join(INPUT_DIR, f))
            break

assert files, f"no test files matched {KEYWORDS}"
print(f"测试文件: {[os.path.basename(f) for f in files]}")

print("启动 VLM ...")
ok, msg = ensure_vlm_ready()
if not ok:
    print(f"VLM 启动失败: {msg}")
    sys.exit(1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "v6_merge_smoke")
os.makedirs(OUT_DIR, exist_ok=True)
clear_y_box_records()

summary = []
for fp in files:
    base = os.path.basename(fp)
    print("=" * 60)
    print(f"处理: {base}")
    img, _ = load_file(fp)
    enhanced = _enhance_vertical_lines(img)
    regions = detect_all_regions(enhanced)

    fn_codes = regions.get("factory_note_codes", [])
    src = regions.get("_metadata", {}).get("factory_note_source", "?")
    print(f"  source = {src}, codes = {len(fn_codes)}")
    for fc in fn_codes:
        b = fc["bbox"]
        print(f"    {fc['code']} @ BBox(x={b.x},y={b.y},w={b.w},h={b.h}) conf={fc['confidence']}")
    summary.append((base, src, len(fn_codes)))

print("=" * 60)
print("汇总：")
for base, src, n in summary:
    print(f"  {base}: source={src}, codes={n}")

csv_path = os.path.join(OUT_DIR, "y_boxes.csv")
n = flush_y_boxes_csv(csv_path)
print(f"\ny_boxes.csv: {csv_path} ({n} 条)")

# 验收
all_v6 = all(s == "v6" for _, s, _ in summary)
all_hit = all(n > 0 for _, _, n in summary)
print(f"\n[{'PASS' if all_v6 and all_hit else 'FAIL'}] all_v6={all_v6} all_hit={all_hit}")
