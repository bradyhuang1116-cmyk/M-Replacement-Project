"""批量替换：全部 Y 字头 TIF 文件，输出到统一文件夹"""
import os
import sys
import time
import logging
import gc

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from config import DEFAULT_PREFIXES
from modules.batch_processor import process_single_file
from modules.docker_manager import ensure_vlm_ready

# ── 输入 ──
_default_dir = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else _default_dir

if os.path.isdir(INPUT_PATH):
    INPUT_FILES = sorted([
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
        and f.upper().startswith("Y")
    ])
else:
    INPUT_FILES = [INPUT_PATH]

if not INPUT_FILES:
    logger.error(f"未找到 Y 字头 TIF 文件: {INPUT_PATH}")
    sys.exit(1)

logger.info(f"共 {len(INPUT_FILES)} 个文件待替换")

# ── 输出目录 ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "batch_replace")
os.makedirs(OUTPUT_DIR, exist_ok=True)

prefixes = DEFAULT_PREFIXES

# ── VLM 服务 ──
logger.info("正在启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 服务启动失败: {msg}")
    sys.exit(1)
logger.info(f"VLM 服务就绪: {msg}")

# ── 批量处理 ──
logger.info("=" * 60)
_total_ok = 0
_total_fail = 0
_start_time = time.time()

for i, file_path in enumerate(INPUT_FILES):
    fname = os.path.basename(file_path)
    logger.info(f"[{i+1}/{len(INPUT_FILES)}] {fname}")
    t0 = time.time()

    try:
        result = process_single_file(
            file_path, OUTPUT_DIR, prefixes=prefixes,
        )
        elapsed = time.time() - t0
        m, s = int(elapsed // 60), int(elapsed % 60)
        total = result.get("total", 0)
        out_path = result.get("output_path", "")
        logger.info(f"  OK: {total} 处替换, {m}m{s:02d}s → {os.path.basename(out_path)}")
        _total_ok += 1
    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)
        _total_fail += 1

    gc.collect()

# ── 汇总 ──
total_elapsed = time.time() - _start_time
tm, ts = int(total_elapsed // 60), int(total_elapsed % 60)
logger.info("=" * 60)
logger.info(f"完成: {_total_ok} 成功, {_total_fail} 失败, 共 {len(INPUT_FILES)} 个文件")
logger.info(f"总耗时: {tm}m{ts:02d}s")
logger.info(f"输出目录: {OUTPUT_DIR}")
