"""测试删除线恢复顺序：白填充 → 删除线 → 文字（文字在最上层）"""
import os
import sys
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from config import DEFAULT_PREFIXES
from modules.batch_processor import process_single_file
from modules.docker_manager import ensure_vlm_ready

_default_dir = r"C:\Users\Brady Huang\Downloads\TIF_Undo"
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else _default_dir

if os.path.isdir(INPUT_PATH):
    INPUT_FILES = sorted([
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
        and f.upper().startswith("Y")
    ])[:3]
else:
    INPUT_FILES = [INPUT_PATH]

if not INPUT_FILES:
    logger.error(f"未找到文件: {INPUT_PATH}")
    sys.exit(1)

logger.info(f"测试文件: {[os.path.basename(f) for f in INPUT_FILES]}")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "strike_order_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

prefixes = DEFAULT_PREFIXES

logger.info("启动 VLM 服务...")
ok, msg = ensure_vlm_ready()
if not ok:
    logger.error(f"VLM 启动失败: {msg}")
    sys.exit(1)
logger.info(f"VLM 就绪: {msg}")

for i, fp in enumerate(INPUT_FILES):
    logger.info(f"[{i+1}/{len(INPUT_FILES)}] {os.path.basename(fp)}")
    try:
        result = process_single_file(fp, OUTPUT_DIR, prefixes=prefixes)
        logger.info(f"  OK: {result.get('total', 0)} 处替换 → {os.path.basename(result.get('output_path', ''))}")
    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)

logger.info(f"输出: {OUTPUT_DIR}")
