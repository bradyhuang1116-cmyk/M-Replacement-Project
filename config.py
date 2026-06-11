"""全局配置

§12 Phase 1: 部署相关常量优先从环境变量读取（支持 .env），不配置时回退到现有默认值。
不改这些 env 时，行为完全等同于改造前。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 从项目根目录加载 .env（不依赖当前工作目录）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# ── 字体路径 ──
FONT_PATH = os.getenv(
    "FONT_PATH",
    os.path.join(BASE_DIR, "fonts", "dingliesongtypeface20241217-2.ttf"),
)
PDF_FONT_PATH = os.getenv("PDF_FONT_PATH", FONT_PATH)

# ── VLM OCR 引擎配置（VLM OCR 引擎 via HTTP）────────────
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "PaddleOCR-VL-1.5-0.9B")
# VLM_PROVIDER:
#   - "vllm": 走本地 NodexelOCR Docker（默认，生产）
#   - "paddleocr_api": 走 PaddleOCR 官方托管 layout-parsing API（可无本地 GPU 测试）
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "vllm").strip().lower()
PADDLEOCR_API_URL = os.getenv(
    "PADDLEOCR_API_URL",
    "https://ebi011tbsdc4t6yc.aistudio-app.com/layout-parsing",
)
PADDLEOCR_API_TOKEN = os.getenv("PADDLEOCR_API_TOKEN", "")

# ── Docker 容器配置（NodexelOCR 自打镜像，模型已封装在镜像内，零挂载启动）──
DOCKER_CONTAINER_NAME = os.getenv("DOCKER_CONTAINER_NAME", "nodexel")
DOCKER_CONTAINER_PORT = int(os.getenv("DOCKER_CONTAINER_PORT", "8080"))
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "nodexelocr:v1")

# ── API 服务网络 ──
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# 逗号分隔；"*" 表示全部放行
API_CORS_ORIGINS = [
    s.strip() for s in os.getenv("API_CORS_ORIGINS", "*").split(",") if s.strip()
]

# ── 输出目录子结构 ──
OUTPUT_SUBDIR = os.getenv("OUTPUT_SUBDIR", "OUTPUT")
VLMOCR_SUBDIR = os.getenv("VLMOCR_SUBDIR", "VLMOCR")
PDF_REPLACEMENT_SUBDIR = os.getenv("PDF_REPLACEMENT_SUBDIR", "PDF_Replacement")
Y_BOXES_CSV_NAME = os.getenv("Y_BOXES_CSV_NAME", "y_boxes.csv")
PROCESSING_REPORT_NAME = os.getenv("PROCESSING_REPORT_NAME", "processing_report.txt")

# ── 持久化数据（队列 / 日志）──（§12 Phase 2+）
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
QUEUE_DB_PATH = os.getenv("QUEUE_DB_PATH", os.path.join(DATA_DIR, "queue.db"))
# 图纸处理日志表（开放给 PLM 后台访问）；字段对齐客户 R_V_TD_FILEPATH
PROCESS_LOG_DB_PATH = os.getenv("PROCESS_LOG_DB_PATH", os.path.join(DATA_DIR, "process_log.db"))
# Worker 处理产物根目录（Phase 2 起，watch folder / API 任务都会落到这里）
WORKER_OUTPUT_DIR = os.getenv("WORKER_OUTPUT_DIR", os.path.join(DATA_DIR, "processed"))
# Worker 空闲时的轮询间隔（秒）
WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "1.5"))
# Worker 单任务失败后的最大重试次数
WORKER_MAX_RETRY = int(os.getenv("WORKER_MAX_RETRY", "1"))
# 内部 API 鉴权（Dashboard 日志查询等）；Bearer token 默认值
DEFAULT_INTERNAL_API_KEY = os.getenv("DEFAULT_INTERNAL_API_KEY", "dev-api-key")

# ── Watch folder（§12 Phase 3）──
# PLM 投递入口 / 处理中 / 输出 / 失败；默认放在 DATA_DIR/watch/* 便于本地测试
WATCH_INBOX_DIR = os.getenv("WATCH_INBOX_DIR", os.path.join(DATA_DIR, "watch", "inbox"))
WATCH_PROCESSING_DIR = os.getenv("WATCH_PROCESSING_DIR", os.path.join(DATA_DIR, "watch", "processing"))
WATCH_OUTPUT_DIR = os.getenv("WATCH_OUTPUT_DIR", os.path.join(DATA_DIR, "watch", "output"))
WATCH_FAILED_DIR = os.getenv("WATCH_FAILED_DIR", os.path.join(DATA_DIR, "watch", "failed"))
# 文件稳定性窗口（秒）：连续 N 秒 size+mtime 不变才入队（防止 PLM 写一半就被读）
WATCH_STABILITY_SECONDS = float(os.getenv("WATCH_STABILITY_SECONDS", "3.0"))
# 扫描 inbox 的间隔（秒）；当前不用 inotify/ReadDirectoryChangesW，纯轮询足够
WATCH_SCAN_INTERVAL = float(os.getenv("WATCH_SCAN_INTERVAL", "2.0"))

# 默认替换前缀
DEFAULT_PREFIXES = ["Y", "X", "B", "H"]
NEW_PREFIX = "H"


def make_pattern(prefixes=None):
    """生成匹配正则，支持多个首字母。如 ["Y","A"] → r'[YA](?=[A-Z0-9]*\d)[A-Z0-9]{6,}'
    Y 编号定义：首字母（Y/X 等） + ≥6 位字母数字，且这 ≥6 位中**至少含 1 个数字**
    （纯字母如 YARIABLE 只是英文词，不是编号）。
    不使用 \b 词边界 —— Y 编号无论前后粘什么字符都应被识别（如 `01.021YA057C800-01`）。"""
    prefixes = prefixes or DEFAULT_PREFIXES
    chars = "".join(p.upper() for p in prefixes)
    if len(chars) == 1:
        return rf"{chars}(?=[A-Z0-9]*\d)[A-Z0-9]{{6,}}"
    return rf"[{chars}](?=[A-Z0-9]*\d)[A-Z0-9]{{6,}}"


# Y 编号匹配正则 —— 保持向后兼容
Y_PATTERN = make_pattern()

# 符号→数字模糊回填映射（绿框/橙框专用，红框不适用）
# 仅限视觉上与数字形似的非字母数字符号
FUZZY_DIGIT_MAP = {
    '/': '1', '!': '1', '|': '1',
    '\\': '1',
    '(': '0', ')': '0',
}

# OCR 配置（保留 lang 常量供下游兼容）
OCR_LANG_EN = "en"
OCR_LANG_CH = "ch"

OCR_MODE = {
    "device": "gpu",
}


def set_ocr_mode(device: str = "gpu", model_type: str = "server"):
    """保留接口兼容。VLM 通过 vLLM HTTP 服务器运行。"""
    OCR_MODE["device"] = device.lower()

# ── 关键词锚定检测配置 ─────────────────────────────────────────
KEYWORD_ANCHORS = {
    "material_code": {
        "keywords": ["MATERIAL CODE", "MATERIALCODE", "MATERIAL",
                     "材料代号", "零部件图号", "PARTS LIST", "代号"],
        "fuzzy_threshold": 0.70,
    },
}

# 表格结束判据：网格线间距 > 中位间距 × 此值时认为表格结束
TABLE_END_GAP_MULTIPLIER = 2.0

# ── 后备百分比区域（关键词检测失败时使用）──────────────────────
DEFAULT_REGIONS = {
    "left_table": {
        "x_min": 0.0, "x_max": 0.18,
        "y_min": 0.0, "y_max": 1.0,
    },
    "bottom_right_title": {
        "x_min": 0.75, "x_max": 1.0,
        "y_min": 0.85, "y_max": 1.0,
    },
    "top_left_number": {
        "x_min": 0.0, "x_max": 0.20,
        "y_min": 0.0, "y_max": 0.08,
    },
}
FALLBACK_REGIONS = DEFAULT_REGIONS

# ── Oracle PLM 对接 ───────────────────────────────────────────
ORACLE_HOST = os.getenv("ORACLE_HOST", "")
ORACLE_PORT = int(os.getenv("ORACLE_PORT", "1521"))
ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME", "")
ORACLE_USER = os.getenv("ORACLE_USER", "")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "")

# 文件格式
SUPPORTED_EXTENSIONS = {".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
