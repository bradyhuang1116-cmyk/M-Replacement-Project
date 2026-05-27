"""全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "dingliesongtypeface20241217-2.ttf")
PDF_FONT_PATH = os.path.join(BASE_DIR, "fonts", "dingliesongtypeface20241217-2.ttf")

# ── VLM OCR 引擎配置（PaddleOCR-VL-1.5 via vLLM HTTP）────────────
VLLM_BASE_URL = "http://localhost:8080/v1"
VLLM_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"

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

# 文件格式
SUPPORTED_EXTENSIONS = {".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
