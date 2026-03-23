"""全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "song.ttf")

# 默认替换前缀
DEFAULT_PREFIXES = ["Y"]
NEW_PREFIX = "H"


def make_pattern(prefixes=None):
    """生成匹配正则，支持多个首字母。如 ["Y","A"] → r'\b[YA][A-Z0-9]{4,}\b'
    匹配首字母 + 至少4位字母数字（总长 >= 5）。"""
    prefixes = prefixes or DEFAULT_PREFIXES
    chars = "".join(p.upper() for p in prefixes)
    if len(chars) == 1:
        return rf"\b{chars}[A-Z0-9]{{4,}}\b"
    return rf"\b[{chars}][A-Z0-9]{{4,}}\b"


# Y 编号匹配正则 —— 保持向后兼容
Y_PATTERN = make_pattern()

# 符号→数字模糊回填映射（绿框/橙框专用，红框不适用）
# 仅限视觉上与数字形似的非字母数字符号
FUZZY_DIGIT_MAP = {
    '/': '1', '!': '1', '|': '1',
    '\\': '1',
    '(': '0', ')': '0',
}

# OCR 配置
OCR_LANG_EN = "en"
OCR_LANG_CH = "ch"

# OCR 模型配置（运行时可通过 set_ocr_mode() 切换）
OCR_MODE = {
    "device": "cpu",        # "cpu" 或 "gpu"
    "model_type": "mobile", # "mobile" 或 "server"
}

# 模型名称映射
OCR_MODEL_NAMES = {
    "mobile": {
        "det": "PP-OCRv5_mobile_det",
        "rec": "PP-OCRv5_mobile_rec",
    },
    "server": {
        "det": "PP-OCRv5_server_det",
        "rec": "PP-OCRv5_server_rec",
    },
}


def set_ocr_mode(device: str = "cpu", model_type: str = "mobile"):
    """切换 OCR 运行模式。切换后需调用 clear_ocr_cache() 重建实例。"""
    OCR_MODE["device"] = device.lower()
    OCR_MODE["model_type"] = model_type.lower()


def get_ocr_model_names() -> tuple[str, str]:
    """返回当前模式下的 (det_model, rec_model)。"""
    mt = OCR_MODE["model_type"]
    names = OCR_MODEL_NAMES.get(mt, OCR_MODEL_NAMES["mobile"])
    return names["det"], names["rec"]

# ── 关键词锚定检测配置 ─────────────────────────────────────────
KEYWORD_ANCHORS = {
    "material_code": {
        "keywords": ["MATERIAL CODE", "MATERIALCODE", "MATERIAL",
                     "材料代号", "零部件图号", "PARTS LIST"],
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
    "annotations": {
        "x_min": 0.15, "x_max": 0.85,
        "y_min": 0.0, "y_max": 0.85,
    },
    "bottom_left_japanese": {
        "x_min": 0.0, "x_max": 0.35,
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
