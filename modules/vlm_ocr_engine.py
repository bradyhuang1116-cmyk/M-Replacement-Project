"""PaddleOCR-VL-1.5 适配器 — 通过 vLLM HTTP API 调用，提供与 PaddleOCR v5 兼容的接口"""

import re
import io
import base64
import logging
import threading

import requests
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_LOC_RE = re.compile(r'([^\n<]+?)(<\|LOC_\d+\|>(?:<\|LOC_\d+\|>){7})')
_LOC_NUM_RE = re.compile(r'<\|LOC_(\d+)\|>')

SPOTTING_UPSCALE_THRESHOLD = 1500


def parse_spotting_output(raw: str, img_w: int, img_h: int) -> list[dict]:
    items = []
    for m in _LOC_RE.finditer(raw):
        text = m.group(1).strip()
        locs = [int(x) for x in _LOC_NUM_RE.findall(m.group(2))]
        if len(locs) != 8:
            continue
        xs = [locs[i] / 1000 * img_w for i in (0, 2, 4, 6)]
        ys = [locs[i] / 1000 * img_h for i in (1, 3, 5, 7)]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        items.append({"text": text, "polygon": polygon})
    return items


def _pil_to_base64(pil_image: Image.Image, max_pixels: int = 1280 * 28 * 28) -> str:
    w, h = pil_image.size
    total = w * h
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _vllm_chat(base_url: str, model_name: str,
               pil_image: Image.Image, prompt: str,
               max_tokens: int = 4096) -> str:
    image_uri = _pil_to_base64(pil_image)
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = requests.post(
        f"{base_url}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class VlmOcrEngine:
    """PaddleOCR-VL-1.5 适配器（vLLM HTTP API）。

    predict(image) 返回与 PaddleOCR v5 兼容的结构，
    可直接被 _parse_ocr_results() 消费。
    """

    def __init__(self, base_url: str, model_name: str):
        self._base_url = base_url
        self._model_name = model_name
        logger.info(f"VLM 引擎: vLLM HTTP → {base_url}, model={model_name}")

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _chat(self, pil_image: Image.Image, prompt: str,
              max_tokens: int = 4096) -> str:
        return _vllm_chat(
            self._base_url, self._model_name,
            pil_image, prompt, max_tokens)

    def predict(self, image) -> list[dict]:
        """兼容 PaddleOCR v5 的 predict() 接口。

        返回 [{"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}]
        """
        pil_img = self._to_pil(image)
        w, h = pil_img.size

        if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
            pil_img = pil_img.resize((w * 2, h * 2), Image.LANCZOS)
            effective_w, effective_h = w * 2, h * 2
        else:
            effective_w, effective_h = w, h

        raw = self._chat(pil_img, "Spotting:")
        spotting_items = parse_spotting_output(raw, effective_w, effective_h)

        if effective_w != w:
            scale = w / effective_w
            for item in spotting_items:
                item["polygon"] = [
                    [int(pt[0] * scale), int(pt[1] * scale)]
                    for pt in item["polygon"]
                ]

        polys = [item["polygon"] for item in spotting_items]
        texts = [item["text"] for item in spotting_items]
        scores = [1.0] * len(spotting_items)

        return [{"dt_polys": polys, "rec_texts": texts, "rec_scores": scores}]

    def query_ocr(self, image) -> str:
        """纯文本 OCR（无坐标），用于快速预扫。"""
        pil_img = self._to_pil(image)
        w, h = pil_img.size
        if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
            pil_img = pil_img.resize((w * 2, h * 2), Image.LANCZOS)
        return self._chat(pil_img, "OCR:")

    def query_spotting(self, image) -> tuple[str, int, int]:
        """Spotting 模式，返回 (raw_output, img_w, img_h)。"""
        pil_img = self._to_pil(image)
        w, h = pil_img.size
        if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
            pil_img = pil_img.resize((w * 2, h * 2), Image.LANCZOS)
        raw = self._chat(pil_img, "Spotting:")
        return raw, w, h

    def health_check(self) -> bool:
        """检查 vLLM 服务器是否可达。"""
        try:
            resp = requests.get(f"{self._base_url}/models", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False


# ── 全局单例管理 ─────────────────────────────────────────────────

_engine_instance: VlmOcrEngine | None = None
_engine_lock = threading.Lock()


def get_vlm_engine() -> VlmOcrEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                from config import VLLM_BASE_URL, VLLM_MODEL_NAME
                _engine_instance = VlmOcrEngine(VLLM_BASE_URL, VLLM_MODEL_NAME)
    return _engine_instance


def clear_vlm_engine():
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
        logger.info("VLM 引擎实例已清除")
