"""VLM OCR 适配器。

支持两种后端：
1. 本地 NodexelOCR Docker（vLLM OpenAI 风格接口）
2. PaddleOCR 官方托管 API layout-parsing（测试阶段可无本地 GPU）

对上层统一暴露 predict(image) -> [{"dt_polys", "rec_texts", "rec_scores"}]
"""

import re
import io
import base64
import json
import logging
import threading

import requests
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_LOC_RE = re.compile(r'([^\n<]+?)(<\|LOC_\d+\|>(?:<\|LOC_\d+\|>){7})')
_LOC_NUM_RE = re.compile(r'<\|LOC_(\d+)\|>')


def _clean_latex(text: str) -> str:
    if "\\" not in text:
        return text
    s = text
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", s)
    s = re.sub(r"_\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    return s.strip()

SPOTTING_UPSCALE_THRESHOLD = 1500


def parse_spotting_output(raw: str, img_w: int, img_h: int) -> list[dict]:
    items = []
    seen = set()
    repeat_count = 0
    for m in _LOC_RE.finditer(raw):
        text = _clean_latex(m.group(1).strip())
        locs = [int(x) for x in _LOC_NUM_RE.findall(m.group(2))]
        if len(locs) != 8:
            continue
        key = (text, tuple(locs))
        if key in seen:
            repeat_count += 1
            if repeat_count >= 3:
                logger.warning(f"VLM 重复输出截断: 已去重 {repeat_count} 条, 保留 {len(items)} 条")
                break
            continue
        seen.add(key)
        repeat_count = 0
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


# todo
def _pil_to_base64_raw(pil_image: Image.Image, max_pixels: int = 1280 * 28 * 28) -> str:
    w, h = pil_image.size
    total = w * h
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
# todo


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
        "repetition_penalty": 1.05,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions", json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout:
            if attempt < 2:
                logger.warning(f"VLM 请求超时，重试 ({attempt+1}/2)")
                continue
            raise


# todo
def _normalize_polygon(poly) -> list[list[int]] | None:
    if not poly or len(poly) < 4:
        return None
    pts = []
    for pt in poly[:4]:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        pts.append([int(round(pt[0])), int(round(pt[1]))])
    return pts


def _extract_spotting_items_from_api(result: dict) -> list[dict]:
    page0 = ((result or {}).get("layoutParsingResults") or [{}])[0] or {}
    pruned = page0.get("prunedResult") or {}
    spotting_res = pruned.get("spotting_res")
    if isinstance(spotting_res, str):
        try:
            spotting_res = json.loads(spotting_res)
        except json.JSONDecodeError:
            spotting_res = None

    items = []
    seen = set()

    def _append_item(text, poly):
        polygon = _normalize_polygon(poly)
        text = _clean_latex((text or "").strip())
        if not polygon or not text:
            return
        key = (text, tuple((p[0], p[1]) for p in polygon))
        if key in seen:
            return
        seen.add(key)
        items.append({"text": text, "polygon": polygon})

    if isinstance(spotting_res, dict):
        candidates = []
        for key in ("texts", "items", "results", "detections", "boxes"):
            value = spotting_res.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if not candidates and {"text", "polygon"} <= set(spotting_res.keys()):
            candidates = [spotting_res]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("transcription") or item.get("label")
            poly = item.get("polygon") or item.get("points") or item.get("box")
            _append_item(text, poly)

    if items:
        return items

    for block in page0.get("layoutParsingResults", []):
        if not isinstance(block, dict):
            continue
        text = block.get("text") or block.get("label")
        poly = block.get("polygon") or block.get("points") or block.get("box")
        _append_item(text, poly)
    return items


def _extract_plain_text_from_api(result: dict) -> str:
    page0 = ((result or {}).get("layoutParsingResults") or [{}])[0] or {}
    pruned = page0.get("prunedResult") or {}
    spotting_res = pruned.get("spotting_res")
    if isinstance(spotting_res, str):
        try:
            spotting_res = json.loads(spotting_res)
        except json.JSONDecodeError:
            spotting_res = None
    if isinstance(spotting_res, dict):
        texts = []
        for key in ("texts", "items", "results", "detections", "boxes"):
            value = spotting_res.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("transcription") or item.get("label")
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)

    md_text = ((page0.get("markdown") or {}).get("text")) or ""
    if md_text:
        return md_text
    return ""
# todo


class VlmOcrEngine:
    """VLM OCR 适配器（vLLM 或 PaddleOCR 托管 API）。

    predict(image) 返回与 PaddleOCR v5 兼容的结构，
    可直接被 _parse_ocr_results() 消费。
    """

    def __init__(self, base_url: str, model_name: str,
                 provider: str = "vllm", api_token: str = ""):
        self._base_url = base_url
        self._model_name = model_name
        # todo
        self._provider = provider
        self._api_token = api_token
        # todo
        logger.info(
            f"VLM 引擎: provider={provider}, base_url={base_url}, model={model_name}"
        )

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
        if self._provider != "vllm":
            raise RuntimeError(
                f"_chat 仅支持 vllm 后端，当前 provider={self._provider}"
            )
        return _vllm_chat(
            self._base_url, self._model_name,
            pil_image, prompt, max_tokens)

    # todo
    def _paddleocr_api_call(self, pil_image: Image.Image, prompt_label: str) -> dict:
        if not self._api_token:
            raise RuntimeError(
                "PADDLEOCR_API_TOKEN 未配置；请在 .env 中填写官方 token。"
            )
        payload = {
            "file": _pil_to_base64_raw(pil_image),
            "fileType": 1,
            "useLayoutDetection": False,
            "promptLabel": prompt_label,
            "useDocUnwarping": False,
            "useDocOrientationClassify": False,
        }
        headers = {
            "Authorization": f"token {self._api_token}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self._base_url, json=payload, headers=headers, timeout=300
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorCode", 0) not in (0, None):
            raise RuntimeError(
                f"PaddleOCR API 返回错误: errorCode={data.get('errorCode')} "
                f"message={data.get('errorMsg') or data.get('message') or data}"
            )
        return data.get("result") or {}
    # todo

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

        if self._provider == "paddleocr_api":
            # todo
            result = self._paddleocr_api_call(pil_img, "spotting")
            spotting_items = _extract_spotting_items_from_api(result)
            # todo
        else:
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
        # todo
        if self._provider == "paddleocr_api":
            result = self._paddleocr_api_call(pil_img, "ocr")
            return _extract_plain_text_from_api(result)
        # todo
        return self._chat(pil_img, "OCR:")

    def query_spotting(self, image) -> tuple[str, int, int]:
        """Spotting 模式，返回 (raw_output, img_w, img_h)。"""
        pil_img = self._to_pil(image)
        w, h = pil_img.size
        if w < SPOTTING_UPSCALE_THRESHOLD and h < SPOTTING_UPSCALE_THRESHOLD:
            pil_img = pil_img.resize((w * 2, h * 2), Image.LANCZOS)
        # todo
        if self._provider == "paddleocr_api":
            result = self._paddleocr_api_call(pil_img, "spotting")
            page0 = ((result or {}).get("layoutParsingResults") or [{}])[0] or {}
            pruned = page0.get("prunedResult") or {}
            raw = pruned.get("spotting_res") or json.dumps(result, ensure_ascii=False)
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            return raw, w, h
        # todo
        raw = self._chat(pil_img, "Spotting:")
        return raw, w, h

    def health_check(self) -> bool:
        """检查 VLM 服务是否可用。"""
        try:
            # todo
            if self._provider == "paddleocr_api":
                return bool(self._api_token and self._base_url)
            # todo
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
                from config import (
                    VLM_PROVIDER,
                    VLLM_BASE_URL,
                    VLLM_MODEL_NAME,
                    PADDLEOCR_API_URL,
                    PADDLEOCR_API_TOKEN,
                )
                # todo
                if VLM_PROVIDER == "paddleocr_api":
                    _engine_instance = VlmOcrEngine(
                        PADDLEOCR_API_URL,
                        "PaddleOCR-VL-1.5",
                        provider="paddleocr_api",
                        api_token=PADDLEOCR_API_TOKEN,
                    )
                # todo
                else:
                    _engine_instance = VlmOcrEngine(
                        VLLM_BASE_URL,
                        VLLM_MODEL_NAME,
                        provider="vllm",
                    )
    return _engine_instance


def clear_vlm_engine():
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
        logger.info("VLM 引擎实例已清除")
