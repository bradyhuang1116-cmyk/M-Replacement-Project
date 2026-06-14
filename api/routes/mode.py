"""推理模式切换接口 — 本地(local) / 云端(cloud) 运行时切换，不持久化。

进入前端默认云端；用户可在选取图纸前切换。切换仅改运行时 config.VLM_PROVIDER
并重置引擎缓存，不写入 config_overrides.json（重启后回到默认云端）。
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config as config_module

router = APIRouter()
logger = logging.getLogger(__name__)

# 模式 ↔ provider 映射
_MODE_TO_PROVIDER = {"cloud": "paddleocr_api", "local": "vllm"}
_PROVIDER_TO_MODE = {v: k for k, v in _MODE_TO_PROVIDER.items()}


def _current_mode() -> str:
    return _PROVIDER_TO_MODE.get(
        getattr(config_module, "VLM_PROVIDER", "paddleocr_api"), "cloud")


@router.get("/mode")
async def get_mode():
    """返回当前推理模式：cloud（云端 API）或 local（本地模型）。"""
    return {"mode": _current_mode()}


class ModeRequest(BaseModel):
    mode: str  # "cloud" | "local"


@router.put("/mode")
async def set_mode(req: ModeRequest):
    """切换推理模式（运行时，不持久化）。重置引擎缓存使下次处理用新模式。"""
    mode = (req.mode or "").strip().lower()
    if mode not in _MODE_TO_PROVIDER:
        raise HTTPException(status_code=422,
                            detail=f"无效模式: {req.mode}，应为 cloud 或 local")

    provider = _MODE_TO_PROVIDER[mode]
    config_module.VLM_PROVIDER = provider

    # 重置引擎缓存（下次 _get_ocr_vlm / _get_ocr_v5 按新 provider 重建）
    try:
        from modules.vlm_ocr_engine import clear_vlm_engine
        clear_vlm_engine()
    except Exception as e:
        logger.warning("清除 VLM 引擎失败: %s", e)
    try:
        import modules.region_detector as rd
        rd._v5_cloud_engine = None
        rd._v5_cache.clear()
    except Exception as e:
        logger.warning("清除 v5 引擎缓存失败: %s", e)
    try:
        # 工厂注意区有独立引擎缓存，也要清，否则切模式后仍用旧引擎
        import modules.factory_note_pixel as fnp
        fnp._vlm_engine = None
        fnp._v5_engine = None
    except Exception as e:
        logger.warning("清除工厂注意区引擎缓存失败: %s", e)

    logger.info("推理模式切换为: %s (provider=%s)", mode, provider)
    return {"mode": mode}
