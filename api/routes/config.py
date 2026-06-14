"""配置管理接口 — 运行时读取/修改系统配置"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app_config.config_service import ConfigService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/config")
async def get_config():
    """返回完整配置数据（含当前值、字段元数据、分组信息）。"""
    service = ConfigService.get_instance()
    return service.get_full_config()


class ConfigUpdateRequest(BaseModel):
    overrides: dict


@router.put("/config")
async def update_config(req: ConfigUpdateRequest):
    """更新配置覆盖值并持久化。"""
    service = ConfigService.get_instance()
    try:
        result = service.update_overrides(req.overrides)
        logger.info("配置已更新: %s", list(req.overrides.keys()))
        return result
    except ValueError as e:
        logger.warning("配置更新被拒绝: %s", e)
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))
