"""PLM 对接 HTTP 接口（供联调 / Postman 测试）"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.plm_interface import plm_process_drawing

router = APIRouter()


class PlmProcessRequest(BaseModel):
    file_path: str = Field(..., description="远程/PLM 侧原图路径")
    output_dir: str = Field(..., description="远程/PLM 侧归档目录")


class PlmProcessResponse(BaseModel):
    status: str
    method: str
    total: int
    file_path: str


@router.post("/plm/process", response_model=PlmProcessResponse)
def process_plm_drawing(req: PlmProcessRequest):
    """PLM 图纸处理 — 获取远程原图 → 本地处理 → 回写远程路径。"""
    try:
        result = plm_process_drawing(req.file_path, req.output_dir)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return result
