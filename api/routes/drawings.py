"""图纸处理接口"""
from fastapi import APIRouter, UploadFile, File
from api.models.response import ProcessResponse
from api.services.drawing_service import DrawingService

router = APIRouter()
service = DrawingService()

@router.post("/drawings/process", response_model=ProcessResponse)
async def process_drawing(file: UploadFile = File(...)):
    """同步处理单个图纸"""
    result = await service.process_single(file)
    return result
