"""API响应模型"""
from pydantic import BaseModel
from typing import List, Optional

class Replacement(BaseModel):
    old_text: str
    new_text: str

class ProcessResponse(BaseModel):
    status: str
    method: str
    total_replacements: int
    replacements: List[Replacement]
    output_file_id: str
    processing_time: float

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None

class PlmProcessResponse(BaseModel):
    status: str
    method: str
    total: int
    file_path: str
    error: Optional[str] = None
