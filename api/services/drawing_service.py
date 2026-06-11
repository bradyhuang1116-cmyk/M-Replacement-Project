"""图纸处理业务逻辑"""
import os
import time
import shutil
from pathlib import Path
from fastapi import UploadFile
from config import DEFAULT_PREFIXES
from modules.batch_processor import process_single_file
from modules.factory_note_pixel import clear_y_box_records

class DrawingService:
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.output_dir = Path("output")
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    async def process_single(self, file: UploadFile):
        """处理单个文件"""
        start_time = time.time()

        # 保存上传文件
        file_path = self.upload_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 清空模块级 Y 编号汇总（防止跨请求累积）
        clear_y_box_records()

        # 调用核心处理逻辑
        result = process_single_file(
            str(file_path),
            str(self.output_dir),
            generate_debug=False,
            prefixes=DEFAULT_PREFIXES,
        )

        processing_time = time.time() - start_time

        # 清理上传文件
        os.remove(file_path)

        return {
            "status": result["status"],
            "method": result["method"],
            "total_replacements": result["total"],
            "replacements": [
                {"old_text": old, "new_text": new}
                for old, new in result["replacements"]
            ],
            "output_file_id": Path(result["output_path"]).stem,
            "processing_time": processing_time
        }
