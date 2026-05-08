"""FastAPI应用入口"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import drawings, health, jobs, gpu, folders, system

app = FastAPI(
    title="三菱图纸替换API",
    description="图纸Y→HY批量替换系统REST API",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/api/v1", tags=["健康检查"])
app.include_router(drawings.router, prefix="/api/v1", tags=["图纸处理"])
app.include_router(jobs.router, prefix="/api/v1", tags=["任务管理"])
app.include_router(gpu.router, prefix="/api/v1", tags=["GPU监控"])
app.include_router(folders.router, prefix="/api/v1", tags=["文件夹"])
app.include_router(system.router, prefix="/api/v1", tags=["系统管理"])

@app.on_event("startup")
async def startup_event():
    """后台线程预热OCR模型，不阻塞API启动"""
    import threading

    def _warmup():
        try:
            from modules.text_replacer import _get_ocr
            _get_ocr("en")
            _get_ocr("ch")
            logging.getLogger(__name__).info("OCR模型预热完成")
        except Exception as e:
            logging.getLogger(__name__).warning(f"OCR模型预热失败: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
