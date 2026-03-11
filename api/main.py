"""FastAPI应用入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import drawings, health

app = FastAPI(
    title="三菱图纸替换API",
    description="图纸Y→HY批量替换系统REST API",
    version="1.0.0"
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

@app.on_event("startup")
async def startup_event():
    """启动时预热OCR模型"""
    try:
        from modules.text_replacer import _get_ocr
        _get_ocr("en")
        _get_ocr("ch")
        print("✓ OCR模型预热完成")
    except Exception as e:
        print(f"⚠ OCR模型预热失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
