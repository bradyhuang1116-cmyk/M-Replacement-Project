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
from api.routes import config, drawings, health, jobs, gpu, folders, system, logs
from config import API_CORS_ORIGINS

app = FastAPI(
    title="NodexelOCR 图纸替换API",
    description="图纸编号批量替换系统REST API",
    version="2.0.0",
    # 关闭自带交互文档页（内部 API，不对用户暴露）
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# CORS配置（默认 ["*"] 全通；可通过 API_CORS_ORIGINS 环境变量逗号分隔限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
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
app.include_router(config.router, prefix="/api/v1", tags=["配置管理"])
app.include_router(logs.router, prefix="/api/v1", tags=["处理日志"])

@app.on_event("startup")
async def startup_event():
    """后台线程预热OCR模型 + 启动PLM轮询器，不阻塞API启动"""
    import threading

    # 先加载持久化覆盖配置，使 setattr 在 PlmPoller/OracleHelper 之前生效
    try:
        from app_config.config_service import ConfigService
        ConfigService.get_instance()
        logging.getLogger(__name__).info("持久化配置已加载")
    except Exception as e:
        logging.getLogger(__name__).warning(f"配置加载失败（使用默认值）: {e}")

    def _warmup():
        try:
            from modules.text_replacer import _get_ocr
            _get_ocr("en")
            _get_ocr("ch")
            logging.getLogger(__name__).info("OCR模型预热完成")
        except Exception as e:
            logging.getLogger(__name__).warning(f"OCR模型预热失败: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    # 启动 PLM Oracle 轮询器（自动拉取远程图纸到 inbox）
    def _start_plm_poller():
        try:
            from modules.plm_poller import PlmPoller
            poller = PlmPoller()
            poller.start()
            logging.getLogger(__name__).info("PLM 轮询器已启动")
        except Exception as e:
            logging.getLogger(__name__).warning(f"PLM 轮询器启动失败: {e}")

    threading.Thread(target=_start_plm_poller, daemon=True).start()

    # 启动 WatchFolder（监控 inbox → 移到 processing → 入队 JobQueue）
    def _start_watch_folder():
        try:
            from modules.job_queue import JobQueue
            from modules.watch_folder import WatchFolder
            queue = JobQueue()
            watcher = WatchFolder(queue)
            watcher.start()
            logging.getLogger(__name__).info("WatchFolder 已启动")
        except Exception as e:
            logging.getLogger(__name__).warning(f"WatchFolder 启动失败: {e}")

    threading.Thread(target=_start_watch_folder, daemon=True).start()

    # 启动 Worker（消费 JobQueue，处理图纸）
    def _start_worker():
        try:
            from modules.job_queue import JobQueue
            from modules.worker import Worker
            queue = JobQueue()
            worker = Worker(queue)
            worker.start()
            logging.getLogger(__name__).info("Worker 已启动")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Worker 启动失败: {e}")

    threading.Thread(target=_start_worker, daemon=True).start()

    # 恢复上次停机时遗留在 processing 中的未处理文件
    def _recover_orphaned():
        try:
            from config import WATCH_PROCESSING_DIR, SUPPORTED_EXTENSIONS
            from modules.filename_parser import parse as parse_filename
            from modules.job_queue import JobQueue
            import os
            proc_dir = WATCH_PROCESSING_DIR
            if not os.path.isdir(proc_dir):
                return
            queue = JobQueue()
            recovered = 0
            for name in sorted(os.listdir(proc_dir)):
                ext = os.path.splitext(name)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                path = os.path.join(proc_dir, name)
                if not os.path.isfile(path):
                    continue
                # 跳过已在队列中的文件
                existing = queue.list_by_status("pending", limit=1000)
                if any(j.file_path == path for j in existing):
                    continue
                parsed = parse_filename(name)
                queue.enqueue(
                    source="watch_folder",
                    source_file=name,
                    file_path=path,
                    drawing_no=parsed.drawing_no,
                    revision=parsed.revision,
                )
                recovered += 1
            if recovered:
                logging.getLogger(__name__).info(
                    f"已恢复 {recovered} 个遗留文件到队列"
                )
        except Exception as e:
            logging.getLogger(__name__).warning(f"恢复遗留文件失败: {e}")

    threading.Thread(target=_recover_orphaned, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理后台服务。"""
    logging.getLogger(__name__).info("应用正在关闭...")

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
