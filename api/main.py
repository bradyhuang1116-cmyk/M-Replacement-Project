"""FastAPI 应用入口。"""
import logging
import os
import sys
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import config, drawings, folders, gpu, health, jobs, logs, mode, plm, system
from config import API_CORS_ORIGINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NodexelOCR 图纸替换API",
    description="图纸编号批量替换系统REST API",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["健康检查"])
app.include_router(drawings.router, prefix="/api/v1", tags=["图纸处理"])
app.include_router(jobs.router, prefix="/api/v1", tags=["任务管理"])
app.include_router(gpu.router, prefix="/api/v1", tags=["GPU监控"])
app.include_router(folders.router, prefix="/api/v1", tags=["文件夹"])
app.include_router(system.router, prefix="/api/v1", tags=["系统管理"])
app.include_router(config.router, prefix="/api/v1", tags=["配置管理"])
app.include_router(logs.router, prefix="/api/v1", tags=["处理日志"])
app.include_router(plm.router, prefix="/api/v1", tags=["PLM对接"])
app.include_router(mode.router, prefix="/api/v1", tags=["推理模式"])


@app.on_event("startup")
async def startup_event():
    """后台预热与启动常驻线程，不阻塞 API 启动。"""

    try:
        from app_config.config_service import ConfigService

        ConfigService.get_instance()
        logger.info("持久化配置覆盖已加载")
    except Exception as e:
        logger.warning("加载持久化配置覆盖失败，使用默认值：%s", e)

    def _warmup():
        try:
            from modules.text_replacer import _get_ocr

            _get_ocr("en")
            _get_ocr("ch")
            logger.info("OCR 预热完成")
        except Exception as e:
            logger.warning("OCR 预热失败：%s", e)

    def _start_plm_poller():
        try:
            from modules.plm_poller import PlmPoller

            poller = PlmPoller()
            poller.start()
            logger.info("PLM 轮询器已启动")
        except Exception as e:
            logger.warning("PLM 轮询器启动失败：%s", e)

    def _start_watch_folder():
        try:
            from modules.job_queue import JobQueue
            from modules.watch_folder import WatchFolder

            queue = JobQueue()
            watcher = WatchFolder(queue)
            watcher.start()
            logger.info("WatchFolder 已启动")
        except Exception as e:
            logger.warning("WatchFolder 启动失败：%s", e)

    def _start_worker():
        try:
            from modules.job_queue import JobQueue
            from modules.worker import Worker

            queue = JobQueue()
            worker = Worker(queue)
            worker.start()
            logger.info("Worker 已启动")

            def _recover_plm_deliveries():
                try:
                    worker.recover_pending_plm_deliveries()
                    logger.info("PLM 未完成交付恢复检查完成")
                except Exception as recovery_error:
                    logger.warning("PLM 未完成交付恢复失败：%s", recovery_error)

            threading.Thread(target=_recover_plm_deliveries, daemon=True).start()
        except Exception as e:
            logger.warning("Worker 启动失败：%s", e)

    def _recover_orphaned():
        try:
            from config import SUPPORTED_EXTENSIONS, WATCH_PROCESSING_DIR
            from modules.filename_parser import parse as parse_filename
            from modules.job_queue import JobQueue

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

                existing = queue.list_by_status("pending", limit=1000)
                if any(job.file_path == path for job in existing):
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
                logger.info("已恢复 %s 个 processing 遗留文件到 JobQueue", recovered)
        except Exception as e:
            logger.warning("恢复 processing 遗留文件失败：%s", e)

    threading.Thread(target=_warmup, daemon=True).start()
    threading.Thread(target=_start_plm_poller, daemon=True).start()
    threading.Thread(target=_start_watch_folder, daemon=True).start()
    threading.Thread(target=_start_worker, daemon=True).start()

    # 当前显式禁用：后端重启后不再自动恢复 processing 目录中的遗留源文件。
    # 如需恢复该行为，可重新启用下一行线程启动代码。
    # threading.Thread(target=_recover_orphaned, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理入口。"""

    logger.info("应用正在关闭...")


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)
