"""配置管理"""
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # 文件配置
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "output"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024

    # 日志配置
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
