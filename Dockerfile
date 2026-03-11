FROM docker.1ms.run/library/python:3.10-slim

WORKDIR /app

# 配置pip使用国内镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p /app/output /app/uploads

# 预下载OCR模型（打包进镜像，内网可用）
# 参数与运行时 _get_ocr() 完全一致，确保不会触发额外下载
RUN FLAGS_use_mkldnn=0 FLAGS_enable_pir_api=0 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=0 \
    python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); \
PaddleOCR(lang='ch', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); \
print('Models downloaded')"

# 暴露端口
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import sys; sys.exit(0)"

# 默认启动API服务
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
