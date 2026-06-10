#!/bin/bash
# NodexelOCR 推理服务启动脚本
set -e

echo "[NodexelOCR] Starting inference service..."

# 降低 vLLM 日志级别，压掉启动时打印的模型加载路径（INFO 级 → WARNING 只留警告/错误）
export VLLM_LOGGING_LEVEL=WARNING

# 模型已内置在镜像中（中性目录名），genai_server 从 --model_dir 加载
exec paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --model_dir /home/paddleocr/models/nodexel-ocr \
    --host 0.0.0.0 \
    --port 8080 \
    --backend vllm \
    --backend_config /home/paddleocr/vllm_config.yaml
