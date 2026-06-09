#!/usr/bin/env bash
# ============================================================
# 离线部署打包脚本 —— 在「有公网的机器」上运行
# 产出一个 offline_bundle/ 目录，U 盘拷进客户内网即可离线部署。
#
# 前置：本机已能 docker pull、pip download、且已 npm install 过 dashboard。
# 用法：bash scripts/离线打包.sh
# ============================================================
set -euo pipefail

# 项目根目录（脚本在 scripts/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUNDLE="offline_bundle"
IMAGE="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu"

echo "=================================================="
echo " 离线包输出目录: $ROOT/$BUNDLE"
echo "=================================================="
mkdir -p "$BUNDLE"

# ── 1. vLLM Docker 镜像 → tar ────────────────────────────
echo ""
echo "[1/4] 导出 vLLM Docker 镜像 (几 GB，较慢)..."
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  镜像已在本地，直接导出"
else
    echo "  本地无镜像，先 docker pull..."
    docker pull "$IMAGE"
fi
docker save "$IMAGE" -o "$BUNDLE/vllm_image.tar"
echo "  → $BUNDLE/vllm_image.tar ($(du -h "$BUNDLE/vllm_image.tar" | cut -f1))"

# ── 2. 模型权重 1.8GB → 直接拷 ───────────────────────────
echo ""
echo "[2/4] 拷贝模型权重 models/PaddleOCR-VL-1.5/ (1.8GB)..."
mkdir -p "$BUNDLE/models"
cp -r "models/PaddleOCR-VL-1.5" "$BUNDLE/models/"
echo "  → $BUNDLE/models/PaddleOCR-VL-1.5/ ($(du -sh "$BUNDLE/models/PaddleOCR-VL-1.5" | cut -f1))"

# ── 3. Python 依赖 → 离线 wheel ──────────────────────────
echo ""
echo "[3/4] 下载 Python 依赖 wheel..."
echo "  ⚠️ 这些 wheel 仅适用于与本机相同的 OS + Python 版本：$(python --version 2>&1)"
mkdir -p "$BUNDLE/offline_wheels"
pip download -r requirements_local.txt -d "$BUNDLE/offline_wheels"
cp requirements_local.txt "$BUNDLE/"
echo "  → $BUNDLE/offline_wheels/ ($(ls "$BUNDLE/offline_wheels" | wc -l) 个包)"

# ── 4. 前端依赖 node_modules → 直接拷 ────────────────────
echo ""
echo "[4/4] 拷贝前端依赖 dashboard/node_modules/..."
if [ -d "dashboard/node_modules" ]; then
    mkdir -p "$BUNDLE/dashboard"
    cp -r "dashboard/node_modules" "$BUNDLE/dashboard/"
    echo "  → $BUNDLE/dashboard/node_modules/ ($(du -sh "$BUNDLE/dashboard/node_modules" | cut -f1))"
else
    echo "  ⚠️ 未发现 dashboard/node_modules，请先在 dashboard/ 下执行 npm install 再重跑"
fi

echo ""
echo "=================================================="
echo " 打包完成。还需手动准备的「基础环境安装包」（去官网下离线安装器）："
echo "   - Docker Desktop 安装包 (.exe)"
echo "   - NVIDIA 驱动 + NVIDIA Container Toolkit"
echo "   - Miniconda / Python 3.13 安装包"
echo "   - Node.js 安装包"
echo " 把它们和 $BUNDLE/ 一起拷进客户内网。"
echo "=================================================="
