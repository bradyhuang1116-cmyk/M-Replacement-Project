#!/usr/bin/env bash
# ============================================================
# NodexelOCR 离线交付打包 —— 在「有公网+GPU 的打包机」运行
# 产出 NodexelOCR_Delivery/，物理拷贝进客户内网即可离线部署。
#
# 前置：① 已 build 自打镜像 nodexelocr:v1（见 docker/Dockerfile.nodexel）
#       ② 已能 pip download；③ 已 npm install 过 dashboard（如需面板）
# 用法：bash scripts/离线打包.sh
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="NodexelOCR_Delivery"
IMAGE="nodexelocr:v1"
PY="C:/Users/Administrator/miniconda3/envs/mitsubishi/python.exe"

echo "=================================================="
echo " 交付包输出: $ROOT/$OUT"
echo "=================================================="
mkdir -p "$OUT/app/modules" "$OUT/offline" "$OUT/config" "$OUT/docs"

# ── 1. NodexelOCR 镜像（模型已封装在内）→ tar.gz ─────────────
echo ""
echo "[1/5] 导出 NodexelOCR 镜像 (~20GB，gzip 后 ~10GB，较慢)..."
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker save "$IMAGE" | gzip > "$OUT/NodexelOCR.tar.gz"
    echo "  → $OUT/NodexelOCR.tar.gz ($(du -h "$OUT/NodexelOCR.tar.gz" | cut -f1))"
else
    echo "  ✗ 镜像 $IMAGE 不存在！先 build：docker build -f docker/Dockerfile.nodexel -t nodexel-ocr:v1 . && docker tag nodexel-ocr:v1 nodexelocr:v1"
    exit 1
fi

# ── 2. Nuitka 编译 modules 全部 + config → .pyd ──────────────
echo ""
echo "[2/5] Nuitka 编译 modules/ 全部 + config..."
bash scripts/compile_core.sh
# config.pyd 放 app 根
cp build/nuitka/config.cp310-win_amd64.pyd "$OUT/app/config.pyd"
echo "  → app/config.pyd（含业务规则/编号正则，无源码）"
# modules 各 .pyd（除 __init__）
for src in modules/*.py; do
    mod="$(basename "$src" .py)"
    [ "$mod" = "__init__" ] && continue
    cp "build/nuitka/$mod".cp310-win_amd64.pyd "$OUT/app/modules/$mod.pyd"
done
echo "  → app/modules/*.pyd（全模块编译，无源码）"

# <!--PART2-PLACEHOLDER-->

# ── 3. 应用代码 + 审核工具 exe + 字体 ──
echo ""
echo "[3/5] 拷贝应用代码与审核工具..."
cp modules/__init__.py "$OUT/app/modules/" 2>/dev/null || true
cp -r api "$OUT/app/"
cp -r fonts "$OUT/app/"
cp start_v2.py "$OUT/app/"
# 审核工具交付编译好的 exe（不拷源码）
if [ -f build_tools/dist/ManualEditor.exe ]; then
    mkdir -p "$OUT/ManualEditor"
    cp build_tools/dist/ManualEditor.exe "$OUT/ManualEditor/"
    echo "  → ManualEditor/ManualEditor.exe"
else
    echo "  ⚠️ build_tools/dist/ManualEditor.exe 不存在，先打包：python -m PyInstaller build_tools/manual_editor.spec"
fi
cp -r frontend "$OUT/" 2>/dev/null || echo "  ⚠️ frontend/ 不存在，请先运行 scripts/build_dashboard.bat"
for f in scripts/deploy/*.bat; do
  [ -f "$f" ] && cp "$f" "$OUT/"
done
rm -rf "$OUT/app/modules/__pycache__" "$OUT/app/api/__pycache__"
echo "  → app/（modules+config 为 .pyd）"

# ── 4. Python 依赖离线 wheel（含 oracledb for PLM 对接）──────
echo ""
echo "[4/5] 下载 Python 依赖 wheel..."
echo "  ⚠️ wheel 仅适用相同 OS + Python 版本：$("$PY" --version 2>&1)"
mkdir -p "$OUT/offline/offline_wheels"
"$PY" -m pip download -r requirements_local.txt -d "$OUT/offline/offline_wheels"
"$PY" -m pip download oracledb -d "$OUT/offline/offline_wheels"   # PLM Oracle thin 模式
cp requirements_local.txt "$OUT/offline/"
echo "  → offline/offline_wheels/（含 oracledb）"

# ── 5. 配置模板 + nssm + 文档 ────────────────────────────────
echo ""
echo "[5/5] 配置模板 + nssm + 文档..."
cp .env.example "$OUT/config/.env.example" 2>/dev/null || true
cat > "$OUT/config/server.properties.example" <<'PROP'
# 客户填本机 IP，决定连哪个 Oracle 库
server.ip=10.237.126.127
PROP
[ -f scripts/nssm.exe ] && cp scripts/nssm.exe "$OUT/offline/" || echo "  ⚠️ scripts/nssm.exe 不存在，请手动放入 offline/"
cp scripts/install_service.bat "$OUT/" 2>/dev/null || true
cp scripts/install_service_云端API.bat "$OUT/" 2>/dev/null || true
cp scripts/install_service_本地GPU.bat "$OUT/" 2>/dev/null || true
cp scripts/_install_service_core.bat "$OUT/" 2>/dev/null || true
cp scripts/离线安装.bat "$OUT/" 2>/dev/null || true
cp scripts/compile_core.bat "$OUT/" 2>/dev/null || true
cp scripts/assemble_delivery.bat "$OUT/" 2>/dev/null || true
for _doc in docs/部署手册.md docs/部署手册_本地GPU模式.md docs/部署手册_云端API模式.md; do
  [ -f "$_doc" ] && cp "$_doc" "$OUT/docs/" || echo "  ⚠️ $_doc 不存在"
done
[ -f docs/打包加密操作手册.md ] && cp docs/打包加密操作手册.md "$OUT/docs/" || true

echo ""
echo "=================================================="
echo " 打包完成：$OUT/"
echo " 还需手动放入 offline/（官网离线安装器）："
echo "   - Docker Desktop / NVIDIA 驱动+Toolkit / Miniconda(Py3.10) / Node(可选)"
echo "   - nssm.exe（如上面提示缺失）"
echo " 整个 $OUT/ 物理拷贝进客户内网。"
echo "=================================================="

