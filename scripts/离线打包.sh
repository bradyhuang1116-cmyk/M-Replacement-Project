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
PY="C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe"

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

# ── 2. Nuitka 编译算法核心 → .pyd ────────────────────────────
echo ""
echo "[2/5] Nuitka 编译 4 个算法核心..."
bash scripts/compile_core.sh
for mod in pdf_vector_handler factory_note_pixel text_replacer region_detector; do
    cp "build/nuitka/$mod"*.pyd "$OUT/app/modules/$mod.pyd"
    echo "  → app/modules/$mod.pyd（编译产物，无源码）"
done

# <!--PART2-PLACEHOLDER-->

# ── 3. 应用代码（管道 .py + api + config + fonts，不含已编译的源 .py）──
echo ""
echo "[3/5] 拷贝应用代码（管道代码保持 .py，算法已是 .pyd）..."
# modules 下除 4 个已编译模块外的 .py 全拷
for f in modules/*.py; do
    base=$(basename "$f" .py)
    case "$base" in
        pdf_vector_handler|factory_note_pixel|text_replacer|region_detector) ;;  # 跳过，用 .pyd
        *) cp "$f" "$OUT/app/modules/" ;;
    esac
done
cp modules/__init__.py "$OUT/app/modules/" 2>/dev/null || true
cp -r api "$OUT/app/"
cp config.py "$OUT/app/"
cp -r fonts "$OUT/app/"
cp start_v2.py "$OUT/app/"
[ -d manual_editor ] && cp -r manual_editor "$OUT/app/" && rm -rf "$OUT/app/manual_editor/__pycache__"
if [ -d "dashboard/node_modules" ]; then
    cp -r dashboard "$OUT/app/" && rm -rf "$OUT/app/dashboard/.next"
fi
rm -rf "$OUT/app/modules/__pycache__" "$OUT/app/api/__pycache__"
echo "  → app/（modules 算法为 .pyd，其余 .py）"

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
cp docs/v4_final_roadmap.md "$OUT/docs/" 2>/dev/null || true

echo ""
echo "=================================================="
echo " 打包完成：$OUT/"
echo " 还需手动放入 offline/（官网离线安装器）："
echo "   - Docker Desktop / NVIDIA 驱动+Toolkit / Miniconda(Py3.10) / Node(可选)"
echo "   - nssm.exe（如上面提示缺失）"
echo " 整个 $OUT/ 物理拷贝进客户内网。"
echo "=================================================="

