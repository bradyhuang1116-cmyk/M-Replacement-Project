#!/usr/bin/env bash
# ============================================================
# 工作包2：Nuitka 编译 4 个算法核心模块为 .pyd（源码保护）
# 在 mitsubishi 环境运行。产物在 build/nuitka/*.pyd
#
# 前置：MinGW 已缓存（首次需用 gh-proxy.com 加速下载，见 docs/v4_final_roadmap.md）
# 用法：bash scripts/compile_core.sh
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe"
OUT="build/nuitka"
MODULES="pdf_vector_handler factory_note_pixel text_replacer region_detector"

mkdir -p "$OUT"
echo "=== Nuitka 编译核心算法模块 ==="
for mod in $MODULES; do
    echo ">>> 编译 modules/$mod.py ..."
    "$PY" -m nuitka --module "modules/$mod.py" --output-dir="$OUT" --assume-yes-for-downloads 2>&1 | tail -2
    if ls "$OUT/$mod"*.pyd >/dev/null 2>&1; then
        echo "    OK $mod.pyd"
    else
        echo "    FAIL $mod" && exit 1
    fi
done

echo ""
echo "=== 编译完成，产物 ==="
ls -la "$OUT"/*.pyd

echo ""
echo "部署时：将 build/nuitka/<mod>.cp310-win_amd64.pyd 改名为 <mod>.pyd"
echo "放回 modules/ 替换对应 .py，并删除源 .py + __pycache__。"
