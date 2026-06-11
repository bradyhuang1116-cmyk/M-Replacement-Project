#!/usr/bin/env bash
# ============================================================
# Nuitka 编译模块为 .pyd（源码保护）
# 范围：modules/ 全部 + config.py（含算法/业务规则/领域知识）
# 不编译：api/（FastAPI 动态加载易坏）、manual_editor/（PySide6 GUI）、
#         __init__.py、start_v2.py（入口）
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

# modules/ 下全部（除 __init__）+ config
MODULE_FILES="config.py"
for f in modules/*.py; do
    [ "$(basename "$f")" = "__init__.py" ] && continue
    MODULE_FILES="$MODULE_FILES $f"
done

mkdir -p "$OUT"
echo "=== Nuitka 编译模块（modules/ 全部 + config）==="
for src in $MODULE_FILES; do
    mod="$(basename "$src" .py)"
    echo ">>> 编译 $src ..."
    "$PY" -m nuitka --module "$src" --output-dir="$OUT" --assume-yes-for-downloads 2>&1 | tail -1
    if ls "$OUT/$mod"*.pyd >/dev/null 2>&1; then
        echo "    OK $mod.pyd"
    else
        echo "    FAIL $mod" && exit 1
    fi
done

echo ""
echo "=== 编译完成，产物 ==="
ls "$OUT"/*.pyd | wc -l
echo "个 .pyd（位于 $OUT/）"

echo ""
echo "部署时：将 <mod>.cp310-win_amd64.pyd 改名为 <mod>.pyd"
echo "  - config.pyd 放项目根，替换 config.py"
echo "  - modules/*.pyd 放 modules/，替换对应 .py"
echo "  删除对应源 .py + __pycache__。api/ 和 manual_editor/ 保持 .py。"

