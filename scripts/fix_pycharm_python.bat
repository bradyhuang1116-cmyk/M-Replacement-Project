@echo off
REM 修复 PyCharm 仍显示 Python 3.13 的问题
chcp 65001 >nul
echo ==================================================
echo  PyCharm Python 解释器修复
echo ==================================================
echo.
echo 已自动写入项目配置:
echo   .idea\misc.xml          - 项目解释器 = Python 3.10
echo   .idea\terminal.xml      - 关闭自动激活 .venv
echo.
echo 请你现在做:
echo   1. 完全关闭 PyCharm（不是只关项目）
echo   2. 重新打开本项目
echo   3. 看右下角应显示 Python 3.10（无 V）
echo   4. 关掉旧终端，新开终端，执行: python --version
echo.
echo 若仍是 3.13:
echo   Settings - Python - Interpreter
echo   选中 Python 3.10（mitsubishi 路径，无 V）
echo   删除 Python 3.13 那条
echo   Apply - OK
echo.
echo mitsubishi 路径:
echo   C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe
echo.
echo 打包不依赖 python 命令，可直接:
echo   scripts\compile_core.bat
echo ==================================================
pause
