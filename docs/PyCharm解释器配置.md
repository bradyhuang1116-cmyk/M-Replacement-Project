# PyCharm 解释器配置（conda mitsubishi 3.10）

本项目打包**只用 conda 环境 `mitsubishi`（Python 3.10）**，**不要用项目 `.venv`**。

解释器路径（按本机用户名修改）：

```
C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe
```

---

## 1. 删除 `.venv`（若存在）

关闭 PyCharm 后，在 PowerShell 中：

```powershell
Set-Location D:\day\2026\202606\20260611\M-Replacement-Project
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
```

---

## 2. 安装打包依赖到 mitsubishi

```powershell
Set-Location D:\day\2026\202606\20260611\M-Replacement-Project

pip install -U pip
pip install nuitka ordered-set zstandard pyinstaller
pip install -r manual_editor\requirements.txt
```

---

## 3. PyCharm 改解释器

1. **File → Settings → Python → Interpreter**
2. 选中带 **V** 的 `Python 3.10 (.venv)` → 点 **-** 删除
3. 点 **+** → **Add Local Interpreter**
4. 选 **Conda Environment → Existing** 或 **System Interpreter**
5. 路径：`C:\Users\Administrator\miniconda3\envs\mitsubishi\python.exe`
6. 应显示 **Python 3.10**，图标**无 V**
7. **Apply → OK**

---

## 4. 终端不要自动激活 venv

**File → Settings → Tools → Terminal**

取消勾选：**Activate virtual environment**

---

## 5. 验证

关掉终端，新开一个：

```powershell
python --version
```

| 检查项 | 要求 |
|--------|------|
| 版本 | `Python 3.10.20` |
| 提示符 | 无 `(.venv)` |
| 右下角 | `Python 3.10`，无 V |

---

## 图标说明

| 图标 | 含义 | 本项目 |
|------|------|--------|
| 无 V | conda / 系统 Python | ✅ 使用 |
| 带 V | 虚拟环境 `.venv` | ❌ 不使用 |

配置完成后，按 `docs/打包加密操作手册.md` 执行打包步骤。
