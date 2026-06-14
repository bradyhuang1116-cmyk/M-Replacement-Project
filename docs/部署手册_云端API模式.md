# NodexelOCR 部署手册 — 云端 API 模式

> 适用对象：联调测试机、**无 NVIDIA 显卡**的处理机  
> 推理方式：PaddleOCR 官方托管 API（`VLM_PROVIDER=paddleocr_api`）  
> **需要处理机能够访问公网 API**；图纸会发送至云端推理，**不适合作为正式生产方案**

 [← 返回部署总览](部署手册.md)

---

## 0. 开始前请确认

| 项目 | 要求 |
|------|------|
| 交付包路径 | 下文统一写作 `D:\NodexelOCR\NodexelOCR_Delivery\`（若路径不同，全文替换） |
| 权限 | bat 脚本需 **右键 → 以管理员身份运行**（WinSCP 安装包可双击） |
| 本模式**不需要** | NVIDIA 驱动、Docker、`NodexelOCR.tar.gz` |
| 本模式**需要** | 公网 HTTPS；交付包含 `frontend\`（Dashboard） |

### 交付包根目录脚本（本模式按序执行）

| 顺序 | 脚本 | 作用 |
|------|------|------|
| 1 | `安装Miniconda.bat` | 安装 Python 3.10 到 `C:\Miniconda3` 并配置 Path |
| 2 | `配置Python环境变量.bat` | 仅重配 Path（步骤 1 已含，可跳过） |
| 3 | 双击 `offline\WinSCP-6.5.6-Setup.exe` | 安装 WinSCP |
| 4 | `离线安装依赖_云端API.bat` | pip 离线依赖 |
| 5 | `初始化配置.bat` | 生成 `app\.env` 等模板 |
| 6 | `注册服务_云端API.bat` | 注册后端(:8000)+前端(:3000) |
| 7 | `启动服务_云端API.bat` | 启动全部服务 |
| 日常 | `停止全部服务.bat` | 停止全部服务 |

**勿运行**：`离线安装_本地GPU.bat`、`注册服务_本地GPU.bat`、`启动服务_本地GPU.bat`

### 目录要点

| 路径 | 说明 |
|------|------|
| `app\` | 后端 API |
| `frontend\` | Dashboard（`node server.js`） |
| `runtime\node\` | 便携 Node（可选） |
| `offline\` | Miniconda、wheel、nssm 等 |

---

## 1. 部署步骤

### 步骤 1：拷贝交付包

拷到 `D:\NodexelOCR\NodexelOCR_Delivery\`。  
**通过标准**：有 `app\`、`frontend\`、`注册服务_云端API.bat`。

### 步骤 2：安装 Python

**右键管理员** → `安装Miniconda.bat` → 新开 PowerShell：

```powershell
python --version
where.exe python
```

应为 `Python 3.10.x` 且 `C:\Miniconda3\python.exe`。否则再运行 `配置Python环境变量.bat`。

### 步骤 3：安装 WinSCP

**双击** `offline\WinSCP-6.5.6-Setup.exe`。

### 步骤 4：安装依赖

**右键管理员** → `离线安装依赖_云端API.bat`。

### 步骤 5：初始化配置

**右键管理员** → `初始化配置.bat`，在 `app\.env` 填：

```ini
VLM_PROVIDER=paddleocr_api
PADDLEOCR_API_TOKEN=your_token_here
```

`app\config\server.properties` 填本机 IP（举例 `server.ip=192.168.10.101`）。

### 步骤 6：验证公网（建议）

```powershell
Test-NetConnection -ComputerName <API域名> -Port 443
```

### 步骤 7：注册并启动

1. **右键管理员** → `注册服务_云端API.bat`
2. **右键管理员** → `启动服务_云端API.bat`

验收：`http://localhost:8000/api/v1/health` 与 `http://localhost:3000/settings`

### 步骤 8：网页配置

打开 `http://localhost:3000/settings` → Save → Restart Backend。

### 步骤 9：验收清单

| 检查项 | 预期 |
|--------|------|
| 8000 health | healthy |
| 3000 settings | 可配置 |
| service_out.log | paddleocr_api |
| ManualEditor.exe | 可启动 |

---

## 2. 日常运维

| 操作 | 脚本 |
|------|------|
| 启动 | `启动服务_云端API.bat` |
| 停止 | `停止全部服务.bat` |

日志：`app\logs\`、`frontend\logs\`

---

## 3. PLM 服务器 OpenSSH（SFTP）

在 PLM 服务器管理员 PowerShell：

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

处理机验证：`Test-NetConnection -ComputerName <PLM_IP> -Port 22`

---

## 4. 常见问题

| 现象 | 处理 |
|------|------|
| 3000 打不开 | 确认 `frontend\` 存在；运行 `注册服务_云端API.bat` 与 `启动服务_云端API.bat` |
| `python` 不是 3.10 | 运行 `配置Python环境变量.bat` |
| 误跑 `离线安装_本地GPU.bat` | 忽略 docker 错误，用 `离线安装依赖_云端API.bat` |
| Token 报错 | 检查 `app\.env` |

---

## 5. 与本地 GPU 模式区别

| 项目 | 云端 | 本地 GPU |
|------|------|----------|
| 依赖脚本 | `离线安装依赖_云端API.bat` | `离线安装_本地GPU.bat` |
| 服务脚本 | `注册/启动服务_云端API.bat` | `注册/启动服务_本地GPU.bat` |
| Docker | 不需要 | 需要 |

正式生产见 [部署手册_本地GPU模式.md](部署手册_本地GPU模式.md)。

---

 [← 返回部署总览](部署手册.md)
