# NodexelOCR 部署手册 — 本地 GPU 模式

> 适用对象：客户现场（**生产环境推荐**）  
> 推理方式：本机 Docker（`VLM_PROVIDER=vllm`）  
> 数据不出厂，不依赖公网 API

 [← 返回部署总览](部署手册.md)

---

## 0. 开始前请确认

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA ≥ 12GB 显存 |
| 磁盘 | ≥ 50GB（含 `NodexelOCR.tar.gz`） |
| 交付包 | 含 `app\`、`frontend\`、`NodexelOCR.tar.gz` |

### 交付包根目录脚本（本模式按序执行）

| 顺序 | 脚本 | 作用 |
|------|------|------|
| 1 | `安装Miniconda.bat` | Python 3.10 + Path |
| 2 | 双击 NVIDIA 驱动安装包 | 显卡驱动 |
| 3 | 双击 `offline\Docker Desktop Installer.exe` | Docker |
| 4 | 双击 `offline\WinSCP-6.5.6-Setup.exe` | WinSCP |
| 5 | `离线安装_本地GPU.bat` | docker load + pip |
| 6 | `初始化配置.bat` | 配置模板 |
| 7 | `注册服务_本地GPU.bat` | 注册后端+前端服务 |
| 8 | `启动服务_本地GPU.bat` | 启动 Docker 容器 + 全部服务 |
| 日常 | `停止全部服务.bat` | 停止服务 |

**勿运行**：`离线安装依赖_云端API.bat`、`注册/启动服务_云端API.bat`

---

## 1. 部署步骤

### 步骤 1：拷贝交付包

确认根目录有 `NodexelOCR.tar.gz`、`frontend\`、`注册服务_本地GPU.bat`。

### 步骤 2：安装 Python

**右键管理员** → `安装Miniconda.bat` → 验证 `python --version` 为 3.10。

### 步骤 3：安装 NVIDIA + Docker + WinSCP

依次双击 `offline\` 中对应安装包。Docker 启动后 `docker version` 无报错。

### 步骤 4：离线安装

**右键管理员** → `离线安装_本地GPU.bat`（载入镜像 + pip）。

### 步骤 5：初始化配置

**右键管理员** → `初始化配置.bat`，确认 `app\.env`：

```ini
VLM_PROVIDER=vllm
```

`server.properties` 填本机 IP（举例 `192.168.20.101`）。

### 步骤 6：注册并启动

1. **右键管理员** → `注册服务_本地GPU.bat`
2. **右键管理员** → `启动服务_本地GPU.bat`（含 `docker run/start nodexel`）

验收：

| 地址 | 预期 |
|------|------|
| `http://localhost:8080/v1/models` | JSON |
| `http://localhost:8000/api/v1/health` | healthy |
| `http://localhost:3000/settings` | 设置页 |

### 步骤 7：网页配置

`http://localhost:3000/settings` → Save → Restart Backend。

---

## 2. 日常运维

| 操作 | 脚本 / 命令 |
|------|-------------|
| 启动全部 | `启动服务_本地GPU.bat` |
| 停止全部 | `停止全部服务.bat` |
| 仅重启容器 | `docker restart nodexel` |

---

## 3. PLM OpenSSH

同云端手册第 3 章（在 PLM 服务器开启 sshd:22）。

---

## 4. 常见问题

| 现象 | 处理 |
|------|------|
| OCR 全空 | 检查 NVIDIA 驱动 / CUDA 12.9 |
| docker load 失败 | 先启动 Docker Desktop |
| 3000 打不开 | 检查 `frontend\` 与服务是否注册 |

---

 [← 返回部署总览](部署手册.md)
