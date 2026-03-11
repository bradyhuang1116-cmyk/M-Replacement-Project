# Docker 网络问题解决方案

## 问题描述
Docker 无法从 Docker Hub 拉取镜像，错误信息：
```
failed to fetch oauth token: Post "https://auth.docker.io/token": dial tcp 150.107.3.176:443: connectex
```

## 解决方案1：配置Docker镜像加速器（推荐）

### Windows Docker Desktop

1. 打开 Docker Desktop
2. 点击右上角设置图标（齿轮）
3. 选择 "Docker Engine"
4. 在配置文件中添加以下内容：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

5. 点击 "Apply & Restart"
6. 等待 Docker 重启完成

### 验证配置
```bash
docker info | grep -A 5 "Registry Mirrors"
```

### 重新构建
```bash
cd Mitsubishi-Electric-Drawing-Replacement-Project
docker-compose down
docker-compose up -d --build
```

---

## 解决方案2：使用阿里云镜像加速器

### 获取专属加速地址
1. 访问：https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors
2. 登录阿里云账号（免费注册）
3. 获取你的专属加速器地址（格式：https://xxxxx.mirror.aliyuncs.com）

### 配置Docker
在 Docker Engine 配置中添加：
```json
{
  "registry-mirrors": [
    "https://xxxxx.mirror.aliyuncs.com"
  ]
}
```

---

## 解决方案3：手动拉取镜像

如果镜像加速器配置后仍然失败，尝试手动拉取：

```bash
# 使用镜像加速器拉取
docker pull docker.mirrors.ustc.edu.cn/library/python:3.11-slim

# 重新标记为原始名称
docker tag docker.mirrors.ustc.edu.cn/library/python:3.11-slim python:3.11-slim

# 然后再构建
docker-compose up -d --build
```

---

## 解决方案4：放弃Docker，使用本地Python部署

如果Docker问题无法解决，直接使用本地Python环境：

### 步骤1：确认Python版本
```bash
python --version
# 必须是 3.11.x 或 3.12.x
```

### 步骤2：安装依赖
```bash
cd Mitsubishi-Electric-Drawing-Replacement-Project
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤3：环境检查
```bash
python env_checker.py
```

### 步骤4：启动服务
```bash
# Windows
start.bat gradio

# 或直接运行
python web_app.py
```

---

## 常见问题

### Q1：配置镜像加速器后仍然失败
**A**：尝试多个镜像源，或使用解决方案4（本地Python部署）

### Q2：Docker Desktop无法启动
**A**：
1. 检查是否启用了虚拟化（BIOS设置）
2. 检查Windows功能中是否启用了"Hyper-V"和"容器"
3. 重启电脑

### Q3：pip安装依赖很慢
**A**：使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 推荐部署方案（按优先级）

### 方案A：Docker + 镜像加速器
- 优点：环境隔离，一键部署
- 缺点：需要配置网络

### 方案B：本地Python环境
- 优点：不依赖Docker，网络要求低
- 缺点：需要手动管理Python版本和依赖

**建议**：
- 如果网络条件好，使用方案A
- 如果网络受限或Docker配置困难，使用方案B
