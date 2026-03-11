# Docker 镜像加速器配置指南（Windows）

## 问题原因
Docker 无法连接到 auth.docker.io，导致无法拉取基础镜像。

## 解决步骤

### 步骤1：打开 Docker Desktop 设置

1. 确保 Docker Desktop 正在运行
2. 点击任务栏的 Docker 图标
3. 点击右上角的 **设置图标**（齿轮）

### 步骤2：配置镜像加速器

1. 在左侧菜单选择 **"Docker Engine"**
2. 你会看到一个 JSON 配置编辑器
3. 将配置修改为以下内容（完整替换）：

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
```

4. 点击 **"Apply & Restart"**
5. 等待 Docker 重启完成（约30秒）

### 步骤3：验证配置

打开命令行，运行：
```bash
docker info
```

查找输出中的 "Registry Mirrors" 部分，应该显示：
```
Registry Mirrors:
  https://docker.mirrors.ustc.edu.cn/
  https://hub-mirror.c.163.com/
```

### 步骤4：重新构建

```bash
cd C:\Users\HYMOD\Mitsubishi-Electric-Drawing-Replacement-Project
docker-compose down
docker-compose up -d --build
```

## 如果仍然失败

### 方案A：手动拉取镜像

```bash
# 尝试手动拉取
docker pull python:3.11-slim
```

如果失败，说明网络环境完全无法访问 Docker Hub。

### 方案B：放弃 Docker，使用 Conda 环境

1. 安装 Miniconda：https://docs.conda.io/en/latest/miniconda.html
2. 创建 Python 3.11 环境：
```bash
conda create -n mitsu python=3.11 -y
conda activate mitsu
```

3. 安装依赖：
```bash
cd C:\Users\HYMOD\Mitsubishi-Electric-Drawing-Replacement-Project
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 启动服务：
```bash
python web_app.py
```

## 推荐方案

**如果你的网络环境无法访问 Docker Hub**，强烈建议使用方案B（Conda环境）。

这样：
- 不依赖 Docker 网络
- 可以使用国内 pip 镜像源
- 部署更快速
- 对单机部署完全够用
