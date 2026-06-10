# NodexelOCR 图纸编号批量替换系统

> 工程图纸自动识别 + 编号批量改写 + PLM 集成 OCR 服务
> 当前状态：NodexelOCR 自打镜像 + Nuitka 编译 + manual_editor exe 均已完成；PLM Oracle 对接由对方实现

---

## 这是什么

把工程图纸（PDF / TIF / TIFF / JPG / PNG）里以 `Y` / `X` / `B` / `H` 开头的零部件编号（如 `YA026D941`）自动改写为新版本号（`HYA026D941`），单张耗时 < 10 秒，准确率 95%+。

**两条处理路径自动切换**：

| 路径 | 适用 | 耗时/张 | 准确率 |
|---|---|---|---|
| 矢量 PDF 直改 | 设计端导出原生 PDF | < 0.5s | 100% |
| VLM OCR 像素改写 | 扫描件 / 旧 TIF | 8–10s | 90%+ |

剩余 < 10% 异常由独立桌面工具 [manual_editor](./manual_editor/README.md) 手动兜底（PySide6，CSV 框定位 + 拖拽改写 + 锁定确认）。

---

## 快速开始

### 环境要求

- Windows 10/11 或 Windows Server 2019+
- NVIDIA GPU，显存 ≥ 12 GB（VLM 推理硬性要求）
- Miniconda + conda env `mitsubishi`（Python 3.10）
- Docker Desktop（跑 NodexelOCR 推理容器用）
- Node.js（前端用）

> ⚠️ paddlepaddle-gpu 必须装 **CUDA 12.9 构建版**（新架构 GPU 如 RTX 50 系必需）；
> 装错 CUDA 版本会导致 OCR 静默返回空。详见 `requirements_local.txt` 注释。

### 三步启动

```bash
# 1. 启动 NodexelOCR 推理容器（模型已封装进镜像，零挂载；端口 8080）
#    点"开始识别"时由 docker_manager.py 自动启动；也可手动启：
docker run -d --gpus all -p 8080:8080 nodexelocr:v1

# 2. 启动后端 API（端口 8000）
.\start_v2.bat

# 3. 启动前端（端口 3000）
cd dashboard && npm install && npm run dev
```

---

## 文档导航

| 文档 | 内容 |
|---|---|
| **[docs/v4_final_roadmap.md](docs/v4_final_roadmap.md)** | **权威方案**：NodexelOCR 自打镜像 + Nuitka 编译 + PLM Oracle 直连 |
| **[docs/实现状态说明.md](docs/实现状态说明.md)** | 当前实现逻辑 + PLM 对接待办 + 未完成事项 |
| [docs/region_detection_pipeline.md](docs/region_detection_pipeline.md) | 区域检测算法说明 |
| [dashboard/README.md](dashboard/README.md) | Next.js 前端说明 |
| [manual_editor/README.md](manual_editor/README.md) | PySide6 桌面工具说明 |

<!--PART2-->

## 仓库结构（精简）

```
.
├── config.py                     全局配置入口（已编译为 config.pyd 交付）
├── start_v2.py / .bat / .vbs     后端启动脚本（.vbs 隐藏窗口）
├── vllm_config.yaml              推理超参（已 COPY 进 NodexelOCR 镜像）
├── requirements_local.txt        host Python 依赖（注意 CUDA 12.9）
├── .env.example                  可调 env 示例
│
├── docker/                       NodexelOCR 镜像构建
│   ├── Dockerfile.nodexel        自打镜像（模型 COPY 进内部）
│   └── entrypoint.sh             容器启动脚本
│
├── api/                          FastAPI 服务端（交付保留 .py）
│   ├── main.py                   app 入口 + CORS + 路由注册
│   └── routes/                   health / jobs / gpu / folders / system / drawings
│
├── modules/                      业务逻辑（交付编译为 .pyd）
│   ├── batch_processor.py        process_single_file() 主入口
│   ├── region_detector.py        区域检测核心算法（不要碰）
│   ├── text_replacer.py          OCR + 字形替换（不要碰）
│   ├── factory_note_pixel.py     工厂注意区检测（不要碰）
│   ├── pdf_vector_handler.py     矢量 PDF 路径
│   ├── vlm_ocr_engine.py         推理服务 HTTP 客户端
│   ├── docker_manager.py         推理容器启停（nodexelocr:v1 零挂载）
│   ├── job_queue.py / worker.py  SQLite 队列 + 单线程 Worker
│   ├── process_log.py            处理日志表（记 O/N，开放给 PLM 访问）
│   └── watch_folder.py / filename_parser.py  文件夹监听（测试/备用入口，生产不启动）
│
├── dashboard/                    Next.js 前端
├── manual_editor/                PySide6 桌面工具（交付为 ManualEditor.exe）
├── build_tools/                  PyInstaller 打包配置
├── scripts/                      编译 / 打包 / 服务注册脚本
├── docs/                         设计文档
├── fonts/                        中日字体（替换字形用）
└── data/                         队列 DB + 处理产物（git 忽略）

注：模型权重不放宿主机，已封装进 NodexelOCR 镜像内部（零挂载）。
```

---

## 关键技术约束（不要踩坑）

| 约束 | 说明 |
|---|---|
| paddle 必须装 CUDA 12.9 构建版 | 装错（如 cu126）会导致 OCR 静默返回空 |
| 不要重构 region_detector / text_replacer / factory_note_pixel | 核心算法已稳定 |
| 不要动 manual_editor 内部 | 独立产品线 |
| 用完整 Python 路径，不用 conda run | `...\miniconda3\envs\mitsubishi\python.exe` |

完整说明见 [docs/实现状态说明.md](docs/实现状态说明.md)。

