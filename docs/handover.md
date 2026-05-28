# 服务化改造 · 交接文档

> **目标读者**：接手后续 §12 Phase 4–10 改造（队列日志、API、前端、打包、部署）的工程师
> **基准日期**：2026-05-28
> **作者**：Brady Huang（前期负责人）
> **配套文档**：详细架构请参见 [plm_integration_design.md](plm_integration_design.md)（本文件是它的执行视图，不重复其内容）

---

## 0. TL;DR

- 这是一个**已在生产可用**的图纸 Y 编号批量替换系统，目前由人工 GUI 触发。
- 正在改造成**服务化形态**，对接客户（三菱电机）的 **Teamcenter PLM**。
- 改造拆 10 个 Phase，**Phase 1–3 已完成**（环境变量化 + SQLite 队列 + Worker + Watch Folder），smoke 测试通过；**Phase 4–10 待做**。
- 核心 OCR / 替换算法**不在改造范围内**——已经稳定，不要碰。
- `manual_editor/` 子目录是另一条独立产品线（PySide6 桌面工具），**完全不在本次服务化改造范围内**。

---

## 1. 项目需求与目标

### 1.1 客户与场景

- **客户**：三菱电机（日本），装备制造行业
- **场景**：日文工程图纸（PDF / TIF / TIFF），单册 200+ 张
- **痛点**：版本迭代时需要把"Y 编号"（8 位字母数字混排零部件编号）批量改写为新版本号。人工：3–5 分钟/张，5–8% 差错率，2–3 天滞后
- **客户系统**：Teamcenter PLM，工程师在 PLM 工作流里推送图纸 → 期望 OCR 服务自动处理 → 改完的图纸回 PLM 给审核

### 1.2 系统当前能力（**改造前后都保持**）

- 输入：PDF / TIF / TIFF / JPG / PNG 单文件或整目录
- 输出：同名替换文件 + `y_boxes.csv`（所有识别框记录）+ `processing_report.txt`
- 两种处理路径：
  - **矢量 PDF 直改**（< 0.5s/张，0 损失）—— Adobe 设计端导出的原生 PDF
  - **VLM OCR 像素改写**（8–10s/张）—— 扫描件 / 旧 TIF，用 PaddleOCR-VL-1.5 视觉大模型
- 替换准确率：矢量 100%，扫描 90%+；剩余 < 10% 由手动编辑器兜底
- 关键算法：4 类框检测（青/绿/橙/红）+ 三方投票（绿/橙）+ 字段锚定（材料代号 / 零件图号 / 工厂注意）

### 1.3 服务化目标

让客户 IT 在不需要工程师操作 GUI 的前提下完成端到端流程：

```
PLM 工作流触发  →  推图纸到 inbox\        ←─┐
                       ↓                    │ 内网共享盘（UNC 路径）
              OCR 服务监听到新文件          │
                       ↓                    │
              稳定性窗口（3s 大小不变）     │
                       ↓                    │
              移到 processing\ + 入 SQLite  │
                       ↓                    │
              单线程 Worker 独占 GPU 处理   │
                       ↓                    │
              产物落到 output\          ────┘  PLM 监听这里 → 进审核
              失败源文件移到 failed\
```

**不变的承诺**：

- 客户 IT **不动 PLM 任何代码**（PLM 用文件流转触发，不调 SOA API）
- OCR 服务对 PLM **零侵入**
- 所有路径 / 端口 / Docker 配置走 env，IT 改 `.env` 即可，不改 Python

---

## 2. 整体架构（当前状态）

### 2.1 三段式

```
┌─────────────────────────────────────────────────────────────────┐
│                       Windows Host                              │
│                                                                 │
│   ┌──────────────┐   localhost:8080   ┌──────────────────┐     │
│   │              │ ◄──────────────────│ Docker 容器       │     │
│   │ FastAPI app  │   POST /v1/...     │ PaddleOCR-VL-vllm│     │
│   │ (api/main.py)│                    │ （第三方镜像）    │     │
│   │              │                    │                  │     │
│   │  + Worker    │                    │ - GPU 直通       │     │
│   │  (queue 消费)│                    │ - 模型卷挂载     │     │
│   │              │                    │ - 业务代码不在内 │     │
│   │  + WatchFold │                    └──────────────────┘     │
│   │  (扫 inbox)  │                                             │
│   │              │   SQLite                                    │
│   │              │ ◄────────┐  data/queue.db (WAL)             │
│   └──────────────┘          │                                  │
│         ▲                   │                                  │
│         │ HTTP              │                                  │
│         │ :8000             │                                  │
└─────────┼───────────────────┼──────────────────────────────────┘
          │                   │
   ┌──────┴───────┐    ┌──────┴───────┐
   │ Dashboard    │    │ UNC 共享盘    │
   │ (Next.js)    │    │ inbox\        │
   │ 当前位置:    │    │ processing\   │
   │ 工程师本地   │    │ output\       │
   └──────────────┘    │ failed\       │
                       │ ← PLM 写入 / 读取 │
                       └────────────────┘
```

### 2.2 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 队列存储 | **SQLite WAL 模式** | 单进程单 Worker，无并发瓶颈；自带持久化、零运维；客户不用部 Redis/Postgres |
| Worker 并发 | **单线程顺序消费** | GPU 显存 12 GB，VLM 推理已经吃满，不能并发 |
| 触发机制 | **文件夹轮询**（不用 watchdog） | UNC 路径上 ReadDirectoryChangesW 经常掉事件；轮询稳定 |
| 稳定性窗口 | **3 秒 size+mtime 不变** | 防止 PLM 写到一半就被读 |
| OCR 推理服务 | **第三方 Docker 镜像** | PaddlePaddle 官方镜像，复杂依赖（CUDA + vLLM + Paddle）打好了 |
| 业务代码部署 | **Windows host 原生 Python**（不进 Docker） | manual_editor 是 PySide6 桌面 GUI，host 上必然要装 Python；分两份维护成本不划算 |

---

## 3. 目录与文件总览

```
Mitsubishi-Electric-Drawing-Replacement-Project/
│
├── config.py                     ★ 全局配置：所有 env 入口在这里
├── start_v2.py                   后端启动脚本（python start_v2.py）
├── start_v2.bat / .vbs           Windows 快捷启动
├── start_vllm_server.bat         单独启 Docker 容器（不启 FastAPI）
├── vllm_config.yaml              vLLM 推理引擎超参
├── requirements_local.txt        host 端 Python 依赖
├── .env.example                  ★ 所有可调 env 的样例与说明
├── .gitignore                    （已加入 data/*.db、data/processed/ 等）
├── CLAUDE.md                     LLM 协作规范（人也能看）
│
├── api/                          ★ FastAPI 服务端
│   ├── main.py                   app 实例、CORS、路由注册、OCR 预热
│   └── routes/
│       ├── health.py             /api/v1/health
│       ├── drawings.py           （V2 历史接口，暂未启用）
│       ├── jobs.py               批处理任务启动/取消/SSE 状态推送（GUI 用）
│       ├── gpu.py                /api/v1/gpu/* GPU 监控
│       ├── folders.py            目录浏览（dashboard 选择 input/output 用）
│       └── system.py             /api/v1/system/* 系统设置（雏形）
│
├── modules/                      ★ 业务逻辑
│   │
│   │  ── 核心算法（不要碰）──
│   ├── file_ingestion.py         读 PDF/TIF/PNG → numpy array
│   ├── pdf_vector_handler.py     矢量 PDF 路径（fast path）
│   ├── region_detector.py        4 类框检测主算法（3136 行，最复杂）
│   ├── factory_note_pixel.py     工厂注意区像素级检测（1200 行）
│   ├── vlm_ocr_engine.py         PaddleOCR-VL-1.5 HTTP 客户端
│   ├── text_replacer.py          行内文字替换 + 字形匹配（1814 行）
│   ├── batch_processor.py        process_single_file() 入口编排
│   │
│   │  ── 服务化新增（Phase 1–3）──
│   ├── docker_manager.py         启停 Docker 容器、健康检查（Phase 1 改 env 化）
│   ├── job_queue.py              ★ SQLite 队列（Phase 2 新增）
│   ├── worker.py                 ★ 单线程 Worker 消费者（Phase 2 新增）
│   ├── watch_folder.py           ★ inbox 轮询监听（Phase 3 新增）
│   └── filename_parser.py        ★ 文件名解析 drawing_no / revision（Phase 3 新增）
│
├── docs/                         ★ 设计文档
│   ├── plm_integration_design.md 架构主文档（必读）
│   ├── handover.md               ← 本文件
│   └── legacy/                   V1 时期部署文档（已 deprecated，Phase 10 会重写）
│
├── manual_editor/                ★★ 完全独立的桌面 GUI 工具，不属于服务化改造范围
│   ├── main.py                   PySide6 入口
│   ├── app/                      MVVM 模块
│   ├── tests/                    pytest 用例
│   └── README.md
│
├── dashboard/                    Next.js 前端（V2，工程师 GUI 用）
│   ├── src/app/                  路由
│   ├── src/components/           组件（FileTable 等）
│   ├── src/types.ts              ★ 改后端常量时这里要同步
│   └── package.json
│
├── data/                         ★ 持久化数据（git 忽略内容，保留目录）
│   ├── queue.db                  SQLite 队列文件（WAL）
│   ├── processed/                Worker 内部产物根
│   └── watch/
│       ├── inbox/                PLM 投递入口（生产换成 UNC）
│       ├── processing/           处理中（防 PLM 重复触发）
│       ├── output/               处理完产物（PLM 监听）
│       └── failed/               失败源文件
│
├── models/                       PaddleOCR-VL 模型权重（>5GB，git 忽略）
├── fonts/                        中日替换字体
├── logs/                         运行日志
│
├── test_queue_worker_smoke.py    ★ Phase 2 烟雾测试
└── test_watch_folder_smoke.py    ★ Phase 3 端到端烟雾测试
```

### 3.1 关键文件单行说明

| 文件 | 行数 | 一句话 |
|---|---|---|
| `config.py` | 154 | 唯一的 env 入口；任何新 env 都加这里 |
| `modules/region_detector.py` | 3136 | 核心算法，4 类框检测；**不要重构** |
| `modules/text_replacer.py` | 1814 | OCR + 字形替换；**不要重构** |
| `modules/factory_note_pixel.py` | 1200 | 工厂注意区检测；`H_FACTORS=[8]` 锁定 |
| `modules/batch_processor.py` | 295 | `process_single_file(file, output_dir)` 是核心入口 |
| `modules/job_queue.py` | 238 | SQLite WAL 队列，6 个方法看完就懂 |
| `modules/worker.py` | 256 | 单线程消费者；watch_folder 任务产物落 output\\、失败移 failed\\ |
| `modules/watch_folder.py` | 221 | 轮询 + 稳定性 + enqueue |
| `modules/filename_parser.py` | 63 | `parse(filename) -> ParsedFilename(drawing_no, revision, matched)` |
| `api/main.py` | 61 | FastAPI 入口；当前**没有挂 Worker / Watch Folder 启动**（见 §6） |
| `api/routes/jobs.py` | 338 | GUI 批处理接口（与新队列**并存**，未合并） |

---

## 4. 环境配置与运行要求

### 4.1 开发环境（前任使用）

- **OS**：Windows 11 / Windows Server 2019+
- **GPU**：NVIDIA RTX 40/50 系，**显存 ≥ 12 GB**（VLM 推理硬性要求）
- **CUDA / 驱动**：随 PaddleOCR-VL Docker 镜像（host 只需要 NVIDIA 驱动 + Docker Desktop）
- **Python**：3.10+ via Miniconda env `mitsubishi`
- **完整 Python 路径（**不要用 `conda run`**）**：
  ```
  C:\Users\Brady Huang\miniconda3\envs\mitsubishi\python.exe
  ```
- **Docker Desktop**：4.x，开启 WSL2 backend + NVIDIA Container Toolkit
- **Node.js**：dashboard 前端用，Next.js 16

### 4.2 启动顺序（手动开发模式）

```powershell
# 1. 启 vLLM 推理容器（等模型加载约 3 分钟，端口 8080）
.\start_vllm_server.bat

# 2. 启后端 API（端口 8000）
.\start_v2.bat
#  or:  python start_v2.py

# 3. 启前端（端口 3000）
cd dashboard && npm run dev
```

### 4.3 关键依赖版本（host 端）

详见 `requirements_local.txt`。重点：

```
paddlepaddle==3.0.0           # 锁版本，新版有 API 兼容问题
paddlex>=3.0.0
PyMuPDF>=1.23                 # 矢量 PDF 路径用
opencv-python-headless>=4.8
Pillow>=10.0
gradio>=4.40.0                # 旧 V1 UI 残留，可考虑下阶段移除
python-dotenv>=1.0            # ★ Phase 1 新加
```

### 4.4 关键环境变量

所有 env 都有默认值（=改造前的硬编码值），**不配置时行为完全等价**。完整清单见 `.env.example`，部署时只需关注：

| env | 默认 | 生产部署典型值 |
|---|---|---|
| `API_PORT` | 8000 | 8000 或客户指定 |
| `API_CORS_ORIGINS` | `*` | `https://plm.intranet.example.com`（收紧） |
| `VLLM_BASE_URL` | `http://localhost:8080/v1` | 同主机 → 不改 |
| `DOCKER_IMAGE` | `paddleocr-genai-vllm-server:latest-nvidia-gpu` | 客户可换私有 registry |
| `VLLM_MODEL_DIR` | `<repo>/models/PaddleOCR-VL-1.5` | 客户共享盘上 → 改 UNC |
| `WATCH_INBOX_DIR` | `data/watch/inbox` | `\\plmserver\share\drawings\inbox` |
| `WATCH_OUTPUT_DIR` | `data/watch/output` | `\\plmserver\share\drawings\processed` |
| `WATCH_FAILED_DIR` | `data/watch/failed` | `\\plmserver\share\drawings\failed` |
| `WATCH_PROCESSING_DIR` | `data/watch/processing` | 建议留在**本机磁盘**（性能 + 防 PLM 二次触发） |
| `WATCH_STABILITY_SECONDS` | `3.0` | 大文件可调高到 5–10 |
| `WORKER_MAX_RETRY` | `1` | 客户 IT 决定 |

### 4.5 路径坑

仓库克隆位置：

- **Windows 视角**：`C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project`
- **Git Bash / Python**：同上
- **Claude Code 的 Write 工具视角**：`C:\Users\Brady Huang\Mitsubishi-Electric-Drawing-Replacement-Project`

两条路径指向**同一个目录**（Windows 用户名映射），不是两份代码。Python / Bash 走 `Brady Huang`，工具链如果路径报"not found"换另一个试一下。

---

## 5. 已完成的改造（Phase 1–3）

### 5.1 Phase 1 — 路径外部化

**目的**：让客户 IT 部署时不动 Python，只改 `.env` 就能改端口/路径/CORS。

**做了什么**：

- `config.py` 顶部加 `load_dotenv()`（可选依赖，未装也能跑）
- 把以前硬编码在多处的路径、端口、容器名、镜像名、CORS、输出子目录名全抽到 `os.getenv()`
- 所有 env 默认值 = 改造前硬编码值，**0 行为差异**
- `requirements_local.txt` 加 `python-dotenv>=1.0`
- `.env.example` 全量补齐 + 中文说明

**改动文件**：`config.py`, `start_v2.py`, `api/main.py`, `api/routes/jobs.py`, `modules/docker_manager.py`, `modules/batch_processor.py`, `.env.example`, `requirements_local.txt`

**怎么验证已完成**：不放 `.env` 跑 `python start_v2.py` → 启动行为与改造前一致。

### 5.2 Phase 2 — SQLite 队列 + Worker

**目的**：把"工程师在前端点开始 → 跑完一次性结束"的人工模式，变成"任务持久化 + Worker 独立消费"，为 Watch Folder 和未来 PLM 接入做基础。

**做了什么**：

- 新增 `modules/job_queue.py`（238 行）：
  - SQLite WAL 模式，单文件 `data/queue.db`
  - 表 `job_queue`：`id / source / source_file / file_path / drawing_no / revision / status / retry_count / created_at / started_at / finished_at / error_msg / result_path`
  - 关键方法：`enqueue / claim_next_pending（原子 pending→running）/ mark_done / mark_failed / requeue_for_retry / counts / reset_stale_running`
  - 所有写入用 `threading.Lock` 串行
- 新增 `modules/worker.py`（256 行）：
  - 单线程 daemon，`start()` / `stop()`
  - 循环：`claim_next_pending → process_single_file → mark_done/failed`
  - 启动时 `reset_stale_running()` 自动把崩溃留下的 running 转回 pending
  - 失败重试：`WORKER_MAX_RETRY` 次以内自动 requeue
  - VLM 预热（`ensure_vlm_ready`）
- `config.py` 加：`DATA_DIR / QUEUE_DB_PATH / WORKER_OUTPUT_DIR / WORKER_POLL_INTERVAL / WORKER_MAX_RETRY`
- `.gitignore` 加：`data/*.db / *.db-journal / *.db-wal / *.db-shm / data/processed/`
- 新增 `test_queue_worker_smoke.py` —— **PASS**（3 个任务全部 done + 重试 retry_count=1 验证通过）

**未挂载到 FastAPI**：当前 `api/main.py` 启动时**不**自动启 Worker。这是故意的——保证 V2 后端在 Phase 4 完成前不破坏现有 GUI 流程。Worker 可以独立跑：`python -m modules.worker`。

### 5.3 Phase 3 — Watch Folder + 文件名解析

**目的**：实现 PLM 通过文件流转触发处理（不调 PLM API），并把 drawing_no / revision 元数据带进队列。

**做了什么**：

- 新增 `modules/filename_parser.py`（63 行）：
  - `parse(filename) -> ParsedFilename(drawing_no, revision, matched)`
  - 优先用 `config.make_pattern()` 项目通用 Y 编号正则
  - 版本号优先级：`_Rev2` > `_R1` > `_v3` > `-A`
  - 全部失败 → 兜底 `(stem, "N/A", matched=False)`，**任务依然入队**，不丢图纸
- 新增 `modules/watch_folder.py`（221 行）：
  - 不依赖 watchdog，纯 `os.listdir` 轮询（UNC 路径上更可靠）
  - 跟踪每个文件的 `(size, mtime)`，连续 `WATCH_STABILITY_SECONDS` 秒不变才认稳定
  - 稳定后：移到 processing\\（同名加 `_001` 后缀防覆盖）→ enqueue (source='watch_folder')
  - `start() / stop()` 同 Worker
- `modules/worker.py` 扩展：
  - 新参数 `watch_output_dir / watch_failed_dir`
  - 处理成功 + `source='watch_folder'`：把产物**复制**一份到 `WATCH_OUTPUT_DIR`（PLM 监听这里），同名加时间戳
  - 失败穷尽重试 + `source='watch_folder'`：把 processing\\ 里的源文件**移动**到 `WATCH_FAILED_DIR`，同名加时间戳
- `config.py` 加：`WATCH_INBOX_DIR / WATCH_PROCESSING_DIR / WATCH_OUTPUT_DIR / WATCH_FAILED_DIR / WATCH_STABILITY_SECONDS / WATCH_SCAN_INTERVAL`
- 新增 `test_watch_folder_smoke.py` —— **PASS 9/9**：
  - 早期扫描（< stability）不入队 ✓
  - done=1（成功） ✓
  - failed=1（故意失败） ✓
  - inbox 清空 ✓
  - watch_output 有 1 个产物 ✓
  - watch_failed 有 1 个源文件 ✓
  - drawing_no='YA111111' 解析正确 ✓
  - source='watch_folder' ✓
  - result_path 实际存在 ✓

### 5.4 当前 git 状态

**Phase 1–3 全部成果 → 尚未提交**：

```
 M .env.example
 M .gitignore
 M api/main.py
 M api/routes/jobs.py
 M config.py
 M modules/batch_processor.py
 M modules/docker_manager.py
 M requirements_local.txt
 M start_v2.py
?? data/                              # 只 .gitkeep
?? docs/plm_integration_design.md     # ★ 架构主文档
?? docs/handover.md                   # ← 本文件
?? modules/filename_parser.py
?? modules/job_queue.py
?? modules/watch_folder.py
?? modules/worker.py
?? test_queue_worker_smoke.py
?? test_watch_folder_smoke.py
```

**推荐提交策略**：3 个原子 commit（Phase 1 / Phase 2 / Phase 3 分开），方便后续 `git revert` 单个阶段；也可合并为一个 commit。最后决定权在你/客户。

`.tmp_pptx/` 是 PPT 临时产物，**不要提交**。

---

## 6. 未完成的改造（Phase 4–10）

按 `plm_integration_design.md` §12 的工作量表：

### 6.1 Phase 4 — 日志表 + N/O 自动判定 + 元数据兜底（2~3 天）

**目的**：

- 队列日志和处理日志分离（queue.db 只管 pending/running/done/failed 状态机；process_log.db 是审计日志）
- 文件能 fail-fast 判断走矢量还是 OCR 路径（目前已经在 `batch_processor.is_vector_pdf()` 里有判定，需要在元数据层显式记录）
- 文件名解析失败时（matched=False），实现 §3.7 描述的 sidecar 兜底（`xxx.tif.meta.json`）

**关键变更点**：

- 新建 `modules/process_log.py`，schema 见 `plm_integration_design.md` §8.2
- `worker.py` 在 `process_single_file` 完成后写一条 process_log（包含 method=vector/ocr、用时、替换数、错误信息）
- `filename_parser.py` 加 sidecar 解析分支
- 新增 `data/process_log.db` 路径配置

### 6.2 Phase 5 — 内部 API（鉴权 + 日志查询 + 设置）（2~3 天）

**目的**：为前端三个新页面准备 REST API。

**关键变更点**：

- `api/routes/` 新增：
  - `auth.py`：简单的 token / basic auth（客户内网，要求不高）
  - `logs.py`：`GET /api/v1/logs?status=&from=&to=&drawing_no=` 查 process_log.db
  - `settings.py`：完善 `system.py`，对接 `app_config/` 的运行时配置（端口、稳定性窗口、重试次数等）
- 与现有 `jobs.py`（GUI 批处理用）的关系：保留，但是**逐步迁移**到队列模式（前端"开始任务"按钮变成 enqueue，进度面板变成订阅队列状态）

**接口规范**：详见 `plm_integration_design.md` §7

### 6.3 Phase 6 — 前端三页（4~6 天）

**目的**：dashboard 增加：

1. **系统设置页**：配置 Watch 目录、Worker 参数、PLM 模板等
2. **待审核列表页**：列出 watch_folder 任务，查看每张图、查看 y_boxes.csv，触发 manual_editor 修正
3. **日志页**：分页 + 过滤的处理日志查询

**关键变更点**：

- `dashboard/src/app/` 新增三个路由 / 页面
- `dashboard/src/types.ts` 同步后端的 status / source / method 枚举
- 复用 `dashboard/src/components/FileTable.tsx` 的列表展示

**注意**：前期决定**前端先延后**（用户原话："等接口讨论再做"）。Phase 5 接口稳定后再启动 Phase 6。

### 6.4 Phase 7 — manual_editor 命令行参数 + URL 协议注册（1~2 天）

**目的**：dashboard 待审核列表点击某张图 → 浏览器跳 `manual-editor://open?file=...` → 调起本地 PySide6 编辑器。

**关键变更点**：

- `manual_editor/main.py` 接受 `--file <path>` 参数
- Windows 注册表写 URL 协议（`HKEY_CLASSES_ROOT\manual-editor`）
- 配合 Phase 9 的安装器自动注册

**重要**：`manual_editor/` 子目录是**独立产品线**，前任明确说过"不要动 manual_editor/"。Phase 7 只是给它加 CLI 入口，**不重构其内部**。

### 6.5 Phase 8 — Docker 化（**已排除，详见 §6.7**）

### 6.6 Phase 9 — PyInstaller 打包（2~3 天）

**目的**：把 host 端 Python 服务打成 Windows .exe，客户 IT 一个文件装完。

**关键变更点**：

- 后端 + Worker + WatchFolder 打成单个 `mitsubishi-ocr-service.exe`，注册为 Windows Service
- `manual_editor` 打成 `mitsubishi-manual-editor.exe`，配合 URL 协议
- 安装器（推荐 Inno Setup）：放权重 + 写 env + 注册服务 + 注册 URL 协议

### 6.7 Phase 8 — Docker 化（**当前明确排除**）

**当前状态**：**不做**。

**为什么排除**：

- 客户 IT 是否能在 Windows Server 上装 WSL2 + Docker Desktop 未确认（>250 人企业用 Docker Desktop 需付费）
- GPU 直通 WSL2 故障定位要在 host / WSL / container 三层之间穿
- manual_editor 是 PySide6 桌面 GUI，**只能 Windows 原生跑**，做了 Docker 也要同时维护一份 host 部署
- UNC 路径多绕一层（PLM → Windows → WSL2 → docker volume），故障面变大

**只有什么时候做**：客户 IT 明确要求"全 docker / docker-compose 部署"时。Phase 9 完成后随时可以补做，不阻塞首期上线。

**注意区分**：**第三方** PaddleOCR-VL Docker 镜像（业务必需，永远要用）≠ **自己打** Docker 镜像（Phase 8，已排除）。`modules/docker_manager.py` 管的是前者，不在排除范围。

### 6.8 Phase 10 — 部署文档 + 操作手册 + 联调（3~5 天）

**目的**：交付物完整闭环。

**关键变更点**：

- 部署手册（客户 IT 视角）：硬件清单、装机步骤、env 调整、故障排查
- 操作手册（工程师视角）：dashboard 怎么用、待审核怎么处理、错乱怎么找回
- 联调：和客户 PLM 工程师对接 inbox / output 路径权限、文件命名约定

---

## 7. 给接手人的具体建议

### 7.1 上手第一周

1. **跑通现有流程**（不改任何代码）：
   - 装 `requirements_local.txt`，启 vLLM 容器，跑 `python start_v2.py`，前端 `npm run dev`
   - 用 dashboard GUI 处理 5 张测试图，确认输出正确
2. **跑两个 smoke 测试**：
   ```bash
   "C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_queue_worker_smoke.py
   "C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_watch_folder_smoke.py
   ```
   两个都应该 PASS。
3. **读三份文档**：
   - `docs/plm_integration_design.md`（架构 + 接口规范 + DB schema 全在这）
   - `docs/handover.md`（本文件）
   - `CLAUDE.md`（协作风格）

### 7.2 推荐做事顺序

```
Phase 4（日志表 + 元数据兜底）       ← 起点，1 周内可以做完
  ↓
把 Worker / WatchFolder 挂到 FastAPI startup  ← 目前是分离的，Phase 5 之前合并
  ↓
Phase 5（API 鉴权 + 日志查询）       ← 给前端备料
  ↓
Phase 7（manual_editor CLI）         ← 简单，穿插着做
  ↓
Phase 6（前端三页）                  ← Phase 5 接口稳定后做
  ↓
Phase 9（PyInstaller）               ← 收尾
  ↓
Phase 10（文档 + 联调）              ← 收尾
```

### 7.3 不要做的事

| 不要 | 为什么 |
|---|---|
| 重构 `region_detector.py` / `text_replacer.py` / `factory_note_pixel.py` | 核心算法，3000+ 行，改一行可能影响识别率 |
| 改 `H_FACTORS` 数组 | 已锁定 `[8]`；`h=15` 弃用 |
| 启用 `scan_all_tables` | 已确认不要多表扫描 |
| 用字符比例换算 | 客户明确反对（前任反复强调） |
| 动 `manual_editor/` 内部 | 独立产品线，不归服务化改造管 |
| 改 dashboard 那几个 V2 历史组件 | 等 Phase 5 接口稳定再统一改 |
| 跑 `conda run` 启 Python | 用完整路径 `C:\Users\Brady Huang\miniconda3\envs\mitsubishi\python.exe` |
| 单步提交 `.tmp_pptx/` 目录 | PPT 临时产物，不入仓库 |
| 自作主张做 Phase 8（Docker 化） | 已排除，做之前先和客户/Brady 确认 |
| 把 V2 GUI 的 `api/routes/jobs.py` 改成走队列 | 那是给工程师本地 GUI 用的，**保留并存**，等前端三页做完再考虑合并 |

### 7.4 通用质量原则

来自 `CLAUDE.md`：

- 不假设，有疑问问；有多种解读直说，不要静默选
- 最小代码解决问题，不为"将来灵活"加抽象
- 改动只动该改的；不顺手"改进"无关代码
- 任务转成可验证目标（"加校验"→"先写失败测试再让它过"）

### 7.5 待客户确认（前任挂起，影响 Phase 4–6）

- 日志是否含 M（manual_editor 修正记录）→ 影响 process_log schema
- failed\\ 是否需要 PLM 监听 / 邮件通知 → 影响 Phase 5 接口
- PLM 元数据要不要带版本字段 → 影响 sidecar 解析
- inbox / output 路径具体是哪个 UNC → 影响 Phase 10 部署清单

详见 `plm_integration_design.md` §10。

---

## 8. 常用命令速查

```bash
# Python 启服务（开发）
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" start_v2.py

# 跑 smoke 测试
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_queue_worker_smoke.py
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_watch_folder_smoke.py

# 单独跑 Worker（不启 FastAPI）
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" -m modules.worker

# 单独跑 WatchFolder（不启 FastAPI / Worker）
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" -m modules.watch_folder

# 检查 vLLM 健康
curl http://localhost:8080/v1/models

# 看队列状态
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" -c "from modules.job_queue import JobQueue; print(JobQueue().counts())"

# 前端
cd dashboard && npm install && npm run dev
```

---

## 9. 联系与背景


- **GitHub 仓库**：见 `git remote -v`
- **环境**：conda env `mitsubishi`
- **开发期最常用工具**：Claude Code（VSCode 扩展），见 `CLAUDE.md`

如果新接手人需要回溯任何一个 Phase 的细节决策，先看 `plm_integration_design.md`，再 `git log -p modules/<file>.py` 看具体改动，最后实在不清楚再回来问。

