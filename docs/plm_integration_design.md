# OCR 服务化与 PLM (Teamcenter) 集成设计

> **状态**：架构设计已定型，待客户/TC 顾问确认少量细节后即可进入实现。
> **更新日期**：2026-05-28
> **作者**：Brady Huang
> **适用范围**：作为客户/TC 顾问对接文档 + 内部实现参考依据。

---

## 1. 项目背景

当前项目包含两个独立模块：

| 模块 | 路径 | 形态 | 职责 |
|---|---|---|---|
| **OCR + 替换管线（V2）** | `api/`、`dashboard/`、`modules/` | FastAPI + Next.js | 端到端处理 TIF/PDF，识别图纸编号并自动替换 |
| **手动修正工具** | `manual_editor/` | PySide6 桌面应用 | 审核员对替换结果手动调整框/文字 |

客户的 Teamcenter (PLM) 系统希望将上述功能集成进图纸审核工作流。为满足"可被 PLM 调用"的要求，两个模块需要从"开发者运行的工具"改造为"客户机房可部署的服务"。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          PLM (Teamcenter)                        │
│                                                                 │
│  设计师 check-in 图纸 → 工作流节点"送 OCR"                         │
│            │                                                    │
│            │ 文件复制到 inbox（文件名保持原图号命名）              │
└────────────┼────────────────────────────────────────────────────┘
             │
             ▼
       ┌──────────────────────┐
       │   inbox 文件夹       │   ← Watch Folder（入口）
       │  \\server\drawings\  │
       │      inbox\          │
       └──────────┬───────────┘
                  │ watchdog 实时监听 + 文件稳定性检查
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OCR 服务（Docker on Windows + WSL2）          │
│                                                                 │
│  ① 文件稳定性检查（3 秒大小不变）                                  │
│  ② 移到 processing\ 目录（防重复触发）                            │
│  ③ 解析文件名 → 提取 drawing_no / revision（含兜底）              │
│  ④ 入 SQLite 持久化队列                                          │
│  ⑤ Worker 处理（单 GPU 顺序）                                    │
│       ├─ 矢量 PDF（可抽文字）→ N 路径 → 直接文字替换              │
│       └─ 位图/扫描 PDF      → O 路径 → VLM OCR + 文字替换       │
│  ⑥ 写日志表 process_log.db                                       │
│  ⑦ 成功 → 输出到 output 文件夹（文件名不变；同名加时间戳）         │
│  ⑧ 失败 → 移到 failed 文件夹 + 日志记 failed                     │
│                                                                 │
│  内部 API（供 dashboard 前端调用）：                              │
│    GET  /api/jobs/queue                                          │
│    GET  /api/jobs/{id}                                           │
│    GET  /api/logs                                                │
│    POST /api/jobs（手动补跑）                                    │
│    GET/PUT /api/settings                                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
       ┌──────────────────────┐
       │  output 文件夹       │   ← Watch Folder（出口）
       │  \\server\drawings\  │     PLM 监听此目录
       │      output\         │
       └──────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          PLM (Teamcenter)                        │
│                                                                 │
│  PLM 监听到新文件 → 按文件名挂回原图纸对象                          │
│              ↓                                                  │
│  工作流推进 → 状态 = 待审核                                        │
│              ↓                                                  │
│  审核员在 PLM 客户端打开"待审核工作区"                             │
│              ↓                                                  │
│        ┌───────────────┐                                        │
│        │  无问题  有问题  │                                        │
│        └───┬───────┬───┘                                        │
│            │       │                                            │
│   [PLM 点【通过】]   [打开 dashboard]                              │
│   [状态=已审核]      [→ 列表 → 点【打开编辑器】]                    │
│                          ↓                                      │
│                  manual-editor.exe 启动                          │
│                  (审核员手动改框/文字)                            │
│                          ↓                                      │
│                  覆盖 output 同名文件                             │
│                          ↓                                      │
│                  PLM 监听到 mtime 变化                            │
│                          ↓                                      │
│                  工作流推进 → 状态 = 已审核                        │
└─────────────────────────────────────────────────────────────────┘

旁路 — V2 Dashboard 前端（保留作为操作员入口）：
  - 队列查看（pending / running / done / failed）
  - 日志查询
  - 待手动审核列表（点击启动 manual-editor）
  - 系统设置（修改 watch folder 路径等运行时配置）
```

---

## 3. OCR 服务设计

### 3.1 部署形态：Docker on Windows + WSL2

- **运行环境**：Windows Server + WSL2 + Docker Desktop / Docker EE
- **容器内**：Linux 基础镜像 + Python + FastAPI + vLLM + PaddleOCR-VL + 业务代码
- **GPU 访问**：通过 NVIDIA Container Toolkit 直通宿主 GPU（消费级 RTX 40/50 系，显存 ≥ 12 GB）
- **模型权重**：不打入镜像，通过 `volumes:` 挂载（首次 U 盘交付，后续走客户内网更新服务器）

### 3.2 触发机制：Watch Folder

| 文件夹 | 用途 | 监听方 |
|---|---|---|
| `inbox\` | PLM 投递入口 | **OCR 服务监听** |
| `processing\` | 处理中文件暂存 | — |
| `output\` | 处理结果 | **PLM 监听** |
| `failed\` | 处理失败的文件 | 当前不监听（见 §10 待确认） |

**文件稳定性检查**：监听到新文件后，等 3 秒，确认文件大小未变化再入队（防止读到半截文件）。等待秒数在前端可配。

### 3.3 队列与处理

- **存储**：SQLite 单文件数据库（`queue.db`），位于持久化卷
- **Worker**：单线程顺序处理，独占 GPU（防 OOM）
- **数据流**：
  ```
  inbox 新文件 → 稳定性 OK → 移到 processing → 入队（status=pending）
                                                   ↓
                                            Worker 取出（status=running）
                                                   ↓
                                            判断走 N 或 O 路径
                                                   ↓
                                            处理完成 → 输出 + 写日志
                                                   ↓
                                            队列记 status=done
  ```

### 3.4 处理路径判断（N / O）

```python
if is_pdf(file_path) and has_extractable_text(file_path):
    result = vector_pdf_path(file_path)   # 矢量 PDF
    log_mode = "N"
else:
    result = vlm_ocr_path(file_path)       # VLM OCR（位图 / 扫描 PDF）
    log_mode = "O"
```

判断逻辑复用 V2 现有代码（[pdf_vector_handler.py](../modules/pdf_vector_handler.py) 已有矢量识别能力）。

### 3.5 输出文件命名规则

- **输入**：`YA116A226-1.tif`（PLM 投递时的原文件名）
- **正常输出**：`output\YA116A226-1.tif`（**文件名不变**）
- **output 已有同名文件时**：加时间戳后缀
  ```
  output\YA116A226-1.tif                         ← 历史输出
  output\YA116A226-1__20260528_143012.tif        ← 本次输出
  ```
- **manual_editor 修改后**：**覆盖** output 同名文件（不加时间戳，详见 §5.3）

### 3.6 日志表（`process_log.db`）

```sql
CREATE TABLE process_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  drawing_no      TEXT NOT NULL,              -- 图号；解析失败时 = 原文件名
  revision        TEXT,                       -- 版本；解析失败时 = "N/A"
  process_date    DATETIME NOT NULL,          -- 处理完成时间
  process_mode    TEXT NOT NULL,              -- "N"=矢量PDF / "O"=VLM OCR
  metadata_source TEXT NOT NULL,              -- "filename" / "fallback" / "plm_api"
  job_id          TEXT,                       -- 对应队列任务 ID
  source_file     TEXT NOT NULL,              -- 原始文件名
  result_path     TEXT,                       -- 输出文件路径
  status          TEXT NOT NULL,              -- "done" / "failed"
  error_msg       TEXT                        -- 失败时的错误信息
);

CREATE INDEX idx_drawing_no ON process_log(drawing_no);
CREATE INDEX idx_process_date ON process_log(process_date);
```

**重要约定**：
- 日志记录在 OCR 服务**自动处理完成时**写入
- manual_editor 的手动修改**不**新增日志条目（见 §10 待确认）
- `metadata_source` 字段为未来接 PLM SOA API 留扩展位

### 3.7 元数据来源（drawing_no / revision）

采用**分层兜底**策略，按以下优先级尝试：

| 层级 | 策略 | 状态 |
|---|---|---|
| L1 | 文件名解析（图号正则匹配 + 版本号识别） | 首期必做 |
| L2 | dashboard 提供"从 PLM 重新拉取"按钮（手动同步） | 远期可选 |
| L3 | OCR 处理时自动调用 PLM SOA API 查询 | 远期可选 |

**兜底**：所有层级均失败时，`drawing_no = 原文件名`，`revision = "N/A"`，其他字段照常写入，服务不报错。

### 3.8 配置管理

**两级配置**：

| 层级 | 位置 | 内容 | 何时修改 |
|---|---|---|---|
| 启动时必需 | `config.yaml` / `.env` | 端口、数据库路径、模型路径、JWT 种子 | 极少 |
| 运行时可调 | SQLite `settings` 表 | watch folder 路径、稳定性等待、API Key、重试次数 | 通过前端"系统设置"页随时改 |

**修改运行时配置无需重启服务**（监听器自动 reload）。

### 3.9 模型升级策略

| 阶段 | 频率 | 传输方式 |
|---|---|---|
| 首次部署 | 一次 | U 盘 / 客户提供的安装介质（最快） |
| 代码升级（高频） | 数周~数月 | Docker 镜像，从内网更新服务器 `docker pull` |
| 模型升级（低频） | 半年~一年 | rsync over SSH / 客户内网文件中转平台 |

**强推**：在客户内网部署一台轻量 VM 作为"更新服务器"（HTTP 服务器或 Harbor 私有镜像仓库），所有后续升级都从这里拉取。一次性配置，长期受益。

---

## 4. 前端 Dashboard

V2 现有的 Next.js dashboard **保留**，作为客户操作员的入口。需要新增/调整几个页面。

### 4.1 现有保留

- 任务队列视图
- 日志查看（增强）
- 文件上传/手动提交

### 4.2 新增页面

#### "待手动审核列表"
- 数据源：`process_log` 表 `status = done` 的记录，按时间倒序
- 每行：图号 / 版本 / 处理日期 / 处理方式（N/O）/ 【打开编辑器】按钮
- 点击按钮 → 通过自定义 URL 协议（`mitsubishi-editor://`）启动 `manual-editor.exe`，带文件路径参数

#### "系统设置"
- watch folder 三个路径（inbox / output / failed）
- 文件稳定性等待秒数
- 失败重试次数
- API Key（用于内部 API 鉴权）
- 模型路径（只读展示，需重启容器才能改）

#### "日志查询"
- 按图号、版本、日期范围、处理方式过滤
- 导出 CSV 功能

---

## 5. Manual Editor 集成

### 5.1 启动方式：路线 A（保留桌面版 + dashboard 列表点击）

```
dashboard "待手动审核列表"
     ↓ 审核员点【打开编辑器】
     ↓
浏览器触发自定义 URL 协议: mitsubishi-editor://open?file=YA116.tif
     ↓ Windows 调用注册的处理器
     ↓
manual-editor.exe 启动，带命令行参数:
  --tif-folder OUTPUT/vlmocr/
  --csv-folder OUTPUT/
  --output-folder OUTPUT/PDF_Replacement/
  --select YA116A226-1.tif
```

### 5.2 默认目录

启动后三个默认目录（用户配置，可在 GUI 内覆盖）：

| 用途 | 默认路径（示意，最终以客户配置为准） |
|---|---|
| 替换后 TIF 文件夹 | `OUTPUT/vlmocr/` |
| CSV 文件夹 | `OUTPUT/` |
| 保存输出文件夹 | `OUTPUT/PDF_Replacement/` |

> 文件夹名为示意，实际命名会修改。

### 5.3 输出覆盖规则

manual_editor 保存时：**覆盖** output 同名文件，**不加时间戳**。

理由：审核员对同一次审核的修订属于"同一张图的迭代"，不是新版本。最终 PLM 取到的就是最新修订结果。

### 5.4 日志写入：**不写**

manual_editor 修改完成后**不**写入 `process_log` 表，理由：
- 日志的语义是"OCR 服务自动处理时走了哪条路径（N/O）"
- 手动修改不属于"自动处理"，不应污染该语义
- PLM 状态由 PLM 工作流引擎自管，不依赖 OCR 日志（见 §6.2）

### 5.5 打包与分发

- **打包**：PyInstaller 打成单 `.exe`（约 150~250 MB，含 PySide6 + PIL）
- **分发**：随 OCR 服务一同交付，安装在审核员图形工作站
- **URL 协议注册**：安装时写入注册表
  ```
  HKEY_CLASSES_ROOT\mitsubishi-editor\shell\open\command
    @="\"C:\\Program Files\\Mitsubishi\\manual-editor.exe\" \"%1\""
  ```

### 5.6 UI 美化

记录：用户后续指挥执行 UI 美化任务。当前 UI 不主动改动。

---

## 6. Teamcenter 集成边界

### 6.1 文件流转（OCR ↔ PLM）

| 方向 | 介质 | 触发 |
|---|---|---|
| PLM → OCR | `inbox\` 文件夹 | PLM 工作流推送 |
| OCR → PLM | `output\` 文件夹 | PLM 监听 mtime |

**完全靠文件系统单向解耦。** OCR 服务不主动通知 PLM、不调用 PLM API。

### 6.2 状态推进（PLM 自管）

| 阶段 | 状态 | 触发条件 |
|---|---|---|
| 投递前 | 草稿 / 已 check-in | PLM 自身工作流 |
| OCR 处理 | 处理中（PLM 工作流自定义） | PLM 工作流推送到 inbox |
| 完成 | 待审核 | PLM 监听 output 出现新文件 |
| 审核通过 | 已审核 | 审核员在 PLM 点【通过】 OR output 文件被手动修改（mtime 变化） |

**关键**：OCR 服务**不修改 PLM 状态**。状态推进由 PLM 工作流引擎根据"文件出现"和"文件 mtime 变化"自动触发。具体工作流配置由 TC 顾问完成。

### 6.3 元数据获取

详见 §3.7。首期采用文件名解析 + 兜底，远期可接 PLM SOA API。

### 6.4 鉴权

- **当前方案**：OCR 服务对 PLM 完全无 API 调用，**不需要 PLM 凭证**
- **远期方案**：若启用 L2/L3 元数据同步，需要 TC 顾问配 SOA API 访问权限 + OCR 服务端配置凭证

内部 API（dashboard 用）通过 **API Key**（HTTP Header `Authorization: Bearer xxx`）鉴权，API Key 在前端"系统设置"页配置。

---

## 7. 接口规范（内部 API）

> 仅供前端 dashboard 调用，**不对 PLM 暴露**。

### 7.1 任务队列

```
GET  /api/jobs/queue
     ?status=pending|running|done|failed
     &limit=50&offset=0
     → 200 OK { items: [...], total: N }

GET  /api/jobs/{id}
     → 200 OK { id, source_file, status, created_at, finished_at, ... }

POST /api/jobs
     Content-Type: multipart/form-data
     file: <binary>
     drawing_no?: string
     revision?: string
     → 201 Created { id, status: "pending" }
```

### 7.2 日志查询

```
GET  /api/logs
     ?drawing_no=YA116A226-1
     &from=2026-05-01&to=2026-05-31
     &mode=N|O
     &limit=100&offset=0
     → 200 OK { items: [...], total: N }

GET  /api/logs/{drawing_no}
     → 200 OK { items: [所有版本的日志记录] }

GET  /api/logs/export?format=csv&...
     → 200 OK (CSV file download)
```

### 7.3 系统设置

```
GET  /api/settings
     → 200 OK { inbox_dir, output_dir, failed_dir, stability_seconds, ... }

PUT  /api/settings
     Body: { inbox_dir, output_dir, ... }
     → 200 OK
     副作用：watcher 重新加载配置（无需重启）
```

### 7.4 错误响应

```
4xx/5xx Body 统一格式:
{
  "error": {
    "code": "INVALID_FOLDER_PATH",
    "message": "human-readable description",
    "details": { ... }
  }
}
```

---

## 8. 数据库 Schema 完整版

### 8.1 `queue.db` — 队列表

```sql
CREATE TABLE job_queue (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  source       TEXT NOT NULL,          -- 'api' | 'watch_folder'
  source_file  TEXT NOT NULL,          -- 原始文件名
  file_path    TEXT NOT NULL,          -- processing\ 下的临时路径
  drawing_no   TEXT,                   -- 解析结果（可为 NULL，由解析层兜底）
  revision     TEXT,                   -- 同上
  status       TEXT NOT NULL,          -- 'pending' | 'running' | 'done' | 'failed'
  retry_count  INTEGER DEFAULT 0,
  created_at   DATETIME NOT NULL,
  started_at   DATETIME,
  finished_at  DATETIME,
  error_msg    TEXT,
  result_path  TEXT
);

CREATE INDEX idx_status ON job_queue(status);
CREATE INDEX idx_created_at ON job_queue(created_at);
```

### 8.2 `process_log.db` — 日志表

见 §3.6。

### 8.3 `settings` 表（与 queue.db 同库）

```sql
CREATE TABLE settings (
  key          TEXT PRIMARY KEY,
  value        TEXT NOT NULL,
  description  TEXT,
  updated_at   DATETIME NOT NULL,
  updated_by   TEXT
);

-- 初始记录（示意）:
INSERT INTO settings VALUES
  ('inbox_dir',          '\\server\drawings\inbox',      'PLM 投递目录',    NOW(), 'system'),
  ('output_dir',         '\\server\drawings\output',     'OCR 输出目录',    NOW(), 'system'),
  ('failed_dir',         '\\server\drawings\failed',     '失败文件归档',    NOW(), 'system'),
  ('stability_seconds',  '3',                            '文件稳定性等待',  NOW(), 'system'),
  ('max_retry',          '2',                            '失败重试次数',    NOW(), 'system'),
  ('api_key',            '<generated>',                  '内部 API 鉴权',   NOW(), 'system');
```

---

## 9. 部署与升级

### 9.1 首次部署清单

| 交付物 | 说明 |
|---|---|
| `ocr-service.tar` | Docker 镜像 tar 包（约 3~5 GB，不含模型） |
| `docker-compose.yml` | 启动配置 |
| `.env.example` | 启动必需配置模板 |
| `models/` | 模型权重（U 盘交付，约 5~15 GB） |
| `manual-editor.exe` | PyInstaller 打包，含 URL 协议注册 |
| `部署文档.pdf` | 客户 IT 用 |
| `操作手册.pdf` | 审核员用 |

### 9.2 客户 IT 部署步骤

1. 安装 Windows Server + WSL2 + Docker Desktop / Docker EE
2. 安装 NVIDIA Container Toolkit
3. 解压模型权重到 `D:\ocr\models\`
4. `docker load -i ocr-service.tar`
5. 配置 `.env`（数据库路径、端口）
6. `docker compose up -d`
7. 浏览器打开 `http://localhost:8080/`，进"系统设置"配置三个 watch folder 路径
8. TC 顾问配置 PLM 工作流（推送到 inbox + 监听 output）
9. 操作员工作站安装 `manual-editor.exe`

### 9.3 升级流程

**代码升级**（高频）：
```bash
# 客户内网更新服务器上
scp new-image.tar ocr-server:/tmp/

# 客户 IT 在 OCR 服务器上
docker load -i /tmp/new-image.tar
docker compose pull && docker compose up -d
```

**模型升级**（低频）：
```bash
# rsync 增量同步（支持断点续传）
rsync -avP --partial new-model/ ocr-server:/d/ocr/models/
# 重启服务
docker compose restart
```

---

## 10. 待客户/TC 顾问确认事项

| # | 问题 | 当前方案 | 影响范围 |
|---|---|---|---|
| 1 | manual_editor 修改是否要写入日志？ | 不写 | 日志表 schema |
| 2 | failed 文件夹是否要被 PLM 监听？ | 不监听，只看日志 | PLM 工作流配置 |
| 3 | 版本字段获取方式？是否可走 PLM SOA API？ | 兜底（N/A） | 元数据来源策略 |
| 4 | inbox / output / failed 三个 UNC 路径 | 部署时前端配置 | 客户 IT 提供具体路径 |

---

## 11. 后续待办（不阻塞首期上线）

| # | 待办 | 触发条件 |
|---|---|---|
| 1 | manual_editor UI 美化 | 用户指挥执行 |
| 2 | L2 元数据同步（dashboard 手动拉取按钮） | 客户答复 §10.3 后评估 |
| 3 | L3 元数据自动同步（每次处理调 PLM API） | 同上 |
| 4 | 失败邮件通知 | 客户答复 §10.2 后评估 |
| 5 | 内网更新服务器（Harbor / HTTP 制品库） | 部署稳定后建设 |

---

## 12. 工作量预估（参考）

| 阶段 | 主要任务 | 工作量 |
|---|---|---|
| 阶段 1 | 路径外部化（环境变量化） | 1~2 天 |
| 阶段 2 | SQLite 队列 + Worker + 单 GPU 顺序处理 | 3~5 天 |
| 阶段 3 | Watch folder + 稳定性检查 + 移动文件 | 2~3 天 |
| 阶段 4 | 日志表 + N/O 自动判定 + 元数据兜底解析 | 2~3 天 |
| 阶段 5 | 内部 API（鉴权 + 日志查询 + 设置） | 2~3 天 |
| 阶段 6 | 前端三个页面（设置 / 待审核列表 / 日志） | 4~6 天 |
| 阶段 7 | manual_editor 命令行参数 + URL 协议注册 | 1~2 天 |
| 阶段 8 | Docker 化 + 镜像构建 + compose 配置 | 3~5 天 |
| 阶段 9 | PyInstaller 打包 manual_editor + 安装器 | 2~3 天 |
| 阶段 10 | 部署文档 + 操作手册 + 联调 | 3~5 天 |
| **合计** | | **23~37 天** |

实际进度依赖客户/TC 顾问的响应速度。核心 OCR + 替换算法**全程不变**。

---

## 附录 A：关键术语

| 术语 | 含义 |
|---|---|
| PLM | Product Lifecycle Management，产品生命周期管理系统。客户用的是 Teamcenter |
| Teamcenter / TC | Siemens 的 PLM 产品，市场主流 |
| Watch Folder | "监听文件夹"，软件持续监视目录变化，新文件出现即触发处理 |
| Sidecar | 与主文件配对的元数据小文件（如 `xxx.tif.meta.json`） |
| SOA | Service Oriented Architecture，Teamcenter 对外的 Web 服务接口 |
| UNC 路径 | `\\server\share\path` 这种 Windows 网络共享路径 |
| N 路径 | 矢量 PDF 直接抽文字替换的处理方式 |
| O 路径 | 通过 VLM OCR 识别后再替换的处理方式 |
| 兜底 | fallback，解析失败时用一组保守的默认值 |

## 附录 B：变更记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| 2026-05-28 | v1.0 | 初稿，架构设计闭环 | Brady Huang |
