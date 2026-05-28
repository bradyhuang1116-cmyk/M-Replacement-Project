# 三菱电机图纸 Y 编号批量替换系统

> 日文工程图纸自动识别 + Y 编号批量改写 + PLM 集成 OCR 服务
> 客户：**三菱电机（日本）** · 装备制造行业
> 当前状态：**生产可用 + 服务化改造进行中（§12 Phase 1–3 已完成，Phase 4–10 待做）**

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
- Miniconda + conda env `mitsubishi`（Python 3.10+）
- Docker Desktop（启 PaddleOCR-VL 推理容器用）
- Node.js（前端开发用）

### 三步启动

```bash
# 1. 启动 vLLM 推理容器（等模型加载约 3 分钟，端口 8080）
.\start_vllm_server.bat

# 2. 启动后端 API（端口 8000）
.\start_v2.bat

# 3. 启动前端（端口 3000）
cd dashboard && npm install && npm run dev
```

完整部署步骤见 [docs/legacy/客户端部署指南.md](docs/legacy/客户端部署指南.md)（注：该文档为 V1 时期产物，已 deprecated，Phase 10 会重写）。

---

## 文档导航

| 文档 | 给谁看 | 内容 |
|---|---|---|
| **[docs/handover.md](docs/handover.md)** | 接手服务化改造的工程师 | **必读**。当前进度、文件用途、未完成任务、不要碰的边界 |
| **[docs/plm_integration_design.md](docs/plm_integration_design.md)** | 架构决策方 | 整体架构、接口规范、数据库 schema、待客户确认事项 |
| [CLAUDE.md](CLAUDE.md) | AI 协作场景 | LLM 协作风格规范（人也能看） |
| [docs/legacy/客户端部署指南.md](docs/legacy/客户端部署指南.md) | 历史参考 | ⚠️ V1 部署清单（已 deprecated，待 Phase 10 重写） |
| [docs/legacy/部署文档.md](docs/legacy/部署文档.md) | 历史参考 | ⚠️ V1 部署说明（同上） |
| [dashboard/README.md](dashboard/README.md) | 前端 | Next.js 项目说明 |
| [manual_editor/README.md](manual_editor/README.md) | 桌面工具 | PySide6 编辑器说明（独立产品线） |

---

## 仓库结构（精简）

```
.
├── config.py                     全局配置入口（所有 env 读取在这）
├── start_v2.py / .bat / .vbs     后端启动脚本
├── start_vllm_server.bat         单独启 vLLM 容器
├── vllm_config.yaml              vLLM 推理超参
├── requirements_local.txt        host Python 依赖
├── .env.example                  所有可调 env 示例 + 说明
│
├── api/                          FastAPI 服务端
│   ├── main.py                   app 入口 + CORS + 路由注册
│   └── routes/                   health / jobs / gpu / folders / system / drawings
│
├── modules/                      业务逻辑
│   ├── batch_processor.py        process_single_file() 主入口
│   ├── region_detector.py        4 类框检测核心算法（不要碰）
│   ├── text_replacer.py          OCR + 字形替换（不要碰）
│   ├── factory_note_pixel.py     工厂注意区检测（不要碰）
│   ├── pdf_vector_handler.py     矢量 PDF 路径
│   ├── vlm_ocr_engine.py         PaddleOCR-VL HTTP 客户端
│   ├── docker_manager.py         vLLM 容器启停 + 健康检查
│   ├── job_queue.py              SQLite 队列（§12 Phase 2）
│   ├── worker.py                 单线程 Worker（§12 Phase 2）
│   ├── watch_folder.py           inbox 轮询监听（§12 Phase 3）
│   └── filename_parser.py        文件名→drawing_no/rev 解析（§12 Phase 3）
│
├── dashboard/                    Next.js 前端（工程师 GUI）
├── manual_editor/                PySide6 桌面工具（独立产品线，不在服务化改造范围）
├── docs/                         设计与交接文档
├── data/                         队列 DB + Worker 产物 + watch 目录（git 忽略内容）
├── models/                       PaddleOCR-VL 模型权重（git 忽略）
├── fonts/                        中日字体（替换字形用）
│
├── test_queue_worker_smoke.py    Phase 2 烟雾测试
└── test_watch_folder_smoke.py    Phase 3 端到端烟雾测试
```

---

## 当前进度（§12 PLM 集成改造）

| Phase | 内容 | 状态 |
|---|---|---|
| 1 | 路径外部化（环境变量化 + `.env.example`） | ✅ 完成（未提交） |
| 2 | SQLite 队列 + 单线程 Worker | ✅ 完成（未提交） |
| 3 | Watch Folder + 文件名解析 + 文件路由 | ✅ 完成（未提交） |
| 4 | 处理日志表 + N/O 自动判定 + 元数据兜底 | ⏳ 待做（推荐起点） |
| 5 | 内部 API（鉴权 + 日志查询 + 设置接口） | ⏳ 待做 |
| 6 | 前端三页（设置 / 待审核列表 / 日志） | ⏳ 待做 |
| 7 | manual_editor CLI 参数 + URL 协议注册 | ⏳ 待做 |
| ~~8~~ | ~~Docker 化（自打镜像 + compose）~~ | ❌ 已排除（详见 handover §6.7） |
| 9 | PyInstaller 打包 + Windows 安装器 | ⏳ 待做 |
| 10 | 部署文档 + 操作手册 + 联调 | ⏳ 待做 |

**详情**：[docs/handover.md](docs/handover.md) §5（已完成）+ §6（未完成）

---

## 验证当前状态

跑两个 smoke 测试，验证 Phase 2/3 仍然工作：

```bash
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_queue_worker_smoke.py
"C:/Users/Brady Huang/miniconda3/envs/mitsubishi/python.exe" test_watch_folder_smoke.py
```

两个都应该输出 `RESULT: PASS`。

---

## 关键技术约束（不要踩坑）

| 约束 | 说明 |
|---|---|
| 不要重构 `region_detector.py` / `text_replacer.py` / `factory_note_pixel.py` | 核心算法 6000+ 行，已稳定 |
| 不要改 `H_FACTORS` | 已锁定 `[8]`，`h=15` 弃用 |
| 不要启用 `scan_all_tables` | 已确认不做多表扫描 |
| 不要用字符比例换算 | 客户明确反对 |
| 不要动 `manual_editor/` 内部 | 独立产品线，只能给它加 CLI 入口 |
| 用完整 Python 路径，不用 `conda run` | `C:\Users\Brady Huang\miniconda3\envs\mitsubishi\python.exe` |

完整清单见 [docs/handover.md](docs/handover.md) §7.3。

---

## 联系

- **前期负责**：Brady Huang
- **仓库**：https://github.com/bradyhuang1116-cmyk/Mitsubishi-Electric-Drawing-Replacement-Project
- **conda env**：`mitsubishi`

接手后续改造请先读 [docs/handover.md](docs/handover.md) §7.1（上手第一周路线）。
