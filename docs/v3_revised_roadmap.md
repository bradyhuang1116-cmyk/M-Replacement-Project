# V3 改造方案修订 — 数据库直连 + 原生推理 + 源码保护

> ⚠️ **已废弃（2026-06-09）**：本文档的"工作线 B：去 vLLM/Docker 改原生推理"经 PoC 验证**否决**。
> 最新方案见 **[v4_final_roadmap.md](v4_final_roadmap.md)**（保留 vLLM + 自打镜像 + Nuitka 编译）。
> 本文件仅作历史参考保留。

> **状态**：方案修订中，PoC 验证进行中（2026-06-08）
> **作者**：Brady Huang
> **关系**：本文档取代 [plm_integration_design.md](plm_integration_design.md) 的 §2/§3/§6/§9（文件夹监听 + Docker 镜像部分）。其余章节（前端、manual editor、术语）仍有效。
> **触发**：客户两份交互文档（《光栅图处理软件与PLM系统的交互机制.docx》《PLM图纸OCR处理服务交互逻辑说明文档.docx》）+ Brady 的源码保护与去 Docker 诉求，推翻了早期预估。

---

## 0. 为什么要修订（早期预估 vs 客户实情）

| 维度 | 早期预估（已作废） | 客户实情 / 新诉求 |
|---|---|---|
| 任务来源 | 监听 inbox 共享文件夹 | **轮询 Oracle 视图 `R_V_TD_FILEPATH`** |
| 原图定位 | 文件名约定 | **查 `SIPM197` 表拿 LOCATION 相对路径** |
| 结果去向 | output 共享文件夹 | **归档到 `D:\SMEC\...` + 回写双表** |
| 服务形态 | Docker 镜像 | **Windows Service**（客户文档明确） |
| 推理引擎 | vLLM + Docker 容器 | **原生 Python 推理**（去 Docker，Brady 诉求） |
| 源码 | 明文 .py | **核心算法 Nuitka 编译**（Brady 诉求） |

---

## 1. 新架构总览

```
┌──────────────────────────────────────────────────────────────┐
│  客户机房一台 Windows + GPU 机器                                │
│                                                                │
│  【Windows Service（NSSM 注册，开机自启 + /health）】           │
│     ├─ 读 config/server.properties → 本机 IP → 选 Oracle 库     │
│     ├─ 轮询 R_V_TD_FILEPATH WHERE ISPROCESS='0'                 │
│     ├─ 查 SIPM197 → LOCATION → 拼绝对路径读原图                 │
│     ├─ 图面处理：                                               │
│     │     ├─ 矢量 PDF → 直接替换（O 标记）                      │
│     │     └─ 扫描件 → 原生 PaddleOCR-VL 推理 + 替换（N 标记）    │
│     │            （进程内加载模型，无 Docker，无 HTTP）          │
│     ├─ 归档 D:\SMEC\<WORK_SEQ>\<TH>-<BBH>\<FILENAME>            │
│     └─ 事务回写 SIPM197 + R_V_TD_FILEPATH                       │
│                                                                │
│  算法核心模块 = 已 Nuitka 编译的 .pyd（客户看不到源码）          │
│  连客户内网 Oracle（10.237.126.x）+ 读写宿主盘 D:\PLM / D:\SMEC │
└──────────────────────────────────────────────────────────────┘
```

**与旧架构最大差异**：没有 Docker 容器，没有 vLLM HTTP 服务，没有共享文件夹监听。模型在 Python 进程内直接加载推理。

---

## 2. 三条工作线（互相独立，可并行）

### 工作线 A — PLM 数据库直连对接（Phase 4 重做，工作量最大）
新增 Oracle 对接层：配置/IP 识别 → 拉任务 → 定位原图 → O/N 映射 → 归档 → 双表事务回写。
详见 §3。

### 工作线 B — 去 vLLM/Docker，改原生推理（本分支 `feat/native-paddle-inference`）
把 `vlm_ocr_engine.py` 内部从"HTTP 调 vLLM"改成"进程内原生加载 PaddleOCR-VL"。
详见 §4。

### 工作线 C — 源码保护 + Windows Service 封装（交付前最后一步）
Nuitka 编译 4 个算法核心 → NSSM 注册成 Windows Service。
详见 §5。

**顺序约束**：C 必须最后做（编译 → 封装）。A、B 可并行开发。

---

## 3. 工作线 A：PLM 数据库直连对接

### 3.1 对接契约（来自客户文档，逐字落地）

```
1. 读 config/server.properties 拿本机 IP
   - 10.237.126.127 → database dbg13, schema meseplm（开发/预发布）
   - 10.237.126.77  → database rtn13, schema meseplm（生产）
2. 轮询任务表：
   SELECT DOCNUMBER, TH, BBH, FILENAME, WORK_SEQ
   FROM R_V_TD_FILEPATH WHERE ISPROCESS='0' ORDER BY UPD_TIMESTAMP ASC
3. 查原图表定位：
   SELECT LOCATION FROM SIPM197
   WHERE DEL=0 AND WKAID<>'3' AND TH=? AND BBH=? AND FNAME=? AND DOCNUMBER=? AND WORK_SEQ=?
4. 拼绝对路径：
   IP=.127 → D:\PLM719\filedata + LOCATION
   IP=.77  → D:\PLM\filedata   + LOCATION
5. 读物理盘原图 → 图面处理
6. 归档：D:\SMEC\<WORK_SEQ>\<TH>-<BBH>\<FILENAME>（先递归建目录）
7. 事务回写双表（任一失败回滚）：
   UPDATE SIPM197 SET OCR=?, PTIME=SYSDATE WHERE ...（O/N）
   UPDATE R_V_TD_FILEPATH SET ISPROCESS='1', OCR=?, UPD_USER='LMT', UPD_TIMESTAMP=SYSDATE WHERE ... AND ISPROCESS='0'
```

### 3.2 O/N 映射（已确认语义）

O/N 只是**日志标记**，对识别/替换逻辑无影响：

| 客户标记 | 含义 | 对应我方处理路径 | 代码依据 |
|---|---|---|---|
| **O** | 矢量 PDF 直接处理，未经 OCR | `method=="vector"` | batch_processor.py:68 |
| **N** | 经 OCR 识别的识别/替换 | `method=="ocr"` | batch_processor.py:180 |

映射就一句：`ocr_flag = "O" if method == "vector" else "N"`，挂在回写环节。

### 3.3 Oracle 驱动

用 `oracledb` **thin 模式**（纯 Python，零客户端库安装）。离线部署只多一个 wheel，不需要 Instant Client（前提：Oracle ≥ 12c，待联调确认版本）。

### 3.4 两阶段开发（绕开"暂无测试库"）

- **阶段一（现在可做，不依赖真库）**：mock 一个本地 SQLite 模拟双表 + 假数据，跑通"拉任务→查路径→处理→O/N→归档→回写"整链路。归档模块、O/N 映射可立即实现。
- **阶段二（拿到测试库后）**：数据源从 mock 切到真 `oracledb`，只动连接层，业务逻辑不重写。

### 3.5 复用 / 作废清单

| 处置 | 模块 | 说明 |
|---|---|---|
| ✅ 复用 | job_queue.py / worker.py | 作为内部处理队列（从 Oracle 拉一批 → 内部排队 → 单 GPU 串行） |
| ✅ 复用 | config.py | 扩展读 server.properties |
| 🟡 降级 | filename_parser.py | 图号/版本改由数据库 TH/BBH 提供，文件名解析仅兜底 |
| 🔴 作废 | watch_folder.py | 文件夹监听用不上 |

<!--SECTION-4-PLACEHOLDER-->

## 4. 工作线 B：去 vLLM/Docker，改原生推理

### 4.1 改造边界（冻结接口 = `VlmOcrEngine.predict()`）

唯一接缝是 `vlm_ocr_engine.py` 的 `predict()` 返回结构：
```python
[{"dt_polys": [[[x,y]×4], ...], "rec_texts": [str, ...], "rec_scores": [float, ...]}]
```
**保住这个结构，下游 6 个调用点（region_detector / factory_note_pixel / text_replacer）零改动。**

| 处置 | 文件/范围 |
|---|---|
| 🔧 改内部（保 API） | vlm_ocr_engine.py:60-201（HTTP → 进程内原生加载推理） |
| 🔧 改 | worker.py:111-116,228-233（`_ensure_vlm_ready` → 原生模型 warmup） |
| 🔧 改 | api/routes/jobs.py:90-101（Phase 1 启动逻辑） |
| 🔴 删 | docker_manager.py（整个文件）+ 4 处 import |
| 🔴 删 | start_vllm_server.bat、vllm_config.yaml |
| 🔴 删 | api/routes/system.py:20-31,100-103（_kill_docker_desktop / _shutdown_wsl） |
| 🔴 删 | config.py DOCKER_* / VLLM_BASE_URL / VLLM_MODEL_NAME；.env.example 对应块 |
| ⛔ 不碰 | 返回结构、两个 _parse_ocr_results、v5/PP-DocLayout 原生代码、所有 engine="vlm" 调用点 |

### 4.2 原生推理方式（官方支持，二选一）

- **路径 B（transformers 直载，PoC 验证用）**：`AutoModelForImageTextToText.from_pretrained(MODEL_DIR)` + `processor.apply_chat_template`，prompt 用 `"Spotting:"` / `"OCR:"`。输出**保留 `<|LOC_n|>` 坐标 token**（已确认 added_tokens.json 含 1001 个），现有 polygon 解析可直接复用。
- 路径 A（`from paddleocr import PaddleOCRVL`）：整页解析用，本项目按元素 spotting 为主，优先路径 B。

许可证 Apache 2.0，商用/改名分发无碍（保留 LICENSE 声明）。

### 4.3 两个硬约束（PoC 必须先验证，否则 B 路线不成立）

1. **GPU 共存**：原生 VL + v5 + PP-DocLayoutV3 同处一个进程抢 12GB 显存。去 vLLM 反而利好（vLLM 默认抢 90% 显存），但确切占用必须实测 `torch.cuda.max_memory_allocated()`。
2. **性能**：文档无可引用数字。vLLM 优势在高并发吞吐；本项目单 GPU 逐张处理，原生损失**可能可接受，必须实测单张延迟**对比。

### 4.4 PoC 验证（脚本：scripts/poc/poc_native_inference.py）

环境隔离：**mitsubishi_poc**（克隆自 mitsubishi），绝不碰 mitsubishi。5 个命门：
1. 新环境导包 + GPU 可见
2. v5 + PP-DocLayout 仍可初始化（共存）
3. **（最关键）原生 VL 推理跑通 + 输出含 `<|LOC_n|>`**
4. 单张 spotting 延迟（对比 vLLM）
5. 显存峰值占用（确认 12GB 三模型够用）

**PoC 全 PASS → 正式改造；某项失败 → 退回保留 Docker 的 A 方案（无界面 Docker 引擎 + 关容器日志）。**

---

## 5. 工作线 C：源码保护 + Windows Service

### 5.1 Nuitka 编译核心算法

编译对象（4 个护城河模块）→ `.pyd` 机器码，客户看不到源码：
- region_detector.py / text_replacer.py / factory_note_pixel.py / pdf_vector_handler.py
- 其余管道代码（队列/对接层/API/config）保持 .py（泄露不致命）
- 编译后 .pyd 可被正常 import，架构零改动

### 5.2 模型/镜像名遮掩（务实版，挡随手查，非防刻意逆向）

- 代码/配置里模型名 → 中性别名 + 编译后藏进二进制
- 去 Docker 后，`docker images` 那个泄露点自然消失（工作线 B 的附带收益）
- 模型目录可改中性名（同步改加载路径）；删 README/LICENSE 纯说明文件
- ⚠️ 接受残留：模型 config.json 里架构类名删不掉（删了加载不了）

### 5.3 NSSM 封装 Windows Service

- NSSM 把 `python.exe start_v2.py` 注册成服务（零代码改动，自动重启 + 日志重定向）
- FastAPI 加 `/health` 路由（满足客户文档第九条）
- ⚠️ 封装只改"怎么启动"，不改"代码什么形式"——源码保护靠 §5.1 编译，不靠封装

---

## 6. 部署与交付（去 Docker 后简化版）

### 6.1 交付物清单（VPN 整包传，客户零联网）
| 交付物 | 来源 | 大小 |
|---|---|---|
| 服务代码（算法已编译 .pyd） | 打包机 Nuitka 编译 | 小 |
| 模型权重 PaddleOCR-VL-1.5/ | U盘单独搬 | 1.8GB |
| Python 依赖 wheel（含 oracledb、transformers） | pip download | 数百 MB |
| nssm.exe | 单文件 | 极小 |
| 基础环境安装包（Python / NVIDIA 驱动） | 官网离线版 | — |

**去 Docker 后不再需要**：Docker Desktop、vLLM 镜像 tar、WSL2、NVIDIA Container Toolkit。部署大幅简化。

### 6.2 模型交互
模型不打包进代码，作为独立资产 U盘搬运，放宿主机目录，`.env`/config 路径指向，**进程内直接加载**（不再 Docker 挂载）。仅扫描件路径（N）用模型；矢量路径（O）不碰模型。

---

## 7. 待客户确认（不阻塞开发）
1. Oracle 版本号（≥12c 则 oracledb thin 模式确定够用）
2. FILENAME 字段后缀（.PDF）与实际输出格式（可能 TIF）不一致时归档命名以哪个为准
3. 测试库连接（10.237.126.127）何时可提供

---

## 8. 风险与回退
- **B 路线性能不达标** → 退回保留 vLLM/Docker，只做遮掩（无界面 Docker 引擎 + `--log-driver none`）
- **paddlepaddle 升级影响 v5** → 当前已是 3.3.0（≥3.2.1 满足），PoC CHECK-2 专门验证 v5/layout 共存
- **不回退 local-test** → B 路线全程在 feat/native-paddle-inference 分支 + mitsubishi_poc 环境，双重隔离
