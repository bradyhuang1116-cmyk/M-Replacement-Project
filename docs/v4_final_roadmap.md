# 最终改造方案

> 推理用 vLLM（NodexelOCR 自打镜像）+ Nuitka 编译源码保护 + PLM Oracle 直连。

---

## 1. 最终架构

```
┌──────────────────────────────────────────────────────────────┐
│  客户机房 Windows + NVIDIA GPU                                  │
│                                                                │
│  【NodexelOCR 容器】(Docker，模型已封装在镜像内部)              │
│     paddleocr genai_server --model_dir /home/paddleocr/models/ │
│     提供 OCR 推理 (localhost:8080，内部组件，客户不直接访问)     │
│              ↑ HTTP                                             │
│  【我方应用服务】(宿主机 Python 进程，Windows Service)          │
│     ├─ 算法核心 = Nuitka 编译的 .pyd (源码不可见)               │
│     ├─ 管道/对接代码 = .py                                      │
│     └─ 由对方实现的 PLM Oracle 对接层驱动                       │
│              ↕                                                  │
│  连客户 Oracle (10.237.126.x) + 读写宿主盘 D:\PLM / D:\SMEC    │
└──────────────────────────────────────────────────────────────┘
```

**三层保护**：模型权重(COPY 进镜像) + 技术来源(镜像改名 NodexelOCR) + 核心算法(Nuitka .pyd)。

---

## 2. 我方工作（三个工作包）

### 工作包 1：自打 NodexelOCR 镜像（模型保护）

**目标**：把模型从"运行时挂载"改为"COPY 进镜像内部"，宿主机看不到模型，镜像名不暴露技术栈。

**已验证可行**（PoC 实测）：
- genai_server 支持 `--model_dir` 指向容器内置路径（不再需要旧的"挂载+cp 到 official_models"hack）
- 内置路径 `/home/paddleocr/models/PaddleOCR-VL-1.5` 起服务成功，`/v1/models` 的 root 显示内置路径
- 输出格式含 `<\|LOC_n\|>`，与现有 `parse_spotting_output` 完全兼容
- 容器内 `/app` 无写权限，须用 `/home/paddleocr/` 路径

**镜像构建**：见 `docker/Dockerfile.nodexel` + `docker/entrypoint.sh`（模型 COPY 进中性目录 nodexel-ocr，entrypoint 降日志级别）。

**构建与打包**：
```bash
docker build -f docker/Dockerfile.nodexel -t nodexel-ocr:v1 .
docker tag nodexel-ocr:v1 nodexelocr:v1
docker save nodexelocr:v1 | gzip > NodexelOCR.tar.gz   # ~10GB
```

**体积说明**：基础镜像 18.6GB（torch 6.8GB + SM120 算子 4.16GB，vLLM 固有成本，砍不掉）+ 模型 1.8GB ≈ 20GB，gzip 后传输 ~10GB。第一版不做瘦身，先 build 可跑版本。

<!--PART2-PLACEHOLDER-->

### 工作包 2：Nuitka 编译（源码保护）

**目标**：`modules/` 全部 + `config.py` 编译成机器码 `.pyd`，客户拿不到可读源码。

| 编译范围 | 内容 | 理由 |
|---|---|---|
| ✅ 编译 | modules/ 全部 13 个（算法核心 + 队列/worker/批处理等业务逻辑） | 算法 + 业务逻辑均保护 |
| ✅ 编译 | config.py | 含编号正则/纠错表/列头锚点等领域知识 |
| ❌ 保留 .py | api/（路由薄转发层） | 无算法，且 FastAPI 动态加载编译易坏 |
| ❌ 保留 .py | manual_editor/（PySide6 GUI） | 编译易坏，独立工具 |
| ❌ 保留 .py | __init__.py、start_v2.py | 包标识 / 入口 |

- 编译后 `.pyd` 可被正常 import，架构零改动（config.pyd 仍可 `from config import`）
- 编译在我方打包机做（`scripts/compile_core.sh`），客户只拿 `.pyd`
- 已验证：14 个模块全编译 + 全 .pyd 替换后 smoke 测试 PASS，链路正常

### 工作包 3：Windows Service 封装 + 离线打包

- 用 NSSM 把应用服务注册成 Windows Service（开机自启），FastAPI 加 `/health` 端点
- 离线依赖打包：`pip download` 所有依赖（含 oracledb thin）成 wheel 包
- 基础环境离线安装包：Docker Desktop / NVIDIA 驱动+Toolkit / Miniconda / nssm.exe

---

## 3. PLM 数据库对接（**对方实现**，我方仅提供实现计划）

> 此部分由客户/对方团队实现，我方不亲自写代码。本节是给对方的实现参考。

### 3.1 对接契约（来自客户文档）

```
1. 读 config/server.properties 拿本机 IP，决定连哪个 Oracle：
   - 10.237.126.127 → database dbg13, schema meseplm（开发）
   - 10.237.126.77  → database rtn13, schema meseplm（生产）
2. 轮询任务表：
   SELECT DOCNUMBER, TH, BBH, FILENAME, WORK_SEQ
   FROM R_V_TD_FILEPATH WHERE ISPROCESS='0' ORDER BY UPD_TIMESTAMP ASC
3. 查原图表定位：
   SELECT LOCATION FROM SIPM197
   WHERE DEL=0 AND WKAID<>'3' AND TH=? AND BBH=? AND FNAME=? AND DOCNUMBER=? AND WORK_SEQ=?
4. 拼绝对路径：IP=.127→D:\PLM719\filedata；IP=.77→D:\PLM\filedata，+ LOCATION
5. 读物理盘原图 → 调我方处理 → 输出
6. 归档：D:\SMEC\<WORK_SEQ>\<TH>-<BBH>\<FILENAME>（先递归建目录）
7. 事务回写双表（任一失败回滚）：
   UPDATE SIPM197 SET OCR=?, PTIME=SYSDATE WHERE ...
   UPDATE R_V_TD_FILEPATH SET ISPROCESS='1', OCR=?, UPD_USER='LMT', UPD_TIMESTAMP=SYSDATE WHERE ... AND ISPROCESS='0'
```

### 3.2 我方提供的接口（对方调用点）

对方的 Oracle 对接层，拿到原图绝对路径后，调用我方一个函数即可：
```python
result = process_single_file(file_path, output_dir)
# 返回 {status, method, total(替换数), output_path}
```
- `method == "ocr"` → O 标记（使用了 OCR 识别，扫描件路径）
- `method == "vector"` → N 标记（未使用 OCR，矢量 PDF 直接处理）
- 对方据此回写 OCR 字段（定义对齐客户 Word 文档：O=是/用OCR，N=否/未用OCR）

### 3.3 实现要点（给对方的建议）

| 要点 | 建议 |
|---|---|
| Oracle 驱动 | `oracledb` thin 模式（纯 Python，无需 Instant Client），离线只多一个 wheel |
| O/N 映射 | `ocr_flag = "O" if method=="ocr" else "N"`（O=用OCR，N=未用OCR，对齐 Word 文档） |
| 事务 | 双表更新用一个事务包裹，任一失败 rollback |
| 防重复消费 | `SELECT ... FOR UPDATE SKIP LOCKED` 或单实例串行 |
| 失败处理 | 路径不存在/定位失败 → 记日志告警，不跳过 |
| 待确认项 | Oracle 版本（≥12c 则 thin 够用）；FILENAME 后缀(.PDF)与实际输出格式不一致时归档命名以哪个为准 |

### 3.4 联调依赖

- 测试库连接（10.237.126.127）由客户提供后方可联调
- 联调前对方可用本地 mock（SQLite 模拟双表）先跑通逻辑

---

## 4. 最终交付物（完整改造后）

```
NodexelOCR_Delivery/
├── NodexelOCR.tar.gz              推理镜像(模型已封装) ~10GB
├── app/                           我方应用服务
│   ├── modules/*.pyd              🔒 编译后的算法核心
│   ├── modules/*.py               管道代码 + 对方的Oracle对接层
│   ├── api/ config.py fonts/
│   ├── dashboard/(可选) manual_editor/
├── offline/                       离线依赖与环境
│   ├── offline_wheels/            Python依赖(含oracledb)
│   ├── DockerDesktop / NVIDIA驱动+Toolkit / Miniconda / nssm.exe
├── config/
│   ├── .env.example               客户填:Oracle连接/D:\SMEC路径
│   └── server.properties.example  客户填:本机IP
└── docs/  部署手册 + 操作手册
```

### 交付方式
- **客户内网隔离 → 移动硬盘/U盘物理拷贝 + 现场部署**（约 12GB）
- 不走公网远程拷（内网无公网，且数据合规禁止）
- 客户 IT 部署：装环境 → `docker load NodexelOCR.tar.gz` → 离线装依赖 → 填配置 → nssm 注册服务 → 启动

### 零泄露/零硬编码保证
- 模型在镜像内、算法是 .pyd、镜像名 NodexelOCR → 无源码/模型泄露
- 镜像内置路径固定；应用路径全走 .env/server.properties → 客户机路径变化不影响

---

## 5. 执行顺序与状态

| # | 工作包 | 负责 | 状态 |
|---|---|---|---|
| 1 | 自打 NodexelOCR 镜像 | 我方 | 地基已验证，待 build |
| 2 | Nuitka 编译算法 | 我方 | 待做 |
| 3 | Windows Service + 离线打包 | 我方 | 待做 |
| 4 | PLM Oracle 对接 | **对方** | 我方已提供实现计划(§3) |
| 5 | 联调 | 双方 | 待客户提供测试库 |

**顺序约束**：编译(2) 在 Service 封装(3) 之前。镜像(1) 可独立先做。

