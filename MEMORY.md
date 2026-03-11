# 三菱图纸编号批量替换系统 — 本地测试版（含越界修复）

## 项目概述
扫描三菱电机图纸（TIF/PDF/JPG），自动识别编号（如 YA026D941），批量替换为新编号（如 HYA026D941）。本分支包含 Docker 部署的全部功能 + 替换越界修复。

## 架构总览

```
┌─────────────────────────────────────────────┐
│              Gradio Web 界面 (7860)          │
│  上传文件 → 选择前缀 → 处理 → 预览/下载     │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│            batch_processor.py                │
│  遍历文件 → 调用检测 → 调用替换 → 保存输出   │
└────────┬───────────────────────┬────────────┘
         │                       │
┌────────▼────────┐   ┌─────────▼───────────┐
│ region_detector  │   │  text_replacer      │
│ 动态区域检测     │   │  OCR + 像素级替换    │
│ 五区域定位       │   │  + 越界修复         │
└────────┬────────┘   └─────────┬───────────┘
         │                       │
┌────────▼───────────────────────▼───────────┐
│          PaddleOCR v5 (CPU)                 │
│  英文模型(en) + 中文模型(ch)                │
└─────────────────────────────────────────────┘
```

## 五区域检测

| 区域 | 标识 | 位置 | 内容 |
|------|------|------|------|
| 红框 | material_code_column | 表格中部 | MATERIAL CODE 数据列 |
| 绿框 | bottom_right_number | 右下角 | 编号栏（主编号） |
| 橙框 | top_left_number | 左上角 | DWG NO 右侧编号 |
| 紫框 | bottom_left_number | 左下角 | 竖排编号（旋转处理） |
| 蓝框 | annotations | 散落标注 | 图面散落编号 |

## 替换流程
1. 加载图纸 → 统一转为 numpy RGB（file_ingestion.py）
2. 动态检测五区域 → 返回 BBox + metadata（region_detector.py）
3. 五方投票校验（绿、橙、红、文件名、紫）→ 确定正确编号
4. 各区域独立替换：白色覆盖 → 渲染新文字 → 贴回（text_replacer.py）
5. 保存输出 + 验证后 OCR 确认

## 越界修复（本分支新增）

### 问题
替换后白色填充超出单元格边界，覆盖网格线和相邻内容。

### 解决方案

#### 1. 新增 `_find_cell_column_bounds()` 函数
- 在 ROI 中用形态学检测竖线（OTSU + 竖线kernel）
- 找到文字中心两侧最近的竖线作为列边界
- 内缩 margin=3px 避开线条本身
- 安全检查：检测宽度过窄时回退到 OCR bbox

#### 2. 红框/蓝框替换循环改进
- 网格模式：用 `_find_cell_column_bounds()` 确定 safe_w（列宽）
- 非网格模式：基于 OCR bbox + 适度扩展（每字符估算宽度）
- 填充/渲染统一内缩 `FILL_MARGIN=2px`

#### 3. 绿框/橙框内缩 `M=3px`
- 填充区域四边各内缩3像素，避开检测到的框线

#### 4. 紫框内缩 `PM=2px`
- 旋转后的替换区域四边内缩2像素

## 参数化前缀替换（来自 docker-deploy）
- `config.py` 中 `DEFAULT_PREFIXES = ["Y"]`，`NEW_PREFIX = "H"`
- `make_pattern(prefixes)` 动态生成正则
- Web 界面 A-Z CheckboxGroup 选择替换前缀
- 替换规则：`"H" + 原文`（如 YA026D941 → HYA026D941）

## 本地运行
```bash
# 创建 Python 3.12 虚拟环境
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements_local.txt

# Web 界面
python web_app.py
# 访问 http://localhost:7860

# CLI
python main.py single TIF_Undo/file.tif -o output/
python main.py batch TIF_Undo/ -o output/
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `web_app.py` | Gradio Web 界面入口 |
| `main.py` | CLI 入口（detect/single/batch） |
| `config.py` | 全局配置、正则、前缀 |
| `modules/region_detector.py` | 五区域动态检测 |
| `modules/text_replacer.py` | OCR + 像素级替换 + 越界修复 |
| `modules/batch_processor.py` | 批量处理协调 |
| `modules/file_ingestion.py` | 文件格式统一加载 |
| `modules/verification.py` | HTML 验证报告 |
| `fonts/SixCaps-Regular.ttf` | 替换用字体 |
| `requirements_local.txt` | 本地依赖（无中文注释，避免 GBK 编码错误） |

## 依赖
- Python 3.12（注意：3.14 不兼容 paddlepaddle）
- PaddleOCR v5, PaddlePaddle 3.0.0 (CPU)
- OpenCV, Pillow, PyMuPDF, numpy
- Gradio >= 4.40.0

## 与 docker-deploy 分支的关系
- `local-test` 包含 `docker-deploy` 的全部改动
- 额外添加：`_find_cell_column_bounds()` + 替换循环越界修复 + 绿/橙/紫框内缩
- 额外添加：`requirements_local.txt`（Windows 本地安装用）
