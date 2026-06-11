# 区域检测流水线（红 / 绿 / 橙框）

来源代码：[modules/region_detector.py](../modules/region_detector.py)
入口函数：`detect_all_regions(image, region_config, prefixes)`（[L2891](../modules/region_detector.py#L2891)）

本文档**不包含** Phase C 工厂注意（factory_note）部分。

---

## 方案总览

### 为什么是 "VLM + v5" 混合

本项目的 OCR 不是单引擎，而是 **PaddleOCR v5（印刷字定位）+ VLM（PaddleOCR-VL-1.5，手写/退化字符识别）** 混合。每种引擎只用在自己擅长的子任务上。

| 子任务 | 难点 | 引擎 | 理由 |
|---|---|---|---|
| 红框：找 MATERIAL CODE 表头关键词 | 印刷英文 / 中文，需高准 | **PaddleOCR v5** | 印刷字识别稳；可英 + 中两次 OCR 合并结果 |
| 红框：列内容验证 | 印刷英文 / 中文 | **PaddleOCR v5** | 同上 |
| 红框：定位列边界 | 依赖竖线 / 横线，**不需 OCR** | OpenCV 形态学 | 比 OCR 准且快几个数量级 |
| 绿框：右下角编号 | 手写 / 半印刷、可能截断、可能跨格 | **VLM** | VLM 对手写鲁棒；可处理整段长文本不被空格切碎 |
| 橙框：左上角图号 | 常与 `DWG NO`/`図番` 写在同一格 | **VLM** | VLM 输出多边形 + 完整文本，便于"合体切割" |

**关键约定**：

- 红框检测**不依赖** Y 编号 OCR——它只锚定列位置；列里的具体编号文字在后续 `text_replacer` 阶段才读
- 绿/橙框都要**先识别真实编号字符串**才能定位 cell，所以两步合一：OCR 出编号 → 用 OCR 的多边形 + morph 线围 cell → 决定最终框
- 所有 OCR 都走统一入口 `_ocr_region`（小图自动放大、大图自动缩到 2100px、加 padding+锐化、predict 失败重试 3 次）

### 三个框的依赖关系

```
                     ┌──────────────────┐
                     │ 红框 (MATERIAL   │
                     │   CODE 列)        │
                     │ ─ v5 锁定位置    │
                     └────────┬─────────┘
                              │ 锁定 mat_bbox
                              ▼
       ┌──────────────────────────────────────┐
       │ 橙框搜索区 = BBox(0, 0, mat.x, mat.y)│
       │ ─ 橙框严格位于红框的"左上"象限         │
       └──────────────────────────────────────┘

┌──────────────────────┐        ┌──────────────────────┐
│ 红框 (并行)           │        │ 绿框 (并行)           │
│ ─ ThreadPoolExecutor │   ↔    │ ─ ThreadPoolExecutor │
│   max_workers=2       │        │   max_workers=2       │
└──────────────────────┘        └──────────────────────┘
                              ↓
                       ┌──────────────┐
                       │ 橙框 (串行)   │
                       │ 等红框结果    │
                       └──────────────┘
```

红、绿并行；橙串行等红框。绿框定位失败时走 `_fallback_detect_bottom_right_number`（百分比兜底框）。

### 绿/橙框共享的"两段式"模式

两个框的核心套路一样：

```
                 [OCR 找编号文本]
                        │
              text + ocr_bbox + 4 点多边形
                        │
        ┌───────────────┴───────────────┐
        │                               │
[_detect_morph_lines]            [文本验证 / 合体切割 / 截断扩展]
   vlines + hlines                       │
        │                               │
        └───────────────┬───────────────┘
                        ▼
              [_find_cell_from_lines]
        基于"text 中心 + 最近的线"围成 cell
                        │
                        ▼
            [_make_green_bbox / _make_orange_bbox]
                cell 与 OCR 框的取舍规则
                        │
                        ▼
                  最终绿框 / 橙框
```

**绿框比橙框多一步"二次裁切"**：用第一轮粗 cell 做基准，剪出更干净的子图再做二次 OCR + morph。
原因：右下角编号栏常与外侧标题栏图框紧贴，第一轮 morph 线会被外框污染。

---

## 总体流程图

```
                              detect_all_regions(image, prefixes)
                                         │
        ┌────────────────── Phase A ─────┴───────────────────┐
        │ 1. 纵向自动旋转 (_auto_rotate_portrait, L2741)     │
        │    img_h > img_w → 逆时针 90°，记录 rotation 元数据 │
        │                                                    │
        │ 2. 图纸边框检测 (_detect_drawing_frame, L563)       │
        │    形态学找最长水平/竖直线 → BBox(frame)            │
        │                                                    │
        │ 3. 三大搜索区静态计算                                │
        │    红框 : x ∈ [8% W, frame.x + 2.8/8·frame.w]      │
        │           y ∈ [0, 95% H]                            │
        │    绿框 : x ∈ [5/8 W, W], y ∈ [5/6 H, H]           │
        │    橙框 : 依赖红框结果，Phase B 后再算               │
        └─────────────────────┬──────────────────────────────┘
                              │
        ┌────────────────── Phase B ─────┴───────────────────┐
        │ _crop_and_scale 各搜索区裁切 + 长边>2100px 时缩放    │
        │                                                    │
        │ ThreadPoolExecutor(max_workers=2) 并行：             │
        │   ┌─ _locate_material_code_column (红框 v5)          │
        │   └─ _locate_bottom_right_number  (绿框 VLM)         │
        │                                                    │
        │ 红框结果出来后：                                     │
        │   橙框搜索区 = BBox(0, 0, red.x, red.y)             │
        │   _locate_top_left_number (橙框 VLM)  ← 串行         │
        │                                                    │
        │ _map_bbox_back 把子图坐标映射回原图                  │
        └─────────────────────┬──────────────────────────────┘
                              │
                       ┌──────┴───────┐
                       │  结果 dict   │
                       │   material_code_column │
                       │   bottom_right_number  │
                       │   top_left_number      │
                       │   _metadata            │
                       └──────────────┘
```

---

## Phase A：全图预处理 + 边框检测

### A-1. 纵向自动旋转 — [`_auto_rotate_portrait`](../modules/region_detector.py#L2741)

```python
if img_h > img_w:
    image, rot_code = _auto_rotate_portrait(image, prefixes=prefixes)
```

- **触发**：图片高 > 宽（图纸约定宽必 > 高）
- **动作**：`cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)`，逆时针 90°
- **元数据**：`regions["_metadata"]["rotation"]` 记录旋转码，供上层把原图同步旋转后再做替换

### A-2. 图纸边框检测 — [`_detect_drawing_frame`](../modules/region_detector.py#L563)

目的：找最外圈实线方框，给红/绿框搜索区的比例公式提供基准。

**算法**：

1. **降采样**：长边 > 2100px 时缩到 2100px（仅用于检测，坐标后还原）
2. **多阈值多长度尝试**：
   - 第一轮：`THRESH_BINARY_INV` 阈值 150 / 130 / 100，长度 ≥ 50% 图宽/高
   - 第二轮：同阈值，长度 ≥ 40%，加合理性校验（边框需贴近图像边缘 + 覆盖 ≥ 70%）
3. **形态学核**：水平线 `(int(w·0.5), 1)`，竖直线 `(1, int(h·0.5))`
4. **失败兜底**：用图片边界 5% 内缩 (`frame = BBox(0.05W, 0.05H, 0.9W, 0.9H)`)

### A-3. 三大搜索区比例公式

```python
# 红框：左 8% 起，到画框 2.8/8 处，高度到 95%
red_x = int(img_w * 0.08)
red_w = frame.x + int(frame.w * (2.8 / 8.0)) - red_x
red_h = int(img_h * 0.95)
red_search = BBox(red_x, 0, red_w, red_h)

# 绿框：右下 3/8 × 1/6 子图
green_x = int(img_w * (5.0 / 8.0))
green_y = int(img_h * (5.0 / 6.0))
green_search = BBox(green_x, green_y, img_w - green_x, img_h - green_y)

# 橙框（Phase B 后，红框结果出来再算）
orange_search = BBox(0, 0, mat_bbox.x, mat_bbox.y)
# 红框未检出时的兜底：BBox(0, 0, img_w*2/8, img_h*2/12)
```

---

## 通用基础设施

### `BBox` — [L77](../modules/region_detector.py#L77)

简单矩形容器：`.x`, `.y`, `.w`, `.h`, `.x2`, `.y2`, `.crop(image)`, `.contains(other)`。

### `_crop_and_scale(image, bbox, max_edge=2100)` — [L317](../modules/region_detector.py#L317)

- 裁切搜索区 → 长边 > 2100 → 等比缩到 2100
- 返回 `(sub_image, scale_factor)`

### `_map_bbox_back(bbox, crop_bbox, scale)` — [L333](../modules/region_detector.py#L333)

子图坐标 + scale → 还原到原图。所有 Phase B 输出最后必走这步。

### `_ocr_region(image, bbox, lang, engine)` — [L347](../modules/region_detector.py#L347)

统一 OCR 入口。

| 步骤 | 行为 |
|---|---|
| 裁 ROI | `bbox.crop(image)` |
| 自适应缩放 | `min(w,h) < 80` → ×5；`< 200` → ×3；`max(w,h) > 2100` → 缩到 2100；上限 3500 |
| 小区域 padding + 锐化 | `h<100 且 w<400` → 白色 pad（≥40），跑 3×3 锐化卷积 |
| 引擎 | `engine="v5"` → PaddleOCR v5；`engine="vlm"` → VLM (PaddleOCR-VL-1.5 via vLLM) |
| 重试 | predict 失败重试 3 次 |
| 坐标回写 | poly 每点 `(px-pad)/scale + bbox.x` 还原到原图坐标 |

返回 `[(text, conf, poly), ...]`，poly 为 4 点多边形。

### `_fuzzy_find_keyword(ocr_results, keywords, threshold)` — [L517](../modules/region_detector.py#L517)

按 `difflib.SequenceMatcher` 比例做模糊匹配，返回 `{text, keyword, score, poly}`。

### `_fuzzy_fix_y_text(raw_text, prefixes)` — [L1287](../modules/region_detector.py#L1287)

去空格后做"符号 → 数字"模糊回填：

- 字母数字保留；`/ ! | \` → `1`；`( )` → `0`（来自 `FUZZY_DIGIT_MAP`）
- 其他符号 → 返回 `None`
- 修复后若首字母 `T` 且能匹配正则 → `T → Y` 替换

### `make_pattern(prefixes)` — config.py

```
prefixes=["Y","X"] → r'\b[YX][A-Z0-9]{6,}\b'   # 总长 ≥ 7
prefixes=["Y"]     → r'\bY[A-Z0-9]{6,}\b'
```

绿/橙框另有宽松正则，见各自章节。

---

## 红框：MATERIAL CODE 列

> **目的**：定位 BOM 表里"材料代号"那一列的矩形位置（精确到列宽 + 数据行 y 范围）。**不读取**列内具体的零件编号——那是 text_replacer 阶段的事。

### 入口 — [`_locate_material_code_column(red_sub)`](../modules/region_detector.py#L742)

输入：红框搜索区裁切子图（已 `_crop_and_scale`）。
返回：`(col_bbox, "vertical")` 或 `(None, None)`。

### 红框完整流程

```
红框搜索区子图（左 8%~画框 2.8/8，全高 95%）
        │
        ▼
┌────────────────────────────────────────────┐
│ 红框-1 表头 OCR + 关键词匹配（v5 英 + 中） │
│   _ocr_region 顶部 20% 带，模糊阈 0.70    │
│   关键词: MATERIAL CODE / 零部件图号 …     │
│   失败 → 整张图视为非标准布局，return None │
└──────────────────┬─────────────────────────┘
                   │ 得到 kw_cx (关键词中心 x)
                   ▼
┌────────────────────────────────────────────┐
│ 红框-2 邻居列锚点                          │
│   在同一批 OCR 结果中找：                  │
│     DEF (左邻) - 阈 0.7                    │
│     品/群 (右邻) - 含汉字"品"或"群"       │
└──────────────────┬─────────────────────────┘
                   │ 得到 def_cx / shin_cx
                   ▼
┌────────────────────────────────────────────┐
│ 红框-3 _trace_vertical_table 竖向追踪      │
│   ① 粗定位 v_lines (窄带 4 轮)             │
│   ② _get_vline_y_extent 算表格 y 范围       │
│      (左右两根竖线 y 范围并集)              │
│   ③ 精定位 v_lines (在表格 y 范围内重做)   │
│   ④ 列宽 = 包住 kw_cx 的两根 v_line        │
│      + DEF/品群锚点校正                    │
│   ⑤ h_lines 数据行 (递归向下延伸)          │
│   ⑥ 高度 < 500px → gap=120 重试            │
└──────────────────┬─────────────────────────┘
                   │ col_bbox
                   ▼
┌────────────────────────────────────────────┐
│ 红框-4 列内容验证                          │
│   _validate_material_code_column           │
│   向上扩 200px 包含表头，再 OCR (英+中)    │
│   找不到 MATERIAL CODE/零部件图号 → return None│
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│ 红框-5 DWG 左边界修正                      │
│   _fix_dwg_left_boundary                   │
│   列内 OCR 若识别到 "DWG..." → 右推左边界 │
└──────────────────┬─────────────────────────┘
                   ▼
            (col_bbox, "vertical")
```

### 红框-1：表头关键词检测

```python
header_bbox = BBox(0, 0, img_w, max(int(img_h * 0.2), 100))
ocr_results = _ocr_region(sub_image, header_bbox)  # 英文 v5

# 模糊关键词
keywords = ["MATERIAL CODE", "MATERIALCODE", "MATERIAL",
            "材料代号", "零部件图号", "PARTS LIST", "代号"]
match = _fuzzy_find_keyword(ocr_results, keywords, threshold=0.70)

if match is None:
    # 中文 v5 回退
    ocr_results = _ocr_region(sub_image, header_bbox, lang="ch")
    match = _fuzzy_find_keyword(ocr_results, keywords, threshold=0.70)
if match is None:
    return None, None  # 非标准布局
```

只在图顶部 20%（最少 100px）做关键词检测，避免在整片红框搜索区扫描。

### 红框-2：邻居列锚点

| 邻列 | 关键字判定 | 位置约束 |
|---|---|---|
| **DEF**（左邻） | 严格 `"DEF"/"DEF."` 或 `len≤5 且 difflib(text,"DEF") ≥ 0.7` | 优先 `cx < mat_cx`；都不在左侧则取最近 |
| **品 / 群**（右邻） | 文本包含汉字 `品` 或 `群` | 优先 `cx > mat_cx`；都不在右侧则取最近 |

这两个锚点防止把 DEF 列或品群列错并入 MATERIAL CODE 列。

### 红框-3：`_trace_vertical_table` 五步 — [L897](../modules/region_detector.py#L897)

#### ① 粗定位 v_lines（窄带四轮兜底）

```
窄带 = ROI 截 y∈[kw_cy ± max(100, roi_h·5%)],
              x∈[kw_cx ± max(400, roi_w·15%)]
```

| 轮 | 检测方式 | 触发条件 |
|---|---|---|
| 1 | morph 竖线（高度阈 30% 带高） | 默认 |
| 2 | morph，y 扩到 `[kw_cy-50, kw_cy + roi_h·30%]`，阈 10% 带高 | 第1轮 < 2 条 |
| 3 | 投影法 `_detect_vlines_by_projection` 替换 morph | 第2轮仍 < 2 条 |
| 4 | 用关键词 poly 自身宽度 + 5% 边距 + DEF 锚点估列宽 | 三轮全失败 |

每轮都做 "v_lines 必须靠近 `kw_cx ± 200px`" 的距离校验。

#### ② 表格 y 范围 — [`_get_vline_y_extent`](../modules/region_detector.py#L850)

对粗定位出的左右两根列边界竖线，向上向下扫描像素，遇到 `gap > 80px` 中断。两边 y 范围取**并集** → `table_y_min / table_y_max`。

#### ③ 精定位 v_lines

在 `table_y_min~table_y_max × [col_left ± rough_col_w]` 内重做 morph 竖线检测，高度阈 15% 表高。

#### ④ 列宽 + 锚点校正

```python
# 找包住 kw_cx 的相邻 v_lines 对
for i in range(len(v_lines)-1):
    if v_lines[i] <= kw_cx <= v_lines[i+1]:
        col_left, col_right = v_lines[i], v_lines[i+1]
        break

# DEF 锚点：若 def_cx 在列内 → 找 def_cx ↔ kw_cx 之间的 v_line 右推 col_left
# 品群锚点：若 shin_cx 在列内 → 找 kw_cx ↔ shin_cx 之间的 v_line 左推 col_right
# 列宽上限 600px：超过则视精定位失败，回退到 ① 的窄带 v_lines
```

#### ⑤ h_lines + 数据行边界

```python
col_region = table_gray[table_y_min:table_y_max, col_left-5 : col_right+5]
h_lines = _detect_horizontal_lines(col_region, min_line_width=max(col_w*0.3, 10))
```

**向下递归延伸**：每次扩 `avg_row_h × 3` 高度的搜索区，直到一段无新横线为止（最多延伸 9 行）。
应对"竖线在与横线交叉处断裂但横线继续"的情况。

- `data_y_top` = 第一条 `y ≥ kw_cy - 5` 的横线
- `data_y_bot` = `h_lines[-1]`

#### ⑥ 高度回退

若 `result.h < 500px`：太矮 → 用 `gap_threshold=120`（更宽松的 y 范围合并）重算一次。

### 红框-4：列内容验证 — [`_validate_material_code_column`](../modules/region_detector.py#L653)

```python
extended_bbox = BBox(col.x, col.y - 200, col.w, col.h + 200)  # 向上扩 200 抓表头
ocr_en = _ocr_region(image, extended_bbox)
ocr_ch = _ocr_region(image, extended_bbox, lang="ch")
```

- 第一轮**精确匹配**：列内任一 OCR 包含 `MATERIALCODE / MATERIAL / 零部件图号`
- 第二轮**模糊匹配**（阈 0.60）：`["MATERIAL CODE", "MATERUL CODE", "零部件图号", "DEF"]`
- 两轮都失败 → 整个红框定位作废，返回 None

### 红框-5：DWG 左边界修正 — [`_fix_dwg_left_boundary`](../modules/region_detector.py#L686)

红框定位偏大时常把左侧 DWG 子列也包进来。对列内 OCR：

- 文本以 `DWGNO` / `DWG` 开头：
  - 若文本 = `DWG`（独立）→ 列新左边界 = DWG poly 右边界
  - 若文本 = `DWGZ212C0008`（合体）→ 用字符比例 `poly_left + poly_w · prefix_chars/total_chars` 估算编号起点

---

## 绿框：右下角编号

> **目的**：在右下角标题栏里找到 Y 编号（如 `YA057C857`），并精确围住它所在的单元格。

### 入口 — [`_locate_bottom_right_number(green_sub, prefixes)`](../modules/region_detector.py#L2069)

输入：绿框搜索区裁切子图（已 `_crop_and_scale`）。
返回：`(text, BBox)` 或 None。

### 绿框完整流程

```
绿框搜索区 (右下 3/8 × 1/6 子图)
        │
        ▼
┌─────────────────────────────────────────────────┐
│ 绿框-1 第一轮 OCR：找到编号文本                  │
│   L1 整片 OCR (VLM)                              │
│     ├ L1a 右侧竖线裁剪 (去跨格残字)               │
│     ├ L1b 直接验证 (≥9字符 + 末位字母数字)       │
│     └ L1c 截断扩展 (≥7字符未通过 → 右扩重新OCR)   │
│   L2 长线裁切 (右数第3+下数第3 长线)              │
│   L3 长线裁切 (右数第2+下数第2 长线)              │
│   L4 预处理增强 (高斯模糊+自适应阈值) + 整片OCR   │
│                                                  │
│  → y_text_orig + ocr_bbox_orig                   │
└─────────────────────────────────────────────────┘
        │
        ▼  v1 cell 粗围 (_find_cell_boundary)
┌─────────────────────────────────────────────────┐
│ 绿框-2 二次裁切                                  │
│   左/上各放 25% pad，右/下保留到子图边           │
│   目的：剥掉外圈图框，让 morph 干净               │
└─────────────────────────────────────────────────┘
        │
        ▼  在二次裁切图上重做：
┌─────────────────────────────────────────────────┐
│ 绿框-3 重 OCR + morph 线 + cell                  │
│   - _locate_bottom_right_number_core 重跑 L1~L4 │
│   - _detect_morph_lines(v_ratio=6.5)            │
│   - 失败 → 回退到第一轮 ocr_bbox + v1 cell      │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│ 绿框-4 围 cell + 合成绿框                        │
│   _find_cell_from_lines(text, vlines, hlines)   │
│   _make_green_bbox(ocr_bbox, cell)               │
└─────────────────────────────────────────────────┘
        │
        ▼  +cx1/+cy1 还原到子图坐标系
   (text, green_bbox)
```

### 绿框-0：图像预处理

由 [`_crop_and_scale`](../modules/region_detector.py#L317) 完成（入口外）：

- 裁出 `BBox(W·5/8, H·5/6, W·3/8, H·1/6)` — 右下角 3/8 × 1/6
- 长边 > 2100 → 等比缩放
- 进入函数后 `gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)`

### 绿框-1：第一轮 OCR — [`_locate_bottom_right_number_core`](../modules/region_detector.py#L2144)

#### 正则定义

```python
y_re     = re.compile(make_pattern(prefixes))                 # 严正则 [YX][A-Z0-9]{6,}
loose_re = re.compile(r"[A-Z][A-Z0-9]*\d{2,}[A-Z]\d{2,}")     # 宽松正则（任意首字母）
```

宽松正则用于 OCR 漏字 / 首字母误识时的兜底。

#### `_search_y_number` — 编号查找核心 — [L1314](../modules/region_detector.py#L1314)

```python
ocr_results = _ocr_region(image, search, engine="vlm")
```

每条 OCR 文本依次尝试：

1. `text.upper().replace(" ", "")` → `y_re.search`
2. 失败 → `_fuzzy_fix_y_text`（符号 → 数字 + T → Y）→ 重新 `y_re.search`
3. 仍失败 → `loose_re.search`，且文本 ≥ 5 字符
4. 若 loose 匹配且首字母非 Y → 尝试 `'Y' + text` 再走 `y_re`（**前导 Y 补全**）
5. 多个匹配中**取 bbox 面积最大**（避免选到页码、小标注）
6. **tight_bbox 缩紧**：OCR 完整文本比正则匹配长时，按字符宽估算编号在文本中的 x 区间，左右各扩 0.3 字符宽

#### 四层定位 (L1~L4)

```
L1 全区域 OCR
  ├ L1a 右侧竖线裁剪：
  │      _find_right_vline 在 ocr_bbox 右侧 ±1 字符宽内找竖线
  │      竖线 x < bbox.x2-5 → bbox 跨了单元格（多读了"A216H"页码）
  │      → _trim_text_by_vline 把文本和 bbox 都截到竖线左侧
  │      → _validate_green_result 通过则返回
  │
  ├ L1b 直接验证通过 → 返回 (text, ocr_bbox)
  │
  └ L1c 截断修复（关键）：
        条件: text.strip() 长度 ≥ 7 但验证失败（<9 字符）
              意味着 OCR 把 9 位编号读成 7~8 位（如 'YA057C85' 漏末尾 '7'）
        动作: char_w = bbox.w / len(text)
              ext_r = min(char_w·2.5, 右边可扩空间)
              ext_v = bbox.h·0.5  (上下各扩 50%)
              在扩展 bbox 上重新 OCR
        修复: _fix_digit_letter_confusion
              I→1, O→0, S→5, B→8, Z→2, G→6, T→7, Q→0, D→0
              （仅在编号已知数字位 2/3/4 和 6+ 替换）
        验证: ≥9 字符且末位字母数字 → 返回

L2 长线裁切 (右数第 3 根长竖线左侧 + 下数第 3 根长横线上方截掉外圈)
  └ _search_and_extend：OCR + 截断扩展 + 数字字母修复 + 验证

L3 长线裁切 (右数第 2 根 + 下数第 2 根)
  └ 同 L2，剥得更紧

L4 预处理增强 (_preprocess_green_ocr: 高斯模糊 + 自适应阈值) + 整片 OCR
  └ 同 L2 流程
```

#### 为什么要 L2 / L3 — 长线检测 `_detect_long_lines` — [L1507](../modules/region_detector.py#L1507)

V2 编号栏外层套着多重边框：表格内框 + 标题栏外框 + 图纸主框。
长线检测先用形态学找所有竖/横线，按长度排序找**最大长度落差点**（间隙 > 中位间隙 × 3 且 > 30px），落差以上为"长线"。
然后用从右数第 N 根长竖线 + 从下数第 N 根长横线把搜索区**逐圈剥离**：

- **N=3（L2）**保守，外圈再加内圈
- **N=2（L3）**激进，剥得更紧

#### `_validate_green_result(text)` — [L1399](../modules/region_detector.py#L1399)

```python
return t.strip() 长度 ≥ 9 且 t[-1].isalnum()
```

编号长度典型 9~13 字符。

### 绿框-2：二次裁切（关键步骤）

第一轮拿到 `(y_text_orig, ocr_bbox_orig)` 后，用 [`_find_cell_boundary`](../modules/region_detector.py#L1714) 做一次粗 cell 定位（v1 版本），得到 `cell_v1` 作为裁切基准。

```python
pad_x = int(cell_v1.w * 0.25)
pad_y = int(cell_v1.h * 0.25)
cx1 = max(0, cell_v1.x - pad_x)   # 左/上各放 25% pad
cy1 = max(0, cell_v1.y - pad_y)
cx2 = img_w                        # 右、下保留到子图边
cy2 = img_h

cropped_rgb  = sub_image[cy1:cy2, cx1:cx2]
cropped_gray = gray[cy1:cy2, cx1:cx2]
```

**为什么这样裁？**

- **右下角编号栏是"角"**——左/上有其他表格列；右/下是图纸白边
- 往左/上扩 25% 保留表格上下文（让 morph 线检测能找到 cell 的左/上边界）
- 往右/下保留到边沿则不损失任何信息
- 二次裁切的核心目的：**剪掉外圈大边框**，让 morph 线检测干净（外圈大框会被识别成"边界"误导 cell 围合）

### 绿框-3：在裁切图上重做 OCR + morph + cell

```python
# 在 cropped 上重跑核心（同样 L1~L4 流程）
new_text, new_ocr = _locate_bottom_right_number_core(
    cropped_rgb, cropped_gray, ch, cw, prefixes
)

# 在 cropped 上检测 morph 线
vlines, hlines = _detect_morph_lines(cropped_gray, v_ratio=6.5)

# 围 cell + 合成绿框
cell = _find_cell_from_lines(new_ocr, vlines, hlines, cw)
green = _make_green_bbox(new_ocr, cell)
```

**二次 OCR 失败的兜底**：用第一轮 `(y_text_orig, ocr_bbox_orig)` + `_find_cell_boundary(v1)` 当最终结果。

### 绿框-4：`_detect_morph_lines` — [L1852](../modules/region_detector.py#L1852)

| 项 | 公式 |
|---|---|
| 竖线最小长度 | `min_vh = max(h / v_ratio, 12)`；绿框 `v_ratio=6.5` |
| 竖线核 | `(1, min_vh)` 形态学开运算，`iterations=2` |
| 横线最小长度 | `min_hw = max(w / 8, 12)` |
| 横线核 | `(min_hw, 1)` 形态学开运算，`iterations=1` |
| 邻线合并 | 同方向相邻线 ≤ 10px 间距合并（取长度并集） |
| `enhance_lines=True` 投影补线 | 列/行像素密度 > `max(mean·3, 0.15)` 视为线，与现有 ±10px 内已存在则跳过 |

**绿框调用时 `enhance_lines=False`**（不补淡线）；橙框调用时启用。

### 绿框-5：`_find_cell_from_lines` — [L1929](../modules/region_detector.py#L1929)

严格规则——每条边界**只能**来自 morph 线或 OCR 边界，**不允许凭空插值**：

| 边界 | 选取规则 | 兜底 |
|---|---|---|
| 上 | `text_cy` **上方**最近的横线 | OCR 上边界 |
| 下 | `text_cy` **下方**最近的横线 | OCR 下边界 |
| 左 | `text_cx` **左侧**且 `x ∈ [OCR.x ± OCR.w·10%]` 内最近的竖线 | OCR 左边界 |
| 右 | `text_cx` **右侧**最近的竖线 | OCR 右边界 |

左边界的 10% 窗口是防止取到远处的列分隔线。
围完后做"必须包住 OCR bbox"的安全裁剪。

### 绿框-6：`_make_green_bbox(ocr, cell)` — [L2025](../modules/region_detector.py#L2025)

| 比较 | 结果 |
|---|---|
| OCR 面积 > cell 面积 | 绿框 = cell；若 OCR 右边界还在 cell 内 → **右边界用 OCR 的**（更紧） |
| OCR 面积 ≤ cell 面积 | 绿框 = OCR；溢出 cell 的边裁回到 cell 边界 |

为什么会出现 OCR > cell？说明 OCR 跨了单元格或包含多余字符——这种情况就用 cell 锁住范围，但右边界跟 OCR 走（OCR 右边界更准）。

最后 `(green.x + cx1, green.y + cy1)` 映射回子图坐标系。

### 绿框 fallback — [`_fallback_detect_bottom_right_number`](../modules/region_detector.py#L2673)

整个 `_locate_bottom_right_number` 返回 None → 按 `region_config["bottom_right_title"]` 百分比兜底（默认 `x ∈ [75%, 100%], y ∈ [85%, 100%]`）。
这种粗框不再做 OCR，后续替换阶段会在该区域内重做 OCR。

---

## 橙框：左上角编号

> **目的**：在左上角找到图纸自身的 Y 编号（如 `YA057C857`，常与 `DWG NO` 或 `図番` 同格）。

### 入口 — [`_locate_top_left_number(orange_sub, mat_bbox, prefixes)`](../modules/region_detector.py#L2631)

输入：橙框搜索区裁切子图（搜索区 = `BBox(0, 0, mat_bbox.x, mat_bbox.y)`，即红框的"左上"象限）。
返回：`(text, BBox)` 或 None。

### 橙框完整流程

```
橙框搜索区 = (0, 0, mat.x, mat.y) 子图
        │
        ▼
┌────────────────────────────────────────────────┐
│ 橙框-1 预先 morph 线检测                       │
│   _detect_morph_lines(gray, v_ratio=6.5,       │
│                       enhance_lines=True)       │
│   → vlines + hlines（含投影补线，淡线也找）     │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│ 橙框-2 核心定位（三策略串行）                  │
│   _locate_top_left_number_core(..., vlines)    │
│                                                 │
│   策略 1: 找独立 Y 编号 OCR 文本               │
│           + 合框竖线裁切                        │
│           + 竖排判定                            │
│                                                 │
│   策略 2: 从 "DWG NO + 编号" 合体文本中提取    │
│           + 中文 OCR 回退                       │
│           + 子图重做 OCR 精确定位 bbox          │
│                                                 │
│   策略 3: 兜底，取第一个能正则匹配的           │
└────────────────────────────────────────────────┘
        │
        ▼  (y_text, ocr_bbox)
┌────────────────────────────────────────────────┐
│ 橙框-3 围 cell + 合成最终橙框                  │
│   _find_cell_from_lines(ocr_bbox, vlines, hlines)│
│   _make_orange_bbox(ocr_bbox, cell)             │
└────────────────────────────────────────────────┘
        │
        ▼  (text, orange_bbox)
```

### 橙框 vs 绿框的差异

| 维度 | 绿框 | 橙框 |
|---|---|---|
| OCR 出现"长文本嵌入编号" | 罕见（主要是页码截断） | **常见**（编号常和 `DWG NO`/`図番` 同格） |
| morph 线检测时机 | 二次裁切后才算 | **入口就算**（含投影补线），传进 core 给合框处理用 |
| 二次裁切 | 有（左/上 25% pad） | **无** |
| 失败处理 | L1~L4 兜底 | 三策略兜底 |
| 重要修正 | 数字字母混淆修复 | **DWG 合体切割** + 中文 OCR 回退 |

橙框搜索区比绿框小得多（只在红框上方）、内容更杂（DWG/図番、客户号、版次等），所以策略不一样。

### 橙框-1：预先检测 morph 线（带投影增强）

```python
gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)
vlines, hlines = _detect_morph_lines(gray, v_ratio=6.5, enhance_lines=True)
```

**为什么 `enhance_lines=True`？**
图纸里 "DWG NO" 和编号之间常有一根**淡竖线**（印刷分隔线），morph 阈值偏严会漏掉。
投影补线用"列像素密度 > `mean·3`"的标准把这种淡线找回来。
这些 vlines 后续传给 core，让它在"OCR 文本嵌入多字段"时用竖线把字段切开。

### 橙框-2：核心定位 — [`_locate_top_left_number_core`](../modules/region_detector.py#L2361)

整片搜索区 `BBox(0, 0, img_w, img_h)` 一次性 OCR（engine="vlm"）。

#### 策略 1：找独立 Y 编号

```python
for text, conf, poly in ocr_results:
    text_up = text.upper().strip()

    # 跳过 DWG / OWG / 図 / 图 开头（这些是合体文本，留给策略 2）
    if text_up.startswith(("DWG", "OWG", "図", "图")): continue

    # 优先比较去空格版本（OCR 可能在编号中插入空格 "YA057 C857"）
    m_orig = y_re.search(text_up)
    m_nosp = y_re.search(text_up.replace(" ", ""))
    m = 取匹配更长的；相同长度优先去空格版

    # 失败 → 符号回填 + T→Y
    if not m:
        fuzzy = _fuzzy_fix_y_text(text_no_sp)
        if fuzzy: m = y_re.search(fuzzy)

    if m:
        # 跳过位于 mat_bbox 内的（与红框重叠 → 干扰）
        if material_code_bbox.contains(ocr_bbox): continue
        ...
```

##### 1.1 合框 OCR 处理（关键！）

```
触发条件: OCR 完整文本比正则匹配长（编号被嵌在更长文本中）
          例: text='DWG.NO YA057 C857', match='YA057C857'

动作:
  ① 在 OCR poly 内列出所有竖线（按 x 排序，左→右）
  ② 逐根竖线：以"竖线 x 为新左边界，OCR 顶/底 ±5px，OCR 右边界为右"截子图
  ③ 子图重做 OCR
  ④ 第一个 OCR 出"纯净编号"（去空格后长度 = 正则匹配长度）的子图 → 用其 bbox 返回
  ⑤ 所有竖线都失败 → 用 OCR 原始 bbox
```

##### 1.2 竖排文字判定

```python
if ocr_bbox.h > ocr_bbox.w * 3:
    # 竖排编号（罕见，但有些图纸图号是竖排）
    return (matched_text, ocr_bbox)
```

##### 1.3 直接返回

策略 1 找到独立编号且不是合框/竖排 → 直接返回 OCR bbox。

#### 策略 2：从 DWG NO 合体提取

策略 1 找不到时启动。

```python
dwg_keywords = ["DWG NO", "DWG NO.", "DWGNO", "DWG",
                "図番", "図面番号", "图号"]
dwg_match = _fuzzy_find_keyword(ocr_results, dwg_keywords, threshold=0.60)
```

##### 2.1 中文 OCR 回退

英文 OCR 没找到关键词 → 用 `lang="ch"` 重做整片 OCR，再找关键词。
原因：扫描图纸里 `図番` 这类汉字 + 英文 OCR 的组合常识别不全。

##### 2.2 切前缀 + 编号匹配

```
text = 'DWG.NO YA057 C857'
  ① 找前缀结束位置（"DWG NO." 长度 7）+ 跳过随后的空格/句点 → prefix_end=8
  ② num_part = 'YA057 C857'
  ③ num_part_clean = 'YA057C857' → y_re.search ✓
  ④ 模糊回填：失败时再 _fuzzy_fix_y_text
```

##### 2.3 编号 bbox 精定位（用子图重 OCR）

> DWG NO 和编号之间有竖线分隔，字符等比例估算不可靠。

```python
# 字符比例粗定位编号首字母 x（已处理 num_part 内空格的位置映射）
char_w = poly_w / len(text)
x_start = poly_left + match_start_orig * char_w

# 从 (x_start, poly_top) 到 (poly_right, poly_bottom) 截子图重 OCR
local_results = _ocr_region(sub_image, BBox(x_start, ..., width, height))

# 在子图 OCR 结果中找匹配的"纯净编号"（OCR 文本长度 == 正则匹配长度）→ 用其 bbox 返回
```

#### 策略 3：兜底

```python
# 遍历所有 ocr_results，第一个能匹配 y_re 的就返回
for text, conf, poly in ocr_results:
    text_clean = text.upper().replace(" ", "")
    m = y_re.search(text_clean)
    if m and poly:
        return (m.group(), poly_to_bbox(poly))
```

### 橙框-3：cell + 最终橙框

```python
cell = _find_cell_from_lines(ocr_bbox, vlines, hlines, img_w)
orange = _make_orange_bbox(ocr_bbox, cell)
```

[`_make_orange_bbox`](../modules/region_detector.py#L1993) 与 `_make_green_bbox` 规则几乎相同：

| 情况 | 行为 |
|---|---|
| OCR 面积 > cell 面积 | 橙框 = cell；若 OCR 右边界在 cell 内 → 右边界用 OCR 的 |
| OCR 面积 ≤ cell 面积 | 橙框 = OCR；溢出 cell 的边裁回 cell 边界 |

`_make_orange_bbox` 单独写主要是语义清晰——把"橙框"的合成逻辑放在自己的函数里，便于以后调整。

返回 `(text, orange_bbox)`，坐标在 orange_sub 子图坐标系内（上层再 `_map_bbox_back` 回原图）。

---

## 统一收尾

回到 [`detect_all_regions`](../modules/region_detector.py#L2891) 尾段：

1. **坐标还原**：红/绿/橙 BBox 全部走 `_map_bbox_back(sub_bbox, search_area, scale)` 还原到原图坐标
2. **元数据**：

   ```python
   metadata = {
       "method": "keyword",            # 关键词检测失败 → "left_half"
       "table_direction": "vertical",
       "rotation": rot_code,           # cv2.ROTATE_90_COUNTERCLOCKWISE 或 None
       "bottom_right_text": green_text,
       "top_left_text": orange_text,
       "table_search_area": red_search,
       "search_areas": {"red_search": ..., "green_search": ..., "orange_search": ...},
       "crop_scales": {"red": ..., "green": ..., "orange": ...},
   }
   ```

3. **返回**：

   ```python
   {
       "material_code_column": red_bbox_or_None,
       "bottom_right_number":  green_bbox_or_None,
       "top_left_number":      orange_bbox_or_None,
       "_metadata":            metadata,
       # Phase C 才会追加 "factory_note_codes"
   }
   ```

---

## OCR 引擎选择速查

| 框 | 关键词定位 | 编号 OCR | 引擎入口 |
|---|---|---|---|
| 红框 | PaddleOCR v5（英 + 中） | PaddleOCR v5（仅列内容验证用） | `_ocr_region(engine="v5")` |
| 绿框 | — | **VLM**（PaddleOCR-VL-1.5） | `_ocr_region(engine="vlm")` |
| 橙框 | — | **VLM**（含合框处理、DWG 合体提取） | `_ocr_region(engine="vlm")` |

