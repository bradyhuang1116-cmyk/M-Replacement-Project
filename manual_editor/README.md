# 手动编号框编辑器 (manual_editor)

独立于主流水线的桌面工具。读取**原图 TIF**、**替换后 TIF (HXXX-R.tif)** 和**y_boxes.csv**，
在交互界面里对单个编号框进行手动校正：拖拽改大小 → 锁定 → 输入文字 → 落到画布上。

> 本工具**不依赖、也不修改**主流水线代码。`modules/` 下任何文件保持原样。

## 安装

```bash
# 复用项目的 conda 环境
/c/Users/Brady\ Huang/miniconda3/envs/mitsubishi/python.exe -m pip install -r manual_editor/requirements.txt
```

或新建独立环境：

```bash
python -m venv .venv-editor
.venv-editor/Scripts/activate  # Windows
pip install -r manual_editor/requirements.txt
```

## 启动

```bash
/c/Users/Brady\ Huang/miniconda3/envs/mitsubishi/python.exe manual_editor/main.py
```

## 使用流程

1. 工具栏依次点击 **选原图文件夹** / **选替换后文件夹** / **选CSV文件夹**
   - 原图：含 `XXX.tif`
   - 替换后：含 `HXXX-R.tif`
   - CSV：含 `y_boxes.csv`（主流水线产出的格式 `source_file,token,x1,y1,x2,y2`）
2. 左侧文件列表点选 `HXXX-R.tif`
   - 右上自动加载对应原图 `XXX.tif`（找不到会显示占位文字）
   - 右下加载替换后图 + 该图全部 CSV 框（**蓝色 = 锁定**）
3. 点击某个框 → 点 **解锁** → 框变红，4 角出现白色拖拽手柄
   - 可拖框移动整体，可拖手柄改大小
4. 点 **确认框** → 框变橙；在文本框里输入文字 → 点 **确认文字**（或回车）
   - 框变绿，框内立刻渲染**白底黑字预览**
5. 点 **保存当前图** → 写出到 `{替换后文件夹}/edited/HXXX-R_edited.tif`
   - 所有 `TEXT_COMMITTED` 状态的框会被绘制成**白底黑字**
   - 其它框不影响图像

> 框 4 态颜色：🔵 锁定 / 🔴 解锁 / 🟠 已确认几何 / 🟢 已确认文字

## 命名约定

替换后文件 `HXXX-R.tif` ←→ 原图 `XXX.tif` ←→ CSV `source_file = XXX`

```python
from manual_editor.app.filename_map import replaced_to_stem
replaced_to_stem("HYA116A226-1_Eg-脱敏-R.tif")  # -> "YA116A226-1_Eg-脱敏"
```

## 目录结构

```
manual_editor/
├── main.py                  # 入口
├── requirements.txt
├── README.md
├── app/
│   ├── main_window.py       # QMainWindow + 工具栏 + 三面板布局
│   ├── original_view.py     # 右上只读原图
│   ├── editor_view.py       # 右下可交互（图+框）
│   ├── box_item.py          # 4 态可拖拽框
│   ├── csv_loader.py        # 读 y_boxes.csv
│   ├── filename_map.py      # HXXX-R <-> XXX 映射
│   ├── render.py            # 保存时把白底黑字落到 PIL 图像
│   └── font_config.py       # 找项目自带宋体字体
└── tests/
    └── test_filename_map.py
```

## 测试

```bash
/c/Users/Brady\ Huang/miniconda3/envs/mitsubishi/python.exe -m pytest manual_editor/tests -v
```

## 已知限制 / TODO

- 当前保存输出到 `edited/` 子目录的新文件；后续会切换为覆盖原 `HXXX-R.tif`
- 撤销/重做未实现
- 大图（>8K）首次加载偶尔有 1–2s 卡顿（PIL → QPixmap 一次性转换）
