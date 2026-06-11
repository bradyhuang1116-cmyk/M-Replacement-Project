"""手动编号框编辑器主窗口。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .box_item import BoxItem, BoxState
from .csv_loader import load_csv
from .editor_view import EditorView
from .filename_map import (
    is_replaced_name,
    replaced_to_original_name,
    replaced_to_stem,
)
from .original_view import OriginalView
from .render import CommittedBox, render_committed_boxes

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手动编号框编辑器")
        self.resize(1600, 1000)

        self._original_dir: Path | None = None
        self._replaced_dir: Path | None = None
        self._csv_dir: Path | None = None
        self._csv_data: dict[str, list] = {}
        self._current_file: Path | None = None

        self._build_ui()
        self._refresh_action_state()

    # ── UI ──
    def _build_ui(self):
        # 顶部工具栏：文件夹选择 + 编辑操作
        toolbar = QToolBar("操作", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.btn_pick_original = QPushButton("选原图文件夹")
        self.btn_pick_original.clicked.connect(self._pick_original)
        toolbar.addWidget(self.btn_pick_original)

        self.btn_pick_replaced = QPushButton("选替换后文件夹")
        self.btn_pick_replaced.clicked.connect(self._pick_replaced)
        toolbar.addWidget(self.btn_pick_replaced)

        self.btn_pick_csv = QPushButton("选CSV文件夹")
        self.btn_pick_csv.clicked.connect(self._pick_csv)
        toolbar.addWidget(self.btn_pick_csv)

        toolbar.addSeparator()

        self.btn_add = QPushButton("+ 新增框")
        self.btn_add.setCheckable(True)
        self.btn_add.clicked.connect(self._toggle_add_mode)
        toolbar.addWidget(self.btn_add)

        self.btn_toggle_lock = QPushButton("解锁")
        self.btn_toggle_lock.clicked.connect(self._toggle_lock_selected)
        toolbar.addWidget(self.btn_toggle_lock)

        self.btn_confirm_box = QPushButton("确认框")
        self.btn_confirm_box.clicked.connect(self._confirm_box_selected)
        toolbar.addWidget(self.btn_confirm_box)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("输入新文字…")
        self.text_input.setFixedWidth(220)
        self.text_input.returnPressed.connect(self._confirm_text_selected)
        toolbar.addWidget(self.text_input)

        self.btn_confirm_text = QPushButton("确认文字")
        self.btn_confirm_text.clicked.connect(self._confirm_text_selected)
        toolbar.addWidget(self.btn_confirm_text)

        toolbar.addSeparator()

        self.btn_save = QPushButton("保存当前图")
        self.btn_save.clicked.connect(self._save_current)
        toolbar.addWidget(self.btn_save)

        # 主体：左侧列表 | 右侧上下双视图
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_file_selected)

        self.original_view = OriginalView()
        self.editor_view = EditorView()
        self.editor_view.selection_changed.connect(self._on_box_selection_changed)
        self.editor_view.add_mode_changed.connect(self.btn_add.setChecked)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._wrap_titled("原图（替换前）", self.original_view))
        right_splitter.addWidget(self._wrap_titled("替换后 + 框（可编辑）", self.editor_view))
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self._wrap_titled("替换后文件列表", self.list_widget))
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([280, 1320])

        self.setCentralWidget(main_splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("请先选择三个文件夹")

    def _wrap_titled(self, title: str, w: QWidget) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; padding: 2px;")
        lay.addWidget(lbl)
        lay.addWidget(w)
        return box

    # ── 文件夹选择 ──
    def _pick_original(self):
        d = QFileDialog.getExistingDirectory(self, "选原图文件夹")
        if d:
            self._original_dir = Path(d)
            self._refresh_status()
            self._reload_current()

    def _pick_replaced(self):
        d = QFileDialog.getExistingDirectory(self, "选替换后文件夹")
        if d:
            self._replaced_dir = Path(d)
            self._refresh_file_list()

    def _pick_csv(self):
        d = QFileDialog.getExistingDirectory(self, "选CSV文件夹")
        if d:
            self._csv_dir = Path(d)
            self._csv_data = load_csv(self._csv_dir)
            self._refresh_status()
            self._reload_current()

    # ── 文件列表 ──
    def _refresh_file_list(self):
        self.list_widget.clear()
        if self._replaced_dir is None or not self._replaced_dir.is_dir():
            self._refresh_status()
            return
        files = sorted(
            p for p in self._replaced_dir.iterdir()
            if is_replaced_name(p.name)
        )
        for p in files:
            self.list_widget.addItem(p.name)
        self._refresh_status()

    def _refresh_status(self):
        parts = []
        if self._original_dir:
            parts.append(f"原图: {self._original_dir.name}")
        if self._replaced_dir:
            parts.append(f"替换后: {self._replaced_dir.name} ({self.list_widget.count()} 个)")
        if self._csv_dir:
            parts.append(f"CSV: {sum(len(v) for v in self._csv_data.values())} 条")
        self.status.showMessage(" | ".join(parts) if parts else "请先选择三个文件夹")

    # ── 文件点击 ──
    def _on_file_selected(self, current, _previous):
        if current is None or self._replaced_dir is None:
            return
        name = current.text()
        replaced_path = self._replaced_dir / name
        original_path = None
        if self._original_dir:
            cand = self._original_dir / replaced_to_original_name(name)
            if cand.exists():
                original_path = cand
            else:
                # 尝试 .tiff
                alt = cand.with_suffix(".tiff")
                if alt.exists():
                    original_path = alt
        self._current_file = replaced_path
        self.original_view.show_image(original_path)
        self.editor_view.show_image(replaced_path)
        stem = replaced_to_stem(name)
        records = self._csv_data.get(stem, [])
        self.editor_view.load_boxes(records, on_state_change=self._on_box_state_change)
        self.status.showMessage(
            f"已加载: {name} | 原图: {'OK' if original_path else '缺失'} | CSV框: {len(records)} 个")
        self._refresh_action_state()

    def _reload_current(self):
        item = self.list_widget.currentItem()
        if item is not None:
            self._on_file_selected(item, None)

    # ── 选中变化 ──
    def _on_box_selection_changed(self, _box):
        self._refresh_action_state()

    def _on_box_state_change(self, _box: BoxItem):
        self._refresh_action_state()

    def _refresh_action_state(self):
        box = self.editor_view.selected_box() if hasattr(self, "editor_view") else None
        has_file = self._current_file is not None
        has_box = box is not None
        state = box.state if has_box else None
        self.btn_add.setEnabled(has_file)
        self.btn_toggle_lock.setEnabled(has_box)
        if has_box:
            if state == BoxState.UNLOCKED:
                self.btn_toggle_lock.setText("锁定（重置）")
            else:
                self.btn_toggle_lock.setText("解锁")
        else:
            self.btn_toggle_lock.setText("解锁")
        self.btn_confirm_box.setEnabled(has_box and state == BoxState.UNLOCKED)
        self.btn_confirm_text.setEnabled(
            has_box and state in (BoxState.BOX_CONFIRMED, BoxState.TEXT_COMMITTED))
        self.text_input.setEnabled(self.btn_confirm_text.isEnabled())
        if has_box and state == BoxState.TEXT_COMMITTED:
            self.text_input.setText(box.committed_text)
        elif has_box and state == BoxState.BOX_CONFIRMED and not self.text_input.text():
            self.text_input.setText(box.token)
        self.btn_save.setEnabled(has_file)

    # ── 编辑操作 ──
    def _toggle_add_mode(self):
        self.editor_view.set_add_mode(self.btn_add.isChecked())

    def _toggle_lock_selected(self):
        box = self.editor_view.selected_box()
        if box is None:
            return
        if box.state == BoxState.UNLOCKED:
            box.reset_to_locked()
        else:
            box.unlock()
        self._refresh_action_state()

    def _confirm_box_selected(self):
        box = self.editor_view.selected_box()
        if box is None or box.state != BoxState.UNLOCKED:
            return
        box.confirm_box()
        self.text_input.setFocus()
        self._refresh_action_state()

    def _confirm_text_selected(self):
        box = self.editor_view.selected_box()
        if box is None:
            return
        if box.state not in (BoxState.BOX_CONFIRMED, BoxState.TEXT_COMMITTED):
            return
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先输入新文字")
            return
        box.commit_text(text)
        self._refresh_action_state()

    # ── 保存 ──
    def _save_current(self):
        if self._current_file is None or not self._current_file.exists():
            return
        committed = self.editor_view.committed_boxes()
        if not committed:
            QMessageBox.information(self, "无修订",
                                     "当前图没有任何已确认文字的框，未保存。")
            return
        # 直接覆盖原图（同目录、原名，不新建子目录、不加后缀）
        out_path = self._current_file
        try:
            render_committed_boxes(
                self._current_file,
                [CommittedBox(x, y, w, h, t) for (x, y, w, h, t) in committed],
                out_path,
            )
        except Exception as e:  # noqa: BLE001 -- 显示给用户即可
            logger.exception("保存失败")
            QMessageBox.critical(self, "保存失败", str(e))
            return
        QMessageBox.information(
            self, "已保存",
            f"已生成：{out_path}\n\n共写入 {len(committed)} 个修订框。")
        self.status.showMessage(f"已保存: {out_path}")
