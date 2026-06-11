"""右下可交互视图：替换后图像 + CSV 框 + 用户编辑。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .box_item import BoxItem, BoxState
from .csv_loader import BoxRecord


class EditorView(QGraphicsView):
    """图像 + 框图层。

    Signals
    -------
    selection_changed(BoxItem | None)
        当前选中的框变化时触发。
    """

    selection_changed = Signal(object)  # BoxItem | None
    add_mode_changed = Signal(bool)     # add_mode 退出/进入时通知工具栏按钮

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._boxes: list[BoxItem] = []
        self._add_mode = False
        self._add_start: QPointF | None = None
        self._add_temp: BoxItem | None = None
        self._panning = False
        self._pan_last: QPoint | None = None
        self._on_state_change: Callable[[BoxItem], None] | None = None
        self._scene.selectionChanged.connect(self._on_scene_selection)

    # ── 加载 ──
    def show_image(self, path: Path | None):
        self._scene.clear()
        self._pixmap_item = None
        self._boxes = []
        if path is None or not path.exists():
            return
        with Image.open(path) as src:
            pil_img = src.convert("RGB") if src.mode != "RGB" else src.copy()
        qimg = ImageQt(pil_img).copy()
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def load_boxes(self, records: list[BoxRecord],
                    on_state_change: Callable[[BoxItem], None] | None = None):
        """把 CSV 记录加成 LOCKED BoxItem。"""
        self._on_state_change = on_state_change
        for rec in records:
            box = BoxItem(rec.x1, rec.y1, rec.w, rec.h,
                          token=rec.token, on_state_change=on_state_change)
            self._scene.addItem(box)
            self._boxes.append(box)

    def boxes(self) -> list[BoxItem]:
        return list(self._boxes)

    def remove_box(self, box: BoxItem):
        if box in self._boxes:
            self._boxes.remove(box)
        self._scene.removeItem(box)

    def selected_box(self) -> BoxItem | None:
        for it in self._scene.selectedItems():
            if isinstance(it, BoxItem):
                return it
        return None

    # ── 新增框模式 ──
    def set_add_mode(self, enabled: bool):
        changed = enabled != self._add_mode
        self._add_mode = enabled
        if enabled:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            self._add_start = None
            if self._add_temp is not None:
                self._scene.removeItem(self._add_temp)
                self._add_temp = None
        if changed:
            self.add_mode_changed.emit(enabled)

    def is_add_mode(self) -> bool:
        return self._add_mode

    # ── 交互 ──
    def wheelEvent(self, event):
        # 滚轮缩放；图很大，滚动条意义不大
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def _is_pannable_at(self, view_pos: QPoint) -> bool:
        """空白处（或仅有底图 pixmap）才允许 LMB 平移；命中 BoxItem/手柄则交给 item。"""
        item = self.itemAt(view_pos)
        return item is None or item is self._pixmap_item

    def mousePressEvent(self, event: QMouseEvent):
        if self._add_mode and event.button() == Qt.MouseButton.LeftButton:
            self._add_start = self.mapToScene(event.position().toPoint())
            return
        if (event.button() == Qt.MouseButton.LeftButton
                and self._is_pannable_at(event.position().toPoint())):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._add_mode and self._add_start is not None:
            cur = self.mapToScene(event.position().toPoint())
            x1 = min(self._add_start.x(), cur.x())
            y1 = min(self._add_start.y(), cur.y())
            w = abs(cur.x() - self._add_start.x())
            h = abs(cur.y() - self._add_start.y())
            if self._add_temp is None:
                self._add_temp = BoxItem(x1, y1, max(w, 1), max(h, 1),
                                         token="(new)",
                                         on_state_change=self._on_state_change)
                self._add_temp.unlock()  # 新框直接进 UNLOCKED
                self._scene.addItem(self._add_temp)
            else:
                self._add_temp.setPos(x1, y1)
                self._add_temp.setRect(0, 0, max(w, 1), max(h, 1))
            return
        if self._panning and self._pan_last is not None:
            pos = event.position().toPoint()
            delta = pos - self._pan_last
            self._pan_last = pos
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._add_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._add_temp is not None and self._add_temp.rect().width() >= 4 \
                    and self._add_temp.rect().height() >= 4:
                self._boxes.append(self._add_temp)
                self._add_temp.setSelected(True)
            elif self._add_temp is not None:
                # 太小，丢弃
                self._scene.removeItem(self._add_temp)
            self._add_temp = None
            self._add_start = None
            self.set_add_mode(False)
            return
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_last = None
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _on_scene_selection(self):
        self.selection_changed.emit(self.selected_box())

    # ── 导出已确认文字的框 ──
    def committed_boxes(self) -> list[tuple[int, int, int, int, str]]:
        out = []
        for b in self._boxes:
            if b.state == BoxState.TEXT_COMMITTED:
                r = b.image_rect()
                out.append((
                    int(round(r.x())), int(round(r.y())),
                    int(round(r.width())), int(round(r.height())),
                    b.committed_text,
                ))
        return out
