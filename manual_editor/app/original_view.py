"""右上只读原图视图。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)


class OriginalView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pixmap_item: QGraphicsPixmapItem | None = None

    def show_image(self, path: Path | None):
        self._scene.clear()
        self._pixmap_item = None
        if path is None or not path.exists():
            placeholder = "原图未找到" if path is None else f"原图未找到：\n{path.name}"
            txt = QGraphicsTextItem(placeholder)
            txt.setDefaultTextColor(Qt.GlobalColor.gray)
            self._scene.addItem(txt)
            self.setSceneRect(txt.boundingRect())
            return
        with Image.open(path) as src:
            pil_img = src.convert("RGB") if src.mode != "RGB" else src.copy()
        qimg = ImageQt(pil_img).copy()
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item = self._scene.addPixmap(pix)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        # 滚轮直接缩放（已在 __init__ 设 ScrollHandDrag，LMB 拖动即平移）
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
