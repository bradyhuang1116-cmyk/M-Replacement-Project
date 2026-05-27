"""可交互编号框 — 4 状态机 + 角拖拽手柄。

状态机
------
LOCKED         初始锁定，蓝色边框，不响应几何编辑
UNLOCKED       已解锁，红色边框 + 4 个角手柄，可拖动/改大小
BOX_CONFIRMED  几何已确认，橙色边框，等待输入文字
TEXT_COMMITTED 文字已确认，绿色边框 + 框内显示文字（保存时变成白底黑字）
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from PySide6.QtCore import QRectF, Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)


class BoxState(Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    BOX_CONFIRMED = "box_confirmed"
    TEXT_COMMITTED = "text_committed"


_STATE_COLORS = {
    BoxState.LOCKED: QColor("#2E86DE"),         # 蓝
    BoxState.UNLOCKED: QColor("#E74C3C"),       # 红
    BoxState.BOX_CONFIRMED: QColor("#F39C12"),  # 橙
    BoxState.TEXT_COMMITTED: QColor("#27AE60"), # 绿
}

HANDLE_SIZE = 10  # 角手柄边长（场景坐标 px）


class _CornerHandle(QGraphicsRectItem):
    """4 个角的拖拽手柄。父项是 BoxItem。"""

    TOP_LEFT = "tl"
    TOP_RIGHT = "tr"
    BOTTOM_LEFT = "bl"
    BOTTOM_RIGHT = "br"

    def __init__(self, corner: str, parent: "BoxItem"):
        super().__init__(parent)
        self._corner = corner
        self._parent_box = parent
        self.setRect(-HANDLE_SIZE / 2, -HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE)
        self.setBrush(QBrush(QColor("#FFFFFF")))
        self.setPen(QPen(QColor("#000000"), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setAcceptHoverEvents(True)
        cursor = {
            self.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            self.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            self.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            self.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
        }[corner]
        self.setCursor(cursor)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self._parent_box.handle_moved(self._corner, self.scenePos())
        return super().itemChange(change, value)


class BoxItem(QGraphicsRectItem):
    """主框。scene 坐标系 = 原图像素坐标系。"""

    def __init__(self, x: float, y: float, w: float, h: float,
                 token: str = "",
                 on_state_change: Callable[["BoxItem"], None] | None = None):
        super().__init__(QRectF(0, 0, max(w, 1), max(h, 1)))
        self.setPos(x, y)
        self._state = BoxState.LOCKED
        self._token = token
        self._committed_text = ""
        self._on_state_change = on_state_change
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        # 不允许 QGraphicsItem.ItemIsMovable 默认开着 —— 由状态控制
        self._update_movability()
        self._handles: dict[str, _CornerHandle] = {}
        self._handles_visible = False
        self._suppress_handle_sync = False

    # ── 状态访问 ──
    @property
    def state(self) -> BoxState:
        return self._state

    @property
    def token(self) -> str:
        return self._token

    @property
    def committed_text(self) -> str:
        return self._committed_text

    def image_rect(self) -> QRectF:
        """框在场景（=原图像素）坐标系下的 QRectF。"""
        return QRectF(self.pos(), self.rect().size())

    # ── 状态转换 ──
    def unlock(self):
        if self._state in (BoxState.LOCKED, BoxState.BOX_CONFIRMED,
                            BoxState.TEXT_COMMITTED):
            self._state = BoxState.UNLOCKED
            self._update_movability()
            self._sync_handles()
            self._notify_state_change()
            self.update()

    def confirm_box(self):
        if self._state == BoxState.UNLOCKED:
            self._state = BoxState.BOX_CONFIRMED
            self._update_movability()
            self._sync_handles()
            self._notify_state_change()
            self.update()

    def commit_text(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        if self._state in (BoxState.BOX_CONFIRMED, BoxState.TEXT_COMMITTED):
            self._committed_text = text
            self._state = BoxState.TEXT_COMMITTED
            self._update_movability()
            self._sync_handles()
            self._notify_state_change()
            self.update()

    def reset_to_locked(self):
        self._state = BoxState.LOCKED
        self._update_movability()
        self._sync_handles()
        self._notify_state_change()
        self.update()

    def _notify_state_change(self):
        if self._on_state_change:
            self._on_state_change(self)

    def _update_movability(self):
        movable = self._state == BoxState.UNLOCKED
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    # ── 角手柄 ──
    def _ensure_handles(self):
        if self._handles:
            return
        for corner in (
            _CornerHandle.TOP_LEFT,
            _CornerHandle.TOP_RIGHT,
            _CornerHandle.BOTTOM_LEFT,
            _CornerHandle.BOTTOM_RIGHT,
        ):
            self._handles[corner] = _CornerHandle(corner, self)

    def _sync_handles(self):
        """让 4 个手柄回到当前 rect 的 4 角。"""
        if self._state == BoxState.UNLOCKED:
            self._ensure_handles()
            self._handles_visible = True
        else:
            self._handles_visible = False

        if not self._handles:
            return
        r = self.rect()  # 局部坐标系
        positions = {
            _CornerHandle.TOP_LEFT: r.topLeft(),
            _CornerHandle.TOP_RIGHT: r.topRight(),
            _CornerHandle.BOTTOM_LEFT: r.bottomLeft(),
            _CornerHandle.BOTTOM_RIGHT: r.bottomRight(),
        }
        self._suppress_handle_sync = True
        try:
            for corner, h in self._handles.items():
                h.setVisible(self._handles_visible)
                h.setPos(positions[corner])
        finally:
            self._suppress_handle_sync = False

    def handle_moved(self, corner: str, scene_pos: QPointF):
        """被角手柄拖动时调整 rect。"""
        if self._suppress_handle_sync:
            return
        # scene_pos 是手柄的场景坐标；转到本 box 局部坐标
        local = self.mapFromScene(scene_pos)
        r = self.rect()
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        if corner == _CornerHandle.TOP_LEFT:
            x1, y1 = local.x(), local.y()
        elif corner == _CornerHandle.TOP_RIGHT:
            x2, y1 = local.x(), local.y()
        elif corner == _CornerHandle.BOTTOM_LEFT:
            x1, y2 = local.x(), local.y()
        elif corner == _CornerHandle.BOTTOM_RIGHT:
            x2, y2 = local.x(), local.y()
        # 维持最小尺寸 4px，且 x1<x2, y1<y2
        if x2 - x1 < 4:
            if corner in (_CornerHandle.TOP_LEFT, _CornerHandle.BOTTOM_LEFT):
                x1 = x2 - 4
            else:
                x2 = x1 + 4
        if y2 - y1 < 4:
            if corner in (_CornerHandle.TOP_LEFT, _CornerHandle.TOP_RIGHT):
                y1 = y2 - 4
            else:
                y2 = y1 + 4
        new_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        # 把局部矩形转回场景：保持 pos 不变，rect 改成 (0,0,w,h)，pos 平移
        scene_topleft = self.mapToScene(new_rect.topLeft())
        self.setPos(scene_topleft)
        self.setRect(0, 0, new_rect.width(), new_rect.height())
        self._sync_handles()
        self.update()

    # ── 绘制 ──
    def paint(self, painter: QPainter,
              option: QStyleOptionGraphicsItem,
              widget: QWidget | None = None):
        color = _STATE_COLORS[self._state]
        r = self.rect()

        if self._state == BoxState.TEXT_COMMITTED:
            painter.fillRect(r, QBrush(QColor("#FFFFFF")))
            # 渲染文字 —— 字号按 box 高度 60% 自适应
            text = self._committed_text or ""
            if text:
                font_size = max(int(r.height() * 0.55), 8)
                font = QFont()
                font.setPixelSize(font_size)
                font.setBold(True)
                painter.setFont(font)
                # 自适应缩字号确保宽度
                fm = painter.fontMetrics()
                while font_size > 8 and fm.horizontalAdvance(text) > r.width() * 0.92:
                    font_size -= 1
                    font.setPixelSize(font_size)
                    painter.setFont(font)
                    fm = painter.fontMetrics()
                painter.setPen(QPen(QColor("#000000")))
                painter.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

        pen_width = 2 if self._state == BoxState.LOCKED else 3
        pen = QPen(color, pen_width)
        pen.setCosmetic(True)  # 不随场景缩放变粗细
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        if self.isSelected():
            sel_pen = QPen(QColor("#FFD700"), 2, Qt.PenStyle.DashLine)
            sel_pen.setCosmetic(True)
            painter.setPen(sel_pen)
            painter.drawRect(r.adjusted(-2, -2, 2, 2))

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        if self._state == BoxState.UNLOCKED:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverEnterEvent(event)

    def itemChange(self, change, value):
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and not self._suppress_handle_sync):
            # 拖动整框时，手柄随父项自动移动（它们是子项）；这里只做边界刷新
            self._sync_handles()
        return super().itemChange(change, value)
