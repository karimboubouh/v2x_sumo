"""LogWidget — compact, colour-coded scrolling interaction log."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer, QRect, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QSizePolicy,
)

import config
from dashboard import theme


# ── Log entry ─────────────────────────────────────────────────────────────────

class _Entry:
    __slots__ = ("text", "color")

    def __init__(self, text: str, color: QColor) -> None:
        self.text  = text
        self.color = color


# ── Custom scroll area that renders log lines ─────────────────────────────────

class _LogCanvas(QWidget):
    """Custom widget that paints all log lines with a native-style scrollbar."""

    _LINE_SPACING = 2  # extra px between lines

    def __init__(self, dpi_scale: float, parent=None) -> None:
        super().__init__(parent)
        self._dpi = dpi_scale
        self._entries: deque[_Entry] = deque(
            maxlen=config.LOG_MAX_LINES if config.LOG_MAX_LINES and config.LOG_MAX_LINES > 0 else None
        )
        self._scroll_offset = 0   # lines scrolled up from bottom
        self._dragging = False
        self._drag_start_y = 0
        self._drag_start_offset = 0

        self.setFont(self._make_font())
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _make_font(self) -> QFont:
        f = QFont("Menlo, Consolas, Courier New, monospace")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(int(config.FONT_SIZE_LOG * 0.9))
        return f

    def on_theme_changed(self) -> None:
        self.update()

    # ── Data ─────────────────────────────────────────────────────────────────

    def add_entries(self, entries: list[_Entry]) -> None:
        at_bottom = self._scroll_offset == 0
        self._entries.extend(entries)
        if at_bottom:
            self._scroll_offset = 0
        else:
            # keep relative position
            self._scroll_offset = min(self._scroll_offset, len(self._entries) - 1)
        self.update()

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _line_height(self) -> int:
        fm = QFontMetrics(self.font())
        return fm.height() + self._LINE_SPACING

    def _scrollbar_rect(self):
        from PySide6.QtCore import QRect
        w = int(8 * self._dpi)
        return QRect(self.width() - w - 2, 0, w, self.height())

    def _thumb_rect(self, sr, total, visible, max_offset):
        from PySide6.QtCore import QRect
        if max_offset <= 0:
            return None
        ratio = visible / max(total, 1)
        thumb_h = max(int(sr.height() * ratio), int(24 * self._dpi))
        thumb_h = min(thumb_h, sr.height())
        travel  = max(sr.height() - thumb_h, 1)
        # scroll_offset 0 = bottom; max_offset = top
        top_frac = (max_offset - self._scroll_offset) / max_offset
        thumb_y = sr.y() + int(top_frac * travel)
        return QRect(sr.x(), thumb_y, sr.width(), thumb_h)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)

        # Background
        painter.fillRect(self.rect(), theme.color("log_bg"))

        lh = self._line_height()
        n_total   = len(self._entries)
        sr        = self._scrollbar_rect()
        text_w    = sr.x() - 6
        n_visible = max(1, self.height() // lh)
        max_off   = max(n_total - n_visible, 0)
        self._scroll_offset = max(0, min(self._scroll_offset, max_off))

        start = max(n_total - n_visible - self._scroll_offset, 0)
        end   = min(start + n_visible, n_total)

        visible = list(self._entries)[start:end]

        painter.setFont(self.font())
        y = 4
        for entry in visible:
            painter.setPen(entry.color)
            # Clip text to text area
            painter.drawText(4, y + lh - 3, entry.text)
            y += lh

        # Scrollbar track
        if max_off > 0:
            track_bg = theme.color_alpha("status_bar_bg", 120)
            painter.fillRect(sr, track_bg)
            thumb = self._thumb_rect(sr, n_total, n_visible, max_off)
            if thumb:
                tc = theme.color("text_secondary") if self._dragging else theme.color("text_dim")
                painter.setBrush(QBrush(tc))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(thumb, 3, 3)

        painter.end()

    # ── Wheel / drag ──────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        step  = max(1, abs(delta) // 40)
        if delta > 0:
            self._scroll_offset = min(self._scroll_offset + step,
                                      max(len(self._entries) - 1, 0))
        else:
            self._scroll_offset = max(self._scroll_offset - step, 0)
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            sr = self._scrollbar_rect()
            if sr.contains(event.pos()):
                self._dragging = True
                self._drag_start_y = event.pos().y()
                self._drag_start_offset = self._scroll_offset
                self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging:
            return
        lh = self._line_height()
        n_total   = len(self._entries)
        n_visible = max(1, self.height() // lh)
        max_off   = max(n_total - n_visible, 0)
        if max_off == 0:
            return
        sr     = self._scrollbar_rect()
        thumb  = self._thumb_rect(sr, n_total, n_visible, max_off)
        if thumb is None:
            return
        travel = max(sr.height() - thumb.height(), 1)
        dy     = event.pos().y() - self._drag_start_y
        delta  = int(dy / travel * max_off)
        self._scroll_offset = max(0, min(self._drag_start_offset - delta, max_off))
        self.update()


# ── Header bar (fully custom-painted — no stylesheet conflicts) ───────────────

class _LogHeader(QWidget):
    """Paints its own background, title, and collapse arrow — no child widgets."""

    toggled = Signal(bool)   # emits True when opening, False when closing

    def __init__(self, dpi_scale: float, parent=None) -> None:
        super().__init__(parent)
        self._open  = True
        self._hover = False
        self._btn_w = int(28 * dpi_scale)
        self.setFixedHeight(int(22 * dpi_scale))
        self.setCursor(Qt.ArrowCursor)
        self._font = QFont()
        self._font.setPointSize(9)
        self._font.setWeight(QFont.DemiBold)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), theme.color("status_bar_bg"))
        p.setFont(self._font)
        # Title
        p.setPen(theme.color("text_secondary"))
        title_rect = QRect(10, 0, self.width() - self._btn_w - 10, self.height())
        p.drawText(title_rect, Qt.AlignVCenter | Qt.AlignLeft,
                   "V2V / DPL Interaction Log")
        # Arrow — brighter on hover
        p.setPen(theme.color("text") if self._hover else theme.color("text_dim"))
        btn_rect = QRect(self.width() - self._btn_w, 0, self._btn_w, self.height())
        p.drawText(btn_rect, Qt.AlignCenter, "▼" if self._open else "▶")
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._open = not self._open
            self.toggled.emit(self._open)
            self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def on_theme_changed(self) -> None:
        self.update()


# ── Public widget ─────────────────────────────────────────────────────────────

class LogWidget(QWidget):
    """Compact log panel with a header, separator, and colour-coded messages."""

    def __init__(self, dpi_scale: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self._dpi = dpi_scale
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header — fully custom-painted, no child widgets, no stylesheet issues
        self._header = _LogHeader(self._dpi)
        self._header.toggled.connect(self._toggle_log)
        layout.addWidget(self._header)

        # Thin separator
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.HLine)
        self._sep.setFixedHeight(1)
        layout.addWidget(self._sep)

        # Canvas
        self._canvas = _LogCanvas(self._dpi)
        layout.addWidget(self._canvas, stretch=1)

        self._apply_colors()

    def _toggle_log(self, opening: bool) -> None:
        self._canvas.setVisible(opening)
        self._sep.setVisible(opening)
        self.setMaximumHeight(16777215 if opening else self._header.height())

    def _apply_colors(self) -> None:
        sep = theme.color("separator")
        self.setStyleSheet(
            f"QFrame {{ background: rgb({sep.red()},{sep.green()},{sep.blue()}); }}"
        )

    def on_theme_changed(self) -> None:
        self._apply_colors()
        self._header.on_theme_changed()
        self._canvas.on_theme_changed()

    # ── Public ───────────────────────────────────────────────────────────────

    def add_messages(self, messages: list, active_links=None) -> None:
        from event_stream import SimulationEvent  # deferred to avoid torch at startup
        dist_map: dict = {}
        qual_map: dict = {}
        if active_links:
            for link in active_links:
                key = tuple(sorted([link.sender_id, link.receiver_id]))
                dist_map[key] = getattr(link, "distance", 0.0)
                qual_map[key] = getattr(link, "quality",  0.0)

        entries: list[_Entry] = []
        for msg in messages:
            if isinstance(msg, SimulationEvent):
                cat = msg.category
                if   cat == "link":     c = theme.color("log_link")
                elif cat == "weight":   c = theme.color("log_weight")
                elif cat == "training": c = theme.color("log_training")
                elif cat == "warning":  c = theme.color("log_warning")
                else:                   c = theme.color("log_status")
                text = f"[{msg.timestamp:6.0f}s] {msg.text}"
            else:
                key  = tuple(sorted([msg.sender_id, msg.receiver_id]))
                dist = dist_map.get(key, 0.0)
                qual = qual_map.get(key, 0.0)
                mt   = msg.msg_type
                if   mt == "hello":                   c = theme.color("log_hello")
                elif mt in {"fl_weights","dl_weights"}: c = theme.color("log_fl")
                else:                                  c = theme.color("log_data")
                text = (
                    f"[{msg.timestamp:6.0f}s] "
                    f"{msg.sender_id:>8s} -> {msg.receiver_id:<8s} : "
                    f"{mt:<10s} (d={dist:5.0f}m, q={qual:.2f})"
                )
            entries.append(_Entry(text, c))

        self._canvas.add_entries(entries)
