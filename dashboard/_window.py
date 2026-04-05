"""MainWindow — top-level Qt window: menu bar, map/log splitter, status bar."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QAction, QKeySequence, QFont, QColor, QPainter
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QMenuBar, QMenu, QFrame, QLabel, QSizePolicy,
)

import config
from dashboard import theme
from dashboard.map_view import MapWidget
from dashboard.log_view import LogWidget
from dashboard.status_bar import StatusWidget


class _LoadingOverlay(QWidget):
    """Full-window translucent overlay shown until the first simulation frame."""

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.setInterval(420)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def sync_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        overlay_rect = parent.frameGeometry()
        screen = parent.screen()
        if screen is not None:
            overlay_rect = overlay_rect.intersected(screen.geometry())
        self.setGeometry(overlay_rect)
        self.raise_()

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Full-window dim
        bg = theme.color("overlay_bg")
        bg.setAlpha(220)
        painter.fillRect(self.rect(), bg)

        # Centred text
        text = "Loading graphics" + "." * self._dots
        font = QFont()
        font.setPointSize(18)
        font.setWeight(QFont.Medium)
        painter.setFont(font)
        painter.setPen(theme.color("accent"))
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        painter.end()

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()


class MainWindow(QMainWindow):
    closed_signal = Signal()

    def __init__(
        self,
        net_bounds: tuple,
        edge_shapes: list,
        scenario_name: str,
        dpi_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self._paused = False
        self._sim_done = False
        self._dpi = dpi_scale

        self.setWindowTitle(f"SUMO V2V Dashboard — {scenario_name}")
        self.setMinimumSize(QSize(800, 500))

        self._build_menu()
        self._build_central(net_bounds, edge_shapes, dpi_scale, scenario_name)
        self._apply_theme()

        # Loading overlay (covers the full top-level window, including the title-bar region)
        self._overlay = _LoadingOverlay(self)
        self._overlay.sync_to_parent()
        self._overlay.show()

    # ── Menu bar ─────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        act_reset = QAction("Reset View", self)
        act_reset.setShortcut(QKeySequence("R"))
        act_reset.triggered.connect(self._on_reset_view)
        file_menu.addAction(act_reset)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence("Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # View
        view_menu = mb.addMenu("&View")
        act_zin = QAction("Zoom In", self)
        act_zin.setShortcut(QKeySequence("+"))
        act_zin.triggered.connect(self._on_zoom_in)
        view_menu.addAction(act_zin)
        act_zout = QAction("Zoom Out", self)
        act_zout.setShortcut(QKeySequence("-"))
        act_zout.triggered.connect(self._on_zoom_out)
        view_menu.addAction(act_zout)
        view_menu.addSeparator()
        act_fs = QAction("Toggle Fullscreen", self)
        act_fs.setShortcut(QKeySequence("F11"))
        act_fs.triggered.connect(self._on_fullscreen)
        view_menu.addAction(act_fs)
        view_menu.addSeparator()
        act_theme = QAction("Toggle Theme", self)
        act_theme.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(act_theme)

        # Simulation
        sim_menu = mb.addMenu("&Simulation")
        act_pause = QAction("Pause / Resume", self)
        act_pause.setShortcut(QKeySequence("Space"))
        act_pause.triggered.connect(self._on_toggle_pause)
        sim_menu.addAction(act_pause)

    # ── Central widget ────────────────────────────────────────────────────────

    def _build_central(
        self,
        net_bounds: tuple,
        edge_shapes: list,
        dpi_scale: float,
        scenario_name: str = "",
    ) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Map + Log splitter
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setHandleWidth(4)
        self._splitter.setChildrenCollapsible(False)

        self._map_widget = MapWidget(net_bounds, edge_shapes, dpi_scale, scenario_name)
        self._log_widget = LogWidget(dpi_scale)

        self._splitter.addWidget(self._map_widget)
        self._splitter.addWidget(self._log_widget)
        self._splitter.setSizes([config.MAP_PANEL_HEIGHT, config.LOG_PANEL_HEIGHT])
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)

        layout.addWidget(self._splitter, stretch=1)

        # Status bar (custom widget, not QMainWindow.statusBar)
        self._status_widget = StatusWidget(dpi_scale)
        self._status_widget.setFixedHeight(int(config.STATUS_BAR_HEIGHT * dpi_scale))
        layout.addWidget(self._status_widget, stretch=0)

        self.setCentralWidget(container)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        bg = theme.color("bg")
        self.setStyleSheet(
            f"QSplitter::handle {{ background: rgb({theme.color('divider').red()},"
            f"{theme.color('divider').green()},{theme.color('divider').blue()}); }}"
        )

    # ── Public API (called by DashboardApp) ───────────────────────────────────

    def update_frame(
        self,
        vehicle_states: dict,
        active_links: list,
        new_messages: list,
        sim_time: float,
        training_status: dict | None = None,
        vehicle_overlays: dict | None = None,
        log_links: list | None = None,
    ) -> None:
        # Dismiss loading overlay on first real data
        if vehicle_states and self._overlay.isVisible():
            self._overlay.dismiss()

        self._map_widget.update_frame(
            vehicle_states or {},
            active_links or [],
            sim_time,
            vehicle_overlays or {},
        )
        if new_messages:
            self._log_widget.add_messages(
                new_messages,
                log_links if log_links is not None else active_links or [],
            )
        self._status_widget.update_status(training_status)

    def mark_done(self, overlay_text: str) -> None:
        self._sim_done = True
        self._map_widget.set_overlay(overlay_text)
        self._status_widget.mark_done()

    @property
    def paused(self) -> bool:
        return self._paused or self._sim_done

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_toggle_pause(self) -> None:
        if not self._sim_done:
            self._paused = not self._paused
            self._map_widget.set_paused(self._paused)

    def _on_reset_view(self) -> None:
        self._map_widget.reset_view()

    def _on_zoom_in(self) -> None:
        self._map_widget.zoom_in()

    def _on_zoom_out(self) -> None:
        self._map_widget.zoom_out()

    def _on_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_toggle_theme(self) -> None:
        theme.toggle()
        self._apply_theme()
        self._map_widget.on_theme_changed()
        self._log_widget.on_theme_changed()
        self._status_widget.update()

    # ── Qt overrides ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:  # noqa: N802
        k = event.key()
        if k in (Qt.Key_Escape,):
            self.close()
        elif k == Qt.Key_Space:
            self._on_toggle_pause()
        elif k == Qt.Key_F11:
            self._on_fullscreen()
        elif k == Qt.Key_R:
            self._on_reset_view()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self._on_zoom_in()
        elif k == Qt.Key_Minus:
            self._on_zoom_out()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_overlay") and self._overlay.isVisible():
            self._overlay.sync_to_parent()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        if hasattr(self, "_overlay") and self._overlay.isVisible():
            self._overlay.sync_to_parent()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed_signal.emit()
        super().closeEvent(event)
