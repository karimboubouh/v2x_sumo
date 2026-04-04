"""DashboardApp — public interface: same contract as the old pygame version."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import config
from dashboard import theme
from dashboard._window import MainWindow


class DashboardApp:
    """
    Main dashboard window.

    Public API (unchanged from the pygame backend):
      initialize()
      render(vehicle_states, active_links, new_messages, sim_time, ...)  → bool
      mark_simulation_done(overlay_text)
      cleanup()
      paused  (property)
    """

    def __init__(
        self,
        net_bounds: tuple,
        edge_shapes: list,
        scenario_name: str,
    ) -> None:
        self._net_bounds   = net_bounds
        self._edge_shapes  = edge_shapes
        self._scenario_name = scenario_name
        self._app: QApplication | None = None
        self._window: MainWindow | None = None
        self._closed = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the QApplication and main window."""
        # High-DPI: must be set before QApplication
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setApplicationName("SUMO V2V Dashboard")
        self._app.setOrganizationName("SUMO")

        # App icon — prefer .icns on macOS, fall back to .png
        _icon_dir = os.path.join(os.path.dirname(__file__), "icons")
        for _name in ("app.icns", "app.png"):
            _icon_path = os.path.join(_icon_dir, _name)
            if os.path.isfile(_icon_path):
                self._app.setWindowIcon(QIcon(_icon_path))
                break

        theme.init(getattr(config, "THEME_MODE", "system"))

        dpi = float(getattr(config, "DPI_SCALE", 1.0))

        self._window = MainWindow(
            net_bounds   = self._net_bounds,
            edge_shapes  = self._edge_shapes,
            scenario_name = self._scenario_name,
            dpi_scale    = dpi,
        )
        self._window.closed_signal.connect(self._on_closed)
        self._window.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self._window.show()

    # ── Per-frame render ──────────────────────────────────────────────────────

    def render(
        self,
        vehicle_states,
        active_links,
        new_messages,
        sim_time: float,
        training_status=None,
        vehicle_overlays=None,
        log_links=None,
    ) -> bool:
        """
        Push one simulation frame to the dashboard and process Qt events.
        Returns False when the user has closed the window.
        """
        if self._closed or self._window is None:
            return False

        self._window.update_frame(
            vehicle_states  = vehicle_states  or {},
            active_links    = active_links    or [],
            new_messages    = new_messages    or [],
            sim_time        = sim_time,
            training_status = training_status,
            vehicle_overlays= vehicle_overlays or {},
            log_links       = log_links,
        )

        self._app.processEvents()
        return not self._closed

    # ── Simulation done ───────────────────────────────────────────────────────

    def mark_simulation_done(self, overlay_text: str = "SIMULATION DONE") -> None:
        if self._window:
            self._window.mark_done(overlay_text)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._app:
            self._app.processEvents()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def paused(self) -> bool:
        if self._window:
            return self._window.paused
        return False

    # ── Internal ─────────────────────────────────────────────────────────────

    def _on_closed(self) -> None:
        self._closed = True
