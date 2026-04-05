"""MapWidget — QGraphicsView-based road network, vehicle, and link renderer."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QFont, QFontMetrics, QPolygonF, QTransform, QLinearGradient, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QFrame,
)

import config
from dashboard import theme

# Mirror of algorithms/base.py constants — avoids importing torch at dashboard startup
LINK_SIDELINK = 0.0
LINK_INTERNET = 1.0


# ── Vehicle colour palette ────────────────────────────────────────────────────

_PALETTE = [
    (255,  80,  80),  # red
    ( 80, 200, 255),  # cyan
    ( 80, 255, 120),  # green
    (255, 200,  50),  # yellow
    (200,  80, 255),  # purple
    (255, 140,  40),  # orange
    ( 40, 180, 180),  # teal
    (255, 100, 180),  # pink
    (160, 255,  80),  # lime
    (100, 140, 255),  # blue
    (255, 255, 100),  # light yellow
    (255,  60, 200),  # magenta
]

_BYZ_FILL    = QColor(210, 50, 50)
_BYZ_OUTLINE = QColor(255, 120, 120)


# ── Scene layers ──────────────────────────────────────────────────────────────

class _RoadsLayer(QGraphicsItem):
    """Immutable road network drawn as a single QPainterPath."""

    def __init__(self, edge_shapes: list, net_bounds: tuple) -> None:
        super().__init__()
        # Cache the rasterised road network. Without this, every frame re-draws
        # the full QPainterPath (thousands of segments), which dominates frame time.
        # The cache is invalidated only when the view zoom changes, which is
        # already reduced to one repaint per zoom step by NoViewportUpdate in _scale_view.
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        x_min, y_min, x_max, y_max = net_bounds
        self._bounds = QRectF(x_min - 200, -y_max - 200,
                              (x_max - x_min) + 400, (y_max - y_min) + 400)
        self._path = QPainterPath()
        for shape in edge_shapes:
            if len(shape) < 2:
                continue
            self._path.moveTo(shape[0][0], -shape[0][1])
            for x, y in shape[1:]:
                self._path.lineTo(x, -y)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter: QPainter, option, widget) -> None:
        # Edge shadow
        edge_pen = QPen(theme.color("road_edge"), 5.0)
        edge_pen.setCapStyle(Qt.RoundCap)
        edge_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(edge_pen)
        painter.drawPath(self._path)
        # Road surface
        road_pen = QPen(theme.color("road"), 3.0)
        road_pen.setCapStyle(Qt.RoundCap)
        road_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(road_pen)
        painter.drawPath(self._path)


class _LinksLayer(QGraphicsItem):
    """All V2V + FL collaboration links — redrawn each frame."""

    def __init__(self) -> None:
        super().__init__()
        self.setZValue(1)
        self._links: list = []
        self._states: dict = {}
        self._INF = QRectF(-1e8, -1e8, 2e8, 2e8)

    def update_data(self, links: list, states: dict) -> None:
        self._links = links
        self._states = states
        self.update()

    def boundingRect(self) -> QRectF:
        return self._INF

    def paint(self, painter: QPainter, option, widget) -> None:
        if not self._links or not self._states:
            return
        painter.setRenderHint(QPainter.Antialiasing)

        # Zoom-aware opacity: Hermite smoothstep, 0 at scale≤0.3 px/m → 1 at scale≥2.5 px/m.
        # At fit-to-view (~0.43 px/m) zoom_t≈0.01 so links are nearly invisible;
        # at 4× zoom (~1.7 px/m) zoom_t≈0.75 — clearly readable.
        scale = painter.worldTransform().m11()
        _t = max(0.0, min(1.0, (scale - 0.3) / 2.2))
        zoom_t = _t * _t * (3.0 - 2.0 * _t)

        for link in self._links:
            s = self._states.get(link.sender_id)
            r = self._states.get(link.receiver_id)
            if s is None or r is None:
                continue
            x1, y1 = s.x, -s.y
            x2, y2 = r.x, -r.y

            offset = getattr(link, "parallel_offset", 0.0)
            if offset:
                dx, dy = x2 - x1, y2 - y1
                ln = math.hypot(dx, dy) + 1e-9
                nx, ny = -dy / ln, dx / ln
                shift = offset * 4.0
                x1 += nx * shift; y1 += ny * shift
                x2 += nx * shift; y2 += ny * shift

            if hasattr(link, "alpha"):
                # FL collaboration link — more prominent than V2V at all zoom levels
                alpha = float(link.alpha)
                base = theme.color("link_sidelink" if link.link_type == LINK_SIDELINK
                                   else "link_internet")
                c = QColor(base)
                c.setAlphaF(min(0.90, 0.15 + alpha * 0.18 + zoom_t * (0.32 + alpha * 0.25)))
                pen = QPen(c, max(0.8, 0.7 + alpha * 1.1))  # 0.80–1.80 px (was 1.0–2.8)
                pen.setCosmetic(True)
                if link.link_type == LINK_INTERNET:
                    pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
            else:
                # V2V sidelink — always blue; quality modulates opacity only
                q = float(getattr(link, "quality", 0.5))
                c = QColor(theme.color("link_sidelink"))
                c.setAlphaF(min(0.90, 0.40 + q * 0.25 + zoom_t * (0.15 + q * 0.10)))
                pen = QPen(c, 1.0)
                pen.setCosmetic(True)
                painter.setPen(pen)

            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


class _VehiclesLayer(QGraphicsItem):
    """All vehicles drawn in a single paint pass — always as rectangles."""

    # Car body in world-metres, pointing north (−Y in scene)
    _CAR_W = 2.4   # width
    _CAR_H = 5.0   # length
    _HALF_W = _CAR_W / 2
    _HALF_H = _CAR_H / 2

    def __init__(self) -> None:
        super().__init__()
        self.setZValue(2)
        self._states: dict  = {}
        self._overlays: dict = {}
        self._color_map: dict[str, QColor] = {}
        self._INF = QRectF(-1e8, -1e8, 2e8, 2e8)

    def update_data(self, states: dict, overlays: dict) -> None:
        self._states  = states
        self._overlays = overlays
        self.update()

    def _vehicle_color(self, vid: str) -> QColor:
        if vid not in self._color_map:
            idx = len(self._color_map) % len(_PALETTE)
            self._color_map[vid] = QColor(*_PALETTE[idx])
        return self._color_map[vid]

    def boundingRect(self) -> QRectF:
        return self._INF

    def paint(self, painter: QPainter, option, widget) -> None:
        if not self._states:
            return
        painter.setRenderHint(QPainter.Antialiasing)

        # Effective pixels per world-unit at current zoom
        scale = painter.worldTransform().m11()
        car_px = self._CAR_W * scale

        # Whether to show speed text labels
        show_labels = scale > 12.0

        for vid, state in self._states.items():
            meta = self._overlays.get(vid, {})
            is_byz   = bool(meta.get("byzantine", False))
            training = bool(meta.get("training_active", False))

            fill_color    = _BYZ_FILL    if is_byz else self._vehicle_color(vid)
            outline_color = _BYZ_OUTLINE if is_byz else fill_color.lighter(150)

            painter.save()
            painter.translate(state.x, -state.y)
            # SUMO angle: 0=north CW. Scene has Y negated, so rotation is correct.
            painter.rotate(state.angle)

            # Always draw a rectangle — clamp to minimum pixel size
            w = max(self._HALF_W, 2.5 / scale)   # minimum 5 px total width
            h = max(self._HALF_H, 5.0 / scale)   # minimum 10 px total height
            body = QRectF(-w, -h, w * 2, h * 2)

            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(outline_color, 0.25))
            painter.drawRoundedRect(body, min(w * 0.4, 0.6 / scale), min(w * 0.4, 0.6 / scale))

            # Detail overlays only when large enough to distinguish
            if car_px >= 6:
                # Windshield tint (front = −Y direction)
                wf = QColor(
                    min(255, fill_color.red()   + 70),
                    min(255, fill_color.green() + 70),
                    min(255, fill_color.blue()  + 70),
                )
                windshield = QRectF(-w * 0.7, -h, w * 2 * 0.7, h * 0.5)
                painter.setBrush(QBrush(wf))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(windshield, w * 0.3, w * 0.3)

                # Rear tint (bottom = +Y direction)
                dk = QColor(
                    max(0, fill_color.red()   - 50),
                    max(0, fill_color.green() - 50),
                    max(0, fill_color.blue()  - 50),
                )
                rear = QRectF(-w * 0.7, h * 0.6, w * 2 * 0.7, h * 0.4)
                painter.setBrush(QBrush(dk))
                painter.drawRoundedRect(rear, w * 0.2, w * 0.2)

            # Training dot — always visible, centred on car
            if training:
                dot_r = max(self._CAR_W * 0.28, 0.7 / scale)
                painter.setBrush(QBrush(theme.color("log_training")))
                painter.setPen(QPen(theme.color("bg"), 0.2))
                painter.drawEllipse(QPointF(0, 0), dot_r, dot_r)

            painter.restore()

            if show_labels:
                speed_kmh = state.speed * 3.6
                lbl = f"{speed_kmh:.0f}"
                font = QFont()
                font.setPointSizeF(self._CAR_W * 1.4)
                painter.setFont(font)
                painter.setPen(theme.color("text"))
                painter.drawText(QPointF(state.x + self._HALF_W + 0.3, -state.y + 1.0), lbl)


# ── Main widget ───────────────────────────────────────────────────────────────

class MapWidget(QGraphicsView):
    """High-quality map view with smooth zoom, pan, HUD, legend, and zoom controls."""

    def __init__(
        self,
        net_bounds: tuple,
        edge_shapes: list,
        dpi_scale: float = 1.0,
        scenario_name: str = "",
    ) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene)

        self._net_bounds = net_bounds
        self._dpi = dpi_scale
        self._scenario_name = scenario_name
        self._sim_time: float = 0.0
        self._vehicle_count: int = 0
        self._avg_speed: float = 0.0
        self._paused: bool = False
        self._overlay_text: str | None = None
        self._vehicle_states: dict = {}
        self._active_links: list = []
        self._base_transform: QTransform | None = None
        self._zoom_btn_rects: dict[str, QRectF] = {}

        # Static overlay pixmap caches — rebuilt only on theme change
        self._legend_px: QPixmap | None = None
        self._kb_px: QPixmap | None = None

        # Pre-allocated fonts (avoids QFont() allocation on every paint call)
        self._font_hud_lbl = QFont(); self._font_hud_lbl.setPointSize(config.FONT_SIZE_MAP)
        self._font_hud_val = QFont(); self._font_hud_val.setPointSize(config.FONT_SIZE_MAP + 1)
        self._font_hud_val.setWeight(QFont.Medium)
        self._fm_hud_lbl = QFontMetrics(self._font_hud_lbl)
        self._fm_hud_val = QFontMetrics(self._font_hud_val)

        # Configure view
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing,
        )
        # NoAnchor: we handle cursor-locking ourselves in _scale_view()
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # DontSavePainterState is safe; DontAdjustForAntialiasing removed (causes jagged zoom)
        self.setOptimizationFlags(QGraphicsView.DontSavePainterState)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setInteractive(False)

        # Scene
        x_min, y_min, x_max, y_max = net_bounds
        scene_rect = QRectF(x_min, -y_max, x_max - x_min, y_max - y_min)
        self._scene.setSceneRect(scene_rect)
        self._scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        # Layers
        self._roads    = _RoadsLayer(edge_shapes, net_bounds)
        self._links    = _LinksLayer()
        self._vehicles = _VehiclesLayer()
        self._scene.addItem(self._roads)
        self._scene.addItem(self._links)
        self._scene.addItem(self._vehicles)

        self.on_theme_changed()

        # Fit after the widget is shown
        QTimer.singleShot(50, self._fit_initial)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_frame(
        self,
        vehicle_states: dict,
        active_links: list,
        sim_time: float,
        vehicle_overlays: dict,
    ) -> None:
        self._vehicle_states = vehicle_states
        self._active_links   = active_links
        self._sim_time       = sim_time
        if vehicle_states:
            self._vehicle_count = len(vehicle_states)
            speeds = [v.speed * 3.6 for v in vehicle_states.values()]
            self._avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        self._vehicles.update_data(vehicle_states, vehicle_overlays)
        self._links.update_data(active_links, vehicle_states)
        self.viewport().update()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.viewport().update()

    def set_overlay(self, text: str | None) -> None:
        self._overlay_text = text
        self.viewport().update()

    def zoom_in(self)  -> None: self._scale_view(1.12,        self.viewport().rect().center())
    def zoom_out(self) -> None: self._scale_view(1.0 / 1.12, self.viewport().rect().center())

    def _scale_view(self, factor: float, anchor_vp) -> None:
        """Scale by *factor*, keeping the viewport point *anchor_vp* fixed in scene space.

        Uses NoViewportUpdate during the compound transform so that scale() and the
        two scrollbar adjustments don't each enqueue a separate paint event — only
        one explicit repaint is issued at the very end.
        """
        new_zoom = self._zoom_level() * factor
        if not (0.2 <= new_zoom <= 50.0):
            return
        scene_pt = self.mapToScene(anchor_vp)
        # Suppress intermediate repaints during the compound transform
        self.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)
        self.scale(factor, factor)
        new_vp = self.mapFromScene(scene_pt)
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + (new_vp.x() - anchor_vp.x())
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + (new_vp.y() - anchor_vp.y())
        )
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.viewport().update()  # single repaint for the whole compound transform

    def reset_view(self) -> None:
        if self._base_transform:
            self.setTransform(self._base_transform)
        else:
            if self.scene() is not None:
                self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self.viewport().update()

    def on_theme_changed(self) -> None:
        self.setBackgroundBrush(QBrush(theme.color("bg")))
        self._legend_px = None   # invalidate static caches
        self._kb_px = None
        self.viewport().update()

    # ── Zoom level helper ─────────────────────────────────────────────────────

    def _zoom_level(self) -> float:
        if self._base_transform is None:
            return 1.0
        base = self._base_transform.m11()
        if base == 0:
            return 1.0
        return self.transform().m11() / base

    # ── Fit on first show ─────────────────────────────────────────────────────

    def _fit_initial(self) -> None:
        if self.scene() is None:
            return
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._base_transform = QTransform(self.transform())

    # ── Events ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            pos = QPointF(event.pos())
            actions = {
                "zoom_in":  self.zoom_in,
                "zoom_out": self.zoom_out,
                "reset":    self.reset_view,
            }
            for name, rect in self._zoom_btn_rects.items():
                if rect.contains(pos):
                    actions[name]()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        # Zoom anchored to the cursor position
        self._scale_view(factor, event.position().toPoint())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._base_transform is None:
            QTimer.singleShot(50, self._fit_initial)

    # ── Foreground: HUD + zoom controls + alpha labels + legend ──────────────

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.resetTransform()
        vp = self.viewport().rect()
        self._draw_hud(painter, vp)
        self._draw_keybindings(painter, vp)
        self._draw_alpha_labels(painter, vp)
        self._draw_zoom_controls(painter, vp)
        self._draw_legend(painter, vp)
        self._draw_range_rings(painter)
        if self._paused:
            self._draw_center_text(painter, vp, "⏸  PAUSED")
        if self._overlay_text:
            self._draw_done_overlay(painter, vp)

    # ── Range rings ───────────────────────────────────────────────────────────

    def _draw_range_rings(self, painter: QPainter) -> None:
        zoom = self._zoom_level()
        if zoom < 1.5 or self._vehicle_count > 30:
            return
        ring_color = theme.color_alpha("log_data", 22)
        pen = QPen(ring_color, 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        r = float(config.COMM_RANGE)
        for state in self._vehicle_states.values():
            sp = self.mapFromScene(QPointF(state.x, -state.y))
            rp = self.mapFromScene(QPointF(state.x + r, -state.y))
            r_px = (rp - sp).x()
            if r_px > 5:
                painter.drawEllipse(sp, r_px, r_px)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, painter: QPainter, vp) -> None:
        pad = 12
        line_h = 19

        rows = [
            ("Scenario",   self._scenario_name),
            ("Time",       f"{self._sim_time:.0f} s"),
            ("Vehicles",   f"{self._vehicle_count}"),
            ("Avg speed",  f"{self._avg_speed:.0f} km/h"),
            ("Zoom",       f"{self._zoom_level():.1f}×"),
        ]

        # Use pre-allocated fonts — no QFont() allocation per frame
        fm_lbl = self._fm_hud_lbl
        fm_val = self._fm_hud_val

        lbl_col = max(fm_lbl.horizontalAdvance(lbl + ":") for lbl, _ in rows) + 10

        x_lbl = pad
        x_val = pad + lbl_col
        col_dim  = theme.color("text_dim")
        col_text = theme.color("text")
        for i, (lbl, val) in enumerate(rows):
            y = pad + 10 + i * line_h
            painter.setFont(self._font_hud_lbl)
            painter.setPen(col_dim)
            painter.drawText(QPointF(x_lbl, y), lbl + ":")
            painter.setFont(self._font_hud_val)
            painter.setPen(col_text)
            painter.drawText(QPointF(x_val, y), val)

    # ── Keyboard hints (bottom-left, no background) ───────────────────────────

    def _draw_keybindings(self, painter: QPainter, vp) -> None:
        """Render keyboard hint strip bottom-left. Content is cached as a QPixmap."""
        if self._kb_px is None:
            self._kb_px = self._build_kb_pixmap()
        y = vp.bottom() - self._kb_px.height() / self._kb_px.devicePixelRatioF() - 12
        painter.drawPixmap(12, int(y), self._kb_px)

    def _build_kb_pixmap(self) -> QPixmap:
        hints = [
            ("R",      "Reset view"),
            ("Space",  "Pause / Resume"),
            ("+ / −",  "Zoom in / out"),
            ("F11",    "Fullscreen"),
            ("Q",      "Quit"),
        ]
        line_h = 16
        font_key = QFont(); font_key.setPointSize(config.FONT_SIZE_MAP - 1); font_key.setWeight(QFont.Bold)
        font_txt = QFont(); font_txt.setPointSize(config.FONT_SIZE_MAP - 1)
        fm_key = QFontMetrics(font_key)
        fm_txt = QFontMetrics(font_txt)
        key_col = max(fm_key.horizontalAdvance(k) for k, _ in hints) + 8
        desc_w  = max(fm_txt.horizontalAdvance(d) for _, d in hints)
        w = key_col + desc_w + 4
        h = len(hints) * line_h

        dpr = self.devicePixelRatioF()
        px = QPixmap(int(w * dpr), int(h * dpr))
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.TextAntialiasing)
        col_key = theme.color("text_secondary")
        col_dsc = theme.color("text_dim")
        for i, (key, desc) in enumerate(hints):
            y = i * line_h + fm_key.ascent()
            p.setFont(font_key)
            p.setPen(col_key)
            p.drawText(QPointF(0, y), key)
            p.setFont(font_txt)
            p.setPen(col_dsc)
            p.drawText(QPointF(key_col, y), desc)
        p.end()
        return px

    # ── Alpha labels ──────────────────────────────────────────────────────────

    def _draw_alpha_labels(self, painter: QPainter, vp) -> None:
        if self._zoom_level() < 4.0 or not self._vehicle_states or not self._active_links:
            return
        font = QFont()
        font.setPointSize(config.FONT_SIZE_MAP - 2)
        fm = QFontMetrics(font)
        painter.setFont(font)
        vp_rect = QRectF(vp)
        for link in self._active_links:
            if not hasattr(link, "alpha"):
                continue
            s = self._vehicle_states.get(link.sender_id)
            r = self._vehicle_states.get(link.receiver_id)
            if s is None or r is None:
                continue
            mid = self.mapFromScene(QPointF((s.x + r.x) / 2, -(s.y + r.y) / 2))
            if not vp_rect.contains(QPointF(mid)):
                continue
            text = f"α {link.alpha:.2f}"
            tw = fm.horizontalAdvance(text)
            bg = theme.color_alpha("surface", 190)
            painter.fillRect(QRectF(mid.x() + 4, mid.y() - 11, tw + 6, 14), bg)
            painter.setPen(theme.color("text_secondary"))
            painter.drawText(QPointF(mid.x() + 7, mid.y()), text)

    # ── Zoom controls + scenario name (top-right, no background) ────────────────

    def _draw_zoom_controls(self, painter: QPainter, vp) -> None:
        btn_size = 28
        gap = 4
        pad = 12

        zoom_text = f"{self._zoom_level():.1f}×"
        font = QFont()
        font.setPointSize(config.FONT_SIZE_MAP)
        fm = QFontMetrics(font)
        label_w = fm.horizontalAdvance(zoom_text) + 14

        # Total strip: [−] [gap] [label] [gap] [+] [gap] [⌂]
        strip_w = btn_size + gap + label_w + gap + btn_size + gap + btn_size
        # Position: right side, just below HUD area
        x0 = vp.right() - strip_w - pad
        y0 = pad - 4  # align top with HUD top

        bg = theme.color_alpha("surface", 210)
        border = theme.color("border")

        def _draw_btn(x, w, label, name):
            rect = QRectF(x, y0, w, btn_size)
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.0))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(theme.color("text"))
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, label)
            if name:
                self._zoom_btn_rects[name] = rect

        cx = x0
        _draw_btn(cx, btn_size, "−", "zoom_out");  cx += btn_size + gap
        _draw_btn(cx, label_w, zoom_text, None);    cx += label_w + gap
        _draw_btn(cx, btn_size, "+", "zoom_in");    cx += btn_size + gap
        _draw_btn(cx, btn_size, "⌂", "reset")

    # ── Legend ────────────────────────────────────────────────────────────────

    def _draw_legend(self, painter: QPainter, vp) -> None:
        """Blit the legend pixmap (built once, cached until theme changes)."""
        if self._legend_px is None:
            self._legend_px = self._build_legend_pixmap()
        dpr = self._legend_px.devicePixelRatioF()
        x = vp.right()  - self._legend_px.width()  / dpr - 12
        y = vp.bottom() - self._legend_px.height() / dpr - 12
        painter.drawPixmap(int(x), int(y), self._legend_px)

    def _build_legend_pixmap(self) -> QPixmap:
        """Render the full legend into a transparent pixmap (origin = 0,0)."""
        pad = 12
        row_h = 22
        swatch = 28
        gap = 8
        section_gap = 8

        link_rows = [
            ("link_sidelink", False, "5G Sidelink"),
            ("link_internet",  True,  "Internet"),
        ]
        veh_rows = [
            (theme.color("log_training"), "dot",  "Training active"),
            (_BYZ_FILL,                   "rect", "Byzantine"),
        ]

        font_sm = QFont(); font_sm.setPointSize(config.FONT_SIZE_MAP - 1); font_sm.setWeight(QFont.Bold)
        font_nm = QFont(); font_nm.setPointSize(config.FONT_SIZE_MAP)
        fm = QFontMetrics(font_nm)
        all_labels = ([r[2] for r in link_rows] + ["V2V (quality)"] +
                      [r[2] for r in veh_rows] + ["Width ∝ α"])
        max_lbl = max(fm.horizontalAdvance(l) for l in all_labels)

        legend_w = int(pad + swatch + gap + max_lbl + pad)
        legend_h = int(pad + 14
                       + len(link_rows) * row_h
                       + row_h                    # V2V gradient
                       + section_gap + 14         # VEHICLES header
                       + len(veh_rows) * row_h
                       + row_h                    # α gradient
                       + pad)

        dpr = self.devicePixelRatioF()
        px = QPixmap(int(legend_w * dpr), int(legend_h * dpr))
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)

        # Background
        p.setBrush(QBrush(theme.color_alpha("surface", 210)))
        p.setPen(QPen(theme.color("border"), 1.0))
        p.drawRoundedRect(QRectF(0, 0, legend_w, legend_h), 10, 10)

        cx = pad
        ty = pad

        # ── LINKS section ──
        p.setFont(font_sm)
        p.setPen(theme.color("text_dim"))
        p.drawText(QPointF(cx, ty + 11), "LINKS")
        ty += 16

        p.setFont(font_nm)
        for color_name, dashed, label in link_rows:
            c = theme.color(color_name)
            mid_y = ty + row_h / 2
            pen = QPen(c, 2.2)
            pen.setCosmetic(True)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(cx, mid_y), QPointF(cx + swatch, mid_y))
            p.setPen(theme.color("text"))
            p.drawText(QPointF(cx + swatch + gap, mid_y + 4), label)
            ty += row_h

        # V2V quality gradient
        mid_y = ty + row_h / 2
        wc = theme.color("link_weak")
        sc = theme.color("link_strong")
        seg = max(swatch // 5, 1)
        for i in range(5):
            t = i / 4.0
            c = QColor(int(wc.red()*(1-t)+sc.red()*t),
                       int(wc.green()*(1-t)+sc.green()*t),
                       int(wc.blue()*(1-t)+sc.blue()*t))
            pen = QPen(c, 2.2); pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(QPointF(cx + i*seg, mid_y), QPointF(min(cx+(i+1)*seg, cx+swatch), mid_y))
        p.setFont(font_nm); p.setPen(theme.color("text"))
        p.drawText(QPointF(cx + swatch + gap, mid_y + 4), "V2V (quality)")
        ty += row_h

        # ── VEHICLES section ──
        ty += section_gap
        p.setFont(font_sm)
        p.setPen(theme.color("text_dim"))
        p.drawText(QPointF(cx, ty + 11), "VEHICLES")
        ty += 16

        p.setFont(font_nm)
        for c, shape, label in veh_rows:
            mid_y = ty + row_h / 2
            p.setBrush(QBrush(c))
            p.setPen(Qt.NoPen)
            if shape == "dot":
                p.drawEllipse(QPointF(cx + swatch / 2, mid_y), 5.0, 5.0)
            else:
                p.drawRoundedRect(QRectF(cx + 2, mid_y - 6, swatch - 4, 12), 2, 2)
            p.setPen(theme.color("text"))
            p.drawText(QPointF(cx + swatch + gap, mid_y + 4), label)
            ty += row_h

        # α gradient row
        mid_y = ty + row_h / 2
        seg = max(swatch // 4, 1)
        for i in range(5):
            a = i / 4.0
            c = theme.color("link_sidelink"); c.setAlphaF(0.35 + a * 0.65)
            pen = QPen(c, max(1.0, 0.8 + a * 2.2)); pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(QPointF(cx + i*seg, mid_y), QPointF(min(cx+(i+1)*seg, cx+swatch), mid_y))
        p.setFont(font_nm); p.setPen(theme.color("text"))
        p.drawText(QPointF(cx + swatch + gap, mid_y + 4), "Width ∝ α")

        p.end()
        return px

    # ── Center overlays ───────────────────────────────────────────────────────

    def _draw_center_text(self, painter: QPainter, vp, text: str) -> None:
        font = QFont()
        font.setPointSize(config.FONT_SIZE_MAP + 11)
        font.setWeight(QFont.Bold)
        painter.setFont(font)
        painter.setPen(theme.color("pause_label"))
        painter.drawText(vp, Qt.AlignCenter, text)

    def _draw_done_overlay(self, painter: QPainter, vp) -> None:
        dim = theme.color_alpha("overlay_bg", 160)
        painter.fillRect(vp, dim)
        font = QFont()
        font.setPointSize(config.FONT_SIZE_MAP + 13)
        font.setWeight(QFont.Bold)
        painter.setFont(font)
        painter.setPen(theme.color("accent"))
        painter.drawText(vp, Qt.AlignCenter, self._overlay_text or "DONE")
