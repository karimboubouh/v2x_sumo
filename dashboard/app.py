"""Dashboard application: Pygame-based split-screen for map and message log."""

import os

# Must be set before pygame.init() to enable HiDPI on macOS Retina displays
os.environ.setdefault('SDL_VIDEO_HIGHDPI', '1')

import pygame  # noqa: E402

import config
from dashboard.map_view import MapView
from dashboard.log_view import LogView
from dashboard.menu import MenuBar
from dashboard.status_bar import StatusBar
from dashboard import theme

_DIVIDER_HIT_RADIUS = 6
_MIN_MAP_H = 150
_MIN_LOG_H = 80


class DashboardApp:
    """Main dashboard window with map (top) and message log (bottom)."""

    def __init__(self, net_bounds, edge_shapes, scenario_name):
        self.scenario_name = scenario_name
        self.net_bounds = net_bounds
        self.edge_shapes = edge_shapes
        self._screen = None
        self._clock = None
        self._map_view = None
        self._log_view = None
        self._status_bar = None
        self._menu_bar = None
        self._initialized = False
        self._fullscreen = False
        self._drag_pos = None
        self._divider_y = None
        self._divider_dragging = False
        self._dpi_scale = 1.0
        self._input_scale_x = 1.0
        self._input_scale_y = 1.0
        self._paused = False
        self._simulation_done = False
        self._overlay_text = None
        self._pause_btn_rect = None
        self._theme_version = -1  # track theme changes

    @property
    def paused(self):
        return self._paused or self._simulation_done

    def mark_simulation_done(self, overlay_text="SIMULATION DONE"):
        """Freeze the simulation and display a centered completion message."""
        self._simulation_done = True
        self._overlay_text = overlay_text

    def _toggle_pause(self):
        if not self._simulation_done:
            self._paused = not self._paused

    def initialize(self):
        """Initialize Pygame and create the window."""
        theme.init()
        pygame.init()
        pygame.display.set_caption(f"SUMO V2V Dashboard - {self.scenario_name}")

        self._screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            pygame.RESIZABLE,
        )
        self._clock = pygame.time.Clock()
        surface_w, surface_h = self._refresh_display_metrics()

        self._menu_bar = MenuBar(self._dpi_scale)
        self._build_panels(surface_w, surface_h)
        self._initialized = True

    def _refresh_display_metrics(self):
        """Update drawable/window scale information for HiDPI displays."""
        surface_w, surface_h = self._screen.get_size()
        if hasattr(pygame.display, "get_window_size"):
            try:
                window_w, window_h = pygame.display.get_window_size()
            except Exception:
                window_w, window_h = surface_w, surface_h
        else:
            window_w, window_h = surface_w, surface_h

        window_w = max(window_w, 1)
        window_h = max(window_h, 1)
        self._input_scale_x = surface_w / window_w
        self._input_scale_y = surface_h / window_h
        self._dpi_scale = max(self._input_scale_x, self._input_scale_y, 1.0)
        return surface_w, surface_h

    def _scale_pos(self, pos):
        """Convert window-space input coordinates to drawable-surface space."""
        x, y = pos
        return (
            int(round(x * self._input_scale_x)),
            int(round(y * self._input_scale_y)),
        )

    def _scale_event(self, event):
        """Return an input event adjusted for drawable-surface coordinates."""
        if self._input_scale_x == 1.0 and self._input_scale_y == 1.0:
            return event

        if event.type not in (
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEMOTION,
        ):
            return event

        data = dict(event.dict)
        if "pos" in data:
            data["pos"] = self._scale_pos(data["pos"])
        if "rel" in data:
            data["rel"] = (
                int(round(data["rel"][0] * self._input_scale_x)),
                int(round(data["rel"][1] * self._input_scale_y)),
            )
        return pygame.event.Event(event.type, data)

    def _build_panels(self, w, h, preserve_view=False):
        """Create/recreate map and log panels for the given window size."""
        menu_h = self._menu_bar.height if self._menu_bar else 0
        status_h = int(config.STATUS_BAR_HEIGHT * self._dpi_scale)
        content_h = h - status_h  # area above the status bar

        if self._divider_y is None:
            ratio = config.MAP_PANEL_HEIGHT / (config.MAP_PANEL_HEIGHT + config.LOG_PANEL_HEIGHT)
            self._divider_y = menu_h + int((content_h - menu_h) * ratio)
        self._divider_y = max(menu_h + _MIN_MAP_H, min(self._divider_y, content_h - _MIN_LOG_H))

        map_h = self._divider_y - menu_h
        log_h = content_h - self._divider_y

        old_zoom = old_pan_x = old_pan_y = None
        old_log_state = None
        if preserve_view and self._map_view is not None:
            old_zoom = self._map_view._zoom
            old_pan_x = self._map_view._pan_x
            old_pan_y = self._map_view._pan_y
        if preserve_view and self._log_view is not None:
            old_log_state = self._log_view.export_state()

        map_rect = pygame.Rect(0, menu_h, w, map_h)
        log_rect = pygame.Rect(0, self._divider_y, w, log_h)
        status_rect = pygame.Rect(0, content_h, w, status_h)
        self._map_view = MapView(map_rect, self.net_bounds, self.edge_shapes, self._dpi_scale)
        self._log_view = LogView(log_rect, self._dpi_scale)
        self._status_bar = StatusBar(status_rect, self._dpi_scale)

        if old_zoom is not None:
            self._map_view._zoom = old_zoom
            self._map_view._pan_x = old_pan_x
            self._map_view._pan_y = old_pan_y
        if old_log_state is not None:
            self._log_view.restore_state(old_log_state)

    def _near_divider(self, my):
        return self._divider_y is not None and abs(my - self._divider_y) <= _DIVIDER_HIT_RADIUS

    def _handle_menu_action(self, action):
        """Dispatch menu item actions. Returns False to quit."""
        if action == "quit":
            return False
        if action == "toggle_pause":
            self._toggle_pause()
        elif action == "reset_view" and self._map_view:
            self._map_view.reset_view()
        elif action == "zoom_in" and self._map_view:
            cx = self._map_view.rect.centerx
            cy = self._map_view.rect.centery
            self._map_view.zoom_at(cx, cy, 1.15)
        elif action == "zoom_out" and self._map_view:
            cx = self._map_view.rect.centerx
            cy = self._map_view.rect.centery
            self._map_view.zoom_at(cx, cy, 1.0 / 1.15)
        elif action == "fullscreen":
            self._toggle_fullscreen()
        elif action == "toggle_theme":
            theme.toggle()
            if self._map_view:
                self._map_view.invalidate_caches()
            if self._log_view:
                self._log_view.invalidate_caches()
            if self._status_bar:
                self._status_bar.invalidate_caches()
        return True

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode(
                (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE
            )
        w, h = self._refresh_display_metrics()
        self._divider_y = None
        self._build_panels(w, h, preserve_view=True)

    def render(self, vehicle_states, active_links, new_messages, sim_time, training_status=None):
        """Render one frame. Returns False if user closed the window."""
        if not self._initialized:
            return False

        w, h = self._screen.get_size()

        for raw_event in pygame.event.get():
            event = self._scale_event(raw_event)
            if raw_event.type == pygame.QUIT:
                return False

            # Menu bar gets first pass on events
            if self._menu_bar:
                menu_action = self._menu_bar.handle_event(event)
                if menu_action:
                    if not self._handle_menu_action(menu_action):
                        return False
                    w, h = self._screen.get_size()
                    continue

            if self._log_view and self._log_view.handle_event(event):
                continue

            if raw_event.type == pygame.KEYDOWN:
                if raw_event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if raw_event.key == pygame.K_SPACE:
                    self._toggle_pause()
                if raw_event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                    w, h = self._screen.get_size()
                if raw_event.key == pygame.K_r and self._map_view:
                    self._map_view.reset_view()
                # Keyboard zoom: + / - / = (unshifted +)
                if raw_event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    if self._map_view:
                        cx = self._map_view.rect.centerx
                        cy = self._map_view.rect.centery
                        self._map_view.zoom_at(cx, cy, 1.15)
                if raw_event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if self._map_view:
                        cx = self._map_view.rect.centerx
                        cy = self._map_view.rect.centery
                        self._map_view.zoom_at(cx, cy, 1.0 / 1.15)

            if raw_event.type == pygame.VIDEORESIZE:
                self._divider_y = None
                w, h = self._refresh_display_metrics()
                self._build_panels(w, h, preserve_view=True)

            if raw_event.type == pygame.MOUSEWHEEL:
                mx, my = self._scale_pos(pygame.mouse.get_pos())
                if self._log_view and self._log_view.handle_wheel((mx, my), raw_event.y):
                    continue
                if self._map_view and self._map_view.rect.collidepoint(mx, my):
                    factor = 1.15 if raw_event.y > 0 else (1.0 / 1.15)
                    self._map_view.zoom_at(mx, my, factor)

            if raw_event.type == pygame.MOUSEBUTTONDOWN and raw_event.button == 1:
                mx, my = event.pos
                if self._pause_btn_rect and self._pause_btn_rect.collidepoint(mx, my):
                    self._toggle_pause()
                elif self._near_divider(my):
                    self._divider_dragging = True
                elif self._map_view and self._map_view.rect.collidepoint(mx, my):
                    self._drag_pos = event.pos

            if raw_event.type == pygame.MOUSEBUTTONUP and raw_event.button == 1:
                self._divider_dragging = False
                self._drag_pos = None

            if raw_event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self._divider_dragging:
                    menu_h = self._menu_bar.height if self._menu_bar else 0
                    status_h = int(config.STATUS_BAR_HEIGHT * self._dpi_scale)
                    content_h = h - status_h
                    self._divider_y = max(menu_h + _MIN_MAP_H, min(my, content_h - _MIN_LOG_H))
                    self._build_panels(w, h, preserve_view=True)
                elif self._drag_pos is not None:
                    dx = event.pos[0] - self._drag_pos[0]
                    dy = event.pos[1] - self._drag_pos[1]
                    if self._map_view:
                        self._map_view.pan(dx, dy)
                    self._drag_pos = event.pos

        # Cursor
        mx, my = self._scale_pos(pygame.mouse.get_pos())
        if self._divider_dragging or self._near_divider(my):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # Clear screen
        self._screen.fill(theme.color("bg"))

        # Draw panels
        self._map_view.draw(
            self._screen, vehicle_states, active_links,
            sim_time, self.scenario_name,
            paused=self._paused,
            overlay_text=self._overlay_text,
        )
        if new_messages:
            self._log_view.add_messages(new_messages, active_links)
        self._log_view.draw(self._screen)

        # Divider
        pygame.draw.rect(self._screen, theme.color("divider"),
                         (0, self._divider_y - 2, w, 4))

        # Status bar (bottom)
        if self._status_bar:
            self._status_bar.draw(self._screen, training_status=training_status)

        # Pause button (top-right, below menu)
        self._draw_pause_button(self._screen, w)

        # Menu bar (drawn last, on top)
        if self._menu_bar:
            self._menu_bar.draw(self._screen, w)

        pygame.display.flip()
        self._clock.tick(config.FPS)

        return True

    def _draw_pause_button(self, surface, w):
        """Draw play/pause toggle button in the top-right corner."""
        size = int(36 * self._dpi_scale)
        menu_h = self._menu_bar.height if self._menu_bar else 0
        margin = int(10 * self._dpi_scale)
        x = w - size - margin
        y = menu_h + margin
        self._pause_btn_rect = pygame.Rect(x, y, size, size)

        bg_color = (80, 60, 60) if self.paused else (60, 60, 80)
        pygame.draw.rect(surface, bg_color, self._pause_btn_rect, border_radius=6)
        pygame.draw.rect(surface, (120, 120, 160), self._pause_btn_rect, 1, border_radius=6)

        cx, cy = x + size // 2, y + size // 2
        ic = int(10 * self._dpi_scale)

        if self.paused:
            pts = [(cx - ic // 2, cy - ic), (cx - ic // 2, cy + ic), (cx + ic, cy)]
            pygame.draw.polygon(surface, (180, 220, 180), pts)
        else:
            bar_w = max(2, int(4 * self._dpi_scale))
            bar_h = int(14 * self._dpi_scale)
            pygame.draw.rect(surface, (220, 180, 180),
                             (cx - bar_w * 2, cy - bar_h // 2, bar_w, bar_h))
            pygame.draw.rect(surface, (220, 180, 180),
                             (cx + bar_w, cy - bar_h // 2, bar_w, bar_h))

    def cleanup(self):
        """Shut down Pygame."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
