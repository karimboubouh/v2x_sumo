"""Map view panel: renders road network, vehicles, and V2V links."""

import pygame
import pygame.freetype

import config
from dashboard import theme


_font_cache = {}


def _make_font(size, bold=False, mono=False):
    """Return a cached pygame.freetype font, trying platform fonts then falling back."""
    key = (size, bold, mono)
    if key in _font_cache:
        return _font_cache[key]
    names = (
        ["menlo", "sfmonomedium", "couriernew", "dejavusansmono"]
        if mono
        else ["helveticaneue", "helvetica", "sfprodisplay", "arial", "dejavusans"]
    )
    for name in names:
        try:
            f = pygame.freetype.SysFont(name, size, bold=bold)
            if f:
                _font_cache[key] = f
                return f
        except Exception:
            pass
    f = pygame.freetype.SysFont(None, size, bold=bold)
    _font_cache[key] = f
    return f


# Distinct colors assigned to vehicles in round-robin order
_VEHICLE_PALETTE = [
    (255, 80, 80),  # red
    (80, 200, 255),  # cyan
    (80, 255, 120),  # green
    (255, 200, 50),  # yellow
    (200, 80, 255),  # purple
    (255, 140, 40),  # orange
    (40, 180, 180),  # teal
    (255, 100, 180),  # pink
    (160, 255, 80),  # lime
    (100, 140, 255),  # blue
    (255, 255, 100),  # light yellow
    (255, 60, 200),  # magenta
]


class MapView:
    """Top panel of the dashboard showing the road network and vehicles."""

    def __init__(self, rect, net_bounds, edge_shapes, dpi_scale=1.0):
        """
        Args:
            rect: pygame.Rect defining the panel area.
            net_bounds: (x_min, y_min, x_max, y_max) in SUMO coordinates.
            edge_shapes: list of edge shapes [(x,y), ...] for road rendering.
            dpi_scale: HiDPI scale factor (e.g. 2.0 on Retina displays).
        """
        self.rect = rect
        self.edge_shapes = edge_shapes
        self.dpi_scale = dpi_scale
        self.margin = int(20 * dpi_scale)

        x_min, y_min, x_max, y_max = net_bounds
        self.net_x_min = x_min
        self.net_y_min = y_min
        self.net_width = max(x_max - x_min, 1.0)
        self.net_height = max(y_max - y_min, 1.0)

        # Pre-compute base scaling (zoom=1 reference)
        draw_w = self.rect.width - 2 * self.margin
        draw_h = self.rect.height - 2 * self.margin
        scale_x = draw_w / self.net_width
        scale_y = draw_h / self.net_height
        self._base_scale = min(scale_x, scale_y)

        # Center the network in the panel (zoom=1 offsets)
        self._base_offset_x = self.rect.x + self.margin + (draw_w - self.net_width * self._base_scale) / 2
        self._base_offset_y = self.rect.y + self.margin + (draw_h - self.net_height * self._base_scale) / 2

        # Zoom and pan state
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Per-vehicle color assignment
        self._vehicle_colors = {}

        # Caches (invalidated on zoom/pan/theme change)
        self._road_surface = None
        self._road_cache_key = None
        self._icon_cache = {}          # color → high-res unrotated icon
        self._rotated_icon_cache = {}  # (color, q_angle, q_scale) → rotated surface
        self._speed_label_cache = {}   # (speed_int, font_size) → composite surface

    def world_to_screen(self, x, y):
        """Convert SUMO world coordinates to screen pixel coordinates."""
        bx = self._base_offset_x + (x - self.net_x_min) * self._base_scale
        by = self._base_offset_y + (self.net_height - (y - self.net_y_min)) * self._base_scale
        cx = self.rect.x + self.rect.width / 2
        cy = self.rect.y + self.rect.height / 2
        sx = cx + (bx - cx) * self._zoom + self._pan_x
        sy = cy + (by - cy) * self._zoom + self._pan_y
        return (int(sx), int(sy))

    def zoom_at(self, mx, my, factor):
        """Zoom in/out keeping the map point under (mx, my) fixed."""
        old_zoom = self._zoom
        new_zoom = max(0.3, min(30.0, self._zoom * factor))
        r = new_zoom / old_zoom
        cx = self.rect.x + self.rect.width / 2
        cy = self.rect.y + self.rect.height / 2
        self._pan_x = (mx - cx) * (1 - r) + r * self._pan_x
        self._pan_y = (my - cy) * (1 - r) + r * self._pan_y
        self._zoom = new_zoom
        self._rotated_icon_cache.clear()
        self._speed_label_cache.clear()

    def pan(self, dx, dy):
        """Shift the map view by (dx, dy) screen pixels."""
        self._pan_x += dx
        self._pan_y += dy

    def reset_view(self):
        """Reset zoom and pan to default."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.invalidate_caches()

    def invalidate_caches(self):
        """Clear all rendering caches (call on theme change, resize, etc.)."""
        self._road_surface = None
        self._road_cache_key = None
        self._rotated_icon_cache.clear()
        self._speed_label_cache.clear()
        self._icon_cache.clear()

    def _render_road_surface(self):
        """Render all roads to an off-screen surface (cached until zoom/pan changes)."""
        cache_key = (self._zoom, self._pan_x, self._pan_y, self.rect.width, self.rect.height)
        if self._road_surface is not None and self._road_cache_key == cache_key:
            return
        surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        surf.fill(theme.color("bg"))
        road_width = max(1, int(2 * self._zoom * self.dpi_scale))
        ox, oy = self.rect.x, self.rect.y
        for shape in self.edge_shapes:
            pts = [(sx - ox, sy - oy) for sx, sy in (self.world_to_screen(x, y) for x, y in shape)]
            if len(pts) >= 2:
                pygame.draw.lines(surf, theme.color("road"), False, pts, road_width)
        self._road_surface = surf
        self._road_cache_key = cache_key

    def draw(self, surface, vehicle_states, active_links, sim_time, scenario_name, paused=False):
        """Draw the map panel."""
        # Roads (cached off-screen surface)
        self._render_road_surface()
        surface.blit(self._road_surface, self.rect.topleft)

        # Draw V2V links
        for link in active_links:
            self._draw_link(surface, vehicle_states, link)

        # Draw vehicles
        for _veh_id, state in vehicle_states.items():
            self._draw_vehicle(surface, state)

        # Paused overlay
        if paused:
            font = _make_font(int(26 * self.dpi_scale), bold=True)
            surf, _ = font.render("⏸  PAUSED", theme.color("pause_label"))
            cx = self.rect.x + self.rect.width // 2
            cy = self.rect.y + self.rect.height // 2
            surface.blit(surf, surf.get_rect(center=(cx, cy)))

        # Draw HUD
        speeds = [s.speed * 3.6 for s in vehicle_states.values()]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        self._draw_hud(surface, sim_time, len(vehicle_states), avg_speed, scenario_name)

        # Bottom separator line
        sep_y = self.rect.y + self.rect.height - 1
        pygame.draw.line(
            surface,
            theme.color("separator"),
            (self.rect.x, sep_y),
            (self.rect.x + self.rect.width, sep_y),
        )

    # Internal canvas: large enough for clean downscaling via rotozoom
    _ICON_IW = 24  # body width
    _ICON_IH = 48  # body height
    _ICON_PAD = 4  # transparent padding so rotozoom doesn't clip corners

    def _get_vehicle_color(self, vehicle_id):
        """Return a persistent color for this vehicle, assigning one if new."""
        if vehicle_id not in self._vehicle_colors:
            idx = len(self._vehicle_colors) % len(_VEHICLE_PALETTE)
            self._vehicle_colors[vehicle_id] = _VEHICLE_PALETTE[idx]
        return self._vehicle_colors[vehicle_id]

    def _build_icon(self, color):
        """Render a clean minimal car icon at internal resolution."""
        IW, IH, PAD = self._ICON_IW, self._ICON_IH, self._ICON_PAD
        surf = pygame.Surface((IW + PAD * 2, IH + PAD * 2), pygame.SRCALPHA)
        bx, by = PAD, PAD
        r = 4  # corner radius

        # Body
        pygame.draw.rect(surf, color, (bx, by, IW, IH), border_radius=r)

        # Windshield strip at front (top 25%) — white tint to show direction
        wh = IH // 4
        wf = (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80))
        pygame.draw.rect(surf, wf, (bx + 3, by + 2, IW - 6, wh), border_radius=2)

        # Dark rear strip (bottom 20%)
        th = IH // 5
        dk = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
        pygame.draw.rect(surf, dk, (bx + 3, by + IH - th, IW - 6, th), border_radius=2)

        # White outline — 2 px for visibility after downscale
        pygame.draw.rect(surf, (255, 255, 255), (bx, by, IW, IH), 2, border_radius=r)

        return surf

    def _draw_vehicle(self, surface, state):
        """Draw a single vehicle icon with cached rotation and speed label."""
        color = self._get_vehicle_color(state.vehicle_id)

        # Icon scales with lane width, floor at dpi_scale
        lane_px = 3.2 * self._base_scale * self._zoom * self.dpi_scale
        s = max(self.dpi_scale, lane_px / 8.0)
        display_w = max(8, int(8 * s))

        # Get or build the high-res base icon for this color
        if color not in self._icon_cache:
            self._icon_cache[color] = self._build_icon(color)
        icon = self._icon_cache[color]

        # Quantize angle (3°) and scale (0.05) for cache key
        q_angle = round(state.angle / 3) * 3
        scale = display_w / self._ICON_IW
        q_scale = round(scale * 20)  # quantize to 0.05 steps
        rot_key = (color, q_angle, q_scale)

        if rot_key not in self._rotated_icon_cache:
            self._rotated_icon_cache[rot_key] = pygame.transform.rotozoom(icon, -q_angle, scale)
        rotated = self._rotated_icon_cache[rot_key]

        sx, sy = self.world_to_screen(state.x, state.y)
        surface.blit(rotated, rotated.get_rect(center=(sx, sy)))

        # Speed label — only when icon is large enough
        if display_w >= 20:
            speed_int = int(state.speed * 3.6)
            font_size = int(9 * self.dpi_scale)
            lbl_key = (speed_int, font_size)

            if lbl_key not in self._speed_label_cache:
                spd_font = _make_font(font_size)
                spd_surf, _ = spd_font.render(str(speed_int), (255, 255, 255))
                pad_x, pad_y = int(4 * self.dpi_scale), int(2 * self.dpi_scale)
                composite = pygame.Surface(
                    (spd_surf.get_width() + pad_x, spd_surf.get_height() + pad_y),
                    pygame.SRCALPHA,
                )
                composite.fill((0, 0, 0, 160))
                composite.blit(spd_surf, (pad_x // 2, pad_y // 2))
                self._speed_label_cache[lbl_key] = composite

            lbl = self._speed_label_cache[lbl_key]
            lbl_x = sx - lbl.get_width() // 2
            lbl_y = sy + rotated.get_height() // 2 + int(2 * self.dpi_scale)
            surface.blit(lbl, (lbl_x, lbl_y))

    def _draw_link(self, surface, vehicle_states, link):
        """Draw a V2V link as a colored line between two vehicles."""
        if link.sender_id not in vehicle_states or link.receiver_id not in vehicle_states:
            return

        state_a = vehicle_states[link.sender_id]
        state_b = vehicle_states[link.receiver_id]
        pos_a = self.world_to_screen(state_a.x, state_a.y)
        pos_b = self.world_to_screen(state_b.x, state_b.y)

        q = link.quality
        r = int(theme.color("link_strong")[0] * q + theme.color("link_weak")[0] * (1 - q))
        g = int(theme.color("link_strong")[1] * q + theme.color("link_weak")[1] * (1 - q))
        b = int(theme.color("link_strong")[2] * q + theme.color("link_weak")[2] * (1 - q))
        color = (min(255, r), min(255, g), min(255, b))

        pygame.draw.line(surface, color, pos_a, pos_b, max(1, int(self.dpi_scale)))

    def _draw_hud(self, surface, sim_time, vehicle_count, avg_speed, scenario_name):
        """Draw heads-up display: status bar + hints."""
        d = self.dpi_scale
        pad = int(8 * d)

        # --- Status bar — top-left (location · time · vehicles · avg speed · zoom) ---
        status_font = _make_font(int(14 * d), bold=False)
        status_text = (
            f"Road: {scenario_name}  |  Time: {sim_time:.0f}s"
            f"  |  Vehicles: {vehicle_count}  |  Avg speed: {avg_speed:.0f} km/h"
            f"  |  Zoom: {self._zoom:.1f}×"
        )
        s_surf, _ = status_font.render(status_text, theme.color("text"))
        surface.blit(s_surf, (self.rect.x + pad, self.rect.y + pad))

        # --- Hints — below status ---
        hint_font = _make_font(int(12 * d))
        hints = "Scroll: zoom  ·  Drag: pan  ·  R: reset  ·  Space: pause  ·  F11: fullscreen"
        h_surf, _ = hint_font.render(hints, theme.color("text_secondary"))
        surface.blit(h_surf, (self.rect.x + pad, self.rect.y + pad + s_surf.get_height() + int(3 * d)))
