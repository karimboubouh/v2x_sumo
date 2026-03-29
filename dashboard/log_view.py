"""Log view panel: scrolling message log for V2V communications."""

from collections import deque
import pygame
import pygame.freetype

import config
from dashboard import theme


_font_cache = {}


def _make_font(size, bold=False):
    """Return a cached pygame.freetype monospace font."""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for name in ["menlo", "sfmonomedium", "couriernew", "dejavusansmono"]:
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


class LogView:
    """Bottom panel showing a scrolling log of V2V messages."""

    def __init__(self, rect, dpi_scale=1.0):
        self.rect = rect
        self.dpi_scale = dpi_scale
        self.max_lines = config.LOG_MAX_LINES
        # Each entry: (text, color, rendered_surface)
        self._lines = deque(maxlen=self.max_lines)
        self._header_surf = None
        self._font_size = int(config.FONT_SIZE * dpi_scale)

    def add_messages(self, messages, active_links=None):
        """Add new messages to the log, pre-rendering text surfaces."""
        dist_map = {}
        qual_map = {}
        if active_links:
            for link in active_links:
                key = tuple(sorted([link.sender_id, link.receiver_id]))
                dist_map[key] = link.distance
                qual_map[key] = link.quality

        font = _make_font(self._font_size)

        for msg in messages:
            key = tuple(sorted([msg.sender_id, msg.receiver_id]))
            dist = dist_map.get(key, 0.0)
            qual = qual_map.get(key, 0.0)

            if msg.msg_type == "hello":
                color = theme.color("log_hello")
            elif msg.msg_type == "fl_weights":
                color = theme.color("log_fl")
            else:
                color = theme.color("log_data")

            line_text = (
                f"[{msg.timestamp:6.0f}s] "
                f"{msg.sender_id:>8s} -> {msg.receiver_id:<8s} : "
                f"{msg.msg_type:<10s} "
                f"(d={dist:5.0f}m, q={qual:.2f})"
            )
            rendered, _ = font.render(line_text, color)
            self._lines.append((line_text, color, rendered))

    def invalidate_caches(self):
        """Re-render all cached surfaces (call on theme change)."""
        self._header_surf = None
        font = _make_font(self._font_size)
        new_lines = deque(maxlen=self.max_lines)
        for text, color, _ in self._lines:
            rendered, _ = font.render(text, color)
            new_lines.append((text, color, rendered))
        self._lines = new_lines

    def draw(self, surface):
        """Draw the log panel."""
        pad = int(4 * self.dpi_scale)

        # Background
        pygame.draw.rect(surface, theme.color("log_bg"), self.rect)

        # Header (cached)
        if self._header_surf is None:
            hdr_font = _make_font(self._font_size, bold=True)
            self._header_surf, _ = hdr_font.render("V2V Message Log", theme.color("text"))
        surface.blit(self._header_surf, (self.rect.x + int(10 * self.dpi_scale), self.rect.y + pad))

        # Separator
        sep_y = self.rect.y + self._header_surf.get_height() + pad * 2
        pygame.draw.line(
            surface, theme.color("separator"),
            (self.rect.x, sep_y),
            (self.rect.x + self.rect.width, sep_y),
        )

        # Message lines — blit pre-rendered surfaces
        line_height = self._font_size + int(3 * self.dpi_scale)
        available_height = self.rect.height - self._header_surf.get_height() - int(18 * self.dpi_scale)
        max_visible = max(1, available_height // line_height)

        visible = list(self._lines)[-max_visible:]
        x = self.rect.x + int(10 * self.dpi_scale)
        y = sep_y + pad

        for _text, _color, rendered in visible:
            surface.blit(rendered, (x, y))
            y += line_height
