"""Log view panel: scrolling message log for V2V communications."""

from collections import deque
import pygame

import config
from dashboard.fonts import get_font
from dashboard import theme
from event_stream import SimulationEvent


class LogView:
    """Bottom panel showing a scrolling log of V2V messages."""

    def __init__(self, rect, dpi_scale=1.0):
        self.rect = rect
        self.dpi_scale = dpi_scale
        self.max_lines = config.LOG_MAX_LINES if config.LOG_MAX_LINES and config.LOG_MAX_LINES > 0 else None
        # Each entry: (text, color, rendered_surface)
        self._lines = deque(maxlen=self.max_lines)
        self._header_surf = None
        self._font_size = int(config.FONT_SIZE * dpi_scale)
        self._scroll_offset = 0
        self._dragging_thumb = False
        self._thumb_drag_offset = 0
        self._thumb_rect = None
        self._track_rect = None

    def _ensure_header(self):
        """Build the cached header surface if needed."""
        if self._header_surf is None:
            hdr_font = get_font(self._font_size + int(self.dpi_scale), bold=True)
            self._header_surf, _ = hdr_font.render("V2V / DPL Interaction Log", theme.color("text"))

    def _line_height(self):
        return self._font_size + int(3 * self.dpi_scale)

    def _layout_metrics(self):
        """Return the current viewport, text, and scrollbar layout metrics."""
        self._ensure_header()
        pad = int(4 * self.dpi_scale)
        left_pad = int(10 * self.dpi_scale)
        right_pad = int(8 * self.dpi_scale)
        bottom_pad = int(8 * self.dpi_scale)
        line_height = self._line_height()

        sep_y = self.rect.y + self._header_surf.get_height() + pad * 2
        text_y = sep_y + pad
        track_w = max(int(10 * self.dpi_scale), 8)
        track_x = self.rect.right - track_w - right_pad
        track_h = max(self.rect.bottom - text_y - bottom_pad, line_height)
        track_rect = pygame.Rect(track_x, text_y, track_w, track_h)

        text_x = self.rect.x + left_pad
        text_w = max(track_x - text_x - right_pad, 20)
        text_rect = pygame.Rect(text_x, text_y, text_w, track_h)

        total_lines = len(self._lines)
        max_visible = max(1, track_h // line_height)
        max_offset = max(total_lines - max_visible, 0)
        self._scroll_offset = max(0, min(self._scroll_offset, max_offset))

        start = max(total_lines - max_visible - self._scroll_offset, 0)
        end = min(start + max_visible, total_lines)

        thumb_rect = None
        if max_offset > 0:
            thumb_h = max(int(track_h * (max_visible / max(total_lines, 1))), int(24 * self.dpi_scale))
            thumb_h = min(thumb_h, track_h)
            travel = max(track_h - thumb_h, 1)
            top_index = start
            normalized = top_index / max_offset if max_offset > 0 else 1.0
            thumb_y = track_rect.y + int(round(normalized * travel))
            thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.width, thumb_h)

        self._track_rect = track_rect
        self._thumb_rect = thumb_rect

        return {
            "pad": pad,
            "sep_y": sep_y,
            "line_height": line_height,
            "text_rect": text_rect,
            "track_rect": track_rect,
            "thumb_rect": thumb_rect,
            "total_lines": total_lines,
            "max_visible": max_visible,
            "max_offset": max_offset,
            "start": start,
            "end": end,
        }

    def _render_line(self, text, color):
        """Render one log line with the current font settings."""
        font = get_font(self._font_size, mono=True)
        rendered, _ = font.render(text, color)
        return rendered

    def export_state(self):
        """Return serializable state for preserving logs across panel rebuilds."""
        return {
            "lines": [(text, color) for text, color, _ in self._lines],
            "scroll_offset": self._scroll_offset,
        }

    def restore_state(self, state):
        """Restore cached log lines and scroll position after a rebuild."""
        self._lines = deque(maxlen=self.max_lines)
        lines = state.get("lines", [])
        if self.max_lines is not None:
            lines = lines[-self.max_lines:]
        for text, color in lines:
            self._lines.append((text, color, self._render_line(text, color)))
        self._scroll_offset = int(state.get("scroll_offset", 0))

    def add_messages(self, messages, active_links=None):
        """Add new messages to the log, pre-rendering text surfaces."""
        stick_to_bottom = self._scroll_offset == 0
        dist_map = {}
        qual_map = {}
        if active_links:
            for link in active_links:
                key = tuple(sorted([link.sender_id, link.receiver_id]))
                dist_map[key] = link.distance
                qual_map[key] = link.quality

        font = get_font(self._font_size, mono=True)

        for msg in messages:
            if isinstance(msg, SimulationEvent):
                if msg.category == "link":
                    color = theme.color("log_link")
                elif msg.category == "weight":
                    color = theme.color("log_weight")
                elif msg.category == "training":
                    color = theme.color("log_training")
                elif msg.category == "warning":
                    color = theme.color("log_warning")
                else:
                    color = theme.color("log_status")

                line_text = f"[{msg.timestamp:6.0f}s] {msg.text}"
            else:
                key = tuple(sorted([msg.sender_id, msg.receiver_id]))
                dist = dist_map.get(key, 0.0)
                qual = qual_map.get(key, 0.0)

                if msg.msg_type == "hello":
                    color = theme.color("log_hello")
                elif msg.msg_type in {"fl_weights", "dl_weights"}:
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

        if stick_to_bottom:
            self._scroll_offset = 0
        else:
            self._scroll_offset = min(self._scroll_offset, len(self._lines) - 1)

    def scroll_lines(self, delta):
        """Scroll the log by a number of lines. Positive delta scrolls upward."""
        metrics = self._layout_metrics()
        if metrics["max_offset"] <= 0:
            self._scroll_offset = 0
            return False
        new_offset = max(0, min(self._scroll_offset + delta, metrics["max_offset"]))
        changed = new_offset != self._scroll_offset
        self._scroll_offset = new_offset
        return changed

    def handle_wheel(self, mouse_pos, wheel_y):
        """Handle mouse-wheel scrolling over the log panel."""
        if not self.rect.collidepoint(mouse_pos):
            return False
        return self.scroll_lines(int(wheel_y) * 3)

    def handle_event(self, event):
        """Handle scrollbar click/drag interactions."""
        metrics = self._layout_metrics()
        thumb_rect = metrics["thumb_rect"]
        track_rect = metrics["track_rect"]

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if thumb_rect and thumb_rect.collidepoint(event.pos):
                self._dragging_thumb = True
                self._thumb_drag_offset = event.pos[1] - thumb_rect.y
                return True
            if track_rect.collidepoint(event.pos) and thumb_rect:
                page = max(metrics["max_visible"] - 1, 1)
                if event.pos[1] < thumb_rect.y:
                    self.scroll_lines(page)
                elif event.pos[1] > thumb_rect.bottom:
                    self.scroll_lines(-page)
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging_thumb:
                self._dragging_thumb = False
                return True

        if event.type == pygame.MOUSEMOTION and self._dragging_thumb and thumb_rect:
            travel = max(track_rect.height - thumb_rect.height, 1)
            new_thumb_y = event.pos[1] - self._thumb_drag_offset
            new_thumb_y = max(track_rect.y, min(new_thumb_y, track_rect.bottom - thumb_rect.height))
            normalized = (new_thumb_y - track_rect.y) / travel
            top_index = int(round(normalized * metrics["max_offset"]))
            self._scroll_offset = max(metrics["max_offset"] - top_index, 0)
            return True

        return False

    def invalidate_caches(self):
        """Re-render all cached surfaces (call on theme change)."""
        self._header_surf = None
        font = get_font(self._font_size, mono=True)
        new_lines = deque(maxlen=self.max_lines)
        for text, color, _ in self._lines:
            rendered, _ = font.render(text, color)
            new_lines.append((text, color, rendered))
        self._lines = new_lines

    def draw(self, surface):
        """Draw the log panel."""
        metrics = self._layout_metrics()
        pad = metrics["pad"]

        # Background
        pygame.draw.rect(surface, theme.color("log_bg"), self.rect)

        # Header (cached)
        surface.blit(self._header_surf, (self.rect.x + int(10 * self.dpi_scale), self.rect.y + pad))

        # Separator
        pygame.draw.line(
            surface, theme.color("separator"),
            (self.rect.x, metrics["sep_y"]),
            (self.rect.x + self.rect.width, metrics["sep_y"]),
        )

        # Message lines — blit pre-rendered surfaces inside the text viewport
        old_clip = surface.get_clip()
        surface.set_clip(metrics["text_rect"])
        visible = list(self._lines)[metrics["start"]:metrics["end"]]
        x = metrics["text_rect"].x
        y = metrics["text_rect"].y

        for _text, _color, rendered in visible:
            surface.blit(rendered, (x, y))
            y += metrics["line_height"]
        surface.set_clip(old_clip)

        # Scrollbar
        if metrics["thumb_rect"] is not None:
            pygame.draw.rect(surface, theme.color("status_bar_bg"), metrics["track_rect"], border_radius=4)
            thumb_color = theme.color("menu_hover") if self._dragging_thumb else theme.color("text_secondary")
            pygame.draw.rect(surface, thumb_color, metrics["thumb_rect"], border_radius=4)
