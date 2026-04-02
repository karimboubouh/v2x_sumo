"""In-app menu bar for the SUMO V2V Dashboard."""

import pygame

from dashboard.fonts import get_font
from dashboard import theme


# Separator sentinel
_SEP = "---"


class MenuBar:
    """Lightweight in-app menu bar with dropdown support."""

    def __init__(self, dpi_scale=1.0):
        self.dpi_scale = dpi_scale
        self._bar_h = int(26 * dpi_scale)
        self._font_size = int(14 * dpi_scale)
        self._open_menu = None  # index of open dropdown, or None

        # Menu structure
        self._menus = [
            {
                "label": "File",
                "items": [
                    {"label": "Reset View", "action": "reset_view", "shortcut": "R"},
                    _SEP,
                    {"label": "Quit", "action": "quit", "shortcut": "Q"},
                ],
            },
            {
                "label": "View",
                "items": [
                    {"label": "Zoom In", "action": "zoom_in", "shortcut": "+"},
                    {"label": "Zoom Out", "action": "zoom_out", "shortcut": "-"},
                    _SEP,
                    {"label": "Fullscreen", "action": "fullscreen", "shortcut": "F11"},
                    _SEP,
                    {"label": "Toggle Theme", "action": "toggle_theme", "shortcut": ""},
                ],
            },
            {
                "label": "Simulation",
                "items": [
                    {"label": "Pause / Resume", "action": "toggle_pause", "shortcut": "Space"},
                ],
            },
        ]

        # Pre-compute label positions
        self._label_rects = []

    @property
    def height(self):
        return self._bar_h

    def handle_event(self, event):
        """Process a pygame event. Returns an action string or None."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if my <= self._bar_h:
                # Click on menu bar — toggle dropdown
                for i, rect in enumerate(self._label_rects):
                    if rect.collidepoint(mx, my):
                        self._open_menu = None if self._open_menu == i else i
                        return None
                self._open_menu = None
                return None

            # Click in open dropdown
            if self._open_menu is not None:
                action = self._hit_test_dropdown(mx, my)
                self._open_menu = None
                return action

            # Click elsewhere closes menu
            self._open_menu = None

        if event.type == pygame.MOUSEMOTION and self._open_menu is not None:
            mx, my = event.pos
            if my <= self._bar_h:
                for i, rect in enumerate(self._label_rects):
                    if rect.collidepoint(mx, my):
                        self._open_menu = i
                        break

        return None

    def _hit_test_dropdown(self, mx, my):
        """Check if (mx, my) is on a dropdown item. Returns action or None."""
        if self._open_menu is None:
            return None
        menu = self._menus[self._open_menu]
        rect = self._label_rects[self._open_menu]
        d = self.dpi_scale
        item_h = int(24 * d)
        sep_h = int(10 * d)
        drop_x = rect.x
        y = self._bar_h

        for item in menu["items"]:
            if item == _SEP:
                y += sep_h
                continue
            item_rect = pygame.Rect(drop_x, y, int(200 * d), item_h)
            if item_rect.collidepoint(mx, my):
                return item["action"]
            y += item_h
        return None

    def draw(self, surface, width):
        """Draw the menu bar and any open dropdown."""
        d = self.dpi_scale
        bg = theme.color("menu_bg")
        text_c = theme.color("menu_text")
        border_c = theme.color("menu_border")
        hover_c = theme.color("menu_hover")

        # Bar background
        pygame.draw.rect(surface, bg, (0, 0, width, self._bar_h))
        pygame.draw.line(surface, border_c, (0, self._bar_h - 1), (width, self._bar_h - 1))

        font = get_font(self._font_size)
        x = int(12 * d)
        self._label_rects = []

        for i, menu in enumerate(self._menus):
            label = menu["label"]
            lbl_surf, lbl_rect = font.render(label, text_c)
            pad_x = int(12 * d)
            total_w = lbl_rect.width + pad_x * 2
            lbl_area = pygame.Rect(x, 0, total_w, self._bar_h)
            self._label_rects.append(lbl_area)

            if self._open_menu == i:
                pygame.draw.rect(surface, hover_c, lbl_area)

            cy = (self._bar_h - lbl_rect.height) // 2
            surface.blit(lbl_surf, (x + pad_x, cy))
            x += total_w

        # Draw open dropdown
        if self._open_menu is not None:
            self._draw_dropdown(surface, self._open_menu)

    def _draw_dropdown(self, surface, menu_idx):
        """Draw a dropdown panel for the given menu index."""
        d = self.dpi_scale
        menu = self._menus[menu_idx]
        rect = self._label_rects[menu_idx]
        bg = theme.color("menu_bg")
        text_c = theme.color("menu_text")
        border_c = theme.color("menu_border")
        hover_c = theme.color("menu_hover")
        secondary_c = theme.color("text_secondary")

        item_h = int(24 * d)
        sep_h = int(10 * d)
        drop_w = int(200 * d)
        drop_x = rect.x

        # Compute total height
        total_h = 0
        for item in menu["items"]:
            total_h += sep_h if item == _SEP else item_h

        # Background + border
        drop_rect = pygame.Rect(drop_x, self._bar_h, drop_w, total_h)
        pygame.draw.rect(surface, bg, drop_rect)
        pygame.draw.rect(surface, border_c, drop_rect, 1)

        font = get_font(self._font_size)
        shortcut_font = get_font(int(12 * d), mono=True)
        mx, my = pygame.mouse.get_pos()
        y = self._bar_h

        for item in menu["items"]:
            if item == _SEP:
                sep_y = y + sep_h // 2
                pygame.draw.line(surface, border_c, (drop_x + int(8 * d), sep_y),
                                 (drop_x + drop_w - int(8 * d), sep_y))
                y += sep_h
                continue

            item_rect = pygame.Rect(drop_x, y, drop_w, item_h)
            if item_rect.collidepoint(mx, my):
                pygame.draw.rect(surface, hover_c, item_rect)

            lbl_surf, _ = font.render(item["label"], text_c)
            surface.blit(lbl_surf, (drop_x + int(12 * d), y + (item_h - lbl_surf.get_height()) // 2))

            if item.get("shortcut"):
                sc_surf, _ = shortcut_font.render(item["shortcut"], secondary_c)
                surface.blit(sc_surf, (drop_x + drop_w - sc_surf.get_width() - int(12 * d),
                                       y + (item_h - sc_surf.get_height()) // 2))
            y += item_h
