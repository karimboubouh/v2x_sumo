"""Shared dashboard font loading using pygame.font and system fonts."""

import os

import pygame

_font_cache = {}

_SANS_NAMES = [
    "Ubuntu",
    "Ubuntu Sans",
    "SF Pro Text",
    "SF Pro Display",
    "Avenir Next",
    "Avenir",
    "Helvetica Neue",
    "Helvetica",
]

_MONO_NAMES = [
    "Ubuntu Mono",
    "SF Mono",
    "Menlo",
    "Monaco",
]

_SANS_PATHS = [
    "/Library/Fonts/Ubuntu-Regular.ttf",
    "/Library/Fonts/Ubuntu.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Avenir.ttc",
]

_MONO_PATHS = [
    "/Library/Fonts/UbuntuMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]


class DashboardFont:
    """Small wrapper to provide a freetype-like API over pygame.font.Font."""

    def __init__(self, font):
        self._font = font

    def render(self, text, color):
        surf = self._font.render(str(text), True, color)
        return surf, surf.get_rect()

    def render_to(self, surface, pos, text, color):
        surf, rect = self.render(text, color)
        surface.blit(surf, pos)
        return rect.move(pos)


def _ensure_font_module():
    if not pygame.font.get_init():
        pygame.font.init()


def _match_font(names, bold=False):
    for name in names:
        try:
            path = pygame.font.match_font(name, bold=bold)
        except Exception:
            path = None
        if path:
            return path
    return None


def _pick_font_path(names, paths, bold=False):
    _ensure_font_module()
    matched = _match_font(names, bold=bold)
    if matched:
        return matched
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def get_font(size, bold=False, mono=False):
    """Return a cached system font wrapper with crisp text rendering."""
    key = (int(size), bool(bold), bool(mono))
    if key in _font_cache:
        return _font_cache[key]

    names = _MONO_NAMES if mono else _SANS_NAMES
    paths = _MONO_PATHS if mono else _SANS_PATHS
    font_path = _pick_font_path(names, paths, bold=bold)

    if font_path is not None:
        font = pygame.font.Font(font_path, int(size))
        font.set_bold(bool(bold))
    else:
        font = pygame.font.SysFont(names[0], int(size), bold=bold)

    _font_cache[key] = DashboardFont(font)
    return _font_cache[key]
