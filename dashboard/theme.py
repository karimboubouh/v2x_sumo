"""Theme manager — dark / light mode, returns QColor values."""

from __future__ import annotations
import subprocess
import sys

from PySide6.QtGui import QColor

import config

# ── Palette definitions ──────────────────────────────────────────────────────

_DARK: dict[str, tuple[int, int, int]] = {
    "bg":               (16,  18,  26),
    "bg_alt":           (22,  24,  34),
    "surface":          (28,  30,  42),
    "surface_raised":   (38,  40,  56),
    "border":           (52,  56,  78),
    "text":             (218, 220, 230),
    "text_secondary":   (138, 142, 162),
    "text_dim":         (88,  92,  112),
    # map
    "road":             (58,  62,  82),
    "road_edge":        (38,  42,  60),
    # link colors
    "link_strong":      (40,  210, 100),
    "link_weak":        (210, 70,  70),
    "link_sidelink":    (60,  160, 255),
    "link_internet":    (220, 80,  80),
    # log
    "log_bg":           (14,  15,  22),
    "log_hello":        (80,  200, 120),
    "log_data":         (80,  150, 255),
    "log_fl":           (255, 188, 58),
    "log_link":         (58,  200, 200),
    "log_weight":       (255, 138, 58),
    "log_training":     (98,  158, 255),
    "log_warning":      (255, 88,  88),
    "log_status":       (158, 163, 183),
    # chrome
    "separator":        (48,  52,  72),
    "divider":          (58,  62,  84),
    "status_bg":        (12,  13,  20),
    "status_bar_bg":    (32,  34,  48),
    "progress_fill":    (48,  178, 118),
    "progress_done":    (98,  158, 255),
    "progress_bg":      (32,  35,  50),
    "accent":           (62,  178, 255),
    "pause_label":      (255, 198, 58),
    "overlay_bg":       (10,  11,  18),
    "menu_bg":          (22,  24,  34),
    "menu_hover":       (42,  44,  60),
    "menu_text":        (218, 220, 230),
    "menu_border":      (52,  56,  78),
    "vehicle_outline":  (255, 255, 255),
}

_LIGHT: dict[str, tuple[int, int, int]] = {
    "bg":               (244, 245, 250),
    "bg_alt":           (234, 236, 244),
    "surface":          (255, 255, 255),
    "surface_raised":   (248, 249, 252),
    "border":           (198, 202, 218),
    "text":             (28,  30,  50),
    "text_secondary":   (88,  93,  118),
    "text_dim":         (148, 153, 175),
    "road":             (178, 184, 204),
    "road_edge":        (205, 208, 222),
    "link_strong":      (18,  168, 78),
    "link_weak":        (200, 48,  48),
    "link_sidelink":    (18,  128, 220),
    "link_internet":    (200, 50,  50),
    "log_bg":           (248, 250, 254),
    "log_hello":        (28,  148, 68),
    "log_data":         (28,  88,  200),
    "log_fl":           (178, 118, 18),
    "log_link":         (18,  138, 138),
    "log_weight":       (178, 88,  18),
    "log_training":     (48,  98,  200),
    "log_warning":      (200, 38,  38),
    "log_status":       (98,  103, 130),
    "separator":        (208, 212, 228),
    "divider":          (193, 197, 218),
    "status_bg":        (224, 227, 240),
    "status_bar_bg":    (208, 212, 226),
    "progress_fill":    (28,  148, 88),
    "progress_done":    (48,  118, 200),
    "progress_bg":      (210, 214, 228),
    "accent":           (28,  138, 220),
    "pause_label":      (200, 140, 18),
    "overlay_bg":       (238, 240, 250),
    "menu_bg":          (244, 245, 250),
    "menu_hover":       (224, 228, 242),
    "menu_text":        (28,  30,  50),
    "menu_border":      (198, 202, 218),
    "vehicle_outline":  (40,  40,  50),
}

# ── State ────────────────────────────────────────────────────────────────────

_current: str = "dark"
_version: int = 0


def _detect_system() -> str:
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and "dark" in r.stdout.lower():
                return "dark"
        except Exception:
            pass
    return "light"


def init(mode: str | None = None) -> None:
    global _current, _version
    if mode is None:
        mode = getattr(config, "THEME_MODE", "system")
    _current = _detect_system() if mode == "system" else (mode if mode in ("dark", "light") else "dark")
    _version += 1


def get() -> str:
    return _current


def toggle() -> str:
    global _current, _version
    _current = "light" if _current == "dark" else "dark"
    _version += 1
    return _current


def color(name: str) -> QColor:
    """Return a QColor for the named color in the current theme."""
    palette = _DARK if _current == "dark" else _LIGHT
    rgb = palette.get(name, (255, 0, 255))  # magenta = missing key
    return QColor(*rgb)


def color_alpha(name: str, alpha: int) -> QColor:
    """Return a QColor with a custom alpha channel (0–255)."""
    c = QColor(color(name))
    c.setAlpha(max(0, min(255, alpha)))
    return c


def version() -> int:
    return _version


# Legacy shim — some call sites use with_alpha(name, alpha)
def with_alpha(name: str, alpha: int) -> QColor:
    return color_alpha(name, alpha)
