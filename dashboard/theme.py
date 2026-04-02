"""Theme manager: dark/light mode with macOS system detection."""

import subprocess

import config

_THEMES = {
    "dark": {
        "bg": (30, 30, 30),
        "road": (80, 80, 80),
        "text": (220, 220, 220),
        "text_secondary": (120, 120, 140),
        "separator": (60, 60, 60),
        "log_bg": (20, 20, 25),
        "log_hello": (100, 200, 100),
        "log_data": (100, 150, 255),
        "log_fl": (255, 180, 50),
        "log_link": (80, 210, 160),
        "log_weight": (255, 180, 50),
        "log_training": (120, 180, 255),
        "log_status": (210, 210, 210),
        "log_warning": (255, 120, 120),
        "link_strong": (0, 200, 80),
        "link_weak": (200, 60, 60),
        "vehicle_outline": (255, 255, 255),
        "menu_bg": (40, 40, 45),
        "menu_hover": (65, 65, 75),
        "menu_text": (210, 210, 210),
        "menu_border": (60, 60, 70),
        "divider": (70, 70, 90),
        "pause_label": (255, 220, 60),
        "status_bg": (25, 25, 30),
        "status_bar_bg": (50, 50, 60),
    },
    "light": {
        "bg": (240, 240, 245),
        "road": (170, 170, 185),
        "text": (30, 30, 35),
        "text_secondary": (100, 100, 120),
        "separator": (195, 195, 205),
        "log_bg": (230, 230, 238),
        "log_hello": (30, 140, 30),
        "log_data": (30, 80, 200),
        "log_fl": (180, 120, 0),
        "log_link": (0, 120, 90),
        "log_weight": (180, 120, 0),
        "log_training": (40, 90, 190),
        "log_status": (60, 60, 70),
        "log_warning": (185, 60, 40),
        "link_strong": (0, 160, 60),
        "link_weak": (200, 40, 40),
        "vehicle_outline": (40, 40, 40),
        "menu_bg": (230, 230, 235),
        "menu_hover": (210, 210, 220),
        "menu_text": (30, 30, 35),
        "menu_border": (195, 195, 205),
        "divider": (180, 180, 195),
        "pause_label": (180, 140, 0),
        "status_bg": (235, 235, 240),
        "status_bar_bg": (205, 205, 215),
    },
}

_current = None
_version = 0


def _detect_system_theme():
    """Detect macOS dark/light mode. Returns 'dark' or 'light'."""
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and "dark" in result.stdout.strip().lower():
            return "dark"
    except Exception:
        pass
    return "light"


def init(mode=None):
    """Initialize theme. mode: 'dark', 'light', or 'system' (default from config)."""
    global _current, _version
    if mode is None:
        mode = getattr(config, "THEME_MODE", "system")
    if mode == "system":
        _current = _detect_system_theme()
    else:
        _current = mode if mode in _THEMES else "dark"
    _version += 1


def get():
    """Return current theme name ('dark' or 'light')."""
    if _current is None:
        init()
    return _current


def toggle():
    """Switch between dark and light. Returns new theme name."""
    global _current, _version
    if _current is None:
        init()
    _current = "light" if _current == "dark" else "dark"
    _version += 1
    return _current


def color(name):
    """Return a color tuple for the given name in the current theme."""
    return _THEMES[get()][name]


def version():
    """Return a monotonically increasing int that changes on every theme switch."""
    return _version
