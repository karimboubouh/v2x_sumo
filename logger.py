"""Colored console logger for SUMO V2V Dashboard."""

try:
    import config as _config
    _default_level = getattr(_config, "LOG_LEVEL", "info")
except Exception:
    _default_level = "info"

_RESET   = "\033[0m"
_REVERSE = "\033[7m"

# severity order: debug < info < success < result < warning < error
_LEVELS = {
    "debug":   {"severity": 0, "color": "\033[36m", "label": "DEBUG  "},
    "info":    {"severity": 1, "color": "\033[34m", "label": "INFO   "},
    "success": {"severity": 2, "color": "\033[32m", "label": "SUCCESS"},
    "result":  {"severity": 3, "color": "\033[35m", "label": "RESULT "},
    "warning": {"severity": 4, "color": "\033[33m", "label": "WARNING"},
    "error":   {"severity": 5, "color": "\033[31m", "label": "ERROR  "},
}

# "[SUCCESS] " = 10 chars — same width for all labels (7-char padded label)
_INDENT = 10

_min_severity = _LEVELS.get(_default_level, _LEVELS["info"])["severity"]
_last_type    = "info"


def set_level(level: str):
    """Set the minimum log level to display. Call once after parsing --verbose."""
    global _min_severity
    level = level.lower()
    if level in _LEVELS:
        _min_severity = _LEVELS[level]["severity"]


def log(message: str, type: str = None):
    """
    Print a colored log line.

    Args:
        message: The text to display.
        type: One of debug|info|success|result|warning|error.
              If omitted, inherits the previous type and indents the line to
              align under the message text of the previous typed line.
    """
    global _last_type

    if type is not None:
        type = type.lower()
        if type not in _LEVELS:
            type = "info"
        _last_type = type
        level_info = _LEVELS[type]

        if level_info["severity"] < _min_severity:
            return

        color = level_info["color"]
        label = level_info["label"]
        # Label drawn in reverse (colored background), message in foreground color
        print(f"{color}{_REVERSE}[{label}]{_RESET} {color}{message}{_RESET}", flush=True)

    else:
        # Continuation: inherit last type's color/filter, indent to align
        if _LEVELS[_last_type]["severity"] < _min_severity:
            return
        color = _LEVELS[_last_type]["color"]
        print(f"{color}{' ' * _INDENT}{message}{_RESET}", flush=True)
