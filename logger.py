"""Colored console logger for SUMO V2V Dashboard."""

import os
import re
import sys
import threading
import time as _time

try:
    import config as _config

    _default_level = getattr(_config, "LOG_LEVEL", "info")
except Exception:
    _default_level = "info"

_RESET = "\033[0m"
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
_last_type = "info"

# ── Progress bar state ────────────────────────────────────────────────────────
_bar_lock        = threading.Lock()
_bar_active      = False
_bar_tty         = sys.stdout.isatty()
_bar_start       = 0.0
_bar_cpu         = 0.0
_bar_mem_mb      = 0.0
_bar_last_sample = 0.0
_bar_proc        = None
_bar_line        = ""
_file_handle     = None
_file_path       = None

_ANSI_ESC = re.compile(r"\033\[[^m]*m")
_BAR_W    = 18   # visual width of the filled/empty block


def set_level(level: str):
    """Set the minimum log level to display. Call once after parsing --verbose."""
    global _min_severity
    level = level.lower()
    if level in _LEVELS:
        _min_severity = _LEVELS[level]["severity"]


def start_file_logging(path: str) -> None:
    """Begin mirroring logger.log(...) output to a plain-text file."""
    global _file_handle, _file_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _bar_lock:
        if _file_handle is not None:
            _file_handle.close()
        _file_path = os.path.abspath(path)
        _file_handle = open(_file_path, "a", encoding="utf-8")


def stop_file_logging() -> None:
    """Stop mirroring logger output to the plain-text file sink."""
    global _file_handle, _file_path
    with _bar_lock:
        if _file_handle is not None:
            _file_handle.close()
            _file_handle = None
        _file_path = None


def _write_file_log_line(text: str) -> None:
    if _file_handle is None:
        return
    _file_handle.write(text + "\n")
    _file_handle.flush()


def log(message: str, type: str = None):
    """
    Print a colored log line.

    When the progress bar is active the bar is erased before printing and
    redrawn after so it always stays at the bottom.

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
        text = f"{color}{_REVERSE}[{label}]{_RESET} {color}{message}{_RESET}"
        file_text = f"[{label.strip()}] {message}"

    else:
        # Continuation: inherit last type's color/filter, indent to align
        if _LEVELS[_last_type]["severity"] < _min_severity:
            return
        color = _LEVELS[_last_type]["color"]
        text = f"{color}{' ' * _INDENT}{message}{_RESET}"
        file_text = f"{' ' * _INDENT}{message}"

    with _bar_lock:
        _write_file_log_line(file_text)
        if _bar_active and _bar_tty:
            # Erase bar line, print log line, redraw bar underneath
            sys.stdout.write(f"\r\033[K{text}\n")
            if _bar_line:
                sys.stdout.write(_bar_line)
            sys.stdout.flush()
        else:
            print(text, flush=True)


# ── Progress bar public API ───────────────────────────────────────────────────

def enable_progress_bar() -> None:
    """Activate the sticky bottom progress bar (no-op when stdout is not a TTY)."""
    global _bar_active, _bar_start, _bar_proc, _bar_last_sample
    if not _bar_tty:
        return
    try:
        import psutil
        _bar_proc = psutil.Process()
        _bar_proc.cpu_percent()   # prime the rolling counter
    except Exception:
        pass
    _bar_start = _time.monotonic()
    _bar_last_sample = _time.monotonic() - 2.0   # trigger first sample immediately
    _bar_active = True


def update_progress_bar(
    training_status: "dict | None",
    sim_time: float,
    vehicle_count: int,
    link_count: int,
    step_count: int,
) -> None:
    """Redraw the progress bar with the latest simulation/training state."""
    global _bar_line, _bar_cpu, _bar_mem_mb, _bar_last_sample
    if not _bar_active:
        return

    now = _time.monotonic()
    if now - _bar_last_sample >= 1.0 and _bar_proc is not None:
        try:
            _bar_cpu    = _bar_proc.cpu_percent()
            _bar_mem_mb = _bar_proc.memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        _bar_last_sample = now

    line = _render_bar(training_status, sim_time, vehicle_count, link_count, step_count, now)
    with _bar_lock:
        _bar_line = line
        sys.stdout.write(f"\r\033[K{line}")
        sys.stdout.flush()


def clear_progress_bar() -> None:
    """Erase the progress bar line and deactivate it (call on shutdown)."""
    global _bar_active, _bar_line
    if not _bar_active:
        return
    with _bar_lock:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        _bar_line = ""
    _bar_active = False


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    s = max(int(seconds), 0)
    h, r = divmod(s, 3600)
    m, sc = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sc:02d}" if h else f"{m:02d}:{sc:02d}"


def _vis(s: str) -> int:
    """Visible (non-ANSI) character count of *s*."""
    return len(_ANSI_ESC.sub("", s))


def _render_bar(
    ts: "dict | None",
    sim_time: float,
    vehicle_count: int,
    link_count: int,
    step_count: int,
    now: float,
) -> str:
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 120
    max_w = max(cols - 2, 40)

    elapsed = now - _bar_start
    DIM = "\033[90m"
    RST = "\033[0m"
    BLD = "\033[1m"
    GRN = "\033[32m"
    BLU = "\033[34m"
    sep = f" {DIM}·{RST} "

    if ts and ts.get("enabled"):
        progress = max(0.0, min(float(ts.get("progress", 0.0)), 1.0))
        filled   = round(_BAR_W * progress)
        empty    = _BAR_W - filled
        done     = bool(ts.get("done"))
        fill_c   = GRN if done else BLU

        bar = (
            f"{DIM}[{RST}"
            f"{fill_c}{'█' * filled}{DIM}{'░' * empty}{RST}"
            f"{DIM}]{RST}"
        )
        pct  = f"{BLD}{progress:5.1%}{RST}"
        algo = str(ts.get("algorithm", "DPL"))
        rn   = int(ts.get("round", 0))
        mr   = max(int(ts.get("max_rounds", 1)), 1)
        eta  = _fmt_dur(float(ts.get("remaining_time", 0.0)))
        est_round = _fmt_dur(float(ts.get("estimated_round_time", ts.get("avg_round_time", 0.0))))
        acc  = float(ts.get("train_acc", 0.0))
        loss = float(ts.get("train_loss", 0.0))
        act  = int(ts.get("active_trainers", 0))
        total  = int(ts.get("vehicle_count", vehicle_count))

        segments = [
            f"{bar} {pct}",
            f"{DIM}Round{RST} {rn}/{mr}",
            f"{DIM}ETA{RST} {eta}",
            f"{DIM}TPR{RST} {est_round}",
            f"{DIM}Acc{RST} {acc:.2%}  {DIM}Loss{RST} {loss:.4f}",
            f"{DIM}Active{RST} {act}/{total}",
        ]

        tgt = float(ts.get("target_acc", 2.0))
        if tgt < 1.0:
            segments.append(f"{DIM}Target{RST} {tgt:.2%}")

        if "PPO" in algo:
            segments.append(f"{DIM}Reward{RST} {ts.get('avg_reward', 0.0):+.3f}")

        eval_acc = ts.get("eval_acc", ts.get("test_acc"))
        if eval_acc is not None:
            eval_label = str(ts.get("eval_label", "Test"))
            t_loss = float(ts.get("eval_loss", ts.get("test_loss", 0.0)))
            t_rnd  = ts.get("eval_round", ts.get("test_round", rn))
            segments.append(
                f"{DIM}{eval_label}{RST} {eval_acc:.2%}  {DIM}loss{RST} {t_loss:.4f}"
                f"  {DIM}@r{RST}{t_rnd}"
            )
    else:
        segments = [
            f"{DIM}Step{RST} {step_count}",
            f"{DIM}Sim{RST} {sim_time:.0f}s",
            f"{DIM}Veh{RST} {vehicle_count}",
            f"{DIM}Links{RST} {link_count}",
            f"{DIM}DPL{RST} off",
        ]

    # System metrics always appended last (lowest priority for truncation)
    segments += [
        f"{DIM}CPU{RST} {_bar_cpu:.1f}%",
        f"{DIM}RAM{RST} {_bar_mem_mb:.0f} MB",
        f"{DIM}⏱{RST} {_fmt_dur(elapsed)}",
    ]

    # Build left-to-right, dropping trailing segments that would overflow
    prefix = "  "
    sep_w  = _vis(sep)
    parts: list[str] = []
    used = len(prefix)
    for seg in segments:
        w = _vis(seg)
        cost = (sep_w if parts else 0) + w
        if used + cost > max_w:
            break
        used += cost
        parts.append(seg)

    return prefix + sep.join(parts)
