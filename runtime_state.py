"""Thread-safe runtime state shared between simulation and dashboard."""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import logger


@dataclass(slots=True)
class FrameSnapshot:
    vehicle_states: dict = field(default_factory=dict)
    render_links: list = field(default_factory=list)
    log_links: list = field(default_factory=list)
    training_status: dict | None = None
    vehicle_overlays: dict | None = None
    sim_time: float = 0.0
    step_count: int = 0
    simulation_done: bool = False
    overlay_text: str | None = None


class LatestFrameBuffer:
    """Single-slot frame buffer: new frames overwrite stale UI frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._frame: FrameSnapshot | None = None

    def publish(self, frame: FrameSnapshot) -> int:
        with self._lock:
            self._version += 1
            self._frame = frame
            return self._version

    def latest(self) -> tuple[int, FrameSnapshot | None]:
        with self._lock:
            return self._version, self._frame


class PerfStats:
    """Small rolling performance logger used only when --perf-log is enabled."""

    def __init__(self, enabled: bool, interval_s: float = 5.0, max_samples: int = 600) -> None:
        self.enabled = bool(enabled)
        self.interval_s = float(interval_s)
        self._max_samples = int(max_samples)
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )
        self._latest: dict[str, Any] = {}
        self._last_log = time.perf_counter()
        self._process = None
        if self.enabled:
            try:
                import psutil

                self._process = psutil.Process()
                self._process.cpu_percent()
            except Exception:
                self._process = None

    def record(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._samples[name].append(float(value))

    def set_latest(self, **values) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latest.update(values)

    @staticmethod
    def _pctl(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1)
        return ordered[idx]

    def maybe_log(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            if now - self._last_log < self.interval_s:
                return
            self._last_log = now
            samples = {key: list(value) for key, value in self._samples.items()}
            latest = dict(self._latest)
            for value in self._samples.values():
                value.clear()

        frame_values = samples.get("ui_frame_s", [])
        frame_p50 = self._pctl(frame_values, 0.50) * 1000.0
        frame_p95 = self._pctl(frame_values, 0.95) * 1000.0
        ui_fps = (1.0 / statistics.mean(frame_values)) if frame_values else 0.0

        def avg_ms(name: str) -> float:
            vals = samples.get(name, [])
            return statistics.mean(vals) * 1000.0 if vals else 0.0

        cpu = mem = None
        if self._process is not None:
            try:
                cpu = self._process.cpu_percent()
                mem = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                cpu = mem = None

        parts = [
            f"ui_fps={ui_fps:.1f}",
            f"frame_p50={frame_p50:.1f}ms",
            f"frame_p95={frame_p95:.1f}ms",
            f"sim_step={avg_ms('sim_step_s'):.1f}ms",
            f"dl_step={avg_ms('dl_step_s'):.1f}ms",
            f"snapshot={avg_ms('snapshot_s'):.1f}ms",
            f"render={avg_ms('render_s'):.1f}ms",
            f"event_backlog={int(latest.get('event_backlog', 0) or 0)}",
            f"active_trainers={int(latest.get('active_trainers', 0) or 0)}",
        ]
        if cpu is not None and mem is not None:
            parts.append(f"cpu={cpu:.1f}%")
            parts.append(f"ram={mem:.0f}MB")
        logger.log("PERF | " + " | ".join(parts), "info")
