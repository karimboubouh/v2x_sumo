"""StatusWidget — bottom bar with training progress and system metrics."""

from __future__ import annotations

import time

import psutil

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QWidget, QSizePolicy

from dashboard import theme


class StatusWidget(QWidget):
    """Compact status bar: header row · progress bar · metrics row."""

    def __init__(self, dpi_scale: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self._dpi = dpi_scale
        self._training_status: dict | None = None
        self._start_time = time.monotonic()

        # System metrics
        self._process = psutil.Process()
        self._process.cpu_percent()           # prime the counter
        self._cpu  = 0.0
        self._mem_mb = 0.0
        self._gpu: float | None = None

        self._elapsed_frozen: float | None = None

        self._sample_timer = QTimer(self)
        self._sample_timer.setInterval(1000)
        self._sample_timer.timeout.connect(self._sample_metrics)
        self._sample_timer.start()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAutoFillBackground(False)

    # ── Data update ───────────────────────────────────────────────────────────

    def update_status(self, training_status: dict | None) -> None:
        self._training_status = training_status
        self.update()

    def mark_done(self) -> None:
        if self._elapsed_frozen is None:
            self._elapsed_frozen = time.monotonic() - self._start_time
        self.update()

    # ── Metrics sampling ──────────────────────────────────────────────────────

    def _sample_metrics(self) -> None:
        self._cpu    = self._process.cpu_percent()
        self._mem_mb = self._process.memory_info().rss / (1024 * 1024)
        self._gpu    = self._sample_gpu()
        self.update()

    @staticmethod
    def _sample_gpu() -> float | None:
        try:
            import subprocess
            r = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=1,
            )
            for line in r.stdout.splitlines():
                if "Device Utilization %" in line:
                    val = line.split("=")[-1].strip().rstrip("%").strip()
                    return float(val)
        except Exception:
            pass
        return None

    @staticmethod
    def _fmt_dur(seconds: float) -> str:
        s = max(int(seconds), 0)
        h, rem = divmod(s, 3600)
        m, sc  = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sc:02d}" if h else f"{m:02d}:{sc:02d}"

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        d = self._dpi
        ts = self._training_status

        # Background + top border
        painter.fillRect(self.rect(), theme.color("status_bg"))
        painter.setPen(QPen(theme.color("separator"), 1.0))
        painter.drawLine(0, 0, self.width(), 0)

        pad_x = int(10 * d)
        pad_y = int(5 * d)
        inner_x  = pad_x
        inner_y  = pad_y
        inner_w  = self.width() - 2 * pad_x
        inner_h  = self.height() - 2 * pad_y

        # ── Header row ────────────────────────────────────────────────────────
        hdr_font = QFont()
        hdr_font.setPointSize(int(8.5 * d))
        hdr_font.setWeight(QFont.DemiBold)
        val_font = QFont()
        val_font.setPointSize(int(8.5 * d))

        hdr_h = QFontMetrics(hdr_font).height()

        segments = self._build_segments(ts)
        self._draw_segments(
            painter,
            QRectF(inner_x, inner_y, inner_w, hdr_h),
            segments,
            hdr_font, val_font,
        )

        if ts and ts.get("enabled"):
            gap     = int(4 * d)
            bar_top = inner_y + hdr_h + gap

            # ── Progress bar ─────────────────────────────────────────────────
            bar_h   = max(int(8 * d), 6)
            bar_w   = inner_w
            progress = max(0.0, min(float(ts.get("progress", 0.0)), 1.0))

            # Track
            track_c = theme.color("progress_bg")
            painter.setBrush(QBrush(track_c))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(inner_x, bar_top, bar_w, bar_h), 3, 3)

            # Fill
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                fc = theme.color("progress_done" if ts.get("done") else "progress_fill")
                painter.setBrush(QBrush(fc))
                painter.drawRoundedRect(QRectF(inner_x, bar_top, fill_w, bar_h), 3, 3)

            # ── Metrics row ───────────────────────────────────────────────────
            met_font = QFont()
            met_font.setPointSize(int(8 * d))
            met_font.setStyleHint(QFont.Monospace)
            met_y = bar_top + bar_h + gap + QFontMetrics(met_font).ascent()

            train_text = (
                f"Train  acc {ts.get('train_acc', 0.0):.2%}  "
                f"loss {ts.get('train_loss', 0.0):.4f}"
            )
            painter.setFont(met_font)
            painter.setPen(theme.color("text"))
            painter.drawText(QPointF(inner_x, met_y), train_text)

            # Right side: init → test
            right = self._right_metrics_text(ts)
            if right:
                fm = QFontMetrics(met_font)
                rw = fm.horizontalAdvance(right)
                rx = inner_x + inner_w - rw
                min_x = inner_x + fm.horizontalAdvance(train_text) + int(16 * d)
                if rx >= min_x:
                    painter.setPen(theme.color("text_secondary"))
                    painter.drawText(QPointF(rx, met_y), right)

        painter.end()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_segments(self, ts: dict | None) -> list[tuple[str, str]]:
        elapsed = self._elapsed_frozen if self._elapsed_frozen is not None else time.monotonic() - self._start_time
        segs: list[tuple[str, str]] = [("Runtime: ", self._fmt_dur(elapsed))]

        if ts and ts.get("enabled"):
            algo = ts.get("algorithm")
            if algo:
                segs.append(("Algo: ", str(algo)))
            rn  = int(ts.get("round", 0))
            mr  = max(int(ts.get("max_rounds", 1)), 1)
            segs.append(("Round: ", f"{rn}/{mr}"))
            segs.append(("ETA: ", self._fmt_dur(ts.get("remaining_time", 0.0))))
            segs.append(("Active: ", f"{ts.get('active_trainers',0)}/{ts.get('vehicle_count',0)}"))
            segs.append(("Done: ", f"{ts.get('done_vehicles',0)}/{ts.get('vehicle_count',0)}"))
            tgt = float(ts.get("target_acc", 2.0))
            if tgt < 1.0:
                segs.append(("Target: ", f"{tgt:.2%}"))
            if "PPO" in str(algo or ""):
                segs.append(("Reward: ", f"{ts.get('avg_reward', 0.0):+.3f}"))
        else:
            segs.append(("DPL: ", "disabled"))

        segs.append(("CPU: ", f"{self._cpu:.1f}%"))
        if self._gpu is not None:
            segs.append(("GPU: ", f"{self._gpu:.0f}%"))
        segs.append(("RAM: ", f"{self._mem_mb:.0f} MB"))
        return segs

    @staticmethod
    def _draw_segments(
        painter: QPainter,
        rect: QRectF,
        segments: list[tuple[str, str]],
        lbl_font: QFont,
        val_font: QFont,
    ) -> None:
        lbl_fm = QFontMetrics(lbl_font)
        val_fm = QFontMetrics(val_font)
        sep_w  = val_fm.horizontalAdvance("  ·  ")
        x      = rect.x()
        y      = rect.y() + lbl_fm.ascent()

        for i, (label, value) in enumerate(segments):
            lbl_w = lbl_fm.horizontalAdvance(label)
            val_w = val_fm.horizontalAdvance(value)
            extra = sep_w if i < len(segments) - 1 else 0
            if x + lbl_w + val_w + extra > rect.right():
                break
            painter.setFont(lbl_font)
            painter.setPen(theme.color("text_secondary"))
            painter.drawText(QPointF(x, y), label)
            x += lbl_w
            painter.setFont(val_font)
            painter.setPen(theme.color("text"))
            painter.drawText(QPointF(x, y), value)
            x += val_w
            if i < len(segments) - 1:
                painter.setPen(theme.color("text_dim"))
                painter.drawText(QPointF(x, y), "  ·  ")
                x += sep_w

    @staticmethod
    def _right_metrics_text(ts: dict) -> str:
        init_acc  = ts.get("init_test_acc")
        test_acc  = ts.get("test_acc")
        test_loss = ts.get("test_loss")
        rnd       = ts.get("test_round")
        if test_acc is not None and init_acc is not None:
            return (f"Init {init_acc:.2%} → Test {test_acc:.2%}  "
                    f"loss {test_loss:.4f}  @ r{rnd}")
        if test_acc is not None:
            return f"Test {test_acc:.2%}  loss {test_loss:.4f}  @ r{rnd}"
        if init_acc is not None:
            return f"Init test {init_acc:.2%}  loss {ts.get('init_test_loss', 0.0):.4f}"
        return ""
