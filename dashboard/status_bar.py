"""Bottom status bar showing system resource usage and DPL progress."""

import time

import psutil
import pygame

from dashboard.fonts import get_font
from dashboard import theme


class StatusBar:
    """Bottom bar showing a single merged header row, progress bar, and metrics."""

    def __init__(self, rect, dpi_scale=1.0):
        self.rect = rect
        self.dpi_scale = dpi_scale
        self._font_size = int(12 * dpi_scale)
        self._small_font_size = max(int(11 * dpi_scale), 10)
        self._start_time = time.monotonic()
        self._process = psutil.Process()

        self._last_sample = 0.0
        self._sample_interval = 1.0
        self._proc_cpu = 0.0
        self._proc_mem_mb = 0.0
        self._gpu_pct = None

        self._process.cpu_percent()

    def _sample(self):
        """Update metrics if enough time has elapsed."""
        now = time.monotonic()
        if now - self._last_sample < self._sample_interval:
            return
        self._last_sample = now

        self._proc_cpu = self._process.cpu_percent()
        mem_info = self._process.memory_info()
        self._proc_mem_mb = mem_info.rss / (1024 * 1024)

        self._gpu_pct = self._sample_gpu()

    def _sample_gpu(self):
        """Try to get GPU utilization. Returns percentage or None."""
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if "Device Utilization %" in result.stdout:
                for line in result.stdout.splitlines():
                    if "Device Utilization %" in line:
                        val = line.split("=")[-1].strip().rstrip("%").strip()
                        return float(val)
        except Exception:
            pass
        return None

    def _format_duration(self, seconds):
        seconds = max(float(seconds or 0.0), 0.0)
        hours, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _draw_segmented_text(self, surface, rect, segments, font, label_font):
        """Draw label/value segments on a single line until space runs out."""
        x = rect.x
        y = rect.y
        for idx, (label, value) in enumerate(segments):
            lbl_surf, _ = label_font.render(label, theme.color("text_secondary"))
            val_surf, _ = font.render(value, theme.color("text"))
            width = lbl_surf.get_width() + val_surf.get_width()
            if idx < len(segments) - 1:
                dot_surf, _ = font.render("  ·  ", theme.color("text_secondary"))
                width += dot_surf.get_width()
            if x + width > rect.right:
                break
            surface.blit(lbl_surf, (x, y))
            x += lbl_surf.get_width()
            surface.blit(val_surf, (x, y))
            x += val_surf.get_width()
            if idx < len(segments) - 1:
                dot_surf, _ = font.render("  ·  ", theme.color("text_secondary"))
                surface.blit(dot_surf, (x, y))
                x += dot_surf.get_width()

    def _draw_merged_header(self, surface, rect, training_status):
        """Single row: Runtime + DPL round/ETA/vehicles + system CPU/RAM."""
        font = get_font(self._small_font_size)
        label_font = get_font(self._small_font_size, bold=True)

        elapsed = time.monotonic() - self._start_time
        segments = [("Runtime: ", self._format_duration(elapsed))]

        if training_status and training_status.get("enabled"):
            round_n = int(training_status["round"])
            max_rounds = max(int(training_status["max_rounds"]), 1)
            segments.append(("Round: ", f"{round_n}/{max_rounds}"))
            segments.append(("ETA: ", self._format_duration(training_status["remaining_time"])))
            segments.append(("Active: ", f"{training_status['active_trainers']}/{training_status['vehicle_count']}"))
            segments.append(("Done: ", f"{training_status['done_vehicles']}/{training_status['vehicle_count']}"))
            target_acc = float(training_status["target_acc"])
            if target_acc <= 1.0:
                segments.append(("Target: ", f"{target_acc:.2%}"))
        else:
            segments.append(("DPL: ", "disabled"))

        segments.append(("CPU: ", f"{self._proc_cpu:.1f}%"))
        if self._gpu_pct is not None:
            segments.append(("GPU: ", f"{self._gpu_pct:.0f}%"))
        segments.append(("RAM: ", f"{self._proc_mem_mb:.0f} MB"))

        self._draw_segmented_text(surface, rect, segments, font, label_font)

    def _draw_progress_and_metrics(self, surface, rect, training_status):
        """Progress bar + train/init-test/test metrics row."""
        d = self.dpi_scale

        if not training_status or not training_status.get("enabled"):
            return

        # Progress bar
        progress = max(0.0, min(float(training_status["progress"]), 1.0))
        bar_h = max(int(10 * d), 8)
        pygame.draw.rect(
            surface,
            theme.color("status_bar_bg"),
            (rect.x, rect.y, rect.width, bar_h),
            border_radius=4,
        )
        fill_w = max(1, int(rect.width * progress)) if progress > 0 else 0
        if fill_w > 0:
            fill_color = (
                theme.color("log_training") if training_status["done"]
                else theme.color("link_strong")
            )
            pygame.draw.rect(
                surface,
                fill_color,
                (rect.x, rect.y, fill_w, bar_h),
                border_radius=4,
            )

        # Metrics row
        small_font = get_font(max(self._small_font_size - 1, 9), mono=True)
        metrics_y = rect.y + bar_h + int(4 * d)

        train_text = (
            f"Train acc {training_status['train_acc']:.2%}  "
            f"loss {training_status['train_loss']:.4f}"
        )
        train_surf, _ = small_font.render(train_text, theme.color("text"))
        surface.blit(train_surf, (rect.x, metrics_y))

        # Right side: "Init 10.16% → Test 72.15%  0.8234  @ r10"
        # or just test if no init, or just init if no current test yet
        init_acc = training_status.get("init_test_acc")
        init_loss = training_status.get("init_test_loss")
        test_acc = training_status.get("test_acc")

        if test_acc is not None and init_acc is not None:
            right_text = (
                f"Init {init_acc:.2%} → "
                f"Test {test_acc:.2%}  "
                f"loss {training_status['test_loss']:.4f}  "
                f"@ r{training_status['test_round']}"
            )
            right_color = theme.color("text_secondary")
        elif test_acc is not None:
            right_text = (
                f"Test acc {test_acc:.2%}  "
                f"loss {training_status['test_loss']:.4f}  "
                f"@ r{training_status['test_round']}"
            )
            right_color = theme.color("text_secondary")
        elif init_acc is not None:
            right_text = f"Init test {init_acc:.2%}  loss {init_loss:.4f}"
            right_color = theme.color("text_secondary")
        else:
            right_text = ""
            right_color = theme.color("text_secondary")

        if right_text:
            right_surf, _ = small_font.render(right_text, right_color)
            right_x = rect.right - right_surf.get_width()
            min_x = rect.x + train_surf.get_width() + int(12 * d)
            if right_x >= min_x:
                surface.blit(right_surf, (right_x, metrics_y))
            else:
                surface.blit(right_surf, (rect.x, metrics_y + int(10 * d)))

    def draw(self, surface, training_status=None):
        """Draw the status bar."""
        self._sample()

        d = self.dpi_scale
        pad_x = int(8 * d)
        pad_y = int(6 * d)

        pygame.draw.rect(surface, theme.color("status_bg"), self.rect)
        pygame.draw.line(
            surface,
            theme.color("separator"),
            (self.rect.x, self.rect.y),
            (self.rect.x + self.rect.width, self.rect.y),
        )

        inner = pygame.Rect(
            self.rect.x + pad_x,
            self.rect.y + pad_y,
            self.rect.width - 2 * pad_x,
            self.rect.height - 2 * pad_y,
        )
        header_h = int(14 * d)
        gap = int(6 * d)

        header_rect = pygame.Rect(inner.x, inner.y, inner.width, header_h)
        progress_rect = pygame.Rect(
            inner.x,
            inner.y + header_h + gap,
            inner.width,
            inner.bottom - (inner.y + header_h + gap),
        )

        self._draw_merged_header(surface, header_rect, training_status)
        self._draw_progress_and_metrics(surface, progress_rect, training_status)

    def invalidate_caches(self):
        """Clear any theme-dependent caches."""
        pass
