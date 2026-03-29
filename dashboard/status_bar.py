"""Bottom status bar showing system resource usage."""

import time
import psutil
import pygame
import pygame.freetype

from dashboard import theme

_font_cache = {}


def _make_font(size, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for name in ["menlo", "sfmonomedium", "couriernew", "dejavusansmono"]:
        try:
            f = pygame.freetype.SysFont(name, size, bold=bold)
            if f:
                _font_cache[key] = f
                return f
        except Exception:
            pass
    f = pygame.freetype.SysFont(None, size, bold=bold)
    _font_cache[key] = f
    return f


class StatusBar:
    """Bottom bar showing CPU, GPU, RAM, and runtime metrics in a single row."""

    def __init__(self, rect, dpi_scale=1.0):
        self.rect = rect
        self.dpi_scale = dpi_scale
        self._font_size = int(11 * dpi_scale)
        self._start_time = time.monotonic()
        self._process = psutil.Process()

        # Sampling state — update metrics every 1s, not every frame
        self._last_sample = 0.0
        self._sample_interval = 1.0
        self._proc_cpu = 0.0
        self._proc_mem_mb = 0.0
        self._sys_cpu = 0.0
        self._sys_mem_pct = 0.0
        self._sys_mem_used_gb = 0.0
        self._gpu_pct = None  # None if unavailable

        # Kick off first CPU measurement (non-blocking)
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

        self._sys_cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        self._sys_mem_pct = vm.percent
        self._sys_mem_used_gb = vm.used / (1024 ** 3)

        self._gpu_pct = self._sample_gpu()

    def _sample_gpu(self):
        """Try to get GPU utilization. Returns percentage or None."""
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=1,
            )
            if "Device Utilization %" in result.stdout:
                for line in result.stdout.splitlines():
                    if "Device Utilization %" in line:
                        val = line.split("=")[-1].strip().rstrip("%").strip()
                        return float(val)
        except Exception:
            pass
        return None

    def draw(self, surface):
        """Draw the horizontal status bar."""
        self._sample()

        d = self.dpi_scale
        pad = int(8 * d)

        # Background + top border
        pygame.draw.rect(surface, theme.color("status_bg"), self.rect)
        pygame.draw.line(
            surface, theme.color("separator"),
            (self.rect.x, self.rect.y),
            (self.rect.x + self.rect.width, self.rect.y),
        )

        font = _make_font(self._font_size)
        lbl_font = _make_font(self._font_size, bold=True)
        lbl_color = theme.color("text_secondary")
        val_color = theme.color("text")
        cy = self.rect.y + (self.rect.height - self._font_size) // 2

        # Running time
        elapsed = time.monotonic() - self._start_time
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        runtime_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        # GPU string
        gpu_str = f"{self._gpu_pct:.0f}%" if self._gpu_pct is not None else "N/A"

        # Build items: (label, value, optional bar_pct)
        items = [
            ("Runtime", runtime_str, None),
            ("Proc CPU", f"{self._proc_cpu:.1f}%", min(self._proc_cpu, 100) / 100),
            ("Proc GPU", gpu_str, min(self._gpu_pct, 100) / 100 if self._gpu_pct is not None else None),
            ("Proc RAM", f"{self._proc_mem_mb:.0f} MB", None),
            ("Sys CPU", f"{self._sys_cpu:.1f}%", min(self._sys_cpu, 100) / 100),
            ("Sys RAM", f"{self._sys_mem_pct:.1f}% ({self._sys_mem_used_gb:.1f} GB)", min(self._sys_mem_pct, 100) / 100),
        ]

        x = self.rect.x + pad

        for i, (label, value, bar_pct) in enumerate(items):
            # Label
            lbl_surf, _ = lbl_font.render(label + ": ", lbl_color)
            surface.blit(lbl_surf, (x, cy))
            x += lbl_surf.get_width()

            # Value
            val_surf, _ = font.render(value, val_color)
            surface.blit(val_surf, (x, cy))
            x += val_surf.get_width()

            # Mini bar after value (for CPU/RAM metrics)
            if bar_pct is not None:
                bar_w = int(40 * d)
                bar_h = int(4 * d)
                bar_x = x + int(4 * d)
                bar_y = cy + self._font_size // 2 - bar_h // 2
                bar_bg = theme.color("status_bar_bg")
                # Green → yellow → red
                if bar_pct < 0.5:
                    r = int(80 + 340 * bar_pct)
                    g = 200
                else:
                    r = 250
                    g = int(200 - 320 * (bar_pct - 0.5))
                bar_fg = (min(255, r), max(0, g), 60)
                pygame.draw.rect(surface, bar_bg, (bar_x, bar_y, bar_w, bar_h), border_radius=2)
                fill_w = max(1, int(bar_w * bar_pct))
                pygame.draw.rect(surface, bar_fg, (bar_x, bar_y, fill_w, bar_h), border_radius=2)
                x = bar_x + bar_w

            # Separator dot between items
            if i < len(items) - 1:
                dot_x = x + int(8 * d)
                dot_surf, _ = font.render("·", lbl_color)
                surface.blit(dot_surf, (dot_x, cy))
                x = dot_x + dot_surf.get_width() + int(8 * d)

    def invalidate_caches(self):
        """Clear any theme-dependent caches."""
        pass
