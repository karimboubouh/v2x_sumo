"""
Generate the SUMO V2V Dashboard application icon.
Produces app.png (512×512) and, on macOS, app.icns.

Run:  python3 dashboard/icons/generate_icon.py
"""

from __future__ import annotations
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

# ── Palette (matches dashboard dark theme) ────────────────────────────────────
BG_OUTER  = (10,  12,  20)
BG_INNER  = (22,  26,  42)
ROAD_CLR  = (50,  56,  80)
CAR_A     = (80,  200, 255)   # cyan  — left vehicle
CAR_B     = (255, 140,  40)   # orange — right vehicle
SIGNAL    = (62,  178, 255)   # accent blue
GLOW      = (40,  120, 220)
DOT       = (98,  158, 255)   # training dot
WHITE     = (218, 220, 230)

SIZE = 512


# ── Helpers ───────────────────────────────────────────────────────────────────

def rr(draw: ImageDraw.ImageDraw, box, radius: float, fill, outline=None, width=1):
    """Draw a rounded rectangle (works on older Pillow without built-in rr)."""
    x0, y0, x1, y1 = box
    r = int(radius)
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
    if outline:
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=width)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=width)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=width)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=width)


def alpha_color(rgb, a):
    return rgb + (a,)


def draw_car(draw: ImageDraw.ImageDraw, cx: float, cy: float,
             color, facing_right: bool = True, scale: float = 1.0) -> None:
    """Draw a simplified top-down car icon."""
    w = int(44 * scale)
    h = int(76 * scale)
    hw, hh = w // 2, h // 2

    # Body
    rr(draw, (cx - hw, cy - hh, cx + hw, cy + hh), 10 * scale, fill=color)

    # Windshield (lighter top section)
    wf = tuple(min(255, c + 70) for c in color)
    ws_h = int(hh * 0.45)
    rr(draw, (cx - int(hw * 0.65), cy - hh, cx + int(hw * 0.65), cy - hh + ws_h),
       6 * scale, fill=wf)

    # Rear (darker bottom)
    dk = tuple(max(0, c - 60) for c in color)
    rr(draw, (cx - int(hw * 0.65), cy + hh - int(hh * 0.35),
              cx + int(hw * 0.65), cy + hh),
       4 * scale, fill=dk)

    # Wheels — four corner ellipses
    wr, ww = int(9 * scale), int(14 * scale)
    wheel = (28, 30, 38)
    for wx, wy in [(cx - hw - 2, cy - int(hh * 0.52)),
                   (cx + hw + 2 - ww, cy - int(hh * 0.52)),
                   (cx - hw - 2, cy + int(hh * 0.18)),
                   (cx + hw + 2 - ww, cy + int(hh * 0.18))]:
        draw.ellipse([wx, wy, wx + ww, wy + wr], fill=wheel)


def draw_signal_arcs(img: Image.Image, cx: float, cy: float,
                     color_rgb, facing_right: bool, n_arcs: int = 3) -> None:
    """Draw concentric signal arcs emanating from a vehicle."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    direction = 1 if facing_right else -1

    for i in range(n_arcs):
        r = int(38 + i * 28)
        alpha = max(30, 140 - i * 40)
        width = max(1, 4 - i)
        color = alpha_color(color_rgb, alpha)
        # Draw arc on the facing side only
        start_angle = -60 if facing_right else 120
        end_angle   =  60 if facing_right else 240
        box = [cx - r, cy - r, cx + r, cy + r]
        d.arc(box, start_angle, end_angle, fill=color, width=width)

    img.alpha_composite(overlay)


def draw_road(draw: ImageDraw.ImageDraw) -> None:
    """Diagonal road strip across the background."""
    # Main road band
    road_pts = [
        (0,   SIZE * 0.42),
        (SIZE, SIZE * 0.42),
        (SIZE, SIZE * 0.60),
        (0,   SIZE * 0.60),
    ]
    draw.polygon(road_pts, fill=ROAD_CLR)

    # Dashed centre line
    dash_y = int(SIZE * 0.51)
    dash_color = (70, 78, 110)
    for x in range(20, SIZE - 20, 40):
        draw.rectangle([x, dash_y - 2, x + 20, dash_y + 2], fill=dash_color)


def radial_gradient(size: int) -> Image.Image:
    """Create a soft radial gradient background (dark outer, slightly lighter centre)."""
    img = Image.new("RGBA", (size, size), BG_OUTER + (255,))
    cx, cy = size / 2, size / 2
    max_r = math.hypot(cx, cy)
    for y in range(size):
        for x in range(size):
            r = math.hypot(x - cx, y - cy) / max_r
            t = max(0.0, 1.0 - r * 1.4)
            blended = tuple(int(BG_OUTER[i] + (BG_INNER[i] - BG_OUTER[i]) * t)
                            for i in range(3))
            img.putpixel((x, y), blended + (255,))
    return img


def glow_circle(img: Image.Image, cx: float, cy: float,
                radius: float, color_rgb, alpha: int = 80) -> None:
    """Paint a soft radial glow blob."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    steps = 8
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        a = int(alpha * (1 - i / steps) * 1.6)
        a = min(255, a)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=alpha_color(color_rgb, a))
    img.alpha_composite(overlay)


# ── Build icon ────────────────────────────────────────────────────────────────

def build_icon(size: int = SIZE) -> Image.Image:
    scale = size / SIZE

    # 1. Background gradient
    img = radial_gradient(size)

    draw = ImageDraw.Draw(img)

    # 2. Road strip
    draw_road(draw)

    # 3. Glow blobs behind vehicles
    glow_circle(img, size * 0.28, size * 0.51, size * 0.22, GLOW, alpha=55)
    glow_circle(img, size * 0.72, size * 0.51, size * 0.22, (200, 100, 20), alpha=45)

    # 4. Signal arcs (drawn on RGBA layer)
    draw_signal_arcs(img, size * 0.28, size * 0.51, SIGNAL, facing_right=True)
    draw_signal_arcs(img, size * 0.72, size * 0.51, (220, 120, 30), facing_right=False)

    # 5. Vehicles
    draw = ImageDraw.Draw(img)
    draw_car(draw, int(size * 0.28), int(size * 0.51), CAR_A, facing_right=True, scale=scale)
    draw_car(draw, int(size * 0.72), int(size * 0.51), CAR_B, facing_right=False, scale=scale)

    # 6. Connection dot between the two cars (centre)
    dot_r = int(7 * scale)
    cx, cy = size // 2, int(size * 0.51)
    draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=DOT)

    # 7. Subtle outer vignette
    vignette = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    steps = min(60, size // 2 - 1)
    for i in range(steps):
        alpha = int(120 * (i / steps) ** 2)
        vd.rectangle([i, i, size - i - 1, size - i - 1], outline=(0, 0, 0, alpha), width=1)
    img.alpha_composite(vignette)

    # 8. Rounded-rect mask so it looks like a proper app icon
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    corner = int(size * 0.22)   # macOS-style ~22 % corner radius
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=corner, fill=255)
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(img, mask=mask)

    return result


# ── Export ────────────────────────────────────────────────────────────────────

def export_png(path: str, size: int = 512) -> None:
    icon = build_icon(size)
    icon.save(path, "PNG")
    print(f"  ✓ {path}  ({size}×{size})")


def export_icns(out_dir: str) -> None:
    """Build an .icns from a set of PNGs (macOS only)."""
    iconset = os.path.join(out_dir, "app.iconset")
    os.makedirs(iconset, exist_ok=True)

    specs = [
        ("icon_16x16.png",       16),
        ("icon_16x16@2x.png",    32),
        ("icon_32x32.png",       32),
        ("icon_32x32@2x.png",    64),
        ("icon_128x128.png",    128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png",    256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png",    512),
        ("icon_512x512@2x.png",1024),
    ]
    for name, sz in specs:
        export_png(os.path.join(iconset, name), sz)

    icns_path = os.path.join(out_dir, "app.icns")
    result = subprocess.run(
        ["iconutil", "-c", "icns", iconset, "-o", icns_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✓ {icns_path}")
        # Clean up iconset folder
        import shutil
        shutil.rmtree(iconset)
    else:
        print(f"  ✗ iconutil failed: {result.stderr.strip()}")
        print("    (PNG files kept in app.iconset/)")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__))
    print("Generating SUMO V2V Dashboard icons …")
    export_png(os.path.join(out, "app.png"), 512)

    if sys.platform == "darwin":
        print("Building macOS .icns …")
        export_icns(out)

    print("Done.")
