"""
Interactive Contour Explorer
============================

Explore elevation contour lines on a procedurally-generated landscape.
Click the terrain to add contour levels, scroll to adjust density,
and toggle display modes with keyboard shortcuts.

The contour extraction is powered by ``xrspatial.contours``, which
uses a marching-squares algorithm with Numba-JIT acceleration.

Controls
--------
* **Left-click**   -- add a contour at the clicked elevation
* **Right-click**  -- remove the nearest custom contour level
* **Scroll up**    -- increase number of evenly-spaced contours
* **Scroll down**  -- decrease number of evenly-spaced contours
* **F**            -- toggle filled regions between contours
* **H**            -- toggle hillshade underlay
* **L**            -- toggle contour elevation labels
* **I**            -- toggle index contours (every 5th line thicker)
* **C**            -- cycle colour scheme (terrain / inferno / mono)
* **R**            -- reset to defaults
* **Q / Escape**   -- quit

Requires: xarray, numpy, matplotlib, xrspatial (this repo)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from xrspatial import generate_terrain, contours
from xrspatial.hillshade import hillshade

# -- Tunable parameters -------------------------------------------------------
GRID_H, GRID_W = 300, 400
ZFACTOR = 800
SEED = 77
INITIAL_N_LEVELS = 12
COLOR_SCHEMES = ["terrain", "inferno", "mono"]
# ------------------------------------------------------------------------------


def make_terrain():
    """Generate synthetic elevation."""
    xs = np.linspace(0, GRID_W, GRID_W)
    ys = np.linspace(0, GRID_H, GRID_H)
    template = xr.DataArray(
        np.zeros((GRID_H, GRID_W), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": ys, "x": xs},
    )
    elevation = generate_terrain(template, zfactor=ZFACTOR, seed=SEED)
    return elevation


def extract_contours(elevation, levels):
    """Run xrspatial.contours and return list of (level, coords) tuples."""
    if len(levels) == 0:
        return []
    return contours(elevation, levels=sorted(levels))


# -- Build the world -----------------------------------------------------------
print("Generating terrain ...")
elevation = make_terrain()
elev_vals = elevation.values
vmin, vmax = float(np.nanmin(elev_vals)), float(np.nanmax(elev_vals))

print("Computing hillshade ...")
hillshade_da = hillshade(elevation)
hillshade_vals = hillshade_da.values

# -- State ---------------------------------------------------------------------
n_auto_levels = INITIAL_N_LEVELS
custom_levels: list[float] = []
show_filled = False
show_hillshade = True
show_labels = False
show_index = True
color_idx = 0

# Cached contour results and artist handles
contour_artists: list = []
fill_artists: list = []
label_artists: list = []


def get_all_levels():
    """Combine auto and custom levels, deduplicating."""
    margin = (vmax - vmin) * 0.02
    auto = np.linspace(vmin + margin, vmax - margin, max(n_auto_levels, 1))
    combined = set(auto.tolist())
    combined.update(custom_levels)
    return sorted(combined)


def get_contour_color(level, levels_list):
    """Pick line colour based on current scheme."""
    scheme = COLOR_SCHEMES[color_idx % len(COLOR_SCHEMES)]
    if scheme == "mono":
        return "sienna"
    norm_val = (level - vmin) / (vmax - vmin + 1e-9)
    if scheme == "terrain":
        return plt.cm.gist_earth(0.25 + 0.65 * norm_val)
    elif scheme == "inferno":
        return plt.cm.inferno(norm_val)
    return "black"


def is_index_contour(idx, total):
    """Every 5th contour is an index contour (drawn thicker)."""
    if total <= 5:
        return False
    return idx % 5 == 0


def clear_contour_artists():
    """Remove all contour line/fill/label artists from axes."""
    for a in contour_artists:
        a.remove()
    contour_artists.clear()
    for a in fill_artists:
        a.remove()
    fill_artists.clear()
    for a in label_artists:
        a.remove()
    label_artists.clear()


def draw_contours():
    """Extract and draw all contour lines."""
    clear_contour_artists()
    levels_list = get_all_levels()

    if not levels_list:
        fig.canvas.draw_idle()
        return

    # Extract contours via xrspatial
    lines = extract_contours(elevation, levels_list)

    # Draw filled regions if enabled
    if show_filled and len(levels_list) >= 2:
        scheme = COLOR_SCHEMES[color_idx % len(COLOR_SCHEMES)]
        if scheme == "mono":
            fill_cmap = "YlOrBr_r"
        elif scheme == "inferno":
            fill_cmap = "inferno"
        else:
            fill_cmap = "gist_earth"
        # Use matplotlib's contourf for fills (our contour lines overlay on top)
        cf = ax.contourf(
            elev_vals, levels=levels_list, cmap=fill_cmap,
            origin="lower", alpha=0.35, extend="both",
        )
        fill_artists.extend(cf.collections)

    # Group lines by level for index-contour detection
    level_to_idx = {lvl: i for i, lvl in enumerate(levels_list)}

    for level, coords in lines:
        if len(coords) < 2:
            continue

        idx_in_list = level_to_idx.get(level, 0)
        is_idx = show_index and is_index_contour(idx_in_list, len(levels_list))
        lw = 1.4 if is_idx else 0.6
        color = get_contour_color(level, levels_list)

        # Check if this is a custom level (draw dashed)
        is_custom = level in custom_levels
        ls = "--" if is_custom else "-"

        line_artist, = ax.plot(
            coords[:, 1], coords[:, 0],
            color=color, linewidth=lw, linestyle=ls, alpha=0.85,
        )
        contour_artists.append(line_artist)

        # Labels on index contours
        if show_labels and is_idx and len(coords) > 4:
            mid = len(coords) // 2
            txt = ax.text(
                coords[mid, 1], coords[mid, 0], f"{level:.0f}",
                fontsize=7, color="white", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                          alpha=0.7, edgecolor="none"),
            )
            label_artists.append(txt)

    update_status()
    fig.canvas.draw_idle()


def update_terrain_display():
    """Update hillshade and terrain base layers."""
    if show_hillshade:
        hillshade_img.set_alpha(1.0)
        terrain_img.set_alpha(0.4)
    else:
        hillshade_img.set_alpha(0.0)
        terrain_img.set_alpha(1.0)
    fig.canvas.draw_idle()


def update_status():
    """Update the status bar text."""
    levels_list = get_all_levels()
    n_custom = len(custom_levels)
    scheme = COLOR_SCHEMES[color_idx % len(COLOR_SCHEMES)]

    parts = [
        f"levels: {len(levels_list)} ({n_auto_levels} auto",
    ]
    if n_custom > 0:
        parts[0] += f" + {n_custom} custom"
    parts[0] += ")"
    parts.append(f"scheme: {scheme}")

    flags = []
    if show_filled:
        flags.append("filled")
    if show_hillshade:
        flags.append("hillshade")
    if show_labels:
        flags.append("labels")
    if show_index:
        flags.append("index")
    if flags:
        parts.append(" ".join(flags))

    status_text.set_text("  |  ".join(parts))


# -- Visualisation -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_title(
    "Contour Explorer  |  L-click: add level  |  R-click: remove  "
    "|  Scroll: density  |  F: fill  |  H: hillshade  |  L: labels",
    color="white", fontsize=10,
)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

# Hillshade layer
hillshade_img = ax.imshow(
    hillshade_vals, cmap="gray", origin="lower",
    aspect="equal", interpolation="bilinear", alpha=1.0,
)

# Terrain colour layer
terrain_img = ax.imshow(
    elev_vals, cmap="gist_earth", origin="lower",
    aspect="equal", interpolation="bilinear", alpha=0.4,
)

# Elevation readout under cursor
elev_text = ax.text(
    0.99, 0.01, "", transform=ax.transAxes, color="white",
    fontsize=9, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Status bar
status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="yellow",
    fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Click markers for custom levels
(custom_markers,) = ax.plot(
    [], [], "w+", markersize=8, markeredgewidth=1.5, alpha=0.8,
)
marker_positions: list[tuple[float, float]] = []  # (col, row)


# -- Event handlers ------------------------------------------------------------

def on_click(event):
    """Left-click: add contour at that elevation. Right-click: remove nearest."""
    if event.inaxes != ax:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if not (0 <= row < GRID_H and 0 <= col < GRID_W):
        return

    if event.button == 1:
        elev = float(elev_vals[row, col])
        if np.isnan(elev):
            return
        # Snap to nearest 10 for cleaner levels
        snapped = round(elev / 10) * 10
        if snapped not in custom_levels:
            custom_levels.append(snapped)
            marker_positions.append((col, row))
            print(f"  Added contour at elevation {snapped:.0f} "
                  f"(clicked {elev:.1f} at pixel {row}, {col})")
            _update_markers()
            draw_contours()

    elif event.button == 3:
        if not custom_levels:
            return
        elev = float(elev_vals[row, col])
        # Find closest custom level to the clicked elevation
        dists = [abs(lvl - elev) for lvl in custom_levels]
        idx = int(np.argmin(dists))
        removed = custom_levels.pop(idx)
        if idx < len(marker_positions):
            marker_positions.pop(idx)
        print(f"  Removed contour at elevation {removed:.0f}")
        _update_markers()
        draw_contours()


def _update_markers():
    """Update custom-level marker positions."""
    if marker_positions:
        xs, ys = zip(*marker_positions)
        custom_markers.set_data(xs, ys)
    else:
        custom_markers.set_data([], [])


def on_scroll(event):
    """Scroll to adjust number of auto contour levels."""
    global n_auto_levels
    if event.inaxes != ax:
        return
    if event.button == "up":
        n_auto_levels = min(n_auto_levels + 2, 80)
    elif event.button == "down":
        n_auto_levels = max(n_auto_levels - 2, 0)
    print(f"  Auto levels: {n_auto_levels}")
    draw_contours()


def on_motion(event):
    """Show elevation under cursor."""
    if event.inaxes != ax:
        elev_text.set_text("")
        fig.canvas.draw_idle()
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if 0 <= row < GRID_H and 0 <= col < GRID_W:
        e = elev_vals[row, col]
        elev_text.set_text(f"elev: {e:.1f} m  ({row}, {col})")
    else:
        elev_text.set_text("")
    fig.canvas.draw_idle()


def on_key(event):
    """Keyboard shortcuts."""
    global show_filled, show_hillshade, show_labels, show_index
    global color_idx, n_auto_levels

    if event.key == "f":
        show_filled = not show_filled
        print(f"  Filled: {'on' if show_filled else 'off'}")
        draw_contours()

    elif event.key == "h":
        show_hillshade = not show_hillshade
        print(f"  Hillshade: {'on' if show_hillshade else 'off'}")
        update_terrain_display()
        draw_contours()

    elif event.key == "l":
        show_labels = not show_labels
        print(f"  Labels: {'on' if show_labels else 'off'}")
        draw_contours()

    elif event.key == "i":
        show_index = not show_index
        print(f"  Index contours: {'on' if show_index else 'off'}")
        draw_contours()

    elif event.key == "c":
        color_idx = (color_idx + 1) % len(COLOR_SCHEMES)
        scheme = COLOR_SCHEMES[color_idx]
        print(f"  Colour scheme: {scheme}")
        draw_contours()

    elif event.key == "r":
        n_auto_levels = INITIAL_N_LEVELS
        custom_levels.clear()
        marker_positions.clear()
        _update_markers()
        show_filled = False
        show_hillshade = True
        show_labels = False
        show_index = True
        color_idx = 0
        update_terrain_display()
        draw_contours()
        print("  Reset to defaults")

    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)

# -- Initial draw --------------------------------------------------------------
print("Extracting initial contours ...")
draw_contours()

plt.tight_layout()
print("\nReady -- scroll to adjust contour density, click to add levels!\n")
plt.show()
