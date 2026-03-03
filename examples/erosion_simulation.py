"""
Interactive Hydraulic Erosion Simulation -- Grand Canyon
========================================================

Watch channels carve into real Grand Canyon terrain as simulated
raindrops erode and deposit sediment.  Uses a Copernicus 30m DEM tile
downloaded from the public AWS bucket; falls back to synthetic terrain
if rasterio is unavailable or the download fails.

Driven by the xrspatial erosion and terrain modules:

  * **erode** -- particle-based hydraulic erosion (Numba-accelerated)
  * **slope** -- terrain slope for visual overlay
  * **generate_terrain** -- procedural fallback terrain

The simulation pre-computes a fully-eroded result, then animates by
interpolating between the original and eroded surfaces.  A diverging
overlay highlights where material was removed (erosion) vs. deposited.

Controls
--------
* **Up / Down**   -- step erosion forward / backward
* **Space**       -- pause / resume automatic progression
* **R**           -- reset to original terrain
* **Q / Escape**  -- quit

Requires: xarray, numpy, matplotlib, xrspatial (this repo)
Optional: rasterio (for real DEM download)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from xrspatial import generate_terrain, slope
from xrspatial.erosion import erode

# -- Tunable parameters -----------------------------------------------------
CELL_SIZE = 30.0                    # metres per pixel
EROSION_ITERATIONS = 100_000        # total droplets for the full erosion run
ALPHA_STEP = 0.02                   # manual step size (Up/Down keys)
AUTO_SPEED = 0.003                  # alpha increment per animation frame
FPS = 30
# ---------------------------------------------------------------------------


def load_dem() -> xr.DataArray:
    """Load a Copernicus 30m DEM subset covering part of the Grand Canyon.

    Downloads a windowed region from the public AWS S3 bucket (no auth
    required).  Falls back to synthetic terrain if rasterio is missing
    or the download fails.
    """
    try:
        import rasterio
        from rasterio.windows import Window

        url = (
            "https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_N36_00_W113_00_DEM/"
            "Copernicus_DSM_COG_10_N36_00_W113_00_DEM.tif"
        )

        print("Downloading Grand Canyon DEM (Copernicus 30m) ...")
        with rasterio.open(url) as src:
            window = Window(col_off=1800, row_off=2400, width=400, height=300)
            data = src.read(1, window=window).astype(np.float64)
            nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan

        h, w = data.shape
        dem = xr.DataArray(data, dims=["y", "x"], name="elevation")
        dem["y"] = np.linspace(h - 1, 0, h)
        dem["x"] = np.linspace(0, w - 1, w)

        print(f"  Loaded DEM: {dem.shape}, "
              f"elevation {np.nanmin(data):.0f} - {np.nanmax(data):.0f} m")
        return dem

    except Exception as e:
        print(f"DEM download failed ({e}), using synthetic terrain")
        h, w = 300, 400
        xs = np.linspace(0, w * CELL_SIZE, w)
        ys = np.linspace(0, h * CELL_SIZE, h)
        template = xr.DataArray(
            np.zeros((h, w), dtype=np.float32),
            dims=["y", "x"],
            coords={"y": ys, "x": xs},
        )
        return generate_terrain(template, zfactor=400, seed=42)


# -- Build the world --------------------------------------------------------
elevation = load_dem()
GRID_H, GRID_W = elevation.shape

print("Computing slope (original terrain) ...")
slope_orig = slope(elevation)
slope_orig_vals = np.nan_to_num(slope_orig.values, nan=0.0)

print(f"Running hydraulic erosion ({EROSION_ITERATIONS:,} droplets) ...")
eroded = erode(elevation, iterations=EROSION_ITERATIONS, seed=42)

print("Computing slope (eroded terrain) ...")
slope_eroded = slope(eroded)
slope_eroded_vals = np.nan_to_num(slope_eroded.values, nan=0.0)

# Difference map: negative = material removed, positive = deposited
elev_orig = elevation.values.astype(np.float64)
elev_eroded = eroded.values.astype(np.float64)
diff = elev_eroded - elev_orig

diff_abs_max = max(np.nanmax(np.abs(diff)), 0.01)
print(f"  Erosion range: {np.nanmin(diff):.2f} to {np.nanmax(diff):.2f} m")
print(f"  Net volume change: {np.nansum(diff) * CELL_SIZE**2:.0f} m^3")

# -- Simulation state -------------------------------------------------------
alpha = 0.0             # 0 = original, 1 = fully eroded
paused = False
auto_play = True

# -- Visualisation -----------------------------------------------------------

# Diverging colourmap for erosion (red) vs deposition (blue)
erosion_cmap = mcolors.LinearSegmentedColormap.from_list("erosion", [
    (0.0,  (0.8, 0.2, 0.1)),       # deep erosion (red)
    (0.3,  (0.95, 0.5, 0.3)),      # moderate erosion (orange)
    (0.5,  (0.95, 0.95, 0.85)),    # neutral (near-white)
    (0.7,  (0.3, 0.55, 0.85)),     # moderate deposition (light blue)
    (1.0,  (0.1, 0.2, 0.7)),       # heavy deposition (blue)
])

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Hydraulic Erosion  |  Up/Down: step  |  Space: pause  "
    "|  R: reset  |  Q: quit",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain layer (interpolated between original and eroded)
current_elev = elev_orig.copy()
terrain_img = ax.imshow(
    current_elev, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Erosion/deposition overlay
erosion_data = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
erosion_img = ax.imshow(
    erosion_data, cmap=erosion_cmap, origin="lower", aspect="equal",
    vmin=-diff_abs_max, vmax=diff_abs_max, alpha=0.55, interpolation="bilinear",
)

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="cyan",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Explanation blurb
ax.text(
    0.99, 0.99,
    "Hydraulic erosion simulates raindrops landing on terrain,\n"
    "picking up sediment as they flow downhill, and depositing\n"
    "it where the water slows down. Over time this carves\n"
    "channels and builds up alluvial fans -- the same process\n"
    "that shaped the Grand Canyon over millions of years.\n"
    "\n"
    "Red/orange = material removed (channel cutting)\n"
    "Blue = material deposited (sediment accumulation)\n"
    "White = little change",
    transform=ax.transAxes, color="white", fontsize=8,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
)


def update_frame(frame: int):
    """Advance erosion interpolation and redraw."""
    global alpha

    if auto_play and not paused:
        alpha = min(alpha + AUTO_SPEED, 1.0)

    # Interpolate terrain between original and fully eroded
    current = elev_orig + alpha * diff
    terrain_img.set_data(current)

    # Update colour limits to track changing elevation range
    vmin = np.nanmin(current)
    vmax = np.nanmax(current)
    terrain_img.set_clim(vmin=vmin, vmax=vmax)

    # Erosion/deposition overlay -- scale by current alpha
    if alpha > 0:
        current_diff = alpha * diff
        erosion_img.set_data(current_diff)
        overlay_max = max(np.nanmax(np.abs(current_diff)), 0.01)
        erosion_img.set_clim(vmin=-overlay_max, vmax=overlay_max)
        erosion_img.set_alpha(min(0.55, alpha * 2))
    else:
        erosion_img.set_data(np.full((GRID_H, GRID_W), np.nan))
        erosion_img.set_alpha(0.0)

    # Interpolate slope for stats
    current_slope = slope_orig_vals + alpha * (slope_eroded_vals - slope_orig_vals)

    # Stats
    current_diff_vals = alpha * diff
    eroded_cells = np.sum(current_diff_vals < -0.01)
    deposited_cells = np.sum(current_diff_vals > 0.01)
    max_erosion = np.nanmin(current_diff_vals)
    max_deposition = np.nanmax(current_diff_vals)
    mean_slope = np.nanmean(current_slope)

    state = "PAUSED" if paused else ("ERODING" if alpha < 1.0 else "COMPLETE")
    status_text.set_text(
        f"{state}  |  progress: {100 * alpha:.1f}%  |  "
        f"eroded: {eroded_cells:,} cells  |  deposited: {deposited_cells:,} cells  |  "
        f"max cut: {max_erosion:.2f} m  |  max fill: {max_deposition:.2f} m  |  "
        f"mean slope: {mean_slope:.1f} deg"
    )

    return (terrain_img, erosion_img, status_text)


def on_key(event):
    """Keyboard: erosion step, pause, reset, quit."""
    global alpha, paused, auto_play

    if event.key == "up":
        auto_play = False
        alpha = min(alpha + ALPHA_STEP, 1.0)
    elif event.key == "down":
        auto_play = False
        alpha = max(alpha - ALPHA_STEP, 0.0)
    elif event.key == " ":
        paused = not paused
        print("  Paused" if paused else "  Resumed")
    elif event.key == "r":
        alpha = 0.0
        auto_play = True
        paused = False
        print("  Reset")
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

anim = FuncAnimation(
    fig, update_frame, interval=1000 // FPS, blit=False, cache_frame_data=False,
)

plt.tight_layout()
print("\nReady -- erosion will progress automatically. "
      "Press Up/Down for manual control.\n")
plt.show()
