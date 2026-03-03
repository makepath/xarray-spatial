"""
Interactive Cost-Distance Wavefront Simulation
===============================================

Click a source point on the terrain and watch an accessibility wavefront
radiate outward, moving fast across flat ground and slow on steep slopes.
Models evacuation planning and search-and-rescue accessibility.

Uses a Copernicus 30m DEM tile downloaded from the public AWS bucket;
falls back to synthetic terrain if rasterio is unavailable or the
download fails.

Driven by xrspatial cost-distance and pathfinding modules:

  * **cost_distance** -- accumulated least-cost traversal from source
  * **a_star_search** -- least-cost path between two points
  * **slope** -- terrain gradient used as friction weight

Controls
--------
* **Left-click**   -- set source point (launches wavefront)
* **Right-click**  -- set destination (computes A* path from source)
* **Space**        -- pause / resume the wavefront sweep
* **Up / Down**    -- increase / decrease animation speed
* **R**            -- reset (clear source, path, and wavefront)
* **Q / Escape**   -- quit

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
from xrspatial.cost_distance import cost_distance
from xrspatial.pathfinding import a_star_search

# -- Tunable parameters -----------------------------------------------------
CELL_SIZE = 30.0                    # metres per pixel
SLOPE_WEIGHT = 0.15                 # how strongly slope inflates friction
SWEEP_SPEED = 40.0                  # cost-units revealed per frame
MAX_COST = np.inf                   # cap for cost_distance search
FPS = 20
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
template = elevation.copy(data=np.zeros((GRID_H, GRID_W), dtype=np.float32))

print("Computing slope ...")
slope_da = slope(elevation)
slope_vals = np.nan_to_num(slope_da.values, nan=0.0).astype(np.float32)

# Friction surface: base cost of 1 plus slope-proportional penalty.
# Steeper terrain is harder to traverse; flat ground is cheapest.
print("Building friction surface ...")
friction_vals = (1.0 + slope_vals * SLOPE_WEIGHT).astype(np.float32)
# Mark NaN elevation cells as impassable
friction_vals[np.isnan(elevation.values)] = np.nan
friction_da = template.copy(data=friction_vals)

print(f"  Friction range: {np.nanmin(friction_vals):.2f} - "
      f"{np.nanmax(friction_vals):.2f}")

# -- Simulation state -------------------------------------------------------
source_point: tuple[int, int] | None = None     # (row, col)
dest_point: tuple[int, int] | None = None       # (row, col)
cost_surface: np.ndarray | None = None           # result of cost_distance
path_pixels: np.ndarray | None = None            # A* path mask
cost_max: float = 1.0                            # max finite cost value
sweep_threshold: float = 0.0                     # current reveal threshold
paused: bool = False
speed_mult: float = 1.0                          # animation speed multiplier


def set_source(row: int, col: int):
    """Compute cost_distance from the chosen source point."""
    global source_point, cost_surface, cost_max, sweep_threshold
    global dest_point, path_pixels

    source_point = (row, col)
    dest_point = None
    path_pixels = None
    sweep_threshold = 0.0

    src_raster = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    src_raster[row, col] = 1
    src_da = template.copy(data=src_raster)

    print(f"  Computing cost-distance from ({row}, {col}) ...")
    cd = cost_distance(src_da, friction_da, max_cost=MAX_COST)
    cost_surface = cd.values.astype(np.float64)

    finite = np.isfinite(cost_surface)
    cost_max = float(cost_surface[finite].max()) if finite.any() else 1.0
    print(f"  Cost range: 0 - {cost_max:.1f}")


def set_destination(row: int, col: int):
    """Compute A* path from source to destination."""
    global dest_point, path_pixels

    if source_point is None:
        print("  Set a source (left-click) first.")
        return

    dest_point = (row, col)
    print(f"  Computing A* path to ({row}, {col}) ...")

    path_da = a_star_search(
        elevation,
        start=source_point,
        goal=dest_point,
        friction=friction_da,
        snap_start=True,
        snap_goal=True,
    )
    path_vals = path_da.values
    path_pixels = np.isfinite(path_vals)
    n_cells = int(path_pixels.sum())

    if n_cells == 0:
        print("  No path found (destination may be unreachable).")
        path_pixels = None
    else:
        path_cost = float(np.nanmax(path_vals))
        print(f"  Path: {n_cells} cells, cost {path_cost:.1f}")


# -- Visualisation -----------------------------------------------------------

# Wavefront colour map: deep blue (near source) -> cyan -> yellow -> red (far)
wave_cmap = mcolors.LinearSegmentedColormap.from_list("wave", [
    (0.0,  (0.05, 0.10, 0.50)),
    (0.25, (0.10, 0.40, 0.85)),
    (0.50, (0.15, 0.75, 0.75)),
    (0.75, (0.90, 0.80, 0.15)),
    (1.0,  (0.85, 0.15, 0.05)),
])

# Frontier highlight colour map: transparent everywhere except alpha=1
# at the wavefront edge
frontier_cmap = mcolors.LinearSegmentedColormap.from_list("frontier", [
    (0.0, (1.0, 1.0, 1.0, 0.0)),
    (1.0, (1.0, 1.0, 1.0, 0.0)),
])

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Cost-Distance Wavefront  |  L-click: source  |  R-click: A* path  "
    "|  Space: pause  |  Up/Down: speed  |  R: reset",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain layer
terrain_img = ax.imshow(
    elevation.values, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Slope shading overlay (subtle hillshade effect)
slope_shade = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
slope_norm = slope_vals / (slope_vals.max() + 1e-9)
slope_shade[..., 3] = slope_norm * 0.25
slope_img = ax.imshow(slope_shade, origin="lower", aspect="equal")

# Wavefront overlay (updated each frame)
wave_data = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
wave_img = ax.imshow(
    wave_data, cmap=wave_cmap, origin="lower", aspect="equal",
    vmin=0, vmax=1, alpha=0.65, interpolation="nearest",
)

# Frontier highlight overlay
frontier_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
frontier_img = ax.imshow(frontier_rgba, origin="lower", aspect="equal")

# Path overlay
path_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
path_img = ax.imshow(path_rgba, origin="lower", aspect="equal")

# Source and destination markers
(source_marker,) = ax.plot([], [], "w*", markersize=14, markeredgecolor="black")
(dest_marker,) = ax.plot([], [], "r*", markersize=14, markeredgecolor="black")

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="cyan",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Explanation blurb
ax.text(
    0.99, 0.99,
    "Cost-distance measures how hard it is to reach every cell\n"
    "from a source point, accounting for terrain. Flat ground\n"
    "is cheap to cross; steep slopes cost more. The wavefront\n"
    "spreads fast across easy terrain and slows on cliffs.\n"
    "\n"
    "Blue = close/easy to reach    Red = far/hard to reach\n"
    "White ring = expanding frontier\n"
    "Magenta line = least-cost A* path (right-click to set)\n"
    "\n"
    "Used in evacuation planning, search-and-rescue, and\n"
    "finding the easiest route through rough terrain.",
    transform=ax.transAxes, color="white", fontsize=8,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
)


def update_frame(frame: int):
    """Advance the wavefront threshold and redraw."""
    global sweep_threshold

    if cost_surface is None:
        wave_img.set_data(np.full((GRID_H, GRID_W), np.nan))
        frontier_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        status_text.set_text(
            "Left-click to place a source point and launch the wavefront."
        )
        return (wave_img, frontier_img, status_text, path_img)

    if not paused and sweep_threshold < cost_max:
        sweep_threshold = min(
            sweep_threshold + SWEEP_SPEED * speed_mult, cost_max,
        )

    # Reveal cells where cost <= current threshold
    revealed = np.isfinite(cost_surface) & (cost_surface <= sweep_threshold)

    # Colour revealed cells by normalised cost
    wave_display = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
    if revealed.any():
        wave_display[revealed] = cost_surface[revealed] / cost_max

    wave_img.set_data(wave_display)

    # Frontier band: cells near the threshold edge
    frontier_width = SWEEP_SPEED * speed_mult * 2
    near_front = (
        revealed
        & (cost_surface >= sweep_threshold - frontier_width)
        & (cost_surface <= sweep_threshold)
    )
    fr = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
    if near_front.any():
        # Bright white ring at the expanding edge
        closeness = 1.0 - (
            (sweep_threshold - cost_surface[near_front]) / (frontier_width + 1e-9)
        )
        fr[near_front, :3] = 1.0
        fr[near_front, 3] = np.clip(closeness * 0.8, 0, 0.8)
    frontier_img.set_data(fr)

    # Path overlay
    pr = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
    if path_pixels is not None and path_pixels.any():
        pr[path_pixels] = [1.0, 0.2, 0.9, 0.9]
    path_img.set_data(pr)

    # Source marker
    if source_point is not None:
        source_marker.set_data([source_point[1]], [source_point[0]])
    # Destination marker
    if dest_point is not None:
        dest_marker.set_data([dest_point[1]], [dest_point[0]])

    # Stats
    n_revealed = int(revealed.sum())
    n_reachable = int(np.isfinite(cost_surface).sum())
    pct = 100 * n_revealed / (n_reachable + 1e-9)
    done = sweep_threshold >= cost_max

    state = "DONE" if done else ("PAUSED" if paused else "SWEEPING")
    speed_str = f"{speed_mult:.1f}x"
    status_text.set_text(
        f"{state}  |  threshold: {sweep_threshold:.0f} / {cost_max:.0f}  |  "
        f"revealed: {n_revealed:,} / {n_reachable:,} ({pct:.1f}%)  |  "
        f"speed: {speed_str}"
    )

    return (wave_img, frontier_img, status_text, path_img)


def on_click(event):
    """Left-click: set source.  Right-click: set destination / A* path."""
    if event.inaxes != ax:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if not (0 <= row < GRID_H and 0 <= col < GRID_W):
        return

    if event.button == 1:
        set_source(row, col)
    elif event.button == 3:
        set_destination(row, col)


def on_key(event):
    """Keyboard: pause, speed, reset, quit."""
    global paused, speed_mult, sweep_threshold
    global source_point, dest_point, cost_surface, path_pixels

    if event.key == " ":
        paused = not paused
        print("  Paused" if paused else "  Resumed")
    elif event.key == "up":
        speed_mult = min(speed_mult * 1.5, 20.0)
        print(f"  Speed: {speed_mult:.1f}x")
    elif event.key == "down":
        speed_mult = max(speed_mult / 1.5, 0.2)
        print(f"  Speed: {speed_mult:.1f}x")
    elif event.key == "r":
        source_point = None
        dest_point = None
        cost_surface = None
        path_pixels = None
        sweep_threshold = 0.0
        speed_mult = 1.0
        paused = False
        source_marker.set_data([], [])
        dest_marker.set_data([], [])
        wave_img.set_data(np.full((GRID_H, GRID_W), np.nan))
        frontier_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        path_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        print("  Reset")
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

anim = FuncAnimation(
    fig, update_frame, interval=1000 // FPS, blit=False, cache_frame_data=False,
)

plt.tight_layout()
print("\nReady -- left-click the terrain to set a source and watch the "
      "wavefront spread.\n")
plt.show()
