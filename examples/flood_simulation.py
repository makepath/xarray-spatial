"""
Interactive Flood Simulation -- Grand Canyon
=============================================

Watch a flood spread across real Grand Canyon terrain as the water
level rises.  Uses a Copernicus 30m DEM tile downloaded from the public
AWS bucket; falls back to synthetic terrain if rasterio is unavailable
or the download fails.

Driven by the xrspatial flood and hydrology modules:

  * **flow_direction** -- D8 flow routing
  * **flow_accumulation** -- upstream cell counts / stream network
  * **hand** -- Height Above Nearest Drainage
  * **flood_depth** -- depth of flooding at a given water level
  * **inundation** -- binary flood extent mask
  * **curve_number_runoff** -- SCS/NRCS rainfall-runoff estimation
  * **travel_time** -- overland flow travel time (Manning's equation)
  * **flow_length** -- distance along flow path to outlet

Controls
--------
* **Up / Down**   -- raise / lower water level
* **Left-click**  -- trigger a rainfall event (shows runoff via curve number)
* **Space**       -- pause / resume automatic water rise
* **T**           -- open travel time analysis window
* **R**           -- reset water level to zero
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
from xrspatial.hydro.flow_direction import flow_direction
from xrspatial.hydro.flow_accumulation import flow_accumulation
from xrspatial.hydro.flow_length import flow_length
from xrspatial.hydro.hand import hand
from xrspatial.flood import (
    curve_number_runoff,
    flood_depth,
    inundation,
    travel_time,
)

# -- Tunable parameters -----------------------------------------------------
CELL_SIZE = 30.0                    # metres per pixel
STREAM_THRESHOLD = 80               # flow-accum cells to define a stream
WATER_LEVEL_STEP = 0.5              # metres per key press / auto-tick
AUTO_RISE_RATE = 0.15               # metres per animation frame when auto-rising
MAX_WATER_LEVEL = 80.0              # cap for water level
CURVE_NUMBER = 75                   # SCS curve number (urban/suburban mix)
MANNINGS_N = 0.035                  # Manning's roughness (mixed ground)
RAINFALL_AMOUNT = 60.0              # mm of rain per click event
RAINFALL_RADIUS = 40                # pixel radius of rain event
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
slope_da = template.copy(data=slope_vals)

print("Computing flow direction ...")
flow_dir_da = flow_direction(elevation)

print("Computing flow accumulation ...")
flow_accum_da = flow_accumulation(flow_dir_da)
flow_accum_vals = flow_accum_da.values

print("Computing HAND (Height Above Nearest Drainage) ...")
hand_da = hand(flow_dir_da, flow_accum_da, elevation, threshold=STREAM_THRESHOLD)
hand_vals = hand_da.values

print("Computing flow length ...")
flow_len_da = flow_length(flow_dir_da, direction='downstream')

print("Computing travel time (Manning's equation) ...")
travel_time_da = travel_time(flow_len_da, slope_da, mannings_n=MANNINGS_N)
travel_time_vals = travel_time_da.values

# Stream network mask (for overlay)
stream_mask = flow_accum_vals >= STREAM_THRESHOLD

print(f"  HAND range: {np.nanmin(hand_vals):.1f} - {np.nanmax(hand_vals):.1f} m")
print(f"  Stream cells: {stream_mask.sum():,} "
      f"({100 * stream_mask.sum() / (GRID_H * GRID_W):.1f}%)")

# -- Simulation state -------------------------------------------------------
water_level = 0.0
paused = False
auto_rise = True
rainfall_events: list[tuple[int, int]] = []  # (row, col) of rain clicks


def compute_flood(wl: float):
    """Compute inundation mask and flood depth at the given water level."""
    inund = inundation(hand_da, wl)
    depth = flood_depth(hand_da, wl)
    return inund.values, depth.values


def trigger_rainfall(row: int, col: int):
    """Apply a rainfall event and show curve-number runoff."""
    rain_vals = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # Circular rainfall footprint
    yy, xx = np.ogrid[:GRID_H, :GRID_W]
    dist_sq = (yy - row) ** 2 + (xx - col) ** 2
    inside = dist_sq <= RAINFALL_RADIUS ** 2
    # Taper rainfall from center to edge
    dist_norm = np.sqrt(dist_sq[inside].astype(np.float32)) / RAINFALL_RADIUS
    rain_vals[inside] = RAINFALL_AMOUNT * (1.0 - 0.5 * dist_norm)

    rainfall_da = template.copy(data=rain_vals)
    runoff_da = curve_number_runoff(rainfall_da, curve_number=CURVE_NUMBER)
    runoff_vals = runoff_da.values

    total_rain = np.nansum(rain_vals)
    total_runoff = np.nansum(runoff_vals)
    pct = 100 * total_runoff / (total_rain + 1e-9)
    print(f"  Rain event at ({row}, {col}): "
          f"rainfall={total_rain:.0f} mm total, "
          f"runoff={total_runoff:.0f} mm total ({pct:.0f}%), "
          f"CN={CURVE_NUMBER}")

    rainfall_events.append((row, col))
    return rain_vals, runoff_vals


def show_travel_time():
    """Open a window with travel time and flow accumulation analysis."""
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Flood Analysis (xrspatial.flood + hydrology)", fontsize=13)

    # Travel time
    tt_display = travel_time_vals.copy()
    tt_display[np.isnan(tt_display)] = 0
    tt_max = np.nanpercentile(tt_display[tt_display > 0], 99)
    im0 = axes[0].imshow(
        tt_display, cmap="plasma", origin="lower", vmin=0, vmax=tt_max,
    )
    axes[0].set_title("Travel Time (s)")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    # Flow accumulation (log scale)
    fa_display = np.log10(flow_accum_vals + 1)
    im1 = axes[1].imshow(fa_display, cmap="Blues", origin="lower")
    axes[1].set_title("Flow Accumulation (log10)")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    # HAND
    hand_display = hand_vals.copy()
    hand_p99 = np.nanpercentile(hand_display[~np.isnan(hand_display)], 99)
    im2 = axes[2].imshow(
        hand_display, cmap="YlOrBr", origin="lower", vmin=0, vmax=hand_p99,
    )
    axes[2].set_title("HAND (m)")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    plt.show(block=False)


# -- Visualisation -----------------------------------------------------------

# Flood depth colour map: transparent -> light blue -> deep blue
flood_cmap = mcolors.LinearSegmentedColormap.from_list("flood", [
    (0.0,  (0.7, 0.85, 1.0)),
    (0.2,  (0.3, 0.6,  0.9)),
    (0.5,  (0.1, 0.4,  0.8)),
    (0.8,  (0.0, 0.2,  0.6)),
    (1.0,  (0.0, 0.1,  0.35)),
])

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Flood Simulation  |  Up/Down: water level  |  L-click: rain  "
    "|  Space: pause  |  T: analysis  |  R: reset",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain layer
terrain_img = ax.imshow(
    elevation.values, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Stream network overlay (thin blue lines)
stream_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
stream_rgba[stream_mask] = [0.1, 0.3, 0.8, 0.7]
stream_img = ax.imshow(stream_rgba, origin="lower", aspect="equal")

# Flood overlay (updated each frame)
flood_data = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
flood_img = ax.imshow(
    flood_data, cmap=flood_cmap, origin="lower", aspect="equal",
    vmin=0, vmax=20, alpha=0.75, interpolation="nearest",
)

# Rainfall overlay
rain_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
rain_img = ax.imshow(rain_rgba, origin="lower", aspect="equal")

# Rain event markers
(rain_marker,) = ax.plot([], [], "c*", markersize=10, markeredgecolor="white")

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="cyan",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Track latest runoff overlay for display
last_runoff: dict[str, np.ndarray | None] = {"vals": None, "fade": 0.0}


def update_frame(frame: int):
    """Advance water level and redraw."""
    global water_level

    if auto_rise and not paused:
        water_level = min(water_level + AUTO_RISE_RATE, MAX_WATER_LEVEL)

    if water_level <= 0:
        flood_img.set_data(np.full((GRID_H, GRID_W), np.nan))
        status_text.set_text(
            "Water level: 0.0 m  |  Press Up or Space to start flooding"
        )
        return (flood_img, status_text, rain_img)

    inund_vals, depth_vals = compute_flood(water_level)

    # Flood depth display
    flood_display = depth_vals.copy()
    flood_display[np.isnan(flood_display)] = np.nan
    flood_img.set_data(flood_display)

    # Adjust colour scale to current max depth
    max_depth = np.nanmax(depth_vals) if not np.all(np.isnan(depth_vals)) else 1.0
    flood_img.set_clim(vmin=0, vmax=max(max_depth, 1.0))

    # Fade runoff overlay
    if last_runoff["vals"] is not None and last_runoff["fade"] > 0:
        last_runoff["fade"] -= 0.02
        rv = last_runoff["vals"]
        rr = rain_rgba.copy()
        ro_max = rv.max() + 1e-9
        show = rv > 0
        rr[show, 0] = 0.0
        rr[show, 1] = 0.8
        rr[show, 2] = 0.4
        rr[show, 3] = (rv[show] / ro_max) * last_runoff["fade"] * 0.6
        rain_img.set_data(rr)
        if last_runoff["fade"] <= 0:
            rain_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
            last_runoff["vals"] = None

    # Rain markers
    if rainfall_events:
        cols, rows = zip(*[(c, r) for r, c in rainfall_events])
        rain_marker.set_data(cols, rows)

    # Stats
    n_inundated = int(np.nansum(inund_vals))
    pct = 100 * n_inundated / (GRID_H * GRID_W)
    mean_depth = np.nanmean(depth_vals[~np.isnan(depth_vals)]) if n_inundated > 0 else 0.0
    max_d = np.nanmax(depth_vals) if n_inundated > 0 else 0.0

    state = "PAUSED" if paused else ("RISING" if auto_rise else "MANUAL")
    status_text.set_text(
        f"{state}  |  water level: {water_level:.1f} m  |  "
        f"flooded: {n_inundated:,} cells ({pct:.1f}%)  |  "
        f"depth: mean {mean_depth:.1f} m, max {max_d:.1f} m"
    )

    return (flood_img, status_text, rain_img)


def on_click(event):
    """Left-click: trigger rainfall event."""
    if event.inaxes != ax or event.button != 1:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if not (0 <= row < GRID_H and 0 <= col < GRID_W):
        return

    rain_vals, runoff_vals = trigger_rainfall(row, col)
    last_runoff["vals"] = runoff_vals
    last_runoff["fade"] = 1.0


def on_key(event):
    """Keyboard: water level, pause, analysis, reset, quit."""
    global water_level, paused, auto_rise

    if event.key == "up":
        auto_rise = False
        water_level = min(water_level + WATER_LEVEL_STEP, MAX_WATER_LEVEL)
    elif event.key == "down":
        auto_rise = False
        water_level = max(water_level - WATER_LEVEL_STEP, 0.0)
    elif event.key == " ":
        paused = not paused
        print("  Paused" if paused else "  Resumed")
    elif event.key == "t":
        show_travel_time()
    elif event.key == "r":
        water_level = 0.0
        auto_rise = True
        paused = False
        rainfall_events.clear()
        rain_marker.set_data([], [])
        rain_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        last_runoff["vals"] = None
        last_runoff["fade"] = 0.0
        print("  Reset")
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

anim = FuncAnimation(
    fig, update_frame, interval=1000 // FPS, blit=False, cache_frame_data=False,
)

plt.tight_layout()
print("\nReady -- water level will rise automatically. "
      "Press Up/Down for manual control.\n")
plt.show()
