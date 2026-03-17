"""
Interactive Watershed Explorer -- Grand Canyon
===============================================

Click anywhere on the terrain to delineate the watershed that drains
through that point.  Each click adds a new basin in a distinct colour.
The stream network is drawn on startup, coloured by Strahler order.

Driven by the xrspatial hydrology modules:

  * **flow_direction** -- D8 flow routing
  * **flow_accumulation** -- upstream cell counts
  * **stream_order** -- Strahler stream ordering
  * **snap_pour_point** -- snap click to nearest high-accumulation cell
  * **watershed** -- delineate the basin draining to a pour point
  * **basin** -- automatic delineation of all drainage basins

Controls
--------
* **Left-click**  -- delineate watershed at clicked location
* **A**           -- toggle automatic basin overlay (all basins)
* **R**           -- clear all user-delineated watersheds
* **Q / Escape**  -- quit

Requires: xarray, numpy, matplotlib, xrspatial (this repo)
Optional: rasterio (for real DEM download)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from xrspatial import generate_terrain
from xrspatial.hydro.flow_direction_d8 import flow_direction_d8 as flow_direction
from xrspatial.hydro.flow_accumulation_d8 import flow_accumulation_d8 as flow_accumulation
from xrspatial.hydro.stream_order_d8 import stream_order_d8 as stream_order
from xrspatial.hydro.snap_pour_point_d8 import snap_pour_point_d8 as snap_pour_point
from xrspatial.hydro.watershed_d8 import watershed_d8 as watershed
from xrspatial.hydro.basin_d8 import basin_d8 as basin

# -- Tunable parameters -----------------------------------------------------
CELL_SIZE = 30.0                    # metres per pixel
STREAM_THRESHOLD = 100              # flow-accum cells to define a stream
SNAP_RADIUS = 10                    # pixels to snap pour point
# ---------------------------------------------------------------------------

# Palette for user-delineated watersheds (up to 12, then cycles)
BASIN_COLORS = [
    (0.12, 0.47, 0.71, 0.40),  # blue
    (1.00, 0.50, 0.05, 0.40),  # orange
    (0.17, 0.63, 0.17, 0.40),  # green
    (0.84, 0.15, 0.16, 0.40),  # red
    (0.58, 0.40, 0.74, 0.40),  # purple
    (0.55, 0.34, 0.29, 0.40),  # brown
    (0.89, 0.47, 0.76, 0.40),  # pink
    (0.50, 0.50, 0.50, 0.40),  # grey
    (0.74, 0.74, 0.13, 0.40),  # olive
    (0.09, 0.75, 0.81, 0.40),  # cyan
    (0.94, 0.89, 0.26, 0.40),  # yellow
    (0.64, 0.08, 0.18, 0.40),  # crimson
]


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

print("Computing flow direction ...")
flow_dir_da = flow_direction(elevation)

print("Computing flow accumulation ...")
flow_accum_da = flow_accumulation(flow_dir_da)
flow_accum_vals = flow_accum_da.values

print("Computing stream order (Strahler) ...")
stream_order_da = stream_order(flow_dir_da, flow_accum_da,
                               threshold=STREAM_THRESHOLD)
stream_order_vals = stream_order_da.values

# Stream network mask
stream_mask = flow_accum_vals >= STREAM_THRESHOLD
max_order = int(np.nanmax(stream_order_vals[stream_mask])) if stream_mask.any() else 1

print(f"  Stream cells: {stream_mask.sum():,} "
      f"({100 * stream_mask.sum() / (GRID_H * GRID_W):.1f}%)")
print(f"  Max Strahler order: {max_order}")

print("Pre-computing basins (for auto-basin toggle) ...")
basin_da = basin(flow_dir_da)
basin_vals = basin_da.values

# Count unique basins (excluding NaN)
unique_basins = np.unique(basin_vals[~np.isnan(basin_vals)])
print(f"  Total basins: {len(unique_basins)}")

# -- Simulation state -------------------------------------------------------
# Each entry: (row, col, label, watershed_mask)
user_watersheds: list[tuple[int, int, int, np.ndarray]] = []
next_label = 1
show_auto_basins = False

# -- Visualisation -----------------------------------------------------------

# Stream order colour map: thin light blue -> thick dark blue
stream_cmap = mcolors.LinearSegmentedColormap.from_list("stream", [
    (0.0, (0.6, 0.85, 1.0)),
    (0.5, (0.2, 0.5,  0.9)),
    (1.0, (0.0, 0.15, 0.5)),
])

# Auto-basin colour map (random-looking distinct colours)
auto_basin_cmap = plt.cm.tab20

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Watershed Explorer  |  L-click: delineate  |  A: auto-basins  "
    "|  R: clear  |  Q: quit",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain layer
terrain_img = ax.imshow(
    elevation.values, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Stream network overlay (coloured by Strahler order)
stream_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
if stream_mask.any():
    orders = stream_order_vals[stream_mask]
    normed = (orders - 1) / max(max_order - 1, 1)
    colors = stream_cmap(normed)
    stream_rgba[stream_mask] = colors
    # Boost alpha for higher-order streams
    stream_rgba[stream_mask, 3] = 0.4 + 0.5 * normed
stream_img = ax.imshow(stream_rgba, origin="lower", aspect="equal")

# User watershed overlay
watershed_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
watershed_img = ax.imshow(watershed_rgba, origin="lower", aspect="equal")

# Auto-basin overlay (hidden by default)
auto_basin_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
auto_basin_img = ax.imshow(auto_basin_rgba, origin="lower", aspect="equal")

# Pour-point markers
(pour_marker,) = ax.plot([], [], "w^", markersize=8, markeredgecolor="black",
                         markeredgewidth=0.8)

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="cyan",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Explanation blurb
ax.text(
    0.99, 0.99,
    "Every point on a landscape drains to an outlet. A watershed\n"
    "is the full area of land that funnels water through a single\n"
    "point. Click anywhere and the colored region shows every\n"
    "cell whose rain eventually flows through your click.\n"
    "\n"
    "Stream lines are colored by Strahler order: thin headwater\n"
    "tributaries (light) merge into larger trunk streams (dark).\n"
    "\n"
    "Watershed boundaries matter for flood risk, water supply\n"
    "planning, and tracking where pollutants end up.",
    transform=ax.transAxes, color="white", fontsize=8,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
)


def build_auto_basin_overlay():
    """Build an RGBA overlay of all basins with distinct colours."""
    rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
    valid = ~np.isnan(basin_vals)
    if not valid.any():
        return rgba
    # Map each basin ID to a colour index
    ids = basin_vals[valid]
    # Use modular indexing into the colourmap
    colour_idx = (ids.astype(np.int64) % 20) / 20.0
    colors = auto_basin_cmap(colour_idx)
    rgba[valid] = colors
    rgba[valid, 3] = 0.30
    return rgba


def rebuild_watershed_overlay():
    """Rebuild the combined user-watershed RGBA overlay."""
    rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
    for i, (_, _, _, mask) in enumerate(user_watersheds):
        color = BASIN_COLORS[i % len(BASIN_COLORS)]
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = color[3]
    return rgba


def update_display():
    """Redraw all overlays and status text."""
    # User watersheds
    watershed_img.set_data(rebuild_watershed_overlay())

    # Auto-basin toggle
    if show_auto_basins:
        auto_basin_img.set_data(build_auto_basin_overlay())
    else:
        auto_basin_img.set_data(
            np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
        )

    # Pour-point markers
    if user_watersheds:
        cols = [c for _, c, _, _ in user_watersheds]
        rows = [r for r, _, _, _ in user_watersheds]
        pour_marker.set_data(cols, rows)
    else:
        pour_marker.set_data([], [])

    # Status
    n_ws = len(user_watersheds)
    total_cells = sum(m.sum() for _, _, _, m in user_watersheds)
    pct = 100 * total_cells / (GRID_H * GRID_W) if total_cells > 0 else 0.0
    auto_str = "ON" if show_auto_basins else "OFF"
    status_text.set_text(
        f"watersheds: {n_ws}  |  "
        f"delineated: {int(total_cells):,} cells ({pct:.1f}%)  |  "
        f"auto-basins: {auto_str}  |  "
        f"streams: order 1-{max_order}"
    )

    fig.canvas.draw_idle()


def delineate_watershed(row: int, col: int):
    """Snap to the nearest stream cell and delineate its watershed."""
    global next_label

    # Build a pour-point raster with a single marked cell
    pp_data = np.full((GRID_H, GRID_W), np.nan, dtype=np.float64)
    pp_data[row, col] = float(next_label)
    pp_da = elevation.copy(data=pp_data)

    # Snap to the highest accumulation cell within the search radius
    snapped_da = snap_pour_point(flow_accum_da, pp_da,
                                 search_radius=SNAP_RADIUS)
    snapped_vals = snapped_da.values

    # Find where the snapped point ended up
    snapped_locs = np.argwhere(~np.isnan(snapped_vals))
    if len(snapped_locs) == 0:
        print(f"  No valid pour point near ({row}, {col})")
        return
    snap_row, snap_col = snapped_locs[0]
    accum_at = flow_accum_vals[snap_row, snap_col]

    # Run watershed delineation
    ws_da = watershed(flow_dir_da, snapped_da)
    ws_vals = ws_da.values
    ws_mask = ws_vals == float(next_label)

    n_cells = int(ws_mask.sum())
    area_km2 = n_cells * (CELL_SIZE ** 2) / 1e6
    print(f"  Watershed #{next_label} at ({snap_row}, {snap_col}), "
          f"accum={accum_at:.0f}, "
          f"cells={n_cells:,}, area={area_km2:.1f} km2")

    user_watersheds.append((snap_row, snap_col, next_label, ws_mask))
    next_label += 1
    update_display()


def on_click(event):
    """Left-click: delineate watershed at clicked location."""
    if event.inaxes != ax or event.button != 1:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if not (0 <= row < GRID_H and 0 <= col < GRID_W):
        return
    delineate_watershed(row, col)


def on_key(event):
    """Keyboard: auto-basins toggle, clear, quit."""
    global show_auto_basins, next_label

    if event.key == "a":
        show_auto_basins = not show_auto_basins
        print(f"  Auto-basins {'ON' if show_auto_basins else 'OFF'}")
        update_display()
    elif event.key == "r":
        user_watersheds.clear()
        next_label = 1
        print("  Cleared all watersheds")
        update_display()
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

# Initial display
update_display()

plt.tight_layout()
print("\nReady -- click the map to delineate a watershed.\n")
plt.show()
