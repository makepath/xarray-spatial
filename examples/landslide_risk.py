"""
Interactive Landslide Susceptibility Analysis
==============================================

Combine multiple terrain-derived factors into a weighted landslide risk
score and explore them interactively.  Uses a Copernicus 30m DEM tile
downloaded from the public AWS bucket; falls back to synthetic terrain
if rasterio is unavailable or the download fails.

Driven by the xrspatial terrain analysis modules:

  * **slope** -- terrain gradient (steep = higher risk)
  * **curvature** -- surface concavity/convexity (concave = higher risk)
  * **flow_direction / flow_accumulation** -- D8 hydrology
  * **twi** -- Topographic Wetness Index (wet = higher risk)
  * **tpi** -- Topographic Position Index (valleys = higher risk)

Risk model
----------
Each factor is normalised to [0, 1] and combined with adjustable weights::

    risk = w1*slope + w2*twi + w3*(-curvature) + w4*(-tpi)

Controls
--------
* **1 / 2 / 3 / 4** -- show individual factor (slope, TWI, curvature, TPI)
* **0**             -- show combined risk map
* **Left / Right**  -- select which weight to adjust
* **Up / Down**     -- increase / decrease the selected weight
* **R**             -- reset all weights to defaults
* **Q / Escape**    -- quit

Requires: xarray, numpy, matplotlib, xrspatial (this repo)
Optional: rasterio (for real DEM download)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from xrspatial import generate_terrain, slope
from xrspatial.curvature import curvature
from xrspatial.hydrology.flow_direction import flow_direction
from xrspatial.hydrology.flow_accumulation import flow_accumulation
from xrspatial.hydrology.twi import twi
from xrspatial.terrain_metrics import tpi

# -- Tunable parameters -----------------------------------------------------
CELL_SIZE = 30.0                    # metres per pixel
STREAM_THRESHOLD = 80               # flow-accum cells to define a stream
WEIGHT_STEP = 0.05                  # per key press
DEFAULT_WEIGHTS = [0.40, 0.25, 0.20, 0.15]  # slope, TWI, curvature, TPI
FACTOR_NAMES = ["Slope", "TWI", "Curvature", "TPI"]
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


def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1] using min-max scaling, ignoring NaN."""
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    rng = hi - lo
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / rng


# -- Build the world --------------------------------------------------------
elevation = load_dem()
GRID_H, GRID_W = elevation.shape
template = elevation.copy(data=np.zeros((GRID_H, GRID_W), dtype=np.float32))

print("Computing slope ...")
slope_da = slope(elevation)
slope_vals = np.nan_to_num(slope_da.values, nan=0.0).astype(np.float32)
slope_da = template.copy(data=slope_vals)
slope_norm = normalise(slope_vals)

print("Computing curvature ...")
curv_da = curvature(elevation, boundary='nan')
curv_vals = np.nan_to_num(curv_da.values, nan=0.0).astype(np.float32)
curv_norm = normalise(curv_vals)

print("Computing flow direction ...")
flow_dir_da = flow_direction(elevation)

print("Computing flow accumulation ...")
flow_accum_da = flow_accumulation(flow_dir_da)

print("Computing TWI (Topographic Wetness Index) ...")
twi_da = twi(flow_accum_da, slope_da)
twi_vals = np.nan_to_num(twi_da.values, nan=0.0).astype(np.float32)
# Clip extreme TWI values (very flat areas produce huge values)
twi_vals = np.clip(twi_vals, 0, np.nanpercentile(twi_vals[twi_vals > 0], 99))
twi_norm = normalise(twi_vals)

print("Computing TPI (Topographic Position Index) ...")
tpi_da = tpi(elevation, boundary='nan')
tpi_vals = np.nan_to_num(tpi_da.values, nan=0.0).astype(np.float32)
tpi_norm = normalise(tpi_vals)

# Store normalised factors for display and combination.
# For curvature and TPI, negative values indicate higher risk (concave
# hollows and valley floors collect water), so we invert them.
factors = [
    slope_norm,            # high slope -> high risk
    twi_norm,              # high wetness -> high risk
    1.0 - curv_norm,       # low (concave) curvature -> high risk
    1.0 - tpi_norm,        # low (valley) TPI -> high risk
]

factor_cmaps = ["YlOrRd", "Blues", "PuOr_r", "RdYlGn"]
factor_labels = [
    "Slope (degrees)",
    "TWI (wetness)",
    "Curvature (concave=high risk)",
    "TPI (valley=high risk)",
]

# Print summary stats
print(f"  Slope: {slope_vals.min():.1f} - {slope_vals.max():.1f} deg")
print(f"  TWI:   {twi_vals.min():.1f} - {twi_vals.max():.1f}")
print(f"  Curvature: {curv_vals.min():.2f} - {curv_vals.max():.2f}")
print(f"  TPI:   {tpi_vals.min():.2f} - {tpi_vals.max():.2f}")

# -- Simulation state -------------------------------------------------------
weights = list(DEFAULT_WEIGHTS)
active_weight = 0   # index of the weight currently selected for adjustment
view_mode = 0       # 0 = combined risk, 1-4 = individual factor


def compute_risk() -> np.ndarray:
    """Compute weighted risk score from current weights."""
    total = sum(weights)
    if total < 1e-12:
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)
    w = [wi / total for wi in weights]
    risk = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(4):
        risk += w[i] * factors[i]
    return risk


# -- Risk colour map: green (safe) -> yellow -> orange -> red (danger) ------
risk_cmap = mcolors.LinearSegmentedColormap.from_list("risk", [
    (0.0,  (0.15, 0.55, 0.20)),
    (0.25, (0.55, 0.75, 0.20)),
    (0.5,  (0.95, 0.85, 0.15)),
    (0.75, (0.90, 0.40, 0.10)),
    (1.0,  (0.70, 0.05, 0.05)),
])

# -- Visualisation -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Landslide Risk  |  0: combined  |  1-4: factors  "
    "|  L/R: select weight  |  Up/Down: adjust  |  R: reset",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain underlay
terrain_img = ax.imshow(
    elevation.values, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Risk / factor overlay (updated on each redraw)
risk_data = compute_risk()
overlay_img = ax.imshow(
    risk_data, cmap=risk_cmap, origin="lower", aspect="equal",
    vmin=0, vmax=1, alpha=0.7, interpolation="nearest",
)

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="cyan",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)

# Explanation blurb
ax.text(
    0.99, 0.99,
    "Landslides happen where multiple risk factors overlap.\n"
    "This tool combines four terrain-derived indicators:\n"
    "\n"
    "  Slope -- steeper ground fails more easily\n"
    "  TWI   -- wetter soil loses strength\n"
    "  Curvature -- concave hollows concentrate water\n"
    "  TPI   -- valley floors collect runoff and saturate\n"
    "\n"
    "Adjust weights to explore how each factor contributes.\n"
    "Press 1-4 to view individual layers. Red = high risk.\n"
    "Used for hazard mapping and land-use planning.",
    transform=ax.transAxes, color="white", fontsize=8,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
)


def update_display():
    """Recompute the risk map (or factor view) and refresh the figure."""
    if view_mode == 0:
        risk = compute_risk()
        overlay_img.set_data(risk)
        overlay_img.set_cmap(risk_cmap)
        overlay_img.set_clim(vmin=0, vmax=1)
    else:
        idx = view_mode - 1
        overlay_img.set_data(factors[idx])
        overlay_img.set_cmap(factor_cmaps[idx])
        overlay_img.set_clim(vmin=0, vmax=1)

    # Build status line
    total = sum(weights)
    parts = []
    for i in range(4):
        marker = ">" if i == active_weight else " "
        pct = 100 * weights[i] / total if total > 1e-12 else 0
        parts.append(f"{marker}{FACTOR_NAMES[i]}: {pct:.0f}%")

    if view_mode == 0:
        layer_label = "COMBINED RISK"
    else:
        layer_label = factor_labels[view_mode - 1].upper()

    # Risk statistics for combined view
    stats = ""
    if view_mode == 0:
        risk = compute_risk()
        high = np.sum(risk > 0.7)
        pct_high = 100 * high / (GRID_H * GRID_W)
        stats = f"  |  high-risk cells: {high:,} ({pct_high:.1f}%)"

    status_text.set_text(
        f"{layer_label}  |  {' | '.join(parts)}{stats}"
    )

    fig.canvas.draw_idle()


def on_key(event):
    """Keyboard handler for weight adjustment and view switching."""
    global active_weight, view_mode, weights

    if event.key in ("1", "2", "3", "4"):
        view_mode = int(event.key)
        update_display()
    elif event.key == "0":
        view_mode = 0
        update_display()
    elif event.key == "left":
        active_weight = (active_weight - 1) % 4
        update_display()
    elif event.key == "right":
        active_weight = (active_weight + 1) % 4
        update_display()
    elif event.key == "up":
        weights[active_weight] = min(weights[active_weight] + WEIGHT_STEP, 1.0)
        update_display()
    elif event.key == "down":
        weights[active_weight] = max(weights[active_weight] - WEIGHT_STEP, 0.0)
        update_display()
    elif event.key == "r":
        weights = list(DEFAULT_WEIGHTS)
        active_weight = 0
        view_mode = 0
        print("  Reset weights")
        update_display()
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

# Initial draw
update_display()

plt.tight_layout()
print("\nReady -- press 1-4 to view individual factors, "
      "0 for combined risk map.\n")
plt.show()
