"""
Interactive Fire Propagation Simulator
=======================================

Click on the map to start fires and watch them spread across a
procedurally-generated landscape.  Spread behaviour is driven by the
xrspatial fire module:

  * **rate_of_spread** -- Rothermel model for fire spread speed
  * **fireline_intensity** -- Byram's fireline intensity
  * **flame_length** -- flame length from intensity
  * **kbdi** -- drought index to derive fuel moisture
  * **dnbr / rdnbr / burn_severity_class** -- post-fire severity analysis

Controls
--------
* **Left-click**  -- ignite a new fire at that location
* **Right-click** -- place a firebreak (barrier)
* **Space**       -- pause / resume the simulation
* **B**           -- open post-fire burn severity analysis
* **R**           -- reset (clear all fires, regenerate wind)
* **Q / Escape**  -- quit

Requires: xarray, numpy, matplotlib, scipy, xrspatial (this repo)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from xrspatial import generate_terrain, slope, perlin
from xrspatial.bump import bump
from xrspatial.cost_distance import cost_distance
from xrspatial.fire import (
    ANDERSON_13,
    burn_severity_class,
    dnbr,
    fireline_intensity,
    flame_length,
    kbdi,
    rate_of_spread,
    rdnbr,
)

# -- Tunable parameters -----------------------------------------------------
GRID_H, GRID_W = 300, 400          # raster dimensions
CELL_SIZE = 30.0                    # metres per pixel
ZFACTOR = 800                       # elevation range (m)
FUEL_MODEL = 1                      # Anderson 13 fuel model (1 = short grass)
DRY_DAYS = 30                       # days of drought for KBDI spin-up
DAILY_MAX_TEMP = 35.0               # degrees C during dry spell
ANNUAL_PRECIP = 900.0               # mm mean annual precipitation
SPREAD_SPEED = 9.0                  # cost-units consumed per animation frame
MAX_COST = 5_000.0                  # fire budget (caps cost_distance search)
BURNOUT_COST = 600.0                # cost before a cell dies out
BARE_THRESHOLD = 0.18               # fuel below this -> impassable
FIRE_SEED = 42                      # reproducible terrain
FPS = 20                            # animation frame rate
BARRIER_RADIUS = 4                  # pixels cleared by right-click firebreak
# ---------------------------------------------------------------------------


def make_template(h: int, w: int, res: float) -> xr.DataArray:
    """Create an empty raster template with proper coordinates."""
    xs = np.linspace(0, w * res, w)
    ys = np.linspace(0, h * res, h)
    return xr.DataArray(
        np.zeros((h, w), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": ys, "x": xs},
    )


def make_fuel_map(
    template: xr.DataArray, elevation: xr.DataArray,
) -> np.ndarray:
    """Procedural fuel density map [0, 1] from Perlin noise, bump, and elevation.

    Perlin noise creates large biome-scale vegetation zones.  bump()
    adds clumps of dense tree clusters within vegetated areas, giving
    the fuel map a patchy, organic look.
    """
    from functools import partial

    # Low-frequency Perlin for large, obvious biome patches
    veg_base = perlin(template, freq=(2, 2), seed=7).values.astype(np.float32)
    # A little medium-frequency detail for texture
    veg_detail = perlin(template, freq=(5, 5), seed=13).values.astype(np.float32)
    veg = 0.75 * veg_base + 0.25 * veg_detail

    # Rescale to [0, 1]
    lo, hi = veg.min(), veg.max()
    veg = (veg - lo) / (hi - lo + 1e-9)

    # Aggressive S-curve: pushes values hard toward 0 or 1, creating
    # large dense patches separated by wide bare corridors
    veg = 1.0 / (1.0 + np.exp(-14 * (veg - 0.45)))

    # Suppress fuel at high elevations (above treeline) and low (water)
    elev = elevation.values
    veg[elev > 700] *= 0.15
    veg[elev < 10] *= 0.1

    # Bare-ground mask: bumps must not override bare zones
    bare_mask = veg < 0.20

    # Tree clusters via bump() -- dense fuel clumps at mid-elevations
    h, w = template.shape

    def tree_heights(locs, src=elev):
        """Tall bumps where elevation is good for trees."""
        out = np.zeros(len(locs), dtype=np.float64)
        for i in range(len(locs)):
            x, y = locs[i]
            e = src[y, x]
            if 100 < e < 600:
                out[i] = 0.35
            elif 50 < e < 100 or 600 < e < 800:
                out[i] = 0.15
        return out

    tree_bumps = bump(
        agg=template,
        count=h * w // 8,
        height_func=partial(tree_heights, src=elev),
        spread=6,
    ).values.astype(np.float32)

    # Bumps amplify existing vegetation
    veg = np.clip(veg * (1.0 + 2.0 * tree_bumps), 0, 1)

    # Stamp bare zones back to zero so corridors stay clear
    veg[bare_mask] = 0.0

    return np.clip(veg, 0, 1).astype(np.float32)


def make_wind(h: int, w: int, rng: np.random.Generator):
    """Strong prevailing wind from the west with gentle spatial variation.

    A dominant westerly (blowing east) makes fire visibly elongate
    downwind.  Small perturbations keep it from looking perfectly
    uniform.
    """
    from scipy.ndimage import gaussian_filter

    # Strong base speed with mild spatial variation
    speed_raw = gaussian_filter(rng.random((h, w)).astype(np.float32), sigma=50)
    lo, hi = speed_raw.min(), speed_raw.max()
    speed_kmh = 15.0 + 20.0 * (speed_raw - lo) / (hi - lo + 1e-9)  # 15-35 km/h

    # Prevailing direction: ~0 rad (east) with small perturbation
    dir_perturb = gaussian_filter(rng.random((h, w)).astype(np.float32), sigma=70)
    dir_perturb = (dir_perturb - dir_perturb.mean()) / (dir_perturb.std() + 1e-9)
    direction = 0.0 + dir_perturb * 0.4  # +/- ~23 degrees around due east

    return speed_kmh.astype(np.float32), direction.astype(np.float32)


# -- Build the world --------------------------------------------------------
print("Generating terrain ...")
rng = np.random.default_rng(FIRE_SEED)
template = make_template(GRID_H, GRID_W, CELL_SIZE)
elevation = generate_terrain(template, zfactor=ZFACTOR, seed=FIRE_SEED)

print("Computing slope ...")
slope_da = slope(elevation)
slope_vals = np.nan_to_num(slope_da.values, nan=0.0).astype(np.float32)
slope_da = template.copy(data=slope_vals)

print("Generating fuel & wind ...")
fuel_density = make_fuel_map(template, elevation)
wind_speed_kmh, wind_dir = make_wind(GRID_H, GRID_W, rng)
wind_speed_da = template.copy(data=wind_speed_kmh)

# -- KBDI: simulate a dry spell to get drought conditions -------------------
print("Computing drought index (KBDI) ...")
kbdi_state = template.copy(data=np.zeros((GRID_H, GRID_W), dtype=np.float32))
daily_temp = template.copy(
    data=np.full((GRID_H, GRID_W), DAILY_MAX_TEMP, dtype=np.float32),
)
no_rain = template.copy(data=np.zeros((GRID_H, GRID_W), dtype=np.float32))

for day in range(DRY_DAYS):
    kbdi_state = kbdi(kbdi_state, daily_temp, no_rain,
                      annual_precip=ANNUAL_PRECIP)

kbdi_vals = kbdi_state.values
print(f"  KBDI after {DRY_DAYS} dry days: "
      f"mean={kbdi_vals.mean():.0f}, max={kbdi_vals.max():.0f}")

# Derive fuel moisture from KBDI.
# KBDI 0 (no drought) -> 30% moisture, KBDI 800 (extreme) -> 3%.
fuel_moisture_vals = 0.30 * (1.0 - kbdi_vals / 800.0) + 0.03
fuel_moisture_vals = np.clip(fuel_moisture_vals, 0.03, 0.30).astype(np.float32)
fuel_moisture_da = template.copy(data=fuel_moisture_vals)

# -- Rate of spread (Rothermel) ---------------------------------------------
print("Computing rate of spread (Rothermel) ...")
ros_da = rate_of_spread(
    slope_da, wind_speed_da, fuel_moisture_da, fuel_model=FUEL_MODEL,
)
ros_vals = ros_da.values  # m/min

# -- Fireline intensity and flame length ------------------------------------
fuel_load_kg_m2 = float(ANDERSON_13[FUEL_MODEL - 1, 0])
heat_content_kj = float(ANDERSON_13[FUEL_MODEL - 1, 4])
fuel_consumed_da = template.copy(
    data=(fuel_load_kg_m2 * fuel_density).astype(np.float32),
)
ros_m_s_da = template.copy(data=(ros_vals / 60.0).astype(np.float32))

print("Computing fireline intensity & flame length ...")
intensity_da = fireline_intensity(
    fuel_consumed_da, ros_m_s_da, heat_content=heat_content_kj,
)
flame_len_da = flame_length(intensity_da)
flame_len_vals = flame_len_da.values

print(f"  Flame length: mean={np.nanmean(flame_len_vals):.2f} m, "
      f"max={np.nanmax(flame_len_vals):.2f} m")

# -- Friction surface for cost_distance from ROS ----------------------------
# Base friction = time per cell = cell_size / ROS (minutes).
# Where ROS is near-zero or fuel is bare, friction is NaN (impassable).
friction_base = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
passable = (ros_vals > 0.01) & (fuel_density >= BARE_THRESHOLD)
friction_base[passable] = CELL_SIZE / ros_vals[passable]

# Directional wind correction.  cost_distance is isotropic, so we bake
# wind direction into the friction: cells that are downwind (neighbours'
# wind blows INTO them) get lower friction; upwind cells get higher friction.
wu = wind_speed_kmh * np.cos(wind_dir)
wv = wind_speed_kmh * np.sin(wind_dir)

inflow = np.zeros_like(wind_speed_kmh)
for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]:
    # Wind vector at neighbouring cell, dotted with the direction
    # from that neighbour toward this cell (-dx, -dy)
    nb_u = np.roll(np.roll(wu, -dy, axis=0), -dx, axis=1)
    nb_v = np.roll(np.roll(wv, -dy, axis=0), -dx, axis=1)
    inflow += np.maximum(nb_u * (-dx) + nb_v * (-dy), 0)

inflow /= inflow.max() + 1e-9  # normalise to [0, 1]

# Map inflow to a friction multiplier.
# inflow=1 (strongly downwind) -> 0.10  (fire races through)
# inflow=0 (strongly upwind)   -> 3.0   (fire crawls against wind)
# This gives a ~30:1 downwind-to-upwind speed ratio.
wind_factor = 3.0 - 2.9 * inflow
friction_base[passable] *= np.clip(wind_factor, 0.10, 3.0)[passable]

# Directional slope correction.  Fire spreads faster uphill (heat rises
# and preheats fuel above) and slower downhill.  For each cell, measure
# how much higher its neighbours are -- high "uphill score" means this
# cell is at the bottom of slopes, so fire reaching it can race upward
# through it easily (low friction).
elev_vals = elevation.values.astype(np.float32)
uphill_score = np.zeros_like(elev_vals)
for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]:
    nb_elev = np.roll(np.roll(elev_vals, -dy, axis=0), -dx, axis=1)
    # Positive when neighbour is higher (this cell feeds fire uphill)
    uphill_score += np.maximum(nb_elev - elev_vals, 0)

uphill_score /= uphill_score.max() + 1e-9  # normalise to [0, 1]

# uphill_score=1 (valley bottom, fire races uphill) -> 0.25
# uphill_score=0 (ridge top, fire crawls downhill)  -> 2.0
# ~8:1 uphill-to-downhill speed ratio.
slope_dir_factor = 2.0 - 1.75 * uphill_score
friction_base[passable] *= np.clip(slope_dir_factor, 0.25, 2.0)[passable]

# Pre-fire NBR proxy from fuel density (for post-fire analysis).
# Denser vegetation = higher NBR.
pre_nbr_vals = (0.8 * fuel_density - 0.1).astype(np.float32)
pre_nbr_da = template.copy(data=pre_nbr_vals)

# -- Simulation state -------------------------------------------------------
fires: list[tuple[int, int, float, np.ndarray]] = []
burned_mask = np.zeros((GRID_H, GRID_W), dtype=bool)
barrier_mask = np.zeros((GRID_H, GRID_W), dtype=bool)
# Tracks which fire index claimed each cell first (-1 = unclaimed).
# Prevents one fire from burning through land another fire already reached.
claimed_by = np.full((GRID_H, GRID_W), -1, dtype=np.int32)
current_threshold = 0.0
paused = False


def _current_friction() -> np.ndarray:
    """Friction with barriers and already-burned cells blocked."""
    friction = friction_base.copy()
    friction[barrier_mask] = np.nan
    friction[burned_mask] = np.nan
    return friction


def add_fire(row: int, col: int):
    """Compute cost_distance for a new ignition and store it."""
    if burned_mask[row, col] or np.isnan(friction_base[row, col]):
        return
    src = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    src[row, col] = 1
    src_da = template.copy(data=src)
    friction_da = template.copy(data=_current_friction())
    cost = cost_distance(src_da, friction_da, max_cost=MAX_COST).values
    fires.append((row, col, current_threshold, cost))


def recompute_barriers():
    """Re-run cost_distance for every fire after a barrier changes."""
    updated: list[tuple[int, int, float, np.ndarray]] = []
    for row, col, t0, _ in fires:
        src = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        src[row, col] = 1
        src_da = template.copy(data=src)
        friction_da = template.copy(data=_current_friction())
        cost = cost_distance(src_da, friction_da, max_cost=MAX_COST).values
        updated.append((row, col, t0, cost))
    fires.clear()
    fires.extend(updated)


def show_burn_severity():
    """Open a window with post-fire dNBR, RdNBR, and severity class."""
    if not burned_mask.any():
        print("  No burned area to analyse.")
        return

    # Post-fire NBR: burned areas lose most vegetation signal
    post_nbr_vals = pre_nbr_vals.copy()
    post_nbr_vals[burned_mask] *= 0.1
    post_nbr_da = template.copy(data=post_nbr_vals)

    dnbr_da = dnbr(pre_nbr_da, post_nbr_da)
    rdnbr_da = rdnbr(dnbr_da, pre_nbr_da)
    severity_da = burn_severity_class(dnbr_da)

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Post-Fire Burn Severity (xrspatial.fire)", fontsize=13)

    im0 = axes[0].imshow(dnbr_da.values, cmap="RdYlGn_r", origin="lower")
    axes[0].set_title("dNBR")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    rdnbr_display = np.clip(rdnbr_da.values, -5, 20)
    im1 = axes[1].imshow(rdnbr_display, cmap="RdYlGn_r", origin="lower")
    axes[1].set_title("RdNBR")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    severity_cmap = mcolors.ListedColormap([
        "#808080",  # 0 = nodata
        "#1a9641",  # 1 = enhanced regrowth (high)
        "#a6d96a",  # 2 = enhanced regrowth (low)
        "#f7f7f7",  # 3 = unburned
        "#fee08b",  # 4 = low severity
        "#fdae61",  # 5 = moderate-low
        "#f46d43",  # 6 = moderate-high
        "#d73027",  # 7 = high severity
    ])
    im2 = axes[2].imshow(
        severity_da.values, cmap=severity_cmap, origin="lower", vmin=0, vmax=7,
    )
    axes[2].set_title("Burn Severity Class")
    cbar = plt.colorbar(im2, ax=axes[2], shrink=0.7, ticks=range(8))
    cbar.ax.set_yticklabels([
        "nodata", "regrowth+", "regrowth", "unburned",
        "low", "mod-low", "mod-high", "high",
    ])

    plt.tight_layout()
    plt.show(block=False)


# -- Visualisation -----------------------------------------------------------

# Fire colour map: transparent -> yellow -> orange -> red -> grey ash
fire_cmap = mcolors.LinearSegmentedColormap.from_list("fire", [
    (0.0,  (1.0, 1.0, 0.0)),
    (0.01, (1.0, 1.0, 0.0)),
    (0.25, (1.0, 0.55, 0.0)),
    (0.5,  (0.9, 0.15, 0.0)),
    (0.75, (0.35, 0.05, 0.0)),
    (1.0,  (0.2, 0.2, 0.2)),
])

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_title(
    "Fire Propagation  |  L-click: ignite  |  R-click: firebreak  "
    "|  Space: pause  |  B: severity  |  R: reset",
    color="white", fontsize=11,
)
ax.tick_params(colors="white")

# Terrain layer
terrain_img = ax.imshow(
    elevation.values, cmap=plt.cm.terrain, origin="lower",
    aspect="equal", interpolation="bilinear",
)

# Fuel overlay
fuel_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
vegetated = fuel_density >= BARE_THRESHOLD
bare = ~vegetated
fuel_rgba[vegetated, 1] = fuel_density[vegetated]
fuel_rgba[vegetated, 3] = fuel_density[vegetated] * 0.3
fuel_rgba[bare] = [0.6, 0.45, 0.2, 0.35]
fuel_img = ax.imshow(fuel_rgba, origin="lower", aspect="equal")

# Fire overlay (updated each frame)
fire_data = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
fire_img = ax.imshow(
    fire_data, cmap=fire_cmap, origin="lower", aspect="equal",
    vmin=0, vmax=1, alpha=0.85, interpolation="nearest",
)

# Scorch overlay
scorch_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
scorch_rgba[..., :3] = 0.05
scorch_img = ax.imshow(scorch_rgba, origin="lower", aspect="equal")

# Barrier overlay
barrier_rgba = np.zeros((GRID_H, GRID_W, 4), dtype=np.float32)
barrier_img = ax.imshow(barrier_rgba, origin="lower", aspect="equal")

# Ignition markers
(marker_plot,) = ax.plot([], [], "w*", markersize=10, markeredgecolor="black")

# Wind barbs (coarse grid)
BARB_STEP = 25
yr = np.arange(0, GRID_H, BARB_STEP)
xr_ = np.arange(0, GRID_W, BARB_STEP)
xg, yg = np.meshgrid(xr_, yr)
u = wind_speed_kmh[yr][:, xr_] * np.cos(wind_dir[yr][:, xr_]) * 0.3
v = wind_speed_kmh[yr][:, xr_] * np.sin(wind_dir[yr][:, xr_]) * 0.3
ax.quiver(xg, yg, u, v, color="white", alpha=0.25, scale=30, width=0.002)

status_text = ax.text(
    0.01, 0.01, "", transform=ax.transAxes, color="yellow",
    fontsize=9, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
)


def update_frame(frame: int):
    """Advance fire front and redraw."""
    global current_threshold

    if not fires:
        fire_img.set_data(np.full((GRID_H, GRID_W), np.nan))
        status_text.set_text("Click to ignite a fire!")
        return (fire_img, status_text, marker_plot, barrier_img)

    if not paused:
        current_threshold += SPREAD_SPEED

    # Merge all fires.  Each cell is owned by the first fire to reach it;
    # later fires cannot burn through land that another fire already claimed.
    min_age = np.full((GRID_H, GRID_W), np.inf, dtype=np.float32)
    any_reached = np.zeros((GRID_H, GRID_W), dtype=bool)

    for fire_idx, (_, _, t0, cost) in enumerate(fires):
        age = (current_threshold - t0) - cost
        reached = age >= 0
        # Only allow this fire into unclaimed cells or cells it already owns
        allowed = (claimed_by == -1) | (claimed_by == fire_idx)
        reached &= allowed
        # Claim newly reached cells for this fire
        newly_claimed = reached & (claimed_by == -1)
        claimed_by[newly_claimed] = fire_idx
        any_reached |= reached
        younger = reached & (age < min_age)
        min_age[younger] = age[younger]

    active = any_reached & (min_age <= BURNOUT_COST)
    dead = any_reached & (min_age > BURNOUT_COST)
    burned_mask[dead] = True

    # Colour active fire cells by age (yellow -> red -> ash)
    fire_display = np.full((GRID_H, GRID_W), np.nan, dtype=np.float32)
    if active.any():
        fire_display[active] = np.clip(min_age[active] / BURNOUT_COST, 0, 1)
    fire_img.set_data(fire_display)

    # Scorch overlay
    scorch = scorch_rgba.copy()
    if dead.any():
        dead_age = (min_age[dead] - BURNOUT_COST) / BURNOUT_COST
        scorch[dead, 3] = np.clip(dead_age * 3, 0, 0.55)
    scorch_img.set_data(scorch)

    # Barrier overlay
    if barrier_mask.any():
        br = barrier_rgba.copy()
        br[barrier_mask] = [0.3, 0.3, 1.0, 0.6]
        barrier_img.set_data(br)

    # Ignition markers
    if fires:
        cols, rows = zip(*[(c, r) for r, c, _, _ in fires])
        marker_plot.set_data(cols, rows)

    # Stats for active zone
    n_active = int(active.sum()) if active.any() else 0
    n_dead = int(dead.sum()) if dead.any() else 0
    pct = 100 * (n_active + n_dead) / (GRID_H * GRID_W)

    avg_flame = ""
    if active.any():
        fl = flame_len_vals[active]
        avg_flame = f"  |  flame: {np.nanmean(fl):.1f} m"

    state = "PAUSED" if paused else "SPREADING"
    status_text.set_text(
        f"{state}  |  fires: {len(fires)}  |  "
        f"burning: {n_active:,}  |  scorched: {n_dead:,}  |  "
        f"area: {pct:.1f}%{avg_flame}"
    )

    return (fire_img, status_text, marker_plot, barrier_img)


def on_click(event):
    """Left-click: ignite.  Right-click: firebreak."""
    if event.inaxes != ax:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if not (0 <= row < GRID_H and 0 <= col < GRID_W):
        return

    if event.button == 1:
        print(f"  Ignition at pixel ({row}, {col})  "
              f"ROS={ros_vals[row, col]:.1f} m/min  "
              f"flame={flame_len_vals[row, col]:.1f} m")
        add_fire(row, col)
    elif event.button == 3:
        r_lo = max(0, row - BARRIER_RADIUS)
        r_hi = min(GRID_H, row + BARRIER_RADIUS + 1)
        c_lo = max(0, col - BARRIER_RADIUS)
        c_hi = min(GRID_W, col + BARRIER_RADIUS + 1)
        barrier_mask[r_lo:r_hi, c_lo:c_hi] = True
        print(f"  Firebreak at pixel ({row}, {col})")
        if fires:
            recompute_barriers()


def on_key(event):
    """Keyboard: pause, severity analysis, reset, quit."""
    global paused, current_threshold

    if event.key == " ":
        paused = not paused
        print("  Paused" if paused else "  Resumed")
    elif event.key == "b":
        show_burn_severity()
    elif event.key == "r":
        fires.clear()
        barrier_mask[:] = False
        burned_mask[:] = False
        claimed_by[:] = -1
        current_threshold = 0.0
        marker_plot.set_data([], [])
        barrier_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        scorch_img.set_data(np.zeros((GRID_H, GRID_W, 4), dtype=np.float32))
        print("  Reset")
    elif event.key in ("q", "escape"):
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

anim = FuncAnimation(
    fig, update_frame, interval=1000 // FPS, blit=False, cache_frame_data=False,
)

plt.tight_layout()
print("\nReady -- click the map to start a fire!\n")
plt.show()
