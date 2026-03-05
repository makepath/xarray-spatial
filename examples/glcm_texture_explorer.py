"""
Interactive GLCM Texture Explorer
==================================

Explore Gray-Level Co-occurrence Matrix texture features on a
procedurally-generated landscape with six distinct texture zones.
Change metrics, window size, quantization levels, and angle
interactively and see the texture map update.

Click any pixel to inspect its local GLCM matrix and per-metric
values in a pop-up panel.

Controls
--------
* **1-6**         -- select metric (contrast, dissimilarity, homogeneity,
                     energy, correlation, entropy)
* **Up / Down**   -- increase / decrease window size (3, 5, 7, 9, 11)
* **Left / Right** -- cycle GLCM angle (0, 45, 90, 135, avg)
* **+ / -**       -- increase / decrease gray levels (8, 16, 32, 64, 128)
* **Left-click**  -- inspect pixel (shows local GLCM matrix)
* **R**           -- regenerate terrain with a new seed
* **Q / Escape**  -- quit

Requires: xarray, numpy, matplotlib, numba, xrspatial (this repo)
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from xrspatial.glcm import glcm_texture, _quantize, VALID_METRICS

# -- Tunable parameters -----------------------------------------------------
GRID_SIZE = 120                      # raster is GRID_SIZE x GRID_SIZE
WINDOW_SIZES = [3, 5, 7, 9, 11]     # available window sizes
LEVEL_OPTIONS = [8, 16, 32, 64, 128] # available gray-level counts
ANGLE_OPTIONS = [0, 45, 90, 135, None]  # None = average all four
INITIAL_METRIC = 0                   # index into VALID_METRICS
INITIAL_WINDOW = 2                   # index into WINDOW_SIZES (7)
INITIAL_LEVELS = 1                   # index into LEVEL_OPTIONS (16)
INITIAL_ANGLE = 4                    # index into ANGLE_OPTIONS (None/avg)
SEED = 42
# ---------------------------------------------------------------------------

METRIC_NAMES = list(VALID_METRICS)
ANGLE_LABELS = ['0 (right)', '45 (upper-right)', '90 (up)',
                '135 (upper-left)', 'avg (all 4)']


def make_texture_raster(size: int, seed: int) -> np.ndarray:
    """Build a raster with six distinct texture zones.

    Layout (3 rows x 2 columns):
      Top-left:     smooth gradient
      Top-right:    random noise
      Mid-left:     vertical stripes
      Mid-right:    horizontal stripes
      Bottom-left:  checkerboard
      Bottom-right: Perlin-like fractal
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((size, size), dtype=np.float64)
    h3 = size // 3
    h2 = size // 2

    # Top-left: smooth gradient
    data[:h3, :h2] = np.linspace(0, 1, h2)[np.newaxis, :]

    # Top-right: random noise
    data[:h3, h2:] = rng.random((h3, size - h2))

    # Mid-left: vertical stripes (period 4)
    stripe_v = np.tile([0.0, 0.0, 1.0, 1.0], size // 4 + 1)[:size - h2]
    data[h3:2*h3, :h2] = stripe_v[np.newaxis, :]

    # Mid-right: horizontal stripes (period 6)
    stripe_h = np.tile([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], h3 // 6 + 1)[:h3]
    data[h3:2*h3, h2:] = stripe_h[:, np.newaxis]

    # Bottom-left: checkerboard
    cb = np.zeros((size - 2*h3, h2))
    cb[::2, 1::2] = 1.0
    cb[1::2, ::2] = 1.0
    data[2*h3:, :h2] = cb

    # Bottom-right: fractal noise (sum of multiple frequency octaves)
    bh = size - 2*h3
    bw = size - h2
    fractal = np.zeros((bh, bw))
    for octave in range(5):
        freq = 2 ** octave
        phase_y = rng.random() * 2 * np.pi
        phase_x = rng.random() * 2 * np.pi
        ys = np.sin(np.linspace(phase_y, phase_y + freq * np.pi, bh))
        xs = np.sin(np.linspace(phase_x, phase_x + freq * np.pi, bw))
        fractal += (ys[:, np.newaxis] * xs[np.newaxis, :]) / (1 + octave)
    fmin, fmax = fractal.min(), fractal.max()
    fractal = (fractal - fmin) / (fmax - fmin + 1e-12)
    data[2*h3:, h2:] = fractal

    return data


# -- State -------------------------------------------------------------------
class State:
    metric_idx = INITIAL_METRIC
    window_idx = INITIAL_WINDOW
    levels_idx = INITIAL_LEVELS
    angle_idx = INITIAL_ANGLE
    seed = SEED
    data = None
    agg = None
    inspect_pixel = None  # (row, col) or None

    @property
    def metric(self): return METRIC_NAMES[self.metric_idx]
    @property
    def window_size(self): return WINDOW_SIZES[self.window_idx]
    @property
    def levels(self): return LEVEL_OPTIONS[self.levels_idx]
    @property
    def angle(self): return ANGLE_OPTIONS[self.angle_idx]
    @property
    def angle_label(self): return ANGLE_LABELS[self.angle_idx]


st = State()


def generate_data():
    st.data = make_texture_raster(GRID_SIZE, st.seed)
    st.agg = xr.DataArray(st.data, dims=['y', 'x'])


def compute_texture():
    t0 = time.time()
    result = glcm_texture(
        st.agg, metric=st.metric,
        window_size=st.window_size,
        levels=st.levels,
        angle=st.angle,
    )
    elapsed = time.time() - t0
    return result.values, elapsed


def compute_local_glcm(row, col):
    """Build the GLCM matrix at a specific pixel for display."""
    half = st.window_size // 2
    quantized = _quantize(st.data, st.levels)
    h, w = st.data.shape
    levels = st.levels

    # Accumulate over requested angle(s)
    angles = [st.angle] if st.angle is not None else [0, 45, 90, 135]
    glcm_acc = np.zeros((levels, levels), dtype=np.float64)
    total_count = 0.0

    from xrspatial.glcm import _ANGLE_OFFSETS
    for ang in angles:
        dy, dx = _ANGLE_OFFSETS[ang]
        for wy in range(row - half, row + half + 1):
            for wx in range(col - half, col + half + 1):
                ny, nx = wy + dy, wx + dx
                if 0 <= wy < h and 0 <= wx < w and 0 <= ny < h and 0 <= nx < w:
                    iv = quantized[wy, wx]
                    jv = quantized[ny, nx]
                    if iv >= 0 and jv >= 0:
                        glcm_acc[iv, jv] += 1.0
                        total_count += 1.0

    if total_count > 0:
        glcm_acc /= total_count
    return glcm_acc


def compute_all_metrics_at(row, col):
    """Return a dict of all metric values at one pixel."""
    result = glcm_texture(
        st.agg, metric=list(VALID_METRICS),
        window_size=st.window_size,
        levels=st.levels,
        angle=st.angle,
    )
    vals = {}
    for m in VALID_METRICS:
        vals[m] = float(result.sel(metric=m).values[row, col])
    return vals


# -- Build the figure --------------------------------------------------------
generate_data()
texture_vals, elapsed = compute_texture()

fig = plt.figure(figsize=(14, 8), layout='constrained')
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2], hspace=0.3, wspace=0.3,
                       figure=fig)

# Top-left: input raster
ax_input = fig.add_subplot(gs[0, 0])
ax_input.set_title('Input raster', color='white', fontsize=11)
input_img = ax_input.imshow(st.data, cmap='gray', origin='upper')
ax_input.tick_params(colors='#888888', labelsize=7)

# Zone labels
h3 = GRID_SIZE // 3
h2 = GRID_SIZE // 2
zone_labels = [
    (h3//2, h2//2, 'gradient'),
    (h3//2, h2 + (GRID_SIZE-h2)//2, 'noise'),
    (h3 + h3//2, h2//2, 'v-stripes'),
    (h3 + h3//2, h2 + (GRID_SIZE-h2)//2, 'h-stripes'),
    (2*h3 + (GRID_SIZE-2*h3)//2, h2//2, 'checker'),
    (2*h3 + (GRID_SIZE-2*h3)//2, h2 + (GRID_SIZE-h2)//2, 'fractal'),
]
for r, c, label in zone_labels:
    ax_input.text(c, r, label, ha='center', va='center',
                  color='cyan', fontsize=8, fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.5, pad=1))

# Inspect crosshair on input
crosshair_v = ax_input.axvline(-1, color='lime', lw=0.8, alpha=0)
crosshair_h = ax_input.axhline(-1, color='lime', lw=0.8, alpha=0)

# Top-center+right: texture map (wider)
ax_texture = fig.add_subplot(gs[0, 1:])
texture_title = ax_texture.set_title('', color='white', fontsize=11)
texture_img = ax_texture.imshow(texture_vals, cmap='inferno', origin='upper')
cbar = plt.colorbar(texture_img, ax=ax_texture, shrink=0.8, pad=0.02)
cbar.ax.tick_params(colors='#888888', labelsize=7)
ax_texture.tick_params(colors='#888888', labelsize=7)

# Bottom-left: GLCM matrix at clicked pixel
ax_glcm = fig.add_subplot(gs[1, 0])
ax_glcm.set_title('GLCM at pixel (click to inspect)', color='white', fontsize=10)
glcm_placeholder = np.zeros((16, 16))
glcm_img = ax_glcm.imshow(glcm_placeholder, cmap='hot', origin='lower',
                           aspect='equal')
ax_glcm.set_xlabel('j', color='#888888', fontsize=9)
ax_glcm.set_ylabel('i', color='#888888', fontsize=9)
ax_glcm.tick_params(colors='#888888', labelsize=7)

# Bottom-center: per-metric bar chart at clicked pixel
ax_bars = fig.add_subplot(gs[1, 1])
ax_bars.set_title('Metrics at pixel', color='white', fontsize=10)
bar_colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
bar_rects = ax_bars.barh(range(6), [0]*6, color=bar_colors, height=0.6)
ax_bars.set_yticks(range(6))
ax_bars.set_yticklabels([m[:5] for m in METRIC_NAMES], color='#cccccc', fontsize=9)
ax_bars.tick_params(colors='#888888', labelsize=7)
ax_bars.set_xlim(0, 1)
ax_bars.set_facecolor('#1a1a2e')

# Bottom-right: parameter display and controls help
ax_info = fig.add_subplot(gs[1, 2])
ax_info.set_facecolor('#1a1a2e')
ax_info.set_xticks([])
ax_info.set_yticks([])
for spine in ax_info.spines.values():
    spine.set_visible(False)

info_text = ax_info.text(
    0.05, 0.95, '', transform=ax_info.transAxes,
    color='#cccccc', fontsize=9, verticalalignment='top',
    fontfamily='monospace', linespacing=1.6,
)


def update_info():
    lines = [
        'PARAMETERS',
        f'  metric : {st.metric}',
        f'  window : {st.window_size}x{st.window_size}',
        f'  levels : {st.levels}',
        f'  angle  : {st.angle_label}',
        '',
        'CONTROLS',
        '  1-6        metric',
        '  Up/Down    window size',
        '  Left/Right angle',
        '  +/-        gray levels',
        '  Click      inspect pixel',
        '  R          new terrain',
        '  Q/Esc      quit',
    ]
    info_text.set_text('\n'.join(lines))


def update_texture_display(texture_vals, elapsed):
    texture_img.set_data(texture_vals)
    vmin = np.nanmin(texture_vals)
    vmax = np.nanmax(texture_vals)
    if vmin == vmax:
        vmax = vmin + 1
    texture_img.set_clim(vmin, vmax)
    ax_texture.set_title(
        f'{st.metric}  (window={st.window_size}  levels={st.levels}'
        f'  angle={st.angle_label})  [{elapsed:.2f}s]',
        color='white', fontsize=11,
    )
    fig.canvas.draw_idle()


def update_inspection(row, col):
    """Update the GLCM matrix and bar chart for a clicked pixel."""
    # GLCM matrix
    glcm = compute_local_glcm(row, col)
    # Only show the occupied portion (trim trailing zero rows/cols)
    used = max(int(np.max(np.where(glcm > 0)[0]) + 1) if glcm.any() else 1,
               int(np.max(np.where(glcm > 0)[1]) + 1) if glcm.any() else 1)
    used = min(used, st.levels)
    display_glcm = glcm[:used, :used]
    glcm_img.set_data(display_glcm)
    glcm_img.set_extent([-0.5, used - 0.5, -0.5, used - 0.5])
    gmax = display_glcm.max()
    glcm_img.set_clim(0, gmax if gmax > 0 else 1)
    ax_glcm.set_title(f'GLCM at ({row}, {col})', color='white', fontsize=10)

    # Bar chart
    metric_vals = compute_all_metrics_at(row, col)
    max_val = 0
    for i, m in enumerate(METRIC_NAMES):
        v = metric_vals[m]
        if np.isnan(v):
            v = 0
        bar_rects[i].set_width(v)
        if v > max_val:
            max_val = v
    ax_bars.set_xlim(0, max(max_val * 1.1, 0.01))
    ax_bars.set_title(f'Metrics at ({row}, {col})', color='white', fontsize=10)

    # Crosshairs
    crosshair_v.set_xdata([col, col])
    crosshair_h.set_ydata([row, row])
    crosshair_v.set_alpha(0.8)
    crosshair_h.set_alpha(0.8)

    fig.canvas.draw_idle()


def recompute_and_display():
    texture_vals, elapsed = compute_texture()
    update_texture_display(texture_vals, elapsed)
    update_info()
    if st.inspect_pixel is not None:
        update_inspection(*st.inspect_pixel)


# -- Event handlers ----------------------------------------------------------

def on_click(event):
    if event.inaxes not in (ax_input, ax_texture):
        return
    if event.button != 1:
        return

    # Map click to pixel coordinates in the raster
    if event.inaxes == ax_input:
        col = int(round(event.xdata))
        row = int(round(event.ydata))
    else:
        col = int(round(event.xdata))
        row = int(round(event.ydata))

    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        return

    st.inspect_pixel = (row, col)
    update_inspection(row, col)


def on_key(event):
    if event.key in ('q', 'escape'):
        plt.close(fig)
        return

    changed = False

    # Metric selection: 1-6
    if event.key in '123456':
        idx = int(event.key) - 1
        if idx < len(METRIC_NAMES):
            st.metric_idx = idx
            changed = True

    # Window size
    elif event.key == 'up':
        st.window_idx = min(st.window_idx + 1, len(WINDOW_SIZES) - 1)
        changed = True
    elif event.key == 'down':
        st.window_idx = max(st.window_idx - 1, 0)
        changed = True

    # Angle
    elif event.key == 'right':
        st.angle_idx = (st.angle_idx + 1) % len(ANGLE_OPTIONS)
        changed = True
    elif event.key == 'left':
        st.angle_idx = (st.angle_idx - 1) % len(ANGLE_OPTIONS)
        changed = True

    # Levels
    elif event.key in ('+', '='):
        st.levels_idx = min(st.levels_idx + 1, len(LEVEL_OPTIONS) - 1)
        changed = True
    elif event.key in ('-', '_'):
        st.levels_idx = max(st.levels_idx - 1, 0)
        changed = True

    # Regenerate terrain
    elif event.key == 'r':
        st.seed += 1
        st.inspect_pixel = None
        crosshair_v.set_alpha(0)
        crosshair_h.set_alpha(0)
        generate_data()
        input_img.set_data(st.data)
        changed = True

    if changed:
        recompute_and_display()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

# -- Initial render ----------------------------------------------------------
update_texture_display(texture_vals, elapsed)
update_info()

print("\nGLCM Texture Explorer ready.")
print("Use 1-6 to pick a metric, arrow keys to change window/angle,")
print("+/- for gray levels, click to inspect a pixel.\n")
plt.show()
