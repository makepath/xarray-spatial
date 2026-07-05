"""Animated hillshade example -- matplotlib / PIL rendering (no datashader).

Generates a synthetic terrain, places trees and water on it, and renders a
rotating-hillshade GIF plus a composited build-up GIF.

Requires: numpy, xarray, matplotlib, pillow.
"""
from functools import partial

import numpy as np
import xarray as xr
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from xrspatial import bump, generate_terrain, hillshade, mean

W = 600
H = 400

x_range = (-20e6, 20e6)
y_range = (-20e6, 20e6)

xs = np.linspace(x_range[0], x_range[1], W)
ys = np.linspace(y_range[1], y_range[0], H)  # origin upper-left to match rasters
template = xr.DataArray(
    np.zeros((H, W), dtype=np.float32),
    coords={'y': ys, 'x': xs}, dims=['y', 'x'], name='terrain',
)

terrain = generate_terrain(template, x_range=x_range, y_range=y_range)


def heights(locations, src, src_range, height=20):
    num_bumps = locations.shape[0]
    out = np.zeros(num_bumps, dtype=np.uint16)
    for r in range(0, num_bumps):
        loc = locations[r]
        x = loc[0]
        y = loc[1]
        val = src[y, x]
        if val >= src_range[0] and val < src_range[1]:
            out[r] = height
    return out


T = 300000  # Number of trees to add per call
src = terrain.data
trees = bump(W, H, count=T, height_func=partial(heights, src=src,
             src_range=(1000, 1300), height=5))
trees += bump(W, H, count=T // 2, height_func=partial(
    heights, src=src, src_range=(1300, 1700), height=20))
trees += bump(W, H, count=T // 3, height_func=partial(
    heights, src=src, src_range=(1700, 2000), height=5))

tree_colorize = trees.copy()
tree_colorize.data[tree_colorize.data == 0] = np.nan

LAND_CONSTANT = 50.0

water = terrain.copy()
water.data = np.where(water.data > 0, LAND_CONSTANT, 0)
water = mean(water, passes=50, excludes=[LAND_CONSTANT])
water.data[water.data == LAND_CONSTANT] = np.nan


# Elevation-style colormap (replaces datashader.colors.Elevation).
_ELEV_STOPS = [
    (0.00, (0.00, 0.30, 0.60)),   # deep water
    (0.20, (0.20, 0.60, 0.85)),   # shallow water
    (0.30, (0.55, 0.78, 0.92)),   # shoreline
    (0.32, (0.40, 0.73, 0.45)),   # beach / lowland
    (0.45, (0.30, 0.55, 0.25)),   # forested
    (0.60, (0.50, 0.50, 0.30)),   # shrubland
    (0.75, (0.65, 0.55, 0.40)),   # highland
    (0.90, (0.80, 0.80, 0.78)),   # bare rock
    (1.00, (1.00, 1.00, 1.00)),   # snow
]
ELEVATION = LinearSegmentedColormap.from_list('elevation',
                                              [s[1] for s in _ELEV_STOPS],
                                              N=256)


def _normalize(arr):
    """Scale a finite-valued array to [0, 1], NaN -> 0 (transparent later)."""
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.where(finite, (arr - lo) / (hi - lo), 0.0)
    return out.astype(np.float32)


def _rgba(arr, cmap, alpha=None):
    """Map a 2D array through a matplotlib colormap to an (H, W, 4) uint8 image.

    NaN pixels become fully transparent (alpha=0). If `alpha` (0-255 int) is
    given, every finite pixel gets that alpha instead of 255.
    """
    norm = _normalize(arr)
    rgba = cm.ScalarMappable(cmap=cmap).to_rgba(norm, bytes=True)
    # to_rgba ignores nan by setting alpha 0; restore finite alpha if requested
    if alpha is not None:
        finite = np.isfinite(arr)
        rgba[..., 3] = np.where(finite, alpha, 0).astype(np.uint8)
    else:
        finite = np.isfinite(arr)
        rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return rgba


def _composite(layers):
    """Alpha-composite a list of (H, W, 4) uint8 arrays (bottom-to-top)."""
    base = layers[0].astype(np.float64)
    for top in layers[1:]:
        a = (top[..., 3:4] / 255.0)
        base = top[..., :3] * a + base[..., :3] * (1 - a)
        base_a = np.maximum(top[..., 3:4], base[..., 3:4] if base.shape[-1] == 4
                            else np.full_like(a, 255))
        base = np.concatenate([base, base_a * 255], axis=-1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _to_pil(rgba):
    return Image.fromarray(rgba, 'RGBA').convert('RGB')


def create_map(azimuth):
    water_cmap = LinearSegmentedColormap.from_list('water',
                                                   ['aqua', 'white'], N=256)
    layers = [
        _rgba(terrain, ELEVATION, alpha=255),
        _rgba(water, water_cmap, alpha=255),
        _rgba(hillshade(terrain + trees, azimuth=azimuth),
              LinearSegmentedColormap.from_list('hs', ['black', 'white'], N=256),
              alpha=128),
        _rgba(tree_colorize, LinearSegmentedColormap.from_list(
            'trees', ['limegreen', 'limegreen'], N=256), alpha=255),
    ]
    img = _to_pil(_composite(layers))
    print('image created')
    return img


def create_map2():
    water_cmap = LinearSegmentedColormap.from_list('water',
                                                   ['aqua', 'white'], N=256)
    bw = LinearSegmentedColormap.from_list('hs', ['black', 'white'], N=256)

    img = _to_pil(_rgba(terrain, bw, alpha=255))
    yield img

    img = _to_pil(_rgba(terrain, ELEVATION, alpha=255))
    yield img

    img = _to_pil(_composite([
        _rgba(terrain, ELEVATION, alpha=255),
        _rgba(hillshade(terrain, azimuth=210), bw, alpha=128),
    ]))
    yield img

    img = _to_pil(_composite([
        _rgba(terrain, ELEVATION, alpha=255),
        _rgba(water, water_cmap, alpha=255),
        _rgba(hillshade(terrain, azimuth=210), bw, alpha=128),
    ]))
    yield img

    img = _to_pil(_composite([
        _rgba(terrain, ELEVATION, alpha=255),
        _rgba(water, water_cmap, alpha=255),
        _rgba(hillshade(terrain + trees, azimuth=210), bw, alpha=128),
        _rgba(tree_colorize, LinearSegmentedColormap.from_list(
            'trees', ['limegreen', 'limegreen'], N=256), alpha=255),
    ]))
    yield img
    yield img
    yield img
    yield img


def gif1():
    images = []
    for i in np.linspace(0, 360, 6):
        images.append(create_map(int(i)))
    images[0].save('animated_hillshade.gif',
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=5000, loop=0)


def gif2():
    images = list(create_map2())
    images[0].save('composite_map.gif',
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=1000, loop=0)


gif2()
