"""Output grid computation and chunk layout for reprojection."""
from __future__ import annotations

import numpy as np


def _compute_output_grid(source_bounds, source_shape, source_crs, target_crs,
                         resolution=None, bounds=None, width=None, height=None):
    """Compute the output raster grid parameters.

    Parameters
    ----------
    source_bounds : tuple
        (left, bottom, right, top) in source CRS.
    source_shape : tuple
        (height, width) of source raster.
    source_crs, target_crs : pyproj.CRS
        Source and target coordinate reference systems.
    resolution : float or tuple or None
        Target resolution. If tuple, (x_res, y_res).
    bounds : tuple or None
        Explicit (left, bottom, right, top) in target CRS.
    width, height : int or None
        Explicit output dimensions.

    Returns
    -------
    dict with keys: bounds, shape, res_x, res_y
    """
    from ._crs_utils import _require_pyproj

    pyproj = _require_pyproj()
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )

    if bounds is None:
        # Transform source corners and edges to target CRS
        src_left, src_bottom, src_right, src_top = source_bounds
        n_edge = 21  # sample points along each edge
        xs = np.concatenate([
            np.linspace(src_left, src_right, n_edge),   # top edge
            np.linspace(src_left, src_right, n_edge),   # bottom edge
            np.full(n_edge, src_left),                   # left edge
            np.full(n_edge, src_right),                  # right edge
        ])
        ys = np.concatenate([
            np.full(n_edge, src_top),
            np.full(n_edge, src_bottom),
            np.linspace(src_bottom, src_top, n_edge),
            np.linspace(src_bottom, src_top, n_edge),
        ])
        tx, ty = transformer.transform(xs, ys)
        tx = np.asarray(tx)
        ty = np.asarray(ty)
        # Filter out inf/nan from failed transforms
        valid = np.isfinite(tx) & np.isfinite(ty)
        if not valid.any():
            raise ValueError(
                "Could not transform any source boundary points "
                "to target CRS."
            )
        tx = tx[valid]
        ty = ty[valid]
        bounds = (float(tx.min()), float(ty.min()),
                  float(tx.max()), float(ty.max()))

    left, bottom, right, top = bounds

    # Determine resolution
    if resolution is not None:
        if isinstance(resolution, (tuple, list)):
            res_x, res_y = float(resolution[0]), float(resolution[1])
        else:
            res_x = res_y = float(resolution)
    elif width is not None and height is not None:
        res_x = (right - left) / width
        res_y = (top - bottom) / height
    else:
        # Estimate from source resolution
        src_h, src_w = source_shape
        src_left, src_bottom, src_right, src_top = source_bounds
        src_res_x = (src_right - src_left) / src_w
        src_res_y = (src_top - src_bottom) / src_h
        # Use the geometric mean of transformed pixel sizes
        center_x = (src_left + src_right) / 2
        center_y = (src_bottom + src_top) / 2
        tx1, ty1 = transformer.transform(center_x, center_y)
        tx2, ty2 = transformer.transform(
            center_x + src_res_x, center_y + src_res_y
        )
        res_x = abs(float(tx2) - float(tx1))
        res_y = abs(float(ty2) - float(ty1))
        if res_x == 0 or res_y == 0:
            res_x = (right - left) / src_w
            res_y = (top - bottom) / src_h

    # Compute dimensions
    if width is None:
        width = max(1, int(np.ceil((right - left) / res_x)))
    if height is None:
        height = max(1, int(np.ceil((top - bottom) / res_y)))

    # Adjust bounds to be exact multiples of resolution
    right = left + width * res_x
    top = bottom + height * res_y

    return {
        'bounds': (left, bottom, right, top),
        'shape': (height, width),
        'res_x': res_x,
        'res_y': res_y,
    }


def _make_output_coords(bounds, shape):
    """Create y and x coordinate arrays for the output grid.

    Coordinates are pixel-center aligned.

    Parameters
    ----------
    bounds : tuple
        (left, bottom, right, top) in target CRS.
    shape : tuple
        (height, width).

    Returns
    -------
    y_coords, x_coords : ndarray
    """
    left, bottom, right, top = bounds
    height, width = shape
    res_x = (right - left) / width
    res_y = (top - bottom) / height
    x_coords = np.linspace(left + res_x / 2, right - res_x / 2, width)
    y_coords = np.linspace(top - res_y / 2, bottom + res_y / 2, height)
    return y_coords, x_coords


def _compute_chunk_layout(shape, chunk_size):
    """Compute chunk sizes along each axis.

    Parameters
    ----------
    shape : tuple
        (height, width).
    chunk_size : int or tuple or None
        Target chunk size. None defaults to 512.

    Returns
    -------
    row_chunks, col_chunks : tuple of int
    """
    if chunk_size is None:
        chunk_size = (512, 512)
    elif isinstance(chunk_size, int):
        chunk_size = (chunk_size, chunk_size)

    height, width = shape
    rcs, ccs = chunk_size

    row_chunks = []
    remaining = height
    while remaining > 0:
        c = min(rcs, remaining)
        row_chunks.append(c)
        remaining -= c

    col_chunks = []
    remaining = width
    while remaining > 0:
        c = min(ccs, remaining)
        col_chunks.append(c)
        remaining -= c

    return tuple(row_chunks), tuple(col_chunks)


def _chunk_bounds(grid_bounds, grid_shape, row_start, row_end, col_start, col_end):
    """Compute geographic bounds for a specific chunk within the output grid.

    Parameters
    ----------
    grid_bounds : tuple
        (left, bottom, right, top) of the full output grid.
    grid_shape : tuple
        (height, width) of the full output grid.
    row_start, row_end, col_start, col_end : int
        Pixel indices of the chunk.

    Returns
    -------
    tuple : (left, bottom, right, top) bounds of the chunk.
    """
    left, bottom, right, top = grid_bounds
    height, width = grid_shape
    res_x = (right - left) / width
    res_y = (top - bottom) / height
    c_left = left + col_start * res_x
    c_right = left + col_end * res_x
    c_top = top - row_start * res_y
    c_bottom = top - row_end * res_y
    return (c_left, c_bottom, c_right, c_top)
