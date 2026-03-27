"""Output grid computation and chunk layout for reprojection."""
from __future__ import annotations

import numpy as np


def _transform_boundary(source_crs, target_crs, xs, ys):
    """Transform coordinate arrays, preferring Numba fast path over pyproj.

    Parameters
    ----------
    source_crs, target_crs : CRS-like
        Source and target coordinate reference systems.
    xs, ys : ndarray
        1-D arrays of x and y coordinates in *source_crs*.

    Returns
    -------
    tx, ty : ndarray
        Transformed coordinates as numpy arrays.
    """
    from ._projections import transform_points

    result = transform_points(source_crs, target_crs, xs, ys)
    if result is not None:
        return result

    # Fall back to pyproj
    from ._crs_utils import _require_pyproj

    pyproj = _require_pyproj()
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    tx, ty = transformer.transform(xs, ys)
    return np.asarray(tx), np.asarray(ty)


def _compute_output_grid(source_bounds, source_shape, source_crs, target_crs,
                         resolution=None, bounds=None, width=None, height=None):
    """Compute the output raster grid parameters.

    Parameters
    ----------
    source_bounds : tuple
        (left, bottom, right, top) in source CRS.
    source_shape : tuple
        (height, width) of source raster.
    source_crs, target_crs : CRS-like
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
    if bounds is None:
        # Transform source corners and edges to target CRS
        src_left, src_bottom, src_right, src_top = source_bounds

        # Clamp geographic coordinates away from singularities.
        # Exactly +/-180 longitude produces inf in many projections;
        # latitudes beyond +/-90 are invalid.
        if source_crs.is_geographic:
            clamp = 0.01  # degrees
            src_left = max(src_left, -180 + clamp)
            src_right = min(src_right, 180 - clamp)
            src_bottom = max(src_bottom, -90 + clamp)
            src_top = min(src_top, 90 - clamp)
            if src_left >= src_right:
                src_left, src_right = source_bounds[0], source_bounds[2]
            if src_bottom >= src_top:
                src_bottom, src_top = source_bounds[1], source_bounds[3]

        # Sample edges densely plus an interior grid so that
        # projections with curvature (e.g. UTM near zone edges)
        # don't underestimate the output bounding box.
        n_edge = 101
        n_interior = 21
        edge_xs = np.concatenate([
            np.linspace(src_left, src_right, n_edge),   # top edge
            np.linspace(src_left, src_right, n_edge),   # bottom edge
            np.full(n_edge, src_left),                   # left edge
            np.full(n_edge, src_right),                  # right edge
        ])
        edge_ys = np.concatenate([
            np.full(n_edge, src_top),
            np.full(n_edge, src_bottom),
            np.linspace(src_bottom, src_top, n_edge),
            np.linspace(src_bottom, src_top, n_edge),
        ])
        # Interior grid catches cases where the projected extent
        # bulges beyond the edges (e.g. Mercator near the poles).
        ix = np.linspace(src_left, src_right, n_interior)
        iy = np.linspace(src_bottom, src_top, n_interior)
        ixx, iyy = np.meshgrid(ix, iy)
        xs = np.concatenate([edge_xs, ixx.ravel()])
        ys = np.concatenate([edge_ys, iyy.ravel()])
        tx, ty = _transform_boundary(source_crs, target_crs, xs, ys)
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

        # Detect antimeridian / polar blow-up: if the transformed extent
        # is absurdly large relative to the source, use a densified
        # interior grid instead of just boundary points.
        raw_bounds = (float(tx.min()), float(ty.min()),
                      float(tx.max()), float(ty.max()))
        src_w_crs = src_right - src_left
        src_h_crs = src_top - src_bottom
        out_w_crs = raw_bounds[2] - raw_bounds[0]
        out_h_crs = raw_bounds[3] - raw_bounds[1]

        # Heuristic: if output span is > 50x source span in either axis,
        # boundary points likely wrapped around a singularity. Fall back
        # to a dense interior grid to get a tighter bounding box.
        ratio_x = out_w_crs / (abs(src_w_crs) + 1e-30)
        ratio_y = out_h_crs / (abs(src_h_crs) + 1e-30)
        if ratio_x > 50 or ratio_y > 50:
            # Sample a dense interior grid instead
            n_dense = 51
            ix = np.linspace(src_left, src_right, n_dense)
            iy = np.linspace(src_bottom, src_top, n_dense)
            ixx, iyy = np.meshgrid(ix, iy)
            itx, ity = _transform_boundary(
                source_crs, target_crs, ixx.ravel(), iyy.ravel()
            )
            itx = np.asarray(itx)
            ity = np.asarray(ity)
            ivalid = np.isfinite(itx) & np.isfinite(ity)
            if ivalid.any():
                itx = itx[ivalid]
                ity = ity[ivalid]
                # Use IQR-based bounds to reject outlier transforms
                q_lo, q_hi = 2, 98
                bounds = (
                    float(np.percentile(itx, q_lo)),
                    float(np.percentile(ity, q_lo)),
                    float(np.percentile(itx, q_hi)),
                    float(np.percentile(ity, q_hi)),
                )
            else:
                bounds = raw_bounds
        else:
            bounds = raw_bounds

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
        # Estimate from source resolution by transforming each axis
        # independently, then taking the geometric mean for a square pixel.
        src_h, src_w = source_shape
        src_left, src_bottom, src_right, src_top = source_bounds
        src_res_x = (src_right - src_left) / src_w
        src_res_y = (src_top - src_bottom) / src_h
        center_x = (src_left + src_right) / 2
        center_y = (src_bottom + src_top) / 2
        # Batch the three resolution-estimation points into one call
        pts_x = np.array([center_x, center_x + src_res_x, center_x])
        pts_y = np.array([center_y, center_y, center_y + src_res_y])
        tp_x, tp_y = _transform_boundary(source_crs, target_crs, pts_x, pts_y)
        tc_x, tc_y = float(tp_x[0]), float(tp_y[0])
        tx_x, tx_y = float(tp_x[1]), float(tp_y[1])
        ty_x, ty_y = float(tp_x[2]), float(tp_y[2])
        dx = np.hypot(tx_x - tc_x, tx_y - tc_y)
        dy = np.hypot(ty_x - tc_x, ty_y - tc_y)
        if dx == 0 or dy == 0:
            res_x = (right - left) / src_w
            res_y = (top - bottom) / src_h
        else:
            # Geometric mean for square pixels
            res_x = res_y = np.sqrt(dx * dy)

    # Compute dimensions.  Use round() instead of ceil() so that
    # floating-point noise (e.g. 677.0000000000001) does not add a
    # spurious extra row/column.
    if width is None:
        width = max(1, int(round((right - left) / res_x)))
    if height is None:
        height = max(1, int(round((top - bottom) / res_y)))

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
