"""Per-backend resampling via numba JIT (nearest/bilinear) or map_coordinates (cubic)."""
from __future__ import annotations

import math

import numpy as np
from numba import njit

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


_RESAMPLING_ORDERS = {
    'nearest': 0,
    'bilinear': 1,
    'cubic': 3,
}


def _validate_resampling(resampling):
    if resampling not in _RESAMPLING_ORDERS:
        raise ValueError(
            f"resampling must be one of {list(_RESAMPLING_ORDERS)}, "
            f"got {resampling!r}"
        )
    return _RESAMPLING_ORDERS[resampling]


# ---------------------------------------------------------------------------
# Numba kernels for nearest and bilinear resampling
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _resample_nearest_jit(src, row_coords, col_coords, nodata):
    """Nearest-neighbor resampling with NaN propagation."""
    h_out, w_out = row_coords.shape
    sh, sw = src.shape
    out = np.empty((h_out, w_out), dtype=np.float64)
    for i in range(h_out):
        for j in range(w_out):
            r = row_coords[i, j]
            c = col_coords[i, j]
            if r < -1.0 or r > sh or c < -1.0 or c > sw:
                out[i, j] = nodata
                continue
            ri = int(r + 0.5)
            ci = int(c + 0.5)
            if ri < 0:
                ri = 0
            if ri >= sh:
                ri = sh - 1
            if ci < 0:
                ci = 0
            if ci >= sw:
                ci = sw - 1
            v = src[ri, ci]
            # NaN check: works for float64
            if v != v:
                out[i, j] = nodata
            else:
                out[i, j] = v
    return out


@njit(nogil=True, cache=True)
def _resample_cubic_jit(src, row_coords, col_coords, nodata):
    """Catmull-Rom cubic resampling with NaN propagation.

    Separable: interpolate 4 row-slices along columns, then combine
    along rows.  Handles NaN inline (no second pass needed).
    """
    h_out, w_out = row_coords.shape
    sh, sw = src.shape
    out = np.empty((h_out, w_out), dtype=np.float64)
    for i in range(h_out):
        for j in range(w_out):
            r = row_coords[i, j]
            c = col_coords[i, j]
            if r < -1.0 or r > sh or c < -1.0 or c > sw:
                out[i, j] = nodata
                continue

            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            fr = r - r0
            fc = c - c0

            # Catmull-Rom column weights (a = -0.5)
            fc2 = fc * fc
            fc3 = fc2 * fc
            wc0 = -0.5 * fc3 + fc2 - 0.5 * fc
            wc1 = 1.5 * fc3 - 2.5 * fc2 + 1.0
            wc2 = -1.5 * fc3 + 2.0 * fc2 + 0.5 * fc
            wc3 = 0.5 * fc3 - 0.5 * fc2

            # Catmull-Rom row weights
            fr2 = fr * fr
            fr3 = fr2 * fr
            wr0 = -0.5 * fr3 + fr2 - 0.5 * fr
            wr1 = 1.5 * fr3 - 2.5 * fr2 + 1.0
            wr2 = -1.5 * fr3 + 2.0 * fr2 + 0.5 * fr
            wr3 = 0.5 * fr3 - 0.5 * fr2

            val = 0.0
            has_nan = False
            for di in range(4):
                ri = r0 - 1 + di
                ric = ri
                if ric < 0:
                    ric = 0
                elif ric >= sh:
                    ric = sh - 1
                # Interpolate along columns for this row
                rv = 0.0
                for dj in range(4):
                    cj = c0 - 1 + dj
                    cjc = cj
                    if cjc < 0:
                        cjc = 0
                    elif cjc >= sw:
                        cjc = sw - 1
                    sv = src[ric, cjc]
                    if sv != sv:  # NaN check
                        has_nan = True
                        break
                    if dj == 0:
                        rv = sv * wc0
                    elif dj == 1:
                        rv += sv * wc1
                    elif dj == 2:
                        rv += sv * wc2
                    else:
                        rv += sv * wc3
                if has_nan:
                    break
                if di == 0:
                    val = rv * wr0
                elif di == 1:
                    val += rv * wr1
                elif di == 2:
                    val += rv * wr2
                else:
                    val += rv * wr3

            out[i, j] = nodata if has_nan else val
    return out


@njit(nogil=True, cache=True)
def _resample_bilinear_jit(src, row_coords, col_coords, nodata):
    """Bilinear resampling with NaN propagation."""
    h_out, w_out = row_coords.shape
    sh, sw = src.shape
    out = np.empty((h_out, w_out), dtype=np.float64)
    for i in range(h_out):
        for j in range(w_out):
            r = row_coords[i, j]
            c = col_coords[i, j]
            if r < -1.0 or r > sh or c < -1.0 or c > sw:
                out[i, j] = nodata
                continue

            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            r1 = r0 + 1
            c1 = c0 + 1
            dr = r - r0
            dc = c - c0

            # Clamp to source bounds
            r0c = r0 if r0 >= 0 else 0
            r1c = r1 if r1 < sh else sh - 1
            c0c = c0 if c0 >= 0 else 0
            c1c = c1 if c1 < sw else sw - 1

            v00 = src[r0c, c0c]
            v01 = src[r0c, c1c]
            v10 = src[r1c, c0c]
            v11 = src[r1c, c1c]

            # If any neighbor is NaN, output nodata
            if v00 != v00 or v01 != v01 or v10 != v10 or v11 != v11:
                out[i, j] = nodata
            else:
                out[i, j] = (v00 * (1.0 - dr) * (1.0 - dc) +
                             v01 * (1.0 - dr) * dc +
                             v10 * dr * (1.0 - dc) +
                             v11 * dr * dc)
    return out


# ---------------------------------------------------------------------------
# Public numpy resampler
# ---------------------------------------------------------------------------

def _resample_numpy(source_window, src_row_coords, src_col_coords,
                    resampling='bilinear', nodata=np.nan):
    """Resample a numpy source window at fractional pixel coordinates.

    Uses numba JIT for nearest and bilinear (fast path), falls back to
    scipy.ndimage.map_coordinates for cubic.

    Parameters
    ----------
    source_window : ndarray (H_src, W_src)
    src_row_coords, src_col_coords : ndarray (H_out, W_out)
    resampling : str
    nodata : float

    Returns
    -------
    ndarray (H_out, W_out)
    """
    order = _validate_resampling(resampling)

    is_integer = np.issubdtype(source_window.dtype, np.integer)
    work = source_window.astype(np.float64) if is_integer else source_window

    # Ensure float64 and contiguous for numba
    if work.dtype != np.float64:
        work = work.astype(np.float64)
    work = np.ascontiguousarray(work)
    rc = np.ascontiguousarray(src_row_coords, dtype=np.float64)
    cc = np.ascontiguousarray(src_col_coords, dtype=np.float64)

    nd = float(nodata)

    if order == 0:
        result = _resample_nearest_jit(work, rc, cc, nd)
        if is_integer:
            result = np.round(result).astype(source_window.dtype)
        return result

    if order == 1:
        return _resample_bilinear_jit(work, rc, cc, nd)

    # Cubic: numba Catmull-Rom (handles NaN inline, no extra passes)
    return _resample_cubic_jit(work, rc, cc, nd)


# ---------------------------------------------------------------------------
# CUDA resampling kernels
# ---------------------------------------------------------------------------

if _HAS_CUDA:

    @_cuda.jit
    def _resample_nearest_cuda(src, row_coords, col_coords, out, nodata):
        """Nearest-neighbor resampling kernel (CUDA)."""
        i, j = _cuda.grid(2)
        h_out = out.shape[0]
        w_out = out.shape[1]
        if i >= h_out or j >= w_out:
            return
        sh = src.shape[0]
        sw = src.shape[1]
        r = row_coords[i, j]
        c = col_coords[i, j]
        if r < -1.0 or r > sh or c < -1.0 or c > sw:
            out[i, j] = nodata
            return
        ri = int(r + 0.5)
        ci = int(c + 0.5)
        if ri < 0:
            ri = 0
        if ri >= sh:
            ri = sh - 1
        if ci < 0:
            ci = 0
        if ci >= sw:
            ci = sw - 1
        v = src[ri, ci]
        # NaN check
        if v != v:
            out[i, j] = nodata
        else:
            out[i, j] = v

    @_cuda.jit
    def _resample_bilinear_cuda(src, row_coords, col_coords, out, nodata):
        """Bilinear resampling kernel (CUDA)."""
        i, j = _cuda.grid(2)
        h_out = out.shape[0]
        w_out = out.shape[1]
        if i >= h_out or j >= w_out:
            return
        sh = src.shape[0]
        sw = src.shape[1]
        r = row_coords[i, j]
        c = col_coords[i, j]
        if r < -1.0 or r > sh or c < -1.0 or c > sw:
            out[i, j] = nodata
            return

        r0 = int(math.floor(r))
        c0 = int(math.floor(c))
        r1 = r0 + 1
        c1 = c0 + 1
        dr = r - r0
        dc = c - c0

        # Clamp to source bounds
        r0c = r0 if r0 >= 0 else 0
        r1c = r1 if r1 < sh else sh - 1
        c0c = c0 if c0 >= 0 else 0
        c1c = c1 if c1 < sw else sw - 1

        v00 = src[r0c, c0c]
        v01 = src[r0c, c1c]
        v10 = src[r1c, c0c]
        v11 = src[r1c, c1c]

        # If any neighbor is NaN, output nodata
        if v00 != v00 or v01 != v01 or v10 != v10 or v11 != v11:
            out[i, j] = nodata
        else:
            out[i, j] = (v00 * (1.0 - dr) * (1.0 - dc) +
                         v01 * (1.0 - dr) * dc +
                         v10 * dr * (1.0 - dc) +
                         v11 * dr * dc)

    @_cuda.jit
    def _resample_cubic_cuda(src, row_coords, col_coords, out, nodata):
        """Catmull-Rom cubic resampling kernel (CUDA)."""
        i, j = _cuda.grid(2)
        h_out = out.shape[0]
        w_out = out.shape[1]
        if i >= h_out or j >= w_out:
            return
        sh = src.shape[0]
        sw = src.shape[1]
        r = row_coords[i, j]
        c = col_coords[i, j]
        if r < -1.0 or r > sh or c < -1.0 or c > sw:
            out[i, j] = nodata
            return

        r0 = int(math.floor(r))
        c0 = int(math.floor(c))
        fr = r - r0
        fc = c - c0

        # Catmull-Rom column weights (a = -0.5)
        fc2 = fc * fc
        fc3 = fc2 * fc
        wc0 = -0.5 * fc3 + fc2 - 0.5 * fc
        wc1 = 1.5 * fc3 - 2.5 * fc2 + 1.0
        wc2 = -1.5 * fc3 + 2.0 * fc2 + 0.5 * fc
        wc3 = 0.5 * fc3 - 0.5 * fc2

        # Catmull-Rom row weights
        fr2 = fr * fr
        fr3 = fr2 * fr
        wr0 = -0.5 * fr3 + fr2 - 0.5 * fr
        wr1 = 1.5 * fr3 - 2.5 * fr2 + 1.0
        wr2 = -1.5 * fr3 + 2.0 * fr2 + 0.5 * fr
        wr3 = 0.5 * fr3 - 0.5 * fr2

        val = 0.0
        has_nan = False

        # Row 0
        ric = r0 - 1
        if ric < 0:
            ric = 0
        elif ric >= sh:
            ric = sh - 1
        # Unrolled column loop for row 0
        cjc = c0 - 1
        if cjc < 0:
            cjc = 0
        elif cjc >= sw:
            cjc = sw - 1
        sv = src[ric, cjc]
        if sv != sv:
            has_nan = True
        if not has_nan:
            rv0 = sv * wc0
            cjc = c0
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv0 += sv * wc1
            cjc = c0 + 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv0 += sv * wc2
            cjc = c0 + 2
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv0 += sv * wc3
            val = rv0 * wr0

        # Row 1
        if not has_nan:
            ric = r0
            if ric < 0:
                ric = 0
            elif ric >= sh:
                ric = sh - 1
            cjc = c0 - 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv1 = sv * wc0
            cjc = c0
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv1 += sv * wc1
            cjc = c0 + 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv1 += sv * wc2
            cjc = c0 + 2
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv1 += sv * wc3
            val += rv1 * wr1

        # Row 2
        if not has_nan:
            ric = r0 + 1
            if ric < 0:
                ric = 0
            elif ric >= sh:
                ric = sh - 1
            cjc = c0 - 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv2 = sv * wc0
            cjc = c0
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv2 += sv * wc1
            cjc = c0 + 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv2 += sv * wc2
            cjc = c0 + 2
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv2 += sv * wc3
            val += rv2 * wr2

        # Row 3
        if not has_nan:
            ric = r0 + 2
            if ric < 0:
                ric = 0
            elif ric >= sh:
                ric = sh - 1
            cjc = c0 - 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv3 = sv * wc0
            cjc = c0
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv3 += sv * wc1
            cjc = c0 + 1
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv3 += sv * wc2
            cjc = c0 + 2
            if cjc < 0:
                cjc = 0
            elif cjc >= sw:
                cjc = sw - 1
            sv = src[ric, cjc]
            if sv != sv:
                has_nan = True
        if not has_nan:
            rv3 += sv * wc3
            val += rv3 * wr3

        if has_nan:
            out[i, j] = nodata
        else:
            out[i, j] = val


# ---------------------------------------------------------------------------
# Native CuPy resampler using CUDA kernels
# ---------------------------------------------------------------------------

def _resample_cupy_native(source_window, src_row_coords, src_col_coords,
                          resampling='bilinear', nodata=np.nan):
    """Resample using custom CUDA kernels (all data stays on GPU).

    Unlike ``_resample_cupy`` which uses ``cupyx.scipy.ndimage.map_coordinates``,
    this function uses hand-written CUDA kernels that match the Numba CPU
    kernels exactly, including inline NaN handling.

    Parameters
    ----------
    source_window : cupy.ndarray (H_src, W_src)
    src_row_coords, src_col_coords : cupy.ndarray (H_out, W_out)
    resampling : str
    nodata : float

    Returns
    -------
    cupy.ndarray (H_out, W_out)
    """
    if not _HAS_CUDA:
        raise RuntimeError("numba.cuda is required for _resample_cupy_native")

    import cupy as cp

    order = _validate_resampling(resampling)

    is_integer = cp.issubdtype(source_window.dtype, cp.integer)
    if is_integer:
        work = source_window.astype(cp.float64)
    else:
        work = source_window
    if work.dtype != cp.float64:
        work = work.astype(cp.float64)

    # Ensure inputs are CuPy arrays
    if not isinstance(src_row_coords, cp.ndarray):
        src_row_coords = cp.asarray(src_row_coords)
    if not isinstance(src_col_coords, cp.ndarray):
        src_col_coords = cp.asarray(src_col_coords)
    rc = cp.ascontiguousarray(src_row_coords, dtype=cp.float64)
    cc = cp.ascontiguousarray(src_col_coords, dtype=cp.float64)

    # Convert sentinel nodata to NaN so kernels can detect it
    if not np.isnan(nodata):
        work = work.copy()
        work[work == nodata] = cp.nan

    h_out, w_out = rc.shape
    out = cp.empty((h_out, w_out), dtype=cp.float64)
    nd = float(nodata)

    # Launch configuration: (16, 16) thread blocks
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (h_out + threads_per_block[0] - 1) // threads_per_block[0],
        (w_out + threads_per_block[1] - 1) // threads_per_block[1],
    )

    if order == 0:
        _resample_nearest_cuda[blocks_per_grid, threads_per_block](
            work, rc, cc, out, nd
        )
        if is_integer:
            out = cp.round(out).astype(source_window.dtype)
        return out

    if order == 1:
        _resample_bilinear_cuda[blocks_per_grid, threads_per_block](
            work, rc, cc, out, nd
        )
        return out

    # Cubic
    _resample_cubic_cuda[blocks_per_grid, threads_per_block](
        work, rc, cc, out, nd
    )
    return out


# ---------------------------------------------------------------------------
# CuPy resampler (uses cupyx.scipy.ndimage.map_coordinates)
# ---------------------------------------------------------------------------

def _resample_cupy(source_window, src_row_coords, src_col_coords,
                   resampling='bilinear', nodata=np.nan):
    """CuPy equivalent of ``_resample_numpy``.

    Control grid is on CPU (pyproj is CPU-only); coordinates are
    transferred to GPU for interpolation.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    order = _validate_resampling(resampling)

    is_integer = cp.issubdtype(source_window.dtype, cp.integer)
    if is_integer:
        work = source_window.astype(cp.float64)
    else:
        work = source_window

    # Transfer coordinate arrays to GPU
    if not isinstance(src_row_coords, cp.ndarray):
        src_row_coords = cp.asarray(src_row_coords)
    if not isinstance(src_col_coords, cp.ndarray):
        src_col_coords = cp.asarray(src_col_coords)

    if cp.issubdtype(work.dtype, cp.floating):
        nan_mask = cp.isnan(work)
        has_nan = bool(nan_mask.any())
    else:
        nan_mask = None
        has_nan = False
    if has_nan:
        if not is_integer:
            work = work.copy()
        work[nan_mask] = 0.0

    coords = cp.array([src_row_coords.ravel(), src_col_coords.ravel()])
    result = map_coordinates(
        work, coords, order=order, mode='constant', cval=0.0
    ).reshape(src_row_coords.shape)

    h, w = source_window.shape
    oob = (
        (src_row_coords < -1.0) | (src_row_coords > h) |
        (src_col_coords < -1.0) | (src_col_coords > w)
    )

    if has_nan:
        nan_weight = map_coordinates(
            nan_mask.astype(cp.float64), coords,
            order=order, mode='constant', cval=1.0
        ).reshape(src_row_coords.shape)
        oob = oob | (nan_weight > 0.1)

    result[oob] = nodata

    if is_integer and resampling == 'nearest':
        result = cp.round(result).astype(source_window.dtype)

    return result
