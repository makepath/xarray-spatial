"""Morphological raster operators.

Applies grayscale morphological operations using a structuring element
(kernel) over a 2D raster.  Erosion computes the local minimum and
dilation the local maximum within the kernel footprint.  Opening
(erosion then dilation) removes small bright features; closing
(dilation then erosion) fills small dark gaps.  Gradient, white
top-hat, and black top-hat are derived by subtracting pairs of the
above.

Supports all four backends: numpy, cupy, dask+numpy, dask+cupy.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr
from xarray import DataArray

from numba import cuda, prange

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _pad_array,
    _validate_boundary,
    _validate_raster,
    calc_cuda_dims,
    has_cuda_and_cupy,
    ngjit,
    not_implemented_func,
)
from xrspatial.dataset_support import supports_dataset


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    # Try /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    # Try psutil
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    # Fallback: 2 GB
    return 2 * 1024 ** 3


def _check_kernel_memory(rows, cols, ky, kx, func_name):
    """Raise MemoryError if the padded float64 array would exceed 50% of RAM.

    Each morphology backend allocates a padded copy of the input of shape
    ``(rows + ky - 1, cols + kx - 1)`` as float64, plus an output of shape
    ``(rows, cols)``.  Budget ~16 bytes per padded cell to cover both
    allocations plus small temporaries.
    """
    padded_rows = rows + ky - 1
    padded_cols = cols + kx - 1
    required = padded_rows * padded_cols * 16
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"{func_name}(): kernel shape ({ky}, {kx}) on a "
            f"({rows}, {cols}) raster needs ~{required / 1e9:.1f} GB "
            f"of padded float64 memory, but only "
            f"{available / 1e9:.1f} GB is available. "
            f"Use a smaller kernel."
        )


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def _circle_kernel(radius):
    """Return a boolean 2D array for a circular structuring element.

    The kernel has shape ``(2*radius+1, 2*radius+1)`` with ``True``
    for cells whose centre is within *radius* cells of the centre.
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def _validate_kernel(kernel, func_name):
    """Ensure *kernel* is a 2D uint8 numpy array with odd dimensions."""
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"{func_name}(): `kernel` must be a numpy ndarray, "
            f"got {type(kernel).__name__}"
        )
    if kernel.ndim != 2:
        raise ValueError(
            f"{func_name}(): `kernel` must be 2D, got {kernel.ndim}D"
        )
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"{func_name}(): `kernel` dimensions must be odd, "
            f"got {kernel.shape}"
        )
    return kernel.astype(np.uint8)


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

@ngjit
def _erode_kernel_numpy(data, kernel, rows, cols, ky, kx):
    """Erosion (local minimum) on a padded array.

    Only kernel cells with non-zero entries contribute to the output.
    The centre cell is included only when ``kernel[hy, hx]`` is non-zero.
    NaN neighbours included by the kernel propagate to NaN. Cells where
    the kernel covers no non-zero entries return NaN.
    """
    out = np.empty((rows, cols), dtype=data.dtype)
    for i in prange(rows):
        for j in range(cols):
            mn = np.nan
            seen = False
            for dy in range(ky):
                for dx in range(kx):
                    if kernel[dy, dx] == 0:
                        continue
                    v = data[i + dy, j + dx]
                    if v != v:  # NaN neighbour propagates
                        mn = v
                        seen = True
                        break
                    if not seen or v < mn:
                        mn = v
                        seen = True
                if seen and mn != mn:
                    break
            out[i, j] = mn
    return out


@ngjit
def _dilate_kernel_numpy(data, kernel, rows, cols, ky, kx):
    """Dilation (local maximum) on a padded array.

    Only kernel cells with non-zero entries contribute to the output.
    The centre cell is included only when ``kernel[hy, hx]`` is non-zero.
    NaN neighbours included by the kernel propagate to NaN. Cells where
    the kernel covers no non-zero entries return NaN.
    """
    out = np.empty((rows, cols), dtype=data.dtype)
    for i in prange(rows):
        for j in range(cols):
            mx = np.nan
            seen = False
            for dy in range(ky):
                for dx in range(kx):
                    if kernel[dy, dx] == 0:
                        continue
                    v = data[i + dy, j + dx]
                    if v != v:  # NaN neighbour propagates
                        mx = v
                        seen = True
                        break
                    if not seen or v > mx:
                        mx = v
                        seen = True
                if seen and mx != mx:
                    break
            out[i, j] = mx
    return out


def _morph_numpy(data, kernel, op, boundary):
    """Run erosion or dilation on a numpy array."""
    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    rows, cols = data.shape
    arr = data.astype(np.float64)
    if boundary == 'nan':
        padded = np.pad(arr, ((hy, hy), (hx, hx)),
                        mode='constant', constant_values=np.nan)
    else:
        padded = _pad_array(arr, (hy, hx), boundary)
    if op == 'erode':
        return _erode_kernel_numpy(padded, kernel, rows, cols, ky, kx)
    else:
        return _dilate_kernel_numpy(padded, kernel, rows, cols, ky, kx)


def _erode_numpy(data, kernel, boundary):
    return _morph_numpy(data, kernel, 'erode', boundary)


def _dilate_numpy(data, kernel, boundary):
    return _morph_numpy(data, kernel, 'dilate', boundary)


def _opening_numpy(data, kernel, boundary):
    tmp = _morph_numpy(data, kernel, 'erode', boundary)
    return _morph_numpy(tmp, kernel, 'dilate', boundary)


def _closing_numpy(data, kernel, boundary):
    tmp = _morph_numpy(data, kernel, 'dilate', boundary)
    return _morph_numpy(tmp, kernel, 'erode', boundary)


# ---------------------------------------------------------------------------
# cupy backend
# ---------------------------------------------------------------------------

@cuda.jit
def _erode_gpu(data, kernel, out, hy, hx, ky, kx):
    i, j = cuda.grid(2)
    rows = out.shape[0]
    cols = out.shape[1]
    if i < rows and j < cols:
        mn = np.nan
        seen = False
        for dy in range(ky):
            for dx in range(kx):
                if kernel[dy, dx] == 0:
                    continue
                v = data[i + dy, j + dx]
                if v != v:
                    mn = v
                    seen = True
                    break
                if not seen or v < mn:
                    mn = v
                    seen = True
            if seen and mn != mn:
                break
        out[i, j] = mn


@cuda.jit
def _dilate_gpu(data, kernel, out, hy, hx, ky, kx):
    i, j = cuda.grid(2)
    rows = out.shape[0]
    cols = out.shape[1]
    if i < rows and j < cols:
        mx = np.nan
        seen = False
        for dy in range(ky):
            for dx in range(kx):
                if kernel[dy, dx] == 0:
                    continue
                v = data[i + dy, j + dx]
                if v != v:
                    mx = v
                    seen = True
                    break
                if not seen or v > mx:
                    mx = v
                    seen = True
            if seen and mx != mx:
                break
        out[i, j] = mx


def _morph_cupy(data, kernel, op, boundary):
    import cupy as cp

    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    rows, cols = data.shape
    arr = cp.asarray(data, dtype=cp.float64)
    kern_dev = cp.asarray(kernel)
    if boundary == 'nan':
        padded = cp.pad(arr, ((hy, hy), (hx, hx)),
                        mode='constant', constant_values=cp.nan)
    else:
        padded = _pad_array(arr, (hy, hx), boundary)
    out = cp.empty((rows, cols), dtype=cp.float64)
    bpg, tpb = calc_cuda_dims((rows, cols))
    if op == 'erode':
        _erode_gpu[bpg, tpb](padded, kern_dev, out, hy, hx, ky, kx)
    else:
        _dilate_gpu[bpg, tpb](padded, kern_dev, out, hy, hx, ky, kx)
    return out


def _erode_cupy(data, kernel, boundary):
    return _morph_cupy(data, kernel, 'erode', boundary)


def _dilate_cupy(data, kernel, boundary):
    return _morph_cupy(data, kernel, 'dilate', boundary)


def _opening_cupy(data, kernel, boundary):
    tmp = _morph_cupy(data, kernel, 'erode', boundary)
    return _morph_cupy(tmp, kernel, 'dilate', boundary)


def _closing_cupy(data, kernel, boundary):
    tmp = _morph_cupy(data, kernel, 'dilate', boundary)
    return _morph_cupy(tmp, kernel, 'erode', boundary)


# ---------------------------------------------------------------------------
# dask + numpy backend
# ---------------------------------------------------------------------------

def _morph_chunk_numpy(chunk, kernel, op, block_info=None):
    """Process a single overlapped dask chunk."""
    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    rows = chunk.shape[0] - 2 * hy
    cols = chunk.shape[1] - 2 * hx
    if op == 'erode':
        interior = _erode_kernel_numpy(chunk, kernel, rows, cols, ky, kx)
    else:
        interior = _dilate_kernel_numpy(chunk, kernel, rows, cols, ky, kx)
    result = chunk.copy()
    result[hy:-hy, hx:-hx] = interior
    return result


def _morph_dask_numpy(data, kernel, op, boundary):
    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    _func = partial(_morph_chunk_numpy, kernel=kernel, op=op)
    return data.astype(np.float64).map_overlap(
        _func,
        depth=(hy, hx),
        boundary=_boundary_to_dask(boundary),
        meta=np.array(()),
    )


def _erode_dask_numpy(data, kernel, boundary):
    return _morph_dask_numpy(data, kernel, 'erode', boundary)


def _dilate_dask_numpy(data, kernel, boundary):
    return _morph_dask_numpy(data, kernel, 'dilate', boundary)


def _opening_dask_numpy(data, kernel, boundary):
    tmp = _morph_dask_numpy(data, kernel, 'erode', boundary)
    return _morph_dask_numpy(tmp, kernel, 'dilate', boundary)


def _closing_dask_numpy(data, kernel, boundary):
    tmp = _morph_dask_numpy(data, kernel, 'dilate', boundary)
    return _morph_dask_numpy(tmp, kernel, 'erode', boundary)


# ---------------------------------------------------------------------------
# dask + cupy backend
# ---------------------------------------------------------------------------

def _morph_chunk_cupy(chunk, kernel, op, block_info=None):
    """Process a single overlapped dask+cupy chunk."""
    import cupy as cp

    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    rows = chunk.shape[0] - 2 * hy
    cols = chunk.shape[1] - 2 * hx
    kern_dev = cp.asarray(kernel)
    out = cp.empty((rows, cols), dtype=cp.float64)
    bpg, tpb = calc_cuda_dims((rows, cols))
    if op == 'erode':
        _erode_gpu[bpg, tpb](chunk, kern_dev, out, hy, hx, ky, kx)
    else:
        _dilate_gpu[bpg, tpb](chunk, kern_dev, out, hy, hx, ky, kx)
    result = chunk.copy()
    result[hy:-hy, hx:-hx] = out
    return result


def _morph_dask_cupy(data, kernel, op, boundary):
    import cupy as cp

    ky, kx = kernel.shape
    hy, hx = ky // 2, kx // 2
    _func = partial(_morph_chunk_cupy, kernel=kernel, op=op)
    return data.astype(cp.float64).map_overlap(
        _func,
        depth=(hy, hx),
        boundary=_boundary_to_dask(boundary, is_cupy=True),
        meta=cp.array(()),
    )


def _erode_dask_cupy(data, kernel, boundary):
    return _morph_dask_cupy(data, kernel, 'erode', boundary)


def _dilate_dask_cupy(data, kernel, boundary):
    return _morph_dask_cupy(data, kernel, 'dilate', boundary)


def _opening_dask_cupy(data, kernel, boundary):
    tmp = _morph_dask_cupy(data, kernel, 'erode', boundary)
    return _morph_dask_cupy(tmp, kernel, 'dilate', boundary)


def _closing_dask_cupy(data, kernel, boundary):
    tmp = _morph_dask_cupy(data, kernel, 'dilate', boundary)
    return _morph_dask_cupy(tmp, kernel, 'erode', boundary)


# ---------------------------------------------------------------------------
# Dispatcher helpers
# ---------------------------------------------------------------------------

def _dispatch(agg, kernel, boundary, name, numpy_fn, cupy_fn, dask_fn, dask_cupy_fn):
    """Common dispatch logic for all four morphological functions."""
    _validate_raster(agg, func_name=name, name='agg', ndim=2)
    kernel = _validate_kernel(kernel, name)
    _validate_boundary(boundary)

    rows, cols = agg.shape
    ky, kx = kernel.shape
    _check_kernel_memory(rows, cols, ky, kx, name)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(numpy_fn, kernel=kernel, boundary=boundary),
        cupy_func=partial(cupy_fn, kernel=kernel, boundary=boundary),
        dask_func=partial(dask_fn, kernel=kernel, boundary=boundary),
        dask_cupy_func=partial(dask_cupy_fn, kernel=kernel, boundary=boundary),
    )

    out = mapper(agg)(agg.data)

    return DataArray(
        out,
        name=name,
        dims=agg.dims,
        coords=agg.coords,
        attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@supports_dataset
def morph_erode(agg, kernel=None, boundary='nan', name='erode'):
    """Apply morphological erosion (local minimum) to a 2D raster.

    Each output cell receives the minimum value found within the
    structuring element footprint centred on that cell.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Non-zero entries
        mark the footprint.  Defaults to a 3x3 square
        (``np.ones((3, 3), dtype=np.uint8)``).
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'erode'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Eroded raster with the same shape, coords, dims, and attrs
        as the input.

    Examples
    --------
    >>> from xrspatial.morphology import morph_erode
    >>> eroded = morph_erode(raster, kernel=np.ones((5, 5), dtype=np.uint8))
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return _dispatch(
        agg, kernel, boundary, name,
        _erode_numpy, _erode_cupy, _erode_dask_numpy, _erode_dask_cupy,
    )


@supports_dataset
def morph_dilate(agg, kernel=None, boundary='nan', name='dilate'):
    """Apply morphological dilation (local maximum) to a 2D raster.

    Each output cell receives the maximum value found within the
    structuring element footprint centred on that cell.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Non-zero entries
        mark the footprint.  Defaults to a 3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'dilate'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Dilated raster.

    Examples
    --------
    >>> from xrspatial.morphology import morph_dilate
    >>> dilated = morph_dilate(raster, kernel=np.ones((5, 5), dtype=np.uint8))
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return _dispatch(
        agg, kernel, boundary, name,
        _dilate_numpy, _dilate_cupy, _dilate_dask_numpy, _dilate_dask_cupy,
    )


@supports_dataset
def morph_opening(agg, kernel=None, boundary='nan', name='opening'):
    """Apply morphological opening (erosion then dilation) to a 2D raster.

    Opening removes small bright features and salt noise while
    preserving the overall shape of larger structures.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Defaults to a
        3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'opening'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Opened raster.

    Examples
    --------
    >>> from xrspatial.morphology import morph_opening
    >>> cleaned = morph_opening(binary_mask)
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return _dispatch(
        agg, kernel, boundary, name,
        _opening_numpy, _opening_cupy, _opening_dask_numpy, _opening_dask_cupy,
    )


@supports_dataset
def morph_closing(agg, kernel=None, boundary='nan', name='closing'):
    """Apply morphological closing (dilation then erosion) to a 2D raster.

    Closing fills small dark gaps and pepper noise while preserving
    the overall shape of larger structures.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Defaults to a
        3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'closing'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Closed raster.

    Examples
    --------
    >>> from xrspatial.morphology import morph_closing
    >>> filled = morph_closing(binary_mask)
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return _dispatch(
        agg, kernel, boundary, name,
        _closing_numpy, _closing_cupy, _closing_dask_numpy, _closing_dask_cupy,
    )


@supports_dataset
def morph_gradient(agg, kernel=None, boundary='nan', name='gradient'):
    """Morphological gradient: dilation minus erosion.

    Highlights edges and transitions in a 2D raster.  The result is
    always non-negative for non-NaN cells.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Defaults to a
        3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'gradient'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Gradient raster (dilate - erode).

    Examples
    --------
    >>> from xrspatial.morphology import morph_gradient
    >>> edges = morph_gradient(elevation)
    """
    dilated = morph_dilate(agg, kernel=kernel, boundary=boundary)
    eroded = morph_erode(agg, kernel=kernel, boundary=boundary)
    result = dilated - eroded
    result.name = name
    return result


@supports_dataset
def morph_white_tophat(agg, kernel=None, boundary='nan', name='white_tophat'):
    """White top-hat: original minus opening.

    Isolates bright features that are smaller than the structuring
    element.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Defaults to a
        3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'white_tophat'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        White top-hat raster (original - opening).

    Examples
    --------
    >>> from xrspatial.morphology import morph_white_tophat
    >>> bright = morph_white_tophat(elevation, kernel=circle_kernel)
    """
    opened = morph_opening(agg, kernel=kernel, boundary=boundary)
    result = agg - opened
    result.name = name
    return result


@supports_dataset
def morph_black_tophat(agg, kernel=None, boundary='nan', name='black_tophat'):
    """Black top-hat: closing minus original.

    Isolates dark features that are smaller than the structuring
    element.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster of numeric values.
    kernel : numpy.ndarray or None
        2D structuring element with odd dimensions.  Defaults to a
        3x3 square.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'black_tophat'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Black top-hat raster (closing - original).

    Examples
    --------
    >>> from xrspatial.morphology import morph_black_tophat
    >>> dark = morph_black_tophat(elevation, kernel=circle_kernel)
    """
    closed = morph_closing(agg, kernel=kernel, boundary=boundary)
    result = closed - agg
    result.name = name
    return result
