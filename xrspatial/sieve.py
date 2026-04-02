"""Sieve filter for removing small raster clumps.

Given a categorical raster and a pixel-count threshold, replaces
connected regions smaller than the threshold with the value of
their largest spatial neighbor.  Pairs with classification functions
(``natural_breaks``, ``reclassify``, etc.) and ``polygonize`` for
cleaning results before vectorization.

Supports all four backends: numpy, cupy, dask+numpy, dask+cupy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import xarray as xr
from xarray import DataArray

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
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
)


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------


def _build_adjacency(region_map, neighborhood):
    """Build a region adjacency dict from a labeled map using vectorized shifts.

    Returns ``{region_id: set_of_neighbor_ids}``.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)

    def _add_pairs(a, b):
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            return
        pairs = np.unique(
            np.column_stack([a[mask].ravel(), b[mask].ravel()]), axis=0
        )
        for x, y in pairs:
            adjacency[int(x)].add(int(y))
            adjacency[int(y)].add(int(x))

    # 4-connected directions (rook)
    _add_pairs(region_map[:-1, :], region_map[1:, :])  # vertical
    _add_pairs(region_map[:, :-1], region_map[:, 1:])  # horizontal

    # 8-connected adds diagonals (queen)
    if neighborhood == 8:
        _add_pairs(region_map[:-1, :-1], region_map[1:, 1:])  # SE
        _add_pairs(region_map[:-1, 1:], region_map[1:, :-1])  # SW

    return adjacency


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------


def _label_all_regions(result, valid, structure):
    """Label connected components per unique value.

    Returns
    -------
    region_map : ndarray of int32
        Each pixel mapped to its region id (0 = nodata).
    region_val : ndarray of float64
        Original raster value for each region id.
    n_total : int
        Total number of regions + 1 (length of *region_val*).
    """
    from scipy.ndimage import label

    unique_vals = np.unique(result[valid])
    region_map = np.zeros(result.shape, dtype=np.int32)
    region_val_list: list[float] = [np.nan]  # id 0 = nodata
    uid = 1

    for v in unique_vals:
        mask = (result == v) & valid
        labeled, n_features = label(mask, structure=structure)
        if n_features > 0:
            nonzero = labeled > 0
            region_map[nonzero] = labeled[nonzero] + (uid - 1)
            region_val_list.extend([float(v)] * n_features)
            uid += n_features

    region_val = np.array(region_val_list, dtype=np.float64)
    return region_map, region_val, uid


def _sieve_numpy(data, threshold, neighborhood, skip_values):
    """Replace connected regions smaller than *threshold* pixels."""
    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if neighborhood == 4
        else np.ones((3, 3), dtype=int)
    )

    result = data.astype(np.float64, copy=True)
    is_float = np.issubdtype(data.dtype, np.floating)
    valid = ~np.isnan(result) if is_float else np.ones(result.shape, dtype=bool)
    skip_set = set(skip_values) if skip_values is not None else set()

    for _ in range(50):  # convergence limit
        region_map, region_val, uid = _label_all_regions(
            result, valid, structure
        )
        region_size = np.bincount(
            region_map.ravel(), minlength=uid
        ).astype(np.int64)

        # Identify small regions eligible for merging
        small_ids = [
            rid
            for rid in range(1, uid)
            if region_size[rid] < threshold
            and region_val[rid] not in skip_set
        ]
        if not small_ids:
            break

        adjacency = _build_adjacency(region_map, neighborhood)

        # Process smallest regions first so they merge into larger neighbors
        small_ids.sort(key=lambda r: region_size[r])

        merged_any = False
        for rid in small_ids:
            if region_size[rid] == 0 or region_size[rid] >= threshold:
                continue

            neighbors = adjacency.get(rid)
            if not neighbors:
                continue  # surrounded by nodata only

            largest_nid = max(neighbors, key=lambda n: region_size[n])
            mask = region_map == rid
            result[mask] = region_val[largest_nid]

            # Update tracking in place
            region_map[mask] = largest_nid
            region_size[largest_nid] += region_size[rid]
            region_size[rid] = 0

            for n in neighbors:
                if n != largest_nid:
                    adjacency[n].discard(rid)
                    adjacency[n].add(largest_nid)
                    adjacency.setdefault(largest_nid, set()).add(n)
            if largest_nid in adjacency:
                adjacency[largest_nid].discard(rid)
            del adjacency[rid]
            merged_any = True

        if not merged_any:
            break

    return result


# ---------------------------------------------------------------------------
# cupy backend  (CPU fallback – merge logic is serial)
# ---------------------------------------------------------------------------


def _sieve_cupy(data, threshold, neighborhood, skip_values):
    """CuPy backend: transfer to CPU, sieve, transfer back."""
    import cupy as cp

    np_result = _sieve_numpy(data.get(), threshold, neighborhood, skip_values)
    return cp.asarray(np_result)


# ---------------------------------------------------------------------------
# dask backends
# ---------------------------------------------------------------------------


def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024**3


def _sieve_dask(data, threshold, neighborhood, skip_values):
    """Dask+numpy backend: compute to numpy, sieve, wrap back."""
    avail = _available_memory_bytes()
    estimated_bytes = np.prod(data.shape) * data.dtype.itemsize
    if estimated_bytes * 5 > 0.5 * avail:
        raise MemoryError(
            f"sieve() needs the full array in memory "
            f"(~{estimated_bytes * 5 / 1e9:.1f} GB) but only "
            f"~{avail / 1e9:.1f} GB is available.  Connected-component "
            f"labeling is a global operation that cannot be chunked.  "
            f"Consider downsampling or tiling the input manually."
        )

    np_data = data.compute()
    result = _sieve_numpy(np_data, threshold, neighborhood, skip_values)
    return da.from_array(result, chunks=data.chunks)


def _sieve_dask_cupy(data, threshold, neighborhood, skip_values):
    """Dask+CuPy backend: compute to cupy, sieve via CPU fallback, wrap back."""
    estimated_bytes = np.prod(data.shape) * data.dtype.itemsize
    try:
        import cupy as cp

        free_gpu, _total = cp.cuda.Device().mem_info
        if estimated_bytes * 5 > 0.5 * free_gpu:
            raise MemoryError(
                f"sieve() needs the full array on GPU "
                f"(~{estimated_bytes * 5 / 1e9:.1f} GB) but only "
                f"~{free_gpu / 1e9:.1f} GB free.  Connected-component "
                f"labeling is a global operation that cannot be chunked.  "
                f"Consider downsampling or tiling the input manually."
            )
    except (ImportError, AttributeError):
        pass

    cp_data = data.compute()
    result = _sieve_cupy(cp_data, threshold, neighborhood, skip_values)
    return da.from_array(result, chunks=data.chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sieve(
    raster: xr.DataArray,
    threshold: int = 10,
    neighborhood: int = 4,
    skip_values: Sequence[float] | None = None,
    name: str = "sieve",
) -> xr.DataArray:
    """Remove small connected regions from a classified raster.

    Identifies connected components of same-value pixels and replaces
    regions smaller than *threshold* pixels with the value of their
    largest spatial neighbor.  NaN pixels are always preserved.

    Parameters
    ----------
    raster : xr.DataArray
        2D classified or categorical raster.
    threshold : int, default=10
        Minimum region size in pixels.  Regions with fewer pixels
        are replaced by their largest neighbor's value.
    neighborhood : int, default=4
        Pixel connectivity: 4 (rook) or 8 (queen).
    skip_values : sequence of float, optional
        Category values whose regions are never replaced, regardless
        of size.  These regions can still serve as merge targets for
        neighboring small regions.
    name : str, default='sieve'
        Output DataArray name.

    Returns
    -------
    xr.DataArray
        Sieved raster with the same shape, dims, coords, and attrs.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.sieve import sieve

        >>> # Classified raster with salt-and-pepper noise
        >>> arr = np.array([[1, 1, 1, 2, 2],
        ...                 [1, 3, 1, 2, 2],
        ...                 [1, 1, 1, 2, 2],
        ...                 [2, 2, 2, 2, 2],
        ...                 [2, 2, 2, 2, 2]], dtype=np.float64)
        >>> raster = xr.DataArray(arr, dims=['y', 'x'])

        >>> # Remove regions smaller than 2 pixels
        >>> result = sieve(raster, threshold=2)
        >>> print(result.values)
        [[1. 1. 1. 2. 2.]
         [1. 1. 1. 2. 2.]
         [1. 1. 1. 2. 2.]
         [2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2.]]

    Notes
    -----
    This is a global operation: for dask-backed arrays the entire raster
    is computed into memory before sieving.  Connected-component labeling
    cannot be performed on individual chunks because regions may span
    chunk boundaries.

    The CuPy backends use a CPU fallback for the merge step, which is
    inherently serial.

    See Also
    --------
    xrspatial.zonal.regions : Connected-component labeling.
    xrspatial.classify.natural_breaks : Classification that may produce
        noisy output suitable for sieving.
    """
    _validate_raster(raster, func_name="sieve", name="raster", ndim=2)

    if neighborhood not in (4, 8):
        raise ValueError("`neighborhood` must be 4 or 8")

    if not isinstance(threshold, (int, np.integer)) or threshold < 1:
        raise ValueError("`threshold` must be a positive integer")

    data = raster.data

    if isinstance(data, np.ndarray):
        out = _sieve_numpy(data, threshold, neighborhood, skip_values)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _sieve_cupy(data, threshold, neighborhood, skip_values)
    elif da is not None and isinstance(data, da.Array):
        if is_dask_cupy(raster):
            out = _sieve_dask_cupy(
                data, threshold, neighborhood, skip_values
            )
        else:
            out = _sieve_dask(data, threshold, neighborhood, skip_values)
    else:
        raise TypeError(
            f"Unsupported array type {type(data).__name__} for sieve()"
        )

    return DataArray(
        out,
        name=name,
        dims=raster.dims,
        coords=raster.coords,
        attrs=raster.attrs,
    )
