"""Sieve filter for removing small raster clumps.

Given a categorical raster and a pixel-count threshold, replaces
connected regions smaller than the threshold with the value of
their largest spatial neighbor.  Pairs with classification functions
(``natural_breaks``, ``reclassify``, etc.) and ``polygonize`` for
cleaning results before vectorization.

Supports all four backends: numpy, cupy, dask+numpy, dask+cupy.
"""

from __future__ import annotations

import warnings
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
    ngjit,
)

_MAX_ITERATIONS = 50


# ---------------------------------------------------------------------------
# Numba union-find labeling
# ---------------------------------------------------------------------------


@ngjit
def _uf_find(parent, x):
    """Find root of *x* with path halving."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@ngjit
def _uf_union(parent, rank, a, b):
    """Union by rank."""
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


@ngjit
def _label_connected(data, valid, neighborhood):
    """Single-pass connected-component labeling via union-find.

    Labels connected regions of same-value pixels in one O(n) pass,
    replacing the previous approach of calling ``scipy.ndimage.label``
    once per unique raster value.

    Returns
    -------
    region_map : ndarray of int32 (2D)
        Each pixel mapped to its region id (0 = nodata).
    region_val : ndarray of float64 (1D)
        Original raster value for each region id.
    n_regions : int
        Total number of regions + 1 (length of *region_val*).
    """
    rows = data.shape[0]
    cols = data.shape[1]
    n = rows * cols
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            idx = r * cols + c
            val = data[r, c]

            # Check left (already visited)
            if c > 0 and valid[r, c - 1] and data[r, c - 1] == val:
                _uf_union(parent, rank, idx, idx - 1)
            # Check up (already visited)
            if r > 0 and valid[r - 1, c] and data[r - 1, c] == val:
                _uf_union(parent, rank, idx, (r - 1) * cols + c)

            if neighborhood == 8:
                if (
                    r > 0
                    and c > 0
                    and valid[r - 1, c - 1]
                    and data[r - 1, c - 1] == val
                ):
                    _uf_union(parent, rank, idx, (r - 1) * cols + (c - 1))
                if (
                    r > 0
                    and c + 1 < cols
                    and valid[r - 1, c + 1]
                    and data[r - 1, c + 1] == val
                ):
                    _uf_union(parent, rank, idx, (r - 1) * cols + (c + 1))

    # Assign contiguous region IDs
    region_map_flat = np.zeros(n, dtype=np.int32)
    root_to_id = np.zeros(n, dtype=np.int32)
    region_val_buf = np.full(n + 1, np.nan, dtype=np.float64)
    next_id = 1

    for i in range(n):
        r = i // cols
        c = i % cols
        if not valid[r, c]:
            continue
        root = _uf_find(parent, i)
        if root_to_id[root] == 0:
            root_to_id[root] = next_id
            region_val_buf[next_id] = data[r, c]
            next_id += 1
        region_map_flat[i] = root_to_id[root]

    region_map = region_map_flat.reshape(rows, cols)
    return region_map, region_val_buf[:next_id], next_id


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------


def _build_adjacency(region_map, neighborhood):
    """Build a region adjacency dict from a labeled map.

    Encodes each (lo, hi) region pair as a single int64 so
    deduplication uses fast 1-D ``np.unique`` instead of the slower
    ``np.unique(axis=0)`` on 2-D pair arrays.

    Returns ``{region_id: set_of_neighbor_ids}``.
    """
    max_id = np.int64(region_map.max()) + 1
    encoded_parts: list[np.ndarray] = []

    def _collect(a, b):
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            return
        am = a[mask].ravel().astype(np.int64)
        bm = b[mask].ravel().astype(np.int64)
        lo = np.minimum(am, bm)
        hi = np.maximum(am, bm)
        encoded_parts.append(lo * max_id + hi)

    _collect(region_map[:-1, :], region_map[1:, :])
    _collect(region_map[:, :-1], region_map[:, 1:])
    if neighborhood == 8:
        _collect(region_map[:-1, :-1], region_map[1:, 1:])
        _collect(region_map[:-1, 1:], region_map[1:, :-1])

    adjacency: dict[int, set[int]] = defaultdict(set)
    if not encoded_parts:
        return adjacency

    encoded = np.unique(np.concatenate(encoded_parts))
    lo_arr = encoded // max_id
    hi_arr = encoded % max_id

    for a, b in zip(lo_arr.tolist(), hi_arr.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------


def _sieve_numpy(data, threshold, neighborhood, skip_values):
    """Replace connected regions smaller than *threshold* pixels."""
    result = data.astype(np.float64, copy=True)
    is_float = np.issubdtype(data.dtype, np.floating)
    valid = ~np.isnan(result) if is_float else np.ones(result.shape, dtype=bool)
    skip_set = set(skip_values) if skip_values is not None else set()

    converged = False
    for _ in range(_MAX_ITERATIONS):
        region_map, region_val, uid = _label_connected(
            result, valid, neighborhood
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
            converged = True
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
            converged = True
            break

    if not converged:
        warnings.warn(
            f"sieve() did not converge after {_MAX_ITERATIONS} iterations. "
            f"The result may still contain regions smaller than "
            f"threshold={threshold}.",
            stacklevel=3,
        )

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
