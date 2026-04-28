"""Sieve filter for removing small raster clumps.

Given a categorical raster and a pixel-count threshold, replaces
connected regions smaller than the threshold with the value of
their largest spatial neighbor that is already at or above the
threshold.  Matches the single-pass semantics of GDAL's
``GDALSieveFilter`` / ``rasterio.features.sieve``.

Pairs with classification functions (``natural_breaks``,
``reclassify``, etc.) and ``polygonize`` for cleaning results
before vectorization.

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
    ngjit,
)



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

    Uses int32 indices internally, so the raster must have fewer than
    ~2.1 billion pixels (roughly 46 000 x 46 000).

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

    # --- Count unique regions first so region_val_buf is right-sized ---
    # Reuse rank array (no longer needed after union-find) as root_to_id.
    # This eliminates a separate n-element int32 allocation.
    root_to_id = rank  # alias; rank is dead after union-find
    for i in range(n):
        root_to_id[i] = 0  # clear

    n_regions = 0
    for i in range(n):
        r = i // cols
        c = i % cols
        if not valid[r, c]:
            continue
        root = _uf_find(parent, i)
        if root_to_id[root] == 0:
            root_to_id[root] = 1  # mark as seen
            n_regions += 1

    # Allocate region_val_buf at actual region count, not pixel count.
    # For a 46K x 46K raster with 100K regions this saves ~16 GB.
    region_val_buf = np.full(n_regions + 1, np.nan, dtype=np.float64)

    # Assign contiguous region IDs
    region_map_flat = np.zeros(n, dtype=np.int32)
    for i in range(n):
        root_to_id[i] = 0  # clear for ID assignment
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
# Memory guards
# ---------------------------------------------------------------------------

# Peak working set for the union-find pass:
#   result copy             8 bytes (float64)
#   parent                  4 bytes (int32)
#   rank / root_to_id       4 bytes (int32, reused)
#   region_map_flat         4 bytes (int32)
#   slack for region_val,
#   region_size, valid mask 8 bytes
# Total ~28 bytes/pixel.  Matches the budget the dask paths already use.
_BYTES_PER_PIXEL = 28


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


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 when CuPy / CUDA is unavailable or the query fails -- callers
    treat that as "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp

        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(rows, cols):
    """Raise MemoryError if the union-find pass would exceed 50% of RAM."""
    required = int(rows) * int(cols) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"sieve() on a {rows}x{cols} raster needs "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  "
            f"Connected-component labeling is a global operation that "
            f"cannot be chunked.  Consider downsampling or tiling the "
            f"input manually."
        )


def _check_gpu_memory(rows, cols):
    """Raise MemoryError when the CuPy round-trip would not fit.

    The CuPy backend transfers to host and runs the CPU sieve, so the
    host budget still applies; we also check free GPU RAM so a user
    with little VRAM gets a clear error before ``data.get()`` runs.
    Skips silently when the GPU memory query fails.
    """
    _check_memory(rows, cols)
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    # Round-trip needs the float64 input on device plus a float64 result.
    required = int(rows) * int(cols) * 16
    if required > 0.5 * available:
        raise MemoryError(
            f"sieve() on a {rows}x{cols} cupy raster needs "
            f"~{required / 1e9:.1f} GB of GPU memory for the round-trip "
            f"but only ~{available / 1e9:.1f} GB is free on the active "
            f"device.  Use a dask+cupy DataArray for out-of-core "
            f"processing or downsample the input."
        )


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------


def _sieve_numpy(data, threshold, neighborhood, skip_values):
    """Single-pass sieve matching GDAL's ``GDALSieveFilter`` semantics.

    A small region is only merged into a neighbor whose size is
    **>= threshold**.  If no such neighbor exists the region stays.
    Regions are processed smallest-first with in-place size updates
    so that earlier merges can grow a neighbor above threshold for
    later ones within the same pass.
    """
    _check_memory(data.shape[0], data.shape[1])
    result = data.astype(np.float64, copy=True)
    is_float = np.issubdtype(data.dtype, np.floating)
    valid = ~np.isnan(result) if is_float else np.ones(result.shape, dtype=bool)
    skip_set = set(skip_values) if skip_values is not None else set()

    region_map, region_val, uid = _label_connected(
        result, valid, neighborhood
    )
    region_size = np.bincount(
        region_map.ravel(), minlength=uid
    ).astype(np.int64)

    small_ids = [
        rid
        for rid in range(1, uid)
        if region_size[rid] < threshold
        and region_val[rid] not in skip_set
    ]
    if not small_ids:
        return result

    adjacency = _build_adjacency(region_map, neighborhood)

    # Process smallest regions first so earlier merges can grow
    # a neighbor above threshold for later candidates.
    small_ids.sort(key=lambda r: region_size[r])

    for rid in small_ids:
        if region_size[rid] == 0 or region_size[rid] >= threshold:
            continue

        neighbors = adjacency.get(rid)
        if not neighbors:
            continue  # surrounded by nodata only

        # Only merge into a neighbor that is already >= threshold.
        valid_neighbors = [
            n for n in neighbors if region_size[n] >= threshold
        ]
        if not valid_neighbors:
            continue

        largest_nid = max(valid_neighbors, key=lambda n: region_size[n])
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

    return result


# ---------------------------------------------------------------------------
# cupy backend  (CPU fallback – merge logic is serial)
# ---------------------------------------------------------------------------


def _sieve_cupy(data, threshold, neighborhood, skip_values):
    """CuPy backend: transfer to CPU, sieve, transfer back."""
    import cupy as cp

    _check_gpu_memory(data.shape[0], data.shape[1])
    np_result = _sieve_numpy(
        data.get(), threshold, neighborhood, skip_values
    )
    return cp.asarray(np_result)


# ---------------------------------------------------------------------------
# dask backends
# ---------------------------------------------------------------------------


def _sieve_dask(data, threshold, neighborhood, skip_values):
    """Dask+numpy backend: compute to numpy, sieve, wrap back."""
    avail = _available_memory_bytes()
    n_pixels = np.prod(data.shape)
    # Peak memory: input + result (float64 each) + parent + rank +
    # region_map_flat (int32 each) = 2*8 + 3*4 = 28 bytes/pixel.
    estimated_bytes = n_pixels * 28
    if estimated_bytes > 0.5 * avail:
        raise MemoryError(
            f"sieve() needs ~{estimated_bytes / 1e9:.1f} GB for the full "
            f"array plus CCL bookkeeping, but only ~{avail / 1e9:.1f} GB "
            f"is available.  Connected-component labeling is a global "
            f"operation that cannot be chunked.  Consider downsampling "
            f"or tiling the input manually."
        )

    np_data = data.compute()
    result = _sieve_numpy(
        np_data, threshold, neighborhood, skip_values
    )
    return da.from_array(result, chunks=data.chunks)


def _sieve_dask_cupy(data, threshold, neighborhood, skip_values):
    """Dask+CuPy backend: compute to cupy, sieve via CPU fallback, wrap back."""
    n_pixels = np.prod(data.shape)
    estimated_bytes = n_pixels * 28
    try:
        import cupy as cp

        free_gpu, _total = cp.cuda.Device().mem_info
        if estimated_bytes > 0.5 * free_gpu:
            raise MemoryError(
                f"sieve() needs ~{estimated_bytes / 1e9:.1f} GB for the "
                f"full array plus CCL bookkeeping, but only "
                f"~{free_gpu / 1e9:.1f} GB free GPU memory.  Connected-"
                f"component labeling is a global operation that cannot be "
                f"chunked.  Consider downsampling or tiling the input."
            )
    except (ImportError, AttributeError):
        pass

    cp_data = data.compute()
    result = _sieve_cupy(
        cp_data, threshold, neighborhood, skip_values
    )
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
    largest spatial neighbor that is already at or above *threshold*.
    Regions whose only neighbors are also below *threshold* are left
    unchanged, matching GDAL's single-pass semantics.  NaN pixels
    are always preserved.

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
    Uses single-pass semantics matching GDAL's ``GDALSieveFilter``.
    A small region is only merged into a neighbor whose current size
    is >= *threshold*.  If no such neighbor exists the region is left
    unchanged.

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
