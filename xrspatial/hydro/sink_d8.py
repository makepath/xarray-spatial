"""Sink identification: find and label depression cells in a D8 flow direction grid.

Identifies cells with direction code 0 (pit/flat with no downhill neighbor)
and labels connected groups using 8-connected BFS.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numba import cuda

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    _validate_raster,
    cuda_args,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for ``_sink_cpu``:
#   labels  : float64 -> 8
#   queue_r : int64   -> 8
#   queue_c : int64   -> 8
# Total ~24 bytes/pixel.  The caller-provided ``flow_dir`` array already
# lives in RAM before the kernel runs and is not double-counted here.
_BYTES_PER_PIXEL = 24

# GPU peak working set per pixel for ``_sink_cupy``:
#   labels : float64 -> 8
# Total ~8 bytes/pixel.  ``flow_dir_data`` already lives on the device
# before the kernel runs and is not double-counted here.
_GPU_BYTES_PER_PIXEL = 8


def _available_memory_bytes():
    """Best-effort estimate of available host memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 if CuPy / CUDA is unavailable or the query fails -- callers
    use that as a sentinel meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(height, width):
    """Raise MemoryError if the BFS kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"sink_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  Use a "
            f"dask-backed DataArray for out-of-core processing."
        )


def _check_gpu_memory(height, width):
    """Raise MemoryError if the CuPy kernel would exceed 50% of free GPU RAM.

    Skips the check (returns silently) when ``_available_gpu_memory_bytes``
    cannot determine the free memory -- e.g. on hosts without CUDA, where
    the kernel will fail at the cupy.asarray boundary anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(height) * int(width) * _GPU_BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"sink_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _sink_cpu(flow_dir, h, w, row_off, col_off, total_w):
    """8-connected BFS flood-fill CCL for sink cells (code 0).

    Labels each connected group of code-0 cells with a unique ID
    based on position: (row_off + r) * total_w + (col_off + c) + 1.
    """
    labels = np.empty((h, w), dtype=np.float64)
    labels[:] = np.nan

    dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                continue
            if v != 0.0:
                continue
            if labels[r, c] == labels[r, c]:  # already labeled
                continue

            label = float((row_off + r) * total_w + (col_off + c) + 1)
            labels[r, c] = label
            head = np.int64(0)
            tail = np.int64(0)
            queue_r[tail] = r
            queue_c[tail] = c
            tail += 1

            while head < tail:
                cr = queue_r[head]
                cc = queue_c[head]
                head += 1

                for k in range(8):
                    nr = cr + dy[k]
                    nc = cc + dx[k]
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    nv = flow_dir[nr, nc]
                    if nv != nv:
                        continue
                    if nv != 0.0:
                        continue
                    if labels[nr, nc] == labels[nr, nc]:
                        continue
                    labels[nr, nc] = label
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return labels


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _sink_init_gpu(flow_dir, labels, H, W):
    """Pits (code 0) get position-based ID, others get 0."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    v = flow_dir[i, j]
    if v != v:  # NaN
        labels[i, j] = 0.0
        return
    if v == 0.0:
        labels[i, j] = float(i * W + j + 1)
    else:
        labels[i, j] = 0.0


@cuda.jit
def _sink_propagate_gpu(labels, changed, H, W):
    """Min-label propagation: each sink cell takes minimum neighbor label."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    my_label = labels[i, j]
    if my_label <= 0.0:
        return

    min_label = my_label
    for k in range(8):
        if k == 0:
            dy, dx = -1, -1
        elif k == 1:
            dy, dx = -1, 0
        elif k == 2:
            dy, dx = -1, 1
        elif k == 3:
            dy, dx = 0, -1
        elif k == 4:
            dy, dx = 0, 1
        elif k == 5:
            dy, dx = 1, -1
        elif k == 6:
            dy, dx = 1, 0
        else:
            dy, dx = 1, 1

        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        nb = labels[ni, nj]
        if nb > 0.0 and nb < min_label:
            min_label = nb

    if min_label < my_label:
        labels[i, j] = min_label
        cuda.atomic.add(changed, 0, 1)


def _sink_cupy(flow_dir_data):
    """GPU driver for sink identification."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    labels = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))
    _sink_init_gpu[griddim, blockdim](flow_dir_f64, labels, H, W)

    max_iter = max(H, W)
    for _ in range(max_iter):
        changed[0] = 0
        _sink_propagate_gpu[griddim, blockdim](labels, changed, H, W)
        if int(changed[0]) == 0:
            break

    return cp.where(labels > 0, labels, cp.nan)


# =====================================================================
# Backend wrappers
# =====================================================================

def _run_numpy(data):
    h, w = data.shape
    return _sink_cpu(data.astype(np.float64), h, w, 0, 0, w)


# =====================================================================
# Cross-tile union-find for dask CCL
# =====================================================================
#
# Per-tile CCL produces globally unique IDs but does not merge
# components that span tile boundaries.  After the per-tile pass we
# walk each shared edge, record an equivalence whenever two adjacent
# boundary cells are both sinks, then union and remap labels.

def _uf_find(parent, x):
    """Path-halving find on a dict-backed union-find."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, a, b):
    """Union two label roots; smaller root wins so labels stay deterministic."""
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if ra < rb:
        parent[rb] = ra
    else:
        parent[ra] = rb


def _collect_boundary_equivalences(labels_np):
    """Return a list of (label_a, label_b) pairs from interior tile edges.

    *labels_np* is the materialized numpy result of the per-tile CCL pass.
    We scan every interior row/column boundary plus the two diagonal
    pairs (NE/SW and NW/SE) so 8-connectivity is preserved across tiles.

    Pairs where either side is NaN or 0 are skipped.  Pairs with the
    same label on both sides are skipped too.
    """
    pairs = []

    def _scan(a, b):
        # a and b are matched-shape slices; record (la, lb) where both
        # are sink labels (non-NaN, non-zero).
        if a.size == 0:
            return
        valid = ~(np.isnan(a) | np.isnan(b))
        if not valid.any():
            return
        am = a[valid]
        bm = b[valid]
        diff = am != bm
        if not diff.any():
            return
        la = am[diff].astype(np.int64)
        lb = bm[diff].astype(np.int64)
        for i in range(la.size):
            pairs.append((int(la[i]), int(lb[i])))

    # Vertical neighbors (up-down): every row boundary
    _scan(labels_np[:-1, :], labels_np[1:, :])
    # Horizontal neighbors (left-right)
    _scan(labels_np[:, :-1], labels_np[:, 1:])
    # Diagonal NW-SE
    _scan(labels_np[:-1, :-1], labels_np[1:, 1:])
    # Diagonal NE-SW
    _scan(labels_np[:-1, 1:], labels_np[1:, :-1])
    return pairs


def _build_label_remap(labels_np):
    """Build a {label: root_label} mapping for cross-tile sink merges.

    Only labels whose root differs from themselves end up in the dict;
    callers can short-circuit when the result is empty.
    """
    pairs = _collect_boundary_equivalences(labels_np)
    if not pairs:
        return {}

    parent = {}
    for a, b in pairs:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        _uf_union(parent, a, b)

    remap = {}
    for label in list(parent):
        root = _uf_find(parent, label)
        if root != label:
            remap[label] = root
    return remap


def _apply_label_remap(block, remap_keys, remap_vals):
    """Replace each label in *block* with its root from the remap arrays."""
    if remap_keys.size == 0:
        return block
    out = block.copy()
    # np.searchsorted gives O(N log K) lookup which beats a Python dict
    # in the inner loop and is easy to vectorize.
    flat = out.ravel()
    valid = ~np.isnan(flat)
    if not valid.any():
        return out
    vals = flat[valid].astype(np.int64)
    idx = np.searchsorted(remap_keys, vals)
    in_range = idx < remap_keys.size
    hits = np.zeros_like(vals, dtype=bool)
    hits[in_range] = remap_keys[idx[in_range]] == vals[in_range]
    if hits.any():
        new_vals = vals.astype(np.float64)
        new_vals[hits] = remap_vals[idx[hits]].astype(np.float64)
        flat[valid] = new_vals
        out = flat.reshape(block.shape)
    return out


def _merge_cross_tile_labels(labels_da):
    """Merge sink labels across tile boundaries.

    Materializes the per-tile CCL result so we can scan all boundaries,
    runs union-find, and applies the remap lazily via map_blocks.
    """
    # Materialize once to scan boundaries.  CCL is fundamentally a global
    # operation so we can't avoid touching every cell; the per-tile pass
    # already streamed.
    labels_np = labels_da.compute()
    remap = _build_label_remap(labels_np)
    if not remap:
        # Nothing to merge — wrap the materialized result back into dask
        # so the caller still gets a dask array with the original chunks.
        return da.from_array(labels_np, chunks=labels_da.chunks)

    keys = np.array(sorted(remap), dtype=np.int64)
    vals = np.array([remap[k] for k in keys], dtype=np.int64)

    def _remap_block(block, _keys=keys, _vals=vals):
        return _apply_label_remap(block, _keys, _vals)

    merged = da.from_array(labels_np, chunks=labels_da.chunks)
    return merged.map_blocks(
        _remap_block,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _run_dask_numpy(data):
    total_w = data.shape[1]

    def _tile_fn(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        row_off = block_info[0]['array-location'][0][0]
        col_off = block_info[0]['array-location'][1][0]
        h, w = block.shape
        return _sink_cpu(np.asarray(block, dtype=np.float64),
                         h, w, row_off, col_off, total_w)

    per_tile = da.map_blocks(
        _tile_fn, data,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )

    return _merge_cross_tile_labels(per_tile)


def _run_dask_cupy(data):
    """Dask+CuPy: convert to numpy dask, run CPU path, convert back."""
    import cupy as cp

    data_np = data.map_blocks(
        lambda b: b.get(), dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    result = _run_dask_numpy(data_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def sink_d8(flow_dir: xr.DataArray,
         name: str = 'sink') -> xr.DataArray:
    """Identify and label depression cells in a D8 flow direction grid.

    Finds cells with direction code 0 (pit/flat with no downhill
    neighbor) and labels connected groups using 8-connected
    component labeling.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid
        (codes 0/1/2/4/8/16/32/64/128; NaN for nodata).
    name : str, default='sink'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each sink cell is labeled with a unique
        group ID.  Non-sink cells and NaN cells are NaN.
    """
    _validate_raster(flow_dir, func_name='sink', name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        _check_memory(*data.shape)
        out = _run_numpy(data)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(*data.shape)
        out = _sink_cupy(data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _run_dask_cupy(data)

    elif da is not None and isinstance(data, da.Array):
        out = _run_dask_numpy(data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
