from __future__ import annotations

import numpy as np
import xarray as xr
from numba import jit

try:
    import cupy
    from numba import cuda
except ImportError:
    class cupy(object):
        ndarray = False
    cuda = None

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import has_cuda_and_cupy, is_cupy_array, is_dask_cupy

# Default erosion parameters
_DEFAULT_PARAMS = dict(
    inertia=0.05,
    capacity=4.0,
    deposition=0.3,
    erosion=0.3,
    evaporation=0.01,
    gravity=4.0,
    min_slope=0.01,
    radius=3,
    max_lifetime=30,
)


def _build_brush(radius):
    """Precompute brush offsets and weights for the erosion kernel."""
    offsets_y = []
    offsets_x = []
    weights = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist2 = dx * dx + dy * dy
            if dist2 <= radius * radius:
                w = max(0.0, radius - dist2 ** 0.5)
                offsets_y.append(dy)
                offsets_x.append(dx)
                weights.append(w)
    weight_sum = sum(weights)
    boy = np.array(offsets_y, dtype=np.int32)
    box = np.array(offsets_x, dtype=np.int32)
    bw = np.array([w / weight_sum for w in weights], dtype=np.float64)
    return boy, box, bw


@jit(nopython=True, nogil=True)
def _erode_cpu(heightmap, random_pos, boy, box, bw,
               inertia, capacity, deposition, erosion_rate,
               evaporation, gravity, min_slope, max_lifetime):
    """Particle-based hydraulic erosion on a 2D heightmap (CPU).

    random_pos : float64 array of shape (n_iterations, 2) with pre-generated
                 random starting positions in [0, 1).
    """
    height, width = heightmap.shape
    n_iterations = random_pos.shape[0]
    n_brush = bw.shape[0]

    for iteration in range(n_iterations):
        pos_x = random_pos[iteration, 0] * (width - 3) + 1
        pos_y = random_pos[iteration, 1] * (height - 3) + 1
        dir_x = 0.0
        dir_y = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for step in range(max_lifetime):
            node_x = int(pos_x)
            node_y = int(pos_y)

            if node_x < 1 or node_x >= width - 2 or node_y < 1 or node_y >= height - 2:
                break

            fx = pos_x - node_x
            fy = pos_y - node_y

            h00 = heightmap[node_y, node_x]
            h10 = heightmap[node_y, node_x + 1]
            h01 = heightmap[node_y + 1, node_x]
            h11 = heightmap[node_y + 1, node_x + 1]

            grad_x = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            grad_y = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

            dir_x = dir_x * inertia - grad_x * (1 - inertia)
            dir_y = dir_y * inertia - grad_y * (1 - inertia)

            dir_len = (dir_x * dir_x + dir_y * dir_y) ** 0.5
            if dir_len < 1e-10:
                break
            dir_x /= dir_len
            dir_y /= dir_len

            new_x = pos_x + dir_x
            new_y = pos_y + dir_y

            if new_x < 1 or new_x >= width - 2 or new_y < 1 or new_y >= height - 2:
                break

            h_old = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + \
                    h01 * (1 - fx) * fy + h11 * fx * fy

            new_node_x = int(new_x)
            new_node_y = int(new_y)
            new_fx = new_x - new_node_x
            new_fy = new_y - new_node_y
            h_new = (heightmap[new_node_y, new_node_x] * (1 - new_fx) * (1 - new_fy) +
                     heightmap[new_node_y, new_node_x + 1] * new_fx * (1 - new_fy) +
                     heightmap[new_node_y + 1, new_node_x] * (1 - new_fx) * new_fy +
                     heightmap[new_node_y + 1, new_node_x + 1] * new_fx * new_fy)

            h_diff = h_new - h_old

            sed_capacity = max(-h_diff, min_slope) * speed * water * capacity

            if sediment > sed_capacity or h_diff > 0:
                if h_diff > 0:
                    amount = min(h_diff, sediment)
                else:
                    amount = (sediment - sed_capacity) * deposition

                sediment -= amount

                heightmap[node_y, node_x] += amount * (1 - fx) * (1 - fy)
                heightmap[node_y, node_x + 1] += amount * fx * (1 - fy)
                heightmap[node_y + 1, node_x] += amount * (1 - fx) * fy
                heightmap[node_y + 1, node_x + 1] += amount * fx * fy
            else:
                amount = min((sed_capacity - sediment) * erosion_rate, -h_diff)

                for k in range(n_brush):
                    ey = node_y + boy[k]
                    ex = node_x + box[k]
                    if 0 <= ey < height and 0 <= ex < width:
                        heightmap[ey, ex] -= amount * bw[k]

                sediment += amount

            speed_sq = speed * speed + h_diff * gravity
            speed = speed_sq ** 0.5 if speed_sq > 0 else 0.0
            water *= (1 - evaporation)

            pos_x = new_x
            pos_y = new_y

    return heightmap


def erode(agg, iterations=50000, seed=42, params=None):
    """Apply particle-based hydraulic erosion to a terrain DataArray.

    Erosion is a global operation that cannot be chunked, so dask arrays
    are materialized before processing and re-wrapped afterwards.

    Parameters
    ----------
    agg : xr.DataArray
        2D terrain heightmap.
    iterations : int
        Number of water droplets to simulate.
    seed : int
        Random seed for droplet placement.
    params : dict, optional
        Override default erosion constants. Keys:
        inertia, capacity, deposition, erosion, evaporation,
        gravity, min_slope, radius, max_lifetime.

    Returns
    -------
    xr.DataArray
        Eroded terrain.
    """
    p = dict(_DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    data = agg.data
    is_dask = da is not None and isinstance(data, da.Array)
    is_gpu = False

    if is_dask:
        if is_dask_cupy(agg):
            is_gpu = True
            data = data.compute()  # cupy ndarray
        else:
            data = data.compute()  # numpy ndarray

    if has_cuda_and_cupy() and is_cupy_array(data):
        is_gpu = True

    # work on a copy
    if is_gpu:
        hm = cupy.asnumpy(data).astype(np.float64).copy()
    else:
        hm = data.astype(np.float64).copy()

    # precompute brush and random positions outside JIT
    boy, box, bw = _build_brush(int(p['radius']))
    rng = np.random.RandomState(seed)
    random_pos = rng.random((iterations, 2))

    hm = _erode_cpu(
        hm, random_pos, boy, box, bw,
        p['inertia'], p['capacity'], p['deposition'], p['erosion'],
        p['evaporation'], p['gravity'], p['min_slope'],
        int(p['max_lifetime']),
    )

    result_data = hm.astype(np.float32)
    if is_gpu:
        result_data = cupy.asarray(result_data)

    if is_dask:
        if is_gpu:
            result_data = da.from_array(result_data,
                                        chunks=agg.data.chunksize,
                                        meta=cupy.array((), dtype=cupy.float32))
        else:
            result_data = da.from_array(result_data, chunks=agg.data.chunksize)

    return xr.DataArray(result_data, dims=agg.dims, coords=agg.coords,
                        attrs=agg.attrs, name=agg.name)
