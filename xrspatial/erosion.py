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

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
)

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


# ---------------------------------------------------------------------------
# GPU (CuPy / CUDA) implementation
# ---------------------------------------------------------------------------

if cuda is not None:

    @cuda.jit
    def _erode_gpu_kernel(
        heightmap, random_pos, boy, box, bw,
        inertia, capacity, deposition, erosion_rate,
        evaporation, gravity, min_slope, max_lifetime,
    ):
        """One CUDA thread per particle.

        Particles within a single launch are independent.  Conflicts at
        shared heightmap cells are resolved via ``cuda.atomic.add``.
        """
        tid = cuda.grid(1)
        n_iterations = random_pos.shape[0]
        if tid >= n_iterations:
            return

        height = heightmap.shape[0]
        width = heightmap.shape[1]
        n_brush = bw.shape[0]

        pos_x = random_pos[tid, 0] * (width - 3) + 1
        pos_y = random_pos[tid, 1] * (height - 3) + 1
        dir_x = 0.0
        dir_y = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for step in range(max_lifetime):
            node_x = int(pos_x)
            node_y = int(pos_y)

            if node_x < 1 or node_x >= width - 2:
                return
            if node_y < 1 or node_y >= height - 2:
                return

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
                return
            dir_x /= dir_len
            dir_y /= dir_len

            new_x = pos_x + dir_x
            new_y = pos_y + dir_y

            if new_x < 1 or new_x >= width - 2:
                return
            if new_y < 1 or new_y >= height - 2:
                return

            h_old = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) +
                     h01 * (1 - fx) * fy + h11 * fx * fy)

            new_node_x = int(new_x)
            new_node_y = int(new_y)
            new_fx = new_x - new_node_x
            new_fy = new_y - new_node_y
            h_new = (heightmap[new_node_y, new_node_x] * (1 - new_fx) * (1 - new_fy) +
                     heightmap[new_node_y, new_node_x + 1] * new_fx * (1 - new_fy) +
                     heightmap[new_node_y + 1, new_node_x] * (1 - new_fx) * new_fy +
                     heightmap[new_node_y + 1, new_node_x + 1] * new_fx * new_fy)

            h_diff = h_new - h_old

            neg_h_diff = -h_diff
            if neg_h_diff < min_slope:
                neg_h_diff = min_slope
            sed_capacity = neg_h_diff * speed * water * capacity

            if sediment > sed_capacity or h_diff > 0:
                if h_diff > 0:
                    amount = h_diff if h_diff < sediment else sediment
                else:
                    amount = (sediment - sed_capacity) * deposition

                sediment -= amount

                cuda.atomic.add(heightmap, (node_y, node_x),
                                amount * (1 - fx) * (1 - fy))
                cuda.atomic.add(heightmap, (node_y, node_x + 1),
                                amount * fx * (1 - fy))
                cuda.atomic.add(heightmap, (node_y + 1, node_x),
                                amount * (1 - fx) * fy)
                cuda.atomic.add(heightmap, (node_y + 1, node_x + 1),
                                amount * fx * fy)
            else:
                neg_h = -h_diff
                sc_minus_sed = (sed_capacity - sediment) * erosion_rate
                amount = sc_minus_sed if sc_minus_sed < neg_h else neg_h

                for k in range(n_brush):
                    ey = node_y + boy[k]
                    ex = node_x + box[k]
                    if 0 <= ey < height and 0 <= ex < width:
                        cuda.atomic.add(heightmap, (ey, ex),
                                        -(amount * bw[k]))

                sediment += amount

            speed_sq = speed * speed + h_diff * gravity
            if speed_sq > 0:
                speed = speed_sq ** 0.5
            else:
                speed = 0.0
            water *= (1 - evaporation)

            pos_x = new_x
            pos_y = new_y


def _erode_cupy(data, random_pos_np, boy, box, bw, params):
    """Run erosion on a CuPy array using the CUDA kernel."""
    hm = data.astype(cupy.float64)

    # Transfer brush and random positions to device
    random_pos_d = cupy.asarray(random_pos_np)
    boy_d = cupy.asarray(boy)
    box_d = cupy.asarray(box)
    bw_d = cupy.asarray(bw)

    n_particles = random_pos_np.shape[0]
    threads_per_block = 256
    blocks = (n_particles + threads_per_block - 1) // threads_per_block

    _erode_gpu_kernel[blocks, threads_per_block](
        hm, random_pos_d, boy_d, box_d, bw_d,
        params['inertia'], params['capacity'], params['deposition'],
        params['erosion'], params['evaporation'], params['gravity'],
        params['min_slope'], int(params['max_lifetime']),
    )
    cuda.synchronize()

    return hm.astype(cupy.float32)


def _erode_numpy(data, random_pos, boy, box, bw, params):
    """Run erosion on a NumPy array using the CPU kernel."""
    hm = data.astype(np.float64).copy()

    hm = _erode_cpu(
        hm, random_pos, boy, box, bw,
        params['inertia'], params['capacity'], params['deposition'],
        params['erosion'], params['evaporation'], params['gravity'],
        params['min_slope'], int(params['max_lifetime']),
    )

    return hm.astype(np.float32)


def _check_erosion_memory(data):
    """Raise MemoryError if the array is too large to materialize."""
    estimated = np.prod(data.shape) * data.dtype.itemsize
    # The erosion kernel needs ~3x: input copy + brush scratch + output
    working = estimated * 3
    try:
        from xrspatial.zonal import _available_memory_bytes
        avail = _available_memory_bytes()
    except ImportError:
        avail = 2 * 1024**3
    if working > 0.8 * avail:
        raise MemoryError(
            f"erode() needs ~{working / 1e9:.1f} GB to materialize and "
            f"process the full grid but only ~{avail / 1e9:.1f} GB "
            f"available.  Particle erosion is a global operation that "
            f"cannot be chunked.  Downsample the input or use a machine "
            f"with more RAM."
        )


def _erode_dask_numpy(data, random_pos, boy, box, bw, params):
    """Run erosion on a dask+numpy array.

    Erosion is a global operation (particles traverse the full grid),
    so we materialize to numpy, run the CPU kernel, then re-wrap.
    """
    _check_erosion_memory(data)
    np_data = data.compute()
    result = _erode_numpy(np_data, random_pos, boy, box, bw, params)
    return da.from_array(result, chunks=data.chunksize)


def _erode_dask_cupy(data, random_pos, boy, box, bw, params):
    """Run erosion on a dask+cupy array.

    Materializes to a single CuPy array, runs the GPU kernel, then
    re-wraps as dask.
    """
    _check_erosion_memory(data)
    cp_data = data.compute()  # CuPy ndarray
    result = _erode_cupy(cp_data, random_pos, boy, box, bw, params)
    return da.from_array(result, chunks=data.chunksize,
                         meta=cupy.array((), dtype=cupy.float32))


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

    # Precompute brush and random positions on the host
    boy, box, bw = _build_brush(int(p['radius']))
    rng = np.random.RandomState(seed)
    random_pos = rng.random((iterations, 2))

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda d: _erode_numpy(d, random_pos, boy, box, bw, p),
        cupy_func=lambda d: _erode_cupy(d, random_pos, boy, box, bw, p),
        dask_func=lambda d: _erode_dask_numpy(d, random_pos, boy, box, bw, p),
        dask_cupy_func=lambda d: _erode_dask_cupy(d, random_pos, boy, box, bw, p),
    )

    result_data = mapper(agg)(agg.data)

    return xr.DataArray(result_data, dims=agg.dims, coords=agg.coords,
                        attrs=agg.attrs, name=agg.name)
