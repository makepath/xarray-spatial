import numpy as np
import xarray as xr

from xrspatial.surface_distance import (
    surface_distance, surface_allocation, surface_direction,
)
from xrspatial.utils import has_cuda_and_cupy

from .common import get_xr_dataarray


def _sparse_source_raster(ny, nx, type):
    """Source raster with a handful of scattered target pixels.

    ``get_xr_dataarray(is_int=True)`` draws integers over ``[-nx, nx)``,
    and surface_distance treats every non-zero finite pixel as a source,
    so all but roughly 1 in ``2 * nx`` pixels seed the search at distance
    zero.  The Dijkstra relaxation body then never runs and the benchmark
    times a heap drain instead of distance propagation.  Scattered point
    sources make the frontier cross the whole grid, which is what these
    functions are for.
    """
    rng = np.random.default_rng(71942)
    z = np.zeros((ny, nx), dtype=np.float32)
    n_sources = max(4, (ny * nx) // 20000)
    rows = rng.integers(0, ny, n_sources)
    cols = rng.integers(0, nx, n_sources)
    z[rows, cols] = np.arange(1, n_sources + 1, dtype=np.float32)

    chunks = (max(1, ny // 2), max(1, nx // 2))
    if type == "cupy":
        if not has_cuda_and_cupy():
            raise NotImplementedError()
        import cupy
        z = cupy.asarray(z)
    elif type == "dask":
        import dask.array as da
        z = da.from_array(z, chunks=chunks)
    elif type == "dask+cupy":
        if not has_cuda_and_cupy():
            raise NotImplementedError()
        import cupy
        import dask.array as da
        z = da.from_array(cupy.asarray(z), chunks=chunks)
    elif type != "numpy":
        raise RuntimeError(f"Unrecognised type {type}")

    y = np.linspace(-90, 90, ny)
    x = np.linspace(-180, 180, nx)
    return xr.DataArray(z, coords=dict(y=y, x=x), dims=["y", "x"])


def _compute(result):
    if hasattr(result.data, "compute"):
        result.data.compute()


class SurfaceDistance:
    params = ([100, 300, 1000], ["numpy", "cupy", "dask", "dask+cupy"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.agg = _sparse_source_raster(ny, nx, type)
        self.elev = get_xr_dataarray((ny, nx), type)

        # A finite max_distance whose pixel radius stays inside one chunk
        # (chunks are ny//2 x nx//2) routes the dask backends through the
        # bounded map_overlap branch instead of the iterative tile one.
        cellsize = min(360.0 / (nx - 1), 180.0 / (ny - 1))
        self.max_distance = 20 * cellsize

    def time_surface_distance(self, nx, type):
        _compute(surface_distance(self.agg, self.elev))

    def time_surface_distance_bounded(self, nx, type):
        _compute(surface_distance(
            self.agg, self.elev, max_distance=self.max_distance))

    def time_surface_allocation(self, nx, type):
        _compute(surface_allocation(self.agg, self.elev))

    def time_surface_direction(self, nx, type):
        _compute(surface_direction(self.agg, self.elev))


class SurfaceDistanceGeodesic:
    """Great-circle horizontal distances from lat/lon coordinates.

    Runs a separate numba kernel (``_dijkstra_geodesic``) behind a
    precomputed per-pixel neighbour-distance grid, and costs about twice
    the planar path.  numpy only: the module raises NotImplementedError
    for geodesic on cupy, dask, and dask+cupy.
    """

    params = [300, 1000]
    param_names = ("nx",)

    def setup(self, nx):
        ny = nx // 2
        self.agg = _sparse_source_raster(ny, nx, "numpy")
        self.elev = get_xr_dataarray((ny, nx), "numpy")

    def time_surface_distance_geodesic(self, nx):
        surface_distance(self.agg, self.elev, method="geodesic")
