import numpy as np
import xarray as xr

from xrspatial import zonal
from xrspatial.utils import has_cuda_and_cupy

from .common import get_xr_dataarray


def create_arr(data=None, H=10, W=10, backend='numpy'):
    assert(backend in ['numpy', 'cupy', 'dask'])
    if data is None:
        data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


class Zonal:
    # Note that rtxpy hillshade includes shadow calculations so timings are
    # not comparable with numpy and cupy hillshade.
    params = ([400, 1600, 3200], [2, 4, 8], ["numpy", "cupy"])
    param_names = ("raster_dim", "zone_dim", "backend")

    def setup(self, raster_dim, zone_dim, backend):
        W = H = raster_dim
        zW = zH = zone_dim
        # Make sure that the raster dim is multiple of the zones dim
        assert(W % zW == 0)
        assert(H % zH == 0)
        # initialize the values raster
        self.values = get_xr_dataarray((H, W), backend)

        # initialize the zones raster
        zones = xr.DataArray(np.zeros((H, W)))
        hstep = H//zH
        wstep = W//zW
        for i in range(zH):
            for j in range(zW):
                zones[i * hstep: (i+1)*hstep, j*wstep: (j+1)*wstep] = i*zW + j

        ''' zones now looks like this
        >>> zones = np.array([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]])
        '''

        self.zones = create_arr(zones, backend=backend)

        # Now setup the custom stat funcs
        if backend == 'cupy':
            import cupy
            l2normKernel = cupy.ReductionKernel(
                in_params='T x', out_params='float64 y',
                map_expr='x*x', reduce_expr='a+b',
                post_map_expr='y = sqrt(a)',
                identity='0', name='l2normKernel'
            )
            self.custom_stats = {
                'double_sum': lambda val: val.sum()*2,
                'l2norm': lambda val: np.sqrt(cupy.sum(val * val)),
                'l2normKernel': lambda val: l2normKernel(val)
            }
        else:
            from xrspatial.utils import ngjit

            @ngjit
            def l2normKernel(arr):
                acc = 0
                for x in arr:
                    acc += x * x
                return np.sqrt(acc)

            self.custom_stats = {
                'double_sum': lambda val: val.sum()*2,
                'l2norm': lambda val: np.sqrt(np.sum(val * val)),
                'l2normKernel': lambda val: l2normKernel(val)
            }

    def time_zonal_stats_default(self, raster_dim, zone_dim, backend):
        zonal.stats(zones=self.zones, values=self.values)

    def time_zonal_stats_custom(self, raster_dim, zone_dim, backend):
        zonal.stats(zones=self.zones, values=self.values,
                    stats_funcs=self.custom_stats)
