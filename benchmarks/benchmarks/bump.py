from xrspatial.bump import bump

from .common import get_xr_dataarray


class Bump:
    # One realistic size plus one small size; spread=2 so the per-bump
    # neighbourhood loop in _finish_bump is non-trivial.  The "dask" type
    # produces a multi-chunk template (common.get_xr_dataarray chunks at
    # ny//2, nx//2), so the dask backend exercises the real per-chunk
    # partition path rather than a single-chunk numpy passthrough.
    params = ([300, 1000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        # bump reads only agg.shape, chunks and backend from the template,
        # not its values.
        self.agg = get_xr_dataarray((ny, nx), type)

    def time_bump(self, nx, type):
        result = bump(agg=self.agg, spread=2)
        if hasattr(result.data, "compute"):
            result.data.compute()
