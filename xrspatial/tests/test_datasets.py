import dask.array as da
import xarray as xr

from xrspatial.datasets import make_terrain


def test_make_terrain():
    terrain = make_terrain()
    assert terrain is not None
    assert isinstance(terrain, xr.DataArray)
    assert isinstance(terrain.data, da.Array)
