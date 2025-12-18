import xarray as xr

from xrspatial.datasets import make_terrain

from xrspatial.tests.general_checks import dask_array_available


@dask_array_available
def test_make_terrain():
    import dask.array as da

    terrain = make_terrain()
    assert terrain is not None
    assert isinstance(terrain, xr.DataArray)
    assert isinstance(terrain.data, da.Array)
