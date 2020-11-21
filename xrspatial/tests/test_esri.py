from os import path

from xrspatial.esri import arcpy_to_xarray
from xrspatial.esri import xarray_to_arcpy

from xrspatial.utils import doesnt_have_arcpy

import pytest
import numpy as np
import xarray as xr

HERE = path.abspath(path.dirname(__file__))
FIXTURES_DIR = path.join(HERE, 'fixtures')


@pytest.mark.skipif(doesnt_have_arcpy(), reason="ArcPy not available")
def test_arcpy_to_xarray():
    import arcpy

    # TODO: Create tiny ArcGIS Grid and add it in a new `fixtures` directory
    test_arcpy_grid = path.join(FIXTURES_DIR, 'test_arcpy_grid')

    arcpy_raster = arcpy.Raster(test_arcpy_grid)

    result = arcpy_to_xarray(arcpy_raster)
    assert isinstance(result, xr.DataArray)


@pytest.mark.skipif(doesnt_have_arcpy(), reason="ArcPy not available")
def test_xarray_to_arcpy():
    import arcpy

    elevation = np.asarray(
        [[1432.6542, 1432.4764, 1432.4764, 1432.1207, 1431.9429, np.nan],
            [1432.6542, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
            [1432.832, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
            [1432.832, 1432.6542, 1432.4764, 1432.4764, 1432.1207, np.nan],
            [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
            [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
            [1432.832, 1432.832, 1432.6542, 1432.4764, 1432.4764, np.nan]],
        dtype=np.float32)

    test_arr = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    result = xarray_to_arcpy(test_arr)

    # TODO: make this assertion correct
    assert isinstance(result, arcpy.Raster)
