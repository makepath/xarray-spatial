import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial import from_template, slope
from xrspatial._template_data import _COUNTRY_BBOXES, _REGIONS
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)


def test_contract():
    agg = from_template("conus")
    assert isinstance(agg, xr.DataArray)
    assert agg.dims == ("y", "x")
    assert "y" in agg.coords and "x" in agg.coords
    res = agg.attrs["res"]
    assert isinstance(res, tuple) and len(res) == 2
    assert res[0] > 0 and res[1] > 0
    assert isinstance(agg.attrs["crs"], int)
    assert agg.x.attrs["units"] == "m"
    assert agg.y.attrs["units"] == "m"


def test_conus_albers():
    agg = from_template("conus")
    assert agg.attrs["crs"] == 5070
    # north-up: y descending
    assert agg.y.values[0] > agg.y.values[-1]
    # x ascending
    assert agg.x.values[0] < agg.x.values[-1]
    # within the registry bounds
    left, bottom, right, top = _REGIONS["conus"]["bounds"]
    assert left <= agg.x.values.min() and agg.x.values.max() <= right
    assert bottom <= agg.y.values.min() and agg.y.values.max() <= top


def test_case_insensitive_region():
    a = from_template("conus")
    b = from_template("CONUS")
    np.testing.assert_array_equal(a.x.values, b.x.values)
    np.testing.assert_array_equal(a.y.values, b.y.values)
    assert a.attrs == b.attrs


def test_nyc_resolves():
    agg = from_template("nyc")
    assert agg.attrs["crs"] == 32618
    assert agg.dims == ("y", "x")


def test_country_code():
    agg = from_template("FRA")
    assert agg.attrs["crs"] == 4326
    assert agg.x.attrs["units"] == "degree"
    assert np.isnan(agg.values).all()
    assert agg.shape[0] > 1 and agg.shape[1] > 1
    assert agg.name == "FRA"


def test_country_code_case_insensitive():
    a = from_template("fra")
    b = from_template("FRA")
    np.testing.assert_array_equal(a.x.values, b.x.values)


def test_resolution_controls_shape():
    coarse = from_template("conus", resolution=10000)
    fine = from_template("conus", resolution=5000)
    assert fine.size > coarse.size
    # realized res tracks the request closely
    assert abs(coarse.attrs["res"][0] - 10000) < 10000
    np.testing.assert_allclose(fine.attrs["res"][0], 5000, rtol=1e-2)


def test_resolution_tuple():
    agg = from_template("conus", resolution=(10000, 5000))
    rx, ry = agg.attrs["res"]
    assert rx > ry


def test_fill_and_dtype():
    agg = from_template("world")
    assert agg.dtype == np.float32
    assert np.isnan(agg.values).all()
    filled = from_template("world", fill=0.0)
    assert (filled.values == 0).all()


def test_world_grid():
    agg = from_template("world", resolution=1.0)
    assert agg.shape == (180, 360)
    assert agg.attrs["crs"] == 4326


@pytest.mark.parametrize("bad", ["does-not-exist", "ZZZ"])
def test_unknown_name_raises(bad):
    with pytest.raises(ValueError, match="Unknown template"):
        from_template(bad)


def test_nonpositive_resolution_raises():
    with pytest.raises(ValueError, match="positive"):
        from_template("conus", resolution=0)
    with pytest.raises(ValueError, match="positive"):
        from_template("conus", resolution=-5)


def test_over_fine_resolution_raises():
    with pytest.raises(ValueError, match="exceeding"):
        from_template("conus", resolution=1)


def test_non_string_name_raises():
    with pytest.raises(TypeError):
        from_template(42)


def test_bad_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        from_template("world", backend="tensorflow")


def test_registry_codes_resolve():
    # every curated region and a sample of country codes build without error
    for name in _REGIONS:
        agg = from_template(name, resolution=None)
        assert agg.dims == ("y", "x")
    for code in ["USA", "FRA", "JPN", "BRA", "RUS", "FJI"]:
        assert code in _COUNTRY_BBOXES
        agg = from_template(code)
        assert agg.attrs["crs"] == 4326


@dask_array_available
def test_dask_numpy_backend():
    import dask.array as da
    agg = from_template("nyc", backend="dask+numpy")
    assert isinstance(agg.data, da.Array)
    ref = from_template("nyc")
    np.testing.assert_array_equal(agg.x.values, ref.x.values)
    np.testing.assert_array_equal(agg.y.values, ref.y.values)
    assert agg.attrs == ref.attrs
    # values match once computed
    assert np.isnan(agg.compute().values).all()


@dask_array_available
def test_dask_alias():
    import dask.array as da
    agg = from_template("world", backend="dask")
    assert isinstance(agg.data, da.Array)


@cuda_and_cupy_available
def test_cupy_backend():
    import cupy
    agg = from_template("hawaii", backend="cupy")
    assert isinstance(agg.data, cupy.ndarray)
    ref = from_template("hawaii")
    np.testing.assert_array_equal(agg.x.values, ref.x.values)
    assert agg.attrs == ref.attrs


@cuda_and_cupy_available
@dask_array_available
def test_dask_cupy_backend():
    import cupy
    import dask.array as da
    agg = from_template("hawaii", backend="dask+cupy")
    assert isinstance(agg.data, da.Array)
    block = agg.data.blocks[0, 0].compute()
    assert isinstance(block, cupy.ndarray)


def test_downstream_slope_accepts_template():
    # an empty template feeds the array contract into a real op without error.
    # slope on an all-NaN grid is expected to emit All-NaN slice warnings; the
    # point here is that the contract is accepted, so silence that noise.
    agg = from_template("conus", resolution=20000)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = slope(agg)
    assert out.dims == ("y", "x")
    assert out.shape == agg.shape


# ---------------------------------------------------------------------------
# preserve (EPSG-coded projection by property)
# ---------------------------------------------------------------------------

pyproj = pytest.importorskip("pyproj")


def _proj(crs):
    return pyproj.CRS.from_epsg(crs).to_dict().get("proj")


@pytest.mark.parametrize("preserve", ["area", "shape"])
def test_preserve_contract(preserve):
    agg = from_template("conus", preserve=preserve)
    assert isinstance(agg.attrs["crs"], int)
    assert agg.dims == ("y", "x")
    assert agg.x.attrs["units"] == "m"
    assert np.isnan(agg.values).all()
    assert agg.dtype == np.float32


def test_preserve_area_curated_and_property():
    agg = from_template("conus", preserve="area")
    assert agg.attrs["crs"] == 5070            # curated US Albers
    assert _proj(agg.attrs["crs"]) == "aea"    # equal-area
    assert from_template("europe", preserve="area").attrs["crs"] == 3035


def test_preserve_area_country_equal_earth_fallback():
    agg = from_template("FRA", preserve="area")
    assert agg.attrs["crs"] == 8857            # Equal Earth fallback
    assert _proj(agg.attrs["crs"]) == "eqearth"


def test_preserve_shape_utm_zone():
    # conus centroid (~-95.85 lon) -> UTM zone 15N
    assert from_template("conus", preserve="shape").attrs["crs"] == 32615
    # Japan -> UTM 53N
    assert from_template("JPN", preserve="shape").attrs["crs"] == 32653
    # any northern UTM zone for France's (overseas-spanning) centroid
    fra = from_template("FRA", preserve="shape").attrs["crs"]
    assert 32601 <= fra <= 32660
    assert _proj(fra) == "utm"


def test_preserve_shape_region_override():
    # europe carries a curated conformal override (ETRS89 LCC Europe)
    assert from_template("europe", preserve="shape").attrs["crs"] == 3034
    assert _proj(3034) == "lcc"


def test_preserve_antimeridian_countries_build():
    for code in ("USA", "RUS", "FJI"):
        for preserve in ("area", "shape"):
            agg = from_template(code, preserve=preserve)
            assert isinstance(agg.attrs["crs"], int)
            assert agg.ndim == 2


def test_preserve_case_insensitive():
    assert from_template("conus", preserve="AREA").attrs["crs"] == 5070


@pytest.mark.parametrize("bad", ["distance", "direction", "bogus", "equalarea"])
def test_preserve_invalid_raises(bad):
    with pytest.raises(ValueError, match="preserve must be one of"):
        from_template("conus", preserve=bad)


def test_preserve_resolution_control():
    default = from_template("conus", preserve="area")
    coarse = from_template("conus", preserve="area", resolution=50000)
    assert coarse.size < default.size
    np.testing.assert_allclose(coarse.attrs["res"][0], 50000, rtol=0.05)


def test_preserve_none_unchanged():
    a = from_template("conus")
    b = from_template("conus", preserve=None)
    assert a.attrs["crs"] == b.attrs["crs"] == 5070
    np.testing.assert_array_equal(a.x.values, b.x.values)


@dask_array_available
def test_preserve_dask_backend():
    import dask.array as da
    agg = from_template("FRA", preserve="shape", backend="dask+numpy")
    assert isinstance(agg.data, da.Array)
    ref = from_template("FRA", preserve="shape")
    np.testing.assert_array_equal(agg.x.values, ref.x.values)
    assert agg.attrs == ref.attrs


@cuda_and_cupy_available
def test_preserve_cupy_backend():
    import cupy
    agg = from_template("conus", preserve="area", backend="cupy")
    assert isinstance(agg.data, cupy.ndarray)
    assert agg.attrs == from_template("conus", preserve="area").attrs


def test_preserve_downstream_slope():
    agg = from_template("conus", preserve="area", resolution=20000)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = slope(agg)
    assert out.shape == agg.shape
