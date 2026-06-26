import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial import from_template, list_templates, slope
from xrspatial._template_data import _CITIES, _CITY_DEFAULT_RESOLUTION, _COUNTRY_BBOXES, _REGIONS
from xrspatial.tests.general_checks import cuda_and_cupy_available, dask_array_available


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
    assert agg.x.attrs["units"] == "degrees_east"
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


def test_resolution_honored_exactly():
    # the nyc bbox is 52814 m tall, not a whole number of 10 m cells, so res_y
    # used to drift to ~10.0008. The far edge is nudged so res stays exact.
    agg = from_template("nyc", resolution=10)
    assert agg.attrs["res"] == (10.0, 10.0)


def test_resolution_tuple_honored_exactly():
    agg = from_template("conus", resolution=(10000, 5000))
    assert agg.attrs["res"] == (10000.0, 5000.0)


def test_coords_match_requested_resolution():
    # pixel spacing equals the requested resolution on both axes
    agg = from_template("nyc", resolution=10)
    np.testing.assert_allclose(np.diff(agg.x.values), 10.0, atol=1e-6)
    np.testing.assert_allclose(-np.diff(agg.y.values), 10.0, atol=1e-6)


def test_nudge_keeps_centers_within_bbox():
    # nudging the far edges out by < half a cell still leaves every pixel
    # center inside the registry bbox.
    agg = from_template("nyc", resolution=10)
    left, bottom, right, top = _REGIONS["nyc"]["bounds"]
    assert left <= agg.x.values.min() and agg.x.values.max() <= right
    assert bottom <= agg.y.values.min() and agg.y.values.max() <= top


def test_country_resolution_honored_exactly():
    # country codes come back in EPSG:4326 (degrees) but go through the same
    # nudge math, so a degree resolution is honored exactly too.
    agg = from_template("FRA", resolution=0.25)
    assert agg.attrs["res"] == (0.25, 0.25)


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


def test_unknown_name_points_to_list_templates():
    # the error tells the user how to discover valid names
    with pytest.raises(ValueError, match="list_templates"):
        from_template("does-not-exist")


def test_list_templates_grouped():
    names = list_templates()
    assert set(names) == {"regions", "cities", "countries"}
    # each group lists exactly its registry keys, sorted
    assert names["regions"] == sorted(_REGIONS)
    assert names["cities"] == sorted(_CITIES)
    assert names["countries"] == sorted(_COUNTRY_BBOXES)


@pytest.mark.parametrize(
    "kind,registry",
    [("regions", _REGIONS), ("cities", _CITIES), ("countries", _COUNTRY_BBOXES)],
)
def test_list_templates_kind_filter(kind, registry):
    assert list_templates(kind) == sorted(registry)


def test_list_templates_bad_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        list_templates("city")


def test_list_templates_names_resolve():
    # every advertised name is a valid from_template argument; build one from
    # each group to confirm the listed names map straight to a template
    names = list_templates()
    for kind in ("regions", "cities", "countries"):
        agg = from_template(names[kind][0])
        assert agg.dims == ("y", "x")


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


def test_city_registry_integrity():
    # the _CITIES block is generated; guard the shape of every entry so a bad
    # regeneration is caught here rather than at from_template() call time.
    for key, entry in _CITIES.items():
        assert key == key.lower() and key.isascii(), key
        assert set(entry) == {"bounds", "crs", "lonlat", "label"}, key
        crs = entry["crs"]
        assert 32601 <= crs <= 32660 or 32701 <= crs <= 32760, (key, crs)
        lon_min, lat_min, lon_max, lat_max = entry["lonlat"]
        assert lon_min < lon_max and lat_min < lat_max, key
        left, bottom, right, top = entry["bounds"]
        assert left < right and bottom < top, key
        assert all(np.isfinite(v) for v in entry["bounds"]), key
    # the curated regions own their names; cities must not shadow them
    assert not set(_CITIES) & set(_REGIONS)


def test_city_sample_builds():
    # a slice of the registry builds and obeys the array contract; include a
    # couple of southern-hemisphere cities so the 327xx build path is exercised
    sample = sorted(_CITIES)[:20] + ["sao_paulo", "sydney"]
    for name in sample:
        agg = from_template(name)
        assert agg.dims == ("y", "x")
        # projected (UTM) coords carry CF metre units
        assert agg.x.attrs["units"] == "m"
        assert agg.x.attrs["standard_name"] == "projection_x_coordinate"
        assert agg.attrs["res"] == (_CITY_DEFAULT_RESOLUTION,
                                    _CITY_DEFAULT_RESOLUTION)
        # north-up, ascending x
        assert agg.y.values[0] > agg.y.values[-1]
        assert agg.x.values[0] < agg.x.values[-1]
        # every pixel center stays inside the registry bbox
        left, bottom, right, top = _CITIES[name]["bounds"]
        assert left <= agg.x.values.min() and agg.x.values.max() <= right
        assert bottom <= agg.y.values.min() and agg.y.values.max() <= top


def test_city_utm_spot_checks():
    # cities resolve to their UTM zone (a standard EPSG code, not a custom one)
    assert from_template("london").attrs["crs"] == 32630   # UTM 30N
    assert from_template("tokyo").attrs["crs"] == 32654     # UTM 54N
    # southern hemisphere -> 327xx
    assert 32701 <= from_template("sao_paulo").attrs["crs"] <= 32760


def test_city_case_insensitive():
    a = from_template("tokyo")
    b = from_template("TOKYO")
    np.testing.assert_array_equal(a.x.values, b.x.values)
    assert a.attrs == b.attrs


def test_city_name_collision_disambiguated():
    # same slug, different cities: the larger keeps the bare name, the other
    # gets an iso2 suffix, and the two resolve to distinct UTM zones
    assert "hyderabad" in _CITIES and "hyderabad_pk" in _CITIES
    bare = from_template("hyderabad").attrs["crs"]
    suffixed = from_template("hyderabad_pk").attrs["crs"]
    assert bare != suffixed


@dask_array_available
def test_dask_numpy_backend():
    import dask.array as da
    agg = from_template("nyc", backend="dask+numpy")
    assert isinstance(agg.data, da.Array)
    ref = from_template("nyc")
    np.testing.assert_array_equal(agg.x.values, ref.x.values)
    np.testing.assert_array_equal(agg.y.values, ref.y.values)
    assert agg.attrs == ref.attrs
    # resolution is honored exactly on the dask path too
    assert from_template("nyc", resolution=10, backend="dask+numpy").attrs[
        "res"
    ] == (10.0, 10.0)
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


def test_preserve_resolution_honored_exactly():
    agg = from_template("conus", preserve="area", resolution=50000)
    assert agg.attrs["res"] == (50000.0, 50000.0)


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


def test_cf_coordinate_units():
    # Units live on the coordinates (CF Conventions sec. 4), not on a
    # crs_units attr. Projected templates use metres; EPSG:4326 uses the
    # CF degree spellings.
    proj = from_template("conus")
    assert "crs_units" not in proj.attrs
    assert proj.x.attrs["units"] == "m"
    assert proj.x.attrs["standard_name"] == "projection_x_coordinate"
    assert proj.y.attrs["units"] == "m"
    assert proj.y.attrs["standard_name"] == "projection_y_coordinate"

    geo = from_template("FRA")
    assert "crs_units" not in geo.attrs
    assert geo.x.attrs["units"] == "degrees_east"
    assert geo.x.attrs["standard_name"] == "longitude"
    assert geo.y.attrs["units"] == "degrees_north"
    assert geo.y.attrs["standard_name"] == "latitude"


def test_cf_grid_mapping_attrs():
    # crs_name is gone; the CF grid-mapping keys identify the projection.
    proj = from_template("conus")
    assert "crs_name" not in proj.attrs
    assert proj.attrs["grid_mapping_name"] == "albers_conical_equal_area"
    assert "NAD83 / Conus Albers" in proj.attrs["crs_wkt"]

    geo = from_template("FRA")
    assert geo.attrs["grid_mapping_name"] == "latitude_longitude"
    assert "WGS 84" in geo.attrs["crs_wkt"]


def test_cf_grid_mapping_preserve_path():
    # The preserve path requires pyproj, so the CF keys are always present.
    agg = from_template("conus", preserve="area")
    assert agg.attrs["grid_mapping_name"] == "albers_conical_equal_area"
    assert "Conus Albers" in agg.attrs["crs_wkt"]


def test_grid_mapping_omitted_for_equal_earth():
    # Equal Earth (the preserve='area' fallback for the world bbox) has no
    # CF grid mapping, so grid_mapping_name is left off and crs_wkt stands
    # alone.
    agg = from_template("world", preserve="area")
    assert agg.attrs["crs"] == 8857
    assert "grid_mapping_name" not in agg.attrs
    assert "Equal Earth" in agg.attrs["crs_wkt"]


def test_cf_attrs_omitted_without_pyproj(monkeypatch):
    # Without pyproj the default (non-reproject) path stays dependency-free:
    # the CF grid-mapping keys are left off rather than raising.
    import sys

    monkeypatch.setitem(sys.modules, "pyproj", None)
    agg = from_template("conus")
    assert "grid_mapping_name" not in agg.attrs
    assert "crs_wkt" not in agg.attrs
    # The rest of the contract still holds.
    assert agg.attrs["crs"] == 5070
    assert agg.x.attrs["units"] == "m"
    assert agg.x.attrs["standard_name"] == "projection_x_coordinate"


def test_lite_crs_name_property():
    from xrspatial.reproject._lite_crs import CRS as LiteCRS

    assert LiteCRS(5070).name == "NAD83 / Conus Albers"
    assert LiteCRS(4326).name == "WGS 84"
