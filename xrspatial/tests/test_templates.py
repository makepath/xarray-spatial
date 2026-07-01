import inspect
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial import from_template, list_templates, slope
from xrspatial._template_data import (_CITIES, _CITY_DEFAULT_RESOLUTION, _COUNTRY_BBOXES,
                                      _REGION_ALIASES, _REGIONS)
from xrspatial.tests.general_checks import cuda_and_cupy_available, dask_array_available


def test_public_functions_have_type_hints():
    # The public DataArray-producing API (e.g. generate_terrain) is fully
    # annotated; the templates entry points should match that convention so
    # the surface stays predictable.
    for func in (from_template, list_templates):
        sig = inspect.signature(func)
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"{func.__name__} is missing a return annotation"
        )
        for pname, param in sig.parameters.items():
            assert param.annotation is not inspect.Parameter.empty, (
                f"{func.__name__} parameter {pname!r} is missing a type hint"
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


# ---------------------------------------------------------------------------
# global-projection templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,crs",
    [("web_mercator", 3857), ("wgs84", 4326), ("latlon", 4326),
     ("equal_earth", 8857), ("pacific", 3832)],
)
def test_global_projection_contract(name, crs):
    agg = from_template(name)
    assert agg.attrs["crs"] == crs
    assert agg.dims == ("y", "x")
    assert agg.shape[0] > 1 and agg.shape[1] > 1
    assert np.isnan(agg.values).all()
    assert agg.dtype == np.float32
    assert agg.name == name
    # north-up (descending y), ascending x
    assert agg.y.values[0] > agg.y.values[-1]
    assert agg.x.values[0] < agg.x.values[-1]


@pytest.mark.parametrize("alias", ["wgs84", "latlon"])
def test_wgs84_latlon_alias_world(alias):
    # the aliases resolve to the EPSG:4326 'world' grid (same coords and attrs);
    # only the DataArray name reflects the spelling that was asked for
    a = from_template(alias)
    world = from_template("world")
    np.testing.assert_array_equal(a.x.values, world.x.values)
    np.testing.assert_array_equal(a.y.values, world.y.values)
    assert a.attrs == world.attrs
    assert a.name == alias


@pytest.mark.parametrize("name", ["web_mercator", "equal_earth", "latlon",
                                  "pacific", "pdc"])
def test_global_projection_case_insensitive(name):
    a = from_template(name)
    b = from_template(name.upper())
    np.testing.assert_array_equal(a.x.values, b.x.values)
    assert a.attrs == b.attrs


def test_pacific_pdc_mercator():
    # the Pacific Disaster Center projection (EPSG:3832, WGS 84 / PDC Mercator)
    # is a Pacific-centered Mercator, so the ocean is continuous; CF names it
    # 'mercator' and the WKT carries the PDC name.
    agg = from_template("pacific")
    assert agg.attrs["crs"] == 3832
    assert agg.attrs["grid_mapping_name"] == "mercator"
    assert "PDC Mercator" in agg.attrs["crs_wkt"]
    assert agg.x.attrs["units"] == "m"
    # 'pdc' is an alias for the same grid (only the name differs)
    pdc = from_template("pdc")
    np.testing.assert_array_equal(pdc.x.values, agg.x.values)
    assert pdc.attrs == agg.attrs
    assert pdc.name == "pdc"
    # conformal, so preserve='area' hands back the Equal Earth equal-area code
    assert from_template("pacific", preserve="area").attrs["crs"] == 8857


def test_web_mercator_metre_coords_within_bounds():
    agg = from_template("web_mercator")
    assert agg.x.attrs["units"] == "m"
    assert agg.x.attrs["standard_name"] == "projection_x_coordinate"
    left, bottom, right, top = _REGIONS["web_mercator"]["bounds"]
    assert left <= agg.x.values.min() and agg.x.values.max() <= right
    assert bottom <= agg.y.values.min() and agg.y.values.max() <= top


def test_global_resolution_honored_exactly():
    agg = from_template("web_mercator", resolution=100000)
    assert agg.attrs["res"] == (100000.0, 100000.0)


# ---------------------------------------------------------------------------
# regional templates (GLANCE continental equal-area projections)
# ---------------------------------------------------------------------------

_REGIONAL = [
    ("southeast_asia", 10594),
    ("central_america", 10598),
    ("caribbean", 10598),
    ("west_africa", 10592),
    ("north_africa", 10592),
    ("east_africa", 10592),
    ("southern_africa", 10592),
    ("south_asia", 10594),
    ("east_asia", 10594),
    ("central_asia", 10594),
    ("middle_east", 10594),
    ("south_america", 10603),
    ("oceania", 10601),
    ("australia", 10601),
    ("new_zealand", 10601),
    ("central_africa", 10592),
    ("north_asia", 10594),
    ("greenland", 10598),
    ("canada", 10598),
    ("mexico", 10598),
    ("great_lakes", 10598),
    ("pacific_northwest", 10598),
    ("gulf_coast", 10598),
    ("new_england", 10598),
    ("great_plains", 10598),
    ("american_southwest", 10598),
    ("amazon_basin", 10603),
    ("andes", 10603),
    ("southern_cone", 10603),
    ("western_europe", 10596),
    ("eastern_europe", 10596),
    ("northern_europe", 10596),
    ("southern_europe", 10596),
]


@pytest.mark.parametrize("name,crs", _REGIONAL)
def test_regional_template_contract(name, crs):
    agg = from_template(name)
    assert agg.attrs["crs"] == crs
    assert agg.dims == ("y", "x")
    assert agg.shape[0] > 1 and agg.shape[1] > 1
    assert np.isnan(agg.values).all()
    assert agg.dtype == np.float32
    assert agg.name == name
    # projected (LAEA) metre coordinates, north-up, ascending x
    assert agg.x.attrs["units"] == "m"
    assert agg.x.attrs["standard_name"] == "projection_x_coordinate"
    assert agg.y.values[0] > agg.y.values[-1]
    assert agg.x.values[0] < agg.x.values[-1]


def test_antarctica_contract():
    # Antarctica is the one region that is not GLANCE LAEA: it uses the de-facto
    # standard Antarctic Polar Stereographic (EPSG:3031), a projected metre CRS.
    agg = from_template("antarctica")
    assert agg.attrs["crs"] == 3031
    assert agg.dims == ("y", "x")
    assert agg.shape[0] > 1 and agg.shape[1] > 1
    assert np.isnan(agg.values).all()
    assert agg.dtype == np.float32
    assert agg.name == "antarctica"
    assert agg.x.attrs["units"] == "m"
    assert agg.x.attrs["standard_name"] == "projection_x_coordinate"
    assert agg.y.values[0] > agg.y.values[-1]
    assert agg.x.values[0] < agg.x.values[-1]


@pytest.mark.parametrize("name,crs", _REGIONAL)
def test_regional_template_centers_within_bounds(name, crs):
    agg = from_template(name)
    left, bottom, right, top = _REGIONS[name]["bounds"]
    assert left <= agg.x.values.min() and agg.x.values.max() <= right
    assert bottom <= agg.y.values.min() and agg.y.values.max() <= top


@pytest.mark.parametrize("name,crs", _REGIONAL)
def test_regional_template_case_insensitive(name, crs):
    a = from_template(name)
    b = from_template(name.upper())
    np.testing.assert_array_equal(a.x.values, b.x.values)
    assert a.attrs == b.attrs


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
    # each group lists exactly its registry keys, sorted; the regions group also
    # advertises the alias spellings (wgs84, latlon) so they are discoverable
    assert names["regions"] == sorted(set(_REGIONS) | set(_REGION_ALIASES))
    assert names["cities"] == sorted(_CITIES)
    assert names["countries"] == sorted(_COUNTRY_BBOXES)


@pytest.mark.parametrize(
    "kind,expected",
    [("regions", sorted(set(_REGIONS) | set(_REGION_ALIASES))),
     ("cities", sorted(_CITIES)),
     ("countries", sorted(_COUNTRY_BBOXES))],
)
def test_list_templates_kind_filter(kind, expected):
    assert list_templates(kind) == expected


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


# new_england at 10 m is ~6.9e9 cells, well past the eager cap, but only ~26k
# chunks at 512 -- a lazy grid that builds and indexes instantly. This is the
# repro from the issue.
_OVER_CAP = dict(name="new_england", resolution=10)


@dask_array_available
def test_chunks_promotes_eager_to_lazy_dask():
    import dask.array as da
    from xrspatial.templates import _MAX_CELLS
    # Supplying chunks promotes the default numpy backend to dask and skips the
    # cap, returning a lazy array that never materializes the full shape.
    agg = from_template(_OVER_CAP["name"], resolution=_OVER_CAP["resolution"],
                        chunks=512)
    assert isinstance(agg.data, da.Array)
    assert agg.size > _MAX_CELLS
    assert agg.data.chunksize == (512, 512)
    assert agg.attrs["res"] == (10.0, 10.0)
    # computing a single cell stays cheap and yields the NaN fill
    assert np.isnan(float(agg.data[0, 0].compute()))


@dask_array_available
def test_dask_backend_skips_cell_cap_without_chunks():
    import dask.array as da
    from xrspatial.templates import _MAX_CELLS
    # An explicit dask backend is lazy too, so the cap is skipped even when no
    # chunks are passed (the dask path falls back to 'auto').
    agg = from_template(_OVER_CAP["name"], resolution=_OVER_CAP["resolution"],
                        backend="dask+numpy")
    assert isinstance(agg.data, da.Array)
    assert agg.size > _MAX_CELLS


@cuda_and_cupy_available
@dask_array_available
def test_chunks_promotes_cupy_to_dask_cupy():
    import cupy
    import dask.array as da
    agg = from_template(_OVER_CAP["name"], resolution=_OVER_CAP["resolution"],
                        backend="cupy", chunks=512)
    assert isinstance(agg.data, da.Array)
    block = agg.data.blocks[0, 0].compute()
    assert isinstance(block, cupy.ndarray)


@dask_array_available
def test_over_fine_dask_chunk_count_raises():
    # The dask path skips the cell cap, but a typo-level resolution with a fixed
    # chunk size builds a runaway task graph. This is the issue #3557 repro:
    # conus at 1 m / chunks=512 is ~7e13 cells / 512^2 ~= 7e7 chunks. The guard
    # must raise from the estimate, BEFORE da.full builds the graph. Match the
    # chunk-count cap text specifically so this can't pass on the eager
    # cell-cap message (which also mentions "chunks").
    with pytest.raises(ValueError, match="chunk limit"):
        from_template("conus", resolution=1, chunks=512)


@dask_array_available
def test_explicit_dask_backend_chunk_count_raises():
    # Same guard via the non-promotion path: an explicit dask backend with a
    # fixed small chunk size on a typo-fine resolution must raise too.
    with pytest.raises(ValueError, match="chunk limit"):
        from_template("conus", resolution=1, backend="dask+numpy", chunks=512)


@dask_array_available
def test_over_fine_dask_coord_alloc_raises():
    # The dask cell-cap exemption keeps the grid data lazy, and the default
    # tiling grows its block so the chunk count stays under _MAX_CHUNKS -- but
    # the x/y coordinate vectors (width + height elements) are built eagerly, so
    # a typo-level fine resolution would allocate tens of GB of coordinates at
    # construction. conus @ 1 mm is ~9e9 coordinate elements (~72 GB) but only
    # ~2e5 chunks, so it slips past the chunk-count guard. The coordinate guard
    # must catch it first. Match its text specifically.
    from xrspatial.templates import _MAX_COORD_CELLS
    with pytest.raises(ValueError, match="coordinate vectors"):
        from_template("conus", resolution=0.001, backend="dask+numpy")
    # The promotion path (chunks given on an eager backend) is guarded too.
    with pytest.raises(ValueError, match="coordinate vectors"):
        from_template("conus", resolution=0.001, chunks=-1)
    assert _MAX_COORD_CELLS == 1_000_000_000


@dask_array_available
def test_auto_chunks_exempt_from_chunk_cap():
    import dask.array as da
    # 'auto' sizes blocks to the dask chunk-size config (~128 MB), so even a very
    # fine resolution stays well under the chunk cap and builds fine. The guard
    # keys on the real block count, so the auto path is not falsely tripped.
    agg = from_template("conus", resolution=1, chunks="auto")
    assert isinstance(agg.data, da.Array)
    from xrspatial.templates import _MAX_CHUNKS
    assert agg.data.npartitions <= _MAX_CHUNKS


@dask_array_available
def test_chunk_count_estimate_matches_dask():
    import dask.array as da
    from xrspatial.templates import _estimate_n_chunks
    # The estimate must agree with the block count dask actually builds, across
    # chunk forms, so the guard fires on the real graph size.
    for chunks in (256, 512, "auto", (300, 400)):
        built = da.full((4000, 5000), np.nan, dtype="float32", chunks=chunks)
        assert _estimate_n_chunks((4000, 5000), chunks) == built.npartitions


@dask_array_available
def test_legit_large_dask_grid_passes():
    import dask.array as da
    # The headroom case: new_england @ 10 m / chunks=512 is past the eager cell
    # cap but only ~26k chunks, far below the 1e6 chunk cap, so it must build.
    agg = from_template("new_england", resolution=10, chunks=512)
    assert isinstance(agg.data, da.Array)
    assert agg.data.npartitions < 1_000_000


@dask_array_available
def test_default_dask_chunks_are_balanced_square_blocks():
    import dask.array as da
    from xrspatial.templates import _DASK_BLOCK
    # The default 'auto' path tiles into even, square-ish blocks (no thin edge
    # slivers) sized near _DASK_BLOCK, so downstream map_overlap ops parallelize
    # cleanly. conus @ 1 km is bigger than one block, so it must split.
    agg = from_template("conus", resolution=1000, backend="dask")
    assert isinstance(agg.data, da.Array)
    assert agg.data.npartitions > 1
    for axis in agg.data.chunks:
        # balanced: every block within one cell of the others, none tiny
        assert max(axis) - min(axis) <= 1
        assert min(axis) > _DASK_BLOCK // 2


@dask_array_available
def test_small_dask_template_stays_one_chunk():
    # A grid at or below ~1.5x the block edge is not worth splitting; it comes
    # back as a single chunk just like before, so tiny templates keep zero
    # task-graph overhead.
    agg = from_template("conus", resolution=5000, backend="dask")
    assert agg.data.npartitions == 1


@dask_array_available
def test_default_tiling_block_grows_for_huge_grids():
    # At a typo-level fine resolution a fixed 2048 block would explode the graph.
    # The default block edge grows instead, keeping the count under the cap so
    # 'auto' never trips the guard on its own (the #3557 contract).
    from xrspatial.templates import _MAX_CHUNKS, _DASK_BLOCK
    agg = from_template("conus", resolution=1, backend="dask")
    assert agg.data.npartitions <= _MAX_CHUNKS
    # the block grew well past the nominal edge
    assert agg.data.chunks[0][0] > _DASK_BLOCK


@dask_array_available
def test_explicit_chunks_bypass_default_tiling():
    # An explicit chunks= is honored verbatim, not replaced by the default
    # tiling, so callers keep full control.
    agg = from_template("conus", resolution=1000, chunks=512)
    assert agg.data.chunks[0][0] == 512
    assert agg.data.chunks[1][0] == 512


@dask_array_available
def test_chunks_tuple_through_public_api():
    import dask.array as da
    # chunks may be a (chunk_y, chunk_x) tuple (a documented form); the public
    # path must honor it verbatim and keep the resolution exact, the same as the
    # int form. Only the int and 'auto' forms were exercised end-to-end before;
    # the tuple form was checked only against the internal _estimate_n_chunks.
    agg = from_template("conus", resolution=1000, chunks=(300, 400))
    assert isinstance(agg.data, da.Array)
    assert agg.data.chunksize == (300, 400)
    assert agg.attrs["res"] == (1000.0, 1000.0)


def test_single_pixel_grid():
    # a resolution coarser than the whole study-area box clamps width and height
    # to the max(1, ...) floor, giving a 1x1 grid that still obeys the contract.
    agg = from_template("conus", resolution=5_000_000)
    assert agg.shape == (1, 1)
    assert agg.attrs["res"] == (5_000_000.0, 5_000_000.0)
    assert agg.dims == ("y", "x")
    assert np.isnan(agg.values).all()


def test_strip_grid():
    # a huge resolution on one axis only clamps that axis to 1, producing an
    # Nx1 / 1xN strip (the other axis stays multi-cell).
    strip = from_template("conus", resolution=(20000, 5_000_000))
    assert strip.shape[0] == 1
    assert strip.shape[1] > 1
    assert strip.dims == ("y", "x")


def test_resolution_tuple_wrong_length_raises():
    # a resolution tuple must be exactly (res_x, res_y); any other length is a
    # validation error.
    for bad in [(1000,), (1000, 2000, 3000)]:
        with pytest.raises(ValueError, match=r"resolution tuple must be"):
            from_template("conus", resolution=bad)


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


@pytest.mark.parametrize("name", ["web_mercator", "equal_earth"])
def test_global_preserve_picks_world_projection(name):
    # the global templates carry the same area/shape EPSG hints as 'world', so
    # preserve='shape' lands on World Mercator (EPSG:3395) instead of a stray
    # UTM zone for the (0, 0) centroid, and preserve='area' lands on Equal Earth
    agg_shape = from_template(name, preserve="shape")
    assert agg_shape.attrs["crs"] == 3395
    assert _proj(3395) == "merc"
    assert from_template(name, preserve="area").attrs["crs"] == 8857


def test_grid_mapping_omitted_for_equal_earth():
    # Equal Earth (the preserve='area' fallback for the world bbox) has no
    # CF grid mapping, so grid_mapping_name is left off and crs_wkt stands
    # alone.
    agg = from_template("world", preserve="area")
    assert agg.attrs["crs"] == 8857
    assert "grid_mapping_name" not in agg.attrs
    assert "Equal Earth" in agg.attrs["crs_wkt"]


@pytest.mark.parametrize("name,crs", _REGIONAL)
def test_regional_bounds_match_reprojected_lonlat(name, crs):
    # the stored bounds are hand-maintained: the lon/lat box projected into the
    # GLANCE CRS. Recompute them here so a future edit or regeneration that
    # drifts bounds out of sync with lonlat (which would misgeoreference the
    # grid) fails loudly instead of shipping a wrong canvas.
    from xrspatial.reproject._crs_utils import _resolve_crs
    from xrspatial.reproject._grid import _edge_samples, _transform_boundary

    lon_min, lat_min, lon_max, lat_max = _REGIONS[name]["lonlat"]
    xs, ys = _edge_samples(lon_min, lat_min, lon_max, lat_max, 101)
    tx, ty = _transform_boundary(_resolve_crs(4326), _resolve_crs(crs), xs, ys)
    tx, ty = np.asarray(tx), np.asarray(ty)
    valid = np.isfinite(tx) & np.isfinite(ty)
    recomputed = (tx[valid].min(), ty[valid].min(),
                  tx[valid].max(), ty[valid].max())
    # bounds are stored rounded to the metre; allow a couple of metres slack
    np.testing.assert_allclose(_REGIONS[name]["bounds"], recomputed, atol=2.0)


@pytest.mark.parametrize("name,crs", _REGIONAL)
def test_regional_template_grid_mapping(name, crs):
    # the GLANCE regions are Lambert azimuthal equal-area, which CF defines, so
    # grid_mapping_name is present and crs_wkt names the GLANCE projection.
    agg = from_template(name)
    assert agg.attrs["grid_mapping_name"] == "lambert_azimuthal_equal_area"
    assert "GLANCE" in agg.attrs["crs_wkt"]
    assert _proj(crs) == "laea"


def test_antarctica_grid_mapping_and_preserve():
    # Antarctic Polar Stereographic is conformal, so grid_mapping_name is
    # 'polar_stereographic' and preserve='area' must hand back a real equal-area
    # code (the south-polar LAEA EPSG:6932), not 3031.
    agg = from_template("antarctica")
    assert agg.attrs["grid_mapping_name"] == "polar_stereographic"
    assert _proj(3031) == "stere"
    assert from_template("antarctica", preserve="area").attrs["crs"] == 6932
    assert _proj(6932) == "laea"
    assert from_template("antarctica", preserve="shape").attrs["crs"] == 3031


@pytest.mark.parametrize(
    "name,crs,wkt_marker",
    [("web_mercator", 3857, "Pseudo-Mercator"),
     ("equal_earth", 8857, "Equal Earth")],
)
def test_global_projection_grid_mapping(name, crs, wkt_marker):
    # neither Pseudo-Mercator nor Equal Earth has a CF grid_mapping_name, so the
    # key is left off and crs_wkt stands alone (carrying the human CRS name).
    agg = from_template(name)
    assert agg.attrs["crs"] == crs
    assert "grid_mapping_name" not in agg.attrs
    assert wkt_marker in agg.attrs["crs_wkt"]
    assert agg.x.attrs["units"] == "m"


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


# ---------------------------------------------------------------------------
# tiling-optimized extents: the default dask grid pads its shape out to whole
# blocks so every chunk is full-size, and explicit height/width set the grid
# shape exactly (extent floats off the region anchor).
# ---------------------------------------------------------------------------


@dask_array_available
def test_default_dask_extent_padded_to_whole_tiles():
    import dask.array as da
    from xrspatial.templates import _DASK_BLOCK
    # conus @ 1 km is a multi-block grid (~3105 x 5865). The default tiling pads
    # the shape up to exact multiples of the block edge so every chunk is a full
    # _DASK_BLOCK square -- no ragged remainder block on the far edge.
    agg = from_template("conus", resolution=1000, backend="dask")
    assert isinstance(agg.data, da.Array)
    assert agg.shape[0] % _DASK_BLOCK == 0
    assert agg.shape[1] % _DASK_BLOCK == 0
    for axis in agg.data.chunks:
        assert set(axis) == {_DASK_BLOCK}


@dask_array_available
def test_padding_keeps_resolution_exact_and_covers_region():
    # Padding grows the grid by whole cells out from the lower-left anchor, so
    # the requested resolution comes back unchanged and the padded extent still
    # covers the original study-area box (it only ever grows).
    agg = from_template("conus", resolution=1000, backend="dask")
    assert agg.attrs["res"] == (1000.0, 1000.0)
    left, bottom, right, top = _REGIONS["conus"]["bounds"]
    # half-cell pixel-center inset, then the padded edges reach at/past the box
    assert agg.x.values.min() <= left + 1000.0
    assert agg.y.values.max() >= top - 1000.0
    assert agg.x.values.max() >= right - 1000.0
    assert agg.y.values.min() <= bottom + 1000.0


@dask_array_available
def test_padding_skipped_for_single_chunk_grid():
    # A grid that stays a single chunk (nyc @ default, ~1760 x 1850) is not
    # padded -- its dask coords match the eager grid cell-for-cell, so backend
    # stays a pure execution detail for templates small enough not to tile.
    eager = from_template("nyc")
    lazy = from_template("nyc", backend="dask+numpy")
    assert lazy.data.npartitions == 1
    np.testing.assert_array_equal(lazy.x.values, eager.x.values)
    np.testing.assert_array_equal(lazy.y.values, eager.y.values)


def test_padding_skipped_for_eager_backend():
    # Padding is a dask tiling concern; the eager numpy grid keeps the exact
    # bbox-derived shape (no point bloating a materialized array).
    from xrspatial.templates import _resolve, _normalize_resolution
    spec = _resolve("conus")
    left, bottom, right, top = spec["bounds"]
    rx, ry = _normalize_resolution(1000, spec["default_resolution"])
    w = max(1, round((right - left) / rx))
    h = max(1, round((top - bottom) / ry))
    agg = from_template("conus", resolution=1000)  # numpy
    assert agg.shape == (h, w)


@dask_array_available
def test_padding_skipped_for_explicit_chunks():
    # An explicit chunks= means the caller is driving the tiling, so the shape
    # is left at the exact bbox-derived size (today's honored-verbatim contract).
    from xrspatial.templates import _resolve, _normalize_resolution
    spec = _resolve("conus")
    left, bottom, right, top = spec["bounds"]
    rx, ry = _normalize_resolution(1000, spec["default_resolution"])
    w = max(1, round((right - left) / rx))
    h = max(1, round((top - bottom) / ry))
    agg = from_template("conus", resolution=1000, chunks=512)
    assert agg.shape == (h, w)


def test_explicit_height_width_exact_shape():
    # Supplying height and width sets the grid shape exactly; the extent floats
    # off the region's lower-left anchor instead of the bbox.
    agg = from_template("conus", height=4096, width=6144)
    assert agg.shape == (4096, 6144)
    assert agg.dims == ("y", "x")
    assert agg.attrs["crs"] == 5070


def test_explicit_height_width_with_resolution_floats_extent():
    # height/width + resolution => extent = shape x resolution, anchored at the
    # region origin. Resolution is honored exactly.
    agg = from_template("conus", resolution=1000, height=4096, width=6144)
    assert agg.shape == (4096, 6144)
    assert agg.attrs["res"] == (1000.0, 1000.0)
    left, bottom, right, top = _REGIONS["conus"]["bounds"]
    # width * res out from the left anchor (pixel centers inset by half a cell)
    assert np.isclose(agg.x.values[0], left + 1000.0 / 2)
    assert np.isclose(agg.x.values[-1], left + (6144 - 0.5) * 1000.0)


def test_explicit_height_width_without_resolution_covers_bbox():
    # height/width alone => resolution is derived so the exact shape spans the
    # region bbox (the grid still covers the named study area).
    agg = from_template("conus", height=600, width=1200)
    assert agg.shape == (600, 1200)
    left, bottom, right, top = _REGIONS["conus"]["bounds"]
    res_x, res_y = agg.attrs["res"]
    assert np.isclose(res_x, (right - left) / 1200)
    assert np.isclose(res_y, (top - bottom) / 600)


@dask_array_available
def test_explicit_height_width_not_padded_on_dask():
    # The user picked the exact (tile-friendly) shape; the dask path tiles it
    # but does not pad it back out to a different size.
    import dask.array as da
    agg = from_template("conus", resolution=1000, height=4096, width=6144,
                        backend="dask")
    assert isinstance(agg.data, da.Array)
    assert agg.shape == (4096, 6144)


def test_partial_height_width_raises():
    # height and width are an all-or-nothing pair.
    with pytest.raises(ValueError, match="both height and width"):
        from_template("conus", height=4096)
    with pytest.raises(ValueError, match="both height and width"):
        from_template("conus", width=6144)


def test_non_positive_height_width_raises():
    with pytest.raises(ValueError, match="height and width must be positive"):
        from_template("conus", height=0, width=10)
    with pytest.raises(ValueError, match="height and width must be positive"):
        from_template("conus", height=10, width=-5)


def test_oversized_explicit_shape_cap_message_names_height_width():
    # The cell-cap message must point at the knob the caller actually set: on
    # the height/width path that is height/width, not the derived resolution.
    with pytest.raises(ValueError, match="exceeding") as exc:
        from_template("conus", height=30000, width=30000)  # 9e8 cells, eager
    assert "height=30000" in str(exc.value)
    assert "width=30000" in str(exc.value)
    assert "resolution" not in str(exc.value)


@dask_array_available
def test_explicit_shape_chunk_count_message_names_height_width():
    # The dask chunk-count guard on the height/width path must name the knob the
    # caller actually set -- height/width -- not the derived resolution. chunks=
    # promotes the eager default to dask and skips the cell cap, so this hits the
    # chunk-count branch (not the cell-cap branch the message-naming test above
    # covers). Only the resolution-path chunk-count message was tested before.
    with pytest.raises(ValueError, match="chunk limit") as exc:
        from_template("conus", height=2_000_000, width=2_000_000, chunks=512)
    assert "height=2000000" in str(exc.value)
    assert "width=2000000" in str(exc.value)
    assert "smaller height/width" in str(exc.value)
    assert "resolution" not in str(exc.value)


@dask_array_available
def test_explicit_height_width_with_preserve():
    # height/width compose with preserve=: the exact shape is anchored at the
    # reprojected bbox and the CRS is the chosen EPSG, resolution derived.
    agg = from_template("FRA", preserve="shape", height=300, width=400)
    assert agg.shape == (300, 400)
    assert agg.attrs["crs"] == 32630  # FRA centroid UTM 30N
    assert agg.x.attrs["units"] == "m"
    res_x, res_y = agg.attrs["res"]
    assert res_x > 0 and res_y > 0


@dask_array_available
def test_explicit_height_width_dask_ragged_shape_stays_exact():
    import dask.array as da
    # An explicit shape that is not a block multiple is left exactly as asked
    # (not padded); the dask path still tiles it into balanced blocks.
    agg = from_template("conus", resolution=1000, height=5000, width=5000,
                        backend="dask")
    assert isinstance(agg.data, da.Array)
    assert agg.shape == (5000, 5000)
    assert agg.data.npartitions > 1
    for axis in agg.data.chunks:
        assert sum(axis) == 5000
        assert max(axis) - min(axis) <= 1  # balanced, no ragged sliver
