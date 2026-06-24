"""GeoDataFrame input + ``.xrs`` accessor + coregister for interpolation (#3480)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401  (registers the .xrs accessor)
from xrspatial import idw, kde, kriging, spline
from xrspatial.tests.general_checks import cuda_and_cupy_available

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
Point = shapely_geometry.Point
LineString = shapely_geometry.LineString


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PX = np.array([1.0, 5.0, 8.0, 3.0, 6.0])
PY = np.array([2.0, 6.0, 4.0, 9.0, 1.0])
PZ = np.array([10.0, 20.0, 15.0, 5.0, 12.0])


def _template(backend="numpy", crs="EPSG:3857"):
    xs = np.linspace(0.0, 10.0, 12)
    ys = np.linspace(10.0, 0.0, 11)
    data = np.zeros((11, 12), dtype=np.float64)
    if backend == "dask":
        import dask.array as da
        data = da.from_array(data, chunks=(6, 6))
    t = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=["y", "x"])
    if crs is not None:
        t.attrs["crs"] = crs
    return t


def _gdf(crs="EPSG:3857", value_col="z"):
    geom = [Point(a, b) for a, b in zip(PX, PY)]
    return gpd.GeoDataFrame({value_col: PZ}, geometry=geom, crs=crs)


def _np(arr):
    data = arr.data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# 1. GeoDataFrame parity with the array form (standalone functions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_idw_geodataframe_parity(backend):
    t = _template(backend)
    arr = idw(PX, PY, PZ, t)
    gdf = idw(_gdf(), template=t, column="z")
    np.testing.assert_allclose(_np(arr), _np(gdf), equal_nan=True)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_spline_geodataframe_parity(backend):
    t = _template(backend)
    arr = spline(PX, PY, PZ, t)
    gdf = spline(_gdf(), template=t, column="z")
    np.testing.assert_allclose(_np(arr), _np(gdf), equal_nan=True)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_kriging_geodataframe_parity(backend):
    t = _template(backend)
    arr = kriging(PX, PY, PZ, t)
    gdf = kriging(_gdf(), template=t, column="z")
    np.testing.assert_allclose(_np(arr), _np(gdf), equal_nan=True)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_kde_geodataframe_parity(backend):
    t = _template(backend)
    # uniform-weight density
    arr = kde(PX, PY, template=t)
    gdf = kde(_gdf(), template=t)
    np.testing.assert_allclose(_np(arr), _np(gdf), equal_nan=True)
    # column maps to per-point weights
    arr_w = kde(PX, PY, weights=PZ, template=t)
    gdf_w = kde(_gdf(), template=t, column="z")
    np.testing.assert_allclose(_np(arr_w), _np(gdf_w), equal_nan=True)


def test_column_autopick_first_numeric():
    t = _template()
    gdf = _gdf(value_col="elevation")
    auto = kriging(gdf, template=t)
    named = kriging(gdf, template=t, column="elevation")
    np.testing.assert_allclose(_np(auto), _np(named), equal_nan=True)


# ---------------------------------------------------------------------------
# 2. Accessor wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["idw", "spline", "kriging"])
def test_accessor_matches_standalone(name):
    t = _template()
    func = globals()[name]
    direct = func(_gdf(), template=t, column="z")
    via = getattr(t.xrs, name)(_gdf(), column="z")
    np.testing.assert_allclose(_np(direct), _np(via), equal_nan=True)
    assert via.attrs.get("crs") == "EPSG:3857"
    assert via.shape == t.shape


def test_accessor_kde():
    t = _template()
    via = t.xrs.kde(_gdf())
    np.testing.assert_allclose(_np(via), _np(kde(_gdf(), template=t)),
                               equal_nan=True)


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_accessor_inherits_chunks(backend):
    t = _template(backend)
    out = t.xrs.idw(_gdf(), column="z")
    if backend == "dask":
        assert out.data.chunksize == (6, 6)
    else:
        assert not hasattr(out.data, "chunks")


def test_accessor_array_form_backward_compatible():
    # the legacy x/y/z array signature on the accessor still works
    t = _template()
    out = t.xrs.idw(PX, PY, PZ)
    np.testing.assert_allclose(_np(out), _np(idw(PX, PY, PZ, t)),
                               equal_nan=True)
    out_k = t.xrs.kde(PX, PY)
    np.testing.assert_allclose(_np(out_k), _np(kde(PX, PY, template=t)),
                               equal_nan=True)


@cuda_and_cupy_available
def test_geodataframe_parity_cupy():
    # the GeoDataFrame branch runs before backend dispatch, so a cupy
    # template should match numpy. One smoke test documents that.
    import cupy
    t_np = _template("numpy")
    t_cp = t_np.copy()
    t_cp.data = cupy.asarray(t_cp.data)
    out_np = idw(_gdf(), template=t_np, column="z")
    out_cp = idw(_gdf(), template=t_cp, column="z")
    np.testing.assert_allclose(_np(out_np), out_cp.data.get(),
                               equal_nan=True)


def test_accessor_dataset():
    t = _template()
    ds = t.to_dataset(name="band")
    out = ds.xrs.idw(_gdf(), column="z")
    np.testing.assert_allclose(_np(out), _np(t.xrs.idw(_gdf(), column="z")),
                               equal_nan=True)


# ---------------------------------------------------------------------------
# 3. Coregister
# ---------------------------------------------------------------------------

def test_coregister_matches_manual_reprojection():
    t = _template(crs="EPSG:3857")
    gdf_3857 = _gdf(crs="EPSG:3857")
    gdf_4326 = gdf_3857.to_crs("EPSG:4326")
    coreg = t.xrs.idw(gdf_4326, column="z", coregister=True)
    direct = t.xrs.idw(gdf_3857, column="z")
    np.testing.assert_allclose(_np(coreg), _np(direct), equal_nan=True,
                               atol=1e-6)


def test_coregister_kde_headline_call():
    t = _template(crs="EPSG:3857")
    gdf_4326 = _gdf(crs="EPSG:3857").to_crs("EPSG:4326")
    out = t.xrs.kde(gdf_4326, coregister=True)
    assert out.shape == t.shape
    assert np.isfinite(_np(out)).all()
    assert out.attrs.get("crs") == "EPSG:3857"


# ---------------------------------------------------------------------------
# 4. CRS-mismatch guard
# ---------------------------------------------------------------------------

def test_crs_mismatch_raises_without_coregister():
    t = _template(crs="EPSG:3857")
    gdf_4326 = _gdf(crs="EPSG:3857").to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        t.xrs.idw(gdf_4326, column="z")


def test_matching_crs_passes():
    t = _template(crs="EPSG:3857")
    out = t.xrs.idw(_gdf(crs="EPSG:3857"), column="z")
    assert out.shape == t.shape


def test_missing_crs_either_side_passes():
    # caller has no CRS -> no-op check, like rasterize
    t = _template(crs=None)
    out = t.xrs.idw(_gdf(crs="EPSG:3857"), column="z")
    assert out.shape == t.shape


def test_coregister_needs_crs_both_sides():
    t = _template(crs="EPSG:3857")
    with pytest.raises(ValueError, match="coregister=True needs a CRS"):
        t.xrs.idw(_gdf(crs=None), column="z", coregister=True)


def test_coregister_requires_geodataframe():
    t = _template(crs="EPSG:3857")
    with pytest.raises(ValueError, match="requires a GeoDataFrame"):
        t.xrs.kde(PX, y=PY, coregister=True)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_non_point_geometry_raises():
    t = _template()
    gdf = gpd.GeoDataFrame(
        {"z": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="Point geometries"):
        idw(gdf, template=t, column="z")


def test_missing_column_no_numeric_raises():
    t = _template()
    gdf = gpd.GeoDataFrame(
        {"label": ["a", "b", "c", "d", "e"]},
        geometry=[Point(a, b) for a, b in zip(PX, PY)], crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="no numeric column"):
        kriging(gdf, template=t)


def test_unknown_column_raises():
    t = _template()
    with pytest.raises(ValueError, match="not in the GeoDataFrame"):
        idw(_gdf(), template=t, column="nope")


def test_non_numeric_column_raises():
    t = _template()
    gdf = gpd.GeoDataFrame(
        {"label": ["a", "b", "c", "d", "e"]},
        geometry=[Point(a, b) for a, b in zip(PX, PY)], crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="not numeric"):
        idw(gdf, template=t, column="label")


def test_positional_yz_with_geodataframe_raises():
    t = _template()
    with pytest.raises(ValueError, match="pass template and column as keywords"):
        idw(_gdf(), PY, PZ, t)


def test_kde_weights_and_column_conflict_raises():
    t = _template()
    with pytest.raises(ValueError, match="either 'weights' or 'column'"):
        kde(_gdf(), template=t, column="z", weights=PZ)


def test_column_without_geodataframe_raises():
    t = _template()
    with pytest.raises(ValueError, match="only valid"):
        idw(PX, PY, PZ, t, column="z")


def test_kde_array_requires_y():
    t = _template()
    with pytest.raises(ValueError, match="y is required"):
        kde(PX, template=t)
