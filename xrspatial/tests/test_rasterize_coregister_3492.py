"""``coregister`` on the ``.xrs.rasterize`` accessor (#3492).

Mirrors the interpolation accessor's ``coregister`` (#3480): a GeoDataFrame
whose CRS differs from the caller raster is reprojected into the caller CRS
before rasterizing, so the burn lands on the caller grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401  (registers the .xrs accessor)

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
Point = shapely_geometry.Point


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
# Coregister
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_coregister_matches_direct(backend):
    # Reproject the 3857 points out to 4326 and let coregister bring them
    # back; the burn must land in the same pixels as rasterizing the
    # native-CRS points directly.
    t = _template(backend, crs="EPSG:3857")
    gdf_3857 = _gdf(crs="EPSG:3857")
    gdf_4326 = gdf_3857.to_crs("EPSG:4326")
    coreg = t.xrs.rasterize(gdf_4326, column="z", coregister=True)
    direct = t.xrs.rasterize(gdf_3857, column="z")
    np.testing.assert_array_equal(_np(coreg), _np(direct))


def test_coregister_inherits_grid_and_crs():
    t = _template(crs="EPSG:3857")
    gdf_4326 = _gdf(crs="EPSG:3857").to_crs("EPSG:4326")
    out = t.xrs.rasterize(gdf_4326, column="z", coregister=True)
    assert out.shape == t.shape
    # five points burn five finite cells; the rest is the NaN fill
    assert np.count_nonzero(np.isfinite(_np(out))) == len(PZ)


def test_coregister_dataset_accessor():
    t = _template(crs="EPSG:3857")
    ds = t.to_dataset(name="band")
    gdf_4326 = _gdf(crs="EPSG:3857").to_crs("EPSG:4326")
    out = ds.xrs.rasterize(gdf_4326, column="z", coregister=True)
    direct = t.xrs.rasterize(_gdf(crs="EPSG:3857"), column="z")
    np.testing.assert_array_equal(_np(out), _np(direct))


# ---------------------------------------------------------------------------
# CRS-mismatch guard (default coregister=False)
# ---------------------------------------------------------------------------

def test_crs_mismatch_raises_without_coregister():
    t = _template(crs="EPSG:3857")
    gdf_4326 = _gdf(crs="EPSG:3857").to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        t.xrs.rasterize(gdf_4326, column="z")


def test_matching_crs_passes():
    t = _template(crs="EPSG:3857")
    out = t.xrs.rasterize(_gdf(crs="EPSG:3857"), column="z")
    assert out.shape == t.shape


def test_coregister_needs_crs_both_sides():
    t = _template(crs="EPSG:3857")
    with pytest.raises(ValueError, match="coregister=True needs a CRS"):
        t.xrs.rasterize(_gdf(crs=None), column="z", coregister=True)

    t_nocrs = _template(crs=None)
    with pytest.raises(ValueError, match="coregister=True needs a CRS"):
        t_nocrs.xrs.rasterize(_gdf(crs="EPSG:3857"), column="z",
                              coregister=True)


def test_coregister_requires_geodataframe():
    t = _template(crs="EPSG:3857")
    pairs = [(Point(a, b), z) for a, b, z in zip(PX, PY, PZ)]
    with pytest.raises(ValueError, match="requires a GeoDataFrame"):
        t.xrs.rasterize(pairs, coregister=True)
