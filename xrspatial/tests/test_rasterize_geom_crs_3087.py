"""Issue #3087: rasterize() must carry the GeoDataFrame CRS onto the
output when no ``like`` template supplies one.

Before the fix, ``_parse_input`` extracted ``gdf.crs`` but it was only
consumed by the #3058 mismatch check.  On the no-``like`` path the
output DataArray had no ``crs`` attr, no ``crs_wkt``, and no
``spatial_ref`` coord, so ``to_geotiff`` wrote an un-georeferenced file
and ``polygonize(rasterize(gdf))`` returned ``crs=None``.

The fix emits the geometry CRS only when the output would otherwise
carry none: ``attrs['crs']`` as an EPSG int when one resolves, else
``attrs['crs_wkt']`` as WKT (the geotiff attrs convention).  A template
CRS always wins.
"""
import numpy as np
import pytest
import xarray as xr

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_geopandas, reason="geopandas not installed"),
]

skip_no_dask = pytest.mark.skipif(not has_dask, reason="dask not installed")
skip_no_cuda = pytest.mark.skipif(not has_cuda, reason="CUDA not available")

# A CRS with no EPSG code: a custom sphere-based equal-area projection.
_NO_EPSG_PROJ = (
    "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 "
    "+a=6370997 +b=6370997 +units=m +no_defs"
)


def _gdf(crs=None):
    return gpd.GeoDataFrame(
        {'value': [1.0]}, geometry=[box(2, 2, 6, 6)], crs=crs,
    )


def _make_like(crs_attr=None, spatial_ref_wkt=None, width=10, height=10):
    x = np.linspace(0.5, width - 0.5, width)
    y = np.linspace(height - 0.5, 0.5, height)
    da_ = xr.DataArray(
        np.zeros((height, width)), dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )
    if crs_attr is not None:
        da_.attrs['crs'] = crs_attr
    if spatial_ref_wkt is not None:
        da_ = da_.assign_coords(spatial_ref=0)
        da_['spatial_ref'].attrs['crs_wkt'] = spatial_ref_wkt
    return da_


class TestGeomCrsPropagation:
    def test_epsg_crs_emitted_without_like(self):
        result = rasterize(
            _gdf(crs='EPSG:32610'), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0,
        )
        assert result.attrs.get('crs') == 32610
        assert 'crs_wkt' not in result.attrs

    def test_non_epsg_crs_emitted_as_wkt(self):
        result = rasterize(
            _gdf(crs=_NO_EPSG_PROJ), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0,
        )
        assert 'crs' not in result.attrs
        wkt = result.attrs.get('crs_wkt')
        assert isinstance(wkt, str)
        # Round-trips through pyproj to the same CRS.
        from pyproj import CRS
        assert CRS(wkt).equals(CRS(_NO_EPSG_PROJ))

    def test_crs_less_gdf_emits_nothing(self):
        result = rasterize(
            _gdf(crs=None), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0,
        )
        assert 'crs' not in result.attrs
        assert 'crs_wkt' not in result.attrs

    def test_geometry_value_pairs_unchanged(self):
        result = rasterize(
            [(box(2, 2, 6, 6), 1.0)], width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0,
        )
        assert 'crs' not in result.attrs
        assert 'crs_wkt' not in result.attrs

    def test_like_crs_attr_wins(self):
        like = _make_like(crs_attr='EPSG:32610')
        result = rasterize(
            _gdf(crs='EPSG:32610'), like=like, column='value', fill=0,
        )
        # Template value kept verbatim, not replaced by the EPSG int.
        assert result.attrs.get('crs') == 'EPSG:32610'
        assert 'crs_wkt' not in result.attrs

    def test_like_spatial_ref_coord_wins(self):
        from pyproj import CRS
        like = _make_like(spatial_ref_wkt=CRS('EPSG:32610').to_wkt())
        result = rasterize(
            _gdf(crs='EPSG:32610'), like=like, column='value', fill=0,
        )
        # spatial_ref carries the georeference; attrs stay untouched.
        assert 'crs' not in result.attrs
        assert 'crs_wkt' not in result.attrs
        assert 'spatial_ref' in result.coords

    def test_like_without_crs_gets_geometry_crs(self):
        like = _make_like()
        result = rasterize(
            _gdf(crs='EPSG:32610'), like=like, column='value', fill=0,
        )
        assert result.attrs.get('crs') == 32610

    def test_polygonize_round_trip_keeps_crs(self):
        from xrspatial.polygonize import polygonize
        result = rasterize(
            _gdf(crs='EPSG:32610'), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0,
        )
        gdf_back = polygonize(result, return_type='geopandas')
        assert gdf_back.crs is not None
        from pyproj import CRS
        assert CRS(gdf_back.crs).to_epsg() == 32610

    @skip_no_dask
    def test_dask_numpy_backend(self):
        result = rasterize(
            _gdf(crs='EPSG:32610'), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0, chunks=5,
        )
        assert result.attrs.get('crs') == 32610

    @skip_no_cuda
    def test_cupy_backend(self):
        result = rasterize(
            _gdf(crs='EPSG:32610'), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0, use_cuda=True,
        )
        assert result.attrs.get('crs') == 32610

    @skip_no_cuda
    @skip_no_dask
    def test_dask_cupy_backend(self):
        result = rasterize(
            _gdf(crs='EPSG:32610'), width=10, height=10,
            bounds=(0, 0, 10, 10), column='value', fill=0,
            use_cuda=True, chunks=5,
        )
        assert result.attrs.get('crs') == 32610
