"""Issue #3058: rasterize() must not silently burn geometries onto a
``like`` template in a different CRS.

Before the fix, ``_parse_input`` discarded ``gdf.crs`` and the output
inherited the template CRS (via ``attrs['crs']`` / the ``spatial_ref``
coord) with no reprojection, producing an authoritative-looking but
wrong raster.  ``check_crs=True`` (default) now compares both sides and
raises ``ValueError`` on mismatch; ``check_crs=False`` opts out.
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


def _make_like(crs_attr=None, crs_wkt=None, spatial_ref_wkt=None,
               width=10, height=10):
    """2D template with georeferenced coords, optionally CRS-tagged.

    ``crs_attr`` sets ``attrs['crs']``; ``crs_wkt`` sets
    ``attrs['crs_wkt']``; ``spatial_ref_wkt`` attaches a rioxarray-style
    ``spatial_ref`` non-dim coord carrying ``crs_wkt``.
    """
    x = np.linspace(0.5, width - 0.5, width)
    y = np.linspace(height - 0.5, 0.5, height)
    da_ = xr.DataArray(
        np.zeros((height, width)), dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )
    if crs_attr is not None:
        da_.attrs['crs'] = crs_attr
    if crs_wkt is not None:
        da_.attrs['crs_wkt'] = crs_wkt
    if spatial_ref_wkt is not None:
        da_ = da_.assign_coords(spatial_ref=0)
        da_['spatial_ref'].attrs['crs_wkt'] = spatial_ref_wkt
    return da_


def _gdf(crs=None):
    return gpd.GeoDataFrame(
        {'value': [1.0]}, geometry=[box(2, 2, 6, 6)], crs=crs,
    )


class TestCrsMismatch:
    def test_matching_crs_attr_ok(self):
        result = rasterize(
            _gdf(crs='EPSG:4326'),
            like=_make_like(crs_attr='EPSG:4326'),
            column='value',
        )
        assert result.attrs.get('crs') == 'EPSG:4326'

    def test_mismatch_crs_attr_raises(self):
        with pytest.raises(ValueError, match='CRS mismatch'):
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(crs_attr='EPSG:3857'),
                column='value',
            )

    def test_mismatch_message_is_compact(self):
        # The error labels each side with a short "EPSG:xxxx" rather than
        # the multi-line pyproj CRS repr.
        with pytest.raises(ValueError) as exc:
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(crs_attr='EPSG:3857'),
                column='value',
            )
        msg = str(exc.value)
        assert 'EPSG:4326' in msg
        assert 'EPSG:3857' in msg
        assert '\n' not in msg

    def test_check_crs_false_bypasses(self):
        # Opt out: the (wrong) template CRS still propagates, no raise.
        result = rasterize(
            _gdf(crs='EPSG:4326'),
            like=_make_like(crs_attr='EPSG:3857'),
            column='value',
            check_crs=False,
        )
        assert result.attrs.get('crs') == 'EPSG:3857'

    def test_int_vs_string_epsg_equivalent(self):
        # gdf.crs as int 4326 vs template "EPSG:4326" string must match.
        result = rasterize(
            _gdf(crs=4326),
            like=_make_like(crs_attr='EPSG:4326'),
            column='value',
        )
        assert result is not None

    def test_crs_wkt_attr_used(self):
        from pyproj import CRS
        wkt = CRS('EPSG:3857').to_wkt()
        with pytest.raises(ValueError, match='CRS mismatch'):
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(crs_wkt=wkt),
                column='value',
            )

    def test_spatial_ref_coord_used(self):
        from pyproj import CRS
        wkt = CRS('EPSG:3857').to_wkt()
        with pytest.raises(ValueError, match='CRS mismatch'):
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(spatial_ref_wkt=wkt),
                column='value',
            )

    def test_no_geometry_crs_is_noop(self):
        # GeoDataFrame without a CRS: nothing to compare, no raise.
        result = rasterize(
            _gdf(crs=None),
            like=_make_like(crs_attr='EPSG:3857'),
            column='value',
        )
        assert result.attrs.get('crs') == 'EPSG:3857'

    def test_no_like_crs_is_noop(self):
        # Template carries no CRS: nothing to compare, no raise.
        result = rasterize(
            _gdf(crs='EPSG:4326'),
            like=_make_like(),
            column='value',
        )
        assert result is not None

    def test_no_like_at_all_is_noop(self):
        # No template -> no CRS comparison path is taken.
        result = rasterize(
            _gdf(crs='EPSG:4326'), width=10, height=10, column='value',
        )
        assert result is not None

    def test_iterable_input_has_no_crs(self):
        # (geometry, value) iterable exposes no CRS, so the check is a
        # no-op even against a CRS-tagged template.
        result = rasterize(
            [(box(2, 2, 6, 6), 1.0)],
            like=_make_like(crs_attr='EPSG:3857'),
        )
        assert result.attrs.get('crs') == 'EPSG:3857'

    def test_invalid_crs_value_surfaces(self):
        # An unparseable CRS on either side must raise rather than
        # silently disable the guard.  Exercise the comparison helper
        # directly with a malformed template CRS value.
        from xrspatial.rasterize import _check_crs_match
        with pytest.raises(ValueError, match='not a valid CRS'):
            _check_crs_match('EPSG:4326', 'not-a-real-crs')

    def test_check_skipped_when_either_side_none(self):
        # The helper is a no-op when either side lacks a CRS.
        from xrspatial.rasterize import _check_crs_match
        _check_crs_match(None, 'EPSG:3857')
        _check_crs_match('EPSG:4326', None)
        _check_crs_match(None, None)


class TestCrsMismatchBackends:
    """The CRS guard runs before backend dispatch, so it must fire the
    same way for dask and cupy outputs.
    """

    @pytest.mark.skipif(not has_dask, reason="dask not installed")
    def test_mismatch_raises_dask(self):
        with pytest.raises(ValueError, match='CRS mismatch'):
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(crs_attr='EPSG:3857'),
                column='value',
                chunks=5,
            )

    @pytest.mark.skipif(not has_dask, reason="dask not installed")
    def test_match_ok_dask(self):
        result = rasterize(
            _gdf(crs='EPSG:4326'),
            like=_make_like(crs_attr='EPSG:4326'),
            column='value',
            chunks=5,
        )
        assert result.chunks is not None
        assert result.attrs.get('crs') == 'EPSG:4326'

    @pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
    def test_mismatch_raises_cupy(self):
        with pytest.raises(ValueError, match='CRS mismatch'):
            rasterize(
                _gdf(crs='EPSG:4326'),
                like=_make_like(crs_attr='EPSG:3857'),
                column='value',
                gpu=True,
            )

    @pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
    def test_match_ok_cupy(self):
        result = rasterize(
            _gdf(crs='EPSG:4326'),
            like=_make_like(crs_attr='EPSG:4326'),
            column='value',
            gpu=True,
        )
        assert result.attrs.get('crs') == 'EPSG:4326'
