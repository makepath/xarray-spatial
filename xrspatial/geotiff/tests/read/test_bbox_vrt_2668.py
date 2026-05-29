"""Tests for ``bbox=`` on ``open_geotiff`` with ``.vrt`` sources (#2668).

The dispatcher resolves ``bbox`` to a pixel ``window`` before dispatching
to a backend. The TIFF resolver (``_bbox_to_window``) reads a TIFF header
and cannot parse a VRT's XML, so VRT sources route through
``_vrt_bbox_to_window`` instead. Both resolvers share the bbox validation
and extent math in ``_geo_info_to_window``. These tests confirm a VRT
``bbox=`` read returns the correctly windowed mosaic and that the shared
validation still rejects malformed / non-georeferenced / rotated inputs.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f'bbox_vrt_2668_{label}_{uuid.uuid4().hex[:8]}'
    d.mkdir()
    return str(d)


@pytest.fixture
def georef_vrt_2668(tmp_path):
    """Build a 256x256 EPSG:4326 georeferenced VRT mosaic.

    Uses ``to_geotiff(..., tile_size=128)`` which writes a tiled mosaic
    plus a ``.vrt`` that wraps the tiles, so the VRT carries a real
    axis-aligned GeoTransform. Spans -106..-105 lon, 40..41 lat
    (north-up). Returns ``(vrt_path, full_array)``.
    """
    d = _unique_dir(tmp_path, 'fixture')
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    ys = np.linspace(41.0, 40.0, 256)  # decreasing -> north-up
    xs = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': ys, 'x': xs},
        attrs={'crs': 4326},
    )
    vrt_path = os.path.join(d, 'mosaic.vrt')
    to_geotiff(raster, vrt_path, tile_size=128)
    return vrt_path, arr


class TestVrtBboxResolution:
    def test_bbox_returns_windowed_mosaic(self, georef_vrt_2668):
        """``bbox=`` on a VRT returns a non-empty sub-region (the bug
        raised ``Invalid TIFF byte order marker`` before this fix)."""
        vrt_path, _ = georef_vrt_2668
        full = open_geotiff(vrt_path)
        bb = (-105.8, 40.2, -105.2, 40.8)
        sub = open_geotiff(vrt_path, bbox=bb)
        assert 0 < sub.shape[0] < full.shape[0]
        assert 0 < sub.shape[1] < full.shape[1]

    def test_bbox_matches_equivalent_pixel_window(self, georef_vrt_2668):
        """``bbox=`` and the pixel window it resolves to produce the same
        array. Resolved via the internal helper so the test doesn't redo
        the affine math. Asserts shape and pixel values."""
        from xrspatial.geotiff import _vrt_bbox_to_window
        vrt_path, _ = georef_vrt_2668
        bb = (-105.7, 40.3, -105.3, 40.7)
        pixel_window = _vrt_bbox_to_window(vrt_path, bb)
        a = open_geotiff(vrt_path, bbox=bb)
        b = open_geotiff(vrt_path, window=pixel_window)
        assert a.shape == b.shape
        np.testing.assert_array_equal(a.values, b.values)
        full = open_geotiff(vrt_path)
        r0, c0, r1, c1 = pixel_window
        np.testing.assert_array_equal(a.values, full.values[r0:r1, c0:c1])

    def test_full_bbox_matches_full_read(self, georef_vrt_2668):
        """A bbox padded outside the extent clamps to file bounds, so the
        result matches a no-window read."""
        vrt_path, _ = georef_vrt_2668
        full = open_geotiff(vrt_path)
        bb = (-106.5, 39.5, -104.5, 41.5)
        sub = open_geotiff(vrt_path, bbox=bb)
        assert sub.shape == full.shape
        np.testing.assert_array_equal(sub.values, full.values)

    def test_bbox_chunked_matches_eager(self, georef_vrt_2668):
        """The resolved window forwards to the chunked VRT backend; the
        computed dask result matches the eager bbox read."""
        import dask.array as da
        vrt_path, _ = georef_vrt_2668
        bb = (-105.8, 40.2, -105.2, 40.8)
        eager = open_geotiff(vrt_path, bbox=bb)
        chunked = open_geotiff(vrt_path, bbox=bb, chunks=64)
        assert isinstance(chunked.data, da.Array)
        np.testing.assert_array_equal(chunked.compute().values, eager.values)


class TestVrtBboxValidation:
    def test_bbox_and_window_are_mutually_exclusive(self, georef_vrt_2668):
        vrt_path, _ = georef_vrt_2668
        with pytest.raises(ValueError, match="mutually exclusive"):
            open_geotiff(
                vrt_path,
                window=(0, 0, 10, 10),
                bbox=(-105.8, 40.2, -105.2, 40.8),
            )

    def test_bbox_wrong_arity_rejected(self, georef_vrt_2668):
        vrt_path, _ = georef_vrt_2668
        with pytest.raises(ValueError, match="4-tuple"):
            open_geotiff(vrt_path, bbox=(-105.8, 40.2, -105.2))

    def test_bbox_inverted_rejected(self, georef_vrt_2668):
        vrt_path, _ = georef_vrt_2668
        with pytest.raises(ValueError, match="non-positive size"):
            open_geotiff(vrt_path, bbox=(-105.2, 40.2, -105.8, 40.8))

    def test_bbox_rejects_nan(self, georef_vrt_2668):
        import math
        vrt_path, _ = georef_vrt_2668
        with pytest.raises(ValueError, match="finite coordinates"):
            open_geotiff(vrt_path, bbox=(math.nan, 40.2, -105.2, 40.8))

    def test_bbox_outside_extent_rejected(self, georef_vrt_2668):
        """A bbox that does not overlap the VRT extent raises rather than
        returning an empty array."""
        vrt_path, _ = georef_vrt_2668
        with pytest.raises(ValueError, match="does not overlap"):
            open_geotiff(vrt_path, bbox=(10.0, 10.0, 20.0, 20.0))

    def test_bbox_requires_georef(self, tmp_path):
        """A VRT with no ``<GeoTransform>`` is rejected with the same
        no-georef error the TIFF path raises."""
        d = _unique_dir(tmp_path, 'nogeoref')
        # Source tile so the VRT references a readable file, but the VRT
        # itself declares no GeoTransform.
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        src_da = xr.DataArray(src, dims=('y', 'x'))
        tif = os.path.join(d, 'tile.tif')
        to_geotiff(src_da, tif, compression='none')
        vrt = os.path.join(d, 'nogeoref.vrt')
        with open(vrt, 'w') as f:
            f.write(
                '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
                '  <VRTRasterBand dataType="Float32" band="1">\n'
                '    <SimpleSource>\n'
                '      <SourceFilename relativeToVRT="1">tile.tif'
                '</SourceFilename>\n'
                '      <SourceBand>1</SourceBand>\n'
                '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
                '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
                '    </SimpleSource>\n'
                '  </VRTRasterBand>\n'
                '</VRTDataset>\n'
            )
        with pytest.raises(ValueError, match="no georeferencing"):
            open_geotiff(vrt, bbox=(-1.0, -1.0, 1.0, 1.0))

    def test_bbox_rejects_rotated_affine(self, tmp_path):
        """A VRT with a rotated (sheared) GeoTransform is rejected with
        the rotated-affine guidance, mirroring the TIFF path."""
        d = _unique_dir(tmp_path, 'rotated')
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        src_da = xr.DataArray(src, dims=('y', 'x'))
        tif = os.path.join(d, 'tile.tif')
        to_geotiff(src_da, tif, compression='none')
        vrt = os.path.join(d, 'rotated.vrt')
        # Non-zero skew terms (positions 2 and 4) make this rotated.
        with open(vrt, 'w') as f:
            f.write(
                '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
                '  <GeoTransform>0, 1, 0.5, 0, 0.5, -1</GeoTransform>\n'
                '  <VRTRasterBand dataType="Float32" band="1">\n'
                '    <SimpleSource>\n'
                '      <SourceFilename relativeToVRT="1">tile.tif'
                '</SourceFilename>\n'
                '      <SourceBand>1</SourceBand>\n'
                '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
                '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
                '    </SimpleSource>\n'
                '  </VRTRasterBand>\n'
                '</VRTDataset>\n'
            )
        with pytest.raises(ValueError, match="rotated affine"):
            open_geotiff(vrt, bbox=(0.0, 0.0, 3.0, 3.0), allow_rotated=True)
