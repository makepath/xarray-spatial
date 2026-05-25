"""Regression tests for issue #1606.

``to_geotiff(da, 'out.vrt')`` (which dispatches to ``_write_vrt_tiled``)
used to drop a chunk of metadata that ``to_geotiff(da, 'out.tif')``
preserved:

* ``attrs['nodatavals']`` / ``attrs['_FillValue']`` -- the VRT path
  read ``attrs['nodata']`` directly instead of going through
  ``_resolve_nodata_attr``.
* ``attrs['gdal_metadata']`` / ``attrs['gdal_metadata_xml']``
* ``attrs['extra_tags']`` and the friendly tag attrs folded in by
  ``_merge_friendly_extra_tags``
* ``attrs['x_resolution']`` / ``attrs['y_resolution']`` /
  ``attrs['resolution_unit']``
* ``attrs['raster_type']``

Each tile under the VRT now carries the same rich tag set the
equivalent single-file ``.tif`` write would emit.
"""
import glob
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _make_rioxarray_style(arr=None):
    """DataArray that looks like rioxarray output: nodata only via aliases."""
    if arr is None:
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(arr.shape[0], dtype=np.float64),
                'x': np.arange(arr.shape[1], dtype=np.float64)},
        attrs={
            # No bare 'nodata' key -- forces _resolve_nodata_attr to walk
            # the alias chain.
            'nodatavals': (-9999.0,),
            '_FillValue': -9999.0,
            'crs': 4326,
            'gdal_metadata': {'AREA_OR_POINT': 'Area', 'foo': 'bar'},
            'x_resolution': 96,
            'y_resolution': 96,
            'resolution_unit': 'inch',
            'raster_type': 'point',
        },
    )


def _first_tile_path(vrt_path):
    tiles_dir = vrt_path[:-len('.vrt')] + '_tiles'
    tiles = sorted(glob.glob(os.path.join(tiles_dir, '*.tif')))
    assert tiles, f'no per-tile .tif files under {tiles_dir}'
    return tiles[0]


class TestVrtTiledMetadataParity:
    def test_nodatavals_alias_propagates_to_tiles(self, tmp_path):
        da = _make_rioxarray_style()
        vrt = str(tmp_path / 'nodatavals.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        # Before the fix this was None: _write_vrt_tiled read
        # attrs['nodata'] directly and ignored the nodatavals alias.
        assert tile_da.attrs.get('nodata') == -9999.0

    def test_fill_value_alias_propagates_to_tiles(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
        da = xr.DataArray(
            arr, dims=('y', 'x'),
            coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
            attrs={'_FillValue': -9999.0, 'crs': 4326},
        )
        vrt = str(tmp_path / 'fillvalue.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        assert tile_da.attrs.get('nodata') == -9999.0

    def test_gdal_metadata_propagates_to_tiles(self, tmp_path):
        da = _make_rioxarray_style()
        vrt = str(tmp_path / 'gdal_meta.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        gm = tile_da.attrs.get('gdal_metadata')
        assert gm == {'AREA_OR_POINT': 'Area', 'foo': 'bar'}

    def test_resolution_tags_propagate_to_tiles(self, tmp_path):
        da = _make_rioxarray_style()
        vrt = str(tmp_path / 'resolution.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        assert tile_da.attrs.get('x_resolution') == 96.0
        assert tile_da.attrs.get('y_resolution') == 96.0
        assert tile_da.attrs.get('resolution_unit') == 'inch'

    def test_raster_type_point_propagates_to_tiles(self, tmp_path):
        da = _make_rioxarray_style()
        vrt = str(tmp_path / 'point.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        assert tile_da.attrs.get('raster_type') == 'point'

    def test_tif_vs_vrt_tile_metadata_parity(self, tmp_path):
        """Same DataArray, two destinations -- per-tile metadata matches."""
        da = _make_rioxarray_style()
        tif_path = str(tmp_path / 'parity.tif')
        vrt_path = str(tmp_path / 'parity.vrt')
        to_geotiff(da, tif_path, tile_size=16)
        to_geotiff(da, vrt_path, tile_size=16)

        tif_da = open_geotiff(tif_path)
        tile_da = open_geotiff(_first_tile_path(vrt_path))

        keys = ('nodata', 'gdal_metadata', 'raster_type',
                'x_resolution', 'y_resolution', 'resolution_unit')
        for k in keys:
            assert tif_da.attrs.get(k) == tile_da.attrs.get(k), (
                f'{k} mismatch: tif={tif_da.attrs.get(k)!r}, '
                f'vrt-tile={tile_da.attrs.get(k)!r}')


class TestVrtTiledRichTagCoverage:
    """Cover the XML / extra_tags / friendly-tag paths the bare
    ``gdal_metadata`` dict assertion above does not exercise."""

    def test_gdal_metadata_xml_string_propagates_to_tiles(self, tmp_path):
        """``attrs['gdal_metadata_xml']`` (pre-built XML string) bypasses
        the dict->XML builder. Verify it still reaches per-tile files."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        xml = (
            '<GDALMetadata>\n'
            '  <Item name="VRT_XML_KEY">vrt_xml_value</Item>\n'
            '</GDALMetadata>\n'
        )
        da = xr.DataArray(
            arr, dims=('y', 'x'),
            coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
            attrs={'crs': 4326, 'gdal_metadata_xml': xml},
        )
        vrt = str(tmp_path / 'gdal_xml.vrt')
        # Rich-tag write surface (PR 4 of epic #2340).
        to_geotiff(da, vrt, tile_size=16,
                   allow_experimental_codecs=True)
        tile_da = open_geotiff(_first_tile_path(vrt))
        # On read, the XML is re-parsed into a dict under
        # attrs['gdal_metadata']; the raw XML lands under
        # attrs['gdal_metadata_xml']. Assert the item shows up in
        # whichever surface the reader emits.
        gm = tile_da.attrs.get('gdal_metadata') or {}
        gm_xml = tile_da.attrs.get('gdal_metadata_xml') or ''
        assert (
            gm.get('VRT_XML_KEY') == 'vrt_xml_value'
            or 'VRT_XML_KEY' in gm_xml
        ), (
            f'gdal_metadata_xml content lost on VRT-tile round-trip; '
            f'gdal_metadata={gm!r}, gdal_metadata_xml={gm_xml!r}'
        )

    def test_extra_tags_entry_propagates_to_tiles(self, tmp_path):
        """A user-supplied ``extra_tags`` entry (Software, tag 305)
        must round-trip through the VRT-tiled writer."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        software = 'xrspatial-vrt-test-1606'
        da = xr.DataArray(
            arr, dims=('y', 'x'),
            coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
            attrs={
                'crs': 4326,
                # (tag_id, type_id, count, value); type 2 = ASCII
                'extra_tags': [(305, 2, len(software) + 1, software)],
            },
        )
        vrt = str(tmp_path / 'extra_tags.vrt')
        # Rich-tag write surface (PR 4 of epic #2340).
        to_geotiff(da, vrt, tile_size=16,
                   allow_experimental_codecs=True)
        tile_da = open_geotiff(_first_tile_path(vrt))
        et = tile_da.attrs.get('extra_tags') or []
        tag_ids = {entry[0] for entry in et}
        assert 305 in tag_ids, (
            f'Software (305) tag missing from VRT tile extra_tags; '
            f'got tag ids {sorted(tag_ids)!r}'
        )

    def test_image_description_friendly_attr_propagates_to_tiles(
            self, tmp_path):
        """``attrs['image_description']`` is folded into ``extra_tags``
        as tag 270 by ``_merge_friendly_extra_tags`` and then surfaces
        on read as ``attrs['image_description']``."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=('y', 'x'),
            coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
            attrs={'crs': 4326,
                   'image_description': 'vrt-tile-friendly-1606'},
        )
        vrt = str(tmp_path / 'image_desc.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        assert (tile_da.attrs.get('image_description')
                == 'vrt-tile-friendly-1606')


class TestVrtTiledMetadataDask:
    def test_nodatavals_alias_dask(self, tmp_path):
        pytest.importorskip('dask.array')
        import dask.array as dska
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = -9999.0
        da_np = xr.DataArray(
            arr, dims=('y', 'x'),
            coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
            attrs={'nodatavals': (-9999.0,), 'crs': 4326,
                   'gdal_metadata': {'k': 'v'}},
        )
        # Dask-back the data so _write_vrt_tiled takes the dask branch.
        da = xr.DataArray(
            dska.from_array(arr, chunks=4),
            dims=da_np.dims, coords=da_np.coords, attrs=da_np.attrs,
        )
        vrt = str(tmp_path / 'dask.vrt')
        to_geotiff(da, vrt, tile_size=16)
        tile_da = open_geotiff(_first_tile_path(vrt))
        assert tile_da.attrs.get('nodata') == -9999.0
        assert tile_da.attrs.get('gdal_metadata') == {'k': 'v'}
