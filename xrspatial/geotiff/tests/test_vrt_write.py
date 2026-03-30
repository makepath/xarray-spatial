"""Tests for VRT tiled output from to_geotiff."""
import numpy as np
import os
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def sample_raster():
    """200x200 float32 raster with coords and CRS."""
    arr = np.random.default_rng(55).random((200, 200), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 200)  # north-to-south
    x = np.linspace(-106.0, -105.0, 200)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326, 'nodata': -9999.0})
    return da


class TestVrtOutputNumpy:
    def test_creates_vrt_and_tiles_dir(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'out_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
        tiles_dir = str(tmp_path / 'out_1083_tiles')
        assert os.path.isdir(tiles_dir)
        tile_files = os.listdir(tiles_dir)
        assert len(tile_files) > 0
        assert all(f.endswith('.tif') for f in tile_files)

    def test_round_trip_numpy(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rt_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tile_naming_convention(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'named_1083.vrt')
        to_geotiff(sample_raster, vrt_path, tile_size=100)
        tiles_dir = str(tmp_path / 'named_1083_tiles')
        files = sorted(os.listdir(tiles_dir))
        # 200x200 with tile_size=100 -> 2x2 grid
        assert files == [
            'tile_00_00.tif', 'tile_00_01.tif',
            'tile_01_00.tif', 'tile_01_01.tif',
        ]

    def test_relative_paths_in_vrt(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rel_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        with open(vrt_path) as f:
            content = f.read()
        # Paths should be relative (no leading /)
        assert 'rel_1083_tiles/' in content
        assert str(tmp_path) not in content

    def test_compression_level_passed_to_tiles(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cl_1083.vrt')
        to_geotiff(sample_raster, vrt_path, compression='zstd',
                   compression_level=1)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)


class TestVrtOutputDask:
    def test_dask_round_trip(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'dask_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_dask_one_tile_per_chunk(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'chunks_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        tiles_dir = str(tmp_path / 'chunks_1083_tiles')
        # 200x200 chunked 100x100 -> 2x2 = 4 tiles
        assert len(os.listdir(tiles_dir)) == 4


class TestVrtEdgeCases:
    def test_cog_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cog_1083.vrt')
        with pytest.raises(ValueError, match='cog.*vrt|vrt.*cog|COG.*VRT|VRT.*COG|cog.*VRT|vrt.*COG'):
            to_geotiff(sample_raster, vrt_path, cog=True)

    def test_overview_levels_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'ovr_1083.vrt')
        with pytest.raises(ValueError, match='overview.*vrt|vrt.*overview|overview.*VRT|VRT.*overview'):
            to_geotiff(sample_raster, vrt_path, overview_levels=[2, 4])

    def test_nonempty_tiles_dir_raises(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'exist_1083_tiles'
        tiles_dir.mkdir()
        (tiles_dir / 'dummy.tif').write_text('x')
        vrt_path = str(tmp_path / 'exist_1083.vrt')
        with pytest.raises(FileExistsError):
            to_geotiff(sample_raster, vrt_path)

    def test_empty_tiles_dir_ok(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'empty_1083_tiles'
        tiles_dir.mkdir()
        vrt_path = str(tmp_path / 'empty_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
