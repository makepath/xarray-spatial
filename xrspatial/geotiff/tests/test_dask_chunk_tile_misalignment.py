"""``read_geotiff_dask`` chunk boundaries misaligned with TIFF tile size.

``read_geotiff_dask`` builds chunks of size ``chunks`` (default 512)
regardless of the underlying file's ``TileWidth``/``TileLength`` tags.
When the requested chunk size does not align with the on-disk tile
grid, the per-window reader must re-tile its decoded tile buffer into
the requested window before returning it to dask. Existing dask tests
all use chunk sizes that line up with the tile boundary; this module
covers the misaligned case so a regression in the windowed re-tile
path (off-by-one cropping, wrong row stride at a tile-spanning chunk,
band-axis misalignment) does not ship undetected.

Three flavours of misalignment are exercised:

  * Chunk smaller than tile (e.g. ``chunks=11`` on a 16-tile file): a
    single tile must be diced into multiple chunks.
  * Chunk larger than tile and not a multiple (e.g. ``chunks=23`` on
    a 16-tile file): a single chunk must stitch fragments from
    multiple tiles.
  * Final chunk that crops both axes simultaneously (image size not a
    multiple of chunk size, and chunk size not a multiple of tile
    size). Catches the corner cell where every boundary is partial.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
dask_array = pytest.importorskip("dask.array")


def _write_tiled(path: str, arr: np.ndarray, tile: int = 16) -> None:
    """Write *arr* as a tiled TIFF with the requested tile size."""
    tifffile.imwrite(str(path), arr, tile=(tile, tile),
                     photometric="minisblack", compression="deflate")


@pytest.fixture(scope="module")
def _arr_64x96():
    """Deterministic 64x96 uint16 raster reused across chunk-size cases."""
    rng = np.random.RandomState(0xC4AE)
    return rng.randint(0, 60_000, size=(64, 96), dtype=np.uint16)


def test_chunk_smaller_than_tile(tmp_path, _arr_64x96):
    """``chunks=11`` on a 16x16-tile file: tile is subdivided across chunks.

    With image 64x96 and chunks=11 the dask layout is ceil(64/11)=6 row
    blocks and ceil(96/11)=9 column blocks (54 chunks total). Each
    chunk straddles a tile boundary -- if the window-to-tile mapping
    is off by one row or column, the computed value will differ from
    the source.
    """
    from xrspatial.geotiff import read_geotiff_dask

    path = tmp_path / "tiled_misalign_small.tif"
    _write_tiled(path, _arr_64x96, tile=16)

    da_arr = read_geotiff_dask(str(path), chunks=11)
    assert isinstance(da_arr.data, dask_array.Array)
    # 11 < 16: every tile is dispersed across at least 2 chunks.
    assert da_arr.data.chunksize[:2] == (11, 11)
    np.testing.assert_array_equal(da_arr.compute().values, _arr_64x96)


def test_chunk_larger_than_tile_nonmultiple(tmp_path, _arr_64x96):
    """``chunks=23`` on a 16x16-tile file: each chunk stitches partial tiles.

    23 % 16 == 7, so every chunk pulls bytes from a partial tile on at
    least one side. If the reader rounds the requested window down to
    the nearest tile boundary, the chunk shape comes out wrong; if it
    rounds up, the values shift.
    """
    from xrspatial.geotiff import read_geotiff_dask

    path = tmp_path / "tiled_misalign_large.tif"
    _write_tiled(path, _arr_64x96, tile=16)

    da_arr = read_geotiff_dask(str(path), chunks=23)
    assert isinstance(da_arr.data, dask_array.Array)
    assert da_arr.data.chunksize[:2] == (23, 23)
    np.testing.assert_array_equal(da_arr.compute().values, _arr_64x96)


def test_chunk_tuple_doubly_unaligned(tmp_path):
    """Image not a multiple of chunk, chunk not a multiple of tile.

    Image 50x70, tile 16x16, chunks (17, 19). The final row chunk and
    final column chunk both crop, and neither chunk dimension is
    aligned with the tile grid. This is the corner-cell case.
    """
    from xrspatial.geotiff import read_geotiff_dask

    rng = np.random.RandomState(0xDCED)
    arr = rng.randint(0, 256, size=(50, 70), dtype=np.uint8)

    path = tmp_path / "tiled_corner_misalign.tif"
    _write_tiled(path, arr, tile=16)

    da_arr = read_geotiff_dask(str(path), chunks=(17, 19))
    assert da_arr.shape == (50, 70)
    # Last block in each axis is the trimmed remainder.
    block_h = da_arr.data.chunks[0]
    block_w = da_arr.data.chunks[1]
    assert block_h == (17, 17, 16), (
        f"row chunks should be 17,17,16 (50-pixel image, chunks=17), "
        f"got {block_h}"
    )
    assert block_w == (19, 19, 19, 13), (
        f"col chunks should be 19,19,19,13 (70-pixel image, chunks=19), "
        f"got {block_w}"
    )
    np.testing.assert_array_equal(da_arr.compute().values, arr)
