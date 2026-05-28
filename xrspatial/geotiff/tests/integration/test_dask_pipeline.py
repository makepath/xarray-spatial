"""Dask-only pipeline integration tests for the GeoTIFF reader.

Covers chunk/tile alignment, integer-nodata chunking, no-op astype
optimisations, overview level selection, planar multiband, max-pixels
guards, streaming writes, and the accessor IO surface.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, to_geotiff
from xrspatial.geotiff._writer import write

# ----------------------------------------------------------
# Section: dask_chunk_tile_misalignment
# ----------------------------------------------------------
tifffile_dask_chunk_tile_misalignment = pytest.importorskip("tifffile")
dask_array_dask_chunk_tile_misalignment = pytest.importorskip("dask.array")


def _write_tiled_dask_chunk_tile_misalignment(path: str, arr: np.ndarray, tile: int = 16) -> None:
    """Write *arr* as a tiled TIFF with the requested tile size."""
    tifffile_dask_chunk_tile_misalignment.imwrite(
        str(path), arr, tile=(tile, tile),
        photometric="minisblack", compression="deflate")


@pytest.fixture(scope="module")
def _arr_64x96_dask_chunk_tile_misalignment():
    """Deterministic 64x96 uint16 raster reused across chunk-size cases."""
    rng = np.random.RandomState(0xC4AE)
    return rng.randint(0, 60_000, size=(64, 96), dtype=np.uint16)


def test_chunk_smaller_than_tile(tmp_path, _arr_64x96_dask_chunk_tile_misalignment):
    """``chunks=11`` on a 16x16-tile file: tile is subdivided across chunks.

    With image 64x96 and chunks=11 the dask layout is ceil(64/11)=6 row
    blocks and ceil(96/11)=9 column blocks (54 chunks total). Each
    chunk straddles a tile boundary -- if the window-to-tile mapping
    is off by one row or column, the computed value will differ from
    the source.
    """
    from xrspatial.geotiff import read_geotiff_dask

    path = tmp_path / "tiled_misalign_small.tif"
    _write_tiled_dask_chunk_tile_misalignment(path, _arr_64x96_dask_chunk_tile_misalignment, tile=16)  # noqa: E501

    da_arr = read_geotiff_dask(str(path), chunks=11)
    assert isinstance(da_arr.data, dask_array_dask_chunk_tile_misalignment.Array)
    # 11 < 16: every tile is dispersed across at least 2 chunks.
    assert da_arr.data.chunksize[:2] == (11, 11)
    np.testing.assert_array_equal(da_arr.compute().values, _arr_64x96_dask_chunk_tile_misalignment)


def test_chunk_larger_than_tile_nonmultiple(tmp_path, _arr_64x96_dask_chunk_tile_misalignment):
    """``chunks=23`` on a 16x16-tile file: each chunk stitches partial tiles.

    23 % 16 == 7, so every chunk pulls bytes from a partial tile on at
    least one side. If the reader rounds the requested window down to
    the nearest tile boundary, the chunk shape comes out wrong; if it
    rounds up, the values shift.
    """
    from xrspatial.geotiff import read_geotiff_dask

    path = tmp_path / "tiled_misalign_large.tif"
    _write_tiled_dask_chunk_tile_misalignment(path, _arr_64x96_dask_chunk_tile_misalignment, tile=16)  # noqa: E501

    da_arr = read_geotiff_dask(str(path), chunks=23)
    assert isinstance(da_arr.data, dask_array_dask_chunk_tile_misalignment.Array)
    assert da_arr.data.chunksize[:2] == (23, 23)
    np.testing.assert_array_equal(da_arr.compute().values, _arr_64x96_dask_chunk_tile_misalignment)


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
    _write_tiled_dask_chunk_tile_misalignment(path, arr, tile=16)

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

# ----------------------------------------------------------
# Section: dask_int_nodata_chunks
# ----------------------------------------------------------


@pytest.fixture
def uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks(tmp_path):
    """Write a uint16 8x8 TIFF whose nodata sentinel is in the
    bottom-right 2x2 quadrant. With ``chunks=4`` the top-left chunk
    never sees a sentinel and used to keep its uint16 dtype.
    """
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1
    arr[6:8, 6:8] = 65535
    path = str(tmp_path / 'uint16_corner_sentinel_1597.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


def test_eager_promotes_to_float64_and_masks(uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks):  # noqa: E501
    """Baseline: the eager path produces float64 with 4 NaNs."""
    path, _ = uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks
    eager = open_geotiff(path)
    assert eager.dtype == np.float64
    assert np.isnan(eager.values).sum() == 4
    assert np.isnan(eager.values[6:8, 6:8]).all()


def test_dask_chunks_4_matches_eager(uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks):
    """The dask compute result matches the eager path bit-for-bit.

    Before the fix this returned a uint16 array with 0s where the
    sentinel had been, because dask coerced the late-arriving float64
    chunk back to uint16 at concat time.
    """
    path, _ = uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks
    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4)
    assert dk.dtype == np.float64
    computed = dk.compute()
    assert computed.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(computed.values),
                                  np.isnan(eager.values))
    finite = ~np.isnan(eager.values)
    np.testing.assert_array_equal(computed.values[finite],
                                  eager.values[finite])


def test_dask_chunks_2_per_chunk_dtype_uniform(
        uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks):
    """Every dask chunk returns float64 regardless of mask hit.

    Iterates the delayed blocks and asserts each one computes to
    float64; the regression had the first chunk's actual data come back
    as uint16 because the mask never matched there.
    """
    path, _ = uint16_with_sentinel_only_in_corner_dask_int_nodata_chunks
    dk = open_geotiff(path, chunks=2)
    blocks = dk.data.to_delayed().flatten()
    for i, block in enumerate(blocks):
        chunk = block.compute()
        assert chunk.dtype == np.float64, (
            f"chunk {i} computed as {chunk.dtype}, expected float64; "
            f"per-chunk dtype divergence is the #1597 regression."
        )


def test_dask_keeps_dtype_for_out_of_range_sentinel(tmp_path):
    """Out-of-range sentinels (uint16 + nodata=-9999) stay uint16.

    When the sentinel cannot match any pixel, no float64 promotion is
    needed and the dask path keeps the file's native dtype.
    """
    arr = np.array([[1, 2, 3, 4]] * 4, dtype=np.uint16)
    path = str(tmp_path / 'uint16_out_of_range_1597.tif')
    write(arr, path, nodata=-9999, compression='none', tiled=False)

    dk = open_geotiff(path, chunks=2)
    assert dk.dtype == np.uint16
    result = dk.compute()
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.values, arr)


def test_dask_float_input_with_sentinel_in_one_chunk(tmp_path):
    """Float rasters with sentinel in non-first chunk also stay float.

    The float path doesn't promote dtype, but it does in-place NaN
    substitution. Verify the substitution holds for chunks with and
    without the sentinel.
    """
    arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 1
    arr[6:8, 6:8] = -9999.0
    path = str(tmp_path / 'float_corner_sentinel_1597.tif')
    write(arr, path, nodata=-9999, compression='none', tiled=False)

    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4).compute()
    np.testing.assert_array_equal(np.isnan(dk.values),
                                  np.isnan(eager.values))

# ----------------------------------------------------------
# Section: dask_max_pixels_default_guard
# ----------------------------------------------------------


tifffile_dask_max_pixels_default_guard = pytest.importorskip("tifffile")

from xrspatial.geotiff._reader import MAX_PIXELS_DEFAULT  # noqa: E402


def _write_oversized_dask_max_pixels_default_guard(path, *, h: int, w: int) -> None:
    """Write a tiny tiled TIFF whose declared dimensions exceed the cap.

    ``tifffile`` will not let us materialise a multi-billion-pixel array,
    so we exploit the fact that the dask reader only consults the
    header's ImageLength / ImageWidth tags for the up-front guard. The
    physical file is one tile; the header advertises a much larger
    image. Reading any window-less chunk would fail at decode time, but
    that is acceptable because the up-front guard is supposed to fire
    long before chunk tasks run.
    """
    # Smallest possible single-tile file; declare a tiny image so the
    # file is valid, then patch the IFD's width/length tags to advertise
    # an oversized image.
    arr = np.zeros((16, 16), dtype=np.uint8)
    tifffile_dask_max_pixels_default_guard.imwrite(
        str(path), arr, tile=(16, 16),
        photometric="minisblack", compression="none")
    # Rewrite the ImageWidth (256) and ImageLength (257) tags in the IFD.
    # tifffile writes a classic TIFF; the IFD starts at offset 8 for a
    # small file. Parsing the offset properly requires reading the
    # header; do that with tifffile's own parser to stay robust.
    with tifffile_dask_max_pixels_default_guard.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        ifd_offset = page.offset
    raw = bytearray(path.read_bytes())
    # Little-endian classic TIFF; first 2 bytes of IFD = entry count.
    n_entries = int.from_bytes(raw[ifd_offset:ifd_offset + 2], 'little')
    for i in range(n_entries):
        entry = ifd_offset + 2 + i * 12
        tag = int.from_bytes(raw[entry:entry + 2], 'little')
        if tag == 256:  # ImageWidth
            raw[entry + 8:entry + 12] = int(w).to_bytes(4, 'little')
        elif tag == 257:  # ImageLength
            raw[entry + 8:entry + 12] = int(h).to_bytes(4, 'little')
    path.write_bytes(bytes(raw))


def test_default_max_pixels_guard_does_not_fire_up_front(tmp_path):
    """The dask path no longer rejects an oversized full image at graph-
    build time (#2501). The cap is now per-chunk, so a forged
    multi-billion-pixel header builds a lazy graph without erroring;
    each chunk task decodes against ``max_pixels`` separately.
    """
    path = tmp_path / "tmp_2501_oversized.tif"
    side = int((MAX_PIXELS_DEFAULT ** 0.5)) + 2
    _write_oversized_dask_max_pixels_default_guard(path, h=side, w=side)
    # The graph builds; the up-front guard is gone. Chunk-level decode
    # may still fail when a task actually runs, but that is a separate
    # path. ``read_geotiff_dask(...)`` itself returns a DataArray.
    da = read_geotiff_dask(str(path))
    assert da.shape == (side, side)


def test_explicit_max_pixels_enforced_per_chunk(tmp_path):
    """Per-chunk decode honours ``max_pixels`` even when the full image
    would have fit under it (#2501). Build a real 64x64 file, request
    chunks of 32, and set ``max_pixels`` below the per-chunk cost.
    """
    arr = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    path = tmp_path / "tmp_2501_per_chunk_cap.tif"
    tifffile_dask_max_pixels_default_guard.imwrite(
        str(path), arr, tile=(16, 16),
        photometric="minisblack", compression="none")
    # 32x32 chunk = 1024 pixels; cap is 100. Graph builds, compute fails.
    da = read_geotiff_dask(str(path), chunks=32, max_pixels=100)
    with pytest.raises(ValueError, match="exceed the safety limit"):
        da.compute()


def test_small_region_unaffected(tmp_path):
    """The default cap must not interfere with normal small reads."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "tmp_1838_small.tif"
    tifffile_dask_max_pixels_default_guard.imwrite(
        str(path), arr, tile=(16, 16),
        photometric="minisblack", compression="none")
    da = read_geotiff_dask(str(path), chunks=8)
    np.testing.assert_array_equal(da.compute().values, arr)

# ----------------------------------------------------------
# Section: dask_no_op_astype
# ----------------------------------------------------------


@pytest.fixture
def float32_no_nodata_tif_dask_no_op_astype(tmp_path):
    """Write a 16x16 float32 TIFF with no nodata sentinel."""
    rng = np.random.RandomState(1624)
    arr = rng.rand(16, 16).astype(np.float32)
    path = str(tmp_path / 'float32_no_nodata_1624.tif')
    write(arr, path, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def uint16_with_sentinel_in_first_chunk_dask_no_op_astype(tmp_path):
    """uint16 raster with sentinel in chunk 0 so the mask hits there."""
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1
    arr[0, 0] = 65535
    arr[6, 6] = 65535
    path = str(tmp_path / 'uint16_sentinel_1624.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


def test_uint16_mask_path_still_promotes(uint16_with_sentinel_in_first_chunk_dask_no_op_astype):
    """The int-sentinel float64 promotion still runs when sentinels are present."""
    path, arr = uint16_with_sentinel_in_first_chunk_dask_no_op_astype
    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4)
    assert dk.dtype == np.float64
    computed = dk.compute()
    assert computed.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(computed.values),
                                  np.isnan(eager.values))
    # Pixels that held the sentinel in the source array are NaN; every
    # other pixel matches the source value byte-for-byte after the
    # uint -> float64 promotion. Anchors the test to fixture values so
    # any regression in the mask path (e.g. wrong sentinel comparison)
    # surfaces here, not just as dtype drift.
    sentinel_mask = arr == 65535
    np.testing.assert_array_equal(np.isnan(computed.values), sentinel_mask)
    np.testing.assert_array_equal(
        computed.values[~sentinel_mask],
        arr[~sentinel_mask].astype(np.float64),
    )


def test_astype_skipped_when_dtypes_match(float32_no_nodata_tif_dask_no_op_astype, monkeypatch):
    """Direct trace: no astype runs on the per-chunk return path when
    ``target_dtype`` already matches.

    Wraps ``read_to_array`` so the array it returns is a subclass that
    flips a flag whenever ``astype`` is called. With the bug, every
    chunk triggers one same-dtype astype. With the fix, none do.
    """
    from xrspatial.geotiff import _reader as reader_mod
    from xrspatial.geotiff._backends import dask as gt

    path, _ = float32_no_nodata_tif_dask_no_op_astype

    class _AstypeTrackingArray(np.ndarray):
        """ndarray subclass that records astype calls."""

        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            obj._astype_calls = []
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._astype_calls = getattr(obj, '_astype_calls', [])

        def astype(self, dtype, *args, **kwargs):
            self._astype_calls.append(np.dtype(dtype))
            return super().astype(dtype, *args, **kwargs)

    captured: list = []

    orig_r2a = reader_mod.read_to_array

    def wrapped_r2a(*args, **kwargs):
        arr, meta = orig_r2a(*args, **kwargs)
        tracked = _AstypeTrackingArray(arr)
        captured.append(tracked)
        return tracked, meta

    # ``read_geotiff_dask``'s per-chunk worker calls the alias
    # ``_read_to_array`` bound in ``xrspatial.geotiff._backends.dask``.
    # Patch that binding; patching ``_reader.read_to_array`` would not
    # affect the already-imported alias.
    monkeypatch.setattr(gt, '_read_to_array', wrapped_r2a)

    dk = read_geotiff_dask(path, chunks=4)
    dk.compute()

    assert captured, "read_to_array was not invoked"
    for tracked in captured:
        same_dtype_calls = [c for c in tracked._astype_calls
                            if c == tracked.dtype]
        assert not same_dtype_calls, (
            f"Same-dtype astype still runs per chunk "
            f"(dtype={tracked.dtype}, calls={tracked._astype_calls}); "
            f"this is the #1624 regression."
        )


def test_caller_supplied_dtype_still_casts(float32_no_nodata_tif_dask_no_op_astype):
    """Explicit ``dtype=float64`` still triggers the cast."""
    path, _ = float32_no_nodata_tif_dask_no_op_astype
    dk = read_geotiff_dask(path, dtype=np.float64, chunks=4)
    assert dk.dtype == np.float64
    out = dk.compute()
    assert out.dtype == np.float64

# ----------------------------------------------------------
# Section: dask_overview_level
# ----------------------------------------------------------


tifffile_dask_overview_level = pytest.importorskip("tifffile")
dask_array_dask_overview_level = pytest.importorskip("dask.array")


def _write_cog_with_overviews_dask_overview_level(path: str, data: np.ndarray) -> None:
    """Write *data* as a tiled TIFF with two precomputed overview IFDs.

    Writes the primary IFD followed by half- and quarter-resolution
    overview IFDs, each tagged ``subfiletype=1`` so the reader treats
    them as a pyramid (matching how ``_write_normal_cog`` in
    ``test_overview_filter.py`` builds COG fixtures). This mirrors what
    GDAL's ``gdaladdo`` emits.
    """
    half = data[::2, ::2]
    quart = data[::4, ::4]
    with tifffile_dask_overview_level.TiffWriter(path) as tw:
        tw.write(data, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
        tw.write(quart, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)


def test_dask_overview_level_zero_matches_full_res(tmp_path):
    """``overview_level=0`` returns full resolution (the base IFD)."""
    from xrspatial.geotiff import read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov.tif")
    _write_cog_with_overviews_dask_overview_level(path, arr)

    da_arr = read_geotiff_dask(path, chunks=32, overview_level=0)
    assert da_arr.shape == arr.shape
    np.testing.assert_array_equal(da_arr.compute().values, arr)


def test_dask_overview_level_one_returns_half_res(tmp_path):
    """``overview_level=1`` materialises the half-resolution overview."""
    from xrspatial.geotiff import open_geotiff, read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov1.tif")
    _write_cog_with_overviews_dask_overview_level(path, arr)

    # Eager reference at the same overview level -- the dask path should
    # pull the same bytes from the same IFD.
    eager = open_geotiff(path, overview_level=1)

    da_arr = read_geotiff_dask(path, chunks=16, overview_level=1)
    assert da_arr.shape == eager.shape, (
        f"dask returned {da_arr.shape} but eager returned {eager.shape} "
        "at overview_level=1"
    )
    assert isinstance(da_arr.data, dask_array_dask_overview_level.Array)
    np.testing.assert_array_equal(da_arr.compute().values, eager.values)


def test_dask_overview_level_two_returns_quarter_res(tmp_path):
    """``overview_level=2`` materialises the quarter-resolution overview."""
    from xrspatial.geotiff import open_geotiff, read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov2.tif")
    _write_cog_with_overviews_dask_overview_level(path, arr)

    eager = open_geotiff(path, overview_level=2)

    da_arr = read_geotiff_dask(path, chunks=8, overview_level=2)
    assert da_arr.shape == eager.shape
    np.testing.assert_array_equal(da_arr.compute().values, eager.values)


def test_dask_overview_level_none_returns_full_res(tmp_path):
    """``overview_level=None`` keeps default behaviour: full resolution."""
    from xrspatial.geotiff import read_geotiff_dask

    rng = np.random.RandomState(0xD0E)
    arr = rng.randint(0, 256, size=(128, 192), dtype=np.uint8)
    path = str(tmp_path / "cog_dask_ov_none.tif")
    _write_cog_with_overviews_dask_overview_level(path, arr)

    da_arr = read_geotiff_dask(path, chunks=32, overview_level=None)
    assert da_arr.shape == arr.shape
    np.testing.assert_array_equal(da_arr.compute().values, arr)

# ----------------------------------------------------------
# Section: dask_planar_multiband
# ----------------------------------------------------------


tifffile_dask_planar_multiband = pytest.importorskip("tifffile")
dask_array_dask_planar_multiband = pytest.importorskip("dask.array")


def _write_planar_tiff_dask_planar_multiband(
        path: str, data: np.ndarray, *,
        planar: str, tiled: bool) -> None:
    """Write *data* shaped ``(bands, height, width)`` with chosen layout.

    tifffile expects ``(bands, h, w)`` for ``planarconfig='separate'`` and
    ``(h, w, bands)`` for ``planarconfig='contig'``. This helper centralises
    the transpose so the test bodies stay focused on the assertion.
    """
    kwargs: dict = {"photometric": "minisblack"}
    if data.shape[0] == 3:
        kwargs["photometric"] = "rgb"
    if tiled:
        kwargs["tile"] = (32, 32)
    if planar == "separate":
        kwargs["planarconfig"] = "separate"
        tifffile_dask_planar_multiband.imwrite(path, data, **kwargs)
    elif planar == "contig":
        kwargs["planarconfig"] = "contig"
        tifffile_dask_planar_multiband.imwrite(path, np.transpose(data, (1, 2, 0)), **kwargs)
    else:
        raise ValueError(f"unknown planar={planar!r}")


def _make_data_dask_planar_multiband(bands: int, height: int, width: int, dtype) -> np.ndarray:
    rng = np.random.RandomState(0xD45C + bands * 100 + height)
    info = np.iinfo(dtype)
    high = min(int(info.max), 60_000) + 1
    return rng.randint(0, high, size=(bands, height, width)).astype(dtype)


@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dask_planar_multiband_matches_numpy(
    tmp_path, planar, tiled, bands, dtype
):
    """``read_geotiff_dask`` returns ``(y, x, band)`` matching the source."""
    from xrspatial.geotiff import read_geotiff_dask

    height, width = 96, 128
    data = _make_data_dask_planar_multiband(bands, height, width, dtype)
    # On disk the file stores ``(bands, h, w)`` but the reader returns
    # the xarray convention ``(y, x, band)``.
    expected = np.transpose(data, (1, 2, 0))

    path = str(tmp_path
               / f"dask_planar_{planar}_{'tile' if tiled else 'strip'}_"
                 f"b{bands}_{np.dtype(dtype).name}.tif")
    _write_planar_tiff_dask_planar_multiband(path, data, planar=planar, tiled=tiled)

    da_arr = read_geotiff_dask(path, chunks=32)

    assert isinstance(da_arr.data, dask_array_dask_planar_multiband.Array), (
        f"expected dask Array, got {type(da_arr.data).__name__}"
    )
    assert da_arr.shape == (height, width, bands), (
        f"shape mismatch: {da_arr.shape} vs {(height, width, bands)}"
    )
    assert da_arr.dtype == np.dtype(dtype)
    assert list(da_arr.dims) == ["y", "x", "band"]

    materialised = da_arr.compute().values
    np.testing.assert_array_equal(materialised, expected)


def test_dask_planar_separate_chunks_tuple(tmp_path):
    """Tuple chunks ``(ch_h, ch_w)`` honoured; band axis stays single chunk."""
    from xrspatial.geotiff import read_geotiff_dask

    bands, height, width = 3, 80, 120
    data = _make_data_dask_planar_multiband(bands, height, width, np.uint8)
    expected = np.transpose(data, (1, 2, 0))

    path = str(tmp_path / "dask_planar_chunktuple.tif")
    _write_planar_tiff_dask_planar_multiband(path, data, planar="separate", tiled=True)

    da_arr = read_geotiff_dask(path, chunks=(40, 60))

    # ``read_geotiff_dask`` builds row-major chunks of (ch_h, ch_w, n_bands).
    # With height=80, width=120, chunks=(40, 60) the expected layout is
    # 2 row blocks x 2 col blocks x 1 band block.
    assert da_arr.data.chunksize[:2] == (40, 60)
    # The band axis is concatenated as one block (n_bands shape).
    assert da_arr.data.chunksize[2] == bands

    np.testing.assert_array_equal(da_arr.compute().values, expected)

# ----------------------------------------------------------
# Section: dask_streaming_write_degenerate
# ----------------------------------------------------------


def _read_raw_pixels_dask_streaming_write_degenerate(path: str) -> np.ndarray:
    """Read the raw pixel array off disk without xrspatial's NaN-mask
    pass.

    ``open_geotiff`` maps the GDAL_NODATA sentinel back to NaN on
    read, so asserting on its output cannot distinguish (a) a writer
    that left NaNs as floats and (b) a writer that wrote the sentinel
    correctly. ``tifffile`` decodes the pixels but does not consult
    ``GDAL_NODATA``, so a raw read surfaces what is actually on disk.
    """
    tifffile = pytest.importorskip("tifffile")

    with tifffile.TiffFile(path) as tif:
        return tif.asarray()


# ---------------------------------------------------------------------------
# Cat 3: 1x1, 1xN, Nx1 dask streaming writes
# ---------------------------------------------------------------------------


class TestStreamingWrite1x1_dask_streaming_write_degenerate:
    """A single-pixel dask raster must round-trip through the streaming writer."""

    def test_1x1_chunk_matches_shape(self, tmp_path):
        arr = np.array([[42.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == pytest.approx(42.0)

    def test_1x1_with_nodata_attr(self, tmp_path):
        """``attrs['nodata']`` must round-trip even for a 1x1 raster."""
        arr = np.array([[7.5]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_nodata.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == pytest.approx(7.5)
        assert result.attrs.get('nodata') == pytest.approx(-9999.0)

    def test_1x1_uint16(self, tmp_path):
        arr = np.array([[255]], dtype=np.uint16)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 1})
        path = str(tmp_path / '1x1_u16.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert result.shape == (1, 1)
        assert int(result.values[0, 0]) == 255


class TestStreamingWrite1xN_dask_streaming_write_degenerate:
    """A 1-pixel-tall raster exercises the single-tile-row streaming path."""

    def test_1xN_single_chunk(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(1, 10)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 10})
        path = str(tmp_path / '1xN_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_1xN_chunks_split_columns(self, tmp_path):
        """Chunk grid splits the row into multiple column-chunks."""
        arr = np.arange(20, dtype=np.float32).reshape(1, 20)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 7})
        path = str(tmp_path / '1xN_b.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_1xN_wide_segmented_by_buffer(self, tmp_path):
        """Wide single row segmented by streaming_buffer_bytes."""
        arr = np.arange(64, dtype=np.float32).reshape(1, 64)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 1, 'x': 16})
        path = str(tmp_path / '1xN_seg.tif')
        # Tiny streaming buffer so the segmenter splits the tile-row.
        to_geotiff(da, path, tile_size=16,
                   streaming_buffer_bytes=1)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


class TestStreamingWriteNx1_dask_streaming_write_degenerate:
    """A 1-pixel-wide raster exercises the column-degenerate streaming path."""

    def test_Nx1_single_chunk(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 10, 'x': 1})
        path = str(tmp_path / 'Nx1_a.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_Nx1_chunks_split_rows(self, tmp_path):
        """Chunk grid splits the column into multiple row-chunks."""
        arr = np.arange(20, dtype=np.float32).reshape(20, 1)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 7, 'x': 1})
        path = str(tmp_path / 'Nx1_b.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


# ---------------------------------------------------------------------------
# Cat 2: NaN / Inf dask streaming writes
# ---------------------------------------------------------------------------


class TestStreamingWriteAllNan_dask_streaming_write_degenerate:
    """All-NaN dask raster must mask every pixel to the nodata sentinel."""

    def test_all_nan_with_sentinel(self, tmp_path):
        arr = np.full((8, 8), np.nan, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 4, 'x': 4})
        path = str(tmp_path / 'allnan.tif')
        to_geotiff(da, path)
        # Raw decode (no NaN-mask pass): every pixel must be the
        # sentinel on disk. Asserting against ``open_geotiff``'s output
        # alone would also pass if the writer left NaNs as floats,
        # because the reader maps both NaN and the sentinel back to NaN.
        raw = _read_raw_pixels_dask_streaming_write_degenerate(path)
        assert (raw == -9999.0).all(), (
            "writer must replace NaN with the GDAL_NODATA sentinel on "
            "disk; raw read shows non-sentinel pixels"
        )
        assert not np.isnan(raw).any()
        # Public read still maps the sentinel back to NaN.
        result = open_geotiff(path)
        assert np.isnan(result.values).all()
        assert result.attrs.get('nodata') == pytest.approx(-9999.0)

    def test_all_nan_default_nodata(self, tmp_path):
        """``attrs['nodata']`` omitted -- the streaming writer must still
        accept the all-NaN input. The reader cannot mask without a
        sentinel so the float NaN survives in the file."""
        arr = np.full((4, 4), np.nan, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allnan_nosen.tif')
        to_geotiff(da, path)
        # No sentinel declared, so the file must carry raw NaN floats
        # on disk -- a regression coercing NaN to some default sentinel
        # would silently change the file's contents and would not be
        # visible through ``open_geotiff`` alone.
        raw = _read_raw_pixels_dask_streaming_write_degenerate(path)
        assert np.isnan(raw).all()
        result = open_geotiff(path)
        assert np.isnan(result.values).all()


class TestStreamingWriteMixedNanInf_dask_streaming_write_degenerate:
    """Mixed NaN / +Inf / -Inf in a single dask raster.

    The streaming writer must (a) replace NaN with the nodata sentinel,
    (b) leave +Inf and -Inf untouched (they are valid IEEE-754 floats).
    """

    def test_mixed_nan_plus_minus_inf(self, tmp_path):
        arr = np.array([
            [1.0, np.nan, 3.0, 4.0],
            [np.inf, 6.0, -np.inf, 8.0],
            [9.0, 10.0, np.nan, 12.0],
            [13.0, np.inf, 15.0, -np.inf],
        ], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0}).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'mixed.tif')
        to_geotiff(da, path)
        # Raw decode pins the on-disk encoding: NaN cells were
        # coerced to the sentinel, Inf cells were left as IEEE-754
        # Inf. A regression that stopped the NaN-to-sentinel coercion
        # would still pass an ``open_geotiff``-only assertion because
        # the reader maps both NaN and the sentinel back to NaN.
        raw = _read_raw_pixels_dask_streaming_write_degenerate(path)
        assert raw[0, 1] == -9999.0
        assert raw[2, 2] == -9999.0
        assert raw[1, 0] == np.inf
        assert raw[3, 1] == np.inf
        assert raw[1, 2] == -np.inf
        assert raw[3, 3] == -np.inf
        assert not np.isnan(raw).any(), (
            "writer must coerce every NaN to the sentinel; raw read "
            "found surviving NaN floats"
        )
        # Public read maps the sentinel back to NaN, keeps Inf as-is.
        result = open_geotiff(path)
        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[2, 2])
        assert result.values[1, 0] == np.inf
        assert result.values[3, 1] == np.inf
        assert result.values[1, 2] == -np.inf
        assert result.values[3, 3] == -np.inf
        assert result.values[0, 0] == pytest.approx(1.0)
        assert result.values[2, 0] == pytest.approx(9.0)


class TestStreamingWriteAllInf_dask_streaming_write_degenerate:
    """All +Inf and all -Inf dask streaming writes.

    +Inf and -Inf are valid IEEE-754 floats; the streaming writer
    should pass them through unchanged. The reader keeps Inf as Inf
    because the nodata mask only matches the sentinel value, not Inf.
    """

    def test_all_plus_inf(self, tmp_path):
        arr = np.full((4, 4), np.inf, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allposinf.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert (result.values == np.inf).all()

    def test_all_minus_inf(self, tmp_path):
        arr = np.full((4, 4), -np.inf, dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 2, 'x': 2})
        path = str(tmp_path / 'allneginf.tif')
        to_geotiff(da, path)
        result = open_geotiff(path)
        assert (result.values == -np.inf).all()


# ---------------------------------------------------------------------------
# Cat 4: predictor=3 floating-point predictor through dask streaming write
# ---------------------------------------------------------------------------


class TestStreamingWriteFloatPredictor_dask_streaming_write_degenerate:
    """``predictor=3`` (TIFF float predictor) on small dask rasters.

    ``unit/test_predictor.py::test_predictor3_streaming_dask``
    already covers a dask-backed streaming write with ``predictor=3``
    on a 128x192 raster and pins the Predictor tag. The tests below
    extend coverage with smaller chunk geometries (16x16) and lock the
    int-dtype ValueError on the streaming path so the dtype guard
    cannot regress silently.
    """

    def test_predictor3_float32_round_trip(self, tmp_path):
        rng = np.random.default_rng(2026_05_15)
        arr = rng.random((40, 40), dtype=np.float32) * 100.0
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_f32.tif')
        to_geotiff(da, path, compression='deflate', predictor=3,
                   tile_size=16)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_predictor3_float64_round_trip(self, tmp_path):
        rng = np.random.default_rng(2026_05_15)
        arr = rng.random((32, 32), dtype=np.float64) * 100.0
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_f64.tif')
        to_geotiff(da, path, compression='deflate', predictor=3,
                   tile_size=16)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_predictor3_int_input_rejected(self, tmp_path):
        """``predictor=3`` requires float dtype; int input must raise."""
        arr = np.arange(32 * 32, dtype=np.int32).reshape(32, 32)
        da = xr.DataArray(arr, dims=['y', 'x']).chunk({'y': 16, 'x': 16})
        path = str(tmp_path / 'pred3_i32.tif')
        with pytest.raises(ValueError, match='predictor'):
            to_geotiff(da, path, compression='deflate', predictor=3,
                       tile_size=16)

# ----------------------------------------------------------
# Section: accessor_io
# ----------------------------------------------------------


def _make_da_accessor_io(height=8, width=10, crs=4326, name='elevation'):
    """Build a georeferenced DataArray for testing."""
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.0, 44.0, height)
    x = np.linspace(-120.0, -119.0, width)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name=name,
        attrs={'crs': crs},
    )


def _make_ds_accessor_io(height=8, width=10, crs=4326):
    """Build a georeferenced Dataset for testing."""
    da = _make_da_accessor_io(height, width, crs, name='elevation')
    return xr.Dataset({'elevation': da})


# ---------------------------------------------------------------------------
# DataArray.xrs.to_geotiff
# ---------------------------------------------------------------------------

class TestDataArrayToGeotiff_accessor_io:
    def test_round_trip(self, tmp_path):
        da = _make_da_accessor_io()
        path = str(tmp_path / 'test_1047_da_roundtrip.tif')
        da.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_with_kwargs(self, tmp_path):
        da = _make_da_accessor_io()
        path = str(tmp_path / 'test_1047_da_kwargs.tif')
        da.xrs.to_geotiff(path, compression='deflate', tiled=True,
                          tile_size=256)

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_preserves_crs(self, tmp_path):
        da = _make_da_accessor_io(crs=32610)
        path = str(tmp_path / 'test_1047_da_crs.tif')
        da.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        assert result.attrs.get('crs') == 32610


# ---------------------------------------------------------------------------
# Dataset.xrs.to_geotiff
# ---------------------------------------------------------------------------

class TestDatasetToGeotiff_accessor_io:
    def test_round_trip(self, tmp_path):
        ds = _make_ds_accessor_io()
        path = str(tmp_path / 'test_1047_ds_roundtrip.tif')
        ds.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, ds['elevation'].values)

    def test_explicit_var(self, tmp_path):
        ds = _make_ds_accessor_io()
        ds['slope'] = ds['elevation'] * 2
        path = str(tmp_path / 'test_1047_ds_var.tif')
        ds.xrs.to_geotiff(path, var='slope', compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, ds['slope'].values)

    def test_no_yx_raises(self, tmp_path):
        ds = xr.Dataset({'vals': xr.DataArray(np.zeros(5), dims=['z'])})
        with pytest.raises(ValueError, match="no variable with 'y' and 'x'"):
            ds.xrs.to_geotiff(str(tmp_path / 'bad.tif'))


# ---------------------------------------------------------------------------
# Dataset.xrs.open_geotiff (spatially-windowed read)
# ---------------------------------------------------------------------------

class TestDatasetOpenGeotiff_accessor_io:
    def test_windowed_read(self, tmp_path):
        """Reading with a Dataset template should return a spatial subset."""
        # Write a 20x20 raster
        big = _make_da_accessor_io(height=20, width=20)
        big_path = str(tmp_path / 'test_1047_big.tif')
        to_geotiff(big, big_path, compression='none')

        # Template dataset covers the center region
        y_sub = big.coords['y'].values[5:15]
        x_sub = big.coords['x'].values[5:15]
        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros((len(y_sub), len(x_sub))),
                dims=['y', 'x'],
                coords={'y': y_sub, 'x': x_sub},
            )
        })

        result = template.xrs.open_geotiff(big_path)
        # Result should be smaller than the full raster
        assert result.shape[0] <= 20
        assert result.shape[1] <= 20
        # And at least as large as the template
        assert result.shape[0] >= len(y_sub)
        assert result.shape[1] >= len(x_sub)

    def test_full_extent_returns_all(self, tmp_path):
        """Template covering full extent should return the whole raster."""
        da = _make_da_accessor_io(height=8, width=10)
        path = str(tmp_path / 'test_1047_full.tif')
        to_geotiff(da, path, compression='none')

        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros_like(da.values),
                dims=['y', 'x'],
                coords={'y': da.coords['y'].values,
                        'x': da.coords['x'].values},
            )
        })
        result = template.xrs.open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_no_coords_raises(self, tmp_path):
        da = _make_da_accessor_io()
        path = str(tmp_path / 'test_1047_nocoords.tif')
        to_geotiff(da, path, compression='none')

        ds = xr.Dataset({'vals': xr.DataArray(np.zeros(5), dims=['z'])})
        with pytest.raises(ValueError, match="'y' and 'x' coordinates"):
            ds.xrs.open_geotiff(path)

    def test_kwargs_forwarded(self, tmp_path):
        """Extra kwargs like name= should be forwarded to open_geotiff."""
        da = _make_da_accessor_io(height=8, width=10)
        path = str(tmp_path / 'test_1047_kwargs.tif')
        to_geotiff(da, path, compression='none')

        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros_like(da.values),
                dims=['y', 'x'],
                coords={'y': da.coords['y'].values,
                        'x': da.coords['x'].values},
            )
        })
        result = template.xrs.open_geotiff(path, name='myname')
        assert result.name == 'myname'


# ---------------------------------------------------------------------------
# DataArray.xrs.open_geotiff (issue #2557 - DataArray-side windowed read
# with backend inference and auto-reproject on CRS mismatch)
# ---------------------------------------------------------------------------


class TestDataArrayOpenGeotiff_2557:
    def test_windowed_read(self, tmp_path):
        big = _make_da_accessor_io(height=20, width=20)
        big_path = str(tmp_path / 'test_2557_da_window.tif')
        to_geotiff(big, big_path, compression='none')

        y_sub = big.coords['y'].values[5:15]
        x_sub = big.coords['x'].values[5:15]
        template = xr.DataArray(
            np.zeros((len(y_sub), len(x_sub)), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': y_sub, 'x': x_sub},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(big_path)
        # Half-pixel expansion may include one extra edge row/col
        assert len(y_sub) <= result.shape[0] <= len(y_sub) + 2
        assert len(x_sub) <= result.shape[1] <= len(x_sub) + 2

    def test_no_coords_raises(self, tmp_path):
        da = _make_da_accessor_io()
        path = str(tmp_path / 'test_2557_da_nocoords.tif')
        to_geotiff(da, path, compression='none')

        bad = xr.DataArray(np.zeros(5), dims=['z'])
        with pytest.raises(ValueError, match="'y' and 'x' coordinates"):
            bad.xrs.open_geotiff(path)

    def test_kwargs_forwarded(self, tmp_path):
        da = _make_da_accessor_io(height=8, width=10)
        path = str(tmp_path / 'test_2557_da_kwargs.tif')
        to_geotiff(da, path, compression='none')

        template = xr.DataArray(
            np.zeros_like(da.values),
            dims=['y', 'x'],
            coords={'y': da.coords['y'].values,
                    'x': da.coords['x'].values},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(path, name='myname')
        assert result.name == 'myname'


# ---------------------------------------------------------------------------
# Backend inference: caller backend -> result backend (issue #2557)
# ---------------------------------------------------------------------------


class TestOpenGeotiffBackendInference_2557:
    def test_numpy_caller_returns_numpy(self, tmp_path):
        big = _make_da_accessor_io(height=10, width=10)
        path = str(tmp_path / 'test_2557_numpy.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            np.zeros((10, 10), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(path)
        assert isinstance(result.data, np.ndarray)
        assert result.chunks is None

    def test_dask_caller_returns_dask_with_inferred_chunks(self, tmp_path):
        import dask.array as dda
        big = _make_da_accessor_io(height=12, width=12)
        path = str(tmp_path / 'test_2557_dask.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            dda.zeros((12, 12), chunks=(4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(path)
        assert isinstance(result.data, dda.Array)
        # First y-chunk should match the caller (4)
        assert result.chunks[0][0] == 4

    def test_explicit_chunks_override_inference(self, tmp_path):
        import dask.array as dda
        big = _make_da_accessor_io(height=12, width=12)
        path = str(tmp_path / 'test_2557_chunks_override.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            dda.zeros((12, 12), chunks=(4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(path, chunks=3)
        assert isinstance(result.data, dda.Array)
        # Caller asked for 3, override wins over inferred 4
        assert result.chunks[0][0] == 3

    def test_dataset_caller_infers_from_first_var(self, tmp_path):
        import dask.array as dda
        big = _make_da_accessor_io(height=10, width=10)
        path = str(tmp_path / 'test_2557_ds_backend.tif')
        to_geotiff(big, path, compression='none')

        var = xr.DataArray(
            dda.zeros((10, 10), chunks=(5, 5), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            attrs={'crs': 4326},
        )
        ds = xr.Dataset({'elevation': var})
        result = ds.xrs.open_geotiff(path)
        assert isinstance(result.data, dda.Array)
        assert result.chunks[0][0] == 5


# ---------------------------------------------------------------------------
# CRS mismatch handling (issue #2557)
# ---------------------------------------------------------------------------


class TestOpenGeotiffCRSMismatch_2557:
    def test_mismatch_raises_by_default(self, tmp_path):
        # File in EPSG:4326
        big = _make_da_accessor_io(height=10, width=10, crs=4326)
        path = str(tmp_path / 'test_2557_mismatch.tif')
        to_geotiff(big, path, compression='none')

        # Caller in EPSG:3857
        template = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': np.linspace(5e6, 4.9e6, 4),
                    'x': np.linspace(-1.3e7, -1.32e7, 4)},
            attrs={'crs': 3857},
        )
        with pytest.raises(ValueError, match="CRS mismatch"):
            template.xrs.open_geotiff(path)

    def test_auto_reproject_returns_caller_crs(self, tmp_path):
        # File in EPSG:4326 over a small mid-latitude box
        height, width = 30, 30
        arr = np.arange(height * width,
                        dtype=np.float32).reshape(height, width)
        y = np.linspace(45.5, 44.5, height)
        x = np.linspace(-120.5, -119.5, width)
        file_da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        path = str(tmp_path / 'test_2557_autoreproj.tif')
        to_geotiff(file_da, path, compression='none')

        # Caller in EPSG:3857 over the same region
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, 3857, always_xy=True)
        x0, y0 = tr.transform(-120.25, 45.25)
        x1, y1 = tr.transform(-119.75, 44.75)
        template = xr.DataArray(
            np.zeros((6, 6), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': np.linspace(max(y0, y1), min(y0, y1), 6),
                    'x': np.linspace(min(x0, x1), max(x0, x1), 6)},
            attrs={'crs': 3857},
        )
        result = template.xrs.open_geotiff(path, auto_reproject=True)
        # Result coords are in caller's CRS (mercator metres). Check the
        # bbox roughly matches the caller's bbox so a future regression
        # in projection direction would be caught. Tolerance is one
        # output pixel's worth of metres at this latitude (~150 km of
        # bbox / 6 pixels).
        tol = abs(float(template.coords['x'][1] - template.coords['x'][0]))
        assert abs(float(result.coords['x'].min())
                   - float(template.coords['x'].min())) < 2 * tol
        assert abs(float(result.coords['x'].max())
                   - float(template.coords['x'].max())) < 2 * tol

    def test_chained_open_geotiff_with_wkt_crs(self, tmp_path):
        # After auto_reproject, xrspatial.reproject sets attrs['crs'] to
        # a WKT string. Calling open_geotiff again on that DataArray must
        # not crash on int() conversion (regression for PR #2598 review).
        height, width = 20, 20
        arr = np.arange(height * width,
                        dtype=np.float32).reshape(height, width)
        y = np.linspace(45.5, 44.5, height)
        x = np.linspace(-120.5, -119.5, width)
        file_da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        path = str(tmp_path / 'test_2557_wkt.tif')
        to_geotiff(file_da, path, compression='none')

        # Synthesize a caller with a WKT-string crs (what reproject sets)
        from pyproj import CRS as _PyprojCRS
        wkt_3857 = _PyprojCRS(3857).to_wkt()
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, 3857, always_xy=True)
        x0, y0 = tr.transform(-120.25, 45.25)
        x1, y1 = tr.transform(-119.75, 44.75)
        template = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': np.linspace(max(y0, y1), min(y0, y1), 4),
                    'x': np.linspace(min(x0, x1), max(x0, x1), 4)},
            attrs={'crs': wkt_3857},
        )
        # Should raise ValueError (mismatch), not TypeError/ValueError
        # from int(wkt_string)
        with pytest.raises(ValueError, match="CRS mismatch"):
            template.xrs.open_geotiff(path)
        # And auto_reproject path also works. reproject preserves the
        # file's native pixel resolution, so the result shape reflects
        # the windowed slice of the file rather than the caller's
        # template shape -- the (20x20-pixel, 1deg-extent) file at a
        # ~0.5deg caller bbox gives ~10 pixels per side plus boundary
        # rounding. Bound generously.
        result = template.xrs.open_geotiff(path, auto_reproject=True)
        assert result.ndim == 2
        assert 4 <= result.shape[0] <= 20
        assert 4 <= result.shape[1] <= 20

    def test_no_caller_crs_no_mismatch_check(self, tmp_path):
        # Caller without attrs['crs'] should skip mismatch logic and
        # proceed (preserves prior behavior).
        big = _make_da_accessor_io(height=10, width=10, crs=4326)
        path = str(tmp_path / 'test_2557_nocrs.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            np.zeros((10, 10), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            # No attrs['crs']
        )
        # Should not raise
        result = template.xrs.open_geotiff(path)
        np.testing.assert_array_equal(result.values, big.values)

    def test_dataset_crs_falls_back_from_ds_attrs(self, tmp_path):
        # CRS on Dataset attrs, not on the data_var, should still be
        # picked up.
        big = _make_da_accessor_io(height=10, width=10, crs=4326)
        path = str(tmp_path / 'test_2557_ds_crs_attr.tif')
        to_geotiff(big, path, compression='none')

        var = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': np.linspace(5e6, 4.9e6, 4),
                    'x': np.linspace(-1.3e7, -1.32e7, 4)},
        )
        ds = xr.Dataset({'elevation': var}, attrs={'crs': 3857})
        with pytest.raises(ValueError, match="CRS mismatch"):
            ds.xrs.open_geotiff(path)

    def test_malformed_crs_raises(self, tmp_path):
        # A garbage attrs['crs'] must raise rather than silently skip
        # the mismatch check (follow-up review hardening).
        big = _make_da_accessor_io(height=10, width=10, crs=4326)
        path = str(tmp_path / 'test_2557_bad_crs.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values[:4],
                    'x': big.coords['x'].values[:4]},
            attrs={'crs': 'not-a-real-crs'},
        )
        with pytest.raises(ValueError, match="not a valid CRS"):
            template.xrs.open_geotiff(path)


# ---------------------------------------------------------------------------
# GPU backend inference (gated on CUDA + cupy availability) - issue #2557
# ---------------------------------------------------------------------------

from xrspatial.tests.general_checks import cuda_and_cupy_available  # noqa: E402


@cuda_and_cupy_available
class TestOpenGeotiffGPUBackendInference_2557:
    def test_cupy_caller_returns_cupy(self, tmp_path):
        import cupy as cp
        big = _make_da_accessor_io(height=10, width=10)
        path = str(tmp_path / 'test_2557_cupy.tif')
        to_geotiff(big, path, compression='none')

        template = xr.DataArray(
            cp.zeros((10, 10), dtype=cp.float32),
            dims=['y', 'x'],
            coords={'y': big.coords['y'].values,
                    'x': big.coords['x'].values},
            attrs={'crs': 4326},
        )
        result = template.xrs.open_geotiff(path)
        # Result should be cupy-backed (gpu=True was inferred)
        assert type(result.data).__module__.split('.')[0] == 'cupy'
