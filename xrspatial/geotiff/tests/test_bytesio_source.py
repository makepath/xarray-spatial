"""Round-trip and concurrency tests for file-like (BytesIO) read/write paths.

Closes #1511. The reader and writer accept any binary file-like that exposes
``read``/``seek`` (reader) or ``write`` (writer), so users can keep encoded
TIFF bytes in memory or stream them through a non-filesystem destination.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._reader import _BytesIOSource, read_to_array


def _make_da(height=32, width=40, dtype=np.float32):
    arr = np.arange(height * width, dtype=dtype).reshape(height, width)
    # Simple geotransform with negative pixel_height (north-up)
    x = np.arange(width, dtype=np.float64) * 0.5 + 100.0 + 0.25
    y = np.arange(height, dtype=np.float64) * (-0.5) + 50.0 - 0.25
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='test',
        attrs={
            'crs': 4326,
            'transform': (0.5, 0.0, 100.0, 0.0, -0.5, 50.0),
        },
    )
    return da


class TestBytesIORoundTrip:
    def test_round_trip_basic(self):
        """to_geotiff(BytesIO) + open_geotiff(BytesIO) round-trips data and crs."""
        da = _make_da()
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate')
        buf.seek(0)
        # Read back from a fresh BytesIO so the read path has to seek itself
        round_tripped = open_geotiff(io.BytesIO(buf.getvalue()))

        np.testing.assert_array_equal(round_tripped.values, da.values)
        assert round_tripped.attrs.get('crs') == 4326
        # Transform tuple preserved verbatim
        assert round_tripped.attrs.get('transform') == da.attrs['transform']

    def test_round_trip_uint8(self):
        """uint8 round-trip without compression."""
        da = _make_da(dtype=np.uint8)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='none')
        result = open_geotiff(io.BytesIO(buf.getvalue()))
        np.testing.assert_array_equal(result.values, da.values)


class TestBytesIOWindowedRead:
    def test_windowed_read(self):
        """Windowed read from a BytesIO source returns the right slice."""
        da = _make_da(height=64, width=64)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate', tiled=True, tile_size=16)

        # Window covering tile-aligned and unaligned regions
        window = (8, 12, 40, 48)
        arr, _ = read_to_array(io.BytesIO(buf.getvalue()), window=window)
        expected = da.values[8:40, 12:48]
        np.testing.assert_array_equal(arr, expected)


class TestBytesIORejectsCog:
    def test_cog_with_file_like_rejected(self):
        """cog=True with a file-like destination raises ValueError."""
        da = _make_da()
        buf = io.BytesIO()
        with pytest.raises(ValueError, match='cog=True'):
            to_geotiff(da, buf, cog=True)


class TestBytesIORejectsVrt:
    def test_vrt_extension_with_file_like_is_treated_as_geotiff(self):
        """A file-like destination cannot carry a VRT extension marker.

        VRT output is filesystem-only (it writes per-tile sidecar GeoTIFFs
        to a directory), so passing a buffer with an unrelated name should
        not select the VRT path. The check uses isinstance(path, str), so a
        BytesIO simply gets treated as a plain GeoTIFF destination.
        """
        # No way to "name" a BytesIO as .vrt, so this asserts the gate works:
        # passing a BytesIO produces a normal GeoTIFF.
        da = _make_da()
        buf = io.BytesIO()
        to_geotiff(da, buf)
        # Header magic for classic TIFF is II*\0 or MM\0*
        head = buf.getvalue()[:4]
        assert head in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+')

    def test_explicit_vrt_path_string_with_file_like_data_works(self, tmp_path):
        """A real .vrt path with file-like data is unrelated to this PR.

        This test just sanity-checks that string-path VRT handling still
        rejects cog=True (existing behaviour) -- the regression we worry
        about is the writer accidentally taking the VRT branch when
        ``path`` is a buffer.
        """
        da = _make_da()
        with pytest.raises(ValueError, match='cog'):
            to_geotiff(da, str(tmp_path / 'out.vrt'), cog=True)


class TestBytesIOConcurrentReads:
    def test_concurrent_windowed_reads(self):
        """Many threads reading disjoint windows from one BytesIOSource see
        consistent bytes (lock around seek+read prevents cursor races)."""
        da = _make_da(height=128, width=128)
        buf = io.BytesIO()
        to_geotiff(da, buf, compression='deflate', tiled=True, tile_size=16)
        encoded = buf.getvalue()

        # All threads share one source instance.
        shared = io.BytesIO(encoded)
        src = _BytesIOSource(shared)

        # Pick byte ranges scattered through the file.
        size = src.size
        rng = np.random.default_rng(0)
        ranges = []
        for _ in range(64):
            start = int(rng.integers(0, max(1, size - 32)))
            length = int(rng.integers(1, min(128, size - start) + 1))
            ranges.append((start, length))

        # Compute the truth values single-threaded.
        truth = [encoded[s:s + n] for s, n in ranges]

        results = [None] * len(ranges)
        errors: list[BaseException] = []

        def worker(i):
            try:
                s, n = ranges[i]
                results[i] = src.read_range(s, n)
            except BaseException as e:  # pragma: no cover - defensive
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(len(ranges))))

        assert not errors, errors
        for got, want in zip(results, truth):
            assert got == want


# ---------------------------------------------------------------------------
# PR #1512 review followups: pathlib.Path normalisation, GPU+file-like reject,
# truncate-on-rewrite semantics, _is_file_like requires tell.
# ---------------------------------------------------------------------------


def _make_small_da():
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326},
    )


def test_pathlib_path_write_and_read_roundtrip(tmp_path):
    """``pathlib.Path`` should behave identically to ``str``."""
    p = tmp_path / 'roundtrip.tif'
    da = _make_small_da()
    to_geotiff(da, p)  # Path, not str
    out = open_geotiff(p)  # Path, not str
    np.testing.assert_array_equal(out.values, da.values)


def test_pathlib_path_vrt_routes_to_read_vrt(tmp_path):
    """``Path('x.vrt')`` must dispatch to the VRT path, not the TIFF reader."""
    import pathlib

    da = _make_small_da()
    tif_path = tmp_path / 'src.tif'
    to_geotiff(da, str(tif_path))

    vrt_path = tmp_path / 'mosaic.vrt'
    vrt_path.write_text(f"""<VRTDataset rasterXSize="8" rasterYSize="8">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tif_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="8" ySize="8"/>
      <DstRect xOff="0" yOff="0" xSize="8" ySize="8"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""")
    out = open_geotiff(pathlib.Path(vrt_path))
    np.testing.assert_array_equal(out.values, da.values)


def test_pathlib_path_derives_name(tmp_path):
    """``name`` should derive from ``Path`` stem, same as for ``str``."""
    p = tmp_path / 'mydata.tif'
    to_geotiff(_make_small_da(), p)
    out = open_geotiff(p)
    assert out.name == 'mydata'


def test_to_geotiff_rejects_gpu_with_file_like():
    """``gpu=True`` + file-like is untested territory; reject up front."""
    da = _make_small_da()
    buf = io.BytesIO()
    with pytest.raises(ValueError, match='gpu=True is not supported'):
        to_geotiff(da, buf, gpu=True)


def test_to_geotiff_buffer_rewritten_in_place():
    """Reusing the same BytesIO should overwrite, not append.

    Mirrors ``to_geotiff('/tmp/x.tif', ...)`` followed by another call to
    the same path -- the second call replaces, not concatenates. PR #1512
    review found that file-like writes used to ``write()`` at the cursor
    so two writes to one buffer produced two TIFFs back-to-back.
    """
    buf = io.BytesIO()
    to_geotiff(_make_small_da(), buf)
    first_size = len(buf.getvalue())
    assert first_size > 0

    # Write a second, different DataArray to the same buffer.
    arr2 = np.full((4, 4), 7.0, dtype=np.float32)
    da2 = xr.DataArray(
        arr2, dims=['y', 'x'],
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        attrs={'crs': 4326},
    )
    to_geotiff(da2, buf)
    second_size = len(buf.getvalue())

    # The second TIFF should fully replace the first, not append.
    out = open_geotiff(io.BytesIO(buf.getvalue()))
    np.testing.assert_array_equal(out.values, arr2)
    # And the buffer must be smaller than first+second (i.e. no concat).
    assert second_size < first_size + 100


def test_is_file_like_requires_tell():
    """Objects with read+seek but no tell are not accepted as file-like.

    ``_BytesIOSource`` needs ``tell`` to compute the buffer size. We refuse
    seekable-but-not-tellable inputs at the gate rather than crashing inside
    ``__init__``.
    """
    from xrspatial.geotiff._reader import _is_file_like

    class ReadSeekNoTell:
        def read(self, n=-1):
            return b''

        def seek(self, *a, **k):
            return 0

    assert _is_file_like(io.BytesIO(b'x')) is True
    assert _is_file_like(ReadSeekNoTell()) is False


class TestWriteGeotiffGpuBytesIO:
    """Regression coverage for ``write_geotiff_gpu`` file-like behaviour.

    ``to_geotiff(gpu=True, ...)`` always rejects BytesIO destinations paired
    with ``cog=True`` (the auto-dispatch path's existing guard). The explicit
    GPU writer used to silently accept that combo and produce a COG into the
    buffer, so the two entry points disagreed on what ``to_geotiff(gpu=True,
    cog=True, path=BytesIO)`` does. These tests pin the mirrored gate added
    by issue #1652 and confirm the non-cog file-like path still works.
    """

    def test_cog_with_bytesio_rejected_1652(self):
        cupy = pytest.importorskip("cupy")
        da = xr.DataArray(
            cupy.asarray(np.random.rand(64, 64).astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        buf = io.BytesIO()
        with pytest.raises(ValueError, match='cog=True'):
            write_geotiff_gpu(da, buf, cog=True)

    def test_non_cog_bytesio_still_works_1652(self):
        cupy = pytest.importorskip("cupy")
        arr_cpu = np.random.rand(64, 64).astype(np.float32)
        da = xr.DataArray(
            cupy.asarray(arr_cpu),
            dims=['y', 'x'],
            coords={'y': np.arange(64.0), 'x': np.arange(64.0)},
            attrs={'crs': 4326},
        )
        from xrspatial.geotiff import write_geotiff_gpu

        buf = io.BytesIO()
        # Non-cog file-like write is still supported on the explicit GPU
        # writer; only cog=True is gated.
        write_geotiff_gpu(da, buf)
        assert len(buf.getvalue()) > 0

        # Verify it round-trips through open_geotiff
        rd = open_geotiff(io.BytesIO(buf.getvalue()))
        np.testing.assert_allclose(np.asarray(rd.values), arr_cpu)
