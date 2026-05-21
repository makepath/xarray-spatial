"""Lazy chunked read_vrt builds a real dask graph (issue #1814).

The pre-fix ``read_vrt(chunks=...)`` materialised the full VRT mosaic
on host RAM, then wrapped the resulting numpy array via ``.chunk()``.
That defeated the purpose of ``chunks=`` for memory protection and
made ``gpu=True`` + ``chunks=`` even worse: the entire mosaic was
moved to the device before chunking.

These tests cover the new lazy path:

* construction does not decode any pixels;
* per-chunk decode happens at ``.compute()`` time;
* the resulting array is byte-identical to the eager read;
* the chunk task count is bounded so a typo in ``chunks=`` cannot
  build a graph the scheduler refuses to dispatch.
"""
from __future__ import annotations

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_vrt, to_geotiff
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


def _gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


@pytest.fixture
def single_tile_vrt():
    """One 128x128 float32 tile wrapped in a VRT."""
    arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    y = np.linspace(41.0, 40.0, 128)
    x = np.linspace(-106.0, -105.0, 128)
    raster = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_single_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield vrt_path, arr


@pytest.fixture
def two_by_two_vrt():
    """4-tile mosaic via the to_geotiff(.vrt, ...) dask path."""
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    y = np.linspace(41.0, 40.0, 256)
    x = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_2x2_')
    vrt_path = os.path.join(td, 'mosaic.vrt')
    # ``tile_size=128`` produces a 2x2 mosaic of 128x128 tiles.
    to_geotiff(raster, vrt_path, tile_size=128)
    yield vrt_path, arr


@pytest.fixture
def multiband_vrt():
    """3-band single-tile VRT."""
    rng = np.random.default_rng(1814)
    arr = rng.random((64, 64, 3), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 64)
    x = np.linspace(-106.0, -105.0, 64)
    raster = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={'y': y, 'x': x, 'band': np.arange(3)},
        attrs={'crs': 4326},
    )
    td = tempfile.mkdtemp(prefix='tmp_1814_mb_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield vrt_path, arr


# ---------------------------------------------------------------------------
# 1. Construction is lazy: no pixels are decoded before .compute().
# ---------------------------------------------------------------------------

def test_chunks_builds_dask_array_with_multiple_blocks(two_by_two_vrt):
    """``read_vrt(chunks=(N,N))`` returns a dask-backed DataArray
    whose underlying array has more than one chunk along each spatial
    axis. Before the fix the array was numpy-backed under
    ``result.chunk()``, so this asserts the new lazy graph is in
    play.
    """
    vrt_path, _ = two_by_two_vrt
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert isinstance(result.data, da.Array), (
        f"expected dask Array, got {type(result.data).__name__}"
    )
    # 256 / 64 = 4 blocks per axis.
    assert result.data.numblocks == (4, 4), (
        f"expected 4x4 blocks, got {result.data.numblocks}"
    )


def test_chunks_is_lazy_does_not_call_internal_reader(monkeypatch,
                                                     two_by_two_vrt):
    """Construction-time call count of the internal VRT reader is zero;
    after ``.compute()`` it equals the chunk count.
    """
    vrt_path, _ = two_by_two_vrt

    from xrspatial.geotiff import _vrt as vrt_module

    counter = {'calls': 0}
    real_read = vrt_module.read_vrt

    def counting_read(*args, **kwargs):
        counter['calls'] += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(vrt_module, 'read_vrt', counting_read)

    result = read_vrt(vrt_path, chunks=(64, 64))

    assert counter['calls'] == 0, (
        f"_read_vrt_internal called {counter['calls']} times before "
        f".compute(); the chunked path leaked an eager decode"
    )

    computed = result.compute()
    # 4 row blocks * 4 col blocks = 16 expected decodes.
    assert counter['calls'] == 16, (
        f"expected 16 per-chunk decodes after compute, got {counter['calls']}"
    )
    assert computed.shape == (256, 256)


# ---------------------------------------------------------------------------
# 2. Byte-identical to the eager path.
# ---------------------------------------------------------------------------

def test_chunked_compute_matches_eager(two_by_two_vrt):
    vrt_path, _ = two_by_two_vrt
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(64, 64)).compute()
    assert eager.shape == chunked.shape
    assert np.array_equal(eager.values, chunked.values), (
        "chunked compute diverged from eager read"
    )
    # Coords and key attrs must match too.
    np.testing.assert_array_equal(eager['x'].values, chunked['x'].values)
    np.testing.assert_array_equal(eager['y'].values, chunked['y'].values)
    assert eager.attrs.get('transform') == chunked.attrs.get('transform')
    assert eager.attrs.get('crs') == chunked.attrs.get('crs')


def test_chunked_single_tile_matches_eager(single_tile_vrt):
    """Single-tile VRT (one source) should still match eager when
    chunked. Exercises the path where many chunk windows hit the
    same single source.
    """
    vrt_path, _ = single_tile_vrt
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(32, 32)).compute()
    assert np.array_equal(eager.values, chunked.values)


# ---------------------------------------------------------------------------
# 3. Task-count cap.
# ---------------------------------------------------------------------------

def test_chunks_task_cap_raises(two_by_two_vrt):
    """``chunks=(1, 1)`` on a 256x256 VRT would build 65,536 tasks,
    blowing past the 50,000-task cap. The reader should refuse with
    a ValueError that names ``chunks=`` and suggests a larger size.
    """
    vrt_path, _ = two_by_two_vrt
    with pytest.raises(ValueError, match=r"chunks=.*task"):
        read_vrt(vrt_path, chunks=(1, 1))


# ---------------------------------------------------------------------------
# 4. Window + chunks: chunks tile the window, not the full extent.
# ---------------------------------------------------------------------------

def test_window_plus_chunks_matches_eager(two_by_two_vrt):
    """When both ``window=`` and ``chunks=`` are passed, the dask
    graph must tile the window (not the full VRT extent). The output
    shape and pixel values match an eager windowed read.
    """
    vrt_path, _ = two_by_two_vrt
    window = (32, 48, 160, 192)  # 128 high, 144 wide

    eager = read_vrt(vrt_path, window=window)
    chunked = read_vrt(vrt_path, window=window, chunks=(64, 64))

    assert isinstance(chunked.data, da.Array)
    # The chunk grid is sized off the window extent (128, 144) with
    # chunks=64 => (2, 3) numblocks.
    assert chunked.data.numblocks == (2, 3), (
        f"expected (2, 3) numblocks over the window, got "
        f"{chunked.data.numblocks}"
    )

    computed = chunked.compute()
    assert computed.shape == eager.shape == (128, 144)
    assert np.array_equal(eager.values, computed.values)


# ---------------------------------------------------------------------------
# 5. GPU + chunks: each block is a cupy array.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_gpu_plus_chunks_returns_dask_on_cupy(two_by_two_vrt):
    """``read_vrt(gpu=True, chunks=...)`` must build a dask graph whose
    blocks are cupy-backed (not numpy that gets cupy-wrapped at
    compute time on the host).
    """
    import cupy

    vrt_path, _ = two_by_two_vrt
    result = read_vrt(vrt_path, gpu=True, chunks=(64, 64))

    assert isinstance(result.data, da.Array)
    assert isinstance(result.data._meta, cupy.ndarray), (
        f"expected cupy _meta, got "
        f"{type(result.data._meta).__module__}."
        f"{type(result.data._meta).__name__}"
    )
    computed = result.compute()
    assert isinstance(computed.data, cupy.ndarray)


# ---------------------------------------------------------------------------
# 6. Multi-band VRT + chunks.
# ---------------------------------------------------------------------------

def test_multiband_plus_chunks_preserves_band_dim(multiband_vrt):
    """3-band VRT read with ``chunks=`` keeps the band dimension on
    every block and the assembled DataArray.
    """
    vrt_path, src = multiband_vrt
    result = read_vrt(vrt_path, chunks=(32, 32))

    assert isinstance(result.data, da.Array)
    assert result.dims == ('y', 'x', 'band')
    assert result.shape == (64, 64, 3)
    # Per-block shape on the band axis is 3 (whole band axis in one
    # chunk because we did not pass a band-chunk size).
    assert result.data.chunks[2] == (3,)

    computed = result.compute()
    np.testing.assert_allclose(computed.values, src, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 7. Copilot review: ``attrs['vrt_holes']`` must propagate to the chunked
#    path so users switching from eager to chunked keep the #1734 contract.
# ---------------------------------------------------------------------------

def test_chunked_propagates_vrt_holes_when_source_missing(two_by_two_vrt):
    """When a source referenced by the VRT does not exist on disk and
    the caller opts into the lenient ``missing_sources='warn'`` path,
    the chunked reader must populate ``attrs['vrt_holes']`` with the
    same schema the eager reader uses, so callers can branch on
    ``"vrt_holes" in da.attrs`` regardless of which code path produced
    the DataArray.

    Note: the default ``missing_sources='raise'`` raises at build time
    under #2265, so this test exercises the explicit ``'warn'`` opt-in.
    """
    import warnings
    from xrspatial.geotiff import GeoTIFFFallbackWarning
    from xrspatial.geotiff._reader import _mmap_cache

    vrt_path, _ = two_by_two_vrt
    vrt_dir = os.path.dirname(vrt_path)
    # Remove one of the four source tiles. ``to_geotiff(.vrt, tile_size=128)``
    # writes tile files into a ``<stem>_tiles/`` subdirectory next to the
    # .vrt; walk the tree for any .tif and unlink the first one.
    tile_files = []
    for root, _dirs, files in os.walk(vrt_dir):
        for f in files:
            if f.endswith('.tif'):
                tile_files.append(os.path.join(root, f))
    assert len(tile_files) >= 1
    # write_vrt() opens each tile via _FileSource to read its header;
    # _FileSource.close() decrements the refcount but the mmap stays
    # cached. On Windows an active mmap blocks os.unlink (WinError 32).
    _mmap_cache.clear()
    os.unlink(tile_files[0])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', GeoTIFFFallbackWarning)
        result = read_vrt(
            vrt_path, chunks=(64, 64), missing_sources='warn',
        )

    assert 'vrt_holes' in result.attrs, (
        "chunked path dropped vrt_holes contract from #1734"
    )
    holes = result.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1
    entry = holes[0]
    # Schema parity with the eager path (see read_vrt at ~line 3963).
    assert set(entry.keys()) >= {'source', 'band', 'dst_rect', 'error'}
    assert isinstance(entry['dst_rect'], tuple)
    assert len(entry['dst_rect']) == 4


def test_chunked_no_vrt_holes_attr_when_complete(two_by_two_vrt):
    """When every source is on disk the chunked reader must not set
    ``attrs['vrt_holes']`` (eager parity: empty hole list is omitted).
    """
    vrt_path, _ = two_by_two_vrt
    result = read_vrt(vrt_path, chunks=(64, 64))
    assert 'vrt_holes' not in result.attrs


# ---------------------------------------------------------------------------
# 8. Copilot review: integer source with no declared nodata must keep its
#    integer dtype through the chunked path (no spurious float64 promotion).
# ---------------------------------------------------------------------------

def test_chunked_integer_no_nodata_keeps_source_dtype():
    """A uint16 source with no <NoDataValue> declared must produce a
    uint16 chunked DataArray, not float64. The eager path stays integer
    in this case because its runtime ``mask.any()`` is False; the
    chunked path approximates with a static "any band declares nodata?"
    check, which yields the same answer here.
    """
    arr = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    y = np.linspace(41.0, 40.0, 128)
    x = np.linspace(-106.0, -105.0, 128)
    raster = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1814_uint16_nonodata_')
    tile_path = os.path.join(td, 'tile.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'mosaic.vrt')
    # No ``nodata=`` passed: the VRT will not declare <NoDataValue> for
    # this band, exercising the no-promotion branch.
    _write_vrt_internal(vrt_path, [tile_path])

    result = read_vrt(vrt_path, chunks=(32, 32))
    assert result.dtype == np.uint16, (
        f"expected uint16 (source dtype), got {result.dtype}; "
        f"chunked path promoted to float64 despite no declared nodata"
    )
    computed = result.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.values, arr)
