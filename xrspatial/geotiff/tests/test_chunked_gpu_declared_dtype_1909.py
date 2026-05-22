"""Regression test for issue #1909.

The GDS chunked GPU read path (``_read_geotiff_gpu_chunked_gds``)
declares ``float64`` on the dask graph whenever the source file has an
integer nodata sentinel that round-trips through the source dtype, but
each chunk task returned the raw source dtype if no pixel hit the
sentinel. The result was a silent declared/actual dtype mismatch: the
dask array advertised float64 while ``.compute()`` produced uint16
buffers from chunks where the sentinel didn't appear.

The fix casts each chunk to the declared dtype before returning, gated
on ``arr.dtype != declared_dtype`` so the no-op case skips the
``astype(copy=True)`` allocation (same #1624 optimisation as the CPU
dask path).

These tests drive ``_read_geotiff_gpu_chunked_gds`` directly because
``read_geotiff_gpu(chunks=...)`` only routes to the GDS path when
``_gds_chunk_path_available`` qualifies (tiled file, kvikio present,
etc.); otherwise it falls back to CPU dask + upload, which already
has the correct dtype contract via #1597 and would not exercise the
#1909 fix. The mmap fallback inside ``gpu_decode_tiles_from_file``
keeps the test runnable without KvikIO on CI.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _parse_for_gds(path: str):
    """Return ``(ifd, geo_info, header)`` for the GDS entry point."""
    from xrspatial.geotiff._geotags import extract_geo_info_with_overview_inheritance
    from xrspatial.geotiff._header import parse_all_ifds, parse_header, select_overview_ifd
    from xrspatial.geotiff._reader import _FileSource

    fs = _FileSource(path)
    try:
        raw = fs.read_all()
    finally:
        fs.close()
    header = parse_header(raw)
    ifds = parse_all_ifds(raw, header)
    ifd = select_overview_ifd(ifds, None)
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, raw, header.byte_order,
    )
    return ifd, geo_info, header


@pytest.fixture
def uint16_no_sentinel_path(tmp_path):
    """A tiled uint16 GeoTIFF with declared nodata=9999 and no sentinel pixels."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    path = tmp_path / "uint16_no_sentinel.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999, tile_size=16)
    return str(path)


@pytest.fixture
def uint16_sentinel_hit_path(tmp_path):
    """A tiled uint16 GeoTIFF whose first chunk hits the nodata sentinel."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    arr[0, 0] = 9999
    path = tmp_path / "uint16_sentinel_hit.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999, tile_size=16)
    return str(path)


@pytest.fixture
def uint16_no_nodata_path(tmp_path):
    """A tiled uint16 GeoTIFF with no declared nodata."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    path = tmp_path / "uint16_no_nodata.tif"
    to_geotiff(arr, str(path), compression="none", tile_size=16)
    return str(path)


@_gpu_only
def test_chunked_gpu_declared_dtype_matches_computed(uint16_no_sentinel_path):
    """Declared dask graph dtype must equal the computed chunk dtype."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(uint16_no_sentinel_path)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    declared = da.data.dtype
    computed = da.data.compute()
    assert declared == np.float64
    assert declared == computed.dtype, (
        f"GDS chunked path declared {declared} but computed {computed.dtype}; "
        f"this is the issue #1909 silent declared/actual dtype mismatch."
    )


@_gpu_only
def test_chunked_gpu_dtype_matches_cpu_dask(uint16_no_sentinel_path):
    """The dask+cupy declared dtype must match the dask+numpy path."""
    from xrspatial.geotiff import read_geotiff_dask
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    cpu = read_geotiff_dask(uint16_no_sentinel_path, chunks=4)
    ifd, geo_info, header = _parse_for_gds(uint16_no_sentinel_path)
    gpu = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert cpu.data.dtype == gpu.data.dtype, (
        f"CPU dask declared {cpu.data.dtype} but GPU dask declared "
        f"{gpu.data.dtype}; backends must agree on graph dtype."
    )


@_gpu_only
def test_chunked_gpu_no_nodata_keeps_source_dtype(uint16_no_nodata_path):
    """No nodata declared => declared dtype stays at source dtype."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(uint16_no_nodata_path)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_nodata_path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.uint16
    assert da.data.compute().dtype == np.uint16


@_gpu_only
def test_chunked_gpu_explicit_dtype_kwarg_threads_through(uint16_no_sentinel_path):
    """Explicit dtype= overrides nodata promotion and chunks land in it."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(uint16_no_sentinel_path)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path, ifd, geo_info, header,
        dtype="float32", chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.float32
    assert da.data.compute().dtype == np.float32


@_gpu_only
def test_chunked_gpu_sentinel_hit_still_promotes(uint16_sentinel_hit_path):
    """A chunk that hits the sentinel still NaN-masks and lands in float64."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(uint16_sentinel_hit_path)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_sentinel_hit_path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.float64
    computed = da.data.compute()
    assert computed.dtype == np.float64
    # ``.get()`` brings the cupy array to host before NumPy's ``isnan``.
    host = computed.get()
    assert np.isnan(host[0, 0])


def test_chunked_gpu_eager_paths_keep_source_dtype(uint16_no_sentinel_path):
    """Eager paths (no chunks) keep the source uint16 dtype.

    Pinned to lock the contract: the eager paths only promote when an
    actual sentinel pixel hits; the dask paths always promote when a
    sentinel is declared. Both contracts are valid; the fix only ties
    the GDS chunked path to the dask contract. CPU-only so this runs
    in every CI configuration.
    """
    from xrspatial.geotiff import open_geotiff

    np_da = open_geotiff(uint16_no_sentinel_path)
    assert np_da.dtype == np.uint16
