"""``stable_only=True`` must reject advanced-tier HTTP / fsspec sources.

Regression coverage for issue #2821. ``reader.http`` and
``reader.fsspec`` sit at the ``advanced`` tier in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES`, so a caller who asks for
stable-only sources via ``stable_only=True`` must not be served from a
remote source. Before the fix the gate only fired on the VRT dispatch:
the eager non-VRT path called ``read_to_array`` directly and the dask
path only forwarded ``stable_only`` to ``read_vrt`` for ``.vrt`` paths,
so an HTTP / fsspec source slipped through unchecked.

The tests push a small valid (stable-codec, local-style) GeoTIFF into
the fsspec ``memory://`` filesystem, which ships with fsspec and runs in
plain CI without credentials. ``open_geotiff('memory://...')`` walks the
same ``_is_fsspec_uri`` routing an ``s3://`` / ``gs://`` URL would, so
the gate fires before any read. HTTP URLs are checked against the local
file source as the negative control.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, RemoteStableSourcesOnlyError,
                               open_geotiff, read_geotiff_dask, read_geotiff_gpu)
from xrspatial.geotiff.tests._helpers.tiff_builders import make_minimal_tiff

fsspec = pytest.importorskip("fsspec")


def _gpu_available() -> bool:
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


_MEMORY_URL = "memory:///stable_only_2821/sample.tif"


@pytest.fixture
def memory_tiff_url():
    """Push a stable-codec GeoTIFF into fsspec memory; yield its URL.

    The memory filesystem is a process-global singleton, so wipe the
    sample before and after each test to avoid cross-test leakage.
    """
    payload = make_minimal_tiff(
        4, 4, np.dtype("float32"),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )
    fs = fsspec.filesystem("memory")
    with fs.open(_MEMORY_URL, "wb") as handle:
        handle.write(payload)
    try:
        yield _MEMORY_URL
    finally:
        if fs.exists(_MEMORY_URL):
            fs.rm(_MEMORY_URL)


@pytest.fixture
def local_tiff_path(tmp_path):
    """Write a stable-codec GeoTIFF to a local file; yield its path."""
    payload = make_minimal_tiff(
        4, 4, np.dtype("float32"),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )
    path = tmp_path / "stable_only_2821_local.tif"
    path.write_bytes(payload)
    return str(path)


def _assert_remote_stable_error(excinfo) -> None:
    """The rejection names the unlocking opt-in and cites the contract."""
    msg = str(excinfo.value)
    assert "stable_only" in msg or "allow_experimental_codecs" in msg, msg
    assert "release_gate_geotiff.rst" in msg or "#2342" in msg, msg


def test_eager_fsspec_source_rejected_under_stable_only(memory_tiff_url):
    """Eager non-VRT read of an fsspec source rejects ``stable_only=True``."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(memory_tiff_url, stable_only=True)
    _assert_remote_stable_error(excinfo)


def test_dask_fsspec_source_rejected_under_stable_only(memory_tiff_url):
    """Dask non-VRT read of an fsspec source rejects ``stable_only=True``."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(memory_tiff_url, stable_only=True, chunks=2)
    _assert_remote_stable_error(excinfo)


def test_dask_direct_fsspec_source_rejected_under_stable_only(memory_tiff_url):
    """The direct ``read_geotiff_dask`` entry point gates remote sources too."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        read_geotiff_dask(memory_tiff_url, chunks=2, stable_only=True)
    _assert_remote_stable_error(excinfo)


def test_remote_stable_error_is_ambiguous_metadata_subclass():
    """The error stays catchable as the shared stable-only family."""
    assert issubclass(RemoteStableSourcesOnlyError, GeoTIFFAmbiguousMetadataError)


def test_http_source_rejected_under_stable_only():
    """An ``http://`` source is gated before any network fetch.

    The gate runs ahead of the SSRF / fetch machinery, so no real
    request is made; the URL never resolves.
    """
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff("http://example.invalid/sample.tif", stable_only=True)
    _assert_remote_stable_error(excinfo)


def test_fsspec_source_allowed_with_experimental_optin(memory_tiff_url):
    """``allow_experimental_codecs=True`` unlocks the advanced tier."""
    result = open_geotiff(
        memory_tiff_url, stable_only=True, allow_experimental_codecs=True,
    )
    assert result.shape == (4, 4)


def test_fsspec_source_allowed_without_stable_only(memory_tiff_url):
    """The default (``stable_only=False``) keeps the legacy behaviour."""
    result = open_geotiff(memory_tiff_url)
    assert result.shape == (4, 4)


def test_local_eager_source_succeeds_under_stable_only(local_tiff_path):
    """A plain local-file eager read still works under ``stable_only=True``."""
    result = open_geotiff(local_tiff_path, stable_only=True)
    assert result.shape == (4, 4)


def test_local_dask_source_succeeds_under_stable_only(local_tiff_path):
    """A plain local-file dask read still works under ``stable_only=True``."""
    result = open_geotiff(local_tiff_path, stable_only=True, chunks=2)
    assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Direct GPU entry point (issue #2867)
#
# ``read_geotiff_gpu`` is a public direct reader that routes HTTP / fsspec
# sources through the CPU fallback, so it must apply the same remote gate as
# ``open_geotiff`` and ``read_geotiff_dask``. The gate runs before the cupy
# import and the CUDA preflight, so the rejection tests run on a CPU-only
# machine -- no GPU required. Only the unlock test, which performs a real
# read, needs cupy + CUDA.
# ---------------------------------------------------------------------------


def test_gpu_direct_fsspec_source_rejected_under_stable_only(memory_tiff_url):
    """The direct ``read_geotiff_gpu`` entry point gates remote sources too."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        read_geotiff_gpu(memory_tiff_url, stable_only=True)
    _assert_remote_stable_error(excinfo)


def test_gpu_direct_http_source_rejected_under_stable_only():
    """An ``http://`` source is gated on the GPU path before any fetch."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        read_geotiff_gpu("http://example.invalid/sample.tif", stable_only=True)
    _assert_remote_stable_error(excinfo)


@_gpu_only
def test_gpu_direct_fsspec_source_allowed_with_experimental_optin(memory_tiff_url):
    """``allow_experimental_codecs=True`` unlocks the advanced tier on GPU."""
    result = read_geotiff_gpu(
        memory_tiff_url, stable_only=True, allow_experimental_codecs=True,
    )
    assert result.shape == (4, 4)
