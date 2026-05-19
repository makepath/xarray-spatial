"""External `.tif.ovr` sidecar overview reader (issue #2112).

Before the fix, opening a GDAL/rasterio file whose overview pyramid
lives in a sibling ``.tif.ovr`` worked at the base level but raised an
``overview_level out of range`` error for any non-zero level. The
reader only walked the in-file IFD chain.

These tests pin the new contract:

* The base IFD continues to read at level 0 with no behaviour change.
* The reader appends sidecar IFDs onto the pyramid list so that
  ``overview_level=1`` (and onwards) reads from the sidecar.
* Byte parity holds against rasterio for the bundled fixture, which
  exercises a uint16 raster with two sidecar levels at factors 2 and 4.
* Discovery is local-file-only: file-like sources, HTTP / fsspec URIs,
  and missing sidecars all fall back to base-only behaviour.
* Sidecar levels reach all eager-read entry points: ``open_geotiff``,
  ``read_to_array``, and the dask graph builder that goes through
  ``_read_to_array_metadata_only``.
"""
from __future__ import annotations

import io
import pathlib
import shutil

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._sidecar import find_sidecar, load_sidecar


_FIXTURE = (
    pathlib.Path(__file__).resolve().parent
    / "golden_corpus"
    / "fixtures"
    / "overview_external_ovr_uint16.tif"
)


def _fixture_or_skip():
    if not _FIXTURE.exists():
        pytest.skip("sidecar fixture not present")
    if not (_FIXTURE.parent / "overview_external_ovr_uint16.tif.ovr").exists():
        pytest.skip("sidecar .ovr file not present")
    return _FIXTURE


# ---------------------------------------------------------------------------
# Discovery helper: local file paths only.
# ---------------------------------------------------------------------------
def test_find_sidecar_returns_path_for_local_file_with_sidecar():
    src = str(_fixture_or_skip())
    assert find_sidecar(src) == src + ".ovr"


def test_find_sidecar_returns_none_when_sidecar_missing(tmp_path):
    src = tmp_path / "no_sidecar_2112.tif"
    src.write_bytes(b"\x00")
    assert find_sidecar(str(src)) is None


def test_find_sidecar_returns_none_for_file_like_object():
    assert find_sidecar(io.BytesIO(b"")) is None


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/path.tif",
        "gs://bucket/path.tif",
        "az://container/path.tif",
        "http://example.com/x.tif",
        "https://example.com/x.tif",
        "memory://buffer.tif",
    ],
)
def test_find_sidecar_returns_none_for_remote_uri(uri):
    assert find_sidecar(uri) is None


# ---------------------------------------------------------------------------
# Sidecar parser: returns a usable IFD list.
# ---------------------------------------------------------------------------
def test_load_sidecar_returns_two_ifds():
    src = _fixture_or_skip()
    sidecar = load_sidecar(str(src) + ".ovr")
    try:
        # The bundled fixture was written with two overview factors (2, 4).
        assert len(sidecar.ifds) == 2
        # First sidecar IFD is the 32x32 level, second is 16x16.
        assert (sidecar.ifds[0].width, sidecar.ifds[0].height) == (32, 32)
        assert (sidecar.ifds[1].width, sidecar.ifds[1].height) == (16, 16)
    finally:
        sidecar.data.close()


# ---------------------------------------------------------------------------
# End-to-end: each sidecar level reads through open_geotiff.
# ---------------------------------------------------------------------------
def test_open_geotiff_base_level_unchanged():
    src = _fixture_or_skip()
    da = open_geotiff(str(src))
    assert da.shape == (64, 64)
    assert da.dtype == np.uint16


def test_open_geotiff_sidecar_level_1():
    src = _fixture_or_skip()
    da = open_geotiff(str(src), overview_level=1)
    assert da.shape == (32, 32)
    assert da.dtype == np.uint16


def test_open_geotiff_sidecar_level_2():
    src = _fixture_or_skip()
    da = open_geotiff(str(src), overview_level=2)
    assert da.shape == (16, 16)
    assert da.dtype == np.uint16


def test_open_geotiff_out_of_range_after_sidecar_appended():
    src = _fixture_or_skip()
    with pytest.raises(ValueError, match="overview_level=3 is out of range"):
        open_geotiff(str(src), overview_level=3)


# ---------------------------------------------------------------------------
# Byte parity with rasterio: the production reference for sidecar overviews.
# ---------------------------------------------------------------------------
def test_sidecar_level_1_matches_rasterio():
    rasterio = pytest.importorskip("rasterio")
    src = _fixture_or_skip()
    with rasterio.open(str(src)) as ds:
        factor = ds.overviews(1)[0]
        out_shape = (ds.height // factor, ds.width // factor)
        rio_arr = ds.read(1, out_shape=out_shape)
    xr_arr = open_geotiff(str(src), overview_level=1).values
    np.testing.assert_array_equal(xr_arr, rio_arr)


def test_sidecar_level_2_matches_rasterio():
    rasterio = pytest.importorskip("rasterio")
    src = _fixture_or_skip()
    with rasterio.open(str(src)) as ds:
        factor = ds.overviews(1)[1]
        out_shape = (ds.height // factor, ds.width // factor)
        rio_arr = ds.read(1, out_shape=out_shape)
    xr_arr = open_geotiff(str(src), overview_level=2).values
    np.testing.assert_array_equal(xr_arr, rio_arr)


# ---------------------------------------------------------------------------
# Sidecar reaches the reader entry point and the metadata-only helper.
# ---------------------------------------------------------------------------
def test_read_to_array_sidecar_level_1(tmp_path):
    src = _fixture_or_skip()
    arr, geo = read_to_array(str(src), overview_level=1)
    assert arr.shape == (32, 32)
    assert arr.dtype == np.uint16


def test_metadata_only_includes_sidecar_levels(tmp_path):
    src = _fixture_or_skip()
    from xrspatial.geotiff import _read_geo_info

    geo, h, w, dtype, _ = _read_geo_info(
        str(src), overview_level=2)
    assert (h, w) == (16, 16)
    # GeoTIFF tags inherit from the base IFD, so the metadata is non-empty.
    assert geo is not None


# ---------------------------------------------------------------------------
# Missing sidecar gracefully falls back to base-only behaviour.
# ---------------------------------------------------------------------------
def test_missing_sidecar_raises_overview_out_of_range(tmp_path):
    src = _fixture_or_skip()
    # Copy only the base file; the sidecar stays behind.
    copy_path = tmp_path / "no_sidecar_2112.tif"
    shutil.copy(src, copy_path)
    # Base level still works.
    assert open_geotiff(str(copy_path)).shape == (64, 64)
    # Asking for an overview level now fails the same way it did before
    # sidecar support was added.
    with pytest.raises(ValueError, match="out of range"):
        open_geotiff(str(copy_path), overview_level=1)


# ---------------------------------------------------------------------------
# File-like buffer source still works for the base level (no sidecar lookup).
# ---------------------------------------------------------------------------
@pytest.fixture
def _gpu_or_skip():
    if not _gpu_available():
        pytest.skip("cupy + CUDA required")


def _gpu_available() -> bool:
    import importlib.util
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def test_gpu_eager_reads_sidecar_level_1(_gpu_or_skip):
    src = _fixture_or_skip()
    cpu = open_geotiff(str(src), overview_level=1)
    gpu = open_geotiff(str(src), overview_level=1, gpu=True)
    assert gpu.shape == cpu.shape
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)


def test_gpu_eager_reads_sidecar_level_2(_gpu_or_skip):
    src = _fixture_or_skip()
    cpu = open_geotiff(str(src), overview_level=2)
    gpu = open_geotiff(str(src), overview_level=2, gpu=True)
    assert gpu.shape == cpu.shape
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)


def test_gpu_eager_base_level_unchanged(_gpu_or_skip):
    src = _fixture_or_skip()
    gpu = open_geotiff(str(src), gpu=True)
    assert gpu.shape == (64, 64)


# ---------------------------------------------------------------------------
# HTTP sidecar discovery: tests use a tiny local HTTP server so the
# probe and download paths are exercised without a network round-trip.
# ---------------------------------------------------------------------------
def _start_http_server(directory):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw):
            return  # silence

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


def test_find_sidecar_http_probe_returns_url_when_present(
        tmp_path, monkeypatch):
    # The sidecar probe now routes through ``_HTTPSource``, which
    # rejects loopback hostnames under the SSRF guard added in #1664.
    # Loopback is the standard local-server pattern in this repo's HTTP
    # tests (see ``test_golden_corpus_http_1930.py``); opt into the
    # escape hatch the production reader exposes.
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    import shutil
    shutil.copy(src, tmp_path / "x.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) == url + ".ovr"
    finally:
        httpd.shutdown()


def test_find_sidecar_http_probe_returns_none_when_missing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    import shutil
    shutil.copy(src, tmp_path / "x.tif")  # no .ovr copied
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) is None
    finally:
        httpd.shutdown()


def test_find_sidecar_http_probe_rejects_loopback_without_env_override(
        tmp_path):
    """Without ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1``, the SSRF
    guard makes a loopback probe silently return ``None`` -- same
    silent-fail-to-base contract the rest of ``find_sidecar`` uses."""
    src = _fixture_or_skip()
    import shutil
    shutil.copy(src, tmp_path / "x.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) is None
    finally:
        httpd.shutdown()


def test_load_sidecar_http_returns_ifds(tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    import shutil
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif.ovr"
        sidecar = load_sidecar(url)
        assert len(sidecar.ifds) == 2
        assert (sidecar.ifds[0].width, sidecar.ifds[0].height) == (32, 32)
        assert (sidecar.ifds[1].width, sidecar.ifds[1].height) == (16, 16)
    finally:
        httpd.shutdown()


def test_find_sidecar_fsspec_probe_returns_uri_when_present(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    import shutil
    shutil.copy(src, tmp_path / "y.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "y.tif.ovr")
    # file:// is a valid fsspec scheme that uses LocalFileSystem.
    uri = f"file://{tmp_path}/y.tif"
    assert find_sidecar(uri) == uri + ".ovr"


def test_file_like_source_reads_base_without_sidecar():
    src = _fixture_or_skip()
    with open(src, "rb") as f:
        buf = io.BytesIO(f.read())
    da = open_geotiff(buf)
    assert da.shape == (64, 64)
