"""Chunked remote reads honour external `.ovr` sidecars (issue #2239).

The eager local/fsspec reader in ``xrspatial.geotiff._reader`` already
appends sidecar IFDs onto the pyramid list before ``select_overview_ifd``
runs, so ``overview_level=1`` against a GDAL external-overview file
resolves into the sidecar. Before this fix:

* ``xrspatial.geotiff._backends.dask`` went straight through
  ``_parse_cog_http_meta`` for HTTP and fsspec sources and skipped the
  sidecar lookup, so ``open_geotiff(remote, chunks=..., overview_level=1)``
  raised "overview_level out of range" or quietly resolved to a
  different overview than the eager open of the same URL / URI.
* ``xrspatial.geotiff._read_geo_info`` had the same fsspec bypass.
* The eager HTTP path (``_read_cog_http``) also never looked for a
  sidecar, so HTTP eager + HTTP chunked were both wrong in the same way.

These tests cover the parity contract end to end: an HTTP-mocked source
and an fsspec ``memory://``-backed source both resolve sidecar overview
levels chunked the same way they resolve them eagerly, and the
``_read_geo_info`` metadata-only path reports the sidecar overview's
dimensions for fsspec inputs.
"""
from __future__ import annotations

import io
import pathlib
import shutil

import numpy as np
import pytest


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
# HTTP server helper -- serves the base TIFF and its sidecar payload
# with Range support. The chunked HTTP reader needs Range responses
# (the simpler SimpleHTTPRequestHandler returns the whole body on a
# range request, which the reader rejects to avoid silent offset bugs).
# ---------------------------------------------------------------------------
def _start_range_http_server(payloads: dict[str, bytes]):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Class-level mapping so ``do_GET`` resolves the requested path
        # without poking at the server. The handler class is built fresh
        # per call so each server instance has its own payloads dict.
        payloads: dict[str, bytes] = {}

        def log_message(self, *a, **kw):
            return  # silence

        def do_GET(self):  # noqa: N802
            path = self.path
            if path not in self.payloads:
                self.send_response(404)
                self.end_headers()
                return
            body = self.payloads[path]
            rng = self.headers.get('Range')
            if rng and rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/{len(body)}',
                )
                self.send_header('Content-Length', str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    handler_cls = type(
        'RangeHandler2239', (_Handler,), {'payloads': payloads}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


@pytest.fixture
def _http_with_sidecar(monkeypatch):
    """Serve the bundled base + sidecar pair with Range support."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    base_bytes = src.read_bytes()
    side_bytes = (src.parent / (src.name + ".ovr")).read_bytes()
    payloads = {
        "/x.tif": base_bytes,
        "/x.tif.ovr": side_bytes,
    }
    httpd, port = _start_range_http_server(payloads)
    try:
        yield f"http://127.0.0.1:{port}/x.tif"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def _fsspec_memory_with_sidecar():
    """Stage the bundled base + sidecar pair into an fsspec memory store."""
    fsspec = pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    fs = fsspec.filesystem("memory")
    # Use a unique-ish key per test session to avoid collisions when
    # pytest-xdist runs this test in parallel with other sidecar specs.
    base_uri = "memory://issue2239/x.tif"
    side_uri = base_uri + ".ovr"
    with open(src, "rb") as f:
        fs.pipe_file(base_uri.replace("memory://", ""), f.read())
    with open(str(src) + ".ovr", "rb") as f:
        fs.pipe_file(side_uri.replace("memory://", ""), f.read())
    try:
        yield base_uri
    finally:
        # Best-effort cleanup so a repeat invocation in the same
        # interpreter sees a clean store.
        for p in (base_uri, side_uri):
            try:
                fs.rm_file(p.replace("memory://", ""))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Eager-vs-chunked overview parity for fsspec sources. Before the fix
# the chunked open went through ``_read_geo_info``'s fsspec bypass and
# never saw the sidecar -- so ``overview_level=1`` failed with an
# "overview_level out of range" error while the eager open succeeded.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overview_level", [1, 2])
def test_fsspec_chunked_open_resolves_sidecar_overview(
        _fsspec_memory_with_sidecar, overview_level):
    from xrspatial.geotiff import open_geotiff
    uri = _fsspec_memory_with_sidecar
    eager = open_geotiff(uri, overview_level=overview_level)
    chunked = open_geotiff(uri, chunks=16,
                           overview_level=overview_level)
    assert chunked.shape == eager.shape
    np.testing.assert_array_equal(chunked.values, eager.values)


def test_fsspec_chunked_open_base_level_unchanged(
        _fsspec_memory_with_sidecar):
    """Base level (no sidecar in play) keeps its prior dimensions and pixels."""
    from xrspatial.geotiff import open_geotiff
    uri = _fsspec_memory_with_sidecar
    eager = open_geotiff(uri)
    chunked = open_geotiff(uri, chunks=16)
    assert chunked.shape == eager.shape == (64, 64)
    np.testing.assert_array_equal(chunked.values, eager.values)


# ---------------------------------------------------------------------------
# Eager-vs-chunked overview parity for HTTP sources. Same contract as the
# fsspec test, but exercises the HTTP discovery + load + tile-fetch path
# in ``_read_cog_http`` and ``_backends/dask.py``.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overview_level", [1, 2])
def test_http_chunked_open_resolves_sidecar_overview(
        _http_with_sidecar, overview_level):
    from xrspatial.geotiff import open_geotiff
    url = _http_with_sidecar
    eager = open_geotiff(url, overview_level=overview_level)
    chunked = open_geotiff(url, chunks=16,
                           overview_level=overview_level)
    assert chunked.shape == eager.shape
    np.testing.assert_array_equal(chunked.values, eager.values)


def test_http_eager_reads_sidecar_overview(_http_with_sidecar):
    """The eager HTTP path also needs to honour sidecars (issue #2239)."""
    from xrspatial.geotiff import open_geotiff
    url = _http_with_sidecar
    # The bundled fixture's sidecar carries two overview levels at 32x32
    # and 16x16. Before the fix both raised "overview_level out of range"
    # over HTTP because ``_read_cog_http`` only saw the in-file IFD chain.
    da32 = open_geotiff(url, overview_level=1)
    da16 = open_geotiff(url, overview_level=2)
    assert da32.shape == (32, 32)
    assert da16.shape == (16, 16)


def test_http_eager_vs_local_parity(_http_with_sidecar):
    """Eager HTTP reads should match the eager local read byte-for-byte."""
    from xrspatial.geotiff import open_geotiff
    url = _http_with_sidecar
    src = _fixture_or_skip()
    for level in (0, 1, 2):
        http_da = open_geotiff(url, overview_level=level)
        local_da = open_geotiff(str(src), overview_level=level)
        assert http_da.shape == local_da.shape
        np.testing.assert_array_equal(http_da.values, local_da.values)


# ---------------------------------------------------------------------------
# ``_read_geo_info`` metadata-only path (the function the dask graph
# builder uses for local sources, exercised here against fsspec to pin
# the sidecar branch). Reports the right IFD dimensions and ``geo_info``
# for the sidecar overview without downloading pixels.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overview_level,expected", [
    (1, (32, 32)),
    (2, (16, 16)),
])
def test_read_geo_info_fsspec_reports_sidecar_dimensions(
        _fsspec_memory_with_sidecar, overview_level, expected):
    from xrspatial.geotiff import _read_geo_info
    uri = _fsspec_memory_with_sidecar
    geo_info, h, w, dtype, n_bands = _read_geo_info(
        uri, overview_level=overview_level,
    )
    assert (h, w) == expected
    # The base file is single-band uint16; reported dtype / band count
    # should match regardless of which overview we landed on.
    assert dtype == np.dtype('uint16')
    assert n_bands == 0  # single-band convention


# ---------------------------------------------------------------------------
# Out-of-range guard: requesting an overview level beyond what the
# merged pyramid offers should raise with a clear message rather than
# silently swallow the request (issue #2239 contract).
# ---------------------------------------------------------------------------
def test_fsspec_chunked_open_rejects_overview_past_sidecar(
        _fsspec_memory_with_sidecar):
    from xrspatial.geotiff import open_geotiff
    uri = _fsspec_memory_with_sidecar
    # Base + two sidecar levels => max valid level is 2.
    with pytest.raises(ValueError, match="overview_level"):
        open_geotiff(uri, chunks=16, overview_level=3)


def test_http_chunked_open_rejects_overview_past_sidecar(_http_with_sidecar):
    from xrspatial.geotiff import open_geotiff
    url = _http_with_sidecar
    with pytest.raises(ValueError, match="overview_level"):
        open_geotiff(url, chunks=16, overview_level=3)


# ---------------------------------------------------------------------------
# Backstop: file-like buffers (``io.BytesIO``) have no sidecar concept;
# the new HTTP/fsspec discovery must not regress the file-like path.
# ---------------------------------------------------------------------------
def test_file_like_chunked_open_unaffected_by_sidecar_discovery():
    from xrspatial.geotiff import open_geotiff
    src = _fixture_or_skip()
    # ``open_geotiff(io.BytesIO, chunks=...)`` is not supported (the
    # dispatcher rejects file-like + chunks), so verify the eager path
    # still returns the base-level image without trying to probe for
    # a sidecar.
    with open(src, "rb") as f:
        buf = io.BytesIO(f.read())
    da = open_geotiff(buf)
    assert da.shape == (64, 64)
