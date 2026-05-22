"""Mixed-endian remote sidecar reads decode with the sidecar's byte order
(issue #2314).

``_parse_cog_http_meta`` discovers a sibling ``.tif.ovr`` and merges its
IFDs onto the pyramid list, but historically returned the base file's
``TIFFHeader`` even when the selected IFD came from the sidecar. The
HTTP eager and dask paths then handed that base header to
``_decode_strip_or_tile``, which uses ``header.byte_order`` to interpret
the pixel bytes. A big-endian ``.ovr`` paired with a little-endian base
(or the reverse) would scramble every multi-byte sample.

This module pins the parity contract for the remote paths: when the
selected overview lives in the sidecar, the decode step has to use the
sidecar's own byte order. The local sidecar path in ``_reader.py``
already swaps to the sidecar's header (``_reader.py:223``); these tests
make the HTTP and fsspec dask paths reach the same parity.

Each test compares the remote read of a mixed-endian pair against the
expected values written by ``tifffile.imwrite`` so a regression cannot
hide behind "both paths agree" -- the byte values are checked against
the source array, not just against another (possibly equally broken)
read.
"""
from __future__ import annotations

import io
import os
import uuid

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# Fixture builders. Each call writes a fresh base + sidecar pair so
# parallel runs (pytest-xdist) do not race on a shared path. Filenames
# include the issue number so a stray collision is easy to trace.
# ---------------------------------------------------------------------------
def _mixed_endian_pair(tmp_path, base_byteorder: str, sidecar_byteorder: str,
                       *, tile: int | None = None):
    """Return (base_path, sidecar_path, base_arr, sidecar_arr).

    The two arrays are deliberately different so a wrong-endian decode
    on the sidecar cannot accidentally compare equal to the base array.
    Distinct multipliers also keep the byte patterns visually different
    when a failure dump shows the raw values.

    ``tile`` selects the on-disk layout: ``None`` writes stripped TIFFs
    (tifffile's default), an int writes tiled TIFFs with the given
    square tile size. The stripped layout exercises
    ``_fetch_decode_cog_http_strips`` and the tiled layout exercises
    ``_fetch_decode_cog_http_tiles``; both paths read
    ``header.byte_order`` and benefit from the sidecar header swap.
    """
    base_arr = (np.arange(64 * 64, dtype=np.uint16) * 7).reshape(64, 64)
    sidecar_arr = (np.arange(32 * 32, dtype=np.uint16) * 13).reshape(32, 32)
    suffix = uuid.uuid4().hex[:8]
    base_path = tmp_path / f"mixedendian_2314_{suffix}.tif"
    sidecar_path = tmp_path / f"mixedendian_2314_{suffix}.tif.ovr"
    write_kwargs = {}
    if tile is not None:
        write_kwargs['tile'] = (tile, tile)
    tifffile.imwrite(str(base_path), base_arr,
                     byteorder=base_byteorder, compression=None,
                     **write_kwargs)
    tifffile.imwrite(str(sidecar_path), sidecar_arr,
                     byteorder=sidecar_byteorder, compression=None,
                     **write_kwargs)
    return base_path, sidecar_path, base_arr, sidecar_arr


# ---------------------------------------------------------------------------
# HTTP server helper. Reused pattern from ``test_remote_sidecar_chunked_2239``:
# a tiny TCP server that supports Range so the COG reader's per-tile
# range GETs work the same as against a real CDN. Plain
# ``SimpleHTTPRequestHandler`` returns the whole body on a range request
# which the reader rejects.
# ---------------------------------------------------------------------------
def _start_range_http_server(payloads: dict[str, bytes]):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        payloads: dict[str, bytes] = {}

        def log_message(self, *a, **kw):
            return

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
        'RangeHandler2314', (_Handler,), {'payloads': payloads}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


# ---------------------------------------------------------------------------
# Local sanity check. The local sidecar reader already handles this case
# (``_reader.py:223`` swaps to the sidecar's header). Pin it here so a
# regression on the local path is caught alongside the remote ones; the
# remote tests below borrow the same fixture builder.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_local_eager_mixed_endian_sidecar(tmp_path, base_bo, side_bo):
    """The local reader already handles mixed-endian sidecars."""
    from xrspatial.geotiff import open_geotiff
    base_path, _, base_arr, side_arr = _mixed_endian_pair(
        tmp_path, base_bo, side_bo)
    r0 = open_geotiff(str(base_path), overview_level=0)
    np.testing.assert_array_equal(r0.values, base_arr)
    r1 = open_geotiff(str(base_path), overview_level=1)
    np.testing.assert_array_equal(r1.values, side_arr)


# ---------------------------------------------------------------------------
# Eager HTTP path. Pre-fix this scrambled the sidecar bytes because
# ``_read_cog_http`` passed the base file's header straight into
# ``_fetch_decode_cog_http_tiles`` after swapping the byte source to the
# sidecar URL. The fix returns the sidecar's header from
# ``_parse_cog_http_meta`` when ``used_sidecar=True``.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_eager_mixed_endian_sidecar(tmp_path, monkeypatch,
                                         base_bo, side_bo):
    from xrspatial.geotiff import open_geotiff
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        r0 = open_geotiff(url, overview_level=0)
        np.testing.assert_array_equal(r0.values, base_arr)
        r1 = open_geotiff(url, overview_level=1)
        np.testing.assert_array_equal(r1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Dask HTTP path. Pre-fix this set ``http_meta = (http_header, http_ifd)``
# where ``http_header`` was always the base header; per-chunk decoding
# then used the wrong byte order on every chunk. The fix flows through
# ``_parse_cog_http_meta``: when the sidecar is in use, ``http_header``
# is the sidecar's header.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_chunked_mixed_endian_sidecar(tmp_path, monkeypatch,
                                           base_bo, side_bo):
    from xrspatial.geotiff import open_geotiff
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        # ``chunks`` forces the dask metadata path through
        # ``_backends/dask.py``'s ``_parse_cog_http_meta`` branch and
        # builds a per-chunk graph that hits the per-chunk decode step.
        chunked0 = open_geotiff(url, overview_level=0, chunks=16)
        np.testing.assert_array_equal(chunked0.values, base_arr)
        chunked1 = open_geotiff(url, overview_level=1, chunks=16)
        np.testing.assert_array_equal(chunked1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# fsspec dask path. The fsspec branch of ``_backends/dask.py`` goes
# through the same ``_parse_cog_http_meta`` call, so the same header
# swap covers it. ``memory://`` stages the payload bytes in process and
# bypasses the HTTP server entirely, which makes the fsspec path easy
# to test without DNS / port assumptions.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_fsspec_chunked_mixed_endian_sidecar(tmp_path, base_bo, side_bo):
    fsspec = pytest.importorskip("fsspec")
    from xrspatial.geotiff import open_geotiff
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair(
        tmp_path, base_bo, side_bo)
    fs = fsspec.filesystem("memory")
    # memory:// is process-global; collide-proof the key.
    key = f"issue2314-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    base_uri = f"memory://{key}/x.tif"
    side_uri = base_uri + ".ovr"
    fs.pipe_file(base_uri.replace("memory://", ""), base_path.read_bytes())
    fs.pipe_file(side_uri.replace("memory://", ""), sidecar_path.read_bytes())
    try:
        chunked0 = open_geotiff(base_uri, overview_level=0, chunks=16)
        np.testing.assert_array_equal(chunked0.values, base_arr)
        chunked1 = open_geotiff(base_uri, overview_level=1, chunks=16)
        np.testing.assert_array_equal(chunked1.values, side_arr)
    finally:
        for p in (base_uri, side_uri):
            try:
                fs.rm_file(p.replace("memory://", ""))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tiled-layout variant. The stripped path above already exercises the
# header swap end to end, but the tiled HTTP path is the more common
# COG shape in the wild and reads ``header.byte_order`` from a different
# call site (``_fetch_decode_cog_http_tiles``'s ``_decode_one`` closure).
# Run one parametrized case here so a future regression that decouples
# the two paths gets caught.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_eager_mixed_endian_sidecar_tiled(tmp_path, monkeypatch,
                                               base_bo, side_bo):
    from xrspatial.geotiff import open_geotiff
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair(
        tmp_path, base_bo, side_bo, tile=16)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        r0 = open_geotiff(url, overview_level=0)
        np.testing.assert_array_equal(r0.values, base_arr)
        r1 = open_geotiff(url, overview_level=1)
        np.testing.assert_array_equal(r1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Direct ``_parse_cog_http_meta`` contract test. Asserts the returned
# header's byte order matches the sidecar when ``used_sidecar=True`` and
# the base when ``used_sidecar=False``. Without this, the test fleet
# would have to assert byte-order parity only indirectly via decoded
# pixel values; a future refactor that re-routes byte_order through a
# different field could silently regress without tripping the
# end-to-end tests above.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_parse_cog_http_meta_returns_sidecar_header(tmp_path, monkeypatch,
                                                    base_bo, side_bo):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    from xrspatial.geotiff._cog_http import _parse_cog_http_meta
    from xrspatial.geotiff._reader import _HTTPSource
    from xrspatial.geotiff._sidecar import close_sidecar
    base_path, sidecar_path, _, _ = _mixed_endian_pair(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"

        # Level 0: base IFD. Header should be the base file's header.
        src = _HTTPSource(url)
        try:
            hdr, ifd, _, _, sidecar_meta = _parse_cog_http_meta(
                src, overview_level=0, source_path=url, return_sidecar=True)
        finally:
            src.close()
        sidecar, _, used = sidecar_meta
        try:
            assert used is False
            assert hdr.byte_order == base_bo
        finally:
            close_sidecar(sidecar)

        # Level 1: sidecar IFD. Header should switch to the sidecar's
        # byte order so downstream decode honours the right endianness.
        src = _HTTPSource(url)
        try:
            hdr, ifd, _, _, sidecar_meta = _parse_cog_http_meta(
                src, overview_level=1, source_path=url, return_sidecar=True)
        finally:
            src.close()
        sidecar, _, used = sidecar_meta
        try:
            assert used is True
            assert hdr.byte_order == side_bo
        finally:
            close_sidecar(sidecar)
    finally:
        httpd.shutdown()
        httpd.server_close()
