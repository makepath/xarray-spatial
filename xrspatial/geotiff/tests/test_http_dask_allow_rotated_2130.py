"""HTTP dask metadata path must honour ``allow_rotated`` (#2130).

Pre-fix, ``read_geotiff_dask`` called ``_parse_cog_http_meta`` without
forwarding ``allow_rotated``, so opening a rotated GeoTIFF over HTTP
with ``chunks=...`` raised ``NotImplementedError`` from the parser
even when the caller had opted in. The local chunked path forwarded
the kwarg correctly, which made the bug a remote-only inconsistency.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff

tifffile = pytest.importorskip("tifffile")


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Range-aware HTTP handler.

    The simple ``SimpleHTTPRequestHandler`` returns the full file body
    for any GET, which the COG HTTP source rejects (it requires a
    proper 206 Partial Content reply). Mirrors the helper used in
    ``test_http_dask_orientation_1794.py``.
    """

    payload: bytes = b''

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            start_s, _, end_s = spec.partition('-')
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}',
            )
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        pass


def _serve(payload: bytes):
    handler_cls = type(
        'RangeHandler2130', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _write_rotated_tiff(path, arr, *, tile=None):
    """Synthesise a TIFF with a rotated ModelTransformationTag (30-deg)."""
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [(34264, 12, 16, m, False)]
    kwargs = {
        'photometric': 'minisblack',
        'planarconfig': 'contig',
        'extratags': extratags,
    }
    if tile is not None:
        kwargs['tile'] = tile
    tifffile.imwrite(str(path), arr, **kwargs)


def test_http_dask_rotated_default_raises(tmp_path, monkeypatch):
    """Without ``allow_rotated`` the HTTP dask path must still raise."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    src = tmp_path / "tmp_2130_http_dask_default.tif"
    arr = np.arange(64, dtype='<u2').reshape(8, 8)
    _write_rotated_tiff(src, arr, tile=(16, 16))
    payload = src.read_bytes()
    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/{src.name}'
        with pytest.raises(NotImplementedError, match="rotation"):
            open_geotiff(url, chunks=4)
    finally:
        httpd.shutdown()


def test_http_dask_rotated_allow_rotated_reads(tmp_path, monkeypatch):
    """``allow_rotated=True`` over HTTP+dask reads the pixel grid.

    Pre-#2130 this raised ``NotImplementedError`` because
    ``read_geotiff_dask`` did not forward the kwarg to
    ``_parse_cog_http_meta``.
    """
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    src = tmp_path / "tmp_2130_http_dask_optin.tif"
    arr = np.arange(64, dtype='<u2').reshape(8, 8)
    _write_rotated_tiff(src, arr, tile=(16, 16))
    payload = src.read_bytes()
    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/{src.name}'
        da = open_geotiff(url, allow_rotated=True, chunks=4)
        assert da.shape == arr.shape
        np.testing.assert_array_equal(da.compute().values, arr)
    finally:
        httpd.shutdown()
