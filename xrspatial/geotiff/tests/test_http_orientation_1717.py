"""HTTP COG full reads must honour TIFF Orientation tag (274).

Issue #1717: ``_read_cog_http`` skipped ``_apply_orientation`` on the
full-read branch, so opening the same oriented file locally vs over HTTP
returned different pixel orders. This is a backend parity break.

These tests open the same Orientation-tagged TIFF via both paths and
assert the returned array and geo transform agree, for every value of
the tag (1-8). The existing rejection of windowed reads + non-default
orientation must still raise.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff._reader import _read_cog_http, read_to_array

tifffile = pytest.importorskip("tifffile")


_ORIENTATIONS = [1, 2, 3, 4, 5, 6, 7, 8]


def _write_with_orientation(path, arr, orientation):
    tifffile.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serve a single in-memory bytes payload with HTTP Range support."""

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
        'RangeHandler1717', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


@pytest.fixture
def _allow_loopback(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_http_full_read_matches_local_for_orientation(
    tmp_path, _allow_loopback, orientation,
):
    """Local-file vs HTTP full read must produce identical output."""
    rng = np.random.default_rng(orientation)
    arr = rng.integers(0, 255, size=(12, 16), dtype=np.uint8)
    path = tmp_path / f"tmp_1717_orient_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    with open(path, 'rb') as f:
        payload = f.read()

    arr_local, geo_local = read_to_array(str(path))

    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/orient_{orientation}.tif'
        arr_http, geo_http = _read_cog_http(url)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert arr_http.shape == arr_local.shape, (
        f"orientation={orientation}: HTTP shape {arr_http.shape} != "
        f"local shape {arr_local.shape}"
    )
    np.testing.assert_array_equal(
        arr_http, arr_local,
        err_msg=f"orientation={orientation}: HTTP pixels differ from local",
    )
    assert geo_http.transform == geo_local.transform, (
        f"orientation={orientation}: transform mismatch "
        f"http={geo_http.transform} local={geo_local.transform}"
    )


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_http_windowed_read_rejects_non_default_orientation(
    tmp_path, _allow_loopback, orientation,
):
    """Windowed reads against an oriented file should still raise.

    Mirrors the local-path guard so the contract is uniform across
    backends. Resolving windowed-read semantics for oriented files is
    out of scope for #1717.
    """
    arr = np.zeros((8, 8), dtype=np.uint8)
    path = tmp_path / f"tmp_1717_window_reject_{orientation}.tif"
    _write_with_orientation(path, arr, orientation)

    with open(path, 'rb') as f:
        payload = f.read()

    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/window_{orientation}.tif'
        with pytest.raises(ValueError, match="Orientation tag"):
            _read_cog_http(url, window=(0, 0, 4, 4))
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_default_orientation_still_works(tmp_path, _allow_loopback):
    """Sanity: orientation=1 (default) HTTP read is byte-identical to local."""
    arr = np.arange(48, dtype=np.uint8).reshape(6, 8)
    path = tmp_path / "tmp_1717_default.tif"
    _write_with_orientation(path, arr, 1)

    with open(path, 'rb') as f:
        payload = f.read()

    arr_local, _ = read_to_array(str(path))
    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/default.tif'
        arr_http, _ = _read_cog_http(url)
    finally:
        httpd.shutdown()
        httpd.server_close()

    np.testing.assert_array_equal(arr_http, arr_local)
    np.testing.assert_array_equal(arr_http, arr)
