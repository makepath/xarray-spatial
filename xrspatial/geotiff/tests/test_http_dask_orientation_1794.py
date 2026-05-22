"""Remote dask reads must not bypass TIFF Orientation handling (#1794)."""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff

tifffile = pytest.importorskip("tifffile")


def _write_with_orientation(path, arr, orientation):
    tifffile.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


class _RangeHandler(http.server.BaseHTTPRequestHandler):
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
        'RangeHandler1794', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def test_http_dask_read_rejects_non_default_orientation(tmp_path, monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "tmp_1794_orientation_2.tif"
    _write_with_orientation(path, arr, 2)

    payload = path.read_bytes()
    httpd, port = _serve(payload)
    try:
        url = f'http://127.0.0.1:{port}/tmp_1794_orientation_2.tif'
        with pytest.raises(ValueError, match="Orientation tag"):
            open_geotiff(url, chunks=4)
    finally:
        httpd.shutdown()
        httpd.server_close()
