"""MinIsWhite photometric handling must be backend-consistent (#1797)."""
from __future__ import annotations

import http.server
import importlib.util
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff

tifffile = pytest.importorskip("tifffile")


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
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
        'RangeHandler1797', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


@pytest.fixture
def miniswhite_http_url(tmp_path, monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    stored = np.array([[0, 1, 2], [10, 128, 255]], dtype=np.uint8)
    path = tmp_path / "tmp_1797_miniswhite.tif"
    tifffile.imwrite(str(path), stored, photometric='miniswhite')
    httpd, port = _serve(path.read_bytes())
    try:
        yield f'http://127.0.0.1:{port}/tmp_1797_miniswhite.tif', stored
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_miniswhite_matches_local_reader(miniswhite_http_url):
    url, stored = miniswhite_http_url

    got = open_geotiff(url)

    np.testing.assert_array_equal(got.values, np.iinfo(stored.dtype).max - stored)


def test_http_dask_miniswhite_matches_local_reader(miniswhite_http_url):
    url, stored = miniswhite_http_url

    got = open_geotiff(url, chunks=2).compute()

    np.testing.assert_array_equal(got.values, np.iinfo(stored.dtype).max - stored)


@_gpu_only
def test_gpu_miniswhite_matches_cpu_reader(tmp_path):
    from xrspatial.geotiff._writer import write

    stored = np.array([[0, 1, 2], [10, 128, 255]], dtype=np.uint8)
    path = str(tmp_path / "tmp_1797_miniswhite_gpu.tif")
    write(stored, path, compression='deflate', tiled=True, tile_size=16,
          photometric='miniswhite')

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    # After #1836 the writer pre-inverts MinIsWhite pixels so the reader's
    # unconditional inversion restores the user-domain values -- the
    # round-trip is the identity for both backends.
    np.testing.assert_array_equal(cpu.values, stored)
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)
