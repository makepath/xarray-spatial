"""Tests for concurrent tile fetching in _read_cog_http (issue #1480)."""
from __future__ import annotations

import http.server
import socketserver
import threading
import time

import numpy as np
import pytest

from xrspatial.geotiff._reader import (
    _HTTPSource,
    _read_cog_http,
    read_to_array,
)
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# read_ranges: ordering and concurrency
# ---------------------------------------------------------------------------

class _FakeHTTPSource(_HTTPSource):
    """_HTTPSource that fakes read_range with a configurable sleep.

    Used to exercise the threadpool path without a real network. Each
    call sleeps for ``per_request_sleep`` seconds and then returns
    deterministic bytes encoding (start, length) so callers can verify
    ordering.
    """

    def __init__(self, per_request_sleep: float = 0.05):
        # Skip super().__init__ -- we're not making real HTTP calls.
        self._url = 'fake://test'
        self._size = None
        self._pool = None
        self._per_request_sleep = per_request_sleep
        self.call_count = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        with self._lock:
            self.call_count += 1
        time.sleep(self._per_request_sleep)
        return f'{start}:{length}'.encode('ascii')


def test_read_ranges_returns_results_in_input_order():
    src = _FakeHTTPSource(per_request_sleep=0.0)
    ranges = [(0, 10), (100, 5), (50, 20), (200, 7)]
    out = src.read_ranges(ranges, max_workers=4)
    assert len(out) == len(ranges)
    for (start, length), data in zip(ranges, out):
        assert data == f'{start}:{length}'.encode('ascii')


def test_read_ranges_empty_list():
    src = _FakeHTTPSource(per_request_sleep=0.0)
    assert src.read_ranges([]) == []


def test_read_ranges_single_request_skips_pool():
    src = _FakeHTTPSource(per_request_sleep=0.0)
    out = src.read_ranges([(42, 8)], max_workers=8)
    assert out == [b'42:8']
    assert src.call_count == 1


def test_read_ranges_concurrency_masks_latency():
    """N concurrent requests should finish faster than N sequential ones.

    The check is intentionally loose (factor 0.5) to avoid flakiness on
    busy CI nodes, but it's tight enough to fail if the implementation
    accidentally serializes.
    """
    n = 20
    per_req = 0.05  # 50 ms each
    src = _FakeHTTPSource(per_request_sleep=per_req)
    ranges = [(i * 100, 10) for i in range(n)]

    t0 = time.perf_counter()
    out = src.read_ranges(ranges, max_workers=8)
    t_total = time.perf_counter() - t0

    assert src.call_count == n
    assert len(out) == n
    # Sequential would be n * per_req. Require at least 2x speedup.
    assert t_total < n * per_req * 0.5, (
        f'expected concurrent fetch to be <{n * per_req * 0.5:.2f}s, '
        f'got {t_total:.2f}s'
    )


# ---------------------------------------------------------------------------
# _read_cog_http: correctness via local http.server
# ---------------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serve a single in-memory bytes payload with HTTP Range support."""

    payload: bytes = b''

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            # Single range only -- matches what _HTTPSource sends.
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
        # Silence the default access log during tests.
        pass


@pytest.fixture
def cog_http_server(tmp_path):
    """Spin up a local http.server serving a tiled COG, yield (url, arr)."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'tmp_1480_cog.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True, overview_levels=[1])

    with open(path, 'rb') as f:
        payload = f.read()

    handler_cls = type(
        'RangeHandler1480', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        yield f'http://127.0.0.1:{port}/cog.tif', arr
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cog_http_round_trip_matches_local_read(cog_http_server):
    url, expected = cog_http_server
    result, _ = _read_cog_http(url)
    np.testing.assert_array_equal(result, expected)


def test_read_to_array_dispatches_to_http(cog_http_server):
    url, expected = cog_http_server
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)
