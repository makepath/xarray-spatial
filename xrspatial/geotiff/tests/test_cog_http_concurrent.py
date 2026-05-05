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

    Tracks both total call count and the maximum observed in-flight
    concurrency so tests can verify the threadpool dispatch directly
    rather than relying on wall-clock timing (which is flaky on busy
    CI runners).
    """

    def __init__(self, per_request_sleep: float = 0.05):
        # Skip super().__init__ -- we're not making real HTTP calls.
        self._url = 'fake://test'
        self._size = None
        self._pool = None
        self._per_request_sleep = per_request_sleep
        self.call_count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        with self._lock:
            self.call_count += 1
            self.in_flight += 1
            if self.in_flight > self.max_in_flight:
                self.max_in_flight = self.in_flight
        try:
            time.sleep(self._per_request_sleep)
            return f'{start}:{length}'.encode('ascii')
        finally:
            with self._lock:
                self.in_flight -= 1


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


def test_read_ranges_dispatches_concurrently():
    """The threadpool should run multiple requests in flight at once.

    Asserting on observed in-flight concurrency is robust to CI scheduler
    jitter; a wall-clock assertion of the same effect is flaky on busy
    runners (the previous version of this test was a 50 ms per-request
    × 20-request setup that occasionally exceeded its 0.5 s budget by a
    few ms on macOS).
    """
    n = 20
    workers = 8
    src = _FakeHTTPSource(per_request_sleep=0.02)
    ranges = [(i * 100, 10) for i in range(n)]

    out = src.read_ranges(ranges, max_workers=workers)

    assert src.call_count == n
    assert len(out) == n
    # Sequential dispatch would peak at 1 in flight. The pool should
    # run several in parallel; require at least 2 (very loose) to keep
    # the test robust on heavily oversubscribed CI runners.
    assert src.max_in_flight >= 2, (
        f'expected >=2 concurrent in-flight calls, '
        f'got max_in_flight={src.max_in_flight}'
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
