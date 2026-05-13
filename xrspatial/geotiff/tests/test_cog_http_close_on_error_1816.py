"""Regression tests for issue #1816.

``_read_cog_http`` called ``source.close()`` only on the success path:
the fetch/decode step plus the post-processing block (band slice,
orientation, photometric inversion) ran outside any ``try/finally``,
so a raise from any of them skipped the close. ``_HTTPSource.close()``
is currently a no-op (a module-level urllib3 pool, not a per-source
resource), so the leak is latent rather than active, but the structure
needs the guard so a future resource-holding source does not silently
leak.

These tests pin the close-on-error contract by injecting a wrapper
around ``_HTTPSource`` that records every ``close()`` call, then
exercising both the happy path and a tile-fetch failure path.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff import _reader as reader_mod
from xrspatial.geotiff._reader import _read_cog_http
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Loopback HTTP server with Range support (mirrors #1695 pattern).
# ---------------------------------------------------------------------------

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
        'RangeHandler1816', (_RangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/cog.tif', httpd, thread


def _stop(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch):
    """The HTTP source rejects loopback by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# Close-tracking wrapper installed via monkeypatch on the _HTTPSource
# constructor used inside _read_cog_http.
# ---------------------------------------------------------------------------

class _CloseTracker:
    """Delegates every attribute to a real ``_HTTPSource`` while
    recording ``close()`` calls. Used to verify that ``_read_cog_http``
    closes the source on both the success and the failure path.
    """

    def __init__(self, real):
        self._real = real
        self.close_count = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self):
        self.close_count += 1
        return self._real.close()


def _install_tracker(monkeypatch):
    """Replace ``_HTTPSource`` in ``_reader`` with a factory that wraps
    each instance in a ``_CloseTracker`` and stashes the trackers on a
    list so the test can inspect them afterwards.
    """
    trackers: list[_CloseTracker] = []
    real_cls = reader_mod._HTTPSource

    def factory(url, *args, **kwargs):
        tracker = _CloseTracker(real_cls(url, *args, **kwargs))
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(reader_mod, '_HTTPSource', factory)
    return trackers


# ---------------------------------------------------------------------------
# Fixture: a real single-band COG served over loopback.
# ---------------------------------------------------------------------------

@pytest.fixture
def single_band_cog(tmp_path):
    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    path = str(tmp_path / 'tmp_1816_single.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True)
    with open(path, 'rb') as f:
        payload = f.read()
    return path, payload, arr


# ---------------------------------------------------------------------------
# Happy path: close called exactly once after full post-processing.
# ---------------------------------------------------------------------------

def test_http_source_closed_on_success(single_band_cog, monkeypatch):
    """A successful ``_read_cog_http`` closes the source exactly once.

    Establishes the baseline so the failure-path test below isn't just
    catching an unrelated regression in the success path.
    """
    _path, payload, expected = single_band_cog
    trackers = _install_tracker(monkeypatch)
    url, httpd, _ = _serve(payload)
    try:
        arr, _geo = _read_cog_http(url)
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop(httpd)

    assert len(trackers) == 1, (
        f"expected one _HTTPSource construction, got {len(trackers)}")
    assert trackers[0].close_count == 1, (
        f"expected close() called once, got {trackers[0].close_count}")


# ---------------------------------------------------------------------------
# Failure path: tile fetch raises, source still closed.
# ---------------------------------------------------------------------------

def test_http_source_closed_when_tile_fetch_raises(
        single_band_cog, monkeypatch):
    """When ``_fetch_decode_cog_http_tiles`` raises, ``_read_cog_http``
    still closes the source. Before the fix, ``source.close()`` ran
    only on the success path, so any exception in the fetch/decode
    bypassed the close.
    """
    _path, payload, _expected = single_band_cog
    trackers = _install_tracker(monkeypatch)

    def boom(*_args, **_kwargs):
        raise OSError("simulated tile fetch failure")

    monkeypatch.setattr(
        reader_mod, '_fetch_decode_cog_http_tiles', boom)

    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(OSError, match="simulated tile fetch failure"):
            _read_cog_http(url)
    finally:
        _stop(httpd)

    assert len(trackers) == 1
    assert trackers[0].close_count == 1, (
        "source.close() was not called on the exception path; "
        "the try/finally guard in _read_cog_http is missing or broken")


# ---------------------------------------------------------------------------
# Failure path: post-processing (orientation) raises, source still closed.
# ---------------------------------------------------------------------------

def test_http_source_closed_when_post_processing_raises(
        single_band_cog, monkeypatch):
    """An exception from the orientation/photometric step also runs
    through ``finally``. Guards against a future regression that moves
    ``source.close()`` back between the fetch and the post-processing.
    """
    _path, payload, _expected = single_band_cog
    trackers = _install_tracker(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated photometric failure")

    monkeypatch.setattr(
        reader_mod, '_apply_photometric_miniswhite', boom)

    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(RuntimeError, match="simulated photometric"):
            _read_cog_http(url)
    finally:
        _stop(httpd)

    assert len(trackers) == 1
    assert trackers[0].close_count == 1
