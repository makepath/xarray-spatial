"""Regression tests for issue #1735.

``_HTTPSource.read_range`` previously returned the response body
without checking the status code, the ``Content-Range`` header, or the
returned byte length. Three failure modes slipped through silently:

- a 200 (Range ignored) or a 4xx/5xx body was handed to the caller as
  if it were the requested range,
- a ``Content-Range`` header pointing at a different byte range was
  trusted as the requested one,
- a truncated response was passed to a downstream codec where the
  decode error appeared far from the real cause.

These tests stand up tiny loopback HTTP servers that misbehave in each
of those ways and assert that ``read_range`` raises a clear ``OSError``.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

from xrspatial.geotiff._reader import _HTTPSource


class _BaseHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b'0' * 64

    def log_message(self, *_args, **_kwargs):
        pass


def _serve(handler_cls):
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/x.bin', httpd, thread


def _stop(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


def test_range_request_ignored_for_nonzero_start_raises():
    """Server ignores ``Range`` for a non-zero start and returns full
    200 -> OSError. (A 200 with start=0 is harmless because the body
    offsets line up with what the caller wanted.)"""

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            # Ignore Range header; return the full object as 200.
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="Content-Range|range fetch"):
            src.read_range(8, 16)
    finally:
        _stop(httpd)


def test_range_request_wrong_content_range_raises():
    """Server returns 206 but the Content-Range header points at the
    wrong bytes -> OSError."""

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            # Pretend we sent bytes 4-19/64 regardless of what was asked.
            self.send_response(206)
            self.send_header('Content-Length', '16')
            self.send_header('Content-Range', 'bytes 4-19/64')
            self.end_headers()
            self.wfile.write(self.payload[4:20])

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        # Caller asks for 0-15; server says 4-19.
        with pytest.raises(OSError, match="Content-Range"):
            src.read_range(0, 16)
    finally:
        _stop(httpd)


def test_range_request_short_body_raises():
    """Server returns 206 with a body shorter than the requested
    length -> OSError."""

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(206)
            self.send_header('Content-Length', '4')
            self.send_header('Content-Range', 'bytes 0-15/64')
            self.end_headers()
            # Send only 4 bytes despite advertising a 16-byte range.
            self.wfile.write(self.payload[:4])

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="length"):
            src.read_range(0, 16)
    finally:
        _stop(httpd)


def test_range_request_well_formed_succeeds():
    """A correctly-formed 206 response is accepted as-is."""

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            rng = self.headers.get('Range', '')
            spec = rng[len('bytes='):]
            s, _, e = spec.partition('-')
            start = int(s)
            end = int(e)
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Length', str(len(chunk)))
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/'
                f'{len(self.payload)}',
            )
            self.end_headers()
            self.wfile.write(chunk)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(8, 16)
        assert out == _BaseHandler.payload[8:24]
        assert len(out) == 16
    finally:
        _stop(httpd)
