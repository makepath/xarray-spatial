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
        # Post #2264 ``read_range`` rejects on the Content-Length
        # preflight before any body bytes are read; pre-#2264 the
        # ``_validate_range_response`` step rejected on
        # Content-Range/range-fetch grounds after the body was already
        # buffered. Both wordings prove the request was refused.
        with pytest.raises(
                OSError,
                match="Content-Range|Content-Length|range fetch"):
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


def test_read_range_zero_length_returns_empty_without_request():
    """``read_range(start, 0)`` (and negative ``length``) must short-
    circuit to ``b''`` before any HTTP request goes on the wire.

    Without the guard, ``Range: bytes=<start>-<start-1>`` is sent, which
    is an invalid range and either trips a 416 from a well-behaved
    server or pulls down arbitrarily large bytes from a misbehaving one.
    Other source implementations (e.g. ``_BytesIOSource``) already
    follow the ``b''``-on-non-positive-length convention; this test
    pins that contract for ``_HTTPSource`` too.
    """
    hit_count = {'n': 0}

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            # If we ever land here, the guard failed. Record the hit so
            # the assertion below points at the right cause.
            hit_count['n'] += 1
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        assert src.read_range(10, 0) == b''
        assert src.read_range(0, 0) == b''
        assert src.read_range(10, -5) == b''
        assert hit_count['n'] == 0, (
            "read_range(length<=0) should not issue an HTTP request"
        )
    finally:
        _stop(httpd)


def test_range_ignored_200_oversize_rejected_via_content_length(
        monkeypatch):
    """Server ignores ``Range`` for ``start=0`` and returns a 200 with
    a ``Content-Length`` past the full-object slack cap.

    Before #2264, ``read_range`` buffered the entire body into
    ``resp.data`` (urllib3 default ``preload_content=True``) and then
    sliced down to ``length``. That defeated the OOM guard the slice
    comment claimed: a 16 KiB prefetch against a 2 GiB body still
    pulled 2 GiB into memory before the slice ran. The fix caps the
    fallback at :attr:`_HTTPSource._RANGE_IGNORED_FULL_OBJECT_CAP` and
    rejects on the ``Content-Length`` preflight before any body bytes
    are read.

    Drop the cap to a small value here so the test does not have to
    stand up a multi-MiB payload to trigger rejection.
    """
    monkeypatch.setattr(
        _HTTPSource, '_RANGE_IGNORED_FULL_OBJECT_CAP', 1024)

    class _Handler(_BaseHandler):
        # Payload larger than the patched cap so the preflight has
        # something to reject.
        payload = b'\xab' * 4096

        def do_GET(self):  # noqa: N802
            # Ignore Range entirely; return the full object as 200.
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="Content-Length|byte budget"):
            src.read_range(0, 64)
    finally:
        _stop(httpd)


def test_range_ignored_200_full_object_sliced_within_cap():
    """Server ignores ``Range`` for ``start=0`` and returns the full
    object as 200 with no ``Content-Range``. When the body fits
    inside the full-object slack cap, ``read_range`` slices it down
    to the requested length.

    This is the legitimate small-file fallback: the caller asked for
    a 64-byte prefetch, the file is a few KiB, and the server doesn't
    honour Range. Pre-#2264 the slice happened after the whole body
    was already in ``resp.data``; post-#2264 the body is bounded by
    the streaming cap on the wire.
    """

    class _Handler(_BaseHandler):
        payload = b'\xcd' * 4096

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(0, 64)
        # Caller's "at most length bytes" contract holds even when the
        # server returned a much larger body.
        assert out == _Handler.payload[:64]
        assert len(out) == 64
    finally:
        _stop(httpd)


def test_range_ignored_200_short_body_returned_as_is():
    """A 200 fallback whose body is smaller than the requested length
    is returned unchanged (no slicing needed).

    This is the "tiny file served by a Range-blind origin" case: the
    caller asked for a 64-byte header prefetch but the whole object
    is only 40 bytes.
    """

    class _Handler(_BaseHandler):
        payload = b'\xef' * 40

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(0, 64)
        assert out == _Handler.payload
        assert len(out) == 40
    finally:
        _stop(httpd)


def test_range_ignored_200_no_content_length_is_streamed_and_capped(
        monkeypatch):
    """Server omits ``Content-Length`` and streams a body larger than
    the full-object slack cap. ``_read_capped`` must abort once more
    than the cap has arrived, so the body never gets fully buffered
    into Python memory.

    This is the second half of the #2264 fix: the ``Content-Length``
    preflight catches honest oversize, the streaming cap (via chunked
    transfer encoding here, since the server omits ``Content-Length``)
    catches the case where the server volunteers no advertised size.

    Drop the full-object cap to a small value to keep the test fast.
    """
    monkeypatch.setattr(
        _HTTPSource, '_RANGE_IGNORED_FULL_OBJECT_CAP', 2048)

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            # No Content-Length; use chunked transfer encoding.
            self.send_response(200)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            # Each chunk is 1024 bytes; send 8 of them (8192 total),
            # past the 2048-byte patched cap.
            chunk = b'\xee' * 1024
            chunk_header = f'{len(chunk):x}\r\n'.encode()
            for _ in range(8):
                self.wfile.write(chunk_header)
                self.wfile.write(chunk)
                self.wfile.write(b'\r\n')
            self.wfile.write(b'0\r\n\r\n')

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="byte budget|exceeded"):
            src.read_range(0, 64)
    finally:
        _stop(httpd)


def test_range_request_uses_streaming_response(monkeypatch):
    """``read_range`` must request the body with ``preload_content=
    False`` so urllib3 hands back a streaming response instead of
    buffering ``resp.data`` up front.

    This pins the wire-level behaviour the OOM fix depends on. If a
    future refactor flips the default back to ``preload_content=
    True``, the streaming cap and the ``Content-Length`` preflight
    both become advisory rather than enforcing. Issue #2264.
    """

    captured: dict = {}

    class _FakeResp:
        def __init__(self, body):
            self.status = 206
            self._body = body
            self.headers = {
                'Content-Length': str(len(body)),
                'Content-Range': f'bytes 0-{len(body) - 1}/64',
            }

        def stream(self, amt=65536, decode_content=True):
            if self._body:
                yield self._body

        def release_conn(self):
            pass

    class _FakePool:
        def request(self, method, url, headers=None, timeout=None,
                    redirect=None, preload_content=True):
            captured['preload_content'] = preload_content
            captured['headers'] = headers
            return _FakeResp(b'\x01' * 16)

    src = _HTTPSource('http://127.0.0.1:65535/x.bin')
    monkeypatch.setattr(src, '_pool', _FakePool())
    out = src.read_range(0, 16)
    assert out == b'\x01' * 16
    # The hard contract: the GET went out asking for a streaming body.
    assert captured['preload_content'] is False
    assert captured['headers'] == {'Range': 'bytes=0-15'}
