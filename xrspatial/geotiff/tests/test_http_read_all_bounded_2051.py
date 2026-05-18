"""Regression tests for issue #2051.

``_HTTPSource.read_all()`` used to pull the full HTTP body unconditionally:
no ``Content-Length`` check, no streaming cap. A TIFF whose header
declares a tiny raster (which sails past ``_check_dimensions``) could
still be served as a multi-gigabyte body and the whole thing landed in
memory before TIFF parsing got a chance to reject anything.

These tests stand up tiny loopback HTTP servers that misbehave in three
ways:

- declared ``Content-Length`` exceeds the byte budget,
- ``Content-Length`` lies (says small, sends big),
- ``Content-Length`` is omitted entirely (chunked transfer encoding).

Plus a positive test that legitimate full-image reads still work, and
unit tests for the ``_compute_full_image_byte_budget`` helper.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff._reader import (
    _HTTPSource,
    _compute_full_image_byte_budget,
    _FULL_IMAGE_BUDGET_HEADER_SLACK,
    _read_cog_http,
)
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

class _BaseHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b''
    # Subclasses override these to fake misbehaviour.
    lie_content_length: int | None = None
    drop_content_length: bool = False
    truncated_payload: bytes | None = None

    def log_message(self, *_args, **_kwargs):
        pass


def _serve(handler_cls):
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
    # Loopback addresses are blocked by the SSRF allow-list; the escape
    # hatch lets the test reach 127.0.0.1.
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# Unit tests for the budget helper
# ---------------------------------------------------------------------------

def test_budget_uses_max_strip_end_plus_slack():
    """Budget is ``max(offset + byte_count) + slack`` over the strip table."""
    offsets = [1024, 5000, 100_000]
    byte_counts = [512, 1024, 4096]
    budget = _compute_full_image_byte_budget(offsets, byte_counts)
    # Largest end is 100_000 + 4096 = 104_096
    assert budget == 104_096 + _FULL_IMAGE_BUDGET_HEADER_SLACK


def test_budget_empty_strip_table_falls_back_to_per_strip_cap():
    """Empty / missing strip table falls back to the per-strip safety cap."""
    budget = _compute_full_image_byte_budget(None, None)
    assert budget > 0
    budget_empty = _compute_full_image_byte_budget([], [])
    assert budget_empty > 0


def test_budget_all_sparse_falls_back_to_per_strip_cap():
    """A strip table where every strip is sparse (byte_count=0 and
    offset=0) is degenerate; the helper falls back rather than picking
    a useless cap of zero."""
    offsets = [0, 0, 0]
    byte_counts = [0, 0, 0]
    budget = _compute_full_image_byte_budget(offsets, byte_counts)
    # Falls back to per-strip cap + slack, not 0.
    assert budget > _FULL_IMAGE_BUDGET_HEADER_SLACK


# ---------------------------------------------------------------------------
# read_all with a byte budget
# ---------------------------------------------------------------------------

def test_read_all_no_budget_returns_full_body():
    """Without ``max_bytes`` the legacy unbounded behaviour is preserved."""

    class _Handler(_BaseHandler):
        payload = b'A' * 1024

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        data = src.read_all()
        assert data == b'A' * 1024
    finally:
        _stop(httpd)


def test_read_all_rejects_oversized_content_length():
    """Server advertises a Content-Length larger than the budget --
    rejected up front via OSError before any body is read."""

    class _Handler(_BaseHandler):
        payload = b'B' * 2048

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="Content-Length"):
            src.read_all(max_bytes=1024)
    finally:
        _stop(httpd)


def test_read_all_truncates_when_server_lies_about_content_length_small():
    """Server lies low: advertises a small Content-Length but sends a
    much larger body. urllib3 trusts the advertised length and truncates
    at the byte count the server declared, so the client is already
    protected -- the extra bytes never reach Python memory. The cap is
    irrelevant on this path because the body the caller sees never
    exceeds the (truthful or lying) Content-Length. Lock in the
    truncation behaviour so a future urllib3 / stdlib change does not
    quietly turn this back into a vector."""

    class _Handler(_BaseHandler):
        # 100 KiB body, but advertised as 100 bytes.
        big_body = b'L' * (100 * 1024)
        lied_length = 100

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(self.lied_length))
            self.end_headers()
            self.wfile.write(self.big_body)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        # Budget is 1024 bytes, server says 100 -> pre-flight passes.
        # The body returned is the 100 bytes the server claimed, not the
        # 100 KiB it tried to send.
        data = src.read_all(max_bytes=1024)
        assert len(data) <= 100, (
            f"Got {len(data)} bytes from a server that advertised 100; "
            f"the HTTP client failed to truncate at Content-Length and "
            f"the byte budget did not catch the over-shoot."
        )
    finally:
        _stop(httpd)


def test_read_all_catches_missing_content_length():
    """Server omits Content-Length and uses chunked transfer encoding.
    The pre-flight check has nothing to look at; the streaming cap must
    still catch the over-sized body."""

    class _Handler(_BaseHandler):
        def do_GET(self):  # noqa: N802
            body = b'C' * (100 * 1024)
            self.send_response(200)
            # No Content-Length header at all.
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            # Send as a single chunk.
            self.wfile.write(f'{len(body):x}\r\n'.encode('ascii'))
            self.wfile.write(body)
            self.wfile.write(b'\r\n0\r\n\r\n')

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="exceeded the byte budget"):
            src.read_all(max_bytes=1024)
    finally:
        _stop(httpd)


def test_read_all_passes_when_body_fits_budget():
    """Legitimate path: body equals the budget exactly, returns cleanly."""

    class _Handler(_BaseHandler):
        payload = b'D' * 1024

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve(_Handler)
    try:
        src = _HTTPSource(url)
        data = src.read_all(max_bytes=2048)
        assert data == b'D' * 1024
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# End-to-end COG read
# ---------------------------------------------------------------------------

class _RangeHandler(_BaseHandler):
    """Honours Range requests; serves the full body on a no-Range GET."""

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            start_s, _, end_s = spec.partition('-')
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}',
            )
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)


def _serve_payload(payload: bytes):
    handler_cls = type(
        'BoundRangeHandler', (_RangeHandler,), {'payload': payload}
    )
    return _serve(handler_cls)


def test_full_image_http_read_still_works_for_legitimate_cog(tmp_path):
    """Sanity: with the cap in place, a normal stripped COG still reads
    cleanly end-to-end. The strip-table-derived budget is loose enough
    to cover the real on-wire body."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'legit_2051.tif')
    # Stripped (not tiled) to exercise the strips path. ``cog=True``
    # writes COG-friendly tag ordering but stripped layout is the
    # default for non-tiled writes.
    write(arr, path, compression='deflate', tiled=False)

    with open(path, 'rb') as f:
        payload = f.read()

    url, httpd, _ = _serve_payload(payload)
    try:
        result, _geo = _read_cog_http(url)
        np.testing.assert_array_equal(result, arr)
    finally:
        _stop(httpd)


def test_full_image_http_read_rejects_padded_body(tmp_path):
    """Attack scenario: a legitimate TIFF header is followed by gigabytes
    of garbage. The strip-table-derived budget rejects the body before
    it can be allocated."""
    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    path = str(tmp_path / 'padded_2051.tif')
    write(arr, path, compression='deflate', tiled=False)

    with open(path, 'rb') as f:
        legit_payload = f.read()

    # Append 64 MiB of zeros to the body. The strip table only covers
    # the first len(legit_payload) bytes; anything past max(offset +
    # byte_count) + slack is over-budget.
    bloated = legit_payload + (b'\x00' * (64 * 1024 * 1024))

    url, httpd, _ = _serve_payload(bloated)
    try:
        with pytest.raises(OSError, match="Content-Length|byte budget"):
            _read_cog_http(url)
    finally:
        _stop(httpd)
