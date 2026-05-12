"""Tests for HTTP COG metadata prefetch growing past 64 KiB (issue #1718).

The fast path is a single 16 KiB GET; if the IFD chain or its out-of-line
tag arrays (TileOffsets, GeoAsciiParams, GDAL_METADATA) extend past that
window the prefetch buffer doubles until everything fits or it hits
:data:`MAX_HTTP_HEADER_BYTES`.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import (
    INITIAL_HTTP_HEADER_BYTES,
    MAX_HTTP_HEADER_BYTES,
    _HTTPSource,
    _parse_cog_http_meta,
    _read_cog_http,
)
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _InMemoryHTTPSource(_HTTPSource):
    """_HTTPSource backed by an in-memory bytes buffer.

    Counts ``read_range`` calls so tests can lock in the fast-path
    invariant (one GET for headers that fit in 16 KiB).
    """

    def __init__(self, payload: bytes):
        # Skip super().__init__ -- no network, no SSRF validation needed.
        self._url = 'memory://test'
        self._size = len(payload)
        self._pool = None
        self._payload = payload
        self.read_range_calls: list[tuple[int, int]] = []

    def read_range(self, start: int, length: int) -> bytes:
        self.read_range_calls.append((start, length))
        return self._payload[start:start + length]


class _RangeHandler1718(http.server.BaseHTTPRequestHandler):
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
        'RangeHandler1718Bound', (_RangeHandler1718,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _write_cog_with_big_metadata(path: str, arr: np.ndarray,
                                 metadata_pad_bytes: int) -> None:
    """Write a multi-overview COG whose level-0 IFD carries a huge
    GDAL_METADATA tag, pushing the chained overview IFDs past 64 KiB."""
    # GDAL_METADATA is stored as an out-of-line ASCII tag value when
    # large; a multi-kilobyte payload pads the value area between the
    # first IFD and its overviews, forcing the rest of the chain past
    # the 16 KiB / 64 KiB prefetch windows.
    big_xml = (
        '<GDALMetadata>'
        + '<Item name="filler">' + 'x' * metadata_pad_bytes + '</Item>'
        + '</GDALMetadata>'
    )
    write(arr, path, compression='deflate', tiled=True, tile_size=64,
          cog=True, overview_levels=[2, 4, 8],
          gdal_metadata_xml=big_xml)


# ---------------------------------------------------------------------------
# Fast path: small COG should fire a single 16 KiB read
# ---------------------------------------------------------------------------

def test_small_cog_uses_single_initial_read(tmp_path):
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'small_1718_cog.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=32,
          cog=True, overview_levels=[1])

    with open(path, 'rb') as f:
        payload = f.read()

    src = _InMemoryHTTPSource(payload)
    header, ifd, geo_info, header_bytes = _parse_cog_http_meta(src)

    # Fast path is exactly one read_range at the initial size.
    assert len(src.read_range_calls) == 1
    assert src.read_range_calls[0] == (0, INITIAL_HTTP_HEADER_BYTES)
    # And the buffer fully resolves the chain.
    parsed_ifds = parse_all_ifds(header_bytes, header)
    assert parsed_ifds[-1].next_ifd_offset == 0


# ---------------------------------------------------------------------------
# Grow path: COG whose IFD chain extends past 64 KiB still parses
# ---------------------------------------------------------------------------

def test_ifd_chain_past_64kib_resolves(tmp_path):
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    path = str(tmp_path / 'big_meta_1718_cog.tif')
    # 96 KiB of XML padding guarantees subsequent IFDs land well past
    # both the 16 KiB initial fetch and the legacy 64 KiB retry.
    _write_cog_with_big_metadata(path, arr, metadata_pad_bytes=96 * 1024)

    with open(path, 'rb') as f:
        payload = f.read()

    # Sanity: the second IFD's offset really does sit past 64 KiB,
    # otherwise this test is not exercising the grow loop.
    header = parse_header(payload)
    full_ifds = parse_all_ifds(payload, header)
    assert len(full_ifds) >= 2, "fixture must have >=2 IFDs"
    assert full_ifds[0].next_ifd_offset > 65536, (
        "fixture must place IFD #2 past 64 KiB to exercise the grow loop; "
        f"got next_ifd_offset={full_ifds[0].next_ifd_offset}"
    )

    src = _InMemoryHTTPSource(payload)
    _, _, _, header_bytes = _parse_cog_http_meta(src)

    grown_ifds = parse_all_ifds(header_bytes, header)
    assert len(grown_ifds) == len(full_ifds), (
        f"prefetch buffer lost IFDs: got {len(grown_ifds)} of {len(full_ifds)}"
    )
    # Multiple read_range calls confirm the buffer actually grew.
    assert len(src.read_range_calls) > 1
    # And it did not blow past the cap.
    assert src.read_range_calls[-1][1] <= MAX_HTTP_HEADER_BYTES


def test_end_to_end_http_read_with_big_metadata(tmp_path, monkeypatch):
    """_read_cog_http should match local read on a >64 KiB IFD-chain COG."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    path = str(tmp_path / 'http_big_1718_cog.tif')
    _write_cog_with_big_metadata(path, arr, metadata_pad_bytes=96 * 1024)

    with open(path, 'rb') as f:
        payload = f.read()

    httpd, _thread = _serve(payload)
    port = httpd.server_address[1]
    try:
        url = f'http://127.0.0.1:{port}/cog.tif'
        result, _geo = _read_cog_http(url)
        np.testing.assert_array_equal(result, arr)

        # Overview read on the same URL must also succeed.
        result_ov, _ = _read_cog_http(url, overview_level=1)
        assert result_ov.shape[0] < arr.shape[0]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Truncation / cap behaviour
# ---------------------------------------------------------------------------

def test_cap_raises_clear_error_on_excessive_chain(monkeypatch):
    """When the IFD chain refuses to fit, hitting the cap raises ValueError.

    Patches MAX_HTTP_HEADER_BYTES tiny so the test does not need to
    fabricate a multi-megabyte payload to exercise the cap branch.
    """
    from xrspatial.geotiff import _reader

    # Build a payload whose first IFD's next-IFD offset deliberately
    # points to a huge address we will never reach. parse_all_ifds will
    # return the first IFD but tail_next > buffer, forcing the grow loop.
    # The payload itself is small so the server EOF branch is not what
    # raises -- we want the cap branch.
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    # In-memory write
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='_cap_1718.tif', delete=False) as f:
        path = f.name
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True, overview_levels=[1])
    with open(path, 'rb') as f:
        payload = bytearray(f.read())

    header = parse_header(bytes(payload))
    ifds = parse_all_ifds(bytes(payload), header)
    assert len(ifds) >= 2

    # Locate the first IFD's next_ifd_offset slot and rewrite it to a
    # far-off value that no buffer growth will ever satisfy.
    bo = header.byte_order
    first_ifd_off = header.first_ifd_offset
    import struct as _struct
    num_entries = _struct.unpack_from(f'{bo}H', payload, first_ifd_off)[0]
    next_off_pos = first_ifd_off + 2 + num_entries * 12
    far = 10**12  # 1 TB, well past any cap
    _struct.pack_into(f'{bo}I', payload, next_off_pos, far & 0xFFFFFFFF)

    # Shrink the cap so the test is fast.
    monkeypatch.setattr(_reader, 'MAX_HTTP_HEADER_BYTES', 64 * 1024)

    src = _InMemoryHTTPSource(bytes(payload))
    # Wrap read_range so requests past EOF still return the same length
    # we already returned (mimics an HTTPS server returning the full
    # file when asked for more). Without this the EOF branch short-
    # circuits before the cap branch fires.
    real_read = src.read_range

    def padded_read(start, length):
        data = real_read(start, length)
        if len(data) < length:
            # Pretend the file is longer than it is by zero-padding,
            # so the grow loop keeps growing until it hits the cap.
            data = data + b'\x00' * (length - len(data))
        return data

    src.read_range = padded_read  # type: ignore[assignment]

    with pytest.raises(ValueError, match='MAX_HTTP_HEADER_BYTES'):
        _parse_cog_http_meta(src)
