"""HTTP COG read parity for ``window``, ``band``, and ``PlanarConfiguration=2``.

Issue #1669: ``open_geotiff(url, window=..., band=...)`` silently dropped
both kwargs on the HTTP branch. The local path honoured them. The HTTP
tile-index loop also ignored ``PlanarConfiguration=2`` so separate-plane
COGs fetched the wrong byte ranges.

These tests build a tiled COG on disk, serve it over a loopback
``http.server`` with HTTP Range support, and compare the HTTP read
against the local read pixel-for-pixel for several combinations:

* windowed read
* band-selected read of a multi-band COG
* window + band combined
* ``PlanarConfiguration=2`` tiled COG, full read
* ``PlanarConfiguration=2`` tiled COG, windowed read

Per PR #1680 review feedback, none of these fixtures rely on the
optional ``tifffile`` dependency. The single-band and multi-band
planar=1 fixtures use the project's own writer (``write``). The
planar=2 fixture is built by hand from TIFF bytes (the xrspatial
writer only emits planar=1) so the planar=2 HTTP logic is still
exercised in the default test environment.
"""
from __future__ import annotations

import http.server
import math
import socketserver
import struct
import threading

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._reader import _read_cog_http, read_to_array
from xrspatial.geotiff._writer import write

# ---------------------------------------------------------------------------
# Loopback HTTP server with Range support
# ---------------------------------------------------------------------------


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
    """Start a Range-aware HTTP server on a random loopback port.

    Returns ``(url, httpd, thread)`` so the caller can shut it down. The
    URL uses a unique name suffix to avoid hand-rolled caches getting
    confused if multiple servers run in one process.
    """
    handler_cls = type(
        'RangeHandler1669', (_RangeHandler,), {'payload': payload}
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
    """The HTTP source blocks 127.0.0.1 by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# Hand-rolled planar=2 tiled TIFF builder
# ---------------------------------------------------------------------------
#
# The xrspatial writer emits PlanarConfiguration=1 only, so a planar=2
# fixture has to be built from raw bytes. Mirrors the pattern already
# used by ``_make_planar_tiff`` in ``test_features.py`` (uncompressed
# tiles, little-endian classic TIFF, separate-plane tile sequence).
# Kept self-contained so the test does not depend on ``tifffile``.

def _make_planar2_tiled_tiff(width, height, bands, data, *, tile_size=16):
    """Build an uncompressed PlanarConfiguration=2 tiled TIFF.

    ``data`` is shaped ``(bands, height, width)`` in row-major layout.
    Returns the file bytes. Used to assert the HTTP tile-fetch loop
    handles separate-plane tile sequences correctly; the writer only
    emits planar=1 so we have to lay out the TIFF by hand.
    """
    bo = '<'
    assert data.shape == (bands, height, width)
    dtype = data.dtype
    bps = dtype.itemsize * 8
    sf = 1  # unsigned int

    tw = th = tile_size
    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    # planar=2: emit every tile for band 0, then every tile for band 1,
    # then band 2. Each tile is the per-band slice padded to tile_size
    # if the right or bottom edge is short. ``TileOffsets`` is the
    # concatenated list of byte offsets, one per (band, tile_row,
    # tile_col) tuple in row-major order across bands.
    tile_blobs = []
    for b in range(bands):
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                tile = np.zeros((th, tw), dtype=dtype)
                r0, c0 = tr * th, tc * tw
                r1 = min(r0 + th, height)
                c1 = min(c0 + tw, width)
                tile[:r1 - r0, :c1 - c0] = data[b, r0:r1, c0:c1]
                tile_blobs.append(tile.tobytes())

    pixel_bytes = b''.join(tile_blobs)
    tile_byte_counts = [len(t) for t in tile_blobs]
    num_offsets = len(tile_blobs)

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals))
        )

    def add_longs(tag, vals):
        tag_list.append(
            (tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals))
        )

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bps] * bands)
    add_short(259, 1)   # no compression
    add_short(262, 2 if bands >= 3 else 1)  # RGB or BlackIsZero
    add_short(277, bands)
    add_short(284, 2)   # PlanarConfiguration = Separate
    add_shorts(339, [sf] * bands)
    add_short(322, tw)
    add_short(323, th)
    add_longs(324, [0] * num_offsets)   # placeholder, patched below
    add_longs(325, tile_byte_counts)

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # First pass: figure out where overflow + pixel data land.
    overflow_buf = bytearray()
    for _tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
    overflow_start = ifd_start + ifd_size
    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch TileOffsets (324) with real byte positions, then rebuild
    # overflow buffer with the updated tag value.
    offset_tag = 324
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == offset_tag:
            offs = []
            pos = 0
            for blob in tile_blobs:
                offs.append(pixel_data_start + pos)
                pos += len(blob)
            new_raw = struct.pack(f'{bo}{num_offsets}I', *offs)
            patched.append((tag, typ, count, new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))  # next IFD
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


# ---------------------------------------------------------------------------
# Hand-rolled oriented TIFF builder (for parity-with-local-path guard)
# ---------------------------------------------------------------------------

def _make_oriented_tiff(width, height, orientation, data):
    """Build a minimal uncompressed stripped TIFF with the given
    Orientation tag (274).

    Mirrors the local-path orientation tests in ``test_orientation.py``
    but does not depend on ``tifffile``. Used to assert the HTTP path
    rejects ``window`` on non-default-orientation files the same way
    the local path does.
    """
    bo = '<'
    dtype = data.dtype
    bps = dtype.itemsize * 8
    assert data.shape == (height, width)

    pixel_bytes = data.tobytes()

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, bps)
    add_short(259, 1)         # no compression
    add_short(262, 1)         # BlackIsZero
    add_long(273, 0)          # StripOffsets placeholder
    add_short(274, orientation)
    add_short(277, 1)         # SamplesPerPixel
    add_short(278, height)    # RowsPerStrip = full image
    add_long(279, len(pixel_bytes))   # StripByteCounts
    add_short(284, 1)         # PlanarConfiguration = Chunky
    add_short(339, 1)         # SampleFormat = uint

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    pixel_data_start = ifd_start + ifd_size

    # Patch StripOffsets
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            new_raw = struct.pack(f'{bo}I', pixel_data_start)
            patched.append((tag, typ, count, new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        out.extend(raw.ljust(4, b'\x00'))
    out.extend(struct.pack(f'{bo}I', 0))   # next IFD
    out.extend(pixel_bytes)
    return bytes(out)


# ---------------------------------------------------------------------------
# Single-band tiled COG fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_band_cog(tmp_path):
    """64x64 float32 tiled COG. Returns ``(path, expected_arr)``."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'tmp_1669_single.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True)
    return path, arr


# ---------------------------------------------------------------------------
# Window parity
# ---------------------------------------------------------------------------

def test_http_window_parity_single_band(single_band_cog):
    """``open_geotiff(url, window=...)`` returns the same shape and pixels
    as the local read for the same window. The HTTP branch used to drop
    the window kwarg, returning the full raster.
    """
    path, _ = single_band_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (4, 8, 36, 56)  # 32 rows x 48 cols
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        assert remote.shape == local.shape
        assert remote.shape == (32, 48)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


def test_http_window_parity_full_tile_aligned(single_band_cog):
    """Window aligned to tile boundaries -- the common COG access pattern."""
    path, _ = single_band_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (16, 16, 48, 48)
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


def test_http_window_via_read_to_array_low_level(single_band_cog):
    """``read_to_array(url, window=...)`` honours the window at the low
    level too, not just via the public ``open_geotiff`` wrapper.
    """
    path, _ = single_band_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (10, 12, 20, 30)
        local_arr, _ = read_to_array(path, window=window)
        remote_arr, _ = read_to_array(url, window=window)
        assert remote_arr.shape == local_arr.shape
        assert remote_arr.shape == (10, 18)
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop(httpd)


def test_http_window_via_low_level_read_cog_http(single_band_cog):
    """``_read_cog_http`` accepts ``window`` directly. Used by callers
    that bypass ``read_to_array``.
    """
    path, _ = single_band_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (5, 7, 25, 47)
        local_arr, _ = read_to_array(path, window=window)
        remote_arr, _ = _read_cog_http(url, window=window)
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop(httpd)


def test_http_window_out_of_bounds_rejected(single_band_cog):
    """Window outside the source extent raises the same ``ValueError``
    as the local path. Without the validator, the HTTP helper would
    clamp the window silently and return a smaller array.
    """
    path, _ = single_band_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        # 64x64 source; (0, 0, 100, 100) is out of bounds in both axes.
        with pytest.raises(ValueError, match='outside the source extent'):
            read_to_array(url, window=(0, 0, 100, 100))
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Band parity on multi-band tiled COGs (PlanarConfiguration=1, chunky)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_band_chunky_cog(tmp_path):
    """3-band tiled chunky (planar=1) COG. The xrspatial writer emits
    planar=1 by default for ``(H, W, bands)`` input. Returns
    ``(path, expected_arr)`` with expected shape ``(H, W, bands)``.
    """
    h, w, bands = 32, 48, 3
    rng = np.random.RandomState(1669)
    expected = rng.randint(0, 200, size=(h, w, bands)).astype(np.uint8)
    path = str(tmp_path / 'tmp_1669_chunky.tif')
    write(expected, path, compression='deflate', tiled=True,
          tile_size=16, cog=True)
    return path, expected


def test_http_band_parity_multi_band(multi_band_chunky_cog):
    """``band=B`` on HTTP returns the same 2D slice as the local path.

    Before the fix the HTTP branch accepted ``band=`` but never sliced,
    so the returned array kept its 3-band shape and ``open_geotiff``
    raised on coord-vs-shape mismatch.
    """
    path, _ = multi_band_chunky_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        for b in range(3):
            local = open_geotiff(path, band=b)
            remote = open_geotiff(url, band=b)
            assert remote.shape == local.shape
            assert remote.ndim == 2
            np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


def test_http_band_parity_via_read_to_array(multi_band_chunky_cog):
    """Band slicing happens inside ``read_to_array``'s HTTP branch."""
    path, _ = multi_band_chunky_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        local_arr, _ = read_to_array(path, band=1)
        remote_arr, _ = read_to_array(url, band=1)
        assert remote_arr.shape == local_arr.shape
        assert remote_arr.ndim == 2
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Window + band combined
# ---------------------------------------------------------------------------

def test_http_window_and_band_combined(multi_band_chunky_cog):
    path, _ = multi_band_chunky_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (4, 8, 28, 40)
        local = open_geotiff(path, window=window, band=2)
        remote = open_geotiff(url, window=window, band=2)
        assert remote.shape == local.shape
        assert remote.shape == (24, 32)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# PlanarConfiguration=2 (separate planes)
# ---------------------------------------------------------------------------

@pytest.fixture
def planar_separate_tiled_cog(tmp_path):
    """3-band tiled planar=2 (separate planes) TIFF.

    The xrspatial writer only emits planar=1 (PR #1680 review feedback:
    keep the test self-contained without taking on ``tifffile`` as a
    test dep). The fixture builds the planar=2 file from raw bytes so
    the HTTP tile-fetch loop is still exercised for separate-plane
    layouts. The result is a tiled GeoTIFF rather than a strict COG (no
    overviews), which is fine for the HTTP tile-fetch path.
    """
    h, w, bands = 32, 48, 3
    rng = np.random.RandomState(0x16692)
    # planar=2 stores (bands, h, w); convert to expected display layout
    # (h, w, bands) for the parity comparison.
    data = rng.randint(0, 200, size=(bands, h, w)).astype(np.uint8)
    path = str(tmp_path / 'tmp_1669_planar2.tif')
    payload = _make_planar2_tiled_tiff(w, h, bands, data, tile_size=16)
    with open(path, 'wb') as f:
        f.write(payload)
    expected = np.transpose(data, (1, 2, 0))
    return path, expected


def test_http_planar2_full_read(planar_separate_tiled_cog):
    """Full read of a planar=2 tiled COG over HTTP must match the local
    decode. The HTTP tile-index loop previously used
    ``tile_idx = tr * tiles_across + tc`` with no per-band offset; for
    planar=2 layouts that means band 0's TileOffsets get reused for
    every band, so the returned array is garbage.
    """
    path, expected = planar_separate_tiled_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        local = open_geotiff(path)
        remote = open_geotiff(url)
        assert remote.shape == local.shape
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
        np.testing.assert_array_equal(np.asarray(remote), expected)
    finally:
        _stop(httpd)


def test_http_planar2_windowed(planar_separate_tiled_cog):
    """Windowed read on planar=2 tiled COG over HTTP."""
    path, _ = planar_separate_tiled_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        window = (4, 4, 28, 36)
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        assert remote.shape == local.shape
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


def test_http_planar2_band_selection(planar_separate_tiled_cog):
    """Band selection on a planar=2 file over HTTP."""
    path, _ = planar_separate_tiled_cog
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve(payload)
    try:
        for b in range(3):
            local = open_geotiff(path, band=b)
            remote = open_geotiff(url, band=b)
            assert remote.shape == local.shape
            np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Orientation guard parity with the local path (PR #1680 review)
# ---------------------------------------------------------------------------

def test_http_window_on_oriented_tiff_rejected(tmp_path):
    """An oriented TIFF (Orientation tag != 1) with a window= read over
    HTTP must raise the same ``ValueError`` the local path raises.

    Without the guard the HTTP path used to honour the window blindly
    and silently return a region in stored pixel order, while the local
    path rejected the same call. That asymmetry meant a caller could
    swap a local read for an HTTP read on the same file and get
    different bytes back.
    """
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    # Orientation 2 = horizontal flip. Any non-default value triggers
    # the guard; pick 2 to mirror ``test_orientation_with_window_raises``
    # in ``test_orientation.py``.
    payload = _make_oriented_tiff(width=6, height=4, orientation=2, data=arr)

    # Sanity check: the file decodes (without a window) and the local
    # path rejects window= on it. If either of these break, the parity
    # assertion below is meaningless.
    path = str(tmp_path / 'orient2_no_window.tif')
    with open(path, 'wb') as f:
        f.write(payload)
    local_full = open_geotiff(path)
    np.testing.assert_array_equal(np.asarray(local_full), arr[:, ::-1])
    with pytest.raises(ValueError, match='[Oo]rientation'):
        read_to_array(path, window=(0, 0, 2, 2))

    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(ValueError, match='[Oo]rientation'):
            read_to_array(url, window=(0, 0, 2, 2))
        with pytest.raises(ValueError, match='[Oo]rientation'):
            _read_cog_http(url, window=(0, 0, 2, 2))
    finally:
        _stop(httpd)
