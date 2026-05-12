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
"""
from __future__ import annotations

import http.server
import socketserver
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
    planar=1 by default. Returns ``(path, expected_arr)`` with expected
    shape ``(H, W, bands)``.
    """
    tifffile = pytest.importorskip('tifffile')
    h, w, bands = 32, 48, 3
    rng = np.random.RandomState(1669)
    data = rng.randint(0, 200, size=(bands, h, w)).astype(np.uint8)
    expected = np.transpose(data, (1, 2, 0))
    path = str(tmp_path / 'tmp_1669_chunky.tif')
    tifffile.imwrite(
        path,
        expected,
        photometric='rgb',
        planarconfig='contig',
        tile=(16, 16),
        compression='deflate',
    )
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

    The xrspatial writer only emits planar=1. tifffile is the simplest
    way to produce a planar=2 fixture with control over tiling. Note
    that this is a tiled GeoTIFF rather than a strict COG (no
    overviews), which is fine for the HTTP tile-fetch path.
    """
    tifffile = pytest.importorskip('tifffile')
    h, w, bands = 32, 48, 3
    rng = np.random.RandomState(0x16692)
    data = rng.randint(0, 200, size=(bands, h, w)).astype(np.uint8)
    # tifffile with planarconfig='separate' expects (bands, H, W) input.
    path = str(tmp_path / 'tmp_1669_planar2.tif')
    tifffile.imwrite(
        path,
        data,
        photometric='rgb',
        planarconfig='separate',
        tile=(16, 16),
        compression='deflate',
    )
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
