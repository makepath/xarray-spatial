"""Regression tests for issue #1695.

``_read_cog_http`` accepted ``band`` but did not validate the index, so
HTTP reads diverged from local, dask, GPU, and VRT reads on bad input:

* ``band=-1`` silently returned the last channel via numpy negative
  indexing on the post-decode slice (L1638).
* ``band=N`` with ``N >= samples_per_pixel`` leaked a raw numpy
  ``IndexError`` whose message exposed the internal slice shape.
* ``band=N`` (N != 0) on a single-band HTTP COG was dropped because the
  post-decode slice was gated on
  ``arr.ndim == 3 and samples_per_pixel > 1``; the call returned the
  full single-band raster as if ``band`` had been ``None``.

PR #1673 already pinned the local eager and dask paths to the contract
"0-based non-negative index only". The in-file NOTE at the time of
issue #1695 (``_reader.py:2031-2034``) flagged that the HTTP branch had
not been mirrored. These tests pin the HTTP branch to the same contract
and confirm cross-path parity on the error messages.

Each test FAILS before the fix and PASSES after.
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
#
# Mirrors the helper from ``test_http_window_band_planar_1669.py`` so the
# fixtures stay self-contained without depending on ``tifffile`` or a
# live network. Each server holds one payload and shuts down at test
# teardown.


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
        'RangeHandler1695', (_RangeHandler,), {'payload': payload}
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_band_cog(tmp_path):
    """3-band tiled chunky (planar=1) COG. The writer emits planar=1 by
    default for ``(H, W, bands)`` input. Returns ``(path, payload, arr)``
    where ``payload`` is the on-disk bytes ready to serve.
    """
    h, w, bands = 32, 48, 3
    rng = np.random.RandomState(1695)
    arr = rng.randint(0, 200, size=(h, w, bands)).astype(np.uint8)
    path = str(tmp_path / 'tmp_1695_multi.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True)
    with open(path, 'rb') as f:
        payload = f.read()
    return path, payload, arr


@pytest.fixture
def single_band_cog(tmp_path):
    """64x64 single-band float32 tiled COG."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'tmp_1695_single.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True)
    with open(path, 'rb') as f:
        payload = f.read()
    return path, payload, arr


# ---------------------------------------------------------------------------
# Negative band index on multi-band HTTP read
# ---------------------------------------------------------------------------

def test_http_negative_band_rejected(multi_band_cog):
    """``band=-1`` on a multi-band HTTP COG raises ``IndexError`` instead
    of silently selecting the last channel.

    Before the fix, ``arr[:, :, -1]`` returned the trailing band without
    any error. The local path raises
    ``IndexError("band=-1 out of range for 3-band file.")`` via #1673.
    """
    _path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            read_to_array(url, band=-1)
    finally:
        _stop(httpd)


def test_http_negative_band_rejected_via_low_level(multi_band_cog):
    """The low-level ``_read_cog_http`` rejects ``band=-1`` too, not just
    the ``read_to_array`` wrapper. Catches any future caller that bypasses
    the wrapper.
    """
    _path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            _read_cog_http(url, band=-1)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Out-of-range band index on multi-band HTTP read
# ---------------------------------------------------------------------------

def test_http_band_equal_to_samples_rejected(multi_band_cog):
    """``band=samples_per_pixel`` (off-by-one) raises the typed error
    instead of leaking the raw numpy axis-2-out-of-bounds message.
    """
    _path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        # File has 3 bands; valid indices are 0, 1, 2.
        with pytest.raises(IndexError, match=r"band=3 out of range"):
            read_to_array(url, band=3)
    finally:
        _stop(httpd)


def test_http_band_far_above_samples_rejected(multi_band_cog):
    """A wildly out-of-range band index also raises the typed error."""
    _path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError, match=r"band=42 out of range"):
            read_to_array(url, band=42)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Single-band HTTP read
# ---------------------------------------------------------------------------

def test_http_nonzero_band_on_single_band_rejected(single_band_cog):
    """``band=1`` on a single-band HTTP COG raises ``IndexError`` instead
    of silently returning the full raster.

    Before the fix, the post-decode slice at L1660 was gated on
    ``arr.ndim == 3 and samples_per_pixel > 1`` so ``band=1`` on a 2D
    single-band array was dropped on the floor. The local path raises
    ``IndexError("band=1 requested on a single-band file.")`` via #1673.
    """
    _path, payload, _arr = single_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError,
                           match=r"band=1 requested on a single-band file"):
            read_to_array(url, band=1)
    finally:
        _stop(httpd)


def test_http_band_zero_on_single_band_still_works(single_band_cog):
    """``band=0`` on a single-band HTTP COG succeeds.

    Negative case: the guard rejects nonzero indices but must not
    over-reject the only valid index on a single-band file. Mirrors the
    local-path validator's ``if band != 0`` branch.
    """
    _path, payload, expected = single_band_cog
    url, httpd, _ = _serve(payload)
    try:
        arr, _ = read_to_array(url, band=0)
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# band=None preserves multi-band behaviour (regression)
# ---------------------------------------------------------------------------

def test_http_band_none_returns_all_bands(multi_band_cog):
    """``band=None`` on a multi-band HTTP COG returns the full
    ``(H, W, bands)`` array unchanged. Regression for the validator: a
    typo that promoted ``None`` to an integer comparison would break
    every multi-band HTTP read.
    """
    _path, payload, expected = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        arr, _ = read_to_array(url)
        assert arr.shape == expected.shape
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# Cross-path parity with local-path eager read
# ---------------------------------------------------------------------------

def test_local_and_http_negative_band_parity(multi_band_cog):
    """The local eager path and the HTTP path raise the same
    ``IndexError`` class with the same diagnostic substring on
    ``band=-1``. This is the parity guard #1673 set up for local vs dask
    vs GPU; the HTTP branch joins after #1695.
    """
    path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError) as local_exc:
            read_to_array(path, band=-1)
        with pytest.raises(IndexError) as http_exc:
            read_to_array(url, band=-1)
        assert "band=-1 out of range" in str(local_exc.value)
        assert "band=-1 out of range" in str(http_exc.value)
        # Same wording, not just same substring.
        assert str(local_exc.value) == str(http_exc.value)
    finally:
        _stop(httpd)


def test_local_and_http_band_equal_to_samples_parity(multi_band_cog):
    """Local and HTTP agree on the off-by-one rejection message."""
    path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError) as local_exc:
            read_to_array(path, band=3)
        with pytest.raises(IndexError) as http_exc:
            read_to_array(url, band=3)
        assert "band=3 out of range" in str(local_exc.value)
        assert "band=3 out of range" in str(http_exc.value)
        assert str(local_exc.value) == str(http_exc.value)
    finally:
        _stop(httpd)


def test_local_and_http_single_band_nonzero_parity(single_band_cog):
    """Local and HTTP agree on the single-band nonzero rejection
    message. Before the fix, the local path raised
    ``"band=1 requested on a single-band file."`` and the HTTP path
    returned the full single-band raster without erroring at all.
    """
    path, payload, _arr = single_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError) as local_exc:
            read_to_array(path, band=1)
        with pytest.raises(IndexError) as http_exc:
            read_to_array(url, band=1)
        assert "single-band file" in str(local_exc.value)
        assert "single-band file" in str(http_exc.value)
        assert str(local_exc.value) == str(http_exc.value)
    finally:
        _stop(httpd)


# ---------------------------------------------------------------------------
# open_geotiff wrapper passes the rejection through (smoke test)
# ---------------------------------------------------------------------------

def test_open_geotiff_http_negative_band_rejected(multi_band_cog):
    """The public ``open_geotiff`` wrapper also rejects ``band=-1`` on
    HTTP, not just the low-level ``read_to_array``. Users hit the
    wrapper, so a regression there would be invisible to the low-level
    test.
    """
    _path, payload, _arr = multi_band_cog
    url, httpd, _ = _serve(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            open_geotiff(url, band=-1)
    finally:
        _stop(httpd)
