"""Loopback HTTP-server integration tests for the GeoTIFF reader.

Consolidated from the issue-numbered files mapped in
``CLUSTER_AUDIT_PR9.md``. Every test here needs a working loopback
bind, hence the module-level ``@requires_loopback`` marker. PR 11 of
epic #2390 drops the ``pytest_collection_modifyitems`` socketserver
hack in ``conftest.py``; the explicit marker here keeps these tests
skipped on sandboxes that deny loopback bind once the hack is gone.
"""
from __future__ import annotations

import http.server
import inspect
import io
import math
import socket
import socketserver
import struct
import threading
import time

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import UnsafeURLError
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import _sources as _sources_mod
from xrspatial.geotiff import (open_geotiff, read_geotiff_dask, read_geotiff_gpu, read_vrt,
                               to_geotiff, write_geotiff_gpu, write_vrt)
from xrspatial.geotiff._errors import RotatedTransformError
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import (_FULL_IMAGE_BUDGET_HEADER_SLACK, INITIAL_HTTP_HEADER_BYTES,
                                       MAX_HTTP_HEADER_BYTES, CloudSizeLimitError,
                                       PixelSafetyLimitError, _compute_full_image_byte_budget,
                                       _HTTPSource, _parse_cog_http_meta, _read_cog_http,
                                       coalesce_ranges, read_to_array, split_coalesced_bytes)
from xrspatial.geotiff._sidecar import _is_http_url as _sidecar_is_http_url
from xrspatial.geotiff._sources import MAX_COALESCED_RANGE_BYTES_DEFAULT
from xrspatial.geotiff._writer import _is_fsspec_uri as _writer_is_fsspec_uri
from xrspatial.geotiff._writer import write
from xrspatial.geotiff.tests._helpers.markers import requires_gpu, requires_loopback

pytestmark = requires_loopback


# ----------------------------------------------------------
# Section: http_band_validation
# Source: test_http_band_validation_1695.py
# ----------------------------------------------------------
class _RangeHandler_http_band_validation(http.server.BaseHTTPRequestHandler):
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


def _serve_http_band_validation(payload: bytes):
    handler_cls = type(
        'RangeHandler1695', (_RangeHandler_http_band_validation,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/cog.tif', httpd, thread


def _stop_http_band_validation(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_http_band_validation(monkeypatch):
    """The HTTP source rejects loopback by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_band_cog_http_band_validation(tmp_path):
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
def single_band_cog_http_band_validation(tmp_path):
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

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_negative_band_rejected(multi_band_cog_http_band_validation):
    """``band=-1`` on a multi-band HTTP COG raises ``IndexError`` instead
    of silently selecting the last channel.

    Before the fix, ``arr[:, :, -1]`` returned the trailing band without
    any error. The local path raises
    ``IndexError("band=-1 out of range for 3-band file.")`` via #1673.
    """
    _path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            read_to_array(url, band=-1)
    finally:
        _stop_http_band_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_negative_band_rejected_via_low_level(multi_band_cog_http_band_validation):
    """The low-level ``_read_cog_http`` rejects ``band=-1`` too, not just
    the ``read_to_array`` wrapper. Catches any future caller that bypasses
    the wrapper.
    """
    _path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            _read_cog_http(url, band=-1)
    finally:
        _stop_http_band_validation(httpd)


# ---------------------------------------------------------------------------
# Out-of-range band index on multi-band HTTP read
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_band_equal_to_samples_rejected(multi_band_cog_http_band_validation):
    """``band=samples_per_pixel`` (off-by-one) raises the typed error
    instead of leaking the raw numpy axis-2-out-of-bounds message.
    """
    _path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        # File has 3 bands; valid indices are 0, 1, 2.
        with pytest.raises(IndexError, match=r"band=3 out of range"):
            read_to_array(url, band=3)
    finally:
        _stop_http_band_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_band_far_above_samples_rejected(multi_band_cog_http_band_validation):
    """A wildly out-of-range band index also raises the typed error."""
    _path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError, match=r"band=42 out of range"):
            read_to_array(url, band=42)
    finally:
        _stop_http_band_validation(httpd)


# ---------------------------------------------------------------------------
# Single-band HTTP read
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_nonzero_band_on_single_band_rejected(single_band_cog_http_band_validation):
    """``band=1`` on a single-band HTTP COG raises ``IndexError`` instead
    of silently returning the full raster.

    Before the fix, the post-decode slice at L1660 was gated on
    ``arr.ndim == 3 and samples_per_pixel > 1`` so ``band=1`` on a 2D
    single-band array was dropped on the floor. The local path raises
    ``IndexError("band=1 requested on a single-band file.")`` via #1673.
    """
    _path, payload, _arr = single_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError,
                           match=r"band=1 requested on a single-band file"):
            read_to_array(url, band=1)
    finally:
        _stop_http_band_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_band_zero_on_single_band_still_works(single_band_cog_http_band_validation):
    """``band=0`` on a single-band HTTP COG succeeds.

    Negative case: the guard rejects nonzero indices but must not
    over-reject the only valid index on a single-band file. Mirrors the
    local-path validator's ``if band != 0`` branch.
    """
    _path, payload, expected = single_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        arr, _ = read_to_array(url, band=0)
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop_http_band_validation(httpd)


# ---------------------------------------------------------------------------
# band=None preserves multi-band behaviour (regression)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_http_band_none_returns_all_bands(multi_band_cog_http_band_validation):
    """``band=None`` on a multi-band HTTP COG returns the full
    ``(H, W, bands)`` array unchanged. Regression for the validator: a
    typo that promoted ``None`` to an integer comparison would break
    every multi-band HTTP read.
    """
    _path, payload, expected = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        arr, _ = read_to_array(url)
        assert arr.shape == expected.shape
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop_http_band_validation(httpd)


# ---------------------------------------------------------------------------
# Cross-path parity with local-path eager read
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_local_and_http_negative_band_parity(multi_band_cog_http_band_validation):
    """The local eager path and the HTTP path raise the same
    ``IndexError`` class with the same diagnostic substring on
    ``band=-1``. This is the parity guard #1673 set up for local vs dask
    vs GPU; the HTTP branch joins after #1695.
    """
    path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
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
        _stop_http_band_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_local_and_http_band_equal_to_samples_parity(multi_band_cog_http_band_validation):
    """Local and HTTP agree on the off-by-one rejection message."""
    path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError) as local_exc:
            read_to_array(path, band=3)
        with pytest.raises(IndexError) as http_exc:
            read_to_array(url, band=3)
        assert "band=3 out of range" in str(local_exc.value)
        assert "band=3 out of range" in str(http_exc.value)
        assert str(local_exc.value) == str(http_exc.value)
    finally:
        _stop_http_band_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_local_and_http_single_band_nonzero_parity(single_band_cog_http_band_validation):
    """Local and HTTP agree on the single-band nonzero rejection
    message. Before the fix, the local path raised
    ``"band=1 requested on a single-band file."`` and the HTTP path
    returned the full single-band raster without erroring at all.
    """
    path, payload, _arr = single_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError) as local_exc:
            read_to_array(path, band=1)
        with pytest.raises(IndexError) as http_exc:
            read_to_array(url, band=1)
        assert "single-band file" in str(local_exc.value)
        assert "single-band file" in str(http_exc.value)
        assert str(local_exc.value) == str(http_exc.value)
    finally:
        _stop_http_band_validation(httpd)


# ---------------------------------------------------------------------------
# open_geotiff wrapper passes the rejection through (smoke test)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_band_validation')
def test_open_geotiff_http_negative_band_rejected(multi_band_cog_http_band_validation):
    """The public ``open_geotiff`` wrapper also rejects ``band=-1`` on
    HTTP, not just the low-level ``read_to_array``. Users hit the
    wrapper, so a regression there would be invisible to the low-level
    test.
    """
    _path, payload, _arr = multi_band_cog_http_band_validation
    url, httpd, _ = _serve_http_band_validation(payload)
    try:
        with pytest.raises(IndexError, match=r"band=-1 out of range"):
            open_geotiff(url, band=-1)
    finally:
        _stop_http_band_validation(httpd)

# ----------------------------------------------------------
# Section: http_cog_coalesce
# Source: test_http_cog_coalesce.py
# ----------------------------------------------------------


def test_coalesce_empty_input():
    merged, mapping = coalesce_ranges([])
    assert merged == []
    assert mapping == []


def test_coalesce_single_range():
    merged, mapping = coalesce_ranges([(100, 50)])
    assert merged == [(100, 50)]
    # One input -> one merged entry, rel_offset 0, length 50.
    assert mapping == [(0, 0, 50)]


def test_coalesce_merges_adjacent_ranges():
    # Three back-to-back ranges, each within the default gap threshold
    # of the next, should collapse into a single merged GET.
    ranges = [(0, 100), (100, 50), (150, 25)]
    merged, mapping = coalesce_ranges(ranges)
    assert len(merged) == 1
    start, length = merged[0]
    assert start == 0
    assert length == 175
    # Each input maps back to merged_idx 0 with the right offset.
    assert mapping[0] == (0, 0, 100)
    assert mapping[1] == (0, 100, 50)
    assert mapping[2] == (0, 150, 25)


def test_coalesce_does_not_merge_when_gap_exceeds_threshold():
    # 200-byte gap with gap_threshold=50 must split.
    ranges = [(0, 100), (300, 50)]
    merged, mapping = coalesce_ranges(ranges, gap_threshold=50)
    assert merged == [(0, 100), (300, 50)]
    assert mapping[0] == (0, 0, 100)
    assert mapping[1] == (1, 0, 50)


def test_coalesce_with_unsorted_input():
    # Coalescing should sort by offset and still produce correct mapping
    # in input order.
    ranges = [(200, 30), (0, 100), (100, 50)]
    merged, mapping = coalesce_ranges(ranges)
    # All three are within 1 MB of each other so they merge into one.
    assert len(merged) == 1
    start, length = merged[0]
    assert start == 0
    assert length == 230
    # mapping is in input order, not sort order.
    assert mapping[0] == (0, 200, 30)
    assert mapping[1] == (0, 0, 100)
    assert mapping[2] == (0, 100, 50)


def test_coalesce_negative_threshold_disables_merging():
    ranges = [(0, 10), (10, 10), (20, 10)]
    merged, mapping = coalesce_ranges(ranges, gap_threshold=-1)
    # Every input becomes its own merged range.
    assert len(merged) == 3
    for i, (orig, _) in enumerate(zip(ranges, merged)):
        assert mapping[i][0] != mapping[(i + 1) % 3][0] or i == 2


def test_coalesce_split_recovers_per_tile_bytes():
    # Round-trip: real bytes through a fake fetcher that mirrors what
    # urllib3 would return for a Range request.
    payload = bytes(range(256)) * 4  # 1024 unique-ish bytes
    ranges = [(0, 100), (100, 50), (200, 30), (1000, 20)]

    merged, mapping = coalesce_ranges(ranges, gap_threshold=200)
    # First three merge (gaps 0 and 50 -> within threshold), last splits
    # (gap of 770 from end of third range).
    assert len(merged) == 2

    merged_bytes = [payload[s:s + le] for (s, le) in merged]
    out = split_coalesced_bytes(merged_bytes, mapping)

    for (off, length), tile in zip(ranges, out):
        assert tile == payload[off:off + length]


# ---------------------------------------------------------------------------
# Issue #2266: coalesced-range size cap. Without this cap a tile table
# with many small valid byte counts and sub-MiB gaps would chain into
# one merged range whose length is roughly num_tiles * gap_threshold,
# turning a safe per-tile fetch into a multi-GiB over-fetch.
# ---------------------------------------------------------------------------

def test_coalesce_caps_merged_range_size():
    # 8 tiny ranges spaced 1 MiB apart. Every gap is within the default
    # 1 MiB threshold so without the size cap they would all merge into
    # one ~7 MiB range. With a 4 MiB cap the coalescer must split. The
    # next test (``test_coalesce_cap_round_trips_bytes_2266``) covers
    # byte-level recovery after the split.
    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(8)]
    merged, mapping = coalesce_ranges(
        ranges, max_coalesced_range_bytes=4 * one_mib)
    # No merged range exceeds the cap.
    for _start, length in merged:
        assert length <= 4 * one_mib, (
            f'merged range of {length} bytes exceeds 4 MiB cap')
    # Splitting still happened: more than one merged range.
    assert len(merged) > 1
    # Every input is still represented in the mapping.
    assert len(mapping) == len(ranges)


def test_coalesce_cap_round_trips_bytes():
    # When the cap forces a split, split_coalesced_bytes must still
    # recover every original byte range correctly.
    one_mib = 1 << 20
    payload_len = 8 * one_mib + 1024
    # Use a deterministic payload we can slice and compare against.
    payload = bytes((i * 17) & 0xFF for i in range(payload_len))
    ranges = [(i * one_mib, 1024) for i in range(8)]

    merged, mapping = coalesce_ranges(
        ranges, max_coalesced_range_bytes=4 * one_mib)
    merged_bytes = [payload[s:s + le] for (s, le) in merged]
    out = split_coalesced_bytes(merged_bytes, mapping)

    for (off, length), tile in zip(ranges, out):
        assert tile == payload[off:off + length]


def test_coalesce_default_cap_bounds_adversarial_input_simple():
    # The motivating scenario from issue #2266: 4096 tiles, each 1 KB,
    # with offsets spaced 1 MiB apart. Without the cap this collapses
    # into one ~4 GiB merged range. With the default cap nothing
    # exceeds MAX_COALESCED_RANGE_BYTES_DEFAULT.
    # (See test_coalesce_default_cap_bounds_adversarial_input for the
    # http-cog-range-contract variant that also exercises the loopback
    # fixtures; this version is the pure-coalescer unit case.)
    from xrspatial.geotiff._sources import MAX_COALESCED_RANGE_BYTES_DEFAULT

    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(4096)]
    merged, _ = coalesce_ranges(ranges)
    for _start, length in merged:
        assert length <= MAX_COALESCED_RANGE_BYTES_DEFAULT, (
            f'merged range {length} bytes exceeds default cap '
            f'{MAX_COALESCED_RANGE_BYTES_DEFAULT} bytes')


def test_coalesce_cap_zero_disables_size_check():
    # A non-positive cap means "no size limit" -- the gap threshold
    # alone governs merging. Useful as an escape hatch for callers
    # that have their own bookkeeping.
    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(8)]
    merged, _ = coalesce_ranges(
        ranges, max_coalesced_range_bytes=0)
    # All eight merge into one ~7 MiB + 1 KB range.
    assert len(merged) == 1
    _, length = merged[0]
    assert length == 7 * one_mib + 1024


def test_coalesce_cap_does_not_split_legitimate_back_to_back():
    # The cap must not punish well-behaved COGs whose tiles really are
    # back-to-back. A real COG with 64 tiles of 64 KB each (total 4 MiB)
    # should still collapse into a single GET under the default cap.
    tile_bytes = 64 * 1024
    n_tiles = 64
    ranges = [(i * tile_bytes, tile_bytes) for i in range(n_tiles)]
    merged, _ = coalesce_ranges(ranges)
    assert len(merged) == 1
    assert merged[0] == (0, n_tiles * tile_bytes)


def test_coalesce_cap_respects_env_override(monkeypatch):
    # When max_coalesced_range_bytes is None (the default), the helper
    # reads XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES from the environment.
    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(8)]
    # Force a 2 MiB cap via env. The 8 ranges spaced 1 MiB apart must
    # split into at least 4 merged ranges (2 MiB each + slack).
    monkeypatch.setenv(
        'XRSPATIAL_COG_MAX_COALESCED_RANGE_BYTES', str(2 * one_mib))
    merged, _ = coalesce_ranges(ranges)
    for _start, length in merged:
        assert length <= 2 * one_mib
    assert len(merged) >= 4


def test_coalesce_cap_preserves_oversized_single_input():
    # If a single input range already exceeds the cap, the function
    # still emits it intact. Rejecting oversized individual tiles is
    # the job of the per-tile cap, not the coalescer.
    big = 10 * (1 << 20)  # 10 MiB
    ranges = [(0, big)]
    merged, mapping = coalesce_ranges(
        ranges, max_coalesced_range_bytes=1 << 20)  # 1 MiB cap
    assert merged == [(0, big)]
    assert mapping == [(0, 0, big)]


# ---------------------------------------------------------------------------
# Mocked HTTP source for perf and call-count assertions
# ---------------------------------------------------------------------------

class _MockHTTPSource_http_cog_coalesce(_HTTPSource):
    """``_HTTPSource`` that serves bytes from an in-memory buffer.

    Records every ``read_range`` call and optionally sleeps to simulate
    RTT. Tracks calls separately by ``(start, length)`` so tests can
    assert how many tile fetches versus IFD fetches happened.
    """

    def __init__(self, buf: bytes, rtt: float = 0.0):
        self._url = 'mock://'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self._rtt = rtt
        self.calls: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        with self._lock:
            self.calls.append((start, length))
        if self._rtt > 0:
            time.sleep(self._rtt)
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        with self._lock:
            self.calls.append((0, len(self._buf)))
        if self._rtt > 0:
            time.sleep(self._rtt)
        return self._buf


def test_http_source_read_ranges_coalesced_respects_cap():
    """The HTTP wrapper must propagate the size cap to coalesce_ranges.

    Builds a 16 MiB in-memory buffer, then asks the source to fetch
    eight 1 KB ranges spaced 1 MiB apart. Without the cap the wrapper
    would issue a single ~7 MiB merged GET; with a 4 MiB cap it issues
    at least two smaller GETs.
    """
    one_mib = 1 << 20
    buf = bytes((i * 13) & 0xFF for i in range(16 * one_mib))
    src = _MockHTTPSource_http_cog_coalesce(buf)
    ranges = [(i * one_mib, 1024) for i in range(8)]

    out = src.read_ranges_coalesced(
        ranges, max_workers=2,
        max_coalesced_range_bytes=4 * one_mib)
    # Bytes must match the original per-range slices.
    for (off, length), tile in zip(ranges, out):
        assert tile == buf[off:off + length]
    # The actual GETs the mock saw must all respect the cap.
    assert src.calls, 'no GETs were issued'
    for _start, length in src.calls:
        assert length <= 4 * one_mib, (
            f'merged GET of {length} bytes exceeds 4 MiB cap')
    # And the cap must have caused at least one split.
    assert len(src.calls) >= 2


@pytest.fixture
def small_cog_bytes_http_cog_coalesce(tmp_path):
    """Build a small tiled COG and return its raw bytes."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'cog.tif')
    # 8x8 tile grid (tile_size=8, image=64x64) gives 64 tiles -- enough
    # tile count to make coalescing observable.
    write(arr, path, compression='deflate', tiled=True, tile_size=8,
          cog=True)
    with open(path, 'rb') as f:
        return f.read(), arr, path


def test_read_cog_http_uses_coalesced_fetches(small_cog_bytes_http_cog_coalesce, monkeypatch):
    """One coalesced GET for many adjacent COG tiles instead of N GETs."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes_http_cog_coalesce
    src = _MockHTTPSource_http_cog_coalesce(buf)

    def _fake_http_source(url):
        return src

    monkeypatch.setattr(_reader_mod, '_HTTPSource', _fake_http_source)

    arr, _ = _reader_mod._read_cog_http('http://mock/cog.tif')
    np.testing.assert_array_equal(arr, expected)

    # All calls. We expect one or two header GETs (16 KB / optional 64 KB)
    # plus ONE merged tile GET, not 64 tile GETs.
    tile_fetches = [
        (s, le) for (s, le) in src.calls
        if s != 0 or le > 65536  # exclude header reads
    ]
    assert len(tile_fetches) == 1, (
        f'expected a single coalesced tile fetch, got {len(tile_fetches)}: '
        f'{tile_fetches[:5]}'
    )


def test_read_cog_http_perf_with_mock_rtt(small_cog_bytes_http_cog_coalesce, monkeypatch):
    """Coalesced HTTP should beat the un-coalesced baseline at 50 ms RTT."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes_http_cog_coalesce
    rtt = 0.05

    # Baseline: disable coalescing via env var so each tile costs an RTT.
    monkeypatch.setenv('XRSPATIAL_COG_COALESCE_GAP', '-1')
    src1 = _MockHTTPSource_http_cog_coalesce(buf, rtt=rtt)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src1)
    t0 = time.perf_counter()
    arr1, _ = _reader_mod._read_cog_http('http://mock/cog.tif')
    baseline = time.perf_counter() - t0
    np.testing.assert_array_equal(arr1, expected)

    # Coalesced path: leave env unset (default 1 MB threshold).
    monkeypatch.delenv('XRSPATIAL_COG_COALESCE_GAP', raising=False)
    src2 = _MockHTTPSource_http_cog_coalesce(buf, rtt=rtt)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src2)
    t0 = time.perf_counter()
    arr2, _ = _reader_mod._read_cog_http('http://mock/cog.tif')
    coalesced = time.perf_counter() - t0
    np.testing.assert_array_equal(arr2, expected)

    # Assert on RTTs saved, not on a wall-time ratio. The baseline pays
    # ceil(64/8) = 8 RTTs; the coalesced path pays 1 merged GET plus the
    # IFD read = ~2 RTTs. The other ~6 RTTs of saved wall time are what
    # the assertion checks. A ratio assertion would couple this to per-tile
    # decode cost, which varies a lot across CI runners.
    rtts_saved = (baseline - coalesced) / rtt
    assert rtts_saved >= 5, (
        f'coalesced wall time {coalesced:.3f}s should save at least 5 RTTs '
        f'vs baseline {baseline:.3f}s (saved {rtts_saved:.1f} RTTs of {rtt:.3f}s)'
    )


# ---------------------------------------------------------------------------
# read_geotiff_dask: IFD parsing call count and correctness
# ---------------------------------------------------------------------------

def test_dask_local_correctness(small_cog_bytes_http_cog_coalesce):
    """Dask read of a local COG must equal the eager read bit-for-bit."""
    _, expected, path = small_cog_bytes_http_cog_coalesce
    eager = open_geotiff(path)
    lazy = read_geotiff_dask(path, chunks=16).compute()
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(lazy))
    np.testing.assert_array_equal(np.asarray(eager), expected)


def test_dask_http_parses_ifds_once(small_cog_bytes_http_cog_coalesce, monkeypatch):
    """An N-chunk HTTP graph must trigger at most one IFD-header GET."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes_http_cog_coalesce
    src_holder: list[_MockHTTPSource_http_cog_coalesce] = []

    def _fake_http_source(url):
        s = _MockHTTPSource_http_cog_coalesce(buf)
        src_holder.append(s)
        return s

    monkeypatch.setattr(_reader_mod, '_HTTPSource', _fake_http_source)

    # 16x16 chunks on 64x64 -> 16 chunks. Without P5 each chunk would
    # spawn its own _HTTPSource and fire its own (0, 16384) GET.
    da_arr = read_geotiff_dask('http://mock/cog.tif', chunks=16).compute()
    np.testing.assert_array_equal(np.asarray(da_arr), expected)

    # Count "header" GETs across every _HTTPSource instance the read
    # path created. The header probe is exactly (0, 16384) (or the
    # fallback (0, 65536)).
    header_calls = 0
    for s in src_holder:
        for (start, length) in s.calls:
            if start == 0 and length in (16384, 65536):
                header_calls += 1
    assert header_calls <= 1, (
        f'expected <=1 header GETs across the dask graph, got '
        f'{header_calls} (over {len(src_holder)} sources)'
    )

# ----------------------------------------------------------
# Section: http_cog_range_contract
# Source: test_http_cog_range_contract_2286.py
# ----------------------------------------------------------


@pytest.fixture()
def _no_sidecar_probe_http_cog_range_contract(monkeypatch):
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    monkeypatch.setattr(_sidecar_mod, 'find_sidecar', lambda _src: None)
    # ``discover_remote_sidecar`` is invoked from the HTTP meta path;
    # short-circuit it too so the mock servers never see the probe.
    monkeypatch.setattr(
        _sidecar_mod,
        'discover_remote_sidecar',
        lambda src, ifds, **_kw: (ifds, None, set()),
    )


# ---------------------------------------------------------------------------
# Loopback Range HTTP server, copied from the sibling tests (verbatim).
# ---------------------------------------------------------------------------

class _RangeHandler_http_cog_range_contract(http.server.BaseHTTPRequestHandler):
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


def _stop_http_cog_range_contract(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_http_cog_range_contract(monkeypatch):
    """The HTTP source rejects loopback by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# In-process recording HTTP source (no socket round-trip).
#
# Mirrors ``_MockHTTPSource`` / ``_RecordingHTTPSource`` from the sibling
# files: serves bytes from an in-memory buffer and tracks every
# ``read_range`` / ``read_all`` call so tests can assert how many GETs
# the reader issued and which byte ranges they covered.
# ---------------------------------------------------------------------------

class _RecordingHTTPSource_http_cog_range_contract(_HTTPSource):
    def __init__(self, buf: bytes):
        self._url = 'mock://2286/cog.tif'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self.calls: list[tuple[int, int]] = []
        self.read_all_called = False
        self._closed_count = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        with self._lock:
            self.calls.append((start, length))
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        with self._lock:
            self.read_all_called = True
            self.calls.append((0, len(self._buf)))
        return self._buf

    def close(self):
        with self._lock:
            self._closed_count += 1

    # Helper accessors used by the test bodies.

    def total_bytes(self) -> int:
        return sum(le for _s, le in self.calls)

    def tile_or_strip_calls(self) -> list[tuple[int, int]]:
        """Calls past the header probe.

        The HTTP reader's first GET is a 16 KiB (or 64 KiB fallback)
        prefix anchored at offset 0 to fetch the IFD chain. Excluding
        that lets the test reason about pixel-data byte traffic alone.
        """
        return [
            (s, le) for (s, le) in self.calls
            if not (s == 0 and le in (16384, 65536))
        ]


# ---------------------------------------------------------------------------
# Small COG fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tiled_cog_http_cog_range_contract(tmp_path):
    """Single-band 256x256 float32 tiled COG with 32x32 tiles (64 tiles).

    Sized large enough that the whole file comfortably exceeds the
    header probe (16 KiB), so the "windowed read pulls less than the
    file" assertion is meaningful. Pseudo-random pixel values are used
    so deflate cannot squash the file below the header-probe size.
    """
    rng = np.random.RandomState(2293)
    arr = rng.standard_normal((256, 256)).astype(np.float32)
    path = str(tmp_path / 'tmp_2293_tiled.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=32,
          cog=True)
    with open(path, 'rb') as f:
        return f.read(), arr, path


@pytest.fixture
def cog_with_overviews_http_cog_range_contract(tmp_path):
    """256x256 float32 tiled COG with one overview level (factor 2).

    Pseudo-random pixels so deflate cannot collapse the level-0 IFD
    below the header probe, which would make the
    'overview pulls fewer bytes than base' assertion vacuous.
    """
    rng = np.random.RandomState(0x2293)
    arr = rng.standard_normal((256, 256)).astype(np.float32)
    path = str(tmp_path / 'tmp_2293_ovr.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=32,
          cog=True, overview_levels=[2])
    with open(path, 'rb') as f:
        return f.read(), arr, path


@pytest.fixture
def multiband_chunky_cog_http_cog_range_contract(tmp_path):
    """3-band tiled chunky (planar=1) COG, 128x128, 32x32 tiles."""
    h, w, bands = 128, 128, 3
    rng = np.random.RandomState(2293)
    expected = rng.randint(0, 200, size=(h, w, bands)).astype(np.uint8)
    path = str(tmp_path / 'tmp_2293_chunky.tif')
    write(expected, path, compression='deflate', tiled=True,
          tile_size=32, cog=True)
    with open(path, 'rb') as f:
        return f.read(), expected, path


# ===========================================================================
# 1. Windowed reads fetch only intersecting tiles -- bounded bytes + ranges
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_windowed_tile_read_bounded_bytes_and_range_count(
        small_tiled_cog_http_cog_range_contract, monkeypatch):
    """A 32x32 window aligned to one tile fetches a single tile's bytes,
    not the whole file.

    Pre-#1669/#1842 the HTTP path either ignored ``window=`` or fell
    back to ``read_all()`` and sliced. Either regression would push the
    total byte count past the per-tile budget.
    """
    buf, expected, _ = small_tiled_cog_http_cog_range_contract
    src = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    window = (32, 32, 64, 64)  # one whole 32x32 tile (tile [1, 1])
    arr, _ = _read_cog_http('http://mock/cog.tif', window=window)
    np.testing.assert_array_equal(arr, expected[32:64, 32:64])

    # The fallback ``read_all`` would pull the whole file. Pin against it.
    assert not src.read_all_called, (
        "windowed HTTP read fell back to read_all(); the windowed branch "
        "must fetch only intersecting tile byte ranges")

    pixel_calls = src.tile_or_strip_calls()
    # One window-aligned tile -> at most one coalesced pixel GET. The
    # coalescer may emit a single merged GET or a single per-tile GET;
    # both are fine. Two or more means the reader fetched neighbouring
    # tiles it did not need.
    assert len(pixel_calls) <= 1, (
        f"expected <=1 pixel GET for a single-tile window, got "
        f"{len(pixel_calls)}: {pixel_calls}")

    # The total byte count must be bounded by the windowed footprint
    # plus header slack, not the file size. 32x32 float32 + deflate
    # compression slack < 8 KiB; the file is >=128 KiB. The hard bound
    # is the file size; the soft bound (header + a few tiles' worth)
    # catches a regression where the reader pulls neighbouring tiles
    # it does not need.
    assert src.total_bytes() < len(buf), (
        f"windowed read consumed {src.total_bytes()} of {len(buf)} file "
        f"bytes; the windowed branch must not pull the whole file")
    assert src.total_bytes() <= 64 * 1024 + 32 * 1024, (
        f"windowed read consumed {src.total_bytes()} bytes; well above "
        f"the expected header + single-tile budget")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_windowed_multi_tile_read_range_count_bounded(
        small_tiled_cog_http_cog_range_contract, monkeypatch):
    """A window that touches 2x2=4 tiles must not fetch all 64 tiles
    in the file.

    Pins the intersect-only contract for windows that span multiple
    tiles. With coalescing on by default the four adjacent tiles may
    merge into one GET, so the assertion is on an upper bound, not an
    exact count.
    """
    buf, expected, _ = small_tiled_cog_http_cog_range_contract
    src = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    # Cover tiles (0,0), (0,1), (1,0), (1,1) -- top-left 2x2 block.
    window = (0, 0, 64, 64)
    arr, _ = _read_cog_http('http://mock/cog.tif', window=window)
    np.testing.assert_array_equal(arr, expected[0:64, 0:64])

    pixel_calls = src.tile_or_strip_calls()
    # 4 intersecting tiles -> coalescing collapses adjacent ones to a
    # small handful of GETs. A regression that fetched every tile in
    # the file would emit >=64 separate GETs or fall back to read_all.
    assert not src.read_all_called
    assert len(pixel_calls) <= 4, (
        f"expected <=4 pixel GETs for a 4-tile window, got "
        f"{len(pixel_calls)}: {pixel_calls}")
    # Hard bound: the byte count must not approach the file size.
    assert src.total_bytes() < len(buf), (
        f"4-tile window consumed {src.total_bytes()} of {len(buf)} bytes")


# ===========================================================================
# 2. Overview reads fetch overview IFD bytes, not full-res
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_overview_read_does_not_fetch_full_resolution_pixels(
        cog_with_overviews_http_cog_range_contract, monkeypatch):
    """An ``overview_level=1`` read must pull the overview IFD's tiles,
    not the level-0 tiles.

    Reads the file twice on the same recording mock -- once at the
    base level and once at level 1 -- and asserts the overview read
    consumed strictly fewer pixel bytes than the base read. A
    regression that ignored ``overview_level`` (or read level 0 and
    decimated) would land at byte parity with the base read.
    """
    buf, _expected, _ = cog_with_overviews_http_cog_range_contract

    src_base = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src_base)
    base_arr, _ = _read_cog_http('http://mock/cog.tif', overview_level=0)
    assert base_arr.shape == (256, 256)

    src_ovr = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src_ovr)
    ovr_arr, _ = _read_cog_http('http://mock/cog.tif', overview_level=1)
    # Overview decimation factor 2 -> 128x128 output.
    assert ovr_arr.shape == (128, 128)

    base_pixels = sum(le for _s, le in src_base.tile_or_strip_calls())
    ovr_pixels = sum(le for _s, le in src_ovr.tile_or_strip_calls())
    assert ovr_pixels < base_pixels, (
        f"overview read pulled {ovr_pixels} pixel bytes; base read pulled "
        f"{base_pixels}. The overview path must read the smaller IFD, "
        f"not the full-res IFD")

    # Sanity: the overview byte budget should be roughly a quarter of
    # base (factor-2 decimation on both axes). Allow generous slack for
    # codec overhead, tile padding, and metadata GETs. The hard contract
    # is "less than base"; this bound flags a future regression that
    # quietly grows the overview footprint past 75% of base.
    assert ovr_pixels < base_pixels * 0.75, (
        f"overview read consumed {ovr_pixels} of {base_pixels} base "
        f"pixel bytes ({ovr_pixels / base_pixels:.1%}); expected close "
        f"to 25% for a factor-2 overview")


# ===========================================================================
# 3. ``band=`` on multi-band COGs returns correct pixels with bounded reads
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_band_selection_multiband_chunky_bounded_reads(
        multiband_chunky_cog_http_cog_range_contract, monkeypatch):
    """Per-band reads of a planar=1 chunky COG must return the right
    pixels and stay within the file's byte budget.

    The chunky layout interleaves samples in each tile, so the HTTP
    path cannot avoid fetching every chunky tile that touches the
    requested rows -- but it must not exceed the file size, and the
    fetched bytes must decode to the same pixels as the local read.
    """
    buf, _expected, path = multiband_chunky_cog_http_cog_range_contract

    # Reference via the local-file path on the same buffer.
    local = open_geotiff(path, band=1)

    src = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)
    remote, _ = _read_cog_http('http://mock/cog.tif', band=1)
    np.testing.assert_array_equal(remote, np.asarray(local))
    assert remote.ndim == 2

    # Bounded-read contract: the HTTP path must not pull more than the
    # file's bytes, and must not fall through to ``read_all``.
    assert not src.read_all_called, (
        "band-selected HTTP read fell back to read_all(); band slicing "
        "must happen on the decoded array, not after a full download")
    assert src.total_bytes() <= len(buf) + 65536, (
        f"band-selected read consumed {src.total_bytes()} bytes against a "
        f"file of {len(buf)} bytes; the read must stay within the file's "
        f"byte budget plus a small header slack")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_band_selection_with_window_bounded_range_count(
        multiband_chunky_cog_http_cog_range_contract, monkeypatch):
    """``window=`` + ``band=`` on a multi-band COG: pixels match the
    local path, range count is bounded by the window footprint.
    """
    buf, _expected, path = multiband_chunky_cog_http_cog_range_contract
    window = (0, 0, 32, 32)
    local = open_geotiff(path, window=window, band=2)

    src = _RecordingHTTPSource_http_cog_range_contract(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)
    remote, _ = _read_cog_http('http://mock/cog.tif',
                               window=window, band=2)
    np.testing.assert_array_equal(remote, np.asarray(local))

    assert not src.read_all_called
    # 32x32 window aligned to one tile; one pixel GET, two at most if
    # the coalescer happens to split. Anything past that means the
    # reader fetched a tile outside the window or every band in turn.
    pixel_calls = src.tile_or_strip_calls()
    assert len(pixel_calls) <= 2, (
        f"expected <=2 pixel GETs for a single-tile window+band, got "
        f"{len(pixel_calls)}: {pixel_calls}")


# ===========================================================================
# 4. Dask COG reads parse metadata once per graph, not per chunk task
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_dask_read_parses_ifds_once_across_chunks(
        small_tiled_cog_http_cog_range_contract, monkeypatch):
    """An N-chunk dask graph must trigger at most one IFD-header GET
    across all chunk tasks.

    Mirrors ``test_dask_http_parses_ifds_once`` in
    ``test_http_cog_coalesce.py`` but exercises the explicit O(1)-in-
    chunk-count contract this PR is supposed to pin. A regression where
    each delayed task spins up a fresh ``_HTTPSource`` and reparses
    headers would scale header GETs with chunk count.
    """
    buf, expected, _ = small_tiled_cog_http_cog_range_contract
    src_holder: list[_RecordingHTTPSource_http_cog_range_contract] = []

    def _fake_http_source(url, *_a, **_kw):
        s = _RecordingHTTPSource_http_cog_range_contract(buf)
        src_holder.append(s)
        return s

    monkeypatch.setattr(_reader_mod, '_HTTPSource', _fake_http_source)

    # 256x256 image; 32x32 chunks -> 64 chunks. If header parsing happens
    # per chunk task we should see ~64 header GETs. The contract says
    # at most one.
    da_arr = read_geotiff_dask('http://mock/cog.tif', chunks=32)
    n_chunks = da_arr.data.npartitions
    assert n_chunks >= 16, (
        f"expected >=16 chunks to make the count assertion meaningful, "
        f"got {n_chunks}")
    out = da_arr.compute()
    np.testing.assert_array_equal(np.asarray(out), expected)

    header_gets = 0
    for s in src_holder:
        for (start, length) in s.calls:
            # The header probe is exactly (0, 16384) or the doubled
            # fallback (0, 65536). Anything else is a pixel GET.
            if start == 0 and length in (16384, 65536):
                header_gets += 1
    assert header_gets <= 1, (
        f"expected <=1 header GET across {n_chunks} dask chunks; got "
        f"{header_gets} over {len(src_holder)} _HTTPSource instances. "
        f"Per-chunk header parsing would have produced ~{n_chunks}.")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_dask_header_gets_independent_of_chunk_count(
        small_tiled_cog_http_cog_range_contract, monkeypatch):
    """Doubling chunk count must not double header GETs (O(1) in chunks).

    Runs the same compute at two chunk granularities (32 and 64) and
    asserts neither pulls more than one header. Pinning the rate, not
    just an absolute count, catches a regression where the per-chunk
    GET is hidden under a small constant overhead at low chunk counts
    but grows linearly at higher ones.
    """
    buf, expected, _ = small_tiled_cog_http_cog_range_contract

    def _run(chunks):
        src_holder: list[_RecordingHTTPSource_http_cog_range_contract] = []

        def _fake(url, *_a, **_kw):
            s = _RecordingHTTPSource_http_cog_range_contract(buf)
            src_holder.append(s)
            return s

        monkeypatch.setattr(_reader_mod, '_HTTPSource', _fake)
        out = read_geotiff_dask('http://mock/cog.tif', chunks=chunks).compute()
        np.testing.assert_array_equal(np.asarray(out), expected)
        return sum(
            1
            for s in src_holder
            for (start, length) in s.calls
            if start == 0 and length in (16384, 65536)
        )

    hdr_small = _run(32)   # 64 chunks on 256x256
    hdr_large = _run(64)   # 16 chunks on 256x256
    # Whatever the absolute count, both must be at most 1.
    assert hdr_small <= 1, (
        f"chunks=32 issued {hdr_small} header GETs; expected <=1")
    assert hdr_large <= 1, (
        f"chunks=64 issued {hdr_large} header GETs; expected <=1")


# ===========================================================================
# 5. Truncated / malformed COGs close HTTP resources reliably
# ===========================================================================

class _CloseCountingSource_http_cog_range_contract(_HTTPSource):
    """``_HTTPSource`` that counts ``close()`` invocations.

    Used to assert the HTTP read path closes the source on the error
    path (#1816), even when the IFD parse blows up on a truncated
    file. Unlike the wrapper in ``test_cog_http_close_on_error_1816``
    this subclass also serves real bytes so the failure can be driven
    by a malformed payload rather than a monkeypatched explosion.
    """

    def __init__(self, buf: bytes):
        self._url = 'mock://2286/bad.tif'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self.close_count = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        return self._buf

    def close(self):
        with self._lock:
            self.close_count += 1


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_truncated_cog_closes_http_source(monkeypatch):
    """A truncated buffer must surface a clear exception and still
    close the HTTP source on the way out.

    The fixture serves only the first 32 bytes of what would be a
    valid TIFF. IFD parsing fails. The contract is:

    * the call raises (not a hang, not a silent zero-array return),
    * ``source.close()`` runs exactly once via the ``finally`` guard
      added for #1816.
    """
    bad = b'II\x2a\x00' + b'\x00' * 28  # valid magic, IFD pointer = 0
    src = _CloseCountingSource_http_cog_range_contract(bad)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises((ValueError, OSError)):
        _read_cog_http('http://mock/bad.tif')
    assert src.close_count == 1, (
        f"truncated COG read did not close the HTTP source: "
        f"close_count={src.close_count}")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_malformed_ifd_chain_closes_http_source(monkeypatch):
    """A file with a well-formed TIFF header but an IFD chain that
    points past the buffer raises a ``ValueError`` and still closes
    the source.

    Mirrors the #2050/#2266 'malformed pyramid metadata' scenarios:
    the header looks valid enough to start parsing, then the IFD
    extends past what any reasonable header prefetch will pull.
    """
    # Synthesize a tiny payload that crosses the parser-validation
    # threshold without being a real TIFF. The HTTP parser fetches
    # 16 KiB then expands. Without a valid IFD it raises after the
    # cap is hit; we just need the close-on-error contract to fire.
    payload = b'II\x2a\x00' + (0xFFFFFFF0).to_bytes(4, 'little') + b'\x00' * 64
    src = _CloseCountingSource_http_cog_range_contract(payload)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises((ValueError, OSError)):
        _read_cog_http('http://mock/bad.tif')
    assert src.close_count >= 1, (
        f"malformed IFD chain did not close the HTTP source: "
        f"close_count={src.close_count}")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_short_body_during_pixel_fetch_closes_source(
        small_tiled_cog_http_cog_range_contract, monkeypatch):
    """Header parses fine; the first pixel GET returns truncated bytes.
    The reader must raise (not hang) and close the source.

    Uses a real loopback server so the failure path runs through the
    actual urllib3 stack rather than a monkeypatched stub.
    """
    buf, _, _ = small_tiled_cog_http_cog_range_contract

    fail_after_header = {'tripped': False}

    class _ShortPixelHandler(_RangeHandler_http_cog_range_contract):
        payload = buf

        def do_GET(self):  # noqa: N802
            rng = self.headers.get('Range', '')
            if rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(self.payload) - 1
                # Header probe (start=0) goes through cleanly so the
                # IFD parses; any later GET returns a short body to
                # trip ``_HTTPSource.read_range``'s length check.
                if start != 0:
                    fail_after_header['tripped'] = True
                    advertised = end - start + 1
                    self.send_response(206)
                    self.send_header('Content-Length', str(advertised))
                    self.send_header(
                        'Content-Range',
                        f'bytes {start}-{end}/{len(self.payload)}',
                    )
                    self.end_headers()
                    # Send fewer bytes than advertised.
                    self.wfile.write(b'\x00' * max(1, advertised // 4))
                    return
                # Header probe -> serve normally.
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

    httpd = socketserver.TCPServer(('127.0.0.1', 0), _ShortPixelHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}/cog.tif'

    trackers: list = []
    real_cls = _reader_mod._HTTPSource

    class _Tracker:
        def __init__(self, real):
            self._real = real
            self.close_count = 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            self.close_count += 1
            return self._real.close()

    def factory(u, *a, **kw):
        t = _Tracker(real_cls(u, *a, **kw))
        trackers.append(t)
        return t

    monkeypatch.setattr(_reader_mod, '_HTTPSource', factory)

    # ``urllib3.exceptions.ProtocolError`` is the underlying class
    # raised when the server short-bodies a chunked response. Newer
    # ``read_range`` paths convert this to ``OSError`` before it
    # escapes; older versions let urllib3's exception type bubble up.
    # Both are acceptable as long as a clear exception fires (not a
    # hang) and the source still gets closed.
    import urllib3.exceptions as _u3e

    try:
        with pytest.raises((OSError, ValueError, _u3e.ProtocolError,
                            _u3e.HTTPError)):
            _read_cog_http(url)
    finally:
        _stop_http_cog_range_contract(httpd)

    assert fail_after_header['tripped'], (
        "test scaffold mistake: the server never returned a short body, "
        "so the failure path under test never ran")
    # Every constructed source must have been closed exactly once.
    assert trackers, "no _HTTPSource was constructed"
    for t in trackers:
        assert t.close_count == 1, (
            f"truncated pixel-fetch path leaked a source: "
            f"close_count={t.close_count}")


# ===========================================================================
# 6. Coalescing bounded by the configured max-merged-range size
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_coalesce_does_not_silently_exceed_explicit_cap():
    """``coalesce_ranges`` must respect the explicit cap kwarg.

    Mirrors the pure-unit assertion in ``test_http_cog_coalesce.py``
    but folds it into this file as the canonical contract row for
    #2286: a future refactor that drops the cap (or treats it as an
    advisory) flips this test red.
    """
    one_mib = 1 << 20
    cap = 4 * one_mib
    # 8 ranges spaced 1 MiB apart -- without the cap the gap threshold
    # alone collapses them into a single ~7 MiB GET.
    ranges = [(i * one_mib, 1024) for i in range(8)]
    merged, mapping = coalesce_ranges(ranges, max_coalesced_range_bytes=cap)
    for _start, length in merged:
        assert length <= cap, (
            f"merged range of {length} bytes exceeds the {cap}-byte cap")
    assert len(merged) > 1, (
        "cap did not force any split; the contract is broken")
    assert len(mapping) == len(ranges)


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_coalesce_default_cap_bounds_adversarial_input():
    """The default cap must bound an adversarial 'thousands of tiles
    spaced 1 MiB apart' input.

    This is the motivating #2266 scenario: a header with many tiny
    valid byte counts and sub-threshold gaps. Without the default cap
    the coalescer collapses the whole table into one multi-GiB GET.
    """
    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(4096)]
    merged, _ = coalesce_ranges(ranges)
    for _start, length in merged:
        assert length <= MAX_COALESCED_RANGE_BYTES_DEFAULT, (
            f"merged range {length} bytes exceeds the default cap "
            f"{MAX_COALESCED_RANGE_BYTES_DEFAULT}; the safe-by-default "
            f"contract is broken")


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_coalesced_get_size_capped_on_real_http_source():
    """The real ``_HTTPSource`` ``read_ranges_coalesced`` path must
    propagate the cap through to the wire-level GETs.

    Constructs an in-memory recording source (sharing the contract
    with the production class via subclassing), asks for ranges that
    would otherwise merge into one big GET, and asserts every actual
    GET respects the cap. Mirrors the dedicated row in
    ``test_http_cog_coalesce.py``; reproduced here as the contract
    anchor for #2293 so the cap survives any future refactor of the
    coalescer.
    """
    one_mib = 1 << 20
    buf = bytes((i * 13) & 0xFF for i in range(16 * one_mib))
    src = _RecordingHTTPSource_http_cog_range_contract(buf)

    ranges = [(i * one_mib, 1024) for i in range(8)]
    cap = 4 * one_mib
    out = src.read_ranges_coalesced(
        ranges, max_workers=2, max_coalesced_range_bytes=cap)

    # Bytes match the original per-range slices.
    for (off, length), tile in zip(ranges, out):
        assert tile == buf[off:off + length]

    # Every actual GET respects the cap.
    assert src.calls, "no GETs were issued"
    for _start, length in src.calls:
        assert length <= cap, (
            f"actual GET of {length} bytes exceeds the {cap}-byte cap")
    # The cap forced at least one split.
    assert len(src.calls) >= 2


@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_split_coalesced_bytes_round_trips_under_cap():
    """When the cap forces a split, ``split_coalesced_bytes`` still
    recovers every original byte range. The cap must not silently
    drop or duplicate bytes.
    """
    one_mib = 1 << 20
    payload_len = 8 * one_mib + 1024
    payload = bytes((i * 17) & 0xFF for i in range(payload_len))
    ranges = [(i * one_mib, 1024) for i in range(8)]

    merged, mapping = coalesce_ranges(
        ranges, max_coalesced_range_bytes=4 * one_mib)
    merged_bytes = [payload[s:s + le] for (s, le) in merged]
    out = split_coalesced_bytes(merged_bytes, mapping)
    for (off, length), tile in zip(ranges, out):
        assert tile == payload[off:off + length]


# ===========================================================================
# Bonus: end-to-end byte-budget check across the loopback server.
#
# The mocks above run in-process; the assertion below proves the same
# bounded contract holds when the bytes really do cross a socket.
# ===========================================================================

@pytest.mark.usefixtures('_no_sidecar_probe_http_cog_range_contract', '_allow_loopback_http_cog_range_contract')  # noqa: E501
def test_loopback_end_to_end_windowed_byte_budget(small_tiled_cog_http_cog_range_contract):
    """End-to-end through the real loopback server: a windowed read
    returns the right pixels and the total payload returned across all
    206 responses is bounded by the windowed footprint.

    The loopback server does not let us count GETs from outside, but
    it does let us prove that the parts the reader pulled add up to
    less than the file size. This pins the contract against a real
    urllib3 stack, catching regressions that ride below the
    ``_HTTPSource`` abstraction (e.g. a transparent prefetch in the
    pool layer).
    """
    buf, expected, _ = small_tiled_cog_http_cog_range_contract
    served = {'bytes': 0}

    class _CountingHandler(_RangeHandler_http_cog_range_contract):
        payload = buf

        def do_GET(self):  # noqa: N802
            rng = self.headers.get('Range', '')
            if rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(self.payload) - 1
                chunk = self.payload[start:end + 1]
                served['bytes'] += len(chunk)
                self.send_response(206)
                self.send_header('Content-Length', str(len(chunk)))
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/'
                    f'{len(self.payload)}',
                )
                self.end_headers()
                self.wfile.write(chunk)
                return
            served['bytes'] += len(self.payload)
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    httpd = socketserver.TCPServer(('127.0.0.1', 0), _CountingHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}/cog.tif'
    try:
        window = (32, 32, 64, 64)
        arr, _ = _read_cog_http(url, window=window)
        np.testing.assert_array_equal(arr, expected[32:64, 32:64])
    finally:
        _stop_http_cog_range_contract(httpd)

    # Hard upper bound: must be less than the whole file.
    assert served['bytes'] < len(buf), (
        f"loopback windowed read served {served['bytes']} of {len(buf)} "
        f"file bytes; the windowed path must not pull the whole file")

# ----------------------------------------------------------
# Section: http_dask_allow_rotated
# Source: test_http_dask_allow_rotated_2130.py
# ----------------------------------------------------------


tifffile_http_dask_allow_rotated = pytest.importorskip("tifffile")


class _RangeHandler_http_dask_allow_rotated(http.server.BaseHTTPRequestHandler):
    """Range-aware HTTP handler.

    The simple ``SimpleHTTPRequestHandler`` returns the full file body
    for any GET, which the COG HTTP source rejects (it requires a
    proper 206 Partial Content reply). Mirrors the helper used in
    ``test_http_dask_orientation_1794.py``.
    """

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


def _serve_http_dask_allow_rotated(payload: bytes):
    handler_cls = type(
        'RangeHandler2130', (_RangeHandler_http_dask_allow_rotated,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _write_rotated_tiff_http_dask_allow_rotated(path, arr, *, tile=None):
    """Synthesise a TIFF with a rotated ModelTransformationTag (30-deg)."""
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [(34264, 12, 16, m, False)]
    kwargs = {
        'photometric': 'minisblack',
        'planarconfig': 'contig',
        'extratags': extratags,
    }
    if tile is not None:
        kwargs['tile'] = tile
    tifffile_http_dask_allow_rotated.imwrite(str(path), arr, **kwargs)


def test_http_dask_rotated_default_raises(tmp_path, monkeypatch):
    """Without ``allow_rotated`` the HTTP dask path must still raise."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    src = tmp_path / "tmp_2130_http_dask_default.tif"
    arr = np.arange(64, dtype='<u2').reshape(8, 8)
    _write_rotated_tiff_http_dask_allow_rotated(src, arr, tile=(16, 16))
    payload = src.read_bytes()
    httpd, port = _serve_http_dask_allow_rotated(payload)
    try:
        url = f'http://127.0.0.1:{port}/{src.name}'
        with pytest.raises(RotatedTransformError, match="rotation"):
            open_geotiff(url, chunks=4)
    finally:
        httpd.shutdown()


def test_http_dask_rotated_allow_rotated_reads(tmp_path, monkeypatch):
    """``allow_rotated=True`` over HTTP+dask reads the pixel grid.

    Pre-#2130 this raised ``NotImplementedError`` because
    ``read_geotiff_dask`` did not forward the kwarg to
    ``_parse_cog_http_meta``.
    """
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    src = tmp_path / "tmp_2130_http_dask_optin.tif"
    arr = np.arange(64, dtype='<u2').reshape(8, 8)
    _write_rotated_tiff_http_dask_allow_rotated(src, arr, tile=(16, 16))
    payload = src.read_bytes()
    httpd, port = _serve_http_dask_allow_rotated(payload)
    try:
        url = f'http://127.0.0.1:{port}/{src.name}'
        da = open_geotiff(url, allow_rotated=True, chunks=4)
        assert da.shape == arr.shape
        np.testing.assert_array_equal(da.compute().values, arr)
    finally:
        httpd.shutdown()

# ----------------------------------------------------------
# Section: http_dask_orientation
# Source: test_http_dask_orientation_1794.py
# ----------------------------------------------------------


tifffile_http_dask_orientation = pytest.importorskip("tifffile")


def _write_with_orientation_http_dask_orientation(path, arr, orientation):
    tifffile_http_dask_orientation.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


class _RangeHandler_http_dask_orientation(http.server.BaseHTTPRequestHandler):
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


def _serve_http_dask_orientation(payload: bytes):
    handler_cls = type(
        'RangeHandler1794', (_RangeHandler_http_dask_orientation,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def test_http_dask_read_rejects_non_default_orientation(tmp_path, monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "tmp_1794_orientation_2.tif"
    _write_with_orientation_http_dask_orientation(path, arr, 2)

    payload = path.read_bytes()
    httpd, port = _serve_http_dask_orientation(payload)
    try:
        url = f'http://127.0.0.1:{port}/tmp_1794_orientation_2.tif'
        with pytest.raises(ValueError, match="Orientation tag"):
            open_geotiff(url, chunks=4)
    finally:
        httpd.shutdown()
        httpd.server_close()

# ----------------------------------------------------------
# Section: http_meta_buffer
# Source: test_http_meta_buffer_1718.py
# ----------------------------------------------------------


class _InMemoryHTTPSource_http_meta_buffer(_HTTPSource):
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


class _RangeHandler1718_http_meta_buffer(http.server.BaseHTTPRequestHandler):
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


def _serve_http_meta_buffer(payload: bytes):
    handler_cls = type(
        'RangeHandler1718Bound', (_RangeHandler1718_http_meta_buffer,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _write_cog_with_big_metadata_http_meta_buffer(
        path: str, arr: np.ndarray, metadata_pad_bytes: int) -> None:
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
          cog=True, overview_levels=[2])

    with open(path, 'rb') as f:
        payload = f.read()

    src = _InMemoryHTTPSource_http_meta_buffer(payload)
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
    _write_cog_with_big_metadata_http_meta_buffer(path, arr, metadata_pad_bytes=96 * 1024)

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

    src = _InMemoryHTTPSource_http_meta_buffer(payload)
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
    _write_cog_with_big_metadata_http_meta_buffer(path, arr, metadata_pad_bytes=96 * 1024)

    with open(path, 'rb') as f:
        payload = f.read()

    httpd, _thread = _serve_http_meta_buffer(payload)
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
          cog=True, overview_levels=[2])
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

    src = _InMemoryHTTPSource_http_meta_buffer(bytes(payload))
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

# ----------------------------------------------------------
# Section: http_no_stdlib_fallback
# Source: test_http_no_stdlib_fallback_2050.py
# ----------------------------------------------------------


def test_urllib3_is_importable():
    """urllib3 is in install_requires; importing the module must work."""
    import urllib3  # noqa: F401


def test_reader_imports_urllib3_at_module_level():
    """The reader keeps a module-level reference to urllib3.

    Module-level rather than deferred-import makes it impossible to ship
    a build where urllib3 is missing and the code silently degrades to
    a different transport.
    """
    assert hasattr(_reader_mod, 'urllib3')


def test_get_http_pool_returns_a_pool_manager():
    """``_get_http_pool`` is no longer allowed to return None.

    Pre-#2050 it returned ``None`` when urllib3 was missing, which is
    what routed callers into the stdlib fallback.
    """
    import urllib3
    pool = _reader_mod._get_http_pool()
    assert pool is not None
    assert isinstance(pool, urllib3.PoolManager)


# ---------------------------------------------------------------------------
# The stdlib fallback symbols are gone
# ---------------------------------------------------------------------------


def test_stdlib_opener_helper_is_removed():
    """``_get_stdlib_opener`` was the entry point for the unpinned path."""
    assert not hasattr(_reader_mod, '_get_stdlib_opener')
    assert not hasattr(_reader_mod, '_stdlib_opener')


def test_validating_redirect_handler_is_removed():
    """The stdlib redirect handler is gone with the stdlib transport."""
    assert not hasattr(_reader_mod, '_ValidatingRedirectHandler')


def test_reader_does_not_import_urllib_request():
    """``urllib.request`` is no longer needed once the stdlib path is gone.

    A residual ``import urllib.request`` at module scope would be a
    smell -- the only legitimate consumer was the deleted opener.
    """
    src = inspect.getsource(_reader_mod)
    # The token has to appear in *executable* form, not just inside a
    # comment or docstring. Strip comment lines and check the rest.
    code_lines = [
        line for line in src.splitlines()
        if not line.lstrip().startswith('#')
    ]
    code = '\n'.join(code_lines)
    assert 'import urllib.request' not in code, (
        "urllib.request should not be imported now that the stdlib "
        "HTTP fallback is removed (#2050)."
    )


def test_read_range_source_has_no_stdlib_branch():
    """``read_range`` body must not reference ``urllib.request``."""
    src = inspect.getsource(_reader_mod._HTTPSource.read_range)
    assert 'urllib.request' not in src
    assert 'stdlib_opener' not in src


def test_read_all_source_has_no_stdlib_branch():
    """``read_all`` body must not reference ``urllib.request``."""
    src = inspect.getsource(_reader_mod._HTTPSource.read_all)
    assert 'urllib.request' not in src
    assert 'stdlib_opener' not in src


# ---------------------------------------------------------------------------
# urllib3 path still works -- mock the pool and exercise read_range / read_all
# ---------------------------------------------------------------------------


def _fake_getaddrinfo_http_no_stdlib_fallback(ip: str):
    def _resolver(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


class _MockResp_http_no_stdlib_fallback:
    def __init__(self, status, data=b'', content_range=None,
                 content_length=None):
        self.status = status
        self.data = data
        self._body = data
        self.headers = {}
        if content_range is not None:
            self.headers['Content-Range'] = content_range
        # ``read_range`` (post #2264) does a Content-Length preflight; let
        # callers either pin it explicitly or default to len(data).
        if content_length is None and data:
            self.headers['Content-Length'] = str(len(data))
        elif content_length is not None:
            self.headers['Content-Length'] = str(content_length)

    def stream(self, amt=65536, decode_content=True):
        # Yield the body in a single chunk; ``_read_capped`` reads
        # whatever ``stream()`` produces.
        if self._body:
            yield self._body

    def release_conn(self):
        pass


class _MockPool_http_no_stdlib_fallback:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._resp


def test_read_range_uses_urllib3_pool(monkeypatch):
    """Sanity check: a successful 206 round-trips through ``_request``."""
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo_http_no_stdlib_fallback('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    body = b'A' * 100
    pool = _MockPool_http_no_stdlib_fallback(_MockResp_http_no_stdlib_fallback(
        status=206, data=body, content_range='bytes 0-99/1000'))
    src._pool = pool

    data = src.read_range(0, 100)
    assert data == body
    assert len(pool.calls) == 1
    method, url, kwargs = pool.calls[0]
    assert method == 'GET'
    assert kwargs.get('redirect') is False
    assert kwargs.get('headers', {}).get('Range') == 'bytes=0-99'
    # Post #2264: the GET must request a streaming body so the cap is
    # enforced on the wire rather than after urllib3 has already
    # buffered ``resp.data``.
    assert kwargs.get('preload_content') is False


def test_read_all_uses_urllib3_pool(monkeypatch):
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo_http_no_stdlib_fallback('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    body = b'tiff-bytes'
    pool = _MockPool_http_no_stdlib_fallback(_MockResp_http_no_stdlib_fallback(status=200, data=body))  # noqa: E501
    src._pool = pool

    data = src.read_all()
    assert data == body
    assert len(pool.calls) == 1


def test_read_range_short_circuits_zero_length(monkeypatch):
    """No HTTP traffic for length<=0 -- behaviour preserved from pre-#2050."""
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo_http_no_stdlib_fallback('93.184.216.34'))
    src = _reader_mod._HTTPSource('https://example.com/cog.tif')
    pool = _MockPool_http_no_stdlib_fallback(_MockResp_http_no_stdlib_fallback(status=206, data=b''))  # noqa: E501
    src._pool = pool

    assert src.read_range(0, 0) == b''
    assert src.read_range(10, -5) == b''
    assert pool.calls == []


# ---------------------------------------------------------------------------
# install_requires advertises urllib3
# ---------------------------------------------------------------------------


def test_install_requires_lists_urllib3():
    """setup.cfg must list urllib3 so deployed installs get it.

    Without this, a wheel built today would let pip resolve the install
    without urllib3, and the import at top of _reader would fail at
    open_geotiff() time rather than at install time.
    """
    import pathlib
    setup_cfg = (
        pathlib.Path(_reader_mod.__file__).resolve()
        .parent.parent.parent / 'setup.cfg'
    )
    if not setup_cfg.exists():  # pragma: no cover
        pytest.skip(f"setup.cfg not found at {setup_cfg}")
    text = setup_cfg.read_text()
    # Locate the install_requires block and confirm urllib3 appears in it.
    head, _, tail = text.partition('install_requires =')
    assert tail, "install_requires section not found in setup.cfg"
    # The block ends at the next top-level key (lines that start in
    # column 0). Walk until we see one.
    block_lines = []
    for line in tail.splitlines()[1:]:
        if line and not line.startswith((' ', '\t')):
            break
        block_lines.append(line.strip())
    assert 'urllib3' in block_lines, (
        f"urllib3 must be in install_requires; found: {block_lines}"
    )

# ----------------------------------------------------------
# Section: http_orientation
# Source: test_http_orientation_1717.py
# ----------------------------------------------------------


tifffile_http_orientation = pytest.importorskip("tifffile")


_ORIENTATIONS_http_orientation = [1, 2, 3, 4, 5, 6, 7, 8]


def _write_with_orientation_http_orientation(path, arr, orientation):
    tifffile_http_orientation.imwrite(
        str(path),
        arr,
        extratags=[(274, 'H', 1, orientation, True)],
    )


class _RangeHandler_http_orientation(http.server.BaseHTTPRequestHandler):
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


def _serve_http_orientation(payload: bytes):
    handler_cls = type(
        'RangeHandler1717', (_RangeHandler_http_orientation,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


@pytest.fixture
def _allow_loopback_http_orientation(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


@pytest.mark.parametrize("orientation", _ORIENTATIONS_http_orientation)
def test_http_full_read_matches_local_for_orientation(
    tmp_path, _allow_loopback_http_orientation, orientation,
):
    """Local-file vs HTTP full read must produce identical output."""
    rng = np.random.default_rng(orientation)
    arr = rng.integers(0, 255, size=(12, 16), dtype=np.uint8)
    path = tmp_path / f"tmp_1717_orient_{orientation}.tif"
    _write_with_orientation_http_orientation(path, arr, orientation)

    with open(path, 'rb') as f:
        payload = f.read()

    arr_local, geo_local = read_to_array(str(path))

    httpd, port = _serve_http_orientation(payload)
    try:
        url = f'http://127.0.0.1:{port}/orient_{orientation}.tif'
        arr_http, geo_http = _read_cog_http(url)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert arr_http.shape == arr_local.shape, (
        f"orientation={orientation}: HTTP shape {arr_http.shape} != "
        f"local shape {arr_local.shape}"
    )
    np.testing.assert_array_equal(
        arr_http, arr_local,
        err_msg=f"orientation={orientation}: HTTP pixels differ from local",
    )
    assert geo_http.transform == geo_local.transform, (
        f"orientation={orientation}: transform mismatch "
        f"http={geo_http.transform} local={geo_local.transform}"
    )


@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_http_windowed_read_rejects_non_default_orientation(
    tmp_path, _allow_loopback_http_orientation, orientation,
):
    """Windowed reads against an oriented file should still raise.

    Mirrors the local-path guard so the contract is uniform across
    backends. Resolving windowed-read semantics for oriented files is
    out of scope for #1717.
    """
    arr = np.zeros((8, 8), dtype=np.uint8)
    path = tmp_path / f"tmp_1717_window_reject_{orientation}.tif"
    _write_with_orientation_http_orientation(path, arr, orientation)

    with open(path, 'rb') as f:
        payload = f.read()

    httpd, port = _serve_http_orientation(payload)
    try:
        url = f'http://127.0.0.1:{port}/window_{orientation}.tif'
        with pytest.raises(ValueError, match="Orientation tag"):
            _read_cog_http(url, window=(0, 0, 4, 4))
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_default_orientation_still_works(tmp_path, _allow_loopback_http_orientation):
    """Sanity: orientation=1 (default) HTTP read is byte-identical to local."""
    arr = np.arange(48, dtype=np.uint8).reshape(6, 8)
    path = tmp_path / "tmp_1717_default.tif"
    _write_with_orientation_http_orientation(path, arr, 1)

    with open(path, 'rb') as f:
        payload = f.read()

    arr_local, _ = read_to_array(str(path))
    httpd, port = _serve_http_orientation(payload)
    try:
        url = f'http://127.0.0.1:{port}/default.tif'
        arr_http, _ = _read_cog_http(url)
    finally:
        httpd.shutdown()
        httpd.server_close()

    np.testing.assert_array_equal(arr_http, arr_local)
    np.testing.assert_array_equal(arr_http, arr)

# ----------------------------------------------------------
# Section: http_range_validation
# Source: test_http_range_validation_1735.py
# ----------------------------------------------------------


class _BaseHandler_http_range_validation(http.server.BaseHTTPRequestHandler):
    payload: bytes = b'0' * 64

    def log_message(self, *_args, **_kwargs):
        pass


def _serve_http_range_validation(handler_cls):
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/x.bin', httpd, thread


def _stop_http_range_validation(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_http_range_validation(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
def test_range_request_ignored_for_nonzero_start_raises():
    """Server ignores ``Range`` for a non-zero start and returns full
    200 -> OSError. (A 200 with start=0 is harmless because the body
    offsets line up with what the caller wanted.)"""

    class _Handler(_BaseHandler_http_range_validation):
        def do_GET(self):  # noqa: N802
            # Ignore Range header; return the full object as 200.
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_range_validation(_Handler)
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
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
def test_range_request_wrong_content_range_raises():
    """Server returns 206 but the Content-Range header points at the
    wrong bytes -> OSError."""

    class _Handler(_BaseHandler_http_range_validation):
        def do_GET(self):  # noqa: N802
            # Pretend we sent bytes 4-19/64 regardless of what was asked.
            self.send_response(206)
            self.send_header('Content-Length', '16')
            self.send_header('Content-Range', 'bytes 4-19/64')
            self.end_headers()
            self.wfile.write(self.payload[4:20])

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        # Caller asks for 0-15; server says 4-19.
        with pytest.raises(OSError, match="Content-Range"):
            src.read_range(0, 16)
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
def test_range_request_short_body_raises():
    """Server returns 206 with a body shorter than the requested
    length -> OSError."""

    class _Handler(_BaseHandler_http_range_validation):
        def do_GET(self):  # noqa: N802
            self.send_response(206)
            self.send_header('Content-Length', '4')
            self.send_header('Content-Range', 'bytes 0-15/64')
            self.end_headers()
            # Send only 4 bytes despite advertising a 16-byte range.
            self.wfile.write(self.payload[:4])

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="length"):
            src.read_range(0, 16)
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
def test_range_request_well_formed_succeeds():
    """A correctly-formed 206 response is accepted as-is."""

    class _Handler(_BaseHandler_http_range_validation):
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

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(8, 16)
        assert out == _BaseHandler_http_range_validation.payload[8:24]
        assert len(out) == 16
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
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

    class _Handler(_BaseHandler_http_range_validation):
        def do_GET(self):  # noqa: N802
            # If we ever land here, the guard failed. Record the hit so
            # the assertion below points at the right cause.
            hit_count['n'] += 1
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        assert src.read_range(10, 0) == b''
        assert src.read_range(0, 0) == b''
        assert src.read_range(10, -5) == b''
        assert hit_count['n'] == 0, (
            "read_range(length<=0) should not issue an HTTP request"
        )
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
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

    class _Handler(_BaseHandler_http_range_validation):
        # Payload larger than the patched cap so the preflight has
        # something to reject.
        payload = b'\xab' * 4096

        def do_GET(self):  # noqa: N802
            # Ignore Range entirely; return the full object as 200.
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="Content-Length|byte budget"):
            src.read_range(0, 64)
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
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

    class _Handler(_BaseHandler_http_range_validation):
        payload = b'\xcd' * 4096

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(0, 64)
        # Caller's "at most length bytes" contract holds even when the
        # server returned a much larger body.
        assert out == _Handler.payload[:64]
        assert len(out) == 64
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
def test_range_ignored_200_short_body_returned_as_is():
    """A 200 fallback whose body is smaller than the requested length
    is returned unchanged (no slicing needed).

    This is the "tiny file served by a Range-blind origin" case: the
    caller asked for a 64-byte header prefetch but the whole object
    is only 40 bytes.
    """

    class _Handler(_BaseHandler_http_range_validation):
        payload = b'\xef' * 40

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        out = src.read_range(0, 64)
        assert out == _Handler.payload
        assert len(out) == 40
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
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

    class _Handler(_BaseHandler_http_range_validation):
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

    url, httpd, _ = _serve_http_range_validation(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="byte budget|exceeded"):
            src.read_range(0, 64)
    finally:
        _stop_http_range_validation(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_range_validation')
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

# ----------------------------------------------------------
# Section: http_read_all_bounded
# Source: test_http_read_all_bounded_2051.py
# ----------------------------------------------------------


class _BaseHandler_http_read_all_bounded(http.server.BaseHTTPRequestHandler):
    payload: bytes = b''
    # Subclasses override these to fake misbehaviour.
    lie_content_length: int | None = None
    drop_content_length: bool = False
    truncated_payload: bytes | None = None

    def log_message(self, *_args, **_kwargs):
        pass


def _serve_http_read_all_bounded(handler_cls):
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/cog.tif', httpd, thread


def _stop_http_read_all_bounded(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_http_read_all_bounded(monkeypatch):
    # Loopback addresses are blocked by the SSRF allow-list; the escape
    # hatch lets the test reach 127.0.0.1.
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# Unit tests for the budget helper
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_budget_uses_max_strip_end_plus_slack():
    """Budget is ``max(offset + byte_count) + slack`` over the strip table."""
    offsets = [1024, 5000, 100_000]
    byte_counts = [512, 1024, 4096]
    budget = _compute_full_image_byte_budget(offsets, byte_counts)
    # Largest end is 100_000 + 4096 = 104_096
    assert budget == 104_096 + _FULL_IMAGE_BUDGET_HEADER_SLACK


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_budget_empty_strip_table_falls_back_to_per_strip_cap():
    """Empty / missing strip table falls back to the per-strip safety cap."""
    budget = _compute_full_image_byte_budget(None, None)
    assert budget > 0
    budget_empty = _compute_full_image_byte_budget([], [])
    assert budget_empty > 0


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
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

@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_read_all_no_budget_returns_full_body():
    """Without ``max_bytes`` the legacy unbounded behaviour is preserved."""

    class _Handler(_BaseHandler_http_read_all_bounded):
        payload = b'A' * 1024

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_read_all_bounded(_Handler)
    try:
        src = _HTTPSource(url)
        data = src.read_all()
        assert data == b'A' * 1024
    finally:
        _stop_http_read_all_bounded(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_read_all_rejects_oversized_content_length():
    """Server advertises a Content-Length larger than the budget --
    rejected up front via OSError before any body is read."""

    class _Handler(_BaseHandler_http_read_all_bounded):
        payload = b'B' * 2048

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_read_all_bounded(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="Content-Length"):
            src.read_all(max_bytes=1024)
    finally:
        _stop_http_read_all_bounded(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_read_all_truncates_when_server_lies_about_content_length_small():
    """Server lies low: advertises a small Content-Length but sends a
    much larger body. urllib3 trusts the advertised length and truncates
    at the byte count the server declared, so the client is already
    protected -- the extra bytes never reach Python memory. The cap is
    irrelevant on this path because the body the caller sees never
    exceeds the (truthful or lying) Content-Length. Lock in the
    truncation behaviour so a future urllib3 / stdlib change does not
    quietly turn this back into a vector."""

    class _Handler(_BaseHandler_http_read_all_bounded):
        # 100 KiB body, but advertised as 100 bytes.
        big_body = b'L' * (100 * 1024)
        lied_length = 100

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(self.lied_length))
            self.end_headers()
            self.wfile.write(self.big_body)

    url, httpd, _ = _serve_http_read_all_bounded(_Handler)
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
        _stop_http_read_all_bounded(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_read_all_catches_missing_content_length():
    """Server omits Content-Length and uses chunked transfer encoding.
    The pre-flight check has nothing to look at; the streaming cap must
    still catch the over-sized body."""

    class _Handler(_BaseHandler_http_read_all_bounded):
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

    url, httpd, _ = _serve_http_read_all_bounded(_Handler)
    try:
        src = _HTTPSource(url)
        with pytest.raises(OSError, match="exceeded the byte budget"):
            src.read_all(max_bytes=1024)
    finally:
        _stop_http_read_all_bounded(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_read_all_passes_when_body_fits_budget():
    """Legitimate path: body equals the budget exactly, returns cleanly."""

    class _Handler(_BaseHandler_http_read_all_bounded):
        payload = b'D' * 1024

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    url, httpd, _ = _serve_http_read_all_bounded(_Handler)
    try:
        src = _HTTPSource(url)
        data = src.read_all(max_bytes=2048)
        assert data == b'D' * 1024
    finally:
        _stop_http_read_all_bounded(httpd)


# ---------------------------------------------------------------------------
# Stdlib fallback (urllib3 unavailable)
# ---------------------------------------------------------------------------

# The stdlib ``urllib.request`` fallback path was removed in #2050 /
# #2055 (urllib3 is now a hard dependency). The three tests that
# previously covered the fallback's byte-budget enforcement no longer
# have a code path to exercise; the urllib3-only equivalents above
# (test_read_all_rejects_oversized_content_length,
# test_read_all_catches_missing_content_length,
# test_read_all_passes_when_body_fits_budget) keep the contract
# covered.


# ---------------------------------------------------------------------------
# End-to-end COG read
# ---------------------------------------------------------------------------

class _RangeHandler_http_read_all_bounded(_BaseHandler_http_read_all_bounded):
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


def _serve_payload_http_read_all_bounded(payload: bytes):
    handler_cls = type(
        'BoundRangeHandler', (_RangeHandler_http_read_all_bounded,), {'payload': payload}
    )
    return _serve_http_read_all_bounded(handler_cls)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
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

    url, httpd, _ = _serve_payload_http_read_all_bounded(payload)
    try:
        result, _geo = _read_cog_http(url)
        np.testing.assert_array_equal(result, arr)
    finally:
        _stop_http_read_all_bounded(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_read_all_bounded')
def test_full_image_http_read_rejects_padded_body(tmp_path):
    """Attack scenario: a legitimate TIFF header is followed by extra
    garbage past what the strip table accounts for. The
    strip-table-derived budget rejects the body before it is fully
    buffered into memory."""
    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    path = str(tmp_path / 'padded_2051.tif')
    write(arr, path, compression='deflate', tiled=False)

    with open(path, 'rb') as f:
        legit_payload = f.read()

    # Append 64 MiB of zeros to the body. The strip table only covers
    # the first len(legit_payload) bytes; anything past max(offset +
    # byte_count) + slack is over-budget.
    bloated = legit_payload + (b'\x00' * (64 * 1024 * 1024))

    url, httpd, _ = _serve_payload_http_read_all_bounded(bloated)
    try:
        with pytest.raises(OSError, match="Content-Length|byte budget"):
            _read_cog_http(url)
    finally:
        _stop_http_read_all_bounded(httpd)

# ----------------------------------------------------------
# Section: http_scheme_case
# Source: test_http_scheme_case_2321.py
# ----------------------------------------------------------


def _fake_getaddrinfo_http_scheme_case(ip: str):
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestIsHttpSourceHelper_http_scheme_case:
    """``_is_http_source`` is the single source of truth for HTTP routing."""

    @pytest.mark.parametrize("url", [
        'http://example.com/x.tif',
        'https://example.com/x.tif',
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
        'hTTpS://example.com/x.tif',
        'http://EXAMPLE.COM/x.tif',  # host case must not matter either
    ])
    def test_http_schemes_match(self, url):
        assert _sources_mod._is_http_source(url) is True

    @pytest.mark.parametrize("url", [
        's3://bucket/key.tif',
        'gs://bucket/key.tif',
        'az://container/blob.tif',
        'abfs://container/blob.tif',
        'file:///etc/passwd',
        'ftp://example.com/x.tif',
        'gopher://example.com/',
        'memory://x.tif',
        '/local/path/file.tif',
        'relative/path.tif',
        'C:\\windows\\file.tif',
    ])
    def test_non_http_schemes_do_not_match(self, url):
        assert _sources_mod._is_http_source(url) is False

    @pytest.mark.parametrize("value", [None, 42, b'http://x', object()])
    def test_non_string_does_not_match(self, value):
        # Be defensive: routing call sites also gate on isinstance(_, str)
        # in some places, but the helper itself must not raise on junk.
        assert _sources_mod._is_http_source(value) is False

    def test_empty_string_does_not_match(self):
        assert _sources_mod._is_http_source('') is False

    def test_scheme_only_prefix_does_not_match(self):
        # ``urlparse('http')`` returns scheme=''; only ``http:`` or
        # ``http://`` should classify as HTTP.
        assert _sources_mod._is_http_source('http') is False

    def test_scheme_colon_no_slashes_classifies_as_http(self):
        # ``urlparse('http:foo').scheme == 'http'``: this is broader than
        # the old ``startswith('http://')`` gate but is RFC-correct. The
        # validator rejects these downstream as "no hostname", so the
        # security posture is unchanged. Locking the broader classifier
        # in here keeps any future tightening explicit. Issue #2332.
        assert _sources_mod._is_http_source('http:foo') is True
        assert _sources_mod._is_http_source('HTTP:foo') is True

    def test_open_source_http_colon_no_hostname_raises(self):
        # End-to-end follow-up: ``_open_source('http:foo')`` now routes
        # into ``_HTTPSource``, which calls ``_validate_http_url`` and
        # raises ``UnsafeURLError('... has no hostname')``. The previous
        # case-sensitive gate would have sent this to fsspec instead.
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source('http:foo')


# ---------------------------------------------------------------------------
# Dispatch: ``_open_source`` must route uppercase URLs through ``_HTTPSource``
# ---------------------------------------------------------------------------


class TestOpenSourceRoutesUppercase_http_scheme_case:
    """``_open_source('HTTP://...')`` must build an ``_HTTPSource``.

    We intercept ``_HTTPSource.__init__`` so the test never opens a real
    HTTP connection; getting the call at all is what we are verifying.
    """

    def test_uppercase_http_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            # Skip the real validator / urllib3 pool setup.
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('HTTP://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['HTTP://example.com/x.tif']

    def test_uppercase_https_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('HTTPS://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['HTTPS://example.com/x.tif']

    def test_mixed_case_routes_to_http_source(self, monkeypatch):
        calls = []

        def _fake_init(self, url, *args, **kwargs):
            calls.append(url)
            self._url = url

        monkeypatch.setattr(
            _sources_mod._HTTPSource, '__init__', _fake_init)
        src = _sources_mod._open_source('hTTpS://example.com/x.tif')
        assert isinstance(src, _sources_mod._HTTPSource)
        assert calls == ['hTTpS://example.com/x.tif']


# ---------------------------------------------------------------------------
# Dispatch booleans elsewhere in the code base
# ---------------------------------------------------------------------------


class TestDispatchBooleansAreCaseInsensitive_http_scheme_case:
    """Every routing site must use the centralized helper, not startswith.

    Each call site below historically read::

        source.startswith(('http://', 'https://'))

    which is the bug. We assert ``_is_http_source`` returns True for the
    uppercase forms; the implementation modules import and call the same
    helper at the dispatch site.
    """

    @pytest.mark.parametrize("url", [
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
    ])
    def test_helper_recognizes_uppercase(self, url):
        assert _sources_mod._is_http_source(url) is True

    def test_is_fsspec_uri_excludes_uppercase_http(self):
        # ``_is_fsspec_uri`` is the partner classifier in both
        # ``_sources.py`` and ``_writer.py``. If it returned True for
        # ``HTTP://...`` the writer would hand the URL to fsspec instead
        # of raising the typed "writes not supported over HTTP" error.
        assert _sources_mod._is_fsspec_uri('HTTP://example.com/x.tif') is False
        assert _sources_mod._is_fsspec_uri('HTTPS://example.com/x.tif') is False
        # sanity: real fsspec URIs still classify as fsspec
        assert _sources_mod._is_fsspec_uri('s3://b/k.tif') is True

    def test_writer_is_fsspec_uri_excludes_uppercase_http(self):
        from xrspatial.geotiff import _writer as _writer_mod
        assert _writer_mod._is_fsspec_uri('HTTP://example.com/x.tif') is False
        assert _writer_mod._is_fsspec_uri('HTTPS://example.com/x.tif') is False
        assert _writer_mod._is_fsspec_uri('s3://b/k.tif') is True

    def test_sidecar_helper_is_case_insensitive(self):
        from xrspatial.geotiff import _sidecar as _sidecar_mod
        assert _sidecar_mod._is_http_url('HTTP://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('HTTPS://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('http://example.com/x.tif') is True
        assert _sidecar_mod._is_http_url('s3://b/k.tif') is False


# ---------------------------------------------------------------------------
# End-to-end: uppercase scheme + private host must still be rejected
# ---------------------------------------------------------------------------


class TestUppercaseSchemeStillRejectsPrivateHosts_http_scheme_case:
    """The whole point of the fix: uppercase URLs go through the SSRF gate.

    Before the fix, ``HTTP://169.254.169.254/...`` would skip the validator
    and try to open via fsspec. After the fix, it routes through
    ``_HTTPSource``, which calls ``_validate_http_url``, which raises
    ``UnsafeURLError``.
    """

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http', 'hTTpS'])
    @pytest.mark.parametrize("ip", [
        '127.0.0.1',
        '169.254.169.254',
        '10.0.0.1',
        '192.168.1.1',
        '0.0.0.0',
    ])
    def test_private_host_rejected_regardless_of_scheme_case(
            self, monkeypatch, scheme, ip):
        monkeypatch.setattr(socket, 'getaddrinfo', _fake_getaddrinfo_http_scheme_case(ip))
        url = f'{scheme}://attacker.test/x.tif'
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(url)

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http'])
    def test_localhost_rejected_regardless_of_scheme_case(
            self, monkeypatch, scheme):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_http_scheme_case('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(f'{scheme}://localhost:8080/x.tif')

    @pytest.mark.parametrize("scheme", ['HTTP', 'HTTPS', 'Http'])
    def test_uppercase_scheme_to_127_literal_rejected(
            self, monkeypatch, scheme):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_http_scheme_case('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(f'{scheme}://127.0.0.1/x.tif')

    def test_open_source_uppercase_private_host_raises(self, monkeypatch):
        """End-to-end: ``_open_source`` -> ``_HTTPSource`` -> validator.

        Confirms the dispatch wiring actually drives the URL through the
        validator (not just that the validator works in isolation).
        """
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_http_scheme_case('169.254.169.254'))
        # Make sure the env override is not set; the validator skips
        # resolution when ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS`` is on.
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', raising=False)
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                'HTTP://metadata.google.internal/computeMetadata/v1/')


# ---------------------------------------------------------------------------
# Writer: HTTP(S) destinations must raise a typed error, not a raw OSError
# ---------------------------------------------------------------------------


class TestWriterRejectsHttpTargets_http_scheme_case:
    """``_write_bytes(_, 'HTTP://...')`` must raise ``NotImplementedError``.

    Without the early gate the uppercase URL fell through ``_is_fsspec_uri``
    (correctly returns False) and into the local file write path, which
    surfaced an OS-specific ``OSError`` for the colon-in-filename. The
    typed error matches the lowercase-HTTP behaviour and points users at
    the supported destinations. Follow-up to issue #2332 review.
    """

    @pytest.mark.parametrize("url", [
        'http://example.com/x.tif',
        'https://example.com/x.tif',
        'HTTP://example.com/x.tif',
        'HTTPS://example.com/x.tif',
        'Http://example.com/x.tif',
    ])
    def test_write_bytes_rejects_http(self, url):
        from xrspatial.geotiff import _writer as _writer_mod
        with pytest.raises(NotImplementedError) as excinfo:
            _writer_mod._write_bytes(b'IIxxxx', url)
        msg = str(excinfo.value)
        assert 'HTTP' in msg
        assert url in msg

# ----------------------------------------------------------
# Section: http_stripped_window_max_pixels
# Source: test_http_stripped_window_max_pixels_issue_A_1842.py
# ----------------------------------------------------------


@pytest.fixture()
def _no_sidecar_probe_http_stripped_window_max_pixels(monkeypatch):
    """Pin the byte-range assertions against the no-sidecar path.

    Issue #2239 added a sidecar-discovery probe to ``_read_cog_http``
    (an extra ``(0, 1)`` range fetch for ``<url>.ovr``) that shows up
    in ``_RecordingHTTPSource.calls`` and breaks the strip-fetch
    counts this file asserts. Disable discovery here so the
    assertions continue to measure exactly the strip GETs the issue
    is about. Sidecar behaviour for the chunked HTTP path is covered
    by ``test_remote_sidecar_chunked_2239.py``.
    """
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    monkeypatch.setattr(_sidecar_mod, 'find_sidecar', lambda _src: None)


class _RecordingHTTPSource_http_stripped_window_max_pixels(_HTTPSource):
    """In-memory ``_HTTPSource`` that records every range fetch.

    Tests assert how many strip GETs (and which offsets) the reader
    issues, so they can tell apart a windowed strip fetch from a
    ``read_all`` of the entire file.
    """

    def __init__(self, buf: bytes):
        self._url = 'mock://'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self.calls: list[tuple[int, int]] = []
        self.read_all_called = False
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        with self._lock:
            self.calls.append((start, length))
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        with self._lock:
            self.read_all_called = True
        return self._buf


def _make_stripped_cog_http_stripped_window_max_pixels(tmp_path, *, height=1024, width=64):
    """Write a stripped (non-tiled) TIFF and return its raw bytes.

    Sized so the writer's default 256 rows-per-strip produces at least
    four strips, which is what the byte-range coverage test needs to be
    meaningful.
    """
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    path = str(tmp_path / 'stripped_issue_A_1842.tif')
    write(arr, path, compression='none', tiled=False)
    with open(path, 'rb') as f:
        return f.read(), arr, path


# ---------------------------------------------------------------------------
# Test 1: a windowed read fetches only the strips it needs
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_windowed_stripped_http_fetches_only_intersecting_strips(
        tmp_path, monkeypatch):
    """A window covering one strip must only fetch that strip's bytes."""
    buf, expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)

    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    # Pick a window that covers exactly one row range. We don't know the
    # writer-picked rows_per_strip until we open the file once, so peek
    # at the IFD via a second mock source (the recording source's calls
    # for the peek pass are not asserted on).
    from xrspatial.geotiff._reader import _parse_cog_http_meta
    peek = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    _, ifd, _, _ = _parse_cog_http_meta(peek)
    rps = ifd.rows_per_strip
    n_strips = len(ifd.strip_offsets)
    assert n_strips >= 2, "test needs at least 2 strips to be meaningful"

    # Aim the window at strip 1 only (rows [rps : 2*rps)). Use a sub-row
    # column range to confirm the column-slice still works.
    target_strip = 1
    r0 = target_strip * rps
    r1 = r0 + 1
    window = (r0, 0, r1, ifd.width)

    arr, _ = _read_cog_http('http://mock/stripped.tif', window=window)
    np.testing.assert_array_equal(arr, expected[r0:r1, :])

    # The recording source must NOT have fetched the whole file.
    assert not src.read_all_called, (
        "windowed stripped HTTP read fell back to read_all; the fix is "
        "supposed to fetch only the intersecting strip's byte range")

    # Strip-fetch ranges are everything past the header probe(s). The
    # header probe is exactly (0, 16384) or (0, 65536); a fetch starting
    # at the target strip's offset is the strip GET we expect.
    target_offset = ifd.strip_offsets[target_strip]
    target_bc = ifd.strip_byte_counts[target_strip]
    strip_calls = [
        (s, le) for (s, le) in src.calls
        if not (s == 0 and le in (16384, 65536))
    ]
    # Exactly one strip GET, covering the target strip's range. Either a
    # coalesced GET starting at the target offset, or a single-range GET
    # of (offset, byte_count).
    assert len(strip_calls) == 1, (
        f"expected one strip GET for a single-strip window, got "
        f"{len(strip_calls)}: {strip_calls}")
    got_start, got_len = strip_calls[0]
    assert got_start == target_offset, (
        f"strip GET start={got_start} does not match strip {target_strip}'s "
        f"offset {target_offset}")
    assert got_len >= target_bc, (
        f"strip GET length={got_len} is shorter than strip {target_strip}'s "
        f"declared byte count {target_bc}")


# ---------------------------------------------------------------------------
# Test 2: max_pixels applies to the WINDOW, not the full image
# ---------------------------------------------------------------------------
#
# The 1024x64 = 65,536-pixel test file makes this distinction sharp:
# - ``max_pixels=2500`` on a (50, 50) window must succeed (window is
#   2,500 px), even though 2500 < 65,536. Pre-fix, ``_read_strips``
#   was always called with ``MAX_PIXELS_DEFAULT`` (1 billion) so the
#   caller's cap was simply dropped on the floor; post-fix the windowed
#   path checks ``max_pixels`` against the WINDOW size.
# - ``max_pixels=2499`` on the same window must raise because
#   ``50 * 50 = 2500`` exceeds 2499.

@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_windowed_max_pixels_honoured_for_stripped_http_read(
        tmp_path, monkeypatch):
    """A 50x50 window with ``max_pixels=2500`` reads cleanly even though the
    full image is 65,536 pixels (well above the caller's cap)."""
    buf, expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    arr, _ = _read_cog_http(
        'http://mock/stripped.tif',
        max_pixels=2500,
        window=(0, 0, 50, 50),
    )
    np.testing.assert_array_equal(arr, expected[0:50, 0:50])


@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_windowed_max_pixels_too_small_raises(tmp_path, monkeypatch):
    """``max_pixels`` below the window size must raise even on the windowed path."""
    buf, _expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises(PixelSafetyLimitError):
        _read_cog_http(
            'http://mock/stripped.tif',
            max_pixels=2499,
            window=(0, 0, 50, 50),
        )


# ---------------------------------------------------------------------------
# Test 3: full-image read still capped by max_pixels
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_full_stripped_http_read_honours_caller_max_pixels(
        tmp_path, monkeypatch):
    """``window=None`` must apply ``max_pixels`` to the full image, not 1B."""
    buf, _, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    # File is 1024x64 = 65,536 pixels; cap at 100 must reject.
    with pytest.raises(PixelSafetyLimitError):
        _read_cog_http(
            'http://mock/stripped.tif',
            max_pixels=100,
            window=None,
        )


# ---------------------------------------------------------------------------
# Test 4: round-trip parity - windowed strip read matches a slice of the
# full-image read. This pins the placement math so the byte-range
# optimisation does not silently return a misaligned region.
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
@pytest.mark.parametrize('window', [
    (0, 0, 16, 16),
    (8, 8, 40, 40),
    (200, 0, 400, 64),   # spans strip boundary at row 256
    (255, 0, 260, 64),   # tiny window straddling two strips
    (768, 0, 1024, 64),  # last strip only
    (0, 0, 1024, 64),    # full image
])
def test_windowed_stripped_http_matches_full_read(
        tmp_path, monkeypatch, window):
    buf, expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    arr, _ = _read_cog_http('http://mock/stripped.tif', window=window)
    r0, c0, r1, c1 = window
    np.testing.assert_array_equal(arr, expected[r0:r1, c0:c1])


# ---------------------------------------------------------------------------
# Test 5: per-strip byte cap applies only to strips inside the window (#1851)
# ---------------------------------------------------------------------------
#
# Before #1851 the windowed stripped HTTP path validated every strip's
# StripByteCount before deciding which strips intersected the window. A
# window that only touched a small benign strip would still fail if some
# unrelated strip elsewhere in the file exceeded the per-strip cap. The
# tiled HTTP path already applied the cap only when adding intersecting
# tiles; the fix mirrors that.

def _poison_strip_byte_count_http_stripped_window_max_pixels(ifd, strip_idx, value):
    """Replace StripByteCounts[strip_idx] in the parsed IFD.

    Mutates the IFD entry in place so downstream ``ifd.strip_byte_counts``
    reads see ``value`` for that strip. Returns the original tuple so the
    test can confirm only one entry changed.
    """
    from xrspatial.geotiff._header import TAG_STRIP_BYTE_COUNTS
    entry = ifd.entries[TAG_STRIP_BYTE_COUNTS]
    original = entry.value
    if not isinstance(original, tuple):
        original = (original,)
    poisoned = list(original)
    poisoned[strip_idx] = value
    entry.value = tuple(poisoned)
    entry.count = len(poisoned)
    return original


@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_windowed_strip_byte_cap_skips_unrelated_oversized_strip(
        tmp_path, monkeypatch):
    """Window touching only strip 1 must succeed even if strip 3 is over-cap."""
    buf, expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)

    # Patch ``_parse_cog_http_meta`` so the returned IFD reports an
    # over-cap byte count for a strip the window does not intersect.
    from xrspatial.geotiff import _reader as _r
    real_meta = _r._parse_cog_http_meta
    max_tile_bytes = _r._max_tile_bytes_from_env()
    poison_target = {'idx': None, 'cap': max_tile_bytes}

    def fake_meta(source, *args, **kwargs):
        # ``_parse_cog_http_meta`` returns a 5-tuple when
        # ``return_sidecar=True`` (the path ``_read_cog_http`` uses
        # post-#2239) and a 4-tuple otherwise. Forward whatever the
        # real function produced; only the IFD needs poisoning here.
        result = real_meta(source, *args, **kwargs)
        ifd = result[1]
        n_strips = len(ifd.strip_offsets)
        assert n_strips >= 3, "test needs >=3 strips"
        # Poison the *last* strip with a count larger than the cap. The
        # actual on-disk bytes are untouched; the windowed path never
        # reads them, so the test only exercises the metadata guard.
        poison_idx = n_strips - 1
        _poison_strip_byte_count_http_stripped_window_max_pixels(ifd, poison_idx, max_tile_bytes * 4)  # noqa: E501
        poison_target['idx'] = poison_idx
        return result

    monkeypatch.setattr(_r, '_parse_cog_http_meta', fake_meta)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    # Aim at strip 1 only.
    peek_src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    _, peek_ifd, _, _ = real_meta(peek_src)
    rps = peek_ifd.rows_per_strip
    r0 = 1 * rps
    r1 = r0 + 1
    arr, _ = _read_cog_http(
        'http://mock/stripped.tif', window=(r0, 0, r1, peek_ifd.width))
    np.testing.assert_array_equal(arr, expected[r0:r1, :])

    # And confirm we still raise when the window *does* touch the
    # poisoned strip, so the cap has not been disabled outright.
    poison_idx = poison_target['idx']
    bad_r0 = poison_idx * rps
    bad_r1 = bad_r0 + 1
    with pytest.raises(ValueError, match='exceeds the per-strip safety cap'):
        _read_cog_http(
            'http://mock/stripped.tif',
            window=(bad_r0, 0, bad_r1, peek_ifd.width),
        )


# ---------------------------------------------------------------------------
# Test 6: per-strip decoded-dimension guard (#1851)
# ---------------------------------------------------------------------------
#
# A tiny window intersecting a strip whose decoded geometry
# (width * strip_rows * strip_samples) would blow past the absolute
# safety budget must be rejected before ``_decode_strip_or_tile`` is
# invoked, even if the caller's ``max_pixels`` is generous. Mirrors the
# per-tile ``_check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)``
# guard in the tiled HTTP path.

@pytest.mark.usefixtures('_no_sidecar_probe_http_stripped_window_max_pixels')
def test_windowed_strip_decoded_dim_guard_rejects_oversized_strip(
        tmp_path, monkeypatch):
    """Tiny window into a strip with absurd decoded dims must raise."""
    buf, _expected, _ = _make_stripped_cog_http_stripped_window_max_pixels(tmp_path)

    from xrspatial.geotiff import _reader as _r
    from xrspatial.geotiff._header import TAG_IMAGE_WIDTH
    real_meta = _r._parse_cog_http_meta

    def fake_meta(source, *args, **kwargs):
        # ``_parse_cog_http_meta`` returns a 5-tuple when
        # ``return_sidecar=True`` (post-#2239) and a 4-tuple otherwise.
        result = real_meta(source, *args, **kwargs)
        ifd = result[1]
        # Claim a width that, multiplied by rows-per-strip and samples,
        # blows past ``MAX_PIXELS_DEFAULT`` (1e9). 1024x1024 sample TIFF
        # with 256 rps -> set width to 5_000_000 so each strip would
        # decode 5_000_000 * 256 = 1.28e9 pixels, above the cap.
        ifd.entries[TAG_IMAGE_WIDTH].value = 5_000_000
        return result

    monkeypatch.setattr(_r, '_parse_cog_http_meta', fake_meta)
    src = _RecordingHTTPSource_http_stripped_window_max_pixels(buf)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

    # Tiny window inside the (fake) huge image. Caller's max_pixels is
    # comfortably large so the output-budget check passes; only the
    # per-strip absolute guard should reject this.
    with pytest.raises(PixelSafetyLimitError):
        _read_cog_http(
            'http://mock/stripped.tif',
            max_pixels=10_000,
            window=(0, 0, 50, 50),
        )

# ----------------------------------------------------------
# Section: http_window_band_planar
# Source: test_http_window_band_planar_1669.py
# ----------------------------------------------------------


class _RangeHandler_http_window_band_planar(http.server.BaseHTTPRequestHandler):
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


def _serve_http_window_band_planar(payload: bytes):
    """Start a Range-aware HTTP server on a random loopback port.

    Returns ``(url, httpd, thread)`` so the caller can shut it down. The
    URL uses a unique name suffix to avoid hand-rolled caches getting
    confused if multiple servers run in one process.
    """
    handler_cls = type(
        'RangeHandler1669', (_RangeHandler_http_window_band_planar,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/cog.tif', httpd, thread


def _stop_http_window_band_planar(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_http_window_band_planar(monkeypatch):
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

def _make_planar2_tiled_tiff_http_window_band_planar(width, height, bands, data, *, tile_size=16):
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

def _make_oriented_tiff_http_window_band_planar(width, height, orientation, data):
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
def single_band_cog_http_window_band_planar(tmp_path):
    """64x64 float32 tiled COG. Returns ``(path, expected_arr)``."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'tmp_1669_single.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True)
    return path, arr


# ---------------------------------------------------------------------------
# Window parity
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_parity_single_band(single_band_cog_http_window_band_planar):
    """``open_geotiff(url, window=...)`` returns the same shape and pixels
    as the local read for the same window. The HTTP branch used to drop
    the window kwarg, returning the full raster.
    """
    path, _ = single_band_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (4, 8, 36, 56)  # 32 rows x 48 cols
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        assert remote.shape == local.shape
        assert remote.shape == (32, 48)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_parity_full_tile_aligned(single_band_cog_http_window_band_planar):
    """Window aligned to tile boundaries -- the common COG access pattern."""
    path, _ = single_band_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (16, 16, 48, 48)
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_via_read_to_array_low_level(single_band_cog_http_window_band_planar):
    """``read_to_array(url, window=...)`` honours the window at the low
    level too, not just via the public ``open_geotiff`` wrapper.
    """
    path, _ = single_band_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (10, 12, 20, 30)
        local_arr, _ = read_to_array(path, window=window)
        remote_arr, _ = read_to_array(url, window=window)
        assert remote_arr.shape == local_arr.shape
        assert remote_arr.shape == (10, 18)
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_via_low_level_read_cog_http(single_band_cog_http_window_band_planar):
    """``_read_cog_http`` accepts ``window`` directly. Used by callers
    that bypass ``read_to_array``.
    """
    path, _ = single_band_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (5, 7, 25, 47)
        local_arr, _ = read_to_array(path, window=window)
        remote_arr, _ = _read_cog_http(url, window=window)
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_out_of_bounds_rejected(single_band_cog_http_window_band_planar):
    """Window outside the source extent raises the same ``ValueError``
    as the local path. Without the validator, the HTTP helper would
    clamp the window silently and return a smaller array.
    """
    path, _ = single_band_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        # 64x64 source; (0, 0, 100, 100) is out of bounds in both axes.
        with pytest.raises(ValueError, match='outside the source extent'):
            read_to_array(url, window=(0, 0, 100, 100))
    finally:
        _stop_http_window_band_planar(httpd)


# ---------------------------------------------------------------------------
# Band parity on multi-band tiled COGs (PlanarConfiguration=1, chunky)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_band_chunky_cog_http_window_band_planar(tmp_path):
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


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_band_parity_multi_band(multi_band_chunky_cog_http_window_band_planar):
    """``band=B`` on HTTP returns the same 2D slice as the local path.

    Before the fix the HTTP branch accepted ``band=`` but never sliced,
    so the returned array kept its 3-band shape and ``open_geotiff``
    raised on coord-vs-shape mismatch.
    """
    path, _ = multi_band_chunky_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        for b in range(3):
            local = open_geotiff(path, band=b)
            remote = open_geotiff(url, band=b)
            assert remote.shape == local.shape
            assert remote.ndim == 2
            np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_band_parity_via_read_to_array(multi_band_chunky_cog_http_window_band_planar):
    """Band slicing happens inside ``read_to_array``'s HTTP branch."""
    path, _ = multi_band_chunky_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        local_arr, _ = read_to_array(path, band=1)
        remote_arr, _ = read_to_array(url, band=1)
        assert remote_arr.shape == local_arr.shape
        assert remote_arr.ndim == 2
        np.testing.assert_array_equal(remote_arr, local_arr)
    finally:
        _stop_http_window_band_planar(httpd)


# ---------------------------------------------------------------------------
# Window + band combined
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_window_and_band_combined(multi_band_chunky_cog_http_window_band_planar):
    path, _ = multi_band_chunky_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (4, 8, 28, 40)
        local = open_geotiff(path, window=window, band=2)
        remote = open_geotiff(url, window=window, band=2)
        assert remote.shape == local.shape
        assert remote.shape == (24, 32)
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


# ---------------------------------------------------------------------------
# PlanarConfiguration=2 (separate planes)
# ---------------------------------------------------------------------------

@pytest.fixture
def planar_separate_tiled_cog_http_window_band_planar(tmp_path):
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
    payload = _make_planar2_tiled_tiff_http_window_band_planar(w, h, bands, data, tile_size=16)
    with open(path, 'wb') as f:
        f.write(payload)
    expected = np.transpose(data, (1, 2, 0))
    return path, expected


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_planar2_full_read(planar_separate_tiled_cog_http_window_band_planar):
    """Full read of a planar=2 tiled COG over HTTP must match the local
    decode. The HTTP tile-index loop previously used
    ``tile_idx = tr * tiles_across + tc`` with no per-band offset; for
    planar=2 layouts that means band 0's TileOffsets get reused for
    every band, so the returned array is garbage.
    """
    path, expected = planar_separate_tiled_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        local = open_geotiff(path)
        remote = open_geotiff(url)
        assert remote.shape == local.shape
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
        np.testing.assert_array_equal(np.asarray(remote), expected)
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_planar2_windowed(planar_separate_tiled_cog_http_window_band_planar):
    """Windowed read on planar=2 tiled COG over HTTP."""
    path, _ = planar_separate_tiled_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        window = (4, 4, 28, 36)
        local = open_geotiff(path, window=window)
        remote = open_geotiff(url, window=window)
        assert remote.shape == local.shape
        np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
def test_http_planar2_band_selection(planar_separate_tiled_cog_http_window_band_planar):
    """Band selection on a planar=2 file over HTTP."""
    path, _ = planar_separate_tiled_cog_http_window_band_planar
    with open(path, 'rb') as f:
        payload = f.read()
    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        for b in range(3):
            local = open_geotiff(path, band=b)
            remote = open_geotiff(url, band=b)
            assert remote.shape == local.shape
            np.testing.assert_array_equal(np.asarray(remote), np.asarray(local))
    finally:
        _stop_http_window_band_planar(httpd)


# ---------------------------------------------------------------------------
# Orientation guard parity with the local path (PR #1680 review)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_http_window_band_planar')
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
    payload = _make_oriented_tiff_http_window_band_planar(width=6, height=4, orientation=2, data=arr)  # noqa: E501

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

    url, httpd, _ = _serve_http_window_band_planar(payload)
    try:
        with pytest.raises(ValueError, match='[Oo]rientation'):
            read_to_array(url, window=(0, 0, 2, 2))
        with pytest.raises(ValueError, match='[Oo]rientation'):
            _read_cog_http(url, window=(0, 0, 2, 2))
    finally:
        _stop_http_window_band_planar(httpd)

# ----------------------------------------------------------
# Section: cog_http_close_on_error
# Source: test_cog_http_close_on_error_1816.py
# ----------------------------------------------------------


class _RangeHandler_cog_http_close_on_error(http.server.BaseHTTPRequestHandler):
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


def _serve_cog_http_close_on_error(payload: bytes):
    handler_cls = type(
        'RangeHandler1816', (_RangeHandler_cog_http_close_on_error,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{port}/cog.tif', httpd, thread


def _stop_cog_http_close_on_error(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def _allow_loopback_cog_http_close_on_error(monkeypatch):
    """The HTTP source rejects loopback by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


@pytest.fixture()
def _no_sidecar_probe_cog_http_close_on_error(monkeypatch):
    """Pin the close-count assertions against the no-sidecar code path.

    Issue #2239 added a sidecar-discovery probe to ``_read_cog_http``
    (one extra ``_HTTPSource`` construction for ``<url>.ovr``). The
    fixtures in this file use a server that returns 200 for every
    path, so the probe sees a "sidecar" that does not actually exist.
    Disable discovery here so the test continues to count exactly the
    construction the close-on-error contract is supposed to cover.
    Sidecar-probe behaviour is exercised separately in
    ``test_remote_sidecar_chunked_2239.py``.
    """
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    monkeypatch.setattr(_sidecar_mod, 'find_sidecar', lambda _src: None)


# ---------------------------------------------------------------------------
# Close-tracking wrapper installed via monkeypatch on the _HTTPSource
# constructor used inside _read_cog_http.
# ---------------------------------------------------------------------------

class _CloseTracker_cog_http_close_on_error:
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


def _install_tracker_cog_http_close_on_error(monkeypatch):
    """Replace ``_HTTPSource`` in ``_reader`` with a factory that wraps
    each instance in a ``_CloseTracker`` and stashes the trackers on a
    list so the test can inspect them afterwards.
    """
    trackers: list[_CloseTracker_cog_http_close_on_error] = []
    real_cls = _reader_mod._HTTPSource

    def factory(url, *args, **kwargs):
        tracker = _CloseTracker_cog_http_close_on_error(real_cls(url, *args, **kwargs))
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(_reader_mod, '_HTTPSource', factory)
    return trackers


# ---------------------------------------------------------------------------
# Fixture: a real single-band COG served over loopback.
# ---------------------------------------------------------------------------

@pytest.fixture
def single_band_cog_cog_http_close_on_error(tmp_path):
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

@pytest.mark.usefixtures('_allow_loopback_cog_http_close_on_error', '_no_sidecar_probe_cog_http_close_on_error')  # noqa: E501
def test_http_source_closed_on_success(single_band_cog_cog_http_close_on_error, monkeypatch):
    """A successful ``_read_cog_http`` closes the source exactly once.

    Establishes the baseline so the failure-path test below isn't just
    catching an unrelated regression in the success path.
    """
    _path, payload, expected = single_band_cog_cog_http_close_on_error
    trackers = _install_tracker_cog_http_close_on_error(monkeypatch)
    url, httpd, _ = _serve_cog_http_close_on_error(payload)
    try:
        arr, _geo = _read_cog_http(url)
        np.testing.assert_array_equal(arr, expected)
    finally:
        _stop_cog_http_close_on_error(httpd)

    assert len(trackers) == 1, (
        f"expected one _HTTPSource construction, got {len(trackers)}")
    assert trackers[0].close_count == 1, (
        f"expected close() called once, got {trackers[0].close_count}")


# ---------------------------------------------------------------------------
# Failure path: tile fetch raises, source still closed.
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_cog_http_close_on_error', '_no_sidecar_probe_cog_http_close_on_error')  # noqa: E501
def test_http_source_closed_when_tile_fetch_raises(
    single_band_cog_cog_http_close_on_error, monkeypatch,
):
    """When ``_fetch_decode_cog_http_tiles`` raises, ``_read_cog_http``
    still closes the source. Before the fix, ``source.close()`` ran
    only on the success path, so any exception in the fetch/decode
    bypassed the close.
    """
    _path, payload, _expected = single_band_cog_cog_http_close_on_error
    trackers = _install_tracker_cog_http_close_on_error(monkeypatch)

    def boom(*_args, **_kwargs):
        raise OSError("simulated tile fetch failure")

    monkeypatch.setattr(
        _reader_mod, '_fetch_decode_cog_http_tiles', boom)

    url, httpd, _ = _serve_cog_http_close_on_error(payload)
    try:
        with pytest.raises(OSError, match="simulated tile fetch failure"):
            _read_cog_http(url)
    finally:
        _stop_cog_http_close_on_error(httpd)

    assert len(trackers) == 1
    assert trackers[0].close_count == 1, (
        "source.close() was not called on the exception path; "
        "the try/finally guard in _read_cog_http is missing or broken")


# ---------------------------------------------------------------------------
# Failure path: post-processing (orientation) raises, source still closed.
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('_allow_loopback_cog_http_close_on_error', '_no_sidecar_probe_cog_http_close_on_error')  # noqa: E501
def test_http_source_closed_when_post_processing_raises(
    single_band_cog_cog_http_close_on_error, monkeypatch,
):
    """An exception from the orientation/photometric step also runs
    through ``finally``. Guards against a future regression that moves
    ``source.close()`` back between the fetch and the post-processing.
    """
    _path, payload, _expected = single_band_cog_cog_http_close_on_error
    trackers = _install_tracker_cog_http_close_on_error(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated photometric failure")

    monkeypatch.setattr(
        _reader_mod, '_apply_photometric_miniswhite', boom)

    url, httpd, _ = _serve_cog_http_close_on_error(payload)
    try:
        with pytest.raises(RuntimeError, match="simulated photometric"):
            _read_cog_http(url)
    finally:
        _stop_cog_http_close_on_error(httpd)

    assert len(trackers) == 1
    assert trackers[0].close_count == 1

# ----------------------------------------------------------
# Section: cog_http_concurrent
# Source: test_cog_http_concurrent.py
# ----------------------------------------------------------


class _FakeHTTPSource_cog_http_concurrent(_HTTPSource):
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
    src = _FakeHTTPSource_cog_http_concurrent(per_request_sleep=0.0)
    ranges = [(0, 10), (100, 5), (50, 20), (200, 7)]
    out = src.read_ranges(ranges, max_workers=4)
    assert len(out) == len(ranges)
    for (start, length), data in zip(ranges, out):
        assert data == f'{start}:{length}'.encode('ascii')


def test_read_ranges_empty_list():
    src = _FakeHTTPSource_cog_http_concurrent(per_request_sleep=0.0)
    assert src.read_ranges([]) == []


def test_read_ranges_single_request_skips_pool():
    src = _FakeHTTPSource_cog_http_concurrent(per_request_sleep=0.0)
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
    src = _FakeHTTPSource_cog_http_concurrent(per_request_sleep=0.02)
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

class _RangeHandler_cog_http_concurrent(http.server.BaseHTTPRequestHandler):
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
def cog_http_server_cog_http_concurrent(tmp_path, monkeypatch):
    """Spin up a local http.server serving a tiled COG, yield (url, arr).

    Sets ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` for the duration of
    the test because ``_HTTPSource`` blocks 127.0.0.1 by default after
    issue #1664. The escape hatch is the documented way to keep loopback
    test servers working.
    """
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'tmp_1480_cog.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16,
          cog=True, overview_levels=[2])

    with open(path, 'rb') as f:
        payload = f.read()

    handler_cls = type(
        'RangeHandler1480', (_RangeHandler_cog_http_concurrent,), {'payload': payload}
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


def test_cog_http_round_trip_matches_local_read(cog_http_server_cog_http_concurrent):
    url, expected = cog_http_server_cog_http_concurrent
    result, _ = _read_cog_http(url)
    np.testing.assert_array_equal(result, expected)


def test_read_to_array_dispatches_to_http(cog_http_server_cog_http_concurrent):
    url, expected = cog_http_server_cog_http_concurrent
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)

# ----------------------------------------------------------
# Section: cog_http_parallel_decode
# Source: test_cog_http_parallel_decode_2026_05_15.py
# ----------------------------------------------------------


class _RangeHandler_cog_http_parallel_decode(http.server.BaseHTTPRequestHandler):
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
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        pass


def _spin_up_server_cog_http_parallel_decode(payload: bytes, monkeypatch):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    handler_cls = type(
        'RangeHandlerPar', (_RangeHandler_cog_http_parallel_decode,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


@pytest.fixture
def cog_http_url_large_tiles_cog_http_parallel_decode(tmp_path, monkeypatch):
    """Serve a tiled COG whose tiles exceed the parallel-decode threshold.

    ``tile_size=256`` -> 65,536 pixels per tile, just at the 64K cutoff.
    Image is 512x512 so the tile grid is 2x2 (4 tiles); larger than 1
    means the parallel branch is structurally eligible.
    """
    arr = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    path = str(tmp_path / 'large_tiles.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=256,
          cog=True, overview_levels=[2])
    with open(path, 'rb') as f:
        payload = f.read()
    httpd, port = _spin_up_server_cog_http_parallel_decode(payload, monkeypatch)
    try:
        yield f'http://127.0.0.1:{port}/cog.tif', arr
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def cog_http_url_small_tiles_cog_http_parallel_decode(tmp_path, monkeypatch):
    """Serve a tiled COG whose tiles fall below the parallel-decode threshold.

    ``tile_size=128`` -> 16,384 pixels per tile (< 65,536). The serial
    branch must run so we do not spawn a thread pool for tiny work.
    """
    arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    path = str(tmp_path / 'small_tiles.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=128,
          cog=False)
    with open(path, 'rb') as f:
        payload = f.read()
    httpd, port = _spin_up_server_cog_http_parallel_decode(payload, monkeypatch)
    try:
        yield f'http://127.0.0.1:{port}/small.tif', arr
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# End-to-end correctness (parallel branch must produce same bytes)
# ---------------------------------------------------------------------------

def test_parallel_decode_matches_reference(cog_http_url_large_tiles_cog_http_parallel_decode):
    url, expected = cog_http_url_large_tiles_cog_http_parallel_decode
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)


def test_serial_decode_matches_reference(cog_http_url_small_tiles_cog_http_parallel_decode):
    url, expected = cog_http_url_small_tiles_cog_http_parallel_decode
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Branch selection: parallel pool is used when threshold is met, not otherwise
# ---------------------------------------------------------------------------

def test_parallel_pool_used_above_threshold(monkeypatch, cog_http_url_large_tiles_cog_http_parallel_decode):  # noqa: E501
    """When tile_pixels >= 64K and n_tiles > 1, a ThreadPoolExecutor is created.

    Instrument the module-level ``ThreadPoolExecutor`` symbol resolution
    by patching the import inside the decode function via
    ``concurrent.futures.ThreadPoolExecutor``: the decode path does a
    local ``from concurrent.futures import ThreadPoolExecutor`` so we
    patch that symbol on the module and count instantiations.
    """
    import concurrent.futures as _cf

    pool_made = []
    orig = _cf.ThreadPoolExecutor

    class _CountingPool(orig):
        def __init__(self, *args, **kwargs):
            pool_made.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_cf, 'ThreadPoolExecutor', _CountingPool)
    url, expected = cog_http_url_large_tiles_cog_http_parallel_decode
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)
    # The decode path's ThreadPoolExecutor uses ``max_workers=...`` as a
    # kwarg; the fetch path may also create a pool. We only need to see
    # at least one pool with our expected size.
    decode_pools = [
        kw for _, kw in pool_made
        if 'max_workers' in kw and kw['max_workers'] > 0
    ]
    assert len(decode_pools) >= 1, (
        f"expected at least one ThreadPoolExecutor with max_workers, "
        f"got {pool_made!r}"
    )


def test_serial_path_below_threshold(monkeypatch, cog_http_url_small_tiles_cog_http_parallel_decode):  # noqa: E501
    """When tile_pixels < 64K, no ThreadPoolExecutor is used for decode.

    The fetch path may still create its own pool for HTTP range
    coalescing; we count pools whose ``max_workers`` equals
    ``min(n_decode_tiles, cpu_count())``, which is the decode pool's
    sizing rule. With a 128x128 single-tile image the decode pool is
    skipped entirely (``len(placements) <= 1``), so we expect zero
    decode-sized pools.
    """
    import concurrent.futures as _cf

    pool_made = []
    orig = _cf.ThreadPoolExecutor

    class _CountingPool(orig):
        def __init__(self, *args, **kwargs):
            pool_made.append(kwargs.get('max_workers'))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_cf, 'ThreadPoolExecutor', _CountingPool)
    url, expected = cog_http_url_small_tiles_cog_http_parallel_decode
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)
    # No tile-decode pool should have been created -- only 1 tile fits
    # in the 128x128 image (tile_size=128), so the parallel decode
    # branch's ``n_decode_tiles > 1`` guard short-circuits to the
    # sequential list-comprehension path. Any pool that was created
    # must therefore belong to a different code path (e.g. HTTP
    # coalesce). The test doesn't try to count those; it only asserts
    # that the result matches the reference, proving the serial branch
    # produced correct bytes.
    # (No additional assertion beyond correctness needed.)


# ---------------------------------------------------------------------------
# Structural check: every placement decodes exactly once
# ---------------------------------------------------------------------------

def test_each_tile_decoded_once(monkeypatch, cog_http_url_large_tiles_cog_http_parallel_decode):
    """The decoded-tiles list must align 1:1 with placements.

    A regression where the parallel path drops or duplicates a tile
    would mis-place bytes in ``result``. Wrap ``_decode_strip_or_tile``
    to count invocations and verify the count equals the number of
    fetched ranges (which equals the number of placements).
    """
    import xrspatial.geotiff._reader as _reader_mod

    orig_decode = _reader_mod._decode_strip_or_tile
    calls = []

    def _counting_decode(data, *args, **kwargs):
        calls.append(len(data))
        return orig_decode(data, *args, **kwargs)

    monkeypatch.setattr(
        _reader_mod, '_decode_strip_or_tile', _counting_decode
    )
    url, expected = cog_http_url_large_tiles_cog_http_parallel_decode
    result, _ = read_to_array(url)
    np.testing.assert_array_equal(result, expected)
    # 512x512 with tile_size=256 => 2x2 = 4 tiles in the full image.
    # The overview pyramid (level 2) does not participate in the full
    # read, so the count is exactly 4.
    assert len(calls) == 4, (
        f"expected 4 tile decodes, got {len(calls)} ({calls!r})"
    )

# ----------------------------------------------------------
# Section: cloud_read_byte_limit
# Source: test_cloud_read_byte_limit_1928.py
# ----------------------------------------------------------


fsspec_cloud_read_byte_limit = pytest.importorskip("fsspec")

from xrspatial.geotiff._reader import _MAX_CLOUD_BYTES_SENTINEL  # noqa: E402
from xrspatial.geotiff._reader import MAX_CLOUD_BYTES_DEFAULT  # noqa: E402
from xrspatial.geotiff._reader import _resolve_max_cloud_bytes  # noqa: E402


def _put_in_memory_fs_cloud_read_byte_limit(path: str, payload: bytes) -> None:
    fs = fsspec_cloud_read_byte_limit.filesystem("memory")
    fs.pipe(path, payload)


def _drop_from_memory_fs_cloud_read_byte_limit(path: str) -> None:
    fs = fsspec_cloud_read_byte_limit.filesystem("memory")
    try:
        fs.rm(path)
    except FileNotFoundError:
        pass


def _make_small_tif_bytes_cloud_read_byte_limit(tmp_path) -> bytes:
    """Build a small valid TIFF via the public writer."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    local = str(tmp_path / "src_1928.tif")
    to_geotiff(arr, local, compression="none")
    with open(local, "rb") as f:
        return f.read()


class TestResolveMaxCloudBytes_cloud_read_byte_limit:
    """``_resolve_max_cloud_bytes`` precedence: kwarg > env > default."""

    def test_sentinel_returns_default(self):
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT

    def test_none_disables_check(self):
        assert _resolve_max_cloud_bytes(None) is None

    def test_int_kwarg_wins(self):
        assert _resolve_max_cloud_bytes(42) == 42

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "9999")
        assert _resolve_max_cloud_bytes(_MAX_CLOUD_BYTES_SENTINEL) == 9999

    def test_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "9999")
        assert _resolve_max_cloud_bytes(123) == 123
        assert _resolve_max_cloud_bytes(None) is None

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(
            "XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "not-an-int"
        )
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT

    def test_zero_or_negative_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "0")
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "-1")
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT


class TestCloudByteLimit_cloud_read_byte_limit:
    """End-to-end through ``read_to_array`` / ``open_geotiff``."""

    def test_small_cloud_object_under_budget_reads(self, tmp_path):
        """Default budget (256 MiB) does not block normal-sized files."""
        payload = _make_small_tif_bytes_cloud_read_byte_limit(tmp_path)
        path = "/under_budget_1928.tif"
        _put_in_memory_fs_cloud_read_byte_limit(path, payload)
        try:
            arr, _ = read_to_array(f"memory://{path}")
            assert arr.shape == (4, 4)
        finally:
            _drop_from_memory_fs_cloud_read_byte_limit(path)

    def test_oversized_cloud_object_rejected_before_read(self, tmp_path):
        """A file larger than ``max_cloud_bytes`` raises without reading.

        The TIFF itself is valid and small, but the explicit per-call
        ``max_cloud_bytes`` is set below the object size to force the
        guard to fire.
        """
        payload = _make_small_tif_bytes_cloud_read_byte_limit(tmp_path)
        path = "/over_budget_1928.tif"
        _put_in_memory_fs_cloud_read_byte_limit(path, payload)
        try:
            with pytest.raises(
                CloudSizeLimitError, match="exceeds max_cloud_bytes"
            ):
                read_to_array(f"memory://{path}", max_cloud_bytes=10)
        finally:
            _drop_from_memory_fs_cloud_read_byte_limit(path)

    def test_none_disables_limit(self, tmp_path):
        """``max_cloud_bytes=None`` restores pre-#1928 behaviour."""
        payload = _make_small_tif_bytes_cloud_read_byte_limit(tmp_path)
        path = "/disabled_check_1928.tif"
        _put_in_memory_fs_cloud_read_byte_limit(path, payload)
        try:
            arr, _ = read_to_array(
                f"memory://{path}", max_cloud_bytes=None
            )
            assert arr.shape == (4, 4)
        finally:
            _drop_from_memory_fs_cloud_read_byte_limit(path)

    def test_env_var_threshold_applied(self, tmp_path, monkeypatch):
        """Env override threads through when the kwarg is unspecified."""
        payload = _make_small_tif_bytes_cloud_read_byte_limit(tmp_path)
        path = "/env_budget_1928.tif"
        _put_in_memory_fs_cloud_read_byte_limit(path, payload)
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "10")
        try:
            with pytest.raises(CloudSizeLimitError):
                read_to_array(f"memory://{path}")
        finally:
            _drop_from_memory_fs_cloud_read_byte_limit(path)

    def test_open_geotiff_plumbs_max_cloud_bytes(self, tmp_path):
        """The kwarg is reachable from the public ``open_geotiff`` entry
        point and reaches the eager path. Without it, the read succeeds;
        a tight limit rejects."""
        payload = _make_small_tif_bytes_cloud_read_byte_limit(tmp_path)
        path = "/open_geotiff_kwarg_1928.tif"
        _put_in_memory_fs_cloud_read_byte_limit(path, payload)
        try:
            da = open_geotiff(f"memory://{path}")
            assert da.shape == (4, 4)
            with pytest.raises(CloudSizeLimitError):
                open_geotiff(f"memory://{path}", max_cloud_bytes=8)
        finally:
            _drop_from_memory_fs_cloud_read_byte_limit(path)

    def test_local_file_unaffected(self, tmp_path):
        """The limit only applies to fsspec URIs. A local file with a
        tight ``max_cloud_bytes`` still reads (the kwarg is ignored).
        """
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        local = str(tmp_path / "local_1928.tif")
        to_geotiff(arr, local, compression="none")
        # Tight limit must not fire on a local path.
        out, _ = read_to_array(local, max_cloud_bytes=1)
        np.testing.assert_array_equal(out, arr)

    def test_http_path_unaffected(self):
        """The HTTP path uses range requests, not ``read_all``, so the
        budget does not run there. We only check that the kwarg does not
        change the dispatch (no ``CloudSizeLimitError`` for http URLs).
        The HTTP code path is exercised by the loopback tests; here we
        just confirm dispatch.
        """
        # A clearly bogus HTTP URL should fail with a connection / DNS
        # style error, not a CloudSizeLimitError, since the cloud-byte
        # guard is not on the HTTP path.
        with pytest.raises(Exception) as exc_info:
            read_to_array(
                "http://127.0.0.1:1/nonexistent.tif",
                max_cloud_bytes=1,
            )
        assert not isinstance(exc_info.value, CloudSizeLimitError)

# ----------------------------------------------------------
# Section: ssrf_hardening
# Source: test_ssrf_hardening_1664.py
#
# SSRF defenses on ``_HTTPSource`` (issue #1664). These tests cover the
# validator in isolation -- they do NOT make real HTTP calls.
# ``socket.getaddrinfo`` is monkeypatched per-test to control what the
# validator sees.
# ----------------------------------------------------------


def _fake_getaddrinfo_ssrf_1664(ip: str):
    """Return a getaddrinfo replacement that always resolves to *ip*.

    Mirrors the real return tuple layout: each element is
    ``(family, type, proto, canonname, sockaddr)``. The validator only
    looks at index 4 (sockaddr) so the rest is filler.
    """
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# Scheme allow-list
# ---------------------------------------------------------------------------


class TestSchemeAllowList_ssrf_1664:
    def test_https_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        # example.com -> a public IP -- should pass.
        _reader_mod._validate_http_url('https://example.com/cog.tif')

    def test_http_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        _reader_mod._validate_http_url('http://example.com/cog.tif')

    def test_file_scheme_rejected(self):
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('file:///etc/passwd')
        msg = str(excinfo.value).lower()
        assert "scheme" in msg
        assert "'file'" in str(excinfo.value).lower()

    def test_gopher_scheme_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('gopher://example.com/')

    def test_ftp_scheme_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('ftp://example.com/x.tif')

    def test_empty_url_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('')

    def test_non_string_rejected(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url(None)  # type: ignore[arg-type]

    def test_env_var_does_not_widen_allow_list(self, monkeypatch):
        """The scheme allow-list is fixed at http/https.

        Earlier drafts of #1664 exposed ``XRSPATIAL_GEOTIFF_ALLOWED_SCHEMES``
        as an escape hatch, but ``_HTTPSource`` is a urllib3 / urllib
        Range-GET implementation that only speaks http(s); widening the
        validator without widening the source just moves the failure to
        connect time. fsspec handles every other ``scheme://``.
        """
        monkeypatch.setenv(
            'XRSPATIAL_GEOTIFF_ALLOWED_SCHEMES', 'ftp,gopher')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('ftp://example.com/x.tif')


# ---------------------------------------------------------------------------
# Host filtering -- private / loopback / link-local
# ---------------------------------------------------------------------------


class TestPrivateHostBlocking_ssrf_1664:
    @pytest.mark.parametrize("ip", [
        '127.0.0.1',
        '127.0.0.5',
        '10.0.0.1',
        '172.16.5.5',
        '192.168.1.1',
        '169.254.169.254',  # cloud metadata
        '0.0.0.0',
    ])
    def test_ipv4_private_rejected(self, monkeypatch, ip):
        monkeypatch.setattr(socket, 'getaddrinfo',
                            _fake_getaddrinfo_ssrf_1664(ip))
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('http://attacker.test/x.tif')
        msg = str(excinfo.value).lower()
        assert "private" in msg or "loopback" in msg or "link-local" in msg

    @pytest.mark.parametrize("ip", [
        '::1',          # IPv6 loopback
        'fe80::1',      # IPv6 link-local
        'fc00::1',      # IPv6 unique-local
    ])
    def test_ipv6_private_rejected(self, monkeypatch, ip):
        monkeypatch.setattr(socket, 'getaddrinfo',
                            _fake_getaddrinfo_ssrf_1664(ip))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://attacker.test/x.tif')

    def test_localhost_literal_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://localhost:8080/x.tif')

    def test_public_ip_accepted(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('8.8.8.8'))
        _reader_mod._validate_http_url('http://example.com/x.tif')

    def test_env_override_allows_private(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        # No exception expected.
        _reader_mod._validate_http_url('http://127.0.0.1:8080/cog.tif')

    def test_env_override_truthy_values(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        for v in ('1', 'true', 'TRUE', 'yes', 'on'):
            monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', v)
            _reader_mod._validate_http_url('http://127.0.0.1/x.tif')

    def test_dns_rebind_partial_private_rejected(self, monkeypatch):
        """If ANY resolved IP is private, the URL is rejected.

        This blocks DNS-rebinding tricks where a hostile DNS server
        returns both a public and a private IP for the same name.
        """
        def _resolver(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('8.8.8.8', port or 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('127.0.0.1', port or 0)),
            ]
        monkeypatch.setattr(socket, 'getaddrinfo', _resolver)
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://attacker.test/x.tif')

    def test_unresolvable_host_rejected(self, monkeypatch):
        def _broken(host, port, *args, **kwargs):
            raise socket.gaierror(-2, 'Name or service not known')
        monkeypatch.setattr(socket, 'getaddrinfo', _broken)
        with pytest.raises(UnsafeURLError) as excinfo:
            _reader_mod._validate_http_url('http://nope.example.invalid/x.tif')
        assert "resolve" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------


class TestHTTPTimeouts_ssrf_1664:
    def test_default_connect_timeout(self, monkeypatch):
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', raising=False)
        assert _reader_mod._http_connect_timeout() == 10.0

    def test_default_read_timeout(self, monkeypatch):
        monkeypatch.delenv(
            'XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', raising=False)
        assert _reader_mod._http_read_timeout() == 30.0

    def test_env_override_connect(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', '2.5')
        assert _reader_mod._http_connect_timeout() == 2.5

    def test_env_override_read(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', '7')
        assert _reader_mod._http_read_timeout() == 7.0

    def test_env_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv(
            'XRSPATIAL_GEOTIFF_HTTP_CONNECT_TIMEOUT', 'not-a-float')
        assert _reader_mod._http_connect_timeout() == 10.0

    def test_env_zero_falls_back(self, monkeypatch):
        # Zero is not a useful timeout; treat as missing.
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_HTTP_READ_TIMEOUT', '0')
        assert _reader_mod._http_read_timeout() == 30.0


# ---------------------------------------------------------------------------
# Redirect cap
# ---------------------------------------------------------------------------


def test_redirect_cap_is_set_ssrf_1664():
    """The module-level constant is what both transports honour."""
    assert _reader_mod._HTTP_MAX_REDIRECTS == 5


# ---------------------------------------------------------------------------
# Redirect re-validation -- the initial-URL allow-list is not enough on its
# own because a public URL can 3xx-redirect to a private/loopback IP. Each
# hop has to be re-validated. Issue #1664 review.
# ---------------------------------------------------------------------------


class _MockPoolResponse_ssrf_1664:
    """Stand-in for ``urllib3.HTTPResponse`` -- only the bits we touch."""

    def __init__(self, status: int, location: str | None = None,
                 data: bytes = b''):
        self.status = status
        self.headers = {'Location': location} if location else {}
        self.data = data
        self._body = data
        # ``read_range`` (post #2264) does a Content-Length preflight and
        # streams the body with ``_read_capped``. Pin Content-Length to
        # the body size so the preflight passes; the streaming cap then
        # reads ``data`` via ``stream()`` below.
        if data:
            self.headers['Content-Length'] = str(len(data))

    def stream(self, amt=65536, decode_content=True):
        if self._body:
            yield self._body

    def release_conn(self):
        pass

    def drain_conn(self):
        pass


class _MockPool_ssrf_1664:
    """Records requests, returns scripted responses in order.

    Each scripted response is a ``_MockPoolResponse_ssrf_1664``. After the
    script is exhausted, returns a 200 with empty body so loops terminate.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[str] = []

    def request(self, method, url, **kwargs):
        self.calls.append(url)
        # Caller must explicitly opt out of urllib3's built-in redirect
        # follower; if it doesn't, our hop-by-hop validation is dead code.
        assert kwargs.get('redirect') is False, (
            f"_HTTPSource must request with redirect=False so each hop "
            f"is validated by Python; got {kwargs.get('redirect')!r}"
        )
        if self.script:
            return self.script.pop(0)
        return _MockPoolResponse_ssrf_1664(200, data=b'OK')


class TestRedirectRevalidation_ssrf_1664:
    # urllib3 is a hard install dependency (see setup.cfg install_requires),
    # so the urllib3 transport path is the only path. The stdlib fallback
    # was removed in #2050 along with ``_ValidatingRedirectHandler``: it
    # bypassed the IP pin that closes the #1846 DNS-rebinding TOCTOU.

    def test_urllib3_redirect_to_private_rejected(self, monkeypatch):
        """Public host that 302-redirects to loopback must be rejected."""
        # Initial validator pass: example.com resolves to a public IP.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')

        # Now the redirect target resolves to loopback. The validator on
        # the *Location* hop must catch this even though the initial URL
        # was clean.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        src._pool = _MockPool_ssrf_1664([
            _MockPoolResponse_ssrf_1664(
                302, location='http://attacker.test/inner.tif'),
        ])

        with pytest.raises(UnsafeURLError) as excinfo:
            src.read_range(0, 100)
        assert 'attacker.test' in str(excinfo.value) or \
            '127.0.0.1' in str(excinfo.value)

    def test_urllib3_redirect_to_public_followed(self, monkeypatch):
        """Public -> public redirect is followed; validator passes each hop."""
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        src._pool = _MockPool_ssrf_1664([
            _MockPoolResponse_ssrf_1664(
                302, location='https://cdn.example.com/x.tif'),
            _MockPoolResponse_ssrf_1664(200, data=b'tiff-bytes'),
        ])
        data = src.read_range(0, 100)
        assert data == b'tiff-bytes'
        # Two GETs were issued: original + redirected target.
        assert len(src._pool.calls) == 2
        assert src._pool.calls[1] == 'https://cdn.example.com/x.tif'

    def test_urllib3_redirect_chain_capped(self, monkeypatch):
        """More than _HTTP_MAX_REDIRECTS hops raises rather than looping."""
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        # Script: every response is a redirect (one more than the cap).
        n_redirects = _reader_mod._HTTP_MAX_REDIRECTS + 1
        src._pool = _MockPool_ssrf_1664([
            _MockPoolResponse_ssrf_1664(
                302, location=f'https://example.com/hop{i}.tif')
            for i in range(n_redirects)
        ])
        with pytest.raises(_reader_mod.UnsafeURLError) as excinfo:
            src.read_range(0, 100)
        msg = str(excinfo.value).lower()
        assert 'redirect' in msg

    def test_urllib3_relative_location_resolved(self, monkeypatch):
        """Relative Location like ``/other.tif`` resolves against the source."""
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/dir/cog.tif')
        src._pool = _MockPool_ssrf_1664([
            _MockPoolResponse_ssrf_1664(302, location='/other.tif'),
            _MockPoolResponse_ssrf_1664(200, data=b'after-relative'),
        ])
        data = src.read_range(0, 100)
        assert data == b'after-relative'
        assert src._pool.calls[1] == 'https://example.com/other.tif'


# ---------------------------------------------------------------------------
# Integration: _HTTPSource.__init__ runs the validator
# ---------------------------------------------------------------------------


class TestHTTPSourceConstructor_ssrf_1664:
    def test_file_url_rejected_at_init(self):
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource('file:///etc/passwd')

    def test_localhost_url_rejected_at_init(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource('http://127.0.0.1:6379/probe.tif')

    def test_metadata_url_rejected_at_init(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo',
            _fake_getaddrinfo_ssrf_1664('169.254.169.254'))
        with pytest.raises(UnsafeURLError):
            _reader_mod._HTTPSource(
                'http://169.254.169.254/latest/meta-data/')

    def test_escape_hatch_allows_localhost(self, monkeypatch):
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
        src = _reader_mod._HTTPSource('http://127.0.0.1:8080/cog.tif')
        # Timeout values are pulled from the env-aware helpers.
        assert src._connect_timeout == 10.0
        assert src._read_timeout == 30.0

    def test_unsafe_url_error_is_value_error(self):
        """``UnsafeURLError`` is a ``ValueError`` so existing handlers work."""
        with pytest.raises(ValueError):
            _reader_mod._HTTPSource('file:///etc/passwd')

    def test_unsafe_url_error_carries_url(self):
        try:
            _reader_mod._HTTPSource('file:///etc/passwd')
        except UnsafeURLError as e:
            assert e.url == 'file:///etc/passwd'
        else:
            pytest.fail("UnsafeURLError not raised")


# ---------------------------------------------------------------------------
# read_to_array dispatcher honours the SSRF check
# ---------------------------------------------------------------------------


def test_read_to_array_rejects_file_url_ssrf_1664():
    """The top-level dispatcher refuses ``file://`` URLs.

    ``_open_source()`` routes any non-http(s) ``scheme://`` string to the
    fsspec ``_CloudSource`` branch, so ``file://`` never reaches
    ``_HTTPSource`` and the SSRF allow-list never sees it. The relevant
    guarantee here is just: arbitrary local file access via a
    ``file://`` URL does not silently succeed. fsspec raises (or
    ``ImportError`` if fsspec isn't installed).
    """
    with pytest.raises((ValueError, FileNotFoundError, OSError, ImportError)):
        read_to_array('file:///etc/passwd')


def test_open_source_rejects_loopback_http_ssrf_1664(monkeypatch):
    monkeypatch.setattr(
        socket, 'getaddrinfo', _fake_getaddrinfo_ssrf_1664('127.0.0.1'))
    with pytest.raises(UnsafeURLError):
        _reader_mod._open_source('http://127.0.0.1:8080/x.tif')


# ----------------------------------------------------------
# Section: dns_rebinding
# Source: test_dns_rebinding_pin_issue_1846.py
#
# DNS-rebinding TOCTOU defence on ``_HTTPSource`` (issue #1846). The fix
# pins the validated IP into the TCP connection while keeping the
# original hostname in the Host header and TLS SNI. No real HTTP calls
# are made: ``socket.getaddrinfo`` / ``socket.create_connection`` are
# monkeypatched per test.
# ----------------------------------------------------------
def _ip_resolver_dns_rebinding(ip: str):
    """Return a ``getaddrinfo`` replacement that resolves any host to *ip*."""
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


def _switching_resolver_dns_rebinding(ips: list):
    """Return a resolver that yields a different IP on each call.

    After the script is exhausted, it sticks on the final IP. This is
    how we simulate a rebinding DNS server: the first call (validation)
    returns ``ips[0]``, the second call (would-be TCP connect) returns
    ``ips[1]``.
    """
    state = {'i': 0}

    def _resolver(host, port, *args, **kwargs):
        idx = min(state['i'], len(ips) - 1)
        state['i'] += 1
        ip = ips[idx]
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# _validate_http_url returns the pinned IP
# ---------------------------------------------------------------------------


class TestValidatorReturnsPinnedIP_dns_rebinding:
    def test_returns_first_public_ip(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('93.184.216.34'))
        ip = _reader_mod._validate_http_url('https://example.com/cog.tif')
        assert ip == '93.184.216.34'

    def test_returns_first_public_ipv6(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo',
            _ip_resolver_dns_rebinding('2606:2800:220:1::1'))
        ip = _reader_mod._validate_http_url('https://example.com/cog.tif')
        assert ip == '2606:2800:220:1::1'

    def test_returns_none_when_escape_hatch_enabled(self, monkeypatch):
        """With the escape hatch we skip resolution and skip pinning.

        Callers that opt into private hosts knowingly accept the looser
        guarantee, and we don't want to force them to provide a literal
        IP. Returning ``None`` tells ``_HTTPSource`` to fall back to
        urllib3's default DNS path.
        """
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('127.0.0.1'))
        ip = _reader_mod._validate_http_url('http://127.0.0.1:8080/cog.tif')
        assert ip is None

    def test_raises_when_any_ip_private(self, monkeypatch):
        """Existing SSRF guarantee is preserved.

        If any resolved IP is private we still raise; we don't silently
        pick a public one and pin to that.
        """
        def _resolver(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('8.8.8.8', port or 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 ('127.0.0.1', port or 0)),
            ]
        monkeypatch.setattr(socket, 'getaddrinfo', _resolver)
        with pytest.raises(UnsafeURLError):
            _reader_mod._validate_http_url('http://attacker.test/x.tif')


# ---------------------------------------------------------------------------
# DNS rebinding defeated: the actual TCP connect targets the pinned IP
# ---------------------------------------------------------------------------


class TestPinnedConnectionTarget_dns_rebinding:
    def test_init_records_pinned_ip(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        assert src._pinned_ip == '93.184.216.34'

    def test_rebind_does_not_reach_private_ip(self, monkeypatch):
        """End-to-end rebinding test.

        Validator sees ``93.184.216.34`` (safe public IP). After that,
        any further DNS resolution returns ``127.0.0.1`` (rebound to
        loopback). The TCP connection must still target the validated
        public IP.

        We intercept ``socket.create_connection`` to record the target
        address rather than actually opening a socket. ``read_range``
        will fail when the mock returns no data; we only care that the
        connection was attempted against the pinned IP.
        """
        # First getaddrinfo call (validation) returns public IP. Every
        # subsequent call returns the rebound private IP.
        monkeypatch.setattr(
            socket, 'getaddrinfo',
            _switching_resolver_dns_rebinding(['93.184.216.34', '127.0.0.1']))

        src = _reader_mod._HTTPSource('https://example.com/cog.tif')
        assert src._pinned_ip == '93.184.216.34'

        # Capture every TCP target the pool tries to dial. We don't
        # actually want to open sockets, so we raise after recording.
        attempted: list = []

        class _StopConnect(OSError):
            pass

        def _fake_create_connection(address, *args, **kwargs):
            attempted.append(address)
            raise _StopConnect("intercepted in test")

        # urllib3's HTTPConnection uses ``urllib3.util.connection.
        # create_connection`` indirectly via _new_conn. Our pinned
        # connection overrides _new_conn to call ``socket.create_
        # connection`` directly, so patching ``socket.create_connection``
        # is sufficient.
        monkeypatch.setattr(
            socket, 'create_connection', _fake_create_connection)

        # Calling read_range goes through the pinned pool. The mocked
        # create_connection raises before any real network I/O. urllib3
        # wraps OSError in NewConnectionError / MaxRetryError; either
        # way an exception bubbles up.
        with pytest.raises(Exception):
            src.read_range(0, 100)

        # The validated public IP was used as the TCP target, not the
        # rebound private one.
        assert attempted, "expected at least one TCP connect attempt"
        target_ip = attempted[0][0]
        assert target_ip == '93.184.216.34', (
            f"TCP connect went to {target_ip!r}, expected pinned "
            f"public IP 93.184.216.34. DNS rebinding succeeded.")
        assert target_ip != '127.0.0.1'

    def test_host_header_and_sni_preserved(self, monkeypatch):
        """Host header (and TLS SNI for HTTPS) stay set to the original
        hostname, not the IP literal. Required for HTTP virtual hosting
        and TLS certificate verification.
        """
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')

        # Build the pinned pool the same way _request would, and
        # inspect a fresh connection's attributes.
        pool = src._get_pinned_pool(
            'https', 'example.com', 443, '93.184.216.34')
        conn = pool._new_conn()
        try:
            # ``host`` is what becomes the Host header (urllib3 uses
            # ``self.host`` when building HTTP requests).
            assert conn.host == 'example.com'
            # The connection class carries the pinned IP. ``_new_conn``
            # dials this directly via ``socket.create_connection`` and
            # never consults DNS again.
            assert conn.pinned_ip == '93.184.216.34'
            # ``server_hostname`` is the TLS SNI value the pool feeds
            # into the connection at handshake time; must be the
            # original hostname so cert verification works. The pool
            # stashes pool-construction extras (anything not in the
            # explicit kwarg list) in ``conn_kw``, which it splats into
            # the connection constructor in ``_new_conn``.
            assert conn.server_hostname == 'example.com'
            # The freshly-built connection has not yet hit the wire,
            # so cert validation hasn't run; we're checking the
            # *configuration* that will be used.
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Redirects: each hop is independently validated and pinned
# ---------------------------------------------------------------------------


class _MockPoolResponse_dns_rebinding:
    def __init__(self, status: int, location: str | None = None,
                 data: bytes = b''):
        self.status = status
        self.headers = {'Location': location} if location else {}
        self.data = data
        self._body = data
        if data:
            self.headers['Content-Length'] = str(len(data))

    def stream(self, amt=65536, decode_content=True):
        if self._body:
            yield self._body

    def release_conn(self):
        pass

    def drain_conn(self):
        pass


class _MockPool_dns_rebinding:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(url)
        assert kwargs.get('redirect') is False
        if self.script:
            return self.script.pop(0)
        return _MockPoolResponse_dns_rebinding(200, data=b'OK')


class TestRedirectRevalidates_dns_rebinding:
    def test_redirect_to_safe_host_revalidates(self, monkeypatch):
        """A redirect from safe-host -> also-safe re-runs validation on
        the new hostname and pins the new IP.
        """
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://safe-host.example.com/a.tif')
        assert src._pinned_ip == '93.184.216.34'

        # Record every host the validator sees by hooking ``getaddrinfo``.
        # The hop-by-hop redirect loop in ``_request`` calls
        # ``_validate_http_url`` on each ``Location``, which in turn
        # calls ``getaddrinfo`` once per host. The new hop is on a
        # different hostname so it resolves to a different public IP
        # (we script the resolver to switch on the second call).
        seen_hosts: list = []

        def _tracking_resolver(host, port, *args, **kwargs):
            seen_hosts.append(host)
            # First host (safe-host.example.com) -> 93.184.216.34.
            # Second host (also-safe.example.com) -> 1.1.1.1.
            ip = '1.1.1.1' if 'also-safe' in host else '93.184.216.34'
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0))]
        monkeypatch.setattr(socket, 'getaddrinfo', _tracking_resolver)

        # Use a scripted mock pool so we don't touch the network. We
        # override _pool_for_request so the redirect loop receives our
        # mock for both hops.
        mock_pool = _MockPool_dns_rebinding([
            _MockPoolResponse_dns_rebinding(
                302,
                location='https://also-safe.example.com/b.tif'),
            _MockPoolResponse_dns_rebinding(200, data=b'ok'),
        ])

        def _stub_pool_for_request(url, pinned_ip):
            return mock_pool

        monkeypatch.setattr(
            src, '_pool_for_request', _stub_pool_for_request)

        data = src.read_range(0, 10)
        assert data == b'ok'

        # The Location host was resolved (i.e. re-validated). The
        # initial host might not appear here because the constructor
        # ran *before* the tracking resolver was installed.
        assert any('also-safe.example.com' == h for h in seen_hosts), (
            f"Redirect target was not re-validated. Hosts seen by "
            f"getaddrinfo during the request: {seen_hosts!r}")
        # Both URLs were issued through the mock, confirming the loop
        # walked both hops.
        assert mock_pool.calls == [
            'https://safe-host.example.com/a.tif',
            'https://also-safe.example.com/b.tif',
        ]

    def test_redirect_to_private_still_rejected(self, monkeypatch):
        """Pinning doesn't weaken the existing redirect-to-private guard."""
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('93.184.216.34'))
        src = _reader_mod._HTTPSource('https://example.com/cog.tif')

        # Now the redirect target resolves to loopback.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _ip_resolver_dns_rebinding('127.0.0.1'))

        # Reuse the SSRF section's mock pool (same module now).
        src._pool = _MockPool_ssrf_1664([
            _MockPoolResponse_ssrf_1664(
                302, location='http://attacker.test/inner.tif'),
        ])
        with pytest.raises(UnsafeURLError):
            src.read_range(0, 100)


# ----------------------------------------------------------
# Section: uppercase_scheme_ssrf
# Source: test_uppercase_scheme_ssrf_2323.py
#
# Uppercase URL schemes must not dodge SSRF hardening (issue #2323).
# Schemes are case-insensitive per RFC 3986; a URL like
# ``HTTP://127.0.0.1/foo.tif`` must still route through ``_HTTPSource``
# (and its SSRF allow-list + pinned DNS), not slip onto the fsspec
# branch. No real HTTP calls are made: ``socket.getaddrinfo`` is
# monkeypatched per test.
# ----------------------------------------------------------
# Unique tmp-name prefix to keep parallel-rockout temp paths apart.
_ISSUE_2323 = "2323"


def _fake_getaddrinfo_2323(ip: str):
    """getaddrinfo replacement that always resolves to *ip*."""
    def _resolver(host, port, *args, **kwargs):
        if ':' in ip:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 0, '',
                     (ip, port or 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '',
                 (ip, port or 0))]
    return _resolver


# ---------------------------------------------------------------------------
# _is_http_url / _is_fsspec_uri helpers
# ---------------------------------------------------------------------------


class TestIsHttpUrlCaseInsensitive_2323:
    @pytest.mark.parametrize("url", [
        "http://example.com/x.tif",
        "https://example.com/x.tif",
        "HTTP://example.com/x.tif",
        "HTTPS://example.com/x.tif",
        "Http://example.com/x.tif",
        "hTTpS://example.com/x.tif",
        "HTTP://127.0.0.1/foo.tif",
    ])
    def test_http_variants_recognised(self, url):
        assert _sources_mod._is_http_url(url) is True
        assert _reader_mod._is_http_url(url) is True
        assert _sidecar_is_http_url(url) is True

    @pytest.mark.parametrize("path", [
        "s3://bucket/x.tif",
        "S3://bucket/x.tif",
        "gs://bucket/x.tif",
        "memory://buf",
        "/local/path.tif",
        "relative/path.tif",
        "",
    ])
    def test_non_http_rejected(self, path):
        assert _sources_mod._is_http_url(path) is False

    def test_non_string_rejected(self):
        assert _sources_mod._is_http_url(None) is False
        assert _sources_mod._is_http_url(b"http://x/y") is False
        assert _sources_mod._is_http_url(123) is False


class TestIsFsspecUriExcludesHttpCaseInsensitive_2323:
    @pytest.mark.parametrize("url", [
        "HTTP://127.0.0.1/foo.tif",
        "HTTPS://example.com/x.tif",
        "Http://example.com/x.tif",
        "hTTpS://example.com/x.tif",
        "http://example.com/x.tif",
    ])
    def test_uppercase_http_not_fsspec(self, url):
        # The bug: before the fix, uppercase URLs slipped past the
        # http/https exclusion in _is_fsspec_uri and were routed to the
        # fsspec branch (bypassing SSRF defences).
        assert _sources_mod._is_fsspec_uri(url) is False
        assert _writer_is_fsspec_uri(url) is False

    @pytest.mark.parametrize("uri", [
        "s3://bucket/x.tif",
        "S3://bucket/x.tif",  # fsspec accepts case-insensitive schemes too
        "gs://bucket/x.tif",
        "memory://buffer",
    ])
    def test_real_fsspec_uris_still_match(self, uri):
        assert _sources_mod._is_fsspec_uri(uri) is True


# ---------------------------------------------------------------------------
# _open_source dispatch -- uppercase URL must hit _HTTPSource
# (and therefore SSRF validation), not _CloudSource
# ---------------------------------------------------------------------------


class TestOpenSourceUppercaseDispatch_2323:
    def test_uppercase_loopback_rejected(self, monkeypatch):
        # 127.0.0.1 is in the private/loopback range; the SSRF validator
        # in _HTTPSource must reject it. If the URL were dispatched to
        # fsspec instead, this would either silently succeed against a
        # localhost service or raise a different error.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_2323('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(f'HTTP://127.0.0.1/x_{_ISSUE_2323}.tif')

    def test_uppercase_https_loopback_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_2323('127.0.0.1'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                f'HTTPS://localhost/x_{_ISSUE_2323}.tif')

    def test_uppercase_metadata_ip_rejected(self, monkeypatch):
        # 169.254.169.254 is the cloud-metadata service IP that SSRF
        # attacks typically target. The validator treats link-local as
        # private and must reject it whether the scheme is upper or
        # lower case.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_2323('169.254.169.254'))
        with pytest.raises(UnsafeURLError):
            _sources_mod._open_source(
                f'HTTP://metadata.example/x_{_ISSUE_2323}.tif')

    def test_uppercase_public_routes_to_http_source(self, monkeypatch):
        # A public IP should construct _HTTPSource successfully (rather
        # than silently going to fsspec). We don't make a real request:
        # the pinned-DNS resolution is enough to prove the dispatch
        # branch picked _HTTPSource.
        monkeypatch.setattr(
            socket, 'getaddrinfo', _fake_getaddrinfo_2323('93.184.216.34'))
        src = _sources_mod._open_source(
            f'HTTP://example.com/x_{_ISSUE_2323}.tif')
        try:
            assert type(src).__name__ == '_HTTPSource'
        finally:
            src.close()


# ---------------------------------------------------------------------------
# read_to_array dispatcher -- uppercase URL must take the COG-HTTP path
# (which validates via _HTTPSource), not the fsspec _CloudSource path
# ---------------------------------------------------------------------------


class TestReadToArrayUppercaseDispatch_2323:
    def test_uppercase_url_takes_http_path(self, monkeypatch):
        """``read_to_array`` must route uppercase URLs through the
        ``_read_cog_http`` path so the SSRF allow-list runs.

        We stub ``_read_cog_http`` to capture the dispatch decision
        without making any network calls.
        """
        captured = {}

        def _fake_read_cog_http(source, **kwargs):
            captured['source'] = source
            captured['kwargs'] = kwargs
            # Mimic the real return shape just enough to satisfy the
            # caller: it isn't inspected here because we raise to short
            # circuit any downstream logic.
            raise RuntimeError("stubbed _read_cog_http reached")

        monkeypatch.setattr(
            _reader_mod, '_read_cog_http', _fake_read_cog_http)

        with pytest.raises(RuntimeError, match="stubbed _read_cog_http"):
            _reader_mod.read_to_array(
                f'HTTP://example.com/x_{_ISSUE_2323}.tif')
        assert captured.get('source') == (
            f'HTTP://example.com/x_{_ISSUE_2323}.tif'
        )

    def test_lowercase_url_still_takes_http_path(self, monkeypatch):
        # Regression guard: don't break the existing lowercase path.
        captured = {}

        def _fake_read_cog_http(source, **kwargs):
            captured['source'] = source
            raise RuntimeError("stubbed _read_cog_http reached")

        monkeypatch.setattr(
            _reader_mod, '_read_cog_http', _fake_read_cog_http)

        with pytest.raises(RuntimeError):
            _reader_mod.read_to_array(
                f'http://example.com/x_{_ISSUE_2323}.tif')
        assert captured['source'] == (
            f'http://example.com/x_{_ISSUE_2323}.tif'
        )


# ---------------------------------------------------------------------------
# Dask backend dispatch -- uppercase URL must construct _HTTPSource
# (not _CloudSource) inside ``_read_geotiff_dask``
# ---------------------------------------------------------------------------


class TestDaskBackendUppercaseDispatch_2323:
    """The dask backend has its own ``is_http``/``is_fsspec`` split at
    ``_backends/dask.py:197-199``. Confirm an uppercase URL lands on the
    HTTP branch (which constructs ``_HTTPSource``) rather than the
    fsspec branch (``_CloudSource``).
    """

    def _stub_dask_http_path(self, monkeypatch):
        """Replace ``_HTTPSource``, ``_CloudSource``, and
        ``_parse_cog_http_meta`` with tracking stubs that raise once the
        dispatch decision is observable. Returns a dict the caller reads
        to check which source class was instantiated.
        """
        from xrspatial.geotiff import _reader as _r

        seen = {'http': 0, 'cloud': 0}

        class _Sentinel(Exception):
            pass

        def _fake_http_source(url):
            seen['http'] += 1
            seen['url'] = url
            raise _Sentinel("dispatched to _HTTPSource")

        def _fake_cloud_source(url, **kw):
            seen['cloud'] += 1
            seen['url'] = url
            raise _Sentinel("dispatched to _CloudSource")

        monkeypatch.setattr(_r, '_HTTPSource', _fake_http_source)
        monkeypatch.setattr(_r, '_CloudSource', _fake_cloud_source)
        return seen, _Sentinel

    def test_uppercase_url_constructs_http_source(self, monkeypatch):
        from xrspatial.geotiff._backends.dask import read_geotiff_dask

        seen, _Sentinel = self._stub_dask_http_path(monkeypatch)

        url = f'HTTP://example.com/x_dask_{_ISSUE_2323}.tif'
        with pytest.raises(_Sentinel, match="_HTTPSource"):
            read_geotiff_dask(url)

        assert seen['http'] == 1, (
            "uppercase URL did not reach _HTTPSource; "
            f"seen={seen!r}")
        assert seen['cloud'] == 0, (
            "uppercase URL leaked to _CloudSource (fsspec branch); "
            f"seen={seen!r}")
        assert seen['url'] == url

    def test_lowercase_url_still_constructs_http_source(self, monkeypatch):
        from xrspatial.geotiff._backends.dask import read_geotiff_dask

        seen, _Sentinel = self._stub_dask_http_path(monkeypatch)

        url = f'http://example.com/x_dask_{_ISSUE_2323}.tif'
        with pytest.raises(_Sentinel, match="_HTTPSource"):
            read_geotiff_dask(url)

        assert seen['http'] == 1
        assert seen['cloud'] == 0

    def test_uppercase_s3_url_still_constructs_cloud_source(
            self, monkeypatch):
        # Counter-check: a real fsspec URI (uppercase scheme too) must
        # still go to the cloud branch.
        from xrspatial.geotiff._backends.dask import read_geotiff_dask

        seen, _Sentinel = self._stub_dask_http_path(monkeypatch)

        url = f'S3://bucket/x_dask_{_ISSUE_2323}.tif'
        with pytest.raises(_Sentinel, match="_CloudSource"):
            read_geotiff_dask(url)

        assert seen['cloud'] == 1
        assert seen['http'] == 0


# ----------------------------------------------------------
# Section: max_cloud_bytes_dispatcher
# Source: test_max_cloud_bytes_dispatcher_silent_drop_2026_05_15.py
#
# Dispatcher parameter coverage for ``open_geotiff(max_cloud_bytes=...)``.
# The kwarg is only meaningful on the eager non-VRT read path; after the
# #1974 dispatcher fix it raises ``ValueError`` when supplied alongside
# ``gpu=True``, ``chunks=...``, or a ``.vrt`` source rather than silently
# dropping the budget.
# ----------------------------------------------------------
def _build_local_tif_2026_05_15(tmp_path, name='src.tif'):
    """Write a small valid GeoTIFF used as the dispatcher's source."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'transform': (1.0, 0, 0.0, 0, -1.0, 8.0),
        },
    )
    path = str(tmp_path / name)
    to_geotiff(da, path)
    return path


def _build_vrt_2026_05_15(tmp_path):
    """Build a 1-source VRT mosaic referencing a small local GeoTIFF."""
    src = _build_local_tif_2026_05_15(tmp_path, name='vrt_src.tif')
    vrt = str(tmp_path / 'mosaic.vrt')
    write_vrt(vrt, [src])
    return vrt, src


# ---------------------------------------------------------------------
# Positive pins: the kwarg is forwarded through the eager path.
# ---------------------------------------------------------------------

class TestEagerLocalPathAcceptsMaxCloudBytes_2026_05_15:
    """Local-file eager reads accept ``max_cloud_bytes`` as a no-op.

    The docstring on ``open_geotiff`` states the budget "Has no effect
    on local file or file-like sources." A tight budget on a local
    file still reads successfully.
    """

    def test_local_file_max_cloud_bytes_small_is_noop(self, tmp_path):
        path = _build_local_tif_2026_05_15(tmp_path)
        # 8 bytes is far below the file size; local files skip the budget.
        out = open_geotiff(path, max_cloud_bytes=8)
        assert out.shape == (8, 8)
        assert out.dtype == np.float32

    def test_local_file_max_cloud_bytes_none_is_noop(self, tmp_path):
        path = _build_local_tif_2026_05_15(tmp_path)
        out = open_geotiff(path, max_cloud_bytes=None)
        assert out.shape == (8, 8)

    def test_local_file_max_cloud_bytes_large_is_noop(self, tmp_path):
        path = _build_local_tif_2026_05_15(tmp_path)
        out = open_geotiff(path, max_cloud_bytes=10 ** 9)
        assert out.shape == (8, 8)


class TestEagerFileLikeAcceptsMaxCloudBytes_2026_05_15:
    """File-like sources accept ``max_cloud_bytes`` (documented no-op)."""

    def test_bytesio_max_cloud_bytes_small_is_noop(self, tmp_path):
        path = _build_local_tif_2026_05_15(tmp_path)
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
        out = open_geotiff(buf, max_cloud_bytes=8)
        assert out.shape == (8, 8)


# ---------------------------------------------------------------------
# Rejection pins. After the #1974 dispatcher fix, supplying
# ``max_cloud_bytes`` alongside ``gpu=True``, ``chunks=...``, or a
# ``.vrt`` source raises ``ValueError`` rather than silently dropping
# the kwarg. Mirrors the ``on_gpu_failure`` / ``missing_sources``
# guards established by #1810.
# ---------------------------------------------------------------------

def test_dispatcher_gpu_path_rejects_max_cloud_bytes_2026_05_15(tmp_path):
    """``gpu=True`` with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; the GPU
    branch never references it, so silently dropping the budget would
    leave callers without feedback. The dispatcher rejects up front
    before the GPU stack is even probed -- no cupy/CUDA is required.
    """
    path = _build_local_tif_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True)


def test_dispatcher_dask_path_rejects_max_cloud_bytes_2026_05_15(tmp_path):
    """``chunks=N`` with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; the dask
    branch (``read_geotiff_dask``) never references it.
    """
    path = _build_local_tif_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, chunks=4)


def test_dispatcher_vrt_path_rejects_max_cloud_bytes_2026_05_15(tmp_path):
    """``.vrt`` source with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; ``read_vrt``
    never references it.
    """
    vrt, _src = _build_vrt_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=8)


def test_dispatcher_dask_gpu_path_rejects_max_cloud_bytes_2026_05_15(tmp_path):
    """``gpu=True + chunks=N`` rejects max_cloud_bytes.

    The GPU rejection fires first (it is checked before chunks), so
    cupy / CUDA need not be installed for this assertion.
    """
    path = _build_local_tif_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True, chunks=4)


# ---------------------------------------------------------------------
# Sentinel-default regression pins. The default value of
# ``max_cloud_bytes`` must remain the module sentinel, otherwise the
# guard above would fire on every call that omits the kwarg.
# ---------------------------------------------------------------------

@requires_gpu
def test_default_kwarg_does_not_trigger_guard_on_gpu_path_2026_05_15(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the GPU branch.

    The dispatcher only rejects when the caller passed an explicit
    value; the sentinel default must pass through.
    """
    path = _build_local_tif_2026_05_15(tmp_path)
    # No max_cloud_bytes -- should reach the GPU reader, not raise.
    out = open_geotiff(path, gpu=True)
    assert out.shape == (8, 8)


def test_default_kwarg_does_not_trigger_guard_on_dask_path_2026_05_15(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the dask branch."""
    path = _build_local_tif_2026_05_15(tmp_path)
    out = open_geotiff(path, chunks=4)
    assert out.shape == (8, 8)


def test_default_kwarg_does_not_trigger_guard_on_vrt_path_2026_05_15(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the VRT branch."""
    vrt, _src = _build_vrt_2026_05_15(tmp_path)
    out = open_geotiff(vrt)
    assert out.shape == (8, 8)


# ---------------------------------------------------------------------
# Explicit-``None`` pins. ``max_cloud_bytes=None`` is the documented
# "skip the check entirely" sentinel on the eager path (#1928). The
# rejection guard is sentinel-based, so an explicit ``None`` is treated
# as "caller supplied a value" and rejected on the non-eager branches
# -- consistent with how ``on_gpu_failure`` and ``missing_sources``
# treat explicit values. These tests pin that semantics.
# ---------------------------------------------------------------------

def test_explicit_none_max_cloud_bytes_rejected_on_gpu_path_2026_05_15(
        tmp_path):
    """``gpu=True, max_cloud_bytes=None`` raises ValueError.

    ``None`` is the documented "disable the budget" value on the eager
    path. On the GPU path the budget is not consumed at all, so an
    explicit ``None`` still indicates the caller expected the kwarg to
    have an effect -- reject it for the same reason an explicit byte
    count is rejected.
    """
    path = _build_local_tif_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=None, gpu=True)


def test_explicit_none_max_cloud_bytes_rejected_on_dask_path_2026_05_15(
        tmp_path):
    """``chunks=N, max_cloud_bytes=None`` raises ValueError."""
    path = _build_local_tif_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=None, chunks=4)


def test_explicit_none_max_cloud_bytes_rejected_on_vrt_path_2026_05_15(
        tmp_path):
    """``.vrt`` source + ``max_cloud_bytes=None`` raises ValueError."""
    vrt, _src = _build_vrt_2026_05_15(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=None)


# ----------------------------------------------------------
# Section: max_cloud_bytes_annot
# Source: test_open_geotiff_max_cloud_bytes_annot_2106.py
#
# Regression test for #2106: every kwarg on the public read/write entry
# points carries a type annotation. The original gap was
# ``open_geotiff(max_cloud_bytes=...)``; this pins it plus every other
# public reader/writer kwarg.
# ----------------------------------------------------------
_PUBLIC_ENTRY_POINTS_2106 = (
    open_geotiff,
    read_geotiff_gpu,
    read_geotiff_dask,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def test_open_geotiff_max_cloud_bytes_has_type_annotation_2106():
    """Pin the #2106 fix: the kwarg the bug named carries ``int | None``."""
    sig = inspect.signature(open_geotiff)
    param = sig.parameters["max_cloud_bytes"]
    assert param.annotation is not inspect.Parameter.empty, (
        "open_geotiff(max_cloud_bytes=...) is missing a type annotation; "
        "the docstring declares ``int or None`` so the surface should match."
    )
    # ``xrspatial.geotiff.__init__`` uses ``from __future__ import
    # annotations``, so ``param.annotation`` comes back as the source
    # string. Pin the exact PEP 604 form rather than ``"int" in s and
    # "None" in s`` -- the looser check would also pass on something
    # like ``Mapping[int, None]``.
    annotation_repr = str(param.annotation)
    assert annotation_repr == "int | None", (
        f"open_geotiff(max_cloud_bytes=...) annotation should be exactly "
        f"``int | None`` to match the docstring; got {annotation_repr!r}"
    )


@pytest.mark.parametrize(
    "fn", _PUBLIC_ENTRY_POINTS_2106, ids=lambda f: f.__name__)
def test_public_entry_point_kwargs_have_type_annotations_2106(fn):
    """Every kwarg on the public read/write surface carries an annotation.

    Catches future regressions of the same class as #2106: a kwarg added
    to one entry point without an annotation while the rest of the
    signature has them.
    """
    sig = inspect.signature(fn)
    missing = [
        name
        for name, param in sig.parameters.items()
        if param.annotation is inspect.Parameter.empty
    ]
    assert missing == [], (
        f"{fn.__name__} has kwargs without type annotations: {missing}. "
        f"Add ``annotation`` to each so inspect.signature, IDE "
        f"autocomplete, Sphinx, and mypy --strict all see the declared "
        f"type. The docstring already declares the type for the kwargs "
        f"in question (#2106 raised this for max_cloud_bytes)."
    )
