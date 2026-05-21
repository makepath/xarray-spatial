"""Coalesced HTTP COG range reads + once-per-graph IFD parsing.

Covers two performance fixes:

* P2 -- :func:`coalesce_ranges` merges adjacent tile byte ranges into
  fewer larger GETs so HTTP wall time is bounded by ``ceil(N_merged/W) *
  RTT`` rather than ``ceil(N_tiles/W) * RTT``.
* P5 -- :func:`read_geotiff_dask` parses IFDs once per graph and threads
  the parsed metadata into delayed tasks, so an N-chunk HTTP COG no
  longer fires N separate 16 KB header GETs.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff import read_geotiff_dask
from xrspatial.geotiff._reader import (
    COALESCE_GAP_THRESHOLD_DEFAULT,
    _HTTPSource,
    coalesce_ranges,
    split_coalesced_bytes,
)
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Pure unit tests on the coalescer
# ---------------------------------------------------------------------------

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
# Mocked HTTP source for perf and call-count assertions
# ---------------------------------------------------------------------------

class _MockHTTPSource(_HTTPSource):
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


@pytest.fixture
def small_cog_bytes(tmp_path):
    """Build a small tiled COG and return its raw bytes."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = str(tmp_path / 'cog.tif')
    # 8x8 tile grid (tile_size=8, image=64x64) gives 64 tiles -- enough
    # tile count to make coalescing observable.
    write(arr, path, compression='deflate', tiled=True, tile_size=8,
          cog=True)
    with open(path, 'rb') as f:
        return f.read(), arr, path


def test_read_cog_http_uses_coalesced_fetches(small_cog_bytes, monkeypatch):
    """One coalesced GET for many adjacent COG tiles instead of N GETs."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes
    src = _MockHTTPSource(buf)

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


def test_read_cog_http_perf_with_mock_rtt(small_cog_bytes, monkeypatch):
    """Coalesced HTTP should beat the un-coalesced baseline at 50 ms RTT."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes
    rtt = 0.05

    # Baseline: disable coalescing via env var so each tile costs an RTT.
    monkeypatch.setenv('XRSPATIAL_COG_COALESCE_GAP', '-1')
    src1 = _MockHTTPSource(buf, rtt=rtt)
    monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src1)
    t0 = time.perf_counter()
    arr1, _ = _reader_mod._read_cog_http('http://mock/cog.tif')
    baseline = time.perf_counter() - t0
    np.testing.assert_array_equal(arr1, expected)

    # Coalesced path: leave env unset (default 1 MB threshold).
    monkeypatch.delenv('XRSPATIAL_COG_COALESCE_GAP', raising=False)
    src2 = _MockHTTPSource(buf, rtt=rtt)
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

def test_dask_local_correctness(small_cog_bytes):
    """Dask read of a local COG must equal the eager read bit-for-bit."""
    _, expected, path = small_cog_bytes
    eager = open_geotiff(path)
    lazy = read_geotiff_dask(path, chunks=16).compute()
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(lazy))
    np.testing.assert_array_equal(np.asarray(eager), expected)


def test_dask_http_parses_ifds_once(small_cog_bytes, monkeypatch):
    """An N-chunk HTTP graph must trigger at most one IFD-header GET."""
    from xrspatial.geotiff import _reader as _reader_mod

    buf, expected, _ = small_cog_bytes
    src_holder: list[_MockHTTPSource] = []

    def _fake_http_source(url):
        s = _MockHTTPSource(buf)
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
