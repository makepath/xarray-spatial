"""Regression tests for issue #1842.

The stripped branch of ``_fetch_decode_cog_http_tiles`` used to call
``source.read_all()`` and slice the decoded array. That violated two
contracts the tiled branch upholds (#1664, #1823):

1. A windowed HTTP read should fetch only the byte ranges of the strips
   that intersect the window, not the whole file.
2. The caller's ``max_pixels`` should bound the *materialised* pixel
   count (the window for windowed reads, the full image otherwise),
   not be silently swapped for ``MAX_PIXELS_DEFAULT``.

These tests pin both behaviours.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from xrspatial.geotiff import _reader as reader_mod
from xrspatial.geotiff._reader import (
    PixelSafetyLimitError,
    _HTTPSource,
    _read_cog_http,
)
from xrspatial.geotiff._writer import write


class _RecordingHTTPSource(_HTTPSource):
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


def _make_stripped_cog(tmp_path, *, height=1024, width=64):
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

def test_windowed_stripped_http_fetches_only_intersecting_strips(
        tmp_path, monkeypatch):
    """A window covering one strip must only fetch that strip's bytes."""
    buf, expected, _ = _make_stripped_cog(tmp_path)
    src = _RecordingHTTPSource(buf)

    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    # Pick a window that covers exactly one row range. We don't know the
    # writer-picked rows_per_strip until we open the file once, so peek
    # at the IFD via a second mock source (the recording source's calls
    # for the peek pass are not asserted on).
    from xrspatial.geotiff._reader import _parse_cog_http_meta
    peek = _RecordingHTTPSource(buf)
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

def test_windowed_max_pixels_honoured_for_stripped_http_read(
        tmp_path, monkeypatch):
    """A 50x50 window with ``max_pixels=2500`` reads cleanly even though the
    full image is 65,536 pixels (well above the caller's cap)."""
    buf, expected, _ = _make_stripped_cog(tmp_path)
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    arr, _ = _read_cog_http(
        'http://mock/stripped.tif',
        max_pixels=2500,
        window=(0, 0, 50, 50),
    )
    np.testing.assert_array_equal(arr, expected[0:50, 0:50])


def test_windowed_max_pixels_too_small_raises(tmp_path, monkeypatch):
    """``max_pixels`` below the window size must raise even on the windowed path."""
    buf, _expected, _ = _make_stripped_cog(tmp_path)
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises(PixelSafetyLimitError):
        _read_cog_http(
            'http://mock/stripped.tif',
            max_pixels=2499,
            window=(0, 0, 50, 50),
        )


# ---------------------------------------------------------------------------
# Test 3: full-image read still capped by max_pixels
# ---------------------------------------------------------------------------

def test_full_stripped_http_read_honours_caller_max_pixels(
        tmp_path, monkeypatch):
    """``window=None`` must apply ``max_pixels`` to the full image, not 1B."""
    buf, _, _ = _make_stripped_cog(tmp_path)
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

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
    buf, expected, _ = _make_stripped_cog(tmp_path)
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    arr, _ = _read_cog_http('http://mock/stripped.tif', window=window)
    r0, c0, r1, c1 = window
    np.testing.assert_array_equal(arr, expected[r0:r1, c0:c1])
