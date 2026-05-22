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
from xrspatial.geotiff._reader import PixelSafetyLimitError, _HTTPSource, _read_cog_http
from xrspatial.geotiff._writer import write


@pytest.fixture(autouse=True)
def _no_sidecar_probe(monkeypatch):
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

def _poison_strip_byte_count(ifd, strip_idx, value):
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


def test_windowed_strip_byte_cap_skips_unrelated_oversized_strip(
        tmp_path, monkeypatch):
    """Window touching only strip 1 must succeed even if strip 3 is over-cap."""
    buf, expected, _ = _make_stripped_cog(tmp_path)

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
        _poison_strip_byte_count(ifd, poison_idx, max_tile_bytes * 4)
        poison_target['idx'] = poison_idx
        return result

    monkeypatch.setattr(_r, '_parse_cog_http_meta', fake_meta)
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    # Aim at strip 1 only.
    peek_src = _RecordingHTTPSource(buf)
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

def test_windowed_strip_decoded_dim_guard_rejects_oversized_strip(
        tmp_path, monkeypatch):
    """Tiny window into a strip with absurd decoded dims must raise."""
    buf, _expected, _ = _make_stripped_cog(tmp_path)

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
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    # Tiny window inside the (fake) huge image. Caller's max_pixels is
    # comfortably large so the output-budget check passes; only the
    # per-strip absolute guard should reject this.
    with pytest.raises(PixelSafetyLimitError):
        _read_cog_http(
            'http://mock/stripped.tif',
            max_pixels=10_000,
            window=(0, 0, 50, 50),
        )
