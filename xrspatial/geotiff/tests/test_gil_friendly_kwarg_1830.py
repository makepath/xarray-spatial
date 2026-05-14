"""Direct coverage for the ``gil_friendly`` kwarg added in PR #1826 (#1830).

The flag gates a documented optimisation: when ``True`` the deflate path
is forced through stdlib ``zlib.compress`` (GIL-releasing) even when the
optional ``deflate`` PyPI binding (which holds the GIL during compress)
is installed. The writer's parallel strip/tile paths pass
``gil_friendly=True`` so the thread pool actually scales; the sequential
paths leave it at the default ``False`` to pick up libdeflate's per-call
speedup.

Existing tests in ``test_parallel_writer_1800.py`` cover end-to-end
round-trip correctness and that the thread pool is dispatched, but
nothing observes which deflate backend ran. A regression dropping the
``and not gil_friendly`` clause in ``_compression.py`` (or dropping the
``gil_friendly=True`` argument on the parallel writer call sites) would
ship the documented thread-pool scaling regression silently.

These tests directly exercise the flag at every layer it appears.
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
import pytest

import xrspatial.geotiff._compression as comp_mod
from xrspatial.geotiff._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_LZW,
    COMPRESSION_LZ4,
    COMPRESSION_NONE,
    COMPRESSION_PACKBITS,
    COMPRESSION_ZSTD,
    _HAVE_LIBDEFLATE,
    compress,
    deflate_compress,
)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import (
    _PARALLEL_MIN_BYTES,
    _prepare_strip,
    _prepare_tile,
    _write_stripped,
    _write_tiled,
    write,
)


# ---------------------------------------------------------------------------
# deflate_compress(gil_friendly=...) at the codec layer
# ---------------------------------------------------------------------------

def _payload(n: int = 8192) -> bytes:
    """Repeatable payload large enough to exercise real codec branches."""
    rng = np.random.RandomState(1830)
    return (rng.bytes(n))


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_deflate_compress_gil_friendly_true_bypasses_libdeflate(monkeypatch):
    """``gil_friendly=True`` must route through stdlib zlib, not libdeflate.

    A regression dropping the ``and not gil_friendly`` clause would
    silently re-route the parallel writer through the GIL-holding
    libdeflate binding and lose the documented thread-pool scaling
    (5x with zlib vs 1.2x with libdeflate across 8 threads).
    """
    libdeflate_calls = {'n': 0}

    real_zlib_compress = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        libdeflate_calls['n'] += 1
        return real_zlib_compress(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    # Baseline: gil_friendly omitted defaults to False -> libdeflate fires.
    out_default = deflate_compress(raw, level=6)
    assert libdeflate_calls['n'] == 1, (
        'with libdeflate installed and gil_friendly=False (default), '
        'deflate_compress must call the libdeflate binding'
    )

    # gil_friendly=True must skip libdeflate.
    out_gilfriendly = deflate_compress(raw, level=6, gil_friendly=True)
    assert libdeflate_calls['n'] == 1, (
        'gil_friendly=True must bypass the libdeflate binding even when '
        'it is installed; libdeflate.zlib_compress was called'
    )

    # Both outputs decompress to the original bytes (wire-compatible).
    assert zlib.decompress(out_default) == raw
    assert zlib.decompress(out_gilfriendly) == raw
    # gil_friendly=True output is exactly stdlib zlib.compress at level 6.
    assert out_gilfriendly == zlib.compress(raw, 6)


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_deflate_compress_gil_friendly_false_uses_libdeflate(monkeypatch):
    """Default ``gil_friendly=False`` must call libdeflate when present.

    Pins the sequential-writer fast path: a regression flipping the
    default or always routing to stdlib zlib would silently undo the
    ~3x per-call speedup that PR #1826 set out to deliver.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    out = deflate_compress(raw, level=6)
    assert calls['n'] == 1, (
        'gil_friendly=False (default) must call deflate.zlib_compress'
    )
    out_explicit = deflate_compress(raw, level=6, gil_friendly=False)
    assert calls['n'] == 2
    assert zlib.decompress(out) == raw
    assert zlib.decompress(out_explicit) == raw


def test_deflate_compress_gil_friendly_round_trip_both_directions():
    """Round-trip parity across both flag values, regardless of backend.

    Output bytes may differ (libdeflate is a different encoder), but
    both must zlib-decompress back to the input.
    """
    raw = _payload(16384)
    for gf in (True, False):
        for level in (1, 6, 9):
            blob = deflate_compress(raw, level=level, gil_friendly=gf)
            assert zlib.decompress(blob) == raw, (
                f'gil_friendly={gf}, level={level} did not round-trip'
            )


def test_deflate_compress_fallback_warning_fires_when_libdeflate_missing(
        monkeypatch):
    """One-shot UserWarning must fire when libdeflate is absent.

    The existing ``test_deflate_compress_fallback_when_libdeflate_missing``
    test silences this warning to keep its assertion focused on output
    bytes. This test pins the warning behaviour itself: a regression
    removing the warning would let users silently pay the 3x perf hit
    on every install missing the optional dep.
    """
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', False)

    raw = b'1830-warning-fires' * 1024

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = comp_mod.deflate_compress(raw, level=6)

    assert zlib.decompress(out) == raw
    matches = [w for w in caught
               if issubclass(w.category, UserWarning)
               and '`deflate` package is not installed' in str(w.message)]
    assert len(matches) == 1, (
        f'expected exactly one libdeflate-fallback UserWarning, '
        f'got {len(matches)}: {[str(w.message) for w in caught]}'
    )
    # Latch flips after the first call.
    assert comp_mod._zlib_fallback_warned is True


def test_deflate_compress_fallback_warning_is_one_shot(monkeypatch):
    """Subsequent calls after the first must not re-emit the warning.

    The module-global latch ``_zlib_fallback_warned`` is the gate. A
    regression flipping it to per-call would spam every parallel
    writer invocation with the same warning.
    """
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', False)

    raw = b'1830-one-shot' * 512

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp_mod.deflate_compress(raw)
        comp_mod.deflate_compress(raw)
        comp_mod.deflate_compress(raw, level=9)

    matches = [w for w in caught
               if issubclass(w.category, UserWarning)
               and '`deflate` package is not installed' in str(w.message)]
    assert len(matches) == 1, (
        f'fallback warning must fire only on the first call; '
        f'got {len(matches)} emissions'
    )


def test_deflate_compress_fallback_no_warning_when_latch_set(monkeypatch):
    """If the latch is already True, no warning fires (process startup
    typically warms it from the first user write)."""
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', True)

    raw = b'1830-latch-set' * 256

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = comp_mod.deflate_compress(raw)

    assert zlib.decompress(out) == raw
    assert not [w for w in caught if issubclass(w.category, UserWarning)
                and '`deflate` package' in str(w.message)]


# ---------------------------------------------------------------------------
# compress(..., gil_friendly=...) at the codec dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_forwards_gil_friendly_to_deflate(monkeypatch):
    """``compress(DEFLATE, gil_friendly=True)`` must skip libdeflate.

    Pins the dispatcher in ``_compression.compress``: the kwarg must
    thread through to ``deflate_compress``. A regression dropping the
    forward would silently revert the parallel writer to libdeflate.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    # Default (gil_friendly=False) -> libdeflate fires once.
    compress(raw, COMPRESSION_DEFLATE, level=6)
    assert calls['n'] == 1
    # gil_friendly=True -> libdeflate must NOT fire.
    out = compress(raw, COMPRESSION_DEFLATE, level=6, gil_friendly=True)
    assert calls['n'] == 1
    assert zlib.decompress(out) == raw


def test_compress_gil_friendly_ignored_for_non_deflate_codecs():
    """LZW/PackBits/zstd/lz4/none ignore the flag (their bindings already
    release the GIL). Round-trip results must be identical for both
    flag values; this guards against a future change accidentally
    routing a non-deflate codec through a different code path based on
    the flag.
    """
    from xrspatial.geotiff._compression import decompress

    raw = _payload(4096)

    matrix = [
        (COMPRESSION_NONE, raw),
        (COMPRESSION_PACKBITS, raw),
        (COMPRESSION_LZW, raw),
        (COMPRESSION_ZSTD, raw),
        (COMPRESSION_LZ4, raw),
    ]
    for tag, payload in matrix:
        out_false = compress(payload, tag, gil_friendly=False)
        out_true = compress(payload, tag, gil_friendly=True)
        assert out_false == out_true, (
            f'compression={tag}: gil_friendly must not affect non-deflate '
            f'codec output'
        )
        # Spot-check round-trip on the path that has a public decoder.
        if tag in (COMPRESSION_ZSTD, COMPRESSION_LZW, COMPRESSION_LZ4,
                   COMPRESSION_PACKBITS):
            decoded = decompress(out_true, tag, expected_size=len(payload))
            decoded_bytes = (decoded.tobytes()
                             if hasattr(decoded, 'tobytes') else decoded)
            assert decoded_bytes[:len(payload)] == payload
        elif tag == COMPRESSION_NONE:
            assert out_true == payload


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_default_gil_friendly_is_false(monkeypatch):
    """The dispatcher's default must keep callers on libdeflate.

    A regression flipping the default to True would silently revert
    the documented sequential-path 3x speedup for every read-modify-
    write caller of ``compress`` outside the parallel writer.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    compress(raw, COMPRESSION_DEFLATE, level=6)
    assert calls['n'] == 1, (
        'compress() default must call libdeflate when installed'
    )


# ---------------------------------------------------------------------------
# Writer call-site verification: _write_stripped / _write_tiled /
# write_streaming pass the right gil_friendly value into the codec.
# ---------------------------------------------------------------------------

class _DeflateCallSpy:
    """Capture every deflate_compress call's gil_friendly value."""

    def __init__(self, monkeypatch):
        self.calls = []  # list of bool
        self._real = comp_mod.deflate_compress
        # Patch at the module that the dispatcher (``compress``) imports
        # from, so all entry points are observed.
        monkeypatch.setattr(comp_mod, 'deflate_compress', self._spy)

    def _spy(self, data, level=6, gil_friendly=False):
        self.calls.append(bool(gil_friendly))
        return self._real(data, level=level, gil_friendly=gil_friendly)


def test_write_stripped_parallel_path_uses_gil_friendly(monkeypatch):
    """The parallel strip writer must call deflate_compress with
    ``gil_friendly=True`` on every strip.

    Pins the writer call site ``_writer.py:764``. A regression dropping
    the kwarg (or passing False) would silently make 8-thread parallel
    deflate writes scale at 1.2x instead of 5x.
    """
    # Large enough payload to take the parallel branch.
    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    assert arr.nbytes > _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_stripped(arr, COMPRESSION_DEFLATE, predictor=1,
                    rows_per_strip=256)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_stripped'
    )
    assert all(spy.calls), (
        f'parallel strip writer must pass gil_friendly=True to every '
        f'deflate_compress call; observed flags: {spy.calls}'
    )


def test_write_stripped_sequential_path_uses_default(monkeypatch):
    """The sequential strip writer (small payload) must use
    ``gil_friendly=False`` so the sequential path picks up libdeflate.

    Pins the writer call site ``_writer.py:741``. A regression passing
    True here would silently revert the sequential 3x speedup.
    """
    rng = np.random.RandomState(1830)
    arr = rng.rand(32, 64).astype(np.float32)
    assert arr.nbytes < _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_stripped(arr, COMPRESSION_DEFLATE, predictor=1,
                    rows_per_strip=8)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_stripped'
    )
    assert not any(spy.calls), (
        f'sequential strip writer must use gil_friendly=False; '
        f'observed flags: {spy.calls}'
    )


def test_write_tiled_parallel_path_uses_gil_friendly(monkeypatch):
    """Parallel tile writer must pass ``gil_friendly=True`` to deflate."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    assert arr.nbytes > _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=512)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_tiled'
    )
    assert all(spy.calls), (
        f'parallel tile writer must pass gil_friendly=True to every '
        f'deflate_compress call; observed flags: {spy.calls}'
    )


def test_write_tiled_sequential_path_uses_default(monkeypatch):
    """Sequential tile writer (small payload) must keep
    ``gil_friendly=False``."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(128, 128).astype(np.float32)
    assert arr.nbytes < _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=32)

    assert spy.calls
    assert not any(spy.calls), (
        f'sequential tile writer must use gil_friendly=False; '
        f'observed flags: {spy.calls}'
    )


def test_prepare_strip_forwards_gil_friendly(monkeypatch):
    """`_prepare_strip` must forward its ``gil_friendly`` kwarg to compress.

    Direct unit pin: walks the writer's per-strip helper for both flag
    values and asserts the deflate call observed the flag.
    """
    rng = np.random.RandomState(1830)
    arr = rng.rand(64, 64).astype(np.float32)

    spy = _DeflateCallSpy(monkeypatch)
    _prepare_strip(arr, 0, 8, 64, 64, 1, np.float32, 4,
                   predictor=1, compression=COMPRESSION_DEFLATE,
                   gil_friendly=True)
    _prepare_strip(arr, 0, 8, 64, 64, 1, np.float32, 4,
                   predictor=1, compression=COMPRESSION_DEFLATE,
                   gil_friendly=False)

    assert spy.calls == [True, False], (
        f'_prepare_strip must forward gil_friendly to deflate_compress; '
        f'observed flags: {spy.calls}'
    )


def test_prepare_tile_forwards_gil_friendly(monkeypatch):
    """`_prepare_tile` must forward its ``gil_friendly`` kwarg to compress."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(64, 64).astype(np.float32)

    spy = _DeflateCallSpy(monkeypatch)
    _prepare_tile(arr, 0, 0, 32, 32, 64, 64, 1, np.float32, 4,
                  predictor=1, compression=COMPRESSION_DEFLATE,
                  gil_friendly=True)
    _prepare_tile(arr, 0, 0, 32, 32, 64, 64, 1, np.float32, 4,
                  predictor=1, compression=COMPRESSION_DEFLATE,
                  gil_friendly=False)

    assert spy.calls == [True, False], (
        f'_prepare_tile must forward gil_friendly to deflate_compress; '
        f'observed flags: {spy.calls}'
    )


def test_write_tiled_parallel_passes_gil_friendly_positionally(monkeypatch):
    """The parallel tile branch passes ``True`` as the *positional*
    ``gil_friendly`` argument to ``_prepare_tile`` (see _writer.py:943).

    Pin the positional contract: if the keyword-order of _prepare_tile
    changes, this test will flag it instead of silently swapping a
    different bool into ``gil_friendly`` and quietly regressing perf.
    """
    captured = []
    real_prepare = _prepare_tile

    def _wrapper(*args, **kwargs):
        # Positional order matches the signature; kwargs holds the rest.
        # gil_friendly is the trailing arg in the call inside _write_tiled.
        captured.append({'args': args, 'kwargs': kwargs})
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(
        'xrspatial.geotiff._writer._prepare_tile', _wrapper)

    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=512)

    assert captured, '_prepare_tile must be invoked'
    # The parallel branch invokes _prepare_tile with all 15 positional
    # args from data..gil_friendly. Index 14 is gil_friendly. If a
    # future refactor switches to keywords, the flag must still resolve
    # to True.
    import inspect
    sig = inspect.signature(_prepare_tile)
    param_names = list(sig.parameters.keys())
    gil_idx = param_names.index('gil_friendly')

    for call in captured:
        if len(call['args']) > gil_idx:
            assert call['args'][gil_idx] is True, (
                f'_write_tiled parallel branch must pass True as the '
                f'positional gil_friendly arg (index {gil_idx}); '
                f'got {call["args"][gil_idx]!r}'
            )
        else:
            assert call['kwargs'].get('gil_friendly') is True, (
                f'_write_tiled parallel branch must set gil_friendly=True; '
                f'call args={call["args"]!r} kwargs={call["kwargs"]!r}'
            )


# ---------------------------------------------------------------------------
# End-to-end: writes still round-trip with the flag forwarded.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('size,tiled,tile_size', [
    (2048, False, None),     # large strip parallel path
    (2048, True, 512),       # large tile parallel path
    (32, False, None),       # small strip sequential path
    (128, True, 32),         # small tile sequential path
])
def test_write_deflate_round_trip_across_parallelism_modes(
        tmp_path, size, tiled, tile_size):
    """End-to-end round-trip on both the sequential and parallel paths.

    Whichever ``gil_friendly`` value the writer selects, the bytes must
    decode back to the source exactly.
    """
    rng = np.random.RandomState(1830)
    expected = rng.rand(size, size).astype(np.float32)
    path = str(tmp_path / f'gilfriendly_{size}_{tiled}_{tile_size}.tif')
    kwargs = {'compression': 'deflate', 'tiled': tiled}
    if tile_size is not None:
        kwargs['tile_size'] = tile_size
    write(expected, path, **kwargs)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)
