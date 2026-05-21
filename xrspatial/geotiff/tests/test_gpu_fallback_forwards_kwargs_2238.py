"""GPU CPU-fallback paths forward read kwargs (issue #2238).

``read_geotiff_gpu`` has four CPU-fallback call sites to
``_read_to_array``:

- the stripped-layout branch at gpu.py:491 (long-standing)
- the planar=2 per-band stage-2 fallback (gpu.py around line 684)
- the sparse-tile fallback (gpu.py around line 697)
- the GPU-decode-failure fallback (gpu.py around line 784)

Before #2238 the last three dropped ``allow_rotated``, ``window``,
``band``, and ``max_pixels`` and the stripped branch dropped
``allow_rotated``. The later ``_gpu_apply_window_band(...)`` slicer can
repair shape for ``window``/``band`` after the fact, but it cannot
repair ``allow_rotated=False`` (the CPU parser has already raised on
the rotated transform) and cannot raise the caller's ``max_pixels``
ceiling after the parser has rejected the IFD.

These tests pin the fix: each fallback site must hand the caller's
kwargs through to ``_read_to_array``.
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest

from xrspatial.geotiff._geotags import TAG_MODEL_TRANSFORMATION


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


# Rotated 4x4 ModelTransformation: pixel_width 1.0, b=0.1 (column-axis
# rotation), pixel_height -1.0, origin (100, 200). Same shape as the
# fixture in test_allow_rotated_no_crs_2122.py.
_ROTATED_M_2238 = (
    1.0, 0.1, 0.0, 100.0,
    0.0, -1.0, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _write_rotated_tiled_tiff(path, arr: np.ndarray, *,
                              tile_w: int = 16, tile_h: int = 16,
                              sparse: bool = False) -> None:
    """Write a single-IFD tiled TIFF with a rotated ModelTransformation.

    Hand-rolled to keep the fixture independent of rasterio/GDAL. The
    output has TileWidth/TileLength tags so the GPU reader takes the
    tiled branch, plus the rotated transform so ``allow_rotated`` is
    required to read it.

    When ``sparse=True``, the last tile is marked with offset=0 and
    byte_count=0 (or the single tile is, for 1x1 grids). This drives
    the reader's ``has_sparse_tile=True`` path so the GPU sparse-tile
    fallback at ``gpu.py:697`` is exercised.
    """
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    tiles_across = (w + tile_w - 1) // tile_w
    tiles_down = (h + tile_h - 1) // tile_h
    n_tiles = tiles_across * tiles_down
    tile_bytes = tile_w * tile_h * 2

    # Pad each tile up to tile_w x tile_h and pack tiles row-major.
    tile_payloads = []
    for ty in range(tiles_down):
        for tx in range(tiles_across):
            tile = np.zeros((tile_h, tile_w), dtype='<u2')
            r0, c0 = ty * tile_h, tx * tile_w
            r1, c1 = min(r0 + tile_h, h), min(c0 + tile_w, w)
            tile[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            tile_payloads.append(tile.tobytes())

    # When sparse, drop the last tile from the file body and mark it
    # offset=0/bytecount=0 in the IFD tables. A single-tile file in
    # sparse mode marks that one tile sparse.
    if sparse:
        real_tiles = max(n_tiles - 1, 1) if n_tiles > 1 else 0
    else:
        real_tiles = n_tiles

    header_size = 8
    tile_data_off = header_size
    tile_data_size = real_tiles * tile_bytes
    offsets_arr_off = tile_data_off + tile_data_size
    offsets_arr_size = n_tiles * 4
    bytecounts_arr_off = offsets_arr_off + offsets_arr_size
    bytecounts_arr_size = n_tiles * 4
    transform_off = bytecounts_arr_off + bytecounts_arr_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    tile_offsets = [tile_data_off + i * tile_bytes for i in range(real_tiles)]
    tile_byte_counts = [tile_bytes] * real_tiles
    if sparse:
        # Pad missing entries with (0, 0) for the sparse tile(s).
        n_sparse = n_tiles - real_tiles
        tile_offsets.extend([0] * n_sparse)
        tile_byte_counts.extend([0] * n_sparse)

    entries = [
        (256, 3, 1, w),                # ImageWidth
        (257, 3, 1, h),                # ImageLength
        (258, 3, 1, 16),               # BitsPerSample = 16
        (259, 3, 1, 1),                # Compression = none
        (262, 3, 1, 1),                # Photometric = BlackIsZero
        (277, 3, 1, 1),                # SamplesPerPixel
        (322, 3, 1, tile_w),           # TileWidth
        (323, 3, 1, tile_h),           # TileLength
        (324, 4, n_tiles, offsets_arr_off),       # TileOffsets
        (325, 4, n_tiles, bytecounts_arr_off),    # TileByteCounts
        (339, 3, 1, 1),                # SampleFormat = unsigned int
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])

    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:  # SHORT
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)  # next IFD

    with open(path, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        for payload in tile_payloads[:real_tiles]:
            f.write(payload)
        f.write(struct.pack(f'<{n_tiles}I', *tile_offsets))
        f.write(struct.pack(f'<{n_tiles}I', *tile_byte_counts))
        f.write(struct.pack('<16d', *_ROTATED_M_2238))
        f.write(ifd_bytes)


def _make_kwarg_recorder():
    """Build a wrapper that records every kwargs dict it sees.

    The wrapper still calls through to the real ``_read_to_array`` so
    the GPU pipeline keeps working; tests assert against the recorded
    kwargs.
    """
    from xrspatial.geotiff._reader import read_to_array as _real

    seen: list[dict] = []

    def _wrapper(*args, **kwargs):
        seen.append(dict(kwargs))
        return _real(*args, **kwargs)

    return _wrapper, seen


# ---------------------------------------------------------------------------
# Stripped-layout branch (gpu.py:491). Already forwarded window/band/max_pixels
# before #2238; this issue adds allow_rotated.
# ---------------------------------------------------------------------------


@_gpu_only
def test_stripped_fallback_forwards_allow_rotated(tmp_path, monkeypatch):
    """Stripped-layout CPU fallback receives ``allow_rotated``.

    Earlier the stripped branch already forwarded window/band/max_pixels
    via #1732 but dropped ``allow_rotated``. The xfail-flipped test in
    ``test_allow_rotated_no_crs_2122.py`` proves the end-to-end behaviour;
    this one pins the kwarg-forwarding contract via a recorder so a
    later refactor cannot silently drop the kwarg again.
    """
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    # Build a stripped (non-tiled) rotated single-strip TIFF directly.
    src = tmp_path / "2238_stripped_rotated.tif"
    h, w = 4, 5
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    ifd_off = transform_off + 16 * 8

    entries = [
        (256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 16),
        (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, header_size),
        (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, strip_size),
        (339, 3, 1, 1),
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])
    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)
    with open(src, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *_ROTATED_M_2238))
        f.write(ifd_bytes)

    wrapper, seen = _make_kwarg_recorder()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    da = read_geotiff_gpu(str(src), allow_rotated=True)

    assert len(seen) == 1, f"expected one fallback call, got {len(seen)}"
    assert seen[0].get('allow_rotated') is True, (
        f"stripped fallback dropped allow_rotated; kwargs={seen[0]}"
    )
    # Sanity: the call succeeded end-to-end (no NotImplementedError).
    np.testing.assert_array_equal(da.data.get(), arr)


# ---------------------------------------------------------------------------
# Sparse-tile branch (gpu.py:697).
# ---------------------------------------------------------------------------


@_gpu_only
def test_sparse_tile_fallback_forwards_all_kwargs(tmp_path, monkeypatch):
    """Sparse-tile fallback hands every caller kwarg to ``_read_to_array``."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_sparse_rotated.tif"
    h, w = 32, 32
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    _write_rotated_tiled_tiff(str(src), arr, tile_w=16, tile_h=16, sparse=True)

    wrapper, seen = _make_kwarg_recorder()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_window = (0, 0, 16, 16)
    requested_max_pixels = 10_000

    da = read_geotiff_gpu(
        str(src),
        allow_rotated=True,
        window=requested_window,
        max_pixels=requested_max_pixels,
    )

    assert len(seen) == 1, (
        f"expected one sparse-tile fallback call to _read_to_array, "
        f"got {len(seen)} (kwargs sequence: {seen})"
    )
    call = seen[0]
    assert call.get('allow_rotated') is True, (
        f"sparse-tile fallback dropped allow_rotated; kwargs={call}"
    )
    assert call.get('window') == requested_window, (
        f"sparse-tile fallback dropped window; kwargs={call}"
    )
    assert call.get('max_pixels') == requested_max_pixels, (
        f"sparse-tile fallback dropped max_pixels; kwargs={call}"
    )
    # band defaults to None for single-band reads.
    assert 'band' in call, (
        f"sparse-tile fallback did not pass band kwarg; kwargs={call}"
    )

    # The fallback round-trips data correctly on the filled region.
    host = da.data.get()
    assert host.shape == (16, 16), host.shape
    # The unmasked output: top-left quadrant of the source.
    assert np.array_equal(host, arr[:16, :16].astype(host.dtype))


# ---------------------------------------------------------------------------
# Planar=2 fallback (gpu.py:684) and GPU-decode-failure fallback (gpu.py:784).
# Both branches are only reachable via a forced fallback because the GPU
# decoder normally succeeds. Monkeypatching the GPU decode entry points to
# raise/return-None drives control flow through the fallback site.
# ---------------------------------------------------------------------------


@_gpu_only
def test_gpu_decode_failure_fallback_forwards_all_kwargs(tmp_path,
                                                        monkeypatch):
    """``gpu_decode_tiles`` failure routes through fallback with kwargs."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff import _gpu_decode
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_decode_fail.tif"
    h, w = 32, 32
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    _write_rotated_tiled_tiff(str(src), arr, tile_w=16, tile_h=16)

    def _raise_from_file(*args, **kwargs):
        raise RuntimeError("synthetic GDS decode failure (test #2238)")

    def _raise_decode(*args, **kwargs):
        raise RuntimeError("synthetic GPU tile decode failure (test #2238)")

    # ``gpu.py`` imports both decoders lazily from ``_gpu_decode``, so
    # patching the source module is sufficient -- there is no
    # module-level binding to patch on gpu_backend itself.
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _raise_decode, raising=True)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _raise_from_file,
        raising=True)

    wrapper, seen = _make_kwarg_recorder()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_window = (0, 0, 16, 16)
    requested_max_pixels = 5_000

    da = read_geotiff_gpu(
        str(src),
        allow_rotated=True,
        window=requested_window,
        max_pixels=requested_max_pixels,
    )

    # At least one fallback hit the recorder, and the most recent one
    # carried all four kwargs.
    assert seen, "GPU-decode-failure fallback did not call _read_to_array"
    call = seen[-1]
    assert call.get('allow_rotated') is True, (
        f"decode-failure fallback dropped allow_rotated; kwargs={call}"
    )
    assert call.get('window') == requested_window, (
        f"decode-failure fallback dropped window; kwargs={call}"
    )
    assert call.get('max_pixels') == requested_max_pixels, (
        f"decode-failure fallback dropped max_pixels; kwargs={call}"
    )
    assert 'band' in call, (
        f"decode-failure fallback did not pass band kwarg; kwargs={call}"
    )

    host = da.data.get()
    assert host.shape == (16, 16), host.shape
    np.testing.assert_array_equal(host, arr[:16, :16].astype(host.dtype))


@_gpu_only
def test_planar2_fallback_forwards_all_kwargs(tmp_path, monkeypatch):
    """planar=2 per-band stage-2 fallback forwards every read kwarg.

    Uses a non-rotated planar=2 multi-band file (rotated + multi-band is
    a harder fixture to write by hand) and monkeypatches the per-band
    decoder to return ``None``, which drives ``cpu_fallback_needed``
    true and exercises gpu.py:684. The kwarg assertions are the
    invariant; the rotated case is covered by the sparse-tile and
    decode-failure tests above.
    """
    tifffile = pytest.importorskip("tifffile")

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_planar2.tif"
    bands, h, w = 2, 64, 64
    rng = np.random.RandomState(2238)
    data = rng.randint(0, 255, size=(bands, h, w)).astype(np.uint8)
    tifffile.imwrite(
        str(src), data,
        photometric='minisblack',
        planarconfig='separate',
        tile=(32, 32),
    )

    # Force ``_gpu_decode_single_band_tiles`` to signal "stage-2 failed"
    # via its ``return None`` contract, which drives the planar=2 branch
    # into the CPU fallback (gpu.py:684).
    def _none_band(*args, **kwargs):
        return None

    monkeypatch.setattr(
        gpu_backend, '_gpu_decode_single_band_tiles', _none_band,
        raising=True)

    wrapper, seen = _make_kwarg_recorder()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_max_pixels = 50_000  # 64*64*2 = 8192 < 50000

    da = read_geotiff_gpu(
        str(src),
        max_pixels=requested_max_pixels,
    )

    assert seen, "planar=2 fallback did not call _read_to_array"
    call = seen[-1]
    assert call.get('max_pixels') == requested_max_pixels, (
        f"planar=2 fallback dropped max_pixels; kwargs={call}"
    )
    assert 'allow_rotated' in call, (
        f"planar=2 fallback did not pass allow_rotated; kwargs={call}"
    )
    assert call.get('allow_rotated') is False, (
        f"planar=2 fallback default allow_rotated should be False; "
        f"kwargs={call}"
    )
    assert 'window' in call and 'band' in call, (
        f"planar=2 fallback did not pass window/band; kwargs={call}"
    )

    # Sanity: full-image round-trip matches the source data.
    host = da.data.get()
    assert host.shape == (h, w, bands), host.shape
    np.testing.assert_array_equal(host, np.transpose(data, (1, 2, 0)))


@_gpu_only
def test_decode_failure_fallback_applies_window_band(tmp_path, monkeypatch):
    """Window/band selection on the decode-failure fallback path matches.

    The fix forwards ``window``/``band`` to ``_read_to_array`` and then
    skips the post-decode ``_gpu_apply_window_band`` slicer (the buffer
    is already windowed). Verifies the end-to-end shape and values do
    not depend on the slicer for the CPU-fallback path.
    """
    tifffile = pytest.importorskip("tifffile")

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff import _gpu_decode

    src = tmp_path / "2238_windowed_fallback.tif"
    bands, h, w = 3, 64, 64
    rng = np.random.RandomState(0x2238)
    data = rng.randint(0, 200, size=(bands, h, w)).astype(np.uint8)
    # planarconfig='contig' expects (H, W, S) layout for tifffile; the
    # source ``data`` is (S, H, W) so transpose before writing.
    tifffile.imwrite(
        str(src), np.transpose(data, (1, 2, 0)),
        photometric='rgb',
        planarconfig='contig',
        tile=(32, 32),
    )

    def _raise_decode(*args, **kwargs):
        raise RuntimeError("force CPU fallback (test #2238)")

    def _raise_from_file(*args, **kwargs):
        raise RuntimeError("force CPU fallback (test #2238)")

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _raise_decode, raising=True)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _raise_from_file,
        raising=True)

    requested_window = (8, 4, 40, 36)  # (r0, c0, r1, c1) -> 32x32 view
    requested_band = 1

    da = read_geotiff_gpu(
        str(src),
        window=requested_window,
        band=requested_band,
    )

    expected_h = requested_window[2] - requested_window[0]
    expected_w = requested_window[3] - requested_window[1]
    host = da.data.get()
    assert host.shape == (expected_h, expected_w), host.shape

    r0, c0, r1, c1 = requested_window
    expected = data[requested_band, r0:r1, c0:c1]
    np.testing.assert_array_equal(host, expected)
