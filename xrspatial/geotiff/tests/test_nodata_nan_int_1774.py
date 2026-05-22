"""Regression tests for issue #1774.

Reading an integer GeoTIFF whose ``GDAL_NODATA`` tag holds a non-finite
string (``"NaN"`` / ``"nan"`` / ``"Inf"`` / ``"-Inf"``) used to crash with
``ValueError: cannot convert float NaN to integer`` at three call sites in
``xrspatial/geotiff/__init__.py``:

* ``open_geotiff`` eager numpy path
* ``_apply_nodata_mask_gpu`` (GPU)
* ``_delayed_read_window`` (dask)

The fix gates each ``int(nodata)`` cast on ``np.isfinite(nodata)``, mirroring
the ``_resolve_masked_fill`` / ``_sparse_fill_value`` helpers in
``_reader.py``. A non-finite sentinel on an integer file cannot match any
pixel value, so the mask is a no-op and the file dtype is preserved.
``attrs['nodata']`` still carries the raw sentinel so a write round-trip
keeps the original GDAL_NODATA tag.

The same gate is paired with ``float(nodata).is_integer()`` so that a
fractional ``GDAL_NODATA`` string (e.g. ``"3.5"`` on a ``uint16`` file)
also stays a no-op rather than truncating to ``int(3.5) == 3`` and
silently masking real pixel value 3. This mirrors the
``_writer.py`` / ``_vrt.py`` pattern used for #1564 / #1616.
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


def _build_uint16_tiff(nodata_str: str, tmp_path) -> str:
    """Write a minimal 2x2 uint16 TIFF with GDAL_NODATA=<nodata_str>.

    Hand-rolled rather than going through ``to_geotiff`` so the GDAL_NODATA
    tag carries arbitrary string content (``"nan"``, ``"Inf"``, etc.). The
    writer would refuse those at the resolve-nodata step before the file
    ever lands on disk.
    """
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_bytes = nodata_str.encode('ascii') + b'\x00'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag: int, val: int) -> None:
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag: int, val: int) -> None:
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_ascii(tag: int, data: bytes) -> None:
        tag_list.append((tag, 2, len(data), data))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 16)   # BitsPerSample
    add_short(259, 1)    # Compression = none
    add_short(262, 1)    # Photometric = MinIsBlack
    add_short(277, 1)    # SamplesPerPixel
    add_short(278, height)  # RowsPerStrip
    add_long(273, 0)     # StripOffsets (patched after layout)
    add_long(279, len(pixels.tobytes()))  # StripByteCounts
    add_short(339, 1)    # SampleFormat = uint
    add_ascii(42113, nodata_bytes)  # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_base = ifd_start + ifd_size
    overflow_buf = bytearray()

    processed: list[tuple[int, int, int, bytes]] = []
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            ovf_pos = overflow_base + len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
            new_raw = struct.pack(f'{bo}I', ovf_pos)
        else:
            new_raw = raw
        processed.append((tag, typ, count, new_raw))

    pixel_start = overflow_base + len(overflow_buf)
    for i, (tag, typ, count, raw) in enumerate(processed):
        if tag == 273:
            processed[i] = (tag, typ, count,
                            struct.pack(f'{bo}I', pixel_start))

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in processed:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        out.extend(raw.ljust(4, b'\x00'))
    out.extend(struct.pack(f'{bo}I', 0))  # next IFD = 0
    out.extend(overflow_buf)
    out.extend(pixels.tobytes())

    path = str(tmp_path / f'uint16_nodata_{nodata_str.replace("-", "neg")}.tif')
    with open(path, 'wb') as f:
        f.write(bytes(out))
    return path


@pytest.mark.parametrize('nodata_str', ['nan', 'NaN', 'NAN'])
def test_open_geotiff_eager_int_nodata_nan(tmp_path, nodata_str):
    """Eager numpy path: NaN nodata on uint16 file is a no-op (#1774)."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = open_geotiff(path)
    # No pixel can match NaN, so the dtype stays uint16
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    # The raw sentinel survives on attrs so write round-trips keep the tag
    assert 'nodata' in da.attrs
    assert np.isnan(da.attrs['nodata'])


@pytest.mark.parametrize('nodata_str', ['inf', 'Inf', 'INF',
                                        '-inf', '-Inf', '-INF'])
def test_open_geotiff_eager_int_nodata_inf(tmp_path, nodata_str):
    """Eager numpy path: +/-Inf nodata on uint16 file is a no-op (#1774)."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = open_geotiff(path)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    assert 'nodata' in da.attrs
    assert np.isinf(da.attrs['nodata'])


def test_open_geotiff_eager_int_nodata_finite_still_masks(tmp_path):
    """Regression guard: in-range finite sentinel still masks correctly."""
    # 30 is one of the pixel values; using it as a sentinel masks one pixel.
    path = _build_uint16_tiff('30', tmp_path)
    da = open_geotiff(path)
    # uint16 + in-range sentinel hit promotes to float64 with NaN
    assert da.dtype == np.float64
    assert np.isnan(da.values[1, 0])
    assert da.values[0, 0] == 10
    assert da.attrs['nodata'] == 30


def test_read_geotiff_dask_int_nodata_nan(tmp_path):
    """Dask path: NaN nodata on uint16 file is a no-op (#1774)."""
    path = _build_uint16_tiff('nan', tmp_path)
    da = read_geotiff_dask(path, chunks=2)
    # effective_dtype stays uint16 because the sentinel is non-finite
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])
    assert 'nodata' in da.attrs
    assert np.isnan(da.attrs['nodata'])


def test_read_geotiff_dask_int_nodata_inf(tmp_path):
    """Dask path: Inf nodata on uint16 file is a no-op (#1774)."""
    path = _build_uint16_tiff('inf', tmp_path)
    da = read_geotiff_dask(path, chunks=2)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])
    assert np.isinf(da.attrs['nodata'])


@_gpu_only
def test_apply_nodata_mask_gpu_int_nan_noop():
    """GPU helper: NaN nodata on uint16 array is a no-op (#1774)."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, float('nan'))
    # No promotion, same buffer back
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])


@_gpu_only
def test_apply_nodata_mask_gpu_int_inf_noop():
    """GPU helper: Inf nodata on uint16 array is a no-op (#1774)."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, float('inf'))
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])


@_gpu_only
def test_apply_nodata_mask_gpu_int_finite_still_masks():
    """GPU helper regression guard: in-range finite sentinel still masks."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, 3)
    # 3 is in range and hits a pixel; promotes to float64 with NaN
    assert out.dtype == cupy.float64
    arr = out.get()
    assert np.isnan(arr[1, 0])
    assert arr[0, 0] == 1.0


# ----------------------------------------------------------------------
# Fractional GDAL_NODATA on integer files (Copilot follow-up review)
# ----------------------------------------------------------------------
# A fractional sentinel like ``"3.5"`` on a ``uint16`` file is similarly
# nonsensical: ``int(3.5) == 3`` would silently flag a real pixel value
# as nodata. The four masking sites must treat fractional sentinels the
# same as non-finite ones (no-op, preserve dtype, preserve raw attr).


@pytest.mark.parametrize('nodata_str', ['3.5', '29.5', '0.5'])
def test_open_geotiff_eager_int_nodata_fractional_noop(tmp_path, nodata_str):
    """Eager numpy path: fractional nodata on uint16 is a no-op."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = open_geotiff(path)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    assert da.attrs['nodata'] == float(nodata_str)


def test_open_geotiff_eager_int_nodata_fractional_does_not_alias_truncation(
    tmp_path,
):
    """A ``"30.5"`` sentinel must not mask the real pixel value 30
    (which is in the test image). ``int(30.5)`` would truncate to 30
    without the integerness gate.
    """
    path = _build_uint16_tiff('30.5', tmp_path)
    da = open_geotiff(path)
    assert da.dtype == np.uint16
    # pixel @[1,0] is 30; the fractional sentinel must NOT have masked it
    assert da.values[1, 0] == 30
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


def test_read_geotiff_dask_int_nodata_fractional_noop(tmp_path):
    """Dask path: fractional nodata on uint16 is a no-op."""
    path = _build_uint16_tiff('30.5', tmp_path)
    da = read_geotiff_dask(path, chunks=2)
    # effective_dtype stays uint16 because the sentinel is fractional
    assert da.dtype == np.uint16
    computed = da.compute().values
    assert computed[1, 0] == 30
    np.testing.assert_array_equal(computed, [[10, 20], [30, 40]])
    assert da.attrs['nodata'] == 30.5


@_gpu_only
def test_apply_nodata_mask_gpu_int_fractional_noop():
    """GPU helper: fractional nodata on uint16 is a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, 3.5)
    # 3.5 cannot match any uint16 pixel; ``int(3.5) == 3`` would have
    # truncated and masked the real pixel value 3.
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])
