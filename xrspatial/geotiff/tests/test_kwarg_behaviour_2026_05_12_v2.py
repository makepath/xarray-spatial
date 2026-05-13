"""Parameter-coverage gap closure for the geotiff module (pass 10).

Test coverage gap sweep 2026-05-12 (pass 10). Two Cat 4 HIGH
parameter-coverage gaps closed here.

Cat 4 HIGH #1 -- ``write_geotiff_gpu(predictor=)``. The CPU writer has
dense coverage of ``predictor=True``/``2``/``3`` via
``test_predictor_fp_write_1313.py``, ``test_predictor_multisample.py``,
and ``test_predictor2_big_endian.py``. The GPU writer threads
``predictor=`` through ``normalize_predictor`` and
``gpu_compress_tiles`` into the five CUDA encode kernels
(``_predictor_encode_kernel_u8``/``_u16``/``_u32``/``_u64`` for
predictor=2, plus ``_fp_predictor_encode_kernel`` for predictor=3),
but no test calls ``write_geotiff_gpu`` with a non-default predictor.
A regression dropping the predictor-encode call from
``gpu_compress_tiles`` would silently emit files that advertise the
predictor tag but contain un-differenced bytes, breaking decode
through this library's own reader, GDAL, rasterio, and libtiff. A
correctness bug in any of the five CUDA encode kernels would likewise
ship undetected because the only existing GPU-predictor tests cover
the *decode* kernels (see ``test_predictor_multisample.py``,
``test_predictor2_big_endian_gpu_1517.py``).

Cat 4 HIGH #2 -- ``read_vrt(window=)``. The public ``read_vrt``
documents ``window: tuple or None`` and the internal
``_vrt.read_vrt`` implements full windowed-read semantics (window
clipping, dst_rect overlap, src/dst coordinate scaling, per-band
nodata handling, GeoTransform origin shift on coords +
``attrs['transform']``). The only existing window-related VRT test is
the signature-annotation pin in
``test_signature_annotations_1654.py``; no test exercises behaviour.
A regression that ignored the kwarg and read the full mosaic would
silently inflate memory + I/O on the windowed-read fast path that
real callers depend on. A regression in the origin-shift block would
return shifted coords inconsistent with ``open_geotiff(window=)``.
"""
from __future__ import annotations

import importlib.util
import os
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
)
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


# --------------------------------------------------------------------------
# GPU gating
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _read_predictor_tag(path: str) -> int | None:
    """Read TIFF Predictor tag (id=317). Returns None if absent."""
    with open(path, 'rb') as f:
        header = f.read(8)
    assert header[:2] == b'II', "test fixture writes little-endian"
    magic = struct.unpack('<H', header[2:4])[0]
    assert magic == 42, "classic TIFF expected"
    ifd_offset = struct.unpack('<I', header[4:8])[0]

    with open(path, 'rb') as f:
        f.seek(ifd_offset)
        n_entries = struct.unpack('<H', f.read(2))[0]
        for _ in range(n_entries):
            entry = f.read(12)
            tag, _type_id, _count = struct.unpack('<HHI', entry[:8])
            if tag == 317:
                return struct.unpack('<H', entry[8:10])[0]
    return None  # tag absent => predictor 1 (none)


def _da_with_float_coords(arr) -> xr.DataArray:
    """Wrap a 2D or 3D array of any dtype with float64 y/x coords.

    Accepts numpy or cupy arrays. For 2D inputs returns a (y, x)
    DataArray; for 3D inputs returns a (y, x, band) DataArray with
    an integer band index. The element dtype is preserved from the
    input; only the y/x coordinate arrays are forced to float64 so
    pixel-is-area transforms round-trip cleanly through the
    geotiff/VRT writers.
    """
    h, w = arr.shape[:2]
    coords = {
        'y': np.arange(h, dtype=np.float64),
        'x': np.arange(w, dtype=np.float64),
    }
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=('y', 'x'), coords=coords)
    return xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        coords={**coords, 'band': np.arange(arr.shape[2])},
    )


# --------------------------------------------------------------------------
# Cat 4 HIGH #1: write_geotiff_gpu(predictor=)
# --------------------------------------------------------------------------


@_gpu_only
class TestWriteGeotiffGpuPredictor2Uint8:
    """``predictor=True`` / ``predictor=2`` on uint8 data.

    Exercises the ``_predictor_encode_kernel_u8`` CUDA kernel via
    ``_gpu_predictor2_encode`` dispatch.
    """

    def test_predictor_true_uint8_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(0)
        arr = rng.randint(0, 256, size=(8, 16), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=True,
                          tile_size=16)

        # Round-trip through the public reader
        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        # On-disk Predictor tag advertises horizontal differencing
        assert _read_predictor_tag(path) == 2

    def test_predictor_2_uint8_round_trip(self, tmp_path):
        """``predictor=2`` (int form) is equivalent to ``predictor=True``."""
        import cupy
        rng = np.random.RandomState(1)
        arr = rng.randint(0, 256, size=(8, 16), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_int_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2

    def test_predictor_2_uint8_3band_rgb(self, tmp_path):
        """Multi-sample (3-band) uint8 with ``predictor=2``.

        Stride is ``samples_per_pixel`` in the encode kernel, so the
        decode must reverse the same stride. A regression dropping
        ``samples`` from ``_gpu_predictor2_encode`` would write data
        differentiated by 1 byte but advertise multi-sample tiles,
        producing garbled colours on read.
        """
        import cupy
        rng = np.random.RandomState(2)
        arr = rng.randint(0, 256, size=(8, 16, 3), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u8_3band_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2

    def test_predictor_false_no_predictor_tag(self, tmp_path):
        """``predictor=False`` writes no Predictor tag (default behaviour).

        Pins the contrast with ``predictor=True``: without this test, a
        regression that flipped the default to ``predictor=2`` would
        round-trip but advertise predictor=2 in the output file.
        """
        import cupy
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_no_pred_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=False,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        # Predictor tag absent or explicitly 1 (no predictor)
        tag = _read_predictor_tag(path)
        assert tag is None or tag == 1


@_gpu_only
class TestWriteGeotiffGpuPredictor2Uint16:
    """``predictor=2`` on uint16 data.

    Exercises ``_predictor_encode_kernel_u16`` (16-bit sample stride).
    """

    def test_predictor_2_uint16_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(3)
        arr = rng.randint(0, 60000, size=(8, 16), dtype=np.uint16)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u16_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2


@_gpu_only
class TestWriteGeotiffGpuPredictor2Int32:
    """``predictor=2`` on int32 data.

    Exercises ``_predictor_encode_kernel_u32`` (32-bit sample stride).
    Int32 is viewed as uint32 for differencing semantics; the round
    trip must reproduce the signed values exactly.
    """

    def test_predictor_2_int32_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(4)
        # Mix of negative and positive to ensure the unsigned-view
        # differencing round-trips through the signed interpretation
        arr = rng.randint(-1_000_000, 1_000_000, size=(8, 16),
                          dtype=np.int32)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_i32_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2


@_gpu_only
class TestWriteGeotiffGpuPredictor3Float:
    """``predictor=3`` (floating-point predictor).

    Exercises ``_fp_predictor_encode_kernel`` for both float32 and
    float64 (bps=4 and bps=8). The kernel does a byte-swizzle
    (MSB-first lane layout) followed by horizontal differencing per
    TIFF Technical Note 3; both bps must round-trip exactly.
    """

    def test_predictor_3_float32_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(5)
        # Smooth-ish values so fp predictor actually compresses
        # (round-trip semantics do not depend on smoothness, but a
        # mix of magnitudes exercises the byte-swizzle on all 4 lanes)
        arr = rng.uniform(-1000.0, 1000.0, size=(8, 16)).astype(np.float32)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_f32_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                          tile_size=16)

        out = open_geotiff(path)
        # FP predictor is lossless: equality, not allclose
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 3

    def test_predictor_3_float64_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(6)
        arr = rng.uniform(-1e9, 1e9, size=(8, 16)).astype(np.float64)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_f64_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 3

    def test_predictor_3_rejects_int_dtype(self, tmp_path):
        """FP predictor refuses non-float dtypes (parity with CPU writer)."""
        import cupy
        arr = np.arange(64, dtype=np.int32).reshape(8, 8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_reject_2026_05_12_v2.tif')

        with pytest.raises(ValueError,
                           match=r"predictor=3.*requires float"):
            write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                              tile_size=16)


@_gpu_only
class TestWriteGeotiffGpuPredictorCpuParity:
    """Pixel-exact parity between CPU ``to_geotiff(predictor=X)`` and
    GPU ``write_geotiff_gpu(predictor=X)``.

    Predictor encode is a lossless transform: identical inputs must
    produce identical decoded outputs regardless of whether the
    differencing ran on CPU or GPU. The compressed bytes may differ
    (different deflate library calls) but the round-tripped pixels
    must match.
    """

    def test_cpu_gpu_parity_predictor_2_uint16(self, tmp_path):
        import cupy
        rng = np.random.RandomState(7)
        arr = rng.randint(0, 60000, size=(8, 16), dtype=np.uint16)

        cpu_path = str(tmp_path / 'cpu_pred2_u16_v2.tif')
        gpu_path = str(tmp_path / 'gpu_pred2_u16_v2.tif')

        to_geotiff(_da_with_float_coords(arr), cpu_path,
                   compression='deflate', predictor=2, tile_size=16)
        write_geotiff_gpu(_da_with_float_coords(cupy.asarray(arr)), gpu_path,
                          compression='deflate', predictor=2, tile_size=16)

        cpu_out = open_geotiff(cpu_path).values
        gpu_out = open_geotiff(gpu_path).values
        np.testing.assert_array_equal(cpu_out, gpu_out)
        np.testing.assert_array_equal(cpu_out, arr)

    def test_cpu_gpu_parity_predictor_3_float32(self, tmp_path):
        import cupy
        rng = np.random.RandomState(8)
        arr = rng.uniform(-100.0, 100.0, size=(8, 16)).astype(np.float32)

        cpu_path = str(tmp_path / 'cpu_pred3_f32_v2.tif')
        gpu_path = str(tmp_path / 'gpu_pred3_f32_v2.tif')

        to_geotiff(_da_with_float_coords(arr), cpu_path,
                   compression='deflate', predictor=3, tile_size=16)
        write_geotiff_gpu(_da_with_float_coords(cupy.asarray(arr)), gpu_path,
                          compression='deflate', predictor=3, tile_size=16)

        cpu_out = open_geotiff(cpu_path).values
        gpu_out = open_geotiff(gpu_path).values
        np.testing.assert_array_equal(cpu_out, gpu_out)
        np.testing.assert_array_equal(cpu_out, arr)


# --------------------------------------------------------------------------
# Cat 4 HIGH #2: read_vrt(window=)
# --------------------------------------------------------------------------


def _write_tile_to_vrt(tmp_path, name: str, data: np.ndarray) -> str:
    """Write a single-source GeoTIFF tile for VRT inclusion."""
    from xrspatial.geotiff._writer import write
    path = str(tmp_path / name)
    write(data, path, compression='none', tiled=False)
    return path


def _make_single_tile_vrt(tmp_path, arr: np.ndarray) -> str:
    """Create a single-source VRT mosaic.

    Uses ``_vrt.write_vrt`` so source paths land relative to the VRT
    directory; that keeps the issue #1671 containment guard happy
    without environment variables.
    """
    tile_path = _write_tile_to_vrt(tmp_path, 'src_tile.tif', arr)
    vrt_path = str(tmp_path / 'single.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    return vrt_path


def _make_2x1_mosaic_vrt(tmp_path, left: np.ndarray,
                         right: np.ndarray) -> str:
    """Create a 2x1 horizontal mosaic VRT for cross-source window tests.

    Hand-built XML so the dst_rect placements are explicit -- VRT's
    write_vrt helper only handles single-source layouts directly.
    """
    h, lw = left.shape[:2]
    rw = right.shape[1]
    width = lw + rw

    lpath = _write_tile_to_vrt(tmp_path, 'left.tif', left)
    rpath = _write_tile_to_vrt(tmp_path, 'right.tif', right)

    dtype_map = {np.dtype('float32'): 'Float32',
                 np.dtype('float64'): 'Float64',
                 np.dtype('uint8'): 'Byte',
                 np.dtype('int32'): 'Int32',
                 np.dtype('uint16'): 'UInt16'}
    data_type = dtype_map[left.dtype]

    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{h}">',
        '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>',
        f'  <VRTRasterBand dataType="{data_type}" band="1">',
        '    <SimpleSource>',
        f'      <SourceFilename relativeToVRT="1">'
        f'{os.path.basename(lpath)}</SourceFilename>',
        '      <SourceBand>1</SourceBand>',
        f'      <SrcRect xOff="0" yOff="0" xSize="{lw}" ySize="{h}"/>',
        f'      <DstRect xOff="0" yOff="0" xSize="{lw}" ySize="{h}"/>',
        '    </SimpleSource>',
        '    <SimpleSource>',
        f'      <SourceFilename relativeToVRT="1">'
        f'{os.path.basename(rpath)}</SourceFilename>',
        '      <SourceBand>1</SourceBand>',
        f'      <SrcRect xOff="0" yOff="0" xSize="{rw}" ySize="{h}"/>',
        f'      <DstRect xOff="{lw}" yOff="0" xSize="{rw}" ySize="{h}"/>',
        '    </SimpleSource>',
        '  </VRTRasterBand>',
        '</VRTDataset>',
    ]

    vrt_path = str(tmp_path / 'mosaic_2x1.vrt')
    with open(vrt_path, 'w') as f:
        f.write('\n'.join(lines))
    return vrt_path


class TestReadVrtWindowEager:
    """Eager numpy ``read_vrt(window=...)`` slices the assembled raster."""

    def test_window_subregion_of_single_source(self, tmp_path):
        """Window picks a 4x6 sub-block from an 8x16 single-source VRT."""
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        # rows 2..6, cols 4..10
        result = read_vrt(vrt, window=(2, 4, 6, 10))

        assert result.shape == (4, 6)
        np.testing.assert_array_equal(result.values, arr[2:6, 4:10])

    def test_window_full_raster_matches_no_window(self, tmp_path):
        """``window=(0, 0, H, W)`` returns the same data as no window."""
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        full = read_vrt(vrt).values
        windowed = read_vrt(vrt, window=(0, 0, 8, 16)).values

        np.testing.assert_array_equal(windowed, full)

    def test_window_outside_raster_bounds_rejected(self, tmp_path):
        """Window extending past raster bounds raises ``ValueError``.

        ``read_vrt`` used to silently clamp out-of-bounds windows. That
        masked caller bugs (typo'd coords, off-by-one extents) and made
        the returned shape disagree with the caller's coord arrays. As
        of #1697 / #1698 the validator rejects such windows up front
        with a typed ``ValueError`` instead.
        """
        arr = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        with pytest.raises(ValueError, match="outside the VRT extent"):
            read_vrt(vrt, window=(0, 0, 100, 100))

    def test_window_negative_offsets_rejected(self, tmp_path):
        """Negative start offsets raise ``ValueError``.

        Per the post-#1697 contract, ``read_vrt`` validates the window
        against the VRT extent. Negative offsets are rejected the same
        way an over-large window is, rather than being silently clamped
        to zero.
        """
        arr = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        with pytest.raises(ValueError, match="outside the VRT extent"):
            read_vrt(vrt, window=(-1, -2, 3, 4))

    def test_window_across_mosaic_seam(self, tmp_path):
        """Window straddling a multi-source seam reads both sources.

        2x1 mosaic of two 4x4 tiles laid out side-by-side (total 4x8).
        A window from col 0 to col 6 covers cols 0-3 of left and cols
        0-1 of right (the seam sits at col 4). The src_rect coordinate
        mapping inside ``_vrt.read_vrt`` must clip each source's
        source-coords correctly; a regression to the dst-to-src
        translation would return mis-aligned columns.
        """
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = (np.arange(16, dtype=np.float32) + 100).reshape(4, 4)

        vrt = _make_2x1_mosaic_vrt(tmp_path, left, right)

        # Window rows 0..4, cols 0..6 (cuts across seam at col 4)
        result = read_vrt(vrt, window=(0, 0, 4, 6))

        assert result.shape == (4, 6)
        # cols 0-3 of window are cols 0-3 of left
        np.testing.assert_array_equal(result.values[:, :4], left[:, :4])
        # cols 4-5 of window are cols 0-1 of right (after seam)
        np.testing.assert_array_equal(result.values[:, 4:6], right[:, :2])

    def test_window_offset_into_mosaic(self, tmp_path):
        """Window starting past the seam reads only the right source."""
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = (np.arange(16, dtype=np.float32) + 100).reshape(4, 4)

        vrt = _make_2x1_mosaic_vrt(tmp_path, left, right)

        # Window cols 5..8 -> right cols 1..4
        result = read_vrt(vrt, window=(0, 5, 4, 8))

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result.values, right[:, 1:4])

    def test_window_transform_origin_shift(self, tmp_path):
        """``attrs['transform']`` reflects the window origin.

        With GeoTransform ``(origin_x=0, res=1, origin_y=0, res=-1)``
        and a window ``(r0=2, c0=3, ...)``, the output's transform
        must advertise the shifted origin ``origin_x' = origin_x +
        c0*res_x`` and ``origin_y' = origin_y + r0*res_y``. This is
        the metadata-propagation contract that ``open_geotiff
        (window=)`` already honours; ``read_vrt(window=)`` must
        agree.
        """
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 3, 6, 10))

        # GeoTransform from _vrt.write_vrt default: pixel-is-area,
        # res_x=1.0, res_y=-1.0, origin (0, 0).
        # Expected: origin shifts by (3 * 1.0, 2 * -1.0) = (3.0, -2.0)
        assert 'transform' in result.attrs
        pw, _, ox, _, ph, oy = result.attrs['transform']
        assert pw == pytest.approx(1.0)
        assert ph == pytest.approx(-1.0)
        assert ox == pytest.approx(3.0)
        assert oy == pytest.approx(-2.0)

    def test_window_coords_match_transform_shift(self, tmp_path):
        """y/x coords reflect the window's origin shift.

        Pixel-is-area convention: coord(0, 0) sits at the *center* of
        the windowed pixel (0, 0). With res_x=1.0, res_y=-1.0,
        origin (0, 0), and window starting at (r0=2, c0=3), the
        first x coord must be ``0 + (3 + 0.5) * 1.0 = 3.5`` and the
        first y coord must be ``0 + (2 + 0.5) * -1.0 = -2.5``.
        """
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 3, 6, 10))

        assert float(result.x[0]) == pytest.approx(3.5)
        assert float(result.y[0]) == pytest.approx(-2.5)


class TestReadVrtWindowWithBand:
    """``read_vrt(window=, band=)`` combinations.

    A regression in either kwarg's interaction with the other (band
    selection after window slicing, nodata sentinel resolved per
    band) would mis-mask the windowed region.
    """

    def _make_multiband_vrt(self, tmp_path) -> tuple[str, np.ndarray]:
        """Two-band VRT with distinct values per band."""
        h, w = 4, 8
        band0 = np.arange(h * w, dtype=np.float32).reshape(h, w)
        band1 = (band0 * -1.0).astype(np.float32)
        # Stack into 3D so write_vrt produces a multi-band TIFF source
        full = np.stack([band0, band1], axis=-1)

        tile_path = str(tmp_path / 'multi.tif')
        to_geotiff(_da_with_float_coords(full), tile_path, compression='none')

        vrt_path = str(tmp_path / 'multi_band.vrt')
        _write_vrt_internal(vrt_path, [tile_path])
        return vrt_path, full

    def test_window_plus_band_selection(self, tmp_path):
        vrt, full = self._make_multiband_vrt(tmp_path)

        # window rows 1..3, cols 2..6, band 1
        result = read_vrt(vrt, window=(1, 2, 3, 6), band=1)

        assert result.ndim == 2  # band selection yields 2D
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(
            result.values, full[1:3, 2:6, 1]
        )


class TestReadVrtWindowDask:
    """``read_vrt(window=, chunks=)`` returns a dask-chunked DataArray.

    The chunk size must apply to the windowed shape, not the full
    VRT extent. A regression that dropped the window before chunking
    would over-allocate the dask graph.
    """

    def test_window_chunks_returns_dask(self, tmp_path):
        import dask.array as da_mod

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.values, arr[2:6, 4:10]
        )


@_gpu_only
class TestReadVrtWindowGpu:
    """``read_vrt(window=, gpu=True)`` returns a CuPy-backed DataArray.

    The eager VRT decode happens on CPU (the internal reader walks
    SimpleSources and assembles); the final ``if gpu: cupy.asarray``
    block uploads the windowed result. Window slicing must happen
    *before* the upload so the GPU array carries only the requested
    pixels.
    """

    def test_window_gpu_returns_cupy(self, tmp_path):
        import cupy

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), gpu=True)

        assert isinstance(result.data, cupy.ndarray)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.data.get(), arr[2:6, 4:10]
        )

    def test_window_gpu_chunks_returns_dask_cupy(self, tmp_path):
        """``window + gpu + chunks`` -> Dask+CuPy with window-sized data."""
        import cupy
        import dask.array as da_mod

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert isinstance(result.data._meta, cupy.ndarray)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.compute().data.get(), arr[2:6, 4:10]
        )
