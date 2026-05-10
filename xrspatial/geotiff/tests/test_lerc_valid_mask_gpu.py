"""GPU follow-up to PR #1529 (LERC valid-mask on decode).

The CPU LERC reader honours the LERC valid-mask and writes the file's
nodata sentinel into masked pixels.  The GPU LERC tile-decode path used
to discard the mask, so masked pixels read back as LERC's zero fill
(real-looking measurements at z == 0) on GPU but as NaN/sentinel on
CPU.  These tests confirm the GPU path now matches the CPU path for
representative LERC mask combinations.

Mirrors the structure of ``test_lerc_valid_mask.py`` but compares
``read_geotiff_gpu`` output to ``read_to_array`` output for each case.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

lerc = pytest.importorskip("lerc")

from xrspatial.geotiff._compression import LERC_AVAILABLE  # noqa: E402


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not (_HAS_GPU and LERC_AVAILABLE),
    reason="cupy + CUDA + lerc required",
)


@pytest.fixture
def lerc_writer_with_mask(monkeypatch):
    """Patch ``lerc_compress`` to embed a valid-mask the writer can't pass.

    The xrspatial writer hard-codes ``hasMask=False`` in its call to
    ``lerc.encode``.  Tests inject a per-tile mask through this holder's
    ``invalid`` predicate so the masked pixels survive the encode and
    show up at decode time.  Same pattern as the CPU test fixture in
    ``test_lerc_valid_mask.py``.
    """
    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype('float32'), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = lerc.encode(arr, samples, has_mask, mask, max_z_error, 1)
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}")
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


def _read_cpu_gpu(path):
    """Read *path* with both readers and return ``(cpu_array, gpu_host_array)``.

    Uses the low-level ``read_to_array`` for CPU so that nodata sentinels
    stay as the literal value (this module checks LERC mask preservation,
    not the higher-level NaN promotion that ``open_geotiff`` performs).

    The GPU reader (``read_geotiff_gpu``) applies the same nodata masking
    that ``open_geotiff`` does (PR #1542), so its output uses NaN where
    the sentinel was. Callers that want a bit-for-bit comparison should
    pass ``raw_gpu=True`` to skip the high-level masking on the GPU side.
    """
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    cpu, _geo = read_to_array(path)
    gpu_da = read_geotiff_gpu(path, gpu='strict')
    gpu_host = gpu_da.data.get()
    return cpu, gpu_host


def _restore_sentinel(arr, nodata):
    """Replace NaN positions in *arr* with *nodata* so high-level GPU
    reads compare bit-exactly against low-level CPU reads (which keep
    the sentinel value verbatim)."""
    if nodata is None or arr.dtype.kind != 'f' or np.isnan(nodata):
        return arr
    out = arr.copy()
    out[np.isnan(out)] = arr.dtype.type(nodata)
    return out


@_gpu_only
class TestGpuLercValidMask:
    """End-to-end TIFF round-trips comparing GPU vs CPU output."""

    def test_float32_nan_nodata(self, tmp_path, lerc_writer_with_mask):
        """Float32 LERC + NaN nodata: GPU output matches CPU output."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        invalid_positions = {(0, 1), (5, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_nan_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=float("nan"))

        cpu, gpu = _read_cpu_gpu(path)
        # NaN positions
        for (r, c) in invalid_positions:
            assert np.isnan(cpu[r, c])
            assert np.isnan(gpu[r, c])
        # Valid positions agree exactly
        cpu_valid = np.where(np.isnan(cpu), 0.0, cpu)
        gpu_valid = np.where(np.isnan(gpu), 0.0, gpu)
        np.testing.assert_array_equal(cpu_valid, gpu_valid)

    def test_float32_sentinel_nodata(self, tmp_path, lerc_writer_with_mask):
        """Float32 LERC + sentinel nodata (-9999): GPU matches CPU."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        invalid_positions = {(0, 1), (3, 3), (7, 7)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_sentinel_f32_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=-9999.0)

        cpu, gpu = _read_cpu_gpu(path)
        # ``read_geotiff_gpu`` applies the high-level nodata mask (#1542),
        # so masked pixels come back as NaN. ``read_to_array`` keeps the
        # sentinel verbatim. Restore the sentinel on the GPU side so the
        # bit-for-bit comparison still pins LERC mask preservation.
        gpu_with_sentinel = _restore_sentinel(gpu, -9999.0)
        np.testing.assert_array_equal(cpu, gpu_with_sentinel)
        for (r, c) in invalid_positions:
            assert np.isnan(gpu[r, c])
            assert gpu_with_sentinel[r, c] == np.float32(-9999.0)

    def test_uint16_sentinel_nodata(self, tmp_path, lerc_writer_with_mask):
        """Uint16 LERC + sentinel nodata (65535): GPU matches CPU."""
        from xrspatial.geotiff._writer import write

        arr = (np.arange(1, 65, dtype=np.uint16) * 100).reshape(8, 8)
        invalid_positions = {(0, 1), (4, 4)}

        def invalid_pred(a):
            m = np.zeros(a.shape[:2], dtype=bool)
            for r, c in invalid_positions:
                m[r, c] = True
            return m
        lerc_writer_with_mask["invalid"] = invalid_pred

        path = str(tmp_path / "lerc_mask_uint16_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8,
              nodata=65535)

        cpu, gpu = _read_cpu_gpu(path)
        # ``read_geotiff_gpu`` applies the high-level nodata mask on
        # integer rasters (#1542): the array is promoted to float64 with
        # NaN where the sentinel was. ``read_to_array`` keeps uint16 with
        # the sentinel literal. Restore the sentinel + dtype on the GPU
        # side so the bit-for-bit comparison still pins LERC mask
        # preservation. Replace NaN before the uint16 cast to avoid
        # numpy's "invalid value encountered in cast" warning.
        assert gpu.dtype == np.float64
        gpu_no_nan = np.where(np.isnan(gpu), 65535.0, gpu)
        gpu_u16 = gpu_no_nan.astype(np.uint16)
        np.testing.assert_array_equal(cpu, gpu_u16)
        for (r, c) in invalid_positions:
            assert np.isnan(gpu[r, c])
            assert gpu_u16[r, c] == np.uint16(65535)

    def test_no_mask_roundtrip_bitexact(self, tmp_path):
        """All-valid LERC (no encoded mask): GPU and CPU agree bit-exact."""
        from xrspatial.geotiff._writer import write

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / "lerc_no_mask_gpu.tif")
        write(arr, path, compression="lerc", tiled=True, tile_size=8)

        cpu, gpu = _read_cpu_gpu(path)
        np.testing.assert_array_equal(cpu, arr)
        np.testing.assert_array_equal(gpu, arr)
