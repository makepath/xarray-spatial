"""Predictor=3 big-endian round-trip tests.

The floating-point predictor (TIFF Tech Note 3) byte-swizzles each row
into ``bytes_per_sample`` lanes, MSB-first.  The decoder un-transposes
those lanes back to the file's native byte order; the byte position of
the MSB differs between BE (index 0) and LE (index ``bps-1``).

Before the fix the un-transpose always wrote MSB at index ``bps-1``, so
big-endian predictor=3 files decoded to garbage values even though they
came back as a clean ``float32``/``float64`` array (no error, just wrong
numbers - the worst kind of bug for raster data).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._reader import read_to_array

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")  # tifffile needs it for predictor=3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_big_endian_predictor3_round_trip(tmp_path, dtype):
    """xrspatial reads a big-endian predictor=3 file exactly."""
    arr = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.5, 2.5, 3.5, 4.5],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=dtype,
    )
    path = tmp_path / f"be_pred3_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=3, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_little_endian_predictor3_still_round_trips(tmp_path):
    """Sanity: the BE fix did not break the LE path."""
    arr = np.linspace(-100.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    path = tmp_path / "le_pred3_f32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder="<", predictor=3, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_big_endian_predictor3_tiled(tmp_path):
    """BE predictor=3 with tiled layout, multiple tiles per row."""
    rng = np.random.RandomState(20260506)
    arr = rng.standard_normal((32, 48)).astype(np.float32)
    path = tmp_path / "be_pred3_tiled.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=3,
        compression="deflate", tile=(16, 16))

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_big_endian_predictor3_gpu(tmp_path):
    """The GPU decode path also handles big-endian predictor=3."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA not available")

    from xrspatial.geotiff import open_geotiff

    rng = np.random.RandomState(20260506)
    arr = rng.standard_normal((32, 48)).astype(np.float32)
    path = tmp_path / "be_pred3_gpu.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=3,
        compression="deflate", tile=(16, 16))

    cpu = open_geotiff(str(path)).values
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = open_geotiff(str(path), gpu=True)
    gpu_arr = gpu_da.data
    if hasattr(gpu_arr, "get"):
        gpu_arr = gpu_arr.get()
    else:
        gpu_arr = np.asarray(gpu_arr)
    np.testing.assert_array_equal(gpu_arr, arr)
