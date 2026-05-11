"""Regression tests for issue #1599.

``write_geotiff_gpu`` (and ``to_geotiff(..., gpu=True)``) used to emit
raw NaN bytes for missing pixels even when ``nodata=<finite>`` was
supplied, while the CPU writer substituted NaN with the sentinel value
before encoding. The mismatch was invisible on xrspatial-only round
trips (the reader masks both NaN and the sentinel) but external
readers that mask only on the GDAL_NODATA value -- rasterio, GDAL,
QGIS -- treated the NaN pixels in GPU-written files as valid data.

The GPU writer now mirrors the CPU writer's NaN-to-sentinel rewrite so
both backends produce byte-equivalent files for the same input.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff, write_geotiff_gpu
from xrspatial.geotiff._reader import read_to_array


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


def _float_raster_with_nans(dtype=np.float32):
    h, w = 50, 60
    arr = np.arange(h * w, dtype=dtype).reshape(h, w)
    # Two disconnected NaN patches so single-pixel guards don't accidentally
    # mask the bug.
    arr[10:15, 20:25] = np.nan
    arr[40, 50] = np.nan
    y = np.arange(h, dtype=np.float64) * -0.1 + 50.0
    x = np.arange(w, dtype=np.float64) * 0.1 - 100.0
    return arr, y, x


@_gpu_only
def test_gpu_writer_substitutes_nan_with_sentinel(tmp_path):
    """GPU-written file stores the sentinel where the input held NaN."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_nan_sentinel.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    # The raw decoded bytes should contain the sentinel, not NaN.
    assert not np.isnan(raw).any(), \
        "GPU writer left NaN in file bytes despite nodata=-9999"
    nan_locations = np.isnan(arr)
    assert (raw[nan_locations] == -9999.0).all()
    # And the valid pixels round-trip untouched.
    assert np.array_equal(raw[~nan_locations], arr[~nan_locations])


@_gpu_only
def test_gpu_and_cpu_writers_byte_equivalent_on_nan_input(tmp_path):
    """For a float NaN input + finite sentinel, GPU and CPU writers
    must produce identical pixel data."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da_cpu = xr.DataArray(arr.copy(), dims=['y', 'x'],
                          coords={'y': y, 'x': x})
    da_gpu = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                          coords={'y': y, 'x': x})

    p_cpu = str(tmp_path / "cpu.tif")
    p_gpu = str(tmp_path / "gpu.tif")
    to_geotiff(da_cpu, p_cpu, crs=4326, nodata=-9999)
    write_geotiff_gpu(da_gpu, p_gpu, crs=4326, nodata=-9999)

    raw_cpu, _ = read_to_array(p_cpu)
    raw_gpu, _ = read_to_array(p_gpu)
    assert np.array_equal(raw_cpu, raw_gpu)


@_gpu_only
def test_gpu_writer_preserves_caller_cupy_buffer(tmp_path):
    """The NaN-to-sentinel rewrite must NOT mutate the caller's CuPy
    array. The CPU writer takes a defensive copy at the matching step;
    the GPU writer mirrors that behaviour."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    cp_arr = cp.asarray(arr.copy())
    da = xr.DataArray(cp_arr, dims=['y', 'x'], coords={'y': y, 'x': x})

    p = str(tmp_path / "gpu_preserve.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    after = cp.asnumpy(cp_arr)
    # NaNs must still be present at the original locations.
    assert np.isnan(after[10:15, 20:25]).all()
    assert np.isnan(after[40, 50])
    # And non-NaN cells must be unchanged.
    np.testing.assert_array_equal(
        after[~np.isnan(arr)], arr[~np.isnan(arr)])


@_gpu_only
def test_gpu_writer_no_rewrite_when_no_nans(tmp_path):
    """A NaN-free input must round-trip bit-exact with the GPU writer
    regardless of the nodata kwarg."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    # Replace NaNs with finite values so the array has no NaN at all.
    arr = np.where(np.isnan(arr), 1.5, arr).astype(np.float32)
    assert not np.isnan(arr).any()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_no_nans.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    assert np.array_equal(raw, arr)


@_gpu_only
def test_gpu_writer_nan_nodata_skips_substitution(tmp_path):
    """If the requested sentinel is NaN itself, no rewrite is needed.
    The file keeps the NaN bytes verbatim (CPU and GPU agree)."""
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                      coords={'y': y, 'x': x})
    p = str(tmp_path / "gpu_nan_sentinel_nan.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=float('nan'))

    raw, _ = read_to_array(p)
    # NaN pixels remain NaN; finite pixels remain finite.
    nan_locations = np.isnan(arr)
    assert np.isnan(raw[nan_locations]).all()
    np.testing.assert_array_equal(raw[~nan_locations], arr[~nan_locations])


@_gpu_only
def test_gpu_writer_external_reader_sees_correct_nodata_mask(tmp_path):
    """rasterio (and any other GDAL_NODATA-strict reader) must see the
    same valid-pixel set on CPU and GPU outputs. This is the bug from
    #1599: the GPU file used to report 100% valid pixels because the
    sentinel was never written into NaN positions."""
    rasterio = pytest.importorskip("rasterio")
    import cupy as cp

    arr, y, x = _float_raster_with_nans()
    n_nans = int(np.isnan(arr).sum())
    da_cpu = xr.DataArray(arr.copy(), dims=['y', 'x'],
                          coords={'y': y, 'x': x})
    da_gpu = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x'],
                          coords={'y': y, 'x': x})

    p_cpu = str(tmp_path / "cpu.tif")
    p_gpu = str(tmp_path / "gpu.tif")
    to_geotiff(da_cpu, p_cpu, crs=4326, nodata=-9999)
    write_geotiff_gpu(da_gpu, p_gpu, crs=4326, nodata=-9999)

    with rasterio.open(p_cpu) as src:
        cpu_masked = src.read(1, masked=True)
    with rasterio.open(p_gpu) as src:
        gpu_masked = src.read(1, masked=True)

    expected_invalid = n_nans
    assert cpu_masked.size - cpu_masked.count() == expected_invalid
    assert gpu_masked.size - gpu_masked.count() == expected_invalid


@_gpu_only
def test_gpu_writer_multiband_nan_substitution(tmp_path):
    """The substitution must work for 3D (y, x, band) inputs as well."""
    import cupy as cp

    h, w, b = 30, 40, 3
    arr = np.arange(h * w * b, dtype=np.float32).reshape(h, w, b)
    arr[5:10, 8:12, 1] = np.nan
    arr[20, 30, 0] = np.nan
    y = np.arange(h, dtype=np.float64) * -0.1 + 50.0
    x = np.arange(w, dtype=np.float64) * 0.1 - 100.0
    da = xr.DataArray(cp.asarray(arr.copy()), dims=['y', 'x', 'band'],
                      coords={'y': y, 'x': x, 'band': np.arange(b)})

    p = str(tmp_path / "gpu_mb.tif")
    write_geotiff_gpu(da, p, crs=4326, nodata=-9999)

    raw, _ = read_to_array(p)
    nan_locations = np.isnan(arr)
    assert not np.isnan(raw).any()
    assert (raw[nan_locations] == -9999.0).all()
    np.testing.assert_array_equal(
        raw[~nan_locations], arr[~nan_locations])
