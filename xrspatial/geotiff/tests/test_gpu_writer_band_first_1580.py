"""Regression test for issue #1580.

``write_geotiff_gpu`` used to assume the CuPy-backed DataArray was
``(y, x[, band])`` and computed ``height, width = arr.shape[:2]``
unconditionally. A rioxarray-style ``(band, y, x)`` 3D DataArray --
auto-dispatched into the GPU writer from ``to_geotiff`` -- ended up
written with the band axis stored as image width and the actual width
stored as samples-per-pixel.

The CPU eager path in ``to_geotiff`` already handles this with a
``np.moveaxis(arr, 0, -1)`` when ``data.dims[0] in ('band', 'bands',
'channel')``; the GPU writer now mirrors that remap so both backends
produce the same file for the same input.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff, write_geotiff_gpu


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


@_gpu_only
@pytest.mark.parametrize("band_dim_name", ["band", "bands", "channel"])
def test_band_first_layout_written_correctly_via_write_geotiff_gpu(
        tmp_path, band_dim_name):
    """Direct call to write_geotiff_gpu with a (band, y, x) CuPy
    DataArray must produce the same file dimensions as the CPU writer
    would (height=y, width=x, samples=band).
    """
    import cupy
    rng = np.random.default_rng(seed=1580)
    arr_bhw = rng.integers(0, 255, (3, 16, 32), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=[band_dim_name, "y", "x"],
        coords={
            band_dim_name: np.arange(3),
            "y": np.arange(16, dtype=np.float64),
            "x": np.arange(32, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / f"band_first_1580_{band_dim_name}.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes["y"] == 16, (
        f"expected height=16, got {rd.sizes}; the writer is treating the "
        f"band axis as height"
    )
    assert rd.sizes["x"] == 32, f"expected width=32, got {rd.sizes}"
    assert rd.sizes["band"] == 3, (
        f"expected 3 bands, got {rd.sizes}; the writer is treating the "
        f"width as samples-per-pixel"
    )

    # Pixel values should match the source after the (band, y, x) ->
    # (y, x, band) remap.
    np.testing.assert_array_equal(rd.values, np.moveaxis(arr_bhw, 0, -1))


@_gpu_only
def test_band_first_layout_via_to_geotiff_auto_dispatch(tmp_path):
    """The user-facing path: pass a CuPy (band, y, x) DataArray to
    ``to_geotiff`` and let auto-detection pick the GPU writer.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 1)
    arr_bhw = rng.integers(0, 255, (2, 8, 12), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=["band", "y", "x"],
        coords={
            "band": np.arange(2),
            "y": np.arange(8, dtype=np.float64),
            "x": np.arange(12, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / "band_first_1580_autodispatch.tif")
    to_geotiff(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes == {"y": 8, "x": 12, "band": 2}, (
        f"auto-dispatched GPU write produced wrong dims: {rd.sizes}"
    )
    np.testing.assert_array_equal(rd.values, np.moveaxis(arr_bhw, 0, -1))


@_gpu_only
def test_yxbands_layout_unchanged(tmp_path):
    """Regression guard: the original (y, x, band) layout must still
    write correctly after the band-first remap was added.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 2)
    arr_yxb = rng.integers(0, 255, (8, 12, 2), dtype=np.uint8)
    da = xr.DataArray(
        cupy.asarray(arr_yxb),
        dims=["y", "x", "band"],
        coords={
            "y": np.arange(8, dtype=np.float64),
            "x": np.arange(12, dtype=np.float64),
            "band": np.arange(2),
        },
        attrs={"crs": 4326},
    )

    out = str(tmp_path / "yxb_1580.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.sizes == {"y": 8, "x": 12, "band": 2}
    np.testing.assert_array_equal(rd.values, arr_yxb)


@_gpu_only
def test_gpu_band_first_matches_cpu_byte_for_byte_on_pixel_values(tmp_path):
    """Cross-backend parity: GPU and CPU writers must emit the same
    pixel values for the same (band, y, x) input.
    """
    import cupy
    rng = np.random.default_rng(seed=1580 + 3)
    arr_bhw = rng.integers(0, 255, (3, 24, 40), dtype=np.uint8)
    da_cpu = xr.DataArray(
        arr_bhw,
        dims=["band", "y", "x"],
        coords={"band": np.arange(3),
                "y": np.arange(24, dtype=np.float64),
                "x": np.arange(40, dtype=np.float64)},
        attrs={"crs": 4326},
    )
    da_gpu = xr.DataArray(
        cupy.asarray(arr_bhw),
        dims=["band", "y", "x"],
        coords={"band": np.arange(3),
                "y": np.arange(24, dtype=np.float64),
                "x": np.arange(40, dtype=np.float64)},
        attrs={"crs": 4326},
    )

    cpu_path = str(tmp_path / "band_first_1580_cpu.tif")
    gpu_path = str(tmp_path / "band_first_1580_gpu.tif")
    to_geotiff(da_cpu, cpu_path, compression="none", gpu=False)
    write_geotiff_gpu(da_gpu, gpu_path, compression="none")

    cpu_rd = open_geotiff(cpu_path).values
    gpu_rd = open_geotiff(gpu_path).values
    np.testing.assert_array_equal(cpu_rd, gpu_rd)
