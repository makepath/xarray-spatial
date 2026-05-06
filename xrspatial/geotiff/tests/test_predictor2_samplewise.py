"""Predictor=2 sample-wise round-trip tests against libtiff/GDAL.

The historical implementation differenced the byte stream at a sample
stride (so it lost carries between bytes inside a single multi-byte
sample).  Files xrspatial wrote could be read back because both halves
were wrong in matching ways, but reading multi-byte integer files
written by libtiff/GDAL produced silently-wrong pixel values whenever a
difference between adjacent samples crossed a byte boundary.

These tests use rasterio (libtiff) and tifffile/imagecodecs as reference
encoders and check that xrspatial reads them back exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._reader import read_to_array

rasterio = pytest.importorskip("rasterio")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")  # required by tifffile for predictor


def _carry_array(dtype):
    """Pixel grid that crosses a byte boundary between adjacent samples."""
    return np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [256, 257, 258, 259],
            [1000, 2000, 3000, 4000],
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", ["uint16", "int16", "uint32", "int32"])
def test_libtiff_predictor2_le_round_trip(tmp_path, dtype):
    """xrspatial reads libtiff-encoded LE predictor=2 files exactly."""
    arr = _carry_array(dtype)
    path = tmp_path / f"libtiff_pred2_le_{dtype}.tif"
    transform = rasterio.transform.from_origin(0, 4, 1, 1)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=dtype,
        compress="deflate",
        predictor=2,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


@pytest.mark.parametrize("dtype", ["uint16", "int16", "uint32", "int32"])
def test_tifffile_predictor2_be_round_trip(tmp_path, dtype):
    """xrspatial reads big-endian predictor=2 files exactly."""
    arr = _carry_array(dtype)
    path = tmp_path / f"tifffile_pred2_be_{dtype}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_libtiff_multiband_predictor2_chunky(tmp_path):
    """Chunky multi-band uint16 predictor=2 (libtiff per-channel diffs)."""
    arr = np.array(
        [
            [[1, 100], [2, 200]],
            [[3, 300], [4, 400]],
        ],
        dtype="uint16",
    )
    arr_band_first = np.moveaxis(arr, -1, 0)  # rasterio expects (band, y, x)

    path = tmp_path / "libtiff_pred2_multiband.tif"
    transform = rasterio.transform.from_origin(0, 2, 1, 1)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="uint16",
        compress="deflate",
        predictor=2,
        interleave="pixel",
        transform=transform,
    ) as dst:
        dst.write(arr_band_first)

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_xrspatial_round_trip_uint16_predictor2(tmp_path):
    """xrspatial-written predictor=2 uint16 still round-trips."""
    from xrspatial.geotiff import open_geotiff, to_geotiff
    import xarray as xr

    arr = _carry_array("uint16")
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "x": np.arange(4, dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
        },
    )
    path = tmp_path / "self_pred2_u16.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=2)

    out = open_geotiff(str(path))
    np.testing.assert_array_equal(out.values, arr)


def test_xrspatial_uint16_predictor2_readable_by_rasterio(tmp_path):
    """xrspatial-written predictor=2 uint16 is readable by libtiff."""
    from xrspatial.geotiff import to_geotiff
    import xarray as xr

    arr = _carry_array("uint16")
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "x": np.arange(4, dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
        },
    )
    path = tmp_path / "interop_pred2_u16.tif"
    to_geotiff(da, str(path), compression="deflate", predictor=2)

    with rasterio.open(str(path)) as src:
        rio_arr = src.read(1)
    np.testing.assert_array_equal(rio_arr, arr)


def _gpu_to_numpy(da):
    """Materialize a possibly cupy-backed DataArray as a numpy ndarray."""
    arr = da.data
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


@pytest.mark.parametrize("dtype", ["uint16", "int32"])
def test_gpu_predictor2_multibyte_matches_libtiff(tmp_path, dtype):
    """The GPU path also decodes libtiff predictor=2 multi-byte files."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA not available")

    from xrspatial.geotiff import open_geotiff

    arr = _carry_array(dtype)
    path = tmp_path / f"gpu_pred2_{dtype}.tif"
    transform = rasterio.transform.from_origin(0, 4, 1, 1)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=dtype,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=16,
        blockysize=16,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)

    cpu_arr = open_geotiff(str(path)).values
    np.testing.assert_array_equal(cpu_arr, arr)

    gpu_arr = _gpu_to_numpy(open_geotiff(str(path), gpu=True))
    np.testing.assert_array_equal(gpu_arr, arr)
