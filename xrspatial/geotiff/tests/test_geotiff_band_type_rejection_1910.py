"""Regression tests for issue #1910.

The non-VRT read paths reject ``bool`` / ``np.bool_`` (#1786) but they
do not reject non-integer numeric types like ``float`` or strings. A
caller passing ``band=0.0`` slips past the type guard, the range check
evaluates ``0 <= 0.0 < n_bands`` as True, and the read either silently
succeeds on a single-band file or fails with a raw numpy ``IndexError``
on multi-band files. The VRT paths in ``_vrt.py`` and
``_backends/vrt.py`` already use the stricter
``isinstance(band, (int, np.integer))`` form, so the contract differed
across backends.

These tests pin each non-VRT read entry point -- ``read_to_array``,
``open_geotiff``, ``read_geotiff_dask``, ``read_geotiff_gpu`` (when
cupy is available) -- to raise ``TypeError`` for non-int ``band``
values. They also confirm the existing bool rejection from #1786 still
fires (and still raises ``ValueError`` for back-compat).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


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


@pytest.fixture
def multiband_tiff_path_1910(tmp_path):
    """4x6 three-band tiled tiff for the band-type rejection tests."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'tmp_1910_band_type.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


# ---------------------------------------------------------------------------
# read_to_array (local eager path)
# ---------------------------------------------------------------------------


def test_read_to_array_band_float_rejected(multiband_tiff_path_1910):
    """``band=0.0`` no longer silently reads band 0."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_to_array(path, band=0.0)


def test_read_to_array_band_np_float_rejected(multiband_tiff_path_1910):
    """``band=np.float32(0)`` is rejected as well."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_to_array(path, band=np.float32(0))


def test_read_to_array_band_str_rejected(multiband_tiff_path_1910):
    """Strings are rejected too."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_to_array(path, band="0")


def test_read_to_array_band_int_still_works(multiband_tiff_path_1910):
    """``band=1`` is a plain int and still selects band 1."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path_1910
    out, _ = read_to_array(path, band=1)
    np.testing.assert_array_equal(out, arr[:, :, 1])


def test_read_to_array_band_np_integer_still_works(multiband_tiff_path_1910):
    """``np.int64(1)`` is accepted because it is an ``np.integer``."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path_1910
    out, _ = read_to_array(path, band=np.int64(1))
    np.testing.assert_array_equal(out, arr[:, :, 1])


def test_read_to_array_band_bool_still_rejected(multiband_tiff_path_1910):
    """The #1786 bool guard fires first and keeps the ValueError."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path_1910
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_to_array(path, band=True)


# ---------------------------------------------------------------------------
# open_geotiff (public dispatcher)
# ---------------------------------------------------------------------------


def test_open_geotiff_band_float_rejected(multiband_tiff_path_1910):
    """``open_geotiff(..., band=0.0)`` raises ``TypeError``."""
    from xrspatial.geotiff import open_geotiff

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        open_geotiff(path, band=0.0)


def test_open_geotiff_band_str_rejected(multiband_tiff_path_1910):
    """``open_geotiff(..., band='0')`` raises ``TypeError``."""
    from xrspatial.geotiff import open_geotiff

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        open_geotiff(path, band="0")


# ---------------------------------------------------------------------------
# read_geotiff_dask (dask CPU path)
# ---------------------------------------------------------------------------


def test_read_geotiff_dask_band_float_rejected(multiband_tiff_path_1910):
    """``read_geotiff_dask(..., band=0.0)`` is rejected before scheduling."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_geotiff_dask(path, chunks=4, band=0.0)


def test_read_geotiff_dask_band_str_rejected(multiband_tiff_path_1910):
    """``read_geotiff_dask(..., band='0')`` raises ``TypeError``."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_geotiff_dask(path, chunks=4, band="0")


def test_read_geotiff_dask_band_int_still_works(multiband_tiff_path_1910):
    """``band=1`` still routes through and reads band 1."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = multiband_tiff_path_1910
    out = read_geotiff_dask(path, chunks=4, band=1)
    np.testing.assert_array_equal(out.values, arr[:, :, 1])


# ---------------------------------------------------------------------------
# read_geotiff_gpu (GPU path)
# ---------------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_band_float_rejected(multiband_tiff_path_1910):
    """``read_geotiff_gpu(..., band=0.0)`` raises ``TypeError``."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_geotiff_gpu(path, band=0.0)


@_gpu_only
def test_read_geotiff_gpu_band_str_rejected(multiband_tiff_path_1910):
    """``read_geotiff_gpu(..., band='0')`` raises ``TypeError``."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = multiband_tiff_path_1910
    with pytest.raises(TypeError, match="band must be a non-negative int"):
        read_geotiff_gpu(path, band="0")
