"""Default-rejection tests for non-finite / fractional integer nodata (#2441).

Companion to the #1774 opt-in no-op coverage (folded into
``read/test_nodata.py`` by cluster 10 of epic #2424). These tests pin
the release-contract upgrade: integer sources whose ``GDAL_NODATA`` tag
is non-finite or fractional must raise ``InvalidIntegerNodataError`` at
the read boundary unless the caller explicitly opts back into the legacy
silent no-op via ``allow_invalid_nodata=True``.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, InvalidIntegerNodataError,
                               open_geotiff, read_geotiff_dask)

from .read.test_nodata import _build_uint16_tiff_1774 as _build_uint16_tiff


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


# ----------------------------------------------------------------------
# Default behaviour: reject non-finite int sentinels at the read boundary
# ----------------------------------------------------------------------


@pytest.mark.parametrize('nodata_str', ['nan', 'NaN', 'NAN',
                                        'inf', '-inf', 'Inf', '-Inf'])
def test_open_geotiff_eager_int_nodata_nonfinite_rejected_by_default(
    tmp_path, nodata_str,
):
    """Eager numpy path raises ``InvalidIntegerNodataError`` for non-finite
    ``GDAL_NODATA`` on integer sources.
    """
    path = _build_uint16_tiff(nodata_str, tmp_path)
    with pytest.raises(InvalidIntegerNodataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert 'nodata' in msg.lower()
    # Message names the offending sentinel kind and dtype so the user
    # can locate the bad source.
    assert 'non-finite' in msg
    assert 'uint16' in msg
    # The opt-in flag name appears in the message so the caller can
    # discover the escape hatch from the rejection itself.
    assert 'allow_invalid_nodata' in msg


@pytest.mark.parametrize('nodata_str', ['3.5', '29.5', '30.5', '0.25'])
def test_open_geotiff_eager_int_nodata_fractional_rejected_by_default(
    tmp_path, nodata_str,
):
    """Eager numpy path raises ``InvalidIntegerNodataError`` for fractional
    ``GDAL_NODATA`` on integer sources.
    """
    path = _build_uint16_tiff(nodata_str, tmp_path)
    with pytest.raises(InvalidIntegerNodataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert 'nodata' in msg.lower()
    assert 'fractional' in msg
    assert 'uint16' in msg
    assert 'allow_invalid_nodata' in msg


def test_invalid_int_nodata_error_is_geotiff_ambiguous_metadata_error():
    """The new error subclasses ``GeoTIFFAmbiguousMetadataError`` so
    existing ``except GeoTIFFAmbiguousMetadataError`` callers catch it.
    """
    assert issubclass(InvalidIntegerNodataError,
                      GeoTIFFAmbiguousMetadataError)


def test_read_geotiff_dask_int_nodata_nan_rejected_by_default(tmp_path):
    """Dask path raises at graph-build time, before any chunk task fires."""
    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_dask(path, chunks=2)


def test_read_geotiff_dask_int_nodata_fractional_rejected_by_default(
    tmp_path,
):
    """Dask path raises at graph-build time for fractional int sentinels."""
    path = _build_uint16_tiff('30.5', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_dask(path, chunks=2)


# ----------------------------------------------------------------------
# Float sources are unaffected
# ----------------------------------------------------------------------


def test_open_geotiff_float_dtype_nan_nodata_still_allowed(tmp_path):
    """Float-dtype sources with NaN ``GDAL_NODATA`` are the normal case
    and must not raise. NaN matches NaN, masking proceeds.
    """
    import xarray as xr

    from xrspatial.geotiff import to_geotiff

    arr = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    da = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': [0.5, -0.5], 'x': [0.5, 1.5]},
        attrs={'crs': 4326},
    )
    path = str(tmp_path / 'float_nan_nodata_2441.tif')
    to_geotiff(da, path, nodata=float('nan'), compression='none', tiled=False)
    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    assert np.isnan(out.attrs['nodata'])


# ----------------------------------------------------------------------
# Finite, in-range integer sentinels are unaffected
# ----------------------------------------------------------------------


def test_open_geotiff_int_finite_nodata_unaffected(tmp_path):
    """Finite integer-valued sentinels still mask as before; the new
    validator must only reject non-finite / fractional sentinels.
    """
    path = _build_uint16_tiff('30', tmp_path)
    da = open_geotiff(path)
    # 30 matches a real pixel; the sentinel-to-NaN promotion fires.
    assert da.dtype == np.float64
    assert np.isnan(da.values[1, 0])
    assert da.attrs['nodata'] == 30


# ----------------------------------------------------------------------
# Opt-in restores the legacy no-op
# ----------------------------------------------------------------------


@pytest.mark.parametrize('nodata_str', ['nan', 'inf', '3.5'])
def test_open_geotiff_opt_in_restores_noop_eager(tmp_path, nodata_str):
    """``allow_invalid_nodata=True`` keeps the pre-2441 no-op behaviour."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


@pytest.mark.parametrize('nodata_str', ['nan', '30.5'])
def test_read_geotiff_dask_opt_in_restores_noop(tmp_path, nodata_str):
    """``allow_invalid_nodata=True`` keeps the pre-2441 no-op for dask."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = read_geotiff_dask(path, chunks=2, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])


# ----------------------------------------------------------------------
# GPU path mirrors the CPU contract
# ----------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_int_nodata_nan_rejected_by_default(tmp_path):
    """GPU read entry point raises before kicking off the device decode."""
    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_gpu(path)


@_gpu_only
def test_read_geotiff_gpu_int_nodata_opt_in_restores_noop(tmp_path):
    """GPU opt-in keeps the no-op (sentinel cannot match any uint16 pixel)."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    da = read_geotiff_gpu(path, allow_invalid_nodata=True)
    # Buffer stays uint16 on the device.
    assert da.dtype == cupy.uint16
    arr = da.data.get()
    np.testing.assert_array_equal(arr, [[10, 20], [30, 40]])


@_gpu_only
def test_read_geotiff_gpu_chunked_int_nodata_rejected_by_default(tmp_path):
    """dask+cupy backend rejects at metadata parse, before any chunk task
    is scheduled. Closes the four-backend matrix explicitly.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_gpu(path, chunks=2)
