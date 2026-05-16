"""Refuse non-numeric ``nodata=`` / ``attrs['_FillValue']`` (#1973).

The writer compares the resolved nodata against pixel values via
``np.isnan`` and casts it to the array dtype. A non-numeric value
used to fall through ``_resolve_nodata_attr`` (returned verbatim) or
the ``nodata=`` kwarg path and then crash inside NumPy with
``ufunc 'isnan' not supported``. Both the entry point and the attr
resolution path now refuse non-numeric values up front with a clear
error.
"""
from __future__ import annotations

import importlib.util
import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._attrs import _resolve_nodata_attr
from xrspatial.geotiff._validation import _validate_nodata_arg


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


def _nan_square():
    return xr.DataArray(
        np.full((4, 4), np.nan, dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("bad", ['missing', object(), [1, 2]])
def test_validate_nodata_arg_rejects_non_numeric(bad):
    with pytest.raises(ValueError, match="nodata must be numeric"):
        _validate_nodata_arg(bad)


@pytest.mark.parametrize("ok", [None, 0, -9999, 1.5, np.float32(-1), np.int64(0)])
def test_validate_nodata_arg_accepts_numeric_and_none(ok):
    _validate_nodata_arg(ok)


def test_resolve_nodata_attr_rejects_non_numeric_fillvalue():
    with pytest.raises(ValueError, match="_FillValue"):
        _resolve_nodata_attr({'_FillValue': 'missing'})


def test_resolve_nodata_attr_rejects_non_numeric_nodata_attr():
    with pytest.raises(ValueError, match=r"attrs\['nodata'\]"):
        _resolve_nodata_attr({'nodata': 'missing'})


def test_resolve_nodata_attr_skips_non_numeric_in_nodatavals():
    # nodatavals (rioxarray's per-band tuple) keeps its skip-on-non-numeric
    # behaviour: those values often come from arbitrary upstream pipelines
    # and a single bad entry should not block writing.
    assert _resolve_nodata_attr({'nodatavals': ('NaN ', -9999.0)}) == -9999.0


def test_resolve_nodata_attr_still_accepts_numeric_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': -9999}) == -9999


def test_resolve_nodata_attr_returns_none_for_nan_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': float('nan')}) is None


def test_to_geotiff_rejects_non_numeric_nodata_kwarg():
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), buf, nodata='missing')


def test_to_geotiff_rejects_non_numeric_fillvalue_attr():
    da = _nan_square()
    da.attrs['_FillValue'] = 'missing'
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="_FillValue"):
        to_geotiff(da, buf)


def test_to_geotiff_vrt_path_rejects_non_numeric_nodata(tmp_path):
    vrt_path = str(tmp_path / "tmp_1973_vrt.vrt")
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), vrt_path, nodata='missing')


def test_to_geotiff_accepts_numeric_nodata_kwarg():
    buf = io.BytesIO()
    to_geotiff(_nan_square(), buf, nodata=-9999)
    assert buf.getbuffer().nbytes > 0


# ---------------------------------------------------------------------------
# Bool rejection: ``nodata=True`` / ``nodata=False`` must raise TypeError at
# all three writer entry points (eager, GPU, VRT). The eager path already
# rejected bools for #1911 but the GPU/VRT validators previously routed bool
# through ``float(True) == 1.0`` and silently coerced. The shared validator
# now refuses bools so all three paths behave the same.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, False])
def test_validate_nodata_arg_rejects_bool(bad):
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _validate_nodata_arg(bad)


def test_validate_nodata_arg_rejects_numpy_bool():
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _validate_nodata_arg(np.bool_(True))


def test_to_geotiff_eager_rejects_bool_nodata():
    buf = io.BytesIO()
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), buf, nodata=True)


def test_to_geotiff_vrt_rejects_bool_nodata(tmp_path):
    vrt_path = str(tmp_path / "tmp_1973_bool_vrt.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), vrt_path, nodata=True)


@_gpu_only
def test_write_geotiff_gpu_rejects_bool_nodata(tmp_path):
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu

    da_cpu = _nan_square()
    da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
    out = str(tmp_path / "tmp_1973_bool_gpu.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_geotiff_gpu(da_gpu, out, nodata=True)


# ---------------------------------------------------------------------------
# All-non-numeric ``attrs['nodatavals']``: warn but still return None and
# fall through. A tuple where every entry is non-numeric is almost certainly
# a user error rather than a legitimate "no sentinel" signal.
# ---------------------------------------------------------------------------


def test_resolve_nodata_attr_warns_when_nodatavals_all_non_numeric():
    with pytest.warns(UserWarning, match="nodatavals"):
        result = _resolve_nodata_attr({'nodatavals': ('foo', 'bar')})
    assert result is None


def test_resolve_nodata_attr_no_warning_when_nodatavals_has_usable_entry():
    # First entry is non-numeric, second is a real sentinel. The loop
    # returns -9999.0 before reaching the warn site, so no warning fires.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert _resolve_nodata_attr({'nodatavals': ('foo', -9999.0)}) == -9999.0


def test_resolve_nodata_attr_no_warning_when_nodatavals_all_nan():
    # NaN entries are skipped (they signal "the float NaN is the sentinel",
    # which doesn't need a GDAL_NODATA tag) but they ARE numeric, so the
    # all-non-numeric warning must not fire for an all-NaN tuple.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert _resolve_nodata_attr({'nodatavals': (float('nan'),)}) is None
