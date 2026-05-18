"""Regression test for issue #1582.

``to_geotiff`` originally only consulted ``attrs['nodata']`` when
resolving the NoData sentinel from a DataArray. Two ecosystem
conventions were silently dropped:

* ``attrs['nodatavals']`` -- rioxarray's per-band tuple. A user reading
  a GeoTIFF with rioxarray and passing the result straight back to
  ``to_geotiff`` ended up with a file lacking a GDAL_NODATA tag, and
  the sentinel pixels stored as ordinary data.
* ``attrs['_FillValue']`` -- the CF-style xarray pipeline convention.

Both paths now route through ``_resolve_nodata_attr`` so the eager
CPU writer (``to_geotiff``) and the GPU writer
(``write_geotiff_gpu``) honour either alias. xrspatial's own
``attrs['nodata']`` still wins when set, preserving the existing
intra-library round-trip.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


_SENTINEL = -9999.0


def _da_float(arr, **attrs):
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(arr.shape[0], dtype=np.float64),
                "x": np.arange(arr.shape[1], dtype=np.float64)},
        attrs=attrs,
    )


@pytest.fixture
def _arr_with_sentinel():
    return np.array(
        [[1.0, 2.0, _SENTINEL], [3.0, _SENTINEL, 5.0]],
        dtype=np.float32,
    )


def test_nodatavals_tuple_resolves_to_nodata_tag(tmp_path, _arr_with_sentinel):
    """rioxarray-style ``nodatavals`` tuple lands as the file's nodata."""
    da = _da_float(_arr_with_sentinel,
                   crs=4326, nodatavals=(_SENTINEL,))
    out = str(tmp_path / "nodatavals_tuple.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


def test_nodatavals_list_resolves_to_nodata_tag(tmp_path, _arr_with_sentinel):
    """List variant of nodatavals (some readers return list, not tuple)."""
    da = _da_float(_arr_with_sentinel,
                   crs=4326, nodatavals=[_SENTINEL])
    out = str(tmp_path / "nodatavals_list.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


def test_nodatavals_scalar_resolves_to_nodata_tag(tmp_path,
                                                   _arr_with_sentinel):
    """Single-band variant where the attr is a scalar, not a sequence."""
    da = _da_float(_arr_with_sentinel,
                   crs=4326, nodatavals=_SENTINEL)
    out = str(tmp_path / "nodatavals_scalar.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


def test_fill_value_resolves_to_nodata_tag(tmp_path, _arr_with_sentinel):
    """CF-style ``_FillValue`` lands as the file's nodata."""
    da = _da_float(_arr_with_sentinel,
                   crs=4326, **{"_FillValue": _SENTINEL})
    out = str(tmp_path / "fillvalue.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


def test_explicit_nodata_attr_wins_over_aliases(tmp_path, _arr_with_sentinel):
    """``attrs['nodata']`` is xrspatial's canonical key. Issue #1987 PR 7
    replaced the legacy "canonical silently wins, alias dropped" path
    with a fail-closed raise: a DataArray with disagreeing ``nodata`` and
    ``nodatavals`` refuses to write. The explicit ``nodata=`` writer
    kwarg overrides both attrs and bypasses the check."""
    from xrspatial.geotiff import ConflictingNodataError

    da = _da_float(
        _arr_with_sentinel, crs=4326,
        nodata=-8888.0,
        nodatavals=(_SENTINEL,),
        **{"_FillValue": -7777.0},
    )
    out = str(tmp_path / "explicit_wins.tif")

    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, out)

    # Explicit kwarg overrides both attrs.
    to_geotiff(da, out, nodata=-8888.0)
    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -8888.0


def test_kwarg_nodata_wins_over_attrs(tmp_path, _arr_with_sentinel):
    """The ``nodata=`` keyword overrides anything in attrs."""
    da = _da_float(_arr_with_sentinel,
                   crs=4326, nodatavals=(_SENTINEL,))
    out = str(tmp_path / "kwarg_wins.tif")
    to_geotiff(da, out, nodata=-1234.0)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -1234.0


def test_nan_nodatavals_does_not_emit_tag(tmp_path):
    """NaN in nodatavals means the float NaN is already the sentinel;
    no GDAL_NODATA tag should be written for that case (mirrors
    rioxarray behaviour, and keeps the file's metadata clean)."""
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    da = _da_float(arr, crs=4326, nodatavals=(float("nan"),))
    out = str(tmp_path / "nan_nodatavals.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") is None


def test_no_nodata_attrs_means_no_tag(tmp_path, _arr_with_sentinel):
    """Sanity guard: a DataArray with no nodata-related attrs still
    writes without a GDAL_NODATA tag."""
    da = _da_float(_arr_with_sentinel, crs=4326)
    out = str(tmp_path / "no_nodata.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") is None


# ---------------------------------------------------------------------------
# GPU writer parity -- same alias resolution must apply to
# write_geotiff_gpu / the auto-dispatch from to_geotiff.
# ---------------------------------------------------------------------------


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
@pytest.mark.parametrize("attr_key,attr_value", [
    ("nodatavals", (_SENTINEL,)),
    ("nodatavals", [_SENTINEL]),
    ("_FillValue", _SENTINEL),
])
def test_gpu_writer_resolves_alias(tmp_path, _arr_with_sentinel,
                                    attr_key, attr_value):
    """The GPU write path (write_geotiff_gpu) honours the same aliases."""
    import cupy
    from xrspatial.geotiff import write_geotiff_gpu

    da = xr.DataArray(
        cupy.asarray(_arr_with_sentinel),
        dims=["y", "x"],
        coords={"y": np.arange(2, dtype=np.float64),
                "x": np.arange(3, dtype=np.float64)},
        attrs={"crs": 4326, attr_key: attr_value},
    )
    out = str(tmp_path / f"gpu_{attr_key}.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL
