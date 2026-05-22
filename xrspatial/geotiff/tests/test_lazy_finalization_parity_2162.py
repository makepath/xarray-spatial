"""Lazy-read finalization parity between the two dask backends (PR C of #2162).

Wave 2 of issue #2162 migrates ``read_geotiff_dask`` (the CPU+dask
backend) and the dask branch of ``read_geotiff_gpu`` (the GPU+dask
backend) onto the shared :func:`_finalize_lazy_read_attrs` helper from
#2177. Both sites had ~25 lines of validate-then-populate-then-stamp
code that produced the same attrs surface; the helper centralises that
logic so a single bug fix lands in both backends at once.

The tests in this module pin the lazy-attrs contract across the two
backends so a future change to the helper (or to one backend's call
site) cannot drift them apart without a visible failure. Each test
opens the same fixture through ``read_geotiff_dask`` and
``read_geotiff_gpu(chunks=...)`` and compares the attrs dicts.

Pins per the issue body:

* ``attrs['nodata_pixels_present']`` is absent on both backends (the
  per-chunk reduction would force eager compute; #2135 contract).
* ``attrs['nodata_dtype_cast']`` matches when the caller forced a cast.
* ``attrs['georef_status']`` matches across the five reader states
  (full, transform_only, crs_only, none, rotated_dropped).

GPU tests skip when CUDA is unavailable using the project's standard
``cupy + CUDA`` gate.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff_dask, to_geotiff
from xrspatial.geotiff._attrs import (GEOREF_STATUS_CRS_ONLY, GEOREF_STATUS_FULL,
                                      GEOREF_STATUS_NONE, GEOREF_STATUS_ROTATED_DROPPED,
                                      GEOREF_STATUS_TRANSFORM_ONLY)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY

tifffile = pytest.importorskip("tifffile")

from xrspatial.geotiff.tests.test_allow_rotated_geotiff_2115 import \
    _write_rotated_tiff  # noqa: E402


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


def _open_cpu_dask(path, **kwargs):
    return read_geotiff_dask(path, chunks=2, **kwargs)


def _open_gpu_dask(path, **kwargs):
    # Lazy import so the module loads under CPU-only sandboxes.
    from xrspatial.geotiff import read_geotiff_gpu
    return read_geotiff_gpu(path, chunks=2, **kwargs)


_BACKENDS = [
    pytest.param(_open_cpu_dask, id="dask+numpy"),
    pytest.param(_open_gpu_dask, id="dask+cupy", marks=_gpu_only),
]


# ---------------------------------------------------------------------------
# Fixture builders, mirroring the per-state fixtures in test_georef_status_2136
# ---------------------------------------------------------------------------


def _make_full_tiff(path):
    """Float coords + CRS -> ``full``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, path)


def _make_transform_only_tiff(path):
    """Float coords, no CRS -> ``transform_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, path)


def _make_crs_only_tiff(path):
    """No-georef marker + CRS -> ``crs_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, path)


def _make_none_tiff(path):
    """Bare TIFF with no GeoTIFF tags at all -> ``none``."""
    arr = np.zeros((4, 4), dtype=np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )


def _make_rotated_tiff(path):
    """Rotated ``ModelTransformationTag`` (opened with ``allow_rotated``)
    -> ``rotated_dropped``. The data is uint16 because the rotated-TIFF
    writer in the #2115 test only emits integer pixels; that's fine for
    a metadata pin."""
    arr = np.arange(16, dtype='<u2').reshape(4, 4)
    _write_rotated_tiff(path, arr)


def _make_float_with_nodata_tiff(path, sentinel=-9999.0):
    """Float raster carrying a GDAL_NODATA tag. Used to exercise the
    nodata lifecycle attrs without forcing the int->float promotion
    branch."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr[0, 0] = sentinel
    da = xr.DataArray(
        arr,
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326, 'nodata': sentinel},
    )
    to_geotiff(da, path)


def _make_int_with_nodata_tiff(path, sentinel=30):
    """Integer raster carrying a sentinel. Lets the dtype-cast tests
    distinguish "graph dtype auto-promoted by masking" from
    "caller asked for an explicit cast"."""
    arr = np.array([[10, 20, 25], [30, 40, 50]], dtype=np.int16)
    da = xr.DataArray(
        arr,
        coords={
            'y': np.array([200.0, 199.0]),
            'x': np.array([100.0, 101.0, 102.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326, 'nodata': sentinel},
    )
    to_geotiff(da, path)


# ---------------------------------------------------------------------------
# Cross-backend parity tests
# ---------------------------------------------------------------------------


_GEOREF_FIXTURES = [
    pytest.param(_make_full_tiff, GEOREF_STATUS_FULL, False,
                 id="full"),
    pytest.param(_make_transform_only_tiff, GEOREF_STATUS_TRANSFORM_ONLY,
                 False, id="transform_only"),
    pytest.param(_make_crs_only_tiff, GEOREF_STATUS_CRS_ONLY, False,
                 id="crs_only"),
    pytest.param(_make_none_tiff, GEOREF_STATUS_NONE, False,
                 id="none"),
    pytest.param(_make_rotated_tiff, GEOREF_STATUS_ROTATED_DROPPED,
                 True, id="rotated_dropped"),
]


@pytest.mark.parametrize("fixture,expected_status,allow_rotated",
                         _GEOREF_FIXTURES)
def test_georef_status_parity(tmp_path, fixture, expected_status,
                              allow_rotated):
    """Both dask backends emit the same ``georef_status`` for each
    of the five reader states."""
    path = str(tmp_path / f"tmp_2178_status_{expected_status}.tif")
    fixture(path)

    kwargs = {'allow_rotated': True} if allow_rotated else {}
    cpu = _open_cpu_dask(path, **kwargs)
    assert cpu.attrs.get('georef_status') == expected_status

    if _HAS_GPU:
        gpu = _open_gpu_dask(path, **kwargs)
        assert gpu.attrs.get('georef_status') == expected_status
        assert cpu.attrs['georef_status'] == gpu.attrs['georef_status']


@pytest.mark.parametrize("fixture,expected_status,allow_rotated",
                         _GEOREF_FIXTURES)
def test_attrs_dict_parity(tmp_path, fixture, expected_status,
                           allow_rotated):
    """Both dask backends emit the same attrs dict for each fixture."""
    if not _HAS_GPU:
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / f"tmp_2178_parity_{expected_status}.tif")
    fixture(path)

    kwargs = {'allow_rotated': True} if allow_rotated else {}
    cpu = _open_cpu_dask(path, **kwargs)
    gpu = _open_gpu_dask(path, **kwargs)

    cpu_attrs = dict(cpu.attrs)
    gpu_attrs = dict(gpu.attrs)
    assert cpu_attrs == gpu_attrs, (
        f"attrs dicts diverged for fixture={expected_status}:\n"
        f"  cpu only: {set(cpu_attrs) - set(gpu_attrs)}\n"
        f"  gpu only: {set(gpu_attrs) - set(cpu_attrs)}\n"
        f"  shared keys with different values: "
        f"{[k for k in set(cpu_attrs) & set(gpu_attrs) if cpu_attrs[k] != gpu_attrs[k]]}"
    )


@pytest.mark.parametrize("opener", _BACKENDS)
def test_nodata_pixels_present_absent_on_lazy(tmp_path, opener):
    """Lazy contract from #2135: ``nodata_pixels_present`` stays unset
    on both dask backends."""
    path = str(tmp_path / "tmp_2178_pixels_absent.tif")
    _make_float_with_nodata_tiff(path)
    out = opener(path)
    assert 'nodata_pixels_present' not in out.attrs


def test_nodata_pixels_present_cross_backend(tmp_path):
    """Both backends agree on the absence of ``nodata_pixels_present``
    when reading the same fixture."""
    if not _HAS_GPU:
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_pixels_cross.tif")
    _make_float_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path)
    gpu = _open_gpu_dask(path)
    assert 'nodata_pixels_present' not in cpu.attrs
    assert 'nodata_pixels_present' not in gpu.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_absent_without_caller_dtype(tmp_path, opener):
    """No ``dtype=`` kwarg: ``nodata_dtype_cast`` stays unset, even
    when masking auto-promotes the graph dtype to float64."""
    path = str(tmp_path / "tmp_2178_no_cast.tif")
    _make_int_with_nodata_tiff(path)
    out = opener(path)
    # Masking promoted the int source to float64 on the graph dtype,
    # but the caller did not ask for a cast.
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert 'nodata_dtype_cast' not in out.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_records_target(tmp_path, opener):
    """Explicit ``dtype=`` kwarg: ``nodata_dtype_cast`` records the
    requested dtype on both backends."""
    path = str(tmp_path / "tmp_2178_with_cast.tif")
    _make_int_with_nodata_tiff(path)
    out = opener(path, mask_nodata=False, dtype=np.float64)
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert 'nodata_pixels_present' not in out.attrs


def test_dtype_cast_parity_cross_backend(tmp_path):
    """Cross-backend: same input + same ``dtype=`` kwarg yields the
    same ``nodata_dtype_cast`` value."""
    if not _HAS_GPU:
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_cast_cross.tif")
    _make_int_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path, mask_nodata=False, dtype=np.float64)
    gpu = _open_gpu_dask(path, mask_nodata=False, dtype=np.float64)
    assert cpu.attrs.get('nodata_dtype_cast') == gpu.attrs.get('nodata_dtype_cast')
    assert cpu.attrs.get('nodata_dtype_cast') == 'float64'


def test_dtype_cast_absent_parity_cross_backend(tmp_path):
    """Cross-backend: same int input without an explicit ``dtype=``
    leaves ``nodata_dtype_cast`` absent on both backends (the auto-
    promoted graph dtype must not leak as a caller cast)."""
    if not _HAS_GPU:
        pytest.skip("dask+cupy parity requires CUDA")
    path = str(tmp_path / "tmp_2178_no_cast_cross.tif")
    _make_int_with_nodata_tiff(path)
    cpu = _open_cpu_dask(path)
    gpu = _open_gpu_dask(path)
    assert 'nodata_dtype_cast' not in cpu.attrs
    assert 'nodata_dtype_cast' not in gpu.attrs


@pytest.mark.parametrize("opener", _BACKENDS)
def test_dtype_cast_records_integer_target(tmp_path, opener):
    """Caller-supplied integer ``dtype=`` kwarg: ``nodata_dtype_cast``
    records the integer dtype on both backends. Pins the
    ``dtype.kind != 'f'`` branch of the call-site fixup (review
    follow-up for #2178)."""
    path = str(tmp_path / "tmp_2178_int_cast.tif")
    _make_int_with_nodata_tiff(path)
    # ``mask_nodata=False`` keeps the integer dtype; the caller cast
    # then routes the graph dtype to ``int32`` without the masking
    # auto-promotion firing. The pre-helper contract emits
    # ``nodata_dtype_cast='int32'`` and ``masked_nodata=False`` here.
    out = opener(path, mask_nodata=False, dtype=np.int32)
    assert out.dtype == np.int32
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'int32'
    assert 'nodata_pixels_present' not in out.attrs
