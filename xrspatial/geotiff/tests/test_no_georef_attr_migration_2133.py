"""Finish migrating the no-georef signal from coord dtype to ``attrs`` (#2133).

#2120 introduced ``attrs[_NO_GEOREF_KEY] = True`` as the read-side
stamp for files without GeoTIFF transform tags, and switched the writer's
no-georef detection to consult that marker rather than coord shape. One
validator (``_check_write_non_uniform_coords``) still inferred no-georef
from coord dtype: it skipped its uniformity check for any integer-dtype
coord array, on the assumption that integer coords were the reader's
0..N-1 placeholder. After #2087 / #2120 that assumption is wrong --- a
user-authored int-coord grid (the exact case #2087 fixed) bypassed
uniformity validation and silently wrote a misrepresented transform.

These tests pin the rest of the migration:

1. A user-authored int-coord grid with non-uniform spacing now trips
   ``NonUniformCoordsError`` rather than silently writing a transform
   derived from the first two values.
2. The no-georef marker is present on read across every backend
   (CPU eager, CPU dask, GPU eager, GPU dask) so downstream code can
   trust a single attr lookup regardless of how the file was opened.
3. The marker survives ``da.copy()`` and ``da.assign_attrs(...)``
   --- xarray's default propagation occasionally drops
   underscore-prefixed keys, so this is worth pinning explicitly.
4. A 3D (band, y, x) no-georef round-trip preserves the marker.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    NonUniformCoordsError,
    open_geotiff,
    to_geotiff,
)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY, _has_no_georef_marker
from xrspatial.geotiff._writer import write


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
def no_georef_path_2133(tmp_path):
    """4x4 float32 TIFF with no GeoTIFF tags."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = str(tmp_path / "no_georef_2133.tif")
    write(arr, path, compression='none', tiled=False)
    return path


# --- Acceptance criterion 1: validator gates on the marker, not dtype ----


def test_non_uniform_int_coords_without_marker_raise(tmp_path):
    """User-authored int coords with non-uniform spacing must not get a free pass.

    Pre-#2133, ``_check_write_non_uniform_coords`` exempted any integer
    dtype on the assumption that integer coords were the reader's
    placeholder. A non-uniform int-coord grid would slip past the
    validator and either reach the lower-level uniform-spacing check
    in ``coords_to_transform`` (raising a different error class) or,
    in some windowed paths, silently write a misrepresented transform.
    Post-#2133 the validator raises ``NonUniformCoordsError`` directly
    because the marker is absent.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_non_uniform_int.tif")
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, path)


def test_non_uniform_int_coords_with_marker_pass(tmp_path):
    """Marker-stamped arrays skip uniformity entirely (they are placeholders).

    The marker contract: a DataArray carrying ``attrs[_NO_GEOREF_KEY]
    is True`` is treated as no-georef, the writer does not synthesise a
    transform, and uniformity of the placeholder coords is irrelevant.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10, 11, 12], dtype=np.int64),
            'x': np.array([1, 2, 5], dtype=np.int64),  # non-uniform, but marked
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2133_non_uniform_int_marked.tif")
    # Must not raise.
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is None
    assert out.attrs.get(_NO_GEOREF_KEY) is True


def test_non_uniform_float_coords_still_raise(tmp_path):
    """Pre-existing float-coord behaviour is unchanged."""
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([10.0, 11.0, 12.0], dtype=np.float64),
            'x': np.array([1.0, 2.0, 5.0], dtype=np.float64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_non_uniform_float.tif")
    with pytest.raises(NonUniformCoordsError):
        to_geotiff(da, path)


def test_uniform_int_coords_still_write(tmp_path):
    """A uniform int-coord grid without the marker writes a real transform.

    Regression guard: making the validator stricter must not break the
    common case of user-authored int-coord projected grids
    (the #2087 / #2120 fix).
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200, 201, 202], dtype=np.int64),
            'x': np.array([100, 101, 102], dtype=np.int64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_uniform_int.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('transform') is not None


# --- Acceptance criterion 2: marker present after every read backend -----


class TestMarkerOnRead:
    """A no-georef file must surface the marker on every read path."""

    def test_cpu_eager(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133)
        assert _has_no_georef_marker(da)

    def test_cpu_dask(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, chunks=2)
        assert _has_no_georef_marker(da)

    @_gpu_only
    def test_gpu_eager(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, gpu=True)
        assert _has_no_georef_marker(da)

    @_gpu_only
    def test_gpu_dask(self, no_georef_path_2133):
        da = open_geotiff(no_georef_path_2133, gpu=True, chunks=2)
        assert _has_no_georef_marker(da)


def test_georef_file_has_no_marker(tmp_path):
    """Files with a transform must NOT carry the marker.

    Negative companion to the read-side tests: a real georeferenced
    write/read round-trip must leave the marker absent so downstream
    consumers can rely on its presence as a positive signal.
    """
    da = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        coords={
            'y': np.array([200.5, 201.5, 202.5], dtype=np.float64),
            'x': np.array([100.5, 101.5, 102.5], dtype=np.float64),
        },
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2133_real_transform.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert not _has_no_georef_marker(out)
    assert _NO_GEOREF_KEY not in out.attrs


# --- Acceptance criterion 3: marker survives copy / assign_attrs ---------


def test_marker_survives_copy():
    da = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        coords={'y': np.arange(2, dtype=np.int64),
                'x': np.arange(2, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    copied = da.copy()
    assert _has_no_georef_marker(copied)


def test_marker_survives_assign_attrs():
    da = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        coords={'y': np.arange(2, dtype=np.int64),
                'x': np.arange(2, dtype=np.int64)},
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    # assign_attrs returns a new array with the merged attrs dict.
    # The marker must persist because attrs are passed through, not
    # filtered on underscore prefix.
    out = da.assign_attrs(extra='added')
    assert _has_no_georef_marker(out)
    assert out.attrs.get('extra') == 'added'


# --- Acceptance criterion 4: 3D no-georef round-trip ---------------------


def test_3d_no_georef_round_trip(tmp_path):
    """A (band, y, x) no-georef array round-trips with the marker intact."""
    arr = np.zeros((3, 4, 4), dtype=np.float32)
    for i in range(3):
        arr[i] = i
    src = xr.DataArray(
        arr,
        coords={
            'band': np.arange(1, 4, dtype=np.int64),
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('band', 'y', 'x'),
        attrs={_NO_GEOREF_KEY: True},
    )
    path = str(tmp_path / "tmp_2133_3d_no_georef.tif")
    to_geotiff(src, path)
    out = open_geotiff(path)
    assert _has_no_georef_marker(out)
    assert out.attrs.get('transform') is None
    # Spatial coords come back as int64 placeholders.
    assert out.coords['y'].dtype == np.int64
    assert out.coords['x'].dtype == np.int64
    # The reader returns ``(y, x, band)``; transpose before comparing
    # to the ``(band, y, x)`` source.
    band_dim = next(
        d for d in out.dims if d not in ('y', 'x')
    )
    np.testing.assert_array_equal(
        out.transpose(band_dim, 'y', 'x').values, src.values
    )
