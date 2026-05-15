"""Round-trip tests for issue #1949.

A TIFF with no GeoTIFF tags (no ``ModelPixelScale``, no
``ModelTiepoint``, no GeoKeys) reads back with integer pixel coords
``[0, 1, 2, ...]`` and ``int64`` dtype (the no-georef branch added in
#1710 and #1753).

Before the fix, writing that DataArray back through ``to_geotiff``
silently injected a synthetic ``ModelPixelScale`` + ``ModelTiepoint``
because ``_coords_to_transform`` happily inferred a unit transform from
the integer pixel coords. The next read then took the georef branch
and the y/x dtype flipped from ``int64`` to ``float64`` with a
``transform`` attr present.

The fix makes ``_coords_to_transform`` return ``None`` when either
spatial coord array carries an integer dtype, so the no-georef contract
survives an arbitrary number of read -> write -> read cycles across
every writer path.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest
import tifffile
import xarray as xr

from xrspatial.geotiff import _coords_to_transform, open_geotiff, to_geotiff


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


# ---------------------------------------------------------------------------
# Unit tests on _coords_to_transform
# ---------------------------------------------------------------------------


def test_coords_to_transform_returns_none_for_int64_y_coords():
    """Integer y coords ``[0, 1, 2, ...]`` are the no-georef signal the
    read side emits; ``_coords_to_transform`` must not fabricate a
    transform from them."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.float64),
        },
    )
    assert _coords_to_transform(da) is None


def test_coords_to_transform_returns_none_for_int64_x_coords():
    """Same check on the other axis."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.float64),
            'x': np.arange(5, dtype=np.int64),
        },
    )
    assert _coords_to_transform(da) is None


@pytest.mark.parametrize("kind", [np.int32, np.int16, np.uint32])
def test_coords_to_transform_returns_none_for_int32_coords(kind):
    """int32 / int16 / uint also short-circuit -- the integer-dtype
    signal is "no georef", regardless of width."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=kind),
            'x': np.arange(5, dtype=kind),
        },
    )
    assert _coords_to_transform(da) is None, (
        f"expected None for {kind.__name__} coords")


def test_coords_to_transform_float_coords_unchanged():
    """Float coords still produce a GeoTransform (sanity baseline)."""
    da = xr.DataArray(
        np.zeros((4, 5), dtype=np.float32),
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        },
    )
    gt = _coords_to_transform(da)
    assert gt is not None
    assert gt.pixel_width == pytest.approx(10.0)
    assert gt.pixel_height == pytest.approx(1.0)


def test_coords_to_transform_3d_yxband_int_y_returns_none():
    """3D (y, x, band) with int y coords also short-circuits.

    The helper picks the y/x dims (filtering out 'band'); the integer
    check must apply to those, not to the band axis.
    """
    da = xr.DataArray(
        np.zeros((4, 5, 3), dtype=np.float32),
        dims=['y', 'x', 'band'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.int64),
            'band': np.arange(3),
        },
    )
    assert _coords_to_transform(da) is None


# ---------------------------------------------------------------------------
# Round-trip tests through to_geotiff
# ---------------------------------------------------------------------------


def _make_no_georef_tiff(path, shape=(4, 5), seed=1949):
    """Write a TIFF with no GeoTIFF tags."""
    arr = np.random.default_rng(seed=seed).random(shape).astype(np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )
    return arr


def test_round_trip_preserves_int_coords_eager(tmp_path):
    """Eager numpy round-trip: y/x coord dtype stays int64 and no
    transform attr is injected."""
    src = str(tmp_path / "no_georef_src_1949.tif")
    _make_no_georef_tiff(src)

    da = open_geotiff(src)
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    assert 'transform' not in da.attrs

    out = str(tmp_path / "no_georef_out_1949.tif")
    to_geotiff(da, out, compression='none', tiled=False)

    da2 = open_geotiff(out)
    assert da2.coords['y'].dtype == np.int64, (
        f"round-trip flipped y coord dtype from int64 to "
        f"{da2.coords['y'].dtype}: {da2.coords['y'].values}")
    assert da2.coords['x'].dtype == np.int64
    assert 'transform' not in da2.attrs, (
        f"round-trip injected fake transform attr: "
        f"{da2.attrs.get('transform')}")
    np.testing.assert_array_equal(
        da2.coords['y'].values, da.coords['y'].values)
    np.testing.assert_array_equal(
        da2.coords['x'].values, da.coords['x'].values)


def test_round_trip_preserves_int_coords_3d(tmp_path):
    """Same round-trip on a 3D multi-band file."""
    src = str(tmp_path / "no_georef_src_3d_1949.tif")
    arr = np.random.default_rng(seed=1949).random(
        (4, 5, 3)).astype(np.float32)
    tifffile.imwrite(src, arr, photometric='minisblack',
                     planarconfig='contig', metadata=None)

    da = open_geotiff(src)
    assert da.coords['y'].dtype == np.int64
    assert da.coords['x'].dtype == np.int64
    assert 'transform' not in da.attrs

    out = str(tmp_path / "no_georef_out_3d_1949.tif")
    to_geotiff(da, out, compression='none')

    da2 = open_geotiff(out)
    assert da2.coords['y'].dtype == np.int64
    assert da2.coords['x'].dtype == np.int64
    assert 'transform' not in da2.attrs


def test_round_trip_dask_streaming_preserves_int_coords(tmp_path):
    """Same contract via the dask streaming writer path."""
    src = str(tmp_path / "no_georef_src_dask_1949.tif")
    arr = np.random.default_rng(seed=1949).random(
        (64, 64)).astype(np.float32)
    tifffile.imwrite(src, arr, photometric='minisblack',
                     planarconfig='contig', tile=(32, 32),
                     compression='deflate', metadata=None)

    da_dk = open_geotiff(src, chunks=32)
    assert da_dk.coords['y'].dtype == np.int64
    assert da_dk.coords['x'].dtype == np.int64
    assert 'transform' not in da_dk.attrs

    out = str(tmp_path / "no_georef_out_dask_1949.tif")
    to_geotiff(da_dk, out, compression='deflate', tile_size=32)

    da_rt = open_geotiff(out)
    assert da_rt.coords['y'].dtype == np.int64
    assert da_rt.coords['x'].dtype == np.int64
    assert 'transform' not in da_rt.attrs


def test_double_round_trip_stable(tmp_path):
    """Two consecutive read -> write -> read cycles produce identical
    coord arrays and attrs sets."""
    src = str(tmp_path / "no_georef_double_1949.tif")
    _make_no_georef_tiff(src, shape=(6, 8))

    da1 = open_geotiff(src)
    out1 = str(tmp_path / "no_georef_double_out1_1949.tif")
    to_geotiff(da1, out1, compression='none', tiled=False)

    da2 = open_geotiff(out1)
    out2 = str(tmp_path / "no_georef_double_out2_1949.tif")
    to_geotiff(da2, out2, compression='none', tiled=False)

    da3 = open_geotiff(out2)
    np.testing.assert_array_equal(
        da1.coords['y'].values, da3.coords['y'].values)
    np.testing.assert_array_equal(
        da1.coords['x'].values, da3.coords['x'].values)
    assert da3.coords['y'].dtype == da1.coords['y'].dtype
    assert da3.coords['x'].dtype == da1.coords['x'].dtype
    # ``transform`` should never appear in either attrs set
    assert 'transform' not in da1.attrs
    assert 'transform' not in da3.attrs


def test_explicit_transform_attr_still_writes(tmp_path):
    """When the user explicitly sets ``attrs['transform']`` on a
    DataArray with int coords (an odd combination, but allowed), the
    writer still honours that explicit value -- the fix only catches the
    implicit-from-int-coords path."""
    arr = np.zeros((4, 5), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(5, dtype=np.int64),
        },
        attrs={'transform': (30.0, 0.0, 100000.0, 0.0, -30.0, 4000000.0),
               'crs': 32610},
    )
    out = str(tmp_path / "explicit_transform_1949.tif")
    to_geotiff(da, out, compression='none', tiled=False)
    da_rt = open_geotiff(out)
    assert 'transform' in da_rt.attrs
    assert da_rt.attrs['transform'][0] == 30.0
    assert da_rt.attrs['crs'] == 32610


@_gpu_only
def test_gpu_writer_preserves_int_coords(tmp_path):
    """The GPU writer path goes through the same ``_coords_to_transform``
    fallback; the fix has to extend there too."""
    src = str(tmp_path / "no_georef_src_gpu_1949.tif")
    _make_no_georef_tiff(src, shape=(64, 64))

    da_gpu = open_geotiff(src, gpu=True)
    assert da_gpu.coords['y'].dtype == np.int64
    assert da_gpu.coords['x'].dtype == np.int64
    assert 'transform' not in da_gpu.attrs

    out = str(tmp_path / "no_georef_out_gpu_1949.tif")
    to_geotiff(da_gpu, out, compression='deflate')

    da_rt = open_geotiff(out)
    assert da_rt.coords['y'].dtype == np.int64
    assert da_rt.coords['x'].dtype == np.int64
    assert 'transform' not in da_rt.attrs
