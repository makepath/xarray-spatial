"""Write-side tests for the GeoTIFF writer covering issue #1483.

Adds:
- T-5: dtype x compression round-trip matrix.
- T-6: NaN vs sentinel nodata round-trip semantics.
- T-7: COG validity check via rasterio (skipped if rasterio missing).
- T-9: write-to-readonly directory raises a clean OS/PermissionError.

T-10 (planar config 2 round-trip) is intentionally omitted -- the writer
does not currently emit PlanarConfiguration=2 (read-only support).
"""
from __future__ import annotations

import os
import platform
import stat
import sys

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# T-5: dtype x compression matrix
# ---------------------------------------------------------------------------

DTYPES_T5 = [
    np.uint8, np.uint16, np.uint32,
    np.int16, np.int32, np.int64,
    np.float32, np.float64,
]
CODECS_T5 = ['none', 'deflate', 'lzw', 'zstd', 'lz4']


def _make_dtype_arr(dtype, h=32, w=32):
    """Make a small array with values that fit the dtype's positive range."""
    n = h * w
    dt = np.dtype(dtype)
    if dt.kind == 'f':
        # Non-trivial floats; include a few extreme-ish values.
        arr = np.linspace(-1e3, 1e3, n).astype(dt).reshape(h, w)
    elif dt.kind == 'u':
        # Stay below uint16 max so it fits any unsigned dtype here.
        arr = (np.arange(n) % 1000).astype(dt).reshape(h, w)
    else:  # signed int
        arr = ((np.arange(n) % 2000) - 1000).astype(dt).reshape(h, w)
    return arr


def _codec_supports(codec, dtype):
    """Return False for combos the writer rejects, True otherwise."""
    # JPEG is not in the parametrized codec list (only uint8/3-band).
    # All listed codecs accept any of the listed dtypes.
    return True


@pytest.mark.parametrize('codec', CODECS_T5)
@pytest.mark.parametrize('dtype', DTYPES_T5)
def test_dtype_codec_roundtrip_stripped(tmp_path, dtype, codec):
    """Round-trip every dtype x codec in stripped layout."""
    if not _codec_supports(codec, dtype):
        pytest.skip(f"{codec} does not support {np.dtype(dtype).name}")

    expected = _make_dtype_arr(dtype)
    path = str(tmp_path / f'1483_t5_strip_{np.dtype(dtype).name}_{codec}.tif')

    try:
        write(expected, path, compression=codec, tiled=False)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"codec {codec} not available: {e}")

    arr, _geo = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == expected.dtype


@pytest.mark.parametrize('codec', CODECS_T5)
@pytest.mark.parametrize('dtype', DTYPES_T5)
def test_dtype_codec_roundtrip_tiled(tmp_path, dtype, codec):
    """Round-trip every dtype x codec in tiled layout."""
    if not _codec_supports(codec, dtype):
        pytest.skip(f"{codec} does not support {np.dtype(dtype).name}")

    expected = _make_dtype_arr(dtype)
    path = str(tmp_path / f'1483_t5_tile_{np.dtype(dtype).name}_{codec}.tif')

    try:
        write(expected, path, compression=codec, tiled=True, tile_size=16)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"codec {codec} not available: {e}")

    arr, _geo = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == expected.dtype


# ---------------------------------------------------------------------------
# T-6: NaN vs sentinel nodata
# ---------------------------------------------------------------------------

def _float_with_nan(h=8, w=8, dtype=np.float32):
    arr = np.linspace(0.0, 100.0, h * w, dtype=dtype).reshape(h, w)
    arr[0, 0] = np.nan
    arr[3, 5] = np.nan
    arr[-1, -1] = np.nan
    return arr


def test_nodata_nan_float_roundtrip(tmp_path):
    """nodata=NaN: NaN positions in the input round-trip as NaN."""
    expected = _float_with_nan(dtype=np.float32)
    path = str(tmp_path / '1483_t6_nodata_nan.tif')

    da = xr.DataArray(expected, dims=('y', 'x'))
    to_geotiff(da, path, nodata=float('nan'), compression='deflate')

    out = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(out.data), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(out.data[finite], expected[finite])


def test_nodata_sentinel_float_disk_vs_read(tmp_path):
    """nodata=-9999: NaN positions become sentinel on disk, NaN on read-back."""
    expected = _float_with_nan(dtype=np.float32)
    path = str(tmp_path / '1483_t6_nodata_sentinel.tif')

    da = xr.DataArray(expected, dims=('y', 'x'))
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    # On-disk values: NaN positions hold the sentinel.
    raw, _geo = read_to_array(path)
    nan_mask = np.isnan(expected)
    assert np.all(raw[nan_mask] == np.float32(-9999.0))
    # Non-NaN positions match.
    np.testing.assert_array_equal(raw[~nan_mask], expected[~nan_mask])

    # Read back through open_geotiff: sentinel becomes NaN again.
    out = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(out.data), nan_mask)
    np.testing.assert_array_equal(out.data[~nan_mask], expected[~nan_mask])
    assert out.attrs.get('nodata') == -9999.0


def test_nodata_uint8_sentinel(tmp_path):
    """nodata=255 for uint8: sentinel on disk, NaN on read (array promoted to float)."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8).copy()
    arr[0, 0] = 255
    arr[4, 4] = 255
    path = str(tmp_path / '1483_t6_nodata_uint8.tif')

    da = xr.DataArray(arr, dims=('y', 'x'))
    to_geotiff(da, path, nodata=255, compression='deflate')

    # On-disk: still uint8 with 255 in those slots.
    raw, _geo = read_to_array(path)
    assert raw.dtype == np.uint8
    assert raw[0, 0] == 255 and raw[4, 4] == 255
    np.testing.assert_array_equal(raw, arr)

    # Read-back: open_geotiff promotes integer with nodata to float + NaN.
    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    assert np.isnan(out.data[0, 0])
    assert np.isnan(out.data[4, 4])
    finite = ~np.isnan(out.data)
    np.testing.assert_array_equal(out.data[finite].astype(np.uint8),
                                  arr[finite])


# ---------------------------------------------------------------------------
# T-7: COG validity (rasterio-dependent)
# ---------------------------------------------------------------------------

rasterio = pytest.importorskip(
    'rasterio',
    reason='rasterio is optional; COG validity test skipped when missing',
)


def test_cog_layout_and_overviews(tmp_path):
    """A cog=True file is tiled, carries overviews, and (when rio-cogeo is
    installed) passes the COG validator.

    Note: xrspatial does not currently emit GDAL's IMAGE_STRUCTURE.LAYOUT=COG
    tag, so we don't assert that. Structural COG properties (tiled, overviews
    present, GDAL-readable) are what the writer actually guarantees.
    """
    h = w = 1024
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w) % 1000.0
    path = str(tmp_path / '1483_t7_cog.tif')

    da = xr.DataArray(arr, dims=('y', 'x'))
    to_geotiff(
        da, path, crs=4326, cog=True, compression='deflate', tile_size=256,
    )

    with rasterio.open(path) as src:
        assert src.is_tiled, "COG output must be tiled"
        # 1024x1024 with 256 tiles produces at least one halving.
        ovs = src.overviews(1)
        assert len(ovs) >= 1, f"expected at least one overview, got {ovs}"
        assert ovs[0] in (2, 4, 8, 16), f"unexpected first overview: {ovs}"
        # Each overview should be strictly larger than the previous (decimation
        # factors are monotonically increasing).
        assert all(b > a for a, b in zip(ovs, ovs[1:])), \
            f"overview decimations not monotonically increasing: {ovs}"
        # Sanity: full-resolution band should round-trip values.
        sample = src.read(1, window=((0, 4), (0, 4)))
        np.testing.assert_array_equal(sample, arr[:4, :4])

    # If rio-cogeo is installed, run its validator for the gold-standard check.
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        return
    valid, errors, _warnings = cog_validate(path, strict=False)
    assert valid, f"rio-cogeo cog_validate failed: errors={errors}"


# ---------------------------------------------------------------------------
# T-9: write-to-readonly directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='POSIX chmod semantics required',
)
@pytest.mark.skipif(
    hasattr(os, 'geteuid') and os.geteuid() == 0,
    reason='root bypasses directory permissions',
)
def test_write_to_readonly_dir_raises_oserror(tmp_path):
    """Writing into a chmod 0o555 directory must raise OSError/PermissionError."""
    ro_dir = tmp_path / '1483_t9_readonly'
    ro_dir.mkdir()
    target = str(ro_dir / 'out.tif')

    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(arr, dims=('y', 'x'))

    original_mode = ro_dir.stat().st_mode
    try:
        os.chmod(ro_dir, 0o555)
        with pytest.raises((OSError, PermissionError)):
            to_geotiff(da, target, compression='deflate')
    finally:
        os.chmod(ro_dir, original_mode)
