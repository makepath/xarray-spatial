"""GPU reader tests.

This module folds the GPU read-path regression tests into one place so
the reader story sits in a single test target. It uses the shared
``requires_gpu`` marker from
``xrspatial/geotiff/tests/_helpers/markers.py`` so the GPU probe lives
in one place.

Sections, by the behaviour they pin:

- ``_read_geotiff_gpu(..., window=..., band=...)`` kwarg forwarding
- ``stripped multiband`` -- 3-D ``(y, x, band)`` reads via the stripped fallback
- stripped no-georef windowed coord parity
- stripped fallback forwards ``max_pixels`` / ``window`` / ``band``
- GPU read nodata promotion + ``attrs['nodata']``
- ``_write_geotiff_gpu`` rejects MinIsWhite on band-first single-band
- ``_read_geotiff_gpu(chunks=...)`` is a real out-of-core dask graph
- sidecar overview-inheritance georef parity across backends
- ``_read_geotiff_gpu`` accepts HTTP / fsspec URLs on the eager path
- GDS chunked path casts each chunk to the declared dtype

Markers come from ``_helpers/markers.py``. Every test carries
``@requires_gpu`` (aliased ``_gpu_only`` for ergonomic decoration);
the ``requires_loopback`` marker gates URL tests that bind a local
HTTP server.
"""
from __future__ import annotations

import http.server
import importlib.util
import os
import socketserver
import tempfile
import threading

import numpy as np
import pytest
import xarray as xr

from .._helpers.markers import requires_gpu as _gpu_only
from .._helpers.markers import requires_loopback

# ---------------------------------------------------------------------------
# Section: window / band kwarg forwarding on GPU read
# ---------------------------------------------------------------------------


@pytest.fixture
def single_band_tiff_1605(tmp_path):
    """16x20 single-band tiled tiff with a non-trivial transform."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.arange(16, dtype=np.float64) * 0.25 + 100.0,
            'x': np.arange(20, dtype=np.float64) * 0.25 + 200.0,
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'window_band_1605_single.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


@pytest.fixture
def multi_band_tiff_1605(tmp_path):
    """16x20x3 tiled tiff -- exercises the band-selection branch."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(16 * 20 * 3, dtype=np.float32).reshape(16, 20, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.arange(16, dtype=np.float64) * 0.25 + 100.0,
            'x': np.arange(20, dtype=np.float64) * 0.25 + 200.0,
            'band': np.arange(3),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'window_band_1605_multi.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


@_gpu_only
def test_read_geotiff_gpu_window_matches_eager_1605(single_band_tiff_1605):
    """Direct call: GPU window slice matches CPU eager window slice."""
    path, source_arr = single_band_tiff_1605
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    window = (2, 4, 12, 14)

    cpu = open_geotiff(path, window=window)
    gpu = _read_geotiff_gpu(path, window=window)

    assert gpu.shape == cpu.shape == (10, 10)
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)
    np.testing.assert_array_equal(gpu['y'].values, cpu['y'].values)
    np.testing.assert_array_equal(gpu['x'].values, cpu['x'].values)
    assert gpu.attrs.get('transform') == cpu.attrs.get('transform')


@_gpu_only
def test_open_geotiff_gpu_window_no_longer_silently_dropped_1605(
        single_band_tiff_1605):
    """open_geotiff(gpu=True, window=...) honors the window."""
    path, source_arr = single_band_tiff_1605
    from xrspatial.geotiff import open_geotiff

    window = (3, 5, 9, 13)
    gpu = open_geotiff(path, gpu=True, window=window)

    # Pre-fix shape was the full (16, 20). Post-fix it must shrink.
    assert gpu.shape == (6, 8)
    np.testing.assert_array_equal(
        gpu.data.get(),
        source_arr[3:9, 5:13],
    )


@_gpu_only
def test_read_geotiff_gpu_band_selection_1605(multi_band_tiff_1605):
    """Direct call: band=k returns the kth band as a 2D DataArray."""
    path, source_arr = multi_band_tiff_1605
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    cpu = open_geotiff(path, band=1)
    gpu = _read_geotiff_gpu(path, band=1)

    assert gpu.shape == cpu.shape == (16, 20)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)


@_gpu_only
def test_open_geotiff_gpu_band_no_longer_silently_dropped_1605(
        multi_band_tiff_1605):
    """open_geotiff(gpu=True, band=...) honors band."""
    path, source_arr = multi_band_tiff_1605
    from xrspatial.geotiff import open_geotiff

    gpu = open_geotiff(path, gpu=True, band=2)

    # Pre-fix would have returned a (16, 20, 3) DataArray (full file).
    assert gpu.shape == (16, 20)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(
        gpu.data.get(),
        source_arr[:, :, 2],
    )


@_gpu_only
def test_read_geotiff_gpu_window_and_band_1605(multi_band_tiff_1605):
    """window + band combine cleanly."""
    path, source_arr = multi_band_tiff_1605
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    window = (1, 2, 11, 17)
    cpu = open_geotiff(path, window=window, band=0)
    gpu = _read_geotiff_gpu(path, window=window, band=0)

    assert gpu.shape == cpu.shape == (10, 15)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)


@_gpu_only
def test_read_geotiff_gpu_window_bounds_validation_1605(single_band_tiff_1605):
    """Out-of-bounds window raises ValueError, mirroring the dask path."""
    path, _ = single_band_tiff_1605
    from xrspatial.geotiff import _read_geotiff_gpu

    with pytest.raises(ValueError, match="outside the source extent"):
        _read_geotiff_gpu(path, window=(0, 0, 100, 100))

    with pytest.raises(ValueError, match="non-positive size"):
        _read_geotiff_gpu(path, window=(5, 0, 5, 10))

    with pytest.raises(ValueError, match="must be a 4-tuple"):
        _read_geotiff_gpu(path, window=(0, 0, 5))


@_gpu_only
def test_read_geotiff_gpu_band_bounds_validation_1605(
        multi_band_tiff_1605, single_band_tiff_1605):
    """Out-of-range band raises IndexError."""
    multi_path, _ = multi_band_tiff_1605
    single_path, _ = single_band_tiff_1605
    from xrspatial.geotiff import _read_geotiff_gpu

    with pytest.raises(IndexError, match="out of range"):
        _read_geotiff_gpu(multi_path, band=10)

    with pytest.raises(IndexError, match="single-band file"):
        _read_geotiff_gpu(single_path, band=1)


@_gpu_only
def test_read_geotiff_gpu_window_rejected_on_nondefault_orientation_1605(
        tmp_path):
    """Mirror the CPU reader: orientation != 1 + window= is ambiguous, raise.

    ``_reader.read_to_array`` rejects this combo with a specific message;
    the GPU path must match so backend substitutability holds. The
    stripped branch is the easy way to write an Orientation=2 file via
    tifffile.
    """
    tifffile = pytest.importorskip("tifffile")
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    path = tmp_path / "orient2_stripped_1605.tif"
    # Orientation tag 274 = 2 (top-right, horizontal flip).
    tifffile.imwrite(
        str(path), arr,
        extratags=[(274, 'H', 1, 2, True)],
    )

    with pytest.raises(ValueError, match=r"Orientation tag \(274\) is 2"):
        _read_geotiff_gpu(str(path), window=(0, 0, 8, 10))


@_gpu_only
def test_read_geotiff_gpu_stripped_chunks_produces_dask_1605(tmp_path):
    """Stripped-file branch must honour ``chunks=`` like the tiled branch.

    Pre-fix, the stripped branch returned the eager CuPy DataArray without
    calling ``.chunk(...)``, silently dropping ``chunks=`` for stripped
    TIFFs. The tiled branch already chunked correctly; this test pins
    the parity.
    """
    tifffile = pytest.importorskip("tifffile")
    import dask.array as dask_array

    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    path = tmp_path / "stripped_chunks_1605.tif"
    # No ``tile=`` argument -> stripped layout. Orientation defaults to 1.
    tifffile.imwrite(str(path), arr)

    result = _read_geotiff_gpu(str(path), chunks=8)

    # The result should be Dask-backed; .data is a dask Array wrapping CuPy.
    assert isinstance(result.data, dask_array.Array), \
        f"stripped chunks=8 did not produce a dask-backed DataArray; got " \
        f"{type(result.data).__name__}"
    assert result.chunks == ((8, 8), (8, 8, 4))
    # And the underlying chunks are CuPy arrays, not numpy.
    block = result.data.blocks[0, 0].compute()
    assert type(block).__module__.startswith("cupy"), \
        f"expected CuPy chunk, got {type(block).__module__}"
    np.testing.assert_array_equal(result.data.compute().get(), arr)


@_gpu_only
def test_read_geotiff_gpu_stripped_chunks_tuple_1605(tmp_path):
    """Stripped branch accepts the (rh, cw) tuple chunks spec too."""
    tifffile = pytest.importorskip("tifffile")
    import dask.array as dask_array

    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    path = tmp_path / "stripped_chunks_tuple_1605.tif"
    tifffile.imwrite(str(path), arr)

    result = _read_geotiff_gpu(str(path), chunks=(4, 10))
    assert isinstance(result.data, dask_array.Array)
    assert result.chunks == ((4, 4, 4, 4), (10, 10))


# ---------------------------------------------------------------------------
# Section: stripped multiband -- 3-D (y, x, band) via stripped fallback
# ---------------------------------------------------------------------------

@_gpu_only
def test_stripped_3band_uint8_stripped_multiband():
    """3-band uint8 stripped TIFF reads as (y, x, band)."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 200, size=(64, 96, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt.tif')
        to_geotiff(da, p, tiled=False)
        out = _read_geotiff_gpu(p)
        assert out.dims == ('y', 'x', 'band')
        assert out.shape == (64, 96, 3)
        np.testing.assert_array_equal(out.data.get(), data)


@_gpu_only
def test_stripped_2band_uint16_stripped_multiband():
    """2-band uint16 stripped TIFF reads as (y, x, band)."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 60000, size=(48, 80, 2)).astype(np.uint16)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt2.tif')
        to_geotiff(da, p, tiled=False)
        out = _read_geotiff_gpu(p)
        assert out.dims == ('y', 'x', 'band')
        assert out.shape == (48, 80, 2)
        assert out.data.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(out.data.get(), data)


@_gpu_only
def test_stripped_singleband_still_2d_stripped_multiband():
    """Single-band stripped TIFF still produces a 2-D (y, x) DataArray."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260508)
    data = rng.randint(0, 200, size=(40, 60)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'wt1.tif')
        to_geotiff(da, p, tiled=False)
        out = _read_geotiff_gpu(p)
        assert out.dims == ('y', 'x')
        assert out.shape == (40, 60)
        np.testing.assert_array_equal(out.data.get(), data)


# ---------------------------------------------------------------------------
# Section: stripped no-georef windowed coord parity
# ---------------------------------------------------------------------------

@pytest.fixture
def no_georef_stripped_path_1753(tmp_path):
    """8x8 float32 stripped TIFF with NO GeoTIFF tags."""
    from xrspatial.geotiff._writer import write

    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "no_georef_stripped_1753.tif")
    write(arr, path, compression='none', tiled=False)
    return path


@pytest.fixture
def no_georef_stripped_uint16_path_1753(tmp_path):
    """8x8 uint16 stripped TIFF with NO GeoTIFF tags."""
    from xrspatial.geotiff._writer import write

    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = str(tmp_path / "no_georef_stripped_uint16_1753.tif")
    write(arr, path, compression='none', tiled=False)
    return path


@_gpu_only
class TestStrippedGpuWindowedNoGeoref1753:
    """GPU stripped fallback must emit integer pixel coords on no-georef
    windowed reads, matching CPU eager + dask + tiled GPU."""

    def test_window_at_origin(self, no_georef_stripped_path_1753):
        """Window at (0, 0) -> coords ``np.arange(0, 4)``."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64, (
            f"y.dtype should be int64 for no-georef; got {da.y.dtype}"
        )
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_offset_window(self, no_georef_stripped_path_1753):
        """Offset window at (2, 3) -> coords ``np.arange(2, 6) / (3, 7)``."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_dask_cupy_window(self, no_georef_stripped_path_1753):
        """``gpu=True, chunks=...`` shares the same fallback coord path."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          chunks=4, window=(0, 0, 4, 4))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(4))
        np.testing.assert_array_equal(da.x.values, np.arange(4))

    def test_dask_cupy_offset_window(self, no_georef_stripped_path_1753):
        """``gpu=True, chunks=...`` with an offset window."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          chunks=4, window=(2, 3, 6, 7))
        assert da.y.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(2, 6))
        np.testing.assert_array_equal(da.x.values, np.arange(3, 7))

    def test_no_transform_attr(self, no_georef_stripped_path_1753):
        """No fabricated ``attrs['transform']`` for non-georef windowed
        reads on GPU."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                          window=(0, 0, 4, 4))
        assert 'transform' not in da.attrs, (
            f"non-georef stripped GPU windowed read should not carry a "
            f"transform attr; got {dict(da.attrs)}"
        )

    def test_uint16_window(self, no_georef_stripped_uint16_path_1753):
        """Fix is dtype-independent; uint16 source still gets int64 coords."""
        from xrspatial.geotiff import open_geotiff
        da = open_geotiff(no_georef_stripped_uint16_path_1753, gpu=True,
                          window=(1, 2, 5, 6))
        assert da.y.dtype == np.int64
        assert da.x.dtype == np.int64
        np.testing.assert_array_equal(da.y.values, np.arange(1, 5))
        np.testing.assert_array_equal(da.x.values, np.arange(2, 6))


@_gpu_only
class TestStrippedGpuWindowedBackendParity1753:
    """Every backend must agree on coord dtype and values for the same
    windowed read of a non-georef stripped TIFF."""

    @pytest.mark.parametrize("win", [
        (0, 0, 4, 4),
        (2, 3, 6, 7),
        (1, 1, 7, 7),
    ])
    def test_dtype_parity(self, no_georef_stripped_path_1753, win):
        from xrspatial.geotiff import open_geotiff
        eager = open_geotiff(no_georef_stripped_path_1753, window=win)
        dask = open_geotiff(no_georef_stripped_path_1753, chunks=4, window=win)
        gpu = open_geotiff(no_georef_stripped_path_1753, gpu=True, window=win)
        gpu_dk = open_geotiff(no_georef_stripped_path_1753, gpu=True,
                              chunks=4, window=win)
        assert eager.y.dtype == dask.y.dtype == gpu.y.dtype == gpu_dk.y.dtype
        assert eager.y.dtype == np.int64
        assert eager.x.dtype == dask.x.dtype == gpu.x.dtype == gpu_dk.x.dtype
        assert eager.x.dtype == np.int64

    @pytest.mark.parametrize("win", [
        (0, 0, 4, 4),
        (2, 3, 6, 7),
        (1, 1, 7, 7),
    ])
    def test_value_parity(self, no_georef_stripped_path_1753, win):
        from xrspatial.geotiff import open_geotiff
        eager = open_geotiff(no_georef_stripped_path_1753, window=win)
        gpu = open_geotiff(no_georef_stripped_path_1753, gpu=True, window=win)
        np.testing.assert_array_equal(eager.y.values, gpu.y.values)
        np.testing.assert_array_equal(eager.x.values, gpu.x.values)


@_gpu_only
class TestStrippedGpuWindowedGeorefStillWorks1753:
    """The fix must not regress georeferenced stripped GPU reads."""

    def test_georef_pixel_is_area_window(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        from xrspatial.geotiff._geotags import GeoTransform
        from xrspatial.geotiff._writer import write

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_pixel_is_area_1753.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, gpu=True, window=(0, 0, 4, 4))
        assert da.y.dtype == np.float64
        assert da.x.dtype == np.float64
        assert da.x.values[0] == pytest.approx(500015.0)
        assert da.y.values[0] == pytest.approx(3999985.0)
        assert 'transform' in da.attrs

    def test_georef_offset_window(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        from xrspatial.geotiff._geotags import GeoTransform
        from xrspatial.geotiff._writer import write

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gt = GeoTransform(
            origin_x=500000.0, origin_y=4000000.0,
            pixel_width=30.0, pixel_height=-30.0,
        )
        path = str(tmp_path / "georef_offset_1753.tif")
        write(arr, path, geo_transform=gt, crs_epsg=32610,
              compression='none', tiled=False)
        da = open_geotiff(path, gpu=True, window=(2, 3, 6, 7))
        assert da.x.values[0] == pytest.approx(500105.0)
        assert da.y.values[0] == pytest.approx(3999925.0)


# ---------------------------------------------------------------------------
# Section: stripped fallback forwards max_pixels/window/band
# ---------------------------------------------------------------------------

@_gpu_only
def test_stripped_max_pixels_cap_is_enforced_1732():
    """max_pixels smaller than the file must raise before full decode."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(64, 96)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_cap.tif')
        to_geotiff(da, p, tiled=False)
        # 64 * 96 = 6144 pixels; cap at 1000 must reject.
        with pytest.raises(ValueError, match="max_pixels|pixel"):
            _read_geotiff_gpu(p, max_pixels=1000)


@_gpu_only
def test_stripped_window_returns_only_window_1732():
    """Windowed read on a stripped file returns the window-sized array
    with coords and transform that match the window origin."""
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(64, 96)).astype(np.uint8)
    da = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={
            'y': np.arange(64, dtype=np.float64) * 0.5 + 100.0,
            'x': np.arange(96, dtype=np.float64) * 0.5 + 200.0,
        },
        attrs={'crs': 4326},
    )

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_win.tif')
        to_geotiff(da, p, tiled=False)
        win = (8, 16, 40, 80)  # 32x64 window
        out = _read_geotiff_gpu(p, window=win)
        assert out.shape == (32, 64)
        np.testing.assert_array_equal(out.data.get(), data[8:40, 16:80])

        cpu = open_geotiff(p, window=win)
        np.testing.assert_array_equal(out.coords['y'].values,
                                      cpu.coords['y'].values)
        np.testing.assert_array_equal(out.coords['x'].values,
                                      cpu.coords['x'].values)
        assert out.attrs['transform'] == cpu.attrs['transform']
        assert out.attrs['transform'] == (0.5, 0.0, 207.75,
                                          0.0, 0.5, 103.75)


@_gpu_only
def test_stripped_band_selection_returns_2d_1732():
    """Selecting band=1 on a 3-band stripped file returns a 2D array
    matching the requested band."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(48, 80, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_band.tif')
        to_geotiff(da, p, tiled=False)
        out = _read_geotiff_gpu(p, band=1)
        assert out.dims == ('y', 'x')
        assert out.shape == (48, 80)
        np.testing.assert_array_equal(out.data.get(), data[:, :, 1])


@_gpu_only
def test_stripped_window_plus_band_1732():
    """Windowed read with band selection composes correctly."""
    from xrspatial.geotiff import _read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(20260512)
    data = rng.randint(0, 200, size=(48, 80, 3)).astype(np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tmp_1732_wb.tif')
        to_geotiff(da, p, tiled=False)
        win = (4, 8, 36, 72)  # 32x64
        out = _read_geotiff_gpu(p, window=win, band=2)
        assert out.dims == ('y', 'x')
        assert out.shape == (32, 64)
        np.testing.assert_array_equal(
            out.data.get(), data[4:36, 8:72, 2])


# ---------------------------------------------------------------------------
# Section: GPU read nodata promotion + attrs['nodata']
# ---------------------------------------------------------------------------

@_gpu_only
def test_gpu_uint16_nodata_promoted_and_masked_tiled_1542(tmp_path):
    """uint16 + nodata sentinel -> GPU returns float64 with NaN, attrs[nodata] set."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_nodata_1542_tiled.tif')
    write(arr, path, nodata=65535, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path, masked=True)
    gpu = open_geotiff(path, gpu=True, masked=True)

    assert cpu.dtype == gpu.dtype == np.float64
    assert cpu.attrs.get('nodata') == 65535.0
    assert gpu.attrs.get('nodata') == 65535.0
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))
    finite = ~np.isnan(cpu_arr)
    np.testing.assert_array_equal(cpu_arr[finite], gpu_arr[finite])


@_gpu_only
def test_gpu_uint16_nodata_promoted_and_masked_stripped_1542(tmp_path):
    """Stripped fallback path also promotes + masks integer nodata."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_nodata_1542_stripped.tif')
    write(arr, path, nodata=65535, compression='deflate', tiled=False)

    cpu = open_geotiff(path, masked=True)
    gpu = open_geotiff(path, gpu=True, masked=True)

    assert cpu.dtype == gpu.dtype == np.float64
    assert gpu.attrs.get('nodata') == 65535.0
    np.testing.assert_array_equal(np.isnan(cpu.values),
                                  np.isnan(gpu.data.get()))


@_gpu_only
def test_gpu_float32_sentinel_replaced_with_nan_1542(tmp_path):
    """float32 + finite sentinel -> GPU returns float32 with NaN at sentinel positions."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    arr[0, 0] = -9999.0
    arr[2, 3] = -9999.0
    path = str(tmp_path / 'gpu_f32_sentinel_1542.tif')
    write(arr, path, nodata=-9999.0, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    assert cpu.dtype == gpu.dtype == np.float32
    assert gpu.attrs.get('nodata') == -9999.0
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))
    finite = ~np.isnan(cpu_arr)
    np.testing.assert_array_equal(cpu_arr[finite], gpu_arr[finite])


@_gpu_only
def test_gpu_no_nodata_keeps_dtype_1542(tmp_path):
    """No nodata declared -> GPU keeps source dtype, no nodata attr added."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_no_nodata_1542.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16)

    gpu = open_geotiff(path, gpu=True)
    assert gpu.dtype == np.uint16
    assert gpu.attrs.get('nodata') is None
    np.testing.assert_array_equal(gpu.data.get(), arr)


@_gpu_only
def test_gpu_nan_nodata_passes_through_1542(tmp_path):
    """nodata=NaN on float data -> GPU returns NaN positions intact, attrs[nodata]=nan."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'gpu_f32_nan_1542.tif')
    write(arr, path, nodata=float('nan'), compression='deflate',
          tiled=True, tile_size=16)

    gpu = open_geotiff(path, gpu=True)
    assert np.isnan(gpu.attrs.get('nodata'))
    gpu_arr = gpu.data.get()
    assert np.isnan(gpu_arr[1, 0])
    finite = ~np.isnan(gpu_arr)
    np.testing.assert_array_equal(gpu_arr[finite], arr[~np.isnan(arr)])


@_gpu_only
def test_gpu_all_four_backends_agree_on_nodata_1542(tmp_path):
    """numpy / dask+numpy / cupy / dask+cupy all agree on dtype + nodata + NaN."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_4backends_1542.tif')
    write(arr, path, nodata=65535, compression='deflate',
          tiled=True, tile_size=16)

    da_np = open_geotiff(path, masked=True)
    da_dask = open_geotiff(path, chunks=512, masked=True)
    da_gpu = open_geotiff(path, gpu=True, masked=True)
    da_gpu_dask = open_geotiff(path, gpu=True, chunks=512, masked=True)

    for label, da in [('np', da_np), ('dask+np', da_dask),
                      ('gpu', da_gpu), ('gpu+dask', da_gpu_dask)]:
        assert da.dtype == np.float64, f"{label}: dtype={da.dtype}"
        assert da.attrs.get('nodata') == 65535.0, (
            f"{label}: missing nodata attr (got {da.attrs.get('nodata')!r})"
        )

    np_arr = da_np.values
    dask_arr = da_dask.compute().values
    gpu_arr = da_gpu.data.get()
    gpu_dask_arr = da_gpu_dask.compute().data.get()
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(dask_arr))
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(gpu_arr))
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(gpu_dask_arr))


@_gpu_only
def test_gpu_int16_negative_nodata_1542(tmp_path):
    """Signed integer with negative nodata: also promoted to float64 + NaN."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[-9999, 10], [20, -9999]], dtype=np.int16)
    path = str(tmp_path / 'gpu_i16_nodata_1542.tif')
    write(arr, path, nodata=-9999, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path, masked=True)
    gpu = open_geotiff(path, gpu=True, masked=True)
    assert cpu.dtype == gpu.dtype == np.float64
    assert gpu.attrs.get('nodata') == -9999.0
    np.testing.assert_array_equal(np.isnan(cpu.values),
                                  np.isnan(gpu.data.get()))


# ---------------------------------------------------------------------------
# Section: _write_geotiff_gpu rejects MinIsWhite (band-first guard)
#
# The last test in this section
# (``test_samples_hint_band_first_without_gpu_2097``) is CPU-only by design:
# it drives ``_compute_gpu_samples_hint`` directly with a ``_FakeArray2097``
# stand-in so the band-axis logic stays pinned even on a no-GPU CI runner.
# Don't add ``@_gpu_only`` to it.
# ---------------------------------------------------------------------------

@_gpu_only
def test_band_first_single_band_miniswhite_rejected_2097(tmp_path):
    """A band-first single-band DataArray with ``photometric='miniswhite'``
    must raise ``NotImplementedError`` on the GPU writer."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import _write_geotiff_gpu

    arr = cp.zeros((1, 4, 8), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("band", "y", "x"))
    out = tmp_path / "tmp_2097_miniswhite_band_first.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        _write_geotiff_gpu(da, str(out), photometric="miniswhite")
    assert not out.exists()


@_gpu_only
def test_band_last_single_band_miniswhite_still_rejected_2097(tmp_path):
    """The pre-existing band-last single-band rejection must still fire."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import _write_geotiff_gpu

    arr = cp.zeros((4, 8, 1), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("y", "x", "band"))
    out = tmp_path / "tmp_2097_miniswhite_band_last.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        _write_geotiff_gpu(da, str(out), photometric="miniswhite")
    assert not out.exists()


@_gpu_only
def test_2d_single_band_miniswhite_still_rejected_2097(tmp_path):
    """2D inputs are the simplest single-band case."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import _write_geotiff_gpu

    arr = cp.zeros((4, 8), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("y", "x"))
    out = tmp_path / "tmp_2097_miniswhite_2d.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        _write_geotiff_gpu(da, str(out), photometric="miniswhite")
    assert not out.exists()


class _FakeArray2097:
    """Stand-in for a DataArray-like object that exposes ``ndim``,
    ``shape``, and ``dims``. Used to drive ``_compute_gpu_samples_hint``
    on a CI runner without cupy or a CUDA device."""

    def __init__(self, shape, dims):
        self.shape = shape
        self.dims = dims
        self.ndim = len(shape)


def test_samples_hint_band_first_without_gpu_2097():
    """Call ``_compute_gpu_samples_hint`` directly so a regression
    fails here even on a no-GPU CI runner."""
    from xrspatial.geotiff._writers.gpu import _compute_gpu_samples_hint

    cases = [
        ((1, 4, 8), ("band", "y", "x"), 1),    # band-first single-band
        ((4, 8, 1), ("y", "x", "band"), 1),    # band-last single-band
        ((3, 4, 8), ("band", "y", "x"), 3),    # band-first 3-band
        ((4, 8, 3), ("y", "x", "band"), 3),    # band-last 3-band
        ((4, 8), ("y", "x"), 1),                # 2D always 1
        ((4, 8, 1), None, 1),                   # raw 3D defaults band-last
    ]
    for shape, dims, expected in cases:
        got = _compute_gpu_samples_hint(_FakeArray2097(shape, dims))
        assert got == expected, (shape, dims, got, expected)


# ---------------------------------------------------------------------------
# Section: _read_geotiff_gpu(chunks=...) out-of-core dask pipeline
# ---------------------------------------------------------------------------

def _kvikio_available_1876() -> bool:
    return importlib.util.find_spec("kvikio") is not None


_HAS_KVIKIO_1876 = _kvikio_available_1876()
# The GDS path additionally requires kvikio. Stack ``@_gpu_only`` with the
# kvikio probe at the test site so the cupy + CUDA runtime check is shared
# with the rest of the module rather than reimplemented here.
_requires_kvikio_1876 = pytest.mark.skipif(
    not _HAS_KVIKIO_1876, reason="kvikio required for GDS path",
)


@pytest.fixture
def small_raster_path_1876(tmp_path):
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      attrs={'crs': 4326,
                             'transform': (1.0, 0, 0, 0, -1.0, 32.0)})
    path = str(tmp_path / 'small_raster_1876.tif')
    to_geotiff(da, path, compression='deflate', tile_size=16)
    return path


@pytest.fixture
def multi_band_path_1876(tmp_path):
    from xrspatial.geotiff import to_geotiff

    rng = np.random.RandomState(42)
    arr = rng.rand(3, 32, 32).astype(np.float32)
    da = xr.DataArray(arr, dims=['band', 'y', 'x'],
                      attrs={'crs': 4326,
                             'transform': (1.0, 0, 0, 0, -1.0, 32.0)})
    path = str(tmp_path / 'multi_band_1876.tif')
    to_geotiff(da, path, compression='deflate', tile_size=16)
    return path


@_gpu_only
def test_read_geotiff_gpu_chunks_yields_dask_cupy_chunks_1876(
        small_raster_path_1876):
    """Each block of the returned dask array must be a cupy array."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import _read_geotiff_gpu

    result = _read_geotiff_gpu(small_raster_path_1876, chunks=8)

    assert isinstance(result.data, da_mod.Array), (
        f"expected dask Array, got {type(result.data).__name__}"
    )
    assert isinstance(result.data._meta, cupy.ndarray), (
        f"expected cupy chunks, got meta={type(result.data._meta).__name__}"
    )
    assert result.data.numblocks == (4, 4)

    block = result.data.blocks[0, 0].compute()
    assert isinstance(block, cupy.ndarray)
    assert block.shape == (8, 8)


@_gpu_only
def test_read_geotiff_gpu_chunks_values_match_eager_1876(
        small_raster_path_1876):
    """Lazy chunked result must equal the eager GPU result element-wise."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    eager = _read_geotiff_gpu(small_raster_path_1876)
    chunked = _read_geotiff_gpu(small_raster_path_1876, chunks=8)

    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(chunked.compute().data)
    np.testing.assert_array_equal(eager_np, chunked_np)


@_gpu_only
def test_read_geotiff_gpu_no_chunks_returns_eager_cupy_1876(
        small_raster_path_1876):
    """``chunks=None`` keeps the eager GPU decode path."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    result = _read_geotiff_gpu(small_raster_path_1876)

    assert isinstance(result.data, cupy.ndarray)


@_gpu_only
def test_open_geotiff_gpu_chunks_propagates_to_dask_1876(
        small_raster_path_1876):
    """``open_geotiff(gpu=True, chunks=...)`` returns Dask+CuPy."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import open_geotiff

    result = open_geotiff(small_raster_path_1876, gpu=True, chunks=8)

    assert isinstance(result.data, da_mod.Array)
    assert isinstance(result.data._meta, cupy.ndarray)


@_gpu_only
def test_read_geotiff_gpu_chunks_preserves_attrs_1876(
        small_raster_path_1876):
    """Geo attrs (transform, crs) must survive the dask path."""
    from xrspatial.geotiff import _read_geotiff_gpu

    result = _read_geotiff_gpu(small_raster_path_1876, chunks=8)
    assert 'transform' in result.attrs
    assert 'crs' in result.attrs


@_gpu_only
@_requires_kvikio_1876
def test_read_geotiff_gpu_chunks_uses_gds_path_when_available_1876(
        small_raster_path_1876, monkeypatch):
    """When kvikio is installed and the file qualifies, each chunk task
    must call the direct disk->GPU decoder rather than detouring through
    ``_read_geotiff_dask``."""
    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gtmod

    direct_calls = {'n': 0}
    real_direct = gtmod._decode_window_gpu_direct

    def _spy(*args, **kwargs):
        direct_calls['n'] += 1
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(gtmod, '_decode_window_gpu_direct', _spy)

    result = _read_geotiff_gpu(small_raster_path_1876, chunks=8)
    result.compute()

    assert direct_calls['n'] == 16, (
        f"expected one disk->GPU call per chunk (4x4 = 16); "
        f"got {direct_calls['n']}"
    )


@_gpu_only
def test_read_geotiff_gpu_chunks_window_subset_1876(small_raster_path_1876):
    """A window on the dask path produces the same values as a window
    on the eager path."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    eager = _read_geotiff_gpu(small_raster_path_1876, window=(4, 4, 24, 28))
    chunked = _read_geotiff_gpu(small_raster_path_1876, chunks=8,
                                window=(4, 4, 24, 28))

    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(chunked.compute().data)
    assert eager_np.shape == (20, 24)
    np.testing.assert_array_equal(eager_np, chunked_np)


@_gpu_only
def test_read_geotiff_gpu_chunks_multi_band_1876(multi_band_path_1876):
    """Multi-band tiled files chunk along (y, x) with a band axis."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import _read_geotiff_gpu

    result = _read_geotiff_gpu(multi_band_path_1876, chunks=16)
    assert isinstance(result.data, da_mod.Array)
    assert isinstance(result.data._meta, cupy.ndarray)
    assert result.sizes['band'] == 3

    eager = _read_geotiff_gpu(multi_band_path_1876)
    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(result.compute().data)
    np.testing.assert_allclose(eager_np, chunked_np, rtol=1e-5)


@_gpu_only
def test_read_geotiff_gpu_chunks_single_band_selection_1876(
        multi_band_path_1876):
    """``band=k`` collapses to a 2D Dask+CuPy DataArray."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    result = _read_geotiff_gpu(multi_band_path_1876, chunks=16, band=1)
    assert result.ndim == 2
    assert isinstance(result.data._meta, cupy.ndarray)

    eager = _read_geotiff_gpu(multi_band_path_1876, band=1)
    eager_np = cupy.asnumpy(eager.data)
    chunked_np = cupy.asnumpy(result.compute().data)
    np.testing.assert_allclose(eager_np, chunked_np, rtol=1e-5)


@_gpu_only
def test_read_geotiff_gpu_chunks_fallback_when_kvikio_absent_1876(
        small_raster_path_1876, monkeypatch):
    """When kvikio is reported missing, the chunked path falls back to
    the CPU-decode + cupy.asarray graph."""
    import importlib.util as _ilu

    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gtmod

    original_find_spec = _ilu.find_spec

    def _fake_find_spec(name, *a, **kw):
        if name == 'kvikio':
            return None
        return original_find_spec(name, *a, **kw)

    monkeypatch.setattr(_ilu, 'find_spec', _fake_find_spec)

    direct_calls = {'n': 0}
    real_direct = gtmod._decode_window_gpu_direct

    def _spy(*args, **kwargs):
        direct_calls['n'] += 1
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(gtmod, '_decode_window_gpu_direct', _spy)

    result = _read_geotiff_gpu(small_raster_path_1876, chunks=8)
    computed = result.compute()
    assert direct_calls['n'] == 0
    assert isinstance(computed.data, cupy.ndarray)

    eager = _read_geotiff_gpu(small_raster_path_1876)
    np.testing.assert_array_equal(
        cupy.asnumpy(eager.data), cupy.asnumpy(computed.data),
    )


# ---------------------------------------------------------------------------
# Section: sidecar overview-inheritance georef parity
# ---------------------------------------------------------------------------

def _attrs_subset_2324(da):
    """Pull the georef attrs we want to compare across backends."""
    transform = da.attrs.get("transform", ())
    return {
        "transform": tuple(float(x) for x in transform),
        "crs": da.attrs.get("crs"),
    }


def test_sidecar_without_geokeys_attrs_match_cpu_vs_dask_2324(tmp_path):
    """Baseline: CPU eager and dask agree on inherited georef."""
    from xrspatial.geotiff import _read_geotiff_dask, open_geotiff

    from ..integration.test_sidecar import _write_pair

    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=False,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    dsk = _read_geotiff_dask(str(base), overview_level=1, chunks=2)

    assert _attrs_subset_2324(cpu) == _attrs_subset_2324(dsk)
    t = cpu.attrs["transform"]
    assert t[0] == pytest.approx(20.0)
    assert t[2] == pytest.approx(100.0)
    assert t[4] == pytest.approx(-20.0)
    assert t[5] == pytest.approx(200.0)
    assert cpu.attrs.get("crs") == 4326


@_gpu_only
def test_sidecar_without_geokeys_gpu_matches_cpu_2324(tmp_path):
    """GPU eager georef matches CPU / dask."""
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    from ..integration.test_sidecar import _write_pair

    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=False,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    gpu = _read_geotiff_gpu(str(base), overview_level=1)

    assert _attrs_subset_2324(gpu) == _attrs_subset_2324(cpu)
    np.testing.assert_array_equal(np.asarray(gpu.data.get()),
                                  np.asarray(cpu.data))


@_gpu_only
def test_sidecar_with_own_geokeys_gpu_matches_cpu_2324(tmp_path):
    """GPU path routes a sidecar-owned georef payload to sidecar bytes."""
    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    from ..integration.test_sidecar import _write_pair

    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=True,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    gpu = _read_geotiff_gpu(str(base), overview_level=1)

    assert _attrs_subset_2324(gpu) == _attrs_subset_2324(cpu)
    t = gpu.attrs["transform"]
    assert t[0] == pytest.approx(2.5)
    assert t[2] == pytest.approx(500.0)
    assert t[4] == pytest.approx(-2.5)
    assert t[5] == pytest.approx(800.0)
    assert gpu.attrs.get("crs") == 3857


# ---------------------------------------------------------------------------
# Section: _read_geotiff_gpu accepts HTTP / fsspec URLs (eager)
# ---------------------------------------------------------------------------

class _RangeHandler2161(http.server.BaseHTTPRequestHandler):
    payload: bytes = b''

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            start_s, _, end_s = spec.partition('-')
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}',
            )
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        pass


def _serve_2161(payload: bytes):
    handler_cls = type(
        'RangeHandler2161Bound', (_RangeHandler2161,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


@pytest.fixture
def small_tif_bytes_2161(tmp_path):
    """Build a small valid tiled COG and return its bytes + a reference
    numpy array."""
    from xrspatial.geotiff import to_geotiff
    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) * 0.5
    local = str(tmp_path / "src_2161.tif")
    to_geotiff(arr, local, compression="deflate", tiled=True, tile_size=16)
    with open(local, "rb") as f:
        return f.read(), arr, local


@_gpu_only
def test_local_path_still_returns_cupy_2161(small_tif_bytes_2161):
    """``_read_geotiff_gpu(local.tif)`` must still take the disk-based
    eager path and return a CuPy-backed DataArray."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    _payload, arr_ref, local = small_tif_bytes_2161
    da = _read_geotiff_gpu(local)
    assert isinstance(da.data, cupy.ndarray), (
        f"Local path no longer returns CuPy array (got {type(da.data)})"
    )
    np.testing.assert_array_equal(da.data.get(), arr_ref)


@_gpu_only
@requires_loopback
def test_http_url_returns_cupy_matching_cpu_2161(small_tif_bytes_2161,
                                                 monkeypatch):
    """HTTP URLs route through the CPU decode + GPU upload helper."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    payload, arr_ref, _local = small_tif_bytes_2161
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    httpd, _thread = _serve_2161(payload)
    port = httpd.server_address[1]
    try:
        url = f'http://127.0.0.1:{port}/eager_2161.tif'
        da_cpu = open_geotiff(url)
        da_gpu_direct = _read_geotiff_gpu(url)
        da_gpu_open = open_geotiff(url, gpu=True)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert isinstance(da_gpu_direct.data, cupy.ndarray)
    assert isinstance(da_gpu_open.data, cupy.ndarray)
    np.testing.assert_array_equal(da_gpu_direct.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_open.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_direct.data.get(), arr_ref)


@_gpu_only
def test_memory_fsspec_uri_returns_cupy_matching_cpu_2161(
        small_tif_bytes_2161):
    """An fsspec URI (``memory://``) routes through the CPU read + GPU
    upload helper after the fix."""
    fsspec = pytest.importorskip("fsspec")
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu, open_geotiff

    payload, arr_ref, _local = small_tif_bytes_2161
    fs = fsspec.filesystem("memory")
    mem_path = "/eager_url_2161.tif"
    fs.pipe(mem_path, payload)
    try:
        url = f"memory://{mem_path}"
        da_cpu = open_geotiff(url)
        da_gpu_direct = _read_geotiff_gpu(url)
        da_gpu_open = open_geotiff(url, gpu=True)
    finally:
        try:
            fs.rm(mem_path)
        except FileNotFoundError:
            pass

    assert isinstance(da_gpu_direct.data, cupy.ndarray)
    assert isinstance(da_gpu_open.data, cupy.ndarray)
    np.testing.assert_array_equal(da_gpu_direct.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_open.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_direct.data.get(), arr_ref)


@_gpu_only
@requires_loopback
def test_unreachable_http_url_does_not_raise_filenotfound_2161(monkeypatch):
    """A bad URL no longer surfaces as a bare ``FileNotFoundError``."""
    import socket

    from xrspatial.geotiff import _read_geotiff_gpu

    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    url = f'http://127.0.0.1:{port}/nope_2161.tif'
    with pytest.raises(Exception) as excinfo:
        _read_geotiff_gpu(url)

    err = excinfo.value
    assert not (
        isinstance(err, FileNotFoundError)
        and str(err).rstrip("'").endswith(url)
    ), (
        f"URL surfaces as a bare FileNotFoundError again: {err!r}"
    )

    import urllib3.exceptions as _u3
    assert isinstance(
        err,
        (_u3.MaxRetryError, _u3.HTTPError, ConnectionError, OSError),
    ), (
        f"Unreachable URL surfaced as {type(err).__name__} ({err!r}); "
        f"expected a networking exception."
    )


@_gpu_only
@requires_loopback
def test_chunked_url_path_still_uses_chunked_helper_2161(small_tif_bytes_2161,
                                                         monkeypatch):
    """``chunks=`` on a URL must still go through the chunked helper."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import open_geotiff

    payload, arr_ref, _local = small_tif_bytes_2161
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    httpd, _thread = _serve_2161(payload)
    port = httpd.server_address[1]
    try:
        url = f'http://127.0.0.1:{port}/chunked_2161.tif'
        da_chunked = open_geotiff(url, gpu=True, chunks=16)
        assert isinstance(da_chunked.data, da_mod.Array), (
            "chunks= on a URL must produce a dask-backed result"
        )
        materialised = da_chunked.data.compute()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert isinstance(materialised, cupy.ndarray), (
        "chunked URL path must materialise to CuPy blocks"
    )
    np.testing.assert_array_equal(materialised.get(), arr_ref)


# ---------------------------------------------------------------------------
# Section: GDS chunked path casts each chunk to declared dtype
# ---------------------------------------------------------------------------

def _parse_for_gds_1909(path: str):
    """Return ``(ifd, geo_info, header)`` for the GDS entry point."""
    from xrspatial.geotiff._geotags import extract_geo_info_with_overview_inheritance
    from xrspatial.geotiff._header import parse_all_ifds, parse_header, select_overview_ifd
    from xrspatial.geotiff._reader import _FileSource

    with _FileSource(path) as fs:
        raw = fs.read_all()
    header = parse_header(raw)
    ifds = parse_all_ifds(raw, header)
    ifd = select_overview_ifd(ifds, None)
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, raw, header.byte_order,
    )
    return ifd, geo_info, header


@pytest.fixture
def uint16_no_sentinel_path_1909(tmp_path):
    """A tiled uint16 GeoTIFF with declared nodata=9999 and no sentinel pixels."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    path = tmp_path / "uint16_no_sentinel_1909.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999, tile_size=16)
    return str(path)


@pytest.fixture
def uint16_sentinel_hit_path_1909(tmp_path):
    """A tiled uint16 GeoTIFF whose first chunk hits the nodata sentinel."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    arr[0, 0] = 9999
    path = tmp_path / "uint16_sentinel_hit_1909.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999, tile_size=16)
    return str(path)


@pytest.fixture
def uint16_no_nodata_path_1909(tmp_path):
    """A tiled uint16 GeoTIFF with no declared nodata."""
    from xrspatial.geotiff import to_geotiff

    arr = (np.arange(64, dtype=np.uint16) + 100).reshape(8, 8)
    path = tmp_path / "uint16_no_nodata_1909.tif"
    to_geotiff(arr, str(path), compression="none", tile_size=16)
    return str(path)


@_gpu_only
def test_chunked_gpu_declared_dtype_matches_computed_1909(
        uint16_no_sentinel_path_1909):
    """Declared dask graph dtype must equal the computed chunk dtype."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1909(uint16_no_sentinel_path_1909)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path_1909, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    declared = da.data.dtype
    computed = da.data.compute()
    assert declared == np.float64
    assert declared == computed.dtype, (
        f"GDS chunked path declared {declared} but computed {computed.dtype}; "
        f"this is the issue #1909 silent declared/actual dtype mismatch."
    )


@_gpu_only
def test_chunked_gpu_dtype_matches_cpu_dask_1909(
        uint16_no_sentinel_path_1909):
    """The dask+cupy declared dtype must match the dask+numpy path."""
    from xrspatial.geotiff import _read_geotiff_dask
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    cpu = _read_geotiff_dask(
        uint16_no_sentinel_path_1909, chunks=4, mask_nodata=True)
    ifd, geo_info, header = _parse_for_gds_1909(uint16_no_sentinel_path_1909)
    gpu = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path_1909, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None, mask_nodata=True,
    )
    assert cpu.data.dtype == gpu.data.dtype, (
        f"CPU dask declared {cpu.data.dtype} but GPU dask declared "
        f"{gpu.data.dtype}; backends must agree on graph dtype."
    )


@_gpu_only
def test_chunked_gpu_no_nodata_keeps_source_dtype_1909(
        uint16_no_nodata_path_1909):
    """No nodata declared => declared dtype stays at source dtype."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1909(uint16_no_nodata_path_1909)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_nodata_path_1909, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.uint16
    assert da.data.compute().dtype == np.uint16


@_gpu_only
def test_chunked_gpu_explicit_dtype_kwarg_threads_through_1909(
        uint16_no_sentinel_path_1909):
    """Explicit dtype= overrides nodata promotion and chunks land in it."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1909(uint16_no_sentinel_path_1909)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_no_sentinel_path_1909, ifd, geo_info, header,
        dtype="float32", chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.float32
    assert da.data.compute().dtype == np.float32


@_gpu_only
def test_chunked_gpu_sentinel_hit_still_promotes_1909(
        uint16_sentinel_hit_path_1909):
    """A chunk that hits the sentinel still NaN-masks and lands in float64."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1909(uint16_sentinel_hit_path_1909)
    da = _read_geotiff_gpu_chunked_gds(
        uint16_sentinel_hit_path_1909, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    assert da.data.dtype == np.float64
    computed = da.data.compute()
    assert computed.dtype == np.float64
    host = computed.get()
    assert np.isnan(host[0, 0])


def test_chunked_gpu_eager_paths_keep_source_dtype_1909(
        uint16_no_sentinel_path_1909):
    """Eager paths (no chunks) keep the source uint16 dtype.

    CPU-only so this runs in every CI configuration.
    """
    from xrspatial.geotiff import open_geotiff

    np_da = open_geotiff(uint16_no_sentinel_path_1909)
    assert np_da.dtype == np.uint16
