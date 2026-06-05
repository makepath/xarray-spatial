"""COG writer compliance and overview/nodata combinations.

Covers the COG public API, the external-interop compliance suite
(rasterio / rio-cogeo / GDAL validator), invalid-input errors, the
parity rows that exercise xrspatial-write -> external-read and the
mirror direction, and the tile-layout / tile-size pre-flight gates.

HTTP-side COG tests stay separate with the integration tests.
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import os
import pathlib
import signal
import socketserver
import threading
import uuid
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._errors import ConflictingCRSError
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._writer import write
from xrspatial.geotiff._writer import write as _array_write

from .._helpers.markers import gpu_available, requires_loopback

# -------------------------------------------------------------------------
# Section: COG writer (public API)
# -------------------------------------------------------------------------


class TestCOGWriter:
    def test_cog_layout_ifds_before_data(self, tmp_path):
        """COG spec: all IFDs should come before pixel data."""
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2])

        with open(path, 'rb') as f:
            data = f.read()

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        assert len(ifds) >= 2  # full res + at least 1 overview

        # All IFD offsets should be < the first tile data offset
        all_tile_offsets = []
        for ifd in ifds:
            tile_off = ifd.tile_offsets
            if tile_off:
                all_tile_offsets.extend(tile_off)

        if all_tile_offsets:
            first_data_offset = min(all_tile_offsets)
            # The last IFD byte should be before the first tile data
            # (This is the COG layout requirement)
            assert header.first_ifd_offset < first_data_offset

    def test_cog_round_trip(self, tmp_path):
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'cog_rt.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2])

        result, geo = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)
        assert geo.crs_epsg == 4326

    def test_cog_auto_overviews(self, tmp_path):
        """Auto-generate overviews when none specified."""
        arr = np.arange(1024, dtype=np.float32).reshape(32, 32)
        path = str(tmp_path / 'cog_auto.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True)

        with open(path, 'rb') as f:
            data = f.read()

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # Should have at least 2 IFDs (full res + overviews)
        assert len(ifds) >= 2


class TestPublicAPI:
    def test_read_write_round_trip(self, tmp_path):
        """Write a DataArray, read it back, verify values and coords."""
        y = np.linspace(45.0, 44.0, 10)
        x = np.linspace(-120.0, -119.0, 12)
        data = np.random.RandomState(42).rand(10, 12).astype(np.float32)

        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
            name='test',
        )

        path = str(tmp_path / 'round_trip.tif')
        to_geotiff(da, path, compression='deflate', tiled=False)

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, data, decimal=5)
        assert result.attrs.get('crs') == 4326

    def test_open_geotiff_name(self, tmp_path):
        """DataArray name defaults to filename stem."""
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'myfile.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert da.name == 'myfile'

    def test_open_geotiff_custom_name(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'test.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path, name='custom')
        assert da.name == 'custom'

    def test_write_numpy_array(self, tmp_path):
        """to_geotiff should accept raw numpy arrays too."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'numpy.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_3d_rgb(self, tmp_path):
        """3D arrays (height, width, bands) should write multi-band."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # red channel
        path = str(tmp_path / 'rgb.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_rejects_4d(self, tmp_path):
        arr = np.zeros((2, 3, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))


class TestCOGOverviewResampling:
    """Test overview resampling methods produce correct results."""

    def test_overview_mean(self, tmp_path):
        arr = np.array([[1, 3, 5, 7],
                        [2, 4, 6, 8],
                        [9, 11, 13, 15],
                        [10, 12, 14, 16]], dtype=np.float32)
        path = str(tmp_path / 'cog_1150_mean.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='mean')

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 2
        # Overview should be 2x2
        ov_ifd = ifds[1]
        assert ov_ifd.width == 2
        assert ov_ifd.height == 2

    def test_overview_nearest(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'cog_1150_nearest.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='nearest')

        result, _ = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)

    def test_overview_mode(self, tmp_path):
        # Categorical data: mode should pick the most common value
        arr = np.array([[1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [3, 3, 4, 4],
                        [3, 3, 4, 4]], dtype=np.int32)
        path = str(tmp_path / 'cog_1150_mode.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='mode')

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 2

    @pytest.mark.parametrize('method', ['min', 'max', 'median'])
    def test_overview_other_methods(self, tmp_path, method):
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / f'cog_1150_{method}.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2], overview_resampling=method)

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) >= 2


class TestCOGMultipleOverviews:
    def test_multiple_overview_levels(self, tmp_path):
        """Multiple explicit overview levels produce correct number of IFDs."""
        arr = np.arange(4096, dtype=np.float32).reshape(64, 64)
        path = str(tmp_path / 'cog_1150_multi.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2, 4, 8])

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # Full res + 3 overviews
        assert len(ifds) == 4

    def test_auto_overviews_large_raster(self, tmp_path):
        """Auto-generation on a larger raster produces multiple levels."""
        arr = np.random.RandomState(42).rand(512, 512).astype(np.float32)
        path = str(tmp_path / 'cog_1150_auto_large.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=64,
              cog=True)

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # 512 -> 256 -> 128 -> 64: should stop, so 3 overview levels + full = 4
        assert len(ifds) >= 3

    def test_cog_overview_round_trip_values(self, tmp_path):
        """Full-res values are preserved through COG write with overviews."""
        arr = np.random.RandomState(99).rand(32, 32).astype(np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'cog_1150_rt_values.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='deflate', tiled=True, tile_size=16,
              cog=True, overview_levels=[2, 4])

        result, geo = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)
        assert geo.crs_epsg == 4326


class TestCOGPublicAPIOverviews:
    def test_to_geotiff_cog_with_overviews(self, tmp_path):
        """Public to_geotiff() with cog=True writes overviews."""
        y = np.linspace(45.0, 44.0, 32)
        x = np.linspace(-120.0, -119.0, 32)
        data = np.random.RandomState(42).rand(32, 32).astype(np.float32)

        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )

        path = str(tmp_path / 'cog_1150_api.tif')
        to_geotiff(da, path, compression='deflate', cog=True,
                   tile_size=16, overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, data, decimal=5)

        # Verify COG structure
        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_to_geotiff_cog_auto_overviews(self, tmp_path):
        """Public API auto-generates overviews when only cog=True."""
        data = np.random.RandomState(7).rand(64, 64).astype(np.float32)
        da = xr.DataArray(data, dims=['y', 'x'])

        path = str(tmp_path / 'cog_1150_api_auto.tif')
        to_geotiff(da, path, compression='deflate', cog=True, tile_size=16)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2


_HAS_GPU = gpu_available()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
class TestGPUCOGOverviews:
    """GPU-specific COG overview tests (require CuPy + CUDA)."""

    def test_gpu_cog_round_trip(self, tmp_path):
        import cupy
        arr = np.random.RandomState(42).rand(32, 32).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_rt.tif')
        from xrspatial.geotiff import _write_geotiff_gpu
        _write_geotiff_gpu(gpu_arr, path, crs=4326, compression='deflate',
                           cog=True, overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_gpu_cog_auto_overviews(self, tmp_path):
        import cupy
        arr = np.random.RandomState(7).rand(64, 64).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_auto.tif')
        from xrspatial.geotiff import _write_geotiff_gpu
        _write_geotiff_gpu(gpu_arr, path, compression='deflate',
                           cog=True, tile_size=16)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_gpu_overview_resampling_nearest(self, tmp_path):
        import cupy
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_nearest.tif')
        from xrspatial.geotiff import _write_geotiff_gpu
        _write_geotiff_gpu(gpu_arr, path, compression='deflate',
                           cog=True, overview_levels=[2],
                           overview_resampling='nearest')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_gpu_make_overview_values(self):
        """GPU overview block-reduce matches CPU for simple case."""
        import cupy

        from xrspatial.geotiff._gpu_decode import make_overview_gpu
        from xrspatial.geotiff._writer import _make_overview

        arr = np.random.RandomState(42).rand(16, 16).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        for method in ('mean', 'nearest', 'min', 'max'):
            cpu_ov = _make_overview(arr, method=method)
            gpu_ov = make_overview_gpu(gpu_arr, method=method).get()
            np.testing.assert_allclose(gpu_ov, cpu_ov, rtol=1e-5,
                                       err_msg=f"Mismatch for method={method}")

    def test_gpu_to_geotiff_dispatches_with_overviews(self, tmp_path):
        """to_geotiff auto-dispatches CuPy data with overview params."""
        import cupy
        arr = np.random.RandomState(11).rand(32, 32).astype(np.float32)
        da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'],
                          attrs={'crs': 4326})

        path = str(tmp_path / 'cog_1150_gpu_dispatch.tif')
        to_geotiff(da, path, compression='deflate', cog=True,
                   overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2


def read_to_array_local(path):
    """Helper to call read_to_array for local files."""
    from xrspatial.geotiff._reader import read_to_array
    return read_to_array(path)


# -------------------------------------------------------------------------
# Section: COG external-interop compliance suite
# -------------------------------------------------------------------------

# rasterio is imported per-test below so tests that do not need it are
# still collected when rasterio is absent.


# ---------------------------------------------------------------------------
# Test matrix definitions
# ---------------------------------------------------------------------------

# Stable, lossless codecs only. Each row should produce a byte-for-byte
# round-trip on the base level.
STABLE_CODECS = ["none", "deflate", "lzw", "zstd", "packbits"]

DTYPES = [
    pytest.param(np.uint16, id="uint16"),
    pytest.param(np.float32, id="float32"),
]

BAND_COUNTS = [
    pytest.param(1, id="1band"),
    pytest.param(3, id="3band"),
]

# ``raster_type`` attr the writer understands: ``'area'`` (default) or
# ``'point'``. We pass via attrs because that is the public surface.
GEOREF_MODES = [
    pytest.param("area", id="area"),
    pytest.param("point", id="point"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    dtype: np.dtype,
    *,
    bands: int = 1,
    height: int = 64,
    width: int = 64,
    rng_seed: int = 17,
) -> np.ndarray:
    """Deterministic raster shaped (h, w) or (h, w, bands)."""
    dt = np.dtype(dtype)
    rng = np.random.RandomState(rng_seed + bands)
    if dt.kind == "f":
        base = rng.uniform(-100.0, 100.0, size=(height, width)).astype(dt)
    else:
        info = np.iinfo(dt)
        high = min(info.max, 1000)
        base = rng.randint(0, high, size=(height, width)).astype(dt)
    if bands == 1:
        return base
    # Stack with small per-band offsets so bands are distinguishable.
    layers = [base]
    for b in range(1, bands):
        layers.append((base + b * 7).astype(dt))
    return np.stack(layers, axis=-1)  # (h, w, bands)


def _build_da(
    arr: np.ndarray,
    *,
    raster_type: str = "area",
    crs: int | str | None = 4326,
) -> xr.DataArray:
    """Wrap ``arr`` in a DataArray with EPSG:4326 coords and georef attrs."""
    if arr.ndim == 2:
        h, w = arr.shape
        dims = ("y", "x")
    else:
        h, w, _b = arr.shape
        dims = ("y", "x", "band")
    y = np.linspace(45.0, 44.0, h, dtype=np.float64)
    x = np.linspace(-120.0, -119.0, w, dtype=np.float64)
    coords: dict = {"y": y, "x": x}
    attrs: dict = {}
    if crs is not None:
        attrs["crs"] = crs
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)


def _pick_sentinel(dtype: np.dtype) -> float | int:
    """Pick a nodata sentinel that fits the dtype.

    The signed-int branch is unreachable from the current DTYPES list
    (only ``uint16`` and ``float32``) but is kept for the eventual case
    where the matrix grows. Dead branches in a helper are cheap and the
    intent is clearer than special-casing the current matrix here.
    """
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return -9999.0
    if dt.kind == "u":
        return int(np.iinfo(dt).max)  # e.g. 65535 for uint16
    return int(np.iinfo(dt).min)


def _arrange_for_rasterio(arr: np.ndarray) -> np.ndarray:
    """Convert (h, w[, bands]) into rasterio's (bands, h, w)."""
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    # (h, w, bands) -> (bands, h, w)
    return np.transpose(arr, (2, 0, 1))


def _is_tiled(src) -> bool:
    """Rasterio's ``is_tiled`` is deprecated; reproduce its check locally.

    A dataset is tiled when block dimensions are square and smaller than
    the dataset itself (rasterio's old definition). ``block_shapes`` is
    a per-band list of ``(height, width)`` tuples.
    """
    shapes = src.block_shapes
    if not shapes:
        return False
    bh, bw = shapes[0]
    return bh == bw and bh < src.height and bw < src.width


def _assert_ifds_before_data(path: str) -> None:
    """COG layout contract: every IFD sits before any tile data block."""
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) >= 2, (
        f"expected at least 2 IFDs (full res + overview), got {len(ifds)}"
    )
    tile_offsets: list[int] = []
    for ifd in ifds:
        offs = ifd.tile_offsets
        if offs:
            tile_offsets.extend(offs)
    assert tile_offsets, "no tile offsets found; output is not tiled"
    first_data = min(tile_offsets)
    # All IFD anchors must sit before the first tile blob.
    assert header.first_ifd_offset < first_data, (
        f"first IFD offset {header.first_ifd_offset} >= first tile data "
        f"offset {first_data}; IFDs must come before image data in a COG"
    )


def _require_validator_env() -> bool:
    """Return True if ``XRSPATIAL_REQUIRE_COG_VALIDATOR`` is set truthy.

    Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else, including unset / empty, returns False.

    CI sets this to make a missing validator dependency a hard failure
    rather than a silent skip. On a contributor laptop without rio-cogeo
    or GDAL it is unset and the validator step skips cleanly.
    """
    val = os.environ.get("XRSPATIAL_REQUIRE_COG_VALIDATOR", "")
    return val.lower() in {"1", "true", "yes", "on"}


def _try_cog_validate(path: str) -> None:
    """Call rio-cogeo's validator if present, else GDAL's.

    When ``XRSPATIAL_REQUIRE_COG_VALIDATOR=1`` is set in the environment
    and neither validator is importable, fail loudly instead of skipping
    so a misconfigured CI job cannot pretend the gate passed. When the
    env var is unset, missing dependencies skip cleanly.
    """
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        cog_validate = None  # type: ignore[assignment]

    if cog_validate is not None:
        valid, errors, _warns = cog_validate(path, strict=False)
        assert valid, f"rio_cogeo cog_validate failed: errors={errors}"
        return

    try:
        from osgeo_utils.samples import validate_cloud_optimized_geotiff
    except ImportError:
        if _require_validator_env():
            pytest.fail(
                "XRSPATIAL_REQUIRE_COG_VALIDATOR=1 but neither rio-cogeo "
                "nor GDAL validate_cloud_optimized_geotiff is importable. "
                "Install rio-cogeo (and/or GDAL Python bindings) on this "
                "job, or unset XRSPATIAL_REQUIRE_COG_VALIDATOR to allow "
                "the soft skip."
            )
        pytest.skip(
            "neither rio-cogeo nor GDAL validate_cloud_optimized_geotiff "
            "is installed; skipping external COG validator step"
        )
        return

    _warns, errors, _details = validate_cloud_optimized_geotiff.validate(
        path, full_check=True,
    )
    assert not errors, f"GDAL validator errors: {errors}"


# ---------------------------------------------------------------------------
# Codec x dtype x band-count: base pixels + overviews + georef survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bands", BAND_COUNTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", STABLE_CODECS)
def test_codec_dtype_bands_roundtrip(tmp_path, codec, dtype, bands):
    """Stable codec round-trip via rasterio: base pixels byte-exact, georef survives.

    Contracts asserted per row:
    - rasterio.open succeeds and reports a tiled COG.
    - Band count and dtype survive.
    - Base pixels are byte-exact (stable codecs are lossless).
    - Overview decimation factors survive.
    - CRS and transform survive.
    - IFDs sit before any tile data block (COG layout).
    """
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(dtype, bands=bands, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_codec_{codec}_{np.dtype(dtype).name}_b{bands}.tif")
    to_geotiff(
        da, path,
        compression=codec, cog=True, tile_size=16,
        overview_levels=[2],
    )

    expected = _arrange_for_rasterio(arr)
    with rasterio.open(path) as src:
        assert _is_tiled(src), (
            f"{codec} {dtype} b{bands}: COG output must be tiled"
        )
        assert src.count == bands, (
            f"band count mismatch: expected {bands}, got {src.count}"
        )
        assert src.dtypes == tuple([np.dtype(dtype).name] * bands), (
            f"dtype tuple mismatch: expected "
            f"{tuple([np.dtype(dtype).name] * bands)}, got {src.dtypes}"
        )
        # Stable codecs are lossless -> byte-exact at full resolution.
        actual = src.read()
        assert actual.shape == expected.shape, (
            f"shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"base pixels diverged for codec={codec} dtype={dtype}",
        )
        # Overviews
        for b in range(1, bands + 1):
            ovs = src.overviews(b)
            assert ovs == [2], (
                f"band {b}: expected overview factors [2], got {ovs}"
            )
        # CRS / transform
        assert src.crs is not None and src.crs.to_epsg() == 4326, (
            f"CRS round-trip failed: got {src.crs}"
        )
        assert not src.transform.is_identity, (
            "transform should not be identity for a georeferenced raster"
        )
    # COG layout invariant
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# Nodata: sentinel and NaN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_nodata_sentinel_survives(tmp_path, dtype):
    """Integer and float sentinels survive write -> rasterio.open."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(dtype, bands=1, height=64, width=64)
    sentinel = _pick_sentinel(dtype)
    # Mark a couple of cells as nodata.
    arr_with_nd = arr.copy()
    arr_with_nd[0, 0] = sentinel
    arr_with_nd[5, 7] = sentinel
    da = _build_da(arr_with_nd, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_nodata_sentinel_{np.dtype(dtype).name}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], nodata=sentinel,
    )

    with rasterio.open(path) as src:
        assert src.nodata is not None, "nodata tag not set on output"
        # rasterio normalises to float; compare numerically.
        assert float(src.nodata) == float(sentinel), (
            f"nodata mismatch: expected {sentinel}, got {src.nodata}"
        )
        actual = src.read(1)
        # Byte-exact at base level for deflate.
        np.testing.assert_array_equal(actual, arr_with_nd)


def test_nodata_nan_survives(tmp_path):
    """NaN nodata: NaN positions round-trip as NaN through rasterio."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(np.float32, bands=1, height=64, width=64)
    arr[0, 0] = np.nan
    arr[3, 9] = np.nan
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_nodata_nan.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], nodata=float("nan"),
    )

    with rasterio.open(path) as src:
        assert src.nodata is not None and np.isnan(src.nodata), (
            f"nodata tag should be NaN, got {src.nodata}"
        )
        actual = src.read(1)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(arr))
        finite = ~np.isnan(arr)
        np.testing.assert_array_equal(actual[finite], arr[finite])


# ---------------------------------------------------------------------------
# Georef: PixelIsArea vs PixelIsPoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raster_type", GEOREF_MODES)
def test_raster_type_tag_survives(tmp_path, raster_type):
    """AREA_OR_POINT tag survives to rasterio.tags()."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(np.float32, bands=1, height=32, width=32)
    da = _build_da(arr, raster_type=raster_type, crs=4326)

    path = str(tmp_path / f"2292_georef_{raster_type}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    with rasterio.open(path) as src:
        tag = src.tags().get("AREA_OR_POINT")
        expected_tag = "Area" if raster_type == "area" else "Point"
        assert tag == expected_tag, (
            f"AREA_OR_POINT tag mismatch: expected {expected_tag!r}, "
            f"got {tag!r}"
        )
        # Base values still round-trip exactly.
        np.testing.assert_array_equal(src.read(1), arr)


# ---------------------------------------------------------------------------
# Overviews: explicit list vs auto-generated
# ---------------------------------------------------------------------------


def test_overviews_explicit_levels(tmp_path):
    """``overview_levels=[2, 4, 8]`` produces exactly those decimations."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(np.float32, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_overviews_explicit.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2, 4, 8],
    )

    with rasterio.open(path) as src:
        assert src.overviews(1) == [2, 4, 8], (
            f"expected overviews [2, 4, 8], got {src.overviews(1)}"
        )
        # Each native overview should have the expected shape.
        for lvl, factor in enumerate([2, 4, 8]):
            with rasterio.open(path, OVERVIEW_LEVEL=lvl) as ov:
                exp_h = arr.shape[0] // factor
                exp_w = arr.shape[1] // factor
                assert ov.shape == (exp_h, exp_w), (
                    f"overview {factor}x: expected shape ({exp_h}, {exp_w}), "
                    f"got {ov.shape}"
                )
    _assert_ifds_before_data(path)


@pytest.mark.parametrize("resampling", ["mean", "nearest"])
def test_overview_pixels_match_expected(tmp_path, resampling):
    """Overview pixel values agree with a hand-computed 2x decimation.

    Uses a deterministic base array so we can predict the level-1 overview
    in pure numpy. ``mean`` reduces each 2x2 block to its mean; ``nearest``
    keeps the upper-left pixel of each block. The writer should produce
    overviews that match within float tolerance (lossless codec on the
    base, deterministic block reducer on the overview).
    """
    rasterio = pytest.importorskip("rasterio")
    base = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(base, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_ovpix_{resampling}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], overview_resampling=resampling,
    )

    if resampling == "mean":
        # Block-mean 2x2 -> (32, 32). Promote to float64 for the reduction
        # so the comparison is not biased by float32 round-off in the
        # intermediate sum, then cast back to match what the reader
        # returns.
        b = base.astype(np.float64).reshape(32, 2, 32, 2).mean(axis=(1, 3))
        expected_ov = b.astype(np.float32)
    else:  # nearest
        # Upper-left pixel of each 2x2 block.
        expected_ov = base[::2, ::2]

    with rasterio.open(path, OVERVIEW_LEVEL=0) as ov:
        actual = ov.read(1)
    assert actual.shape == expected_ov.shape, (
        f"{resampling}: expected overview shape {expected_ov.shape}, "
        f"got {actual.shape}"
    )
    # Tolerance: the writer's mean reducer accumulates in float64 internally
    # but the on-disk result is float32; comparing against our hand-computed
    # float32 expected leaves <= 1 ULP of slack per cell.
    np.testing.assert_allclose(
        actual, expected_ov, rtol=1e-5, atol=1e-5,
        err_msg=f"{resampling} overview pixels diverged from expected",
    )


def test_overviews_auto_generated(tmp_path):
    """``overview_levels=None`` with cog=True auto-generates a pyramid."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(np.float32, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_overviews_auto.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=32,
    )

    with rasterio.open(path) as src:
        ovs = src.overviews(1)
        assert len(ovs) >= 1, f"expected at least one overview, got {ovs}"
        # Auto-generated pyramid: every level is a power of two, strictly
        # increasing, and large enough that the next halving would not fall
        # below the tile_size of 32. The bitwise test below is the classic
        # power-of-two check: ``o & (o - 1) == 0`` is True iff ``o`` has a
        # single set bit. The ``o >= 2`` guard rules out the false-positive
        # at ``o == 0``.
        assert all((o & (o - 1)) == 0 and o >= 2 for o in ovs), (
            f"auto overviews should be powers of two >= 2, got {ovs}"
        )
        assert all(b > a for a, b in zip(ovs, ovs[1:])), (
            f"auto overviews not strictly increasing: {ovs}"
        )
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# TIFF layout sanity: tiled, sane tile offsets, IFDs before data
# ---------------------------------------------------------------------------


def test_layout_is_cog_shaped(tmp_path):
    """A cog=True file is tiled, has overview IFDs, and IFDs precede data."""
    rasterio = pytest.importorskip("rasterio")
    arr = _make_data(np.uint16, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_layout.tif")
    to_geotiff(
        da, path,
        compression="lzw", cog=True, tile_size=32,
        overview_levels=[2, 4],
    )

    with rasterio.open(path) as src:
        assert _is_tiled(src), "COG output must be tiled, got stripped layout"
        assert src.block_shapes[0] == (32, 32), (
            f"unexpected block shape: {src.block_shapes}"
        )

    # All IFDs come before image data; tile offsets are monotonic-ish
    # (not strictly monotonic across IFDs but every offset must point inside
    # the file).
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) == 3, (
        f"expected 3 IFDs (full + 2 overviews), got {len(ifds)}"
    )
    file_len = len(data)
    for ifd in ifds:
        for off in (ifd.tile_offsets or ()):
            assert 0 <= off < file_len, (
                f"tile offset {off} outside file bounds [0, {file_len})"
            )
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# Optional external validator
# ---------------------------------------------------------------------------


def test_external_cog_validator(tmp_path):
    """Run rio-cogeo / GDAL's COG validator if available, else skip cleanly."""
    arr = _make_data(np.float32, bands=1, height=256, width=256)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_validator.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=64,
        overview_levels=[2, 4],
    )

    _try_cog_validate(path)


# ---------------------------------------------------------------------------
# Validator-mode env contract
# ---------------------------------------------------------------------------


def test_require_validator_env_strict_fails_when_dep_missing(
    tmp_path, monkeypatch,
):
    """``XRSPATIAL_REQUIRE_COG_VALIDATOR=1`` must fail (not skip) if both
    validators are absent.

    This guards the CI gate: if the install step silently drops rio-cogeo
    or GDAL, the compliance suite must fail rather than skip past the
    validator step. Stub both imports as ``ImportError`` so the test runs
    the same on every job, validator-present or not.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        fl = tuple(fromlist) if fromlist else ()
        rio_match = (
            name == "rio_cogeo.cogeo" and "cog_validate" in fl
        )
        gdal_match = (
            name == "osgeo_utils.samples"
            and "validate_cloud_optimized_geotiff" in fl
        )
        if rio_match or gdal_match:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", "1")

    arr = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)
    path = str(tmp_path / "2302_require_strict.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    # ``pytest.fail.Exception`` is a documented alias for
    # ``_pytest.outcomes.Failed`` on pytest >= 7 (which this repo pins
    # via setup.cfg). Update both spots in this file if that pin moves.
    with pytest.raises(pytest.fail.Exception, match="XRSPATIAL_REQUIRE_COG_VALIDATOR"):
        _try_cog_validate(path)


def test_require_validator_env_unset_skips_when_dep_missing(
    tmp_path, monkeypatch,
):
    """With the env var unset, missing validators trigger a clean skip.

    This is the contributor-laptop path: no rio-cogeo / GDAL installed,
    the compliance suite still passes without the optional validator
    step.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        fl = tuple(fromlist) if fromlist else ()
        rio_match = (
            name == "rio_cogeo.cogeo" and "cog_validate" in fl
        )
        gdal_match = (
            name == "osgeo_utils.samples"
            and "validate_cloud_optimized_geotiff" in fl
        )
        if rio_match or gdal_match:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.delenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", raising=False)

    arr = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)
    path = str(tmp_path / "2302_require_unset.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    with pytest.raises(pytest.skip.Exception):
        _try_cog_validate(path)


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_require_validator_env_truthy_values(monkeypatch, val):
    """All documented truthy spellings activate strict mode."""
    monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", val)
    assert _require_validator_env() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "anything"])
def test_require_validator_env_non_truthy_values(monkeypatch, val):
    """Empty or non-truthy spellings leave strict mode off."""
    if val == "":
        monkeypatch.delenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", raising=False)
    else:
        monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", val)
    assert _require_validator_env() is False


# -------------------------------------------------------------------------
# Section: COG invalid-input errors
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_da_small(shape=(8, 8)):
    """A small float32 DataArray suitable for COG writes."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=('y', 'x')
    )


def _uint8_da(shape=(8, 8)):
    """A small uint8 DataArray (JPEG is uint8-only)."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.uint8), dims=('y', 'x')
    )


# ---------------------------------------------------------------------------
# Row 1: Experimental codec without ``allow_experimental_codecs=True``
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('codec', ['lerc', 'lz4', 'jpeg2000', 'j2k'])
def test_experimental_codec_without_opt_in_raises(tmp_path, codec):
    """Experimental codecs are gated; the message names the codec and
    the opt-in flag, and mentions the experimental tier so the caller
    knows why the default refuses the input."""
    da = _float_da_small()
    p = tmp_path / f'cog_exp_codec_{codec}_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, compression=codec)

    msg = str(exc.value)
    assert codec in msg, msg
    assert 'allow_experimental_codecs' in msg, msg
    assert 'experimental' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Row 2: Internal-only JPEG without ``allow_internal_only_jpeg=True``
# ---------------------------------------------------------------------------

def test_internal_only_jpeg_without_opt_in_raises(tmp_path):
    """``compression='jpeg'`` is rejected by default; the message names
    the codec, the opt-in flag, and explains the interop break."""
    da = _uint8_da()
    p = tmp_path / 'cog_jpeg_no_optin_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, compression='jpeg')

    msg = str(exc.value)
    assert 'jpeg' in msg.lower(), msg
    assert 'allow_internal_only_jpeg' in msg, msg


def test_internal_only_jpeg_not_covered_by_experimental_flag(tmp_path):
    """``allow_experimental_codecs=True`` does not cover JPEG. The two
    flags are deliberately separate (internal-only is stricter than
    experimental) so a caller cannot reach the JFIF path by toggling
    only the experimental switch."""
    da = _uint8_da()
    p = tmp_path / 'cog_jpeg_exp_flag_only_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True,
                   compression='jpeg',
                   allow_experimental_codecs=True)

    msg = str(exc.value)
    assert 'jpeg' in msg.lower(), msg
    assert 'allow_internal_only_jpeg' in msg, msg


# ---------------------------------------------------------------------------
# Row 3: Rotated transform on input DataArray
# ---------------------------------------------------------------------------

def test_rotated_affine_attr_without_drop_rotation_raises(tmp_path):
    """The reader stamps ``attrs['rotated_affine']`` when called with
    ``allow_rotated=True``. Writing such a DataArray without
    ``drop_rotation=True`` would silently produce an identity-affine
    output, so the entry point refuses up front."""
    da = _float_da_small()
    da.attrs['rotated_affine'] = (1.0, 0.5, 0.0, 0.0, 0.5, 1.0)
    p = tmp_path / 'cog_rotated_affine_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotated_affine' in msg, msg
    assert 'drop_rotation' in msg, msg


def test_rotated_affine_attr_drop_rotation_opt_in_succeeds(tmp_path):
    """The opt-in path lets the write proceed (lossy but explicit).
    Pinned here so the rejection-message test cannot be 'fixed' by
    removing the opt-in entirely."""
    da = _float_da_small()
    da.attrs['rotated_affine'] = (1.0, 0.5, 0.0, 0.0, 0.5, 1.0)
    p = tmp_path / 'cog_rotated_affine_optin_2301.tif'

    to_geotiff(da, str(p), cog=True, drop_rotation=True)
    assert p.exists()
    assert p.stat().st_size > 0


def test_rotated_transform_tuple_attr_raises(tmp_path):
    """``attrs['transform']`` as a 6-tuple ``(a, b, c, d, e, f)`` with
    non-zero rotation/shear (``b`` or ``d``) is refused by
    ``transform_from_attr``. The message names the rotation/shear
    constraint and the axis-aligned requirement."""
    da = _float_da_small()
    da.attrs['transform'] = (1.0, 0.5, 0.0, 0.0, -1.0, 4.0)  # b = 0.5
    p = tmp_path / 'cog_rotated_tuple_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg
    assert 'axis-aligned' in msg, msg


def test_rotated_transform_affine_attr_raises(tmp_path):
    """``attrs['transform']`` as a rasterio ``Affine`` with non-zero
    rotation/shear used to slip past the 6-tuple gate because
    ``Affine`` iterates as a 9-element augmented matrix. The
    validation hook detects the Affine duck-type and raises the same
    diagnostic the 6-tuple branch already produced."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da_small()
    da.attrs['transform'] = Affine(1.0, 0.5, 0.0, 0.0, -1.0, 4.0)  # b = 0.5
    p = tmp_path / 'cog_rotated_affine_obj_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg
    assert 'axis-aligned' in msg, msg


def test_skewed_transform_affine_attr_raises(tmp_path):
    """The ``d`` shear term (Affine's third row, first column) is also
    rejected. Same validator path as ``b != 0``; pinned separately so a
    refactor that only covers ``b`` is caught."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da_small()
    da.attrs['transform'] = Affine(1.0, 0.0, 0.0, 0.3, -1.0, 4.0)  # d = 0.3
    p = tmp_path / 'cog_skewed_affine_obj_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg


def test_affine_attr_with_unconvertable_b_d_raises(tmp_path):
    """An attrs['transform'] object that quacks like an Affine (has
    ``.b`` and ``.d``) but carries non-numeric values for them is
    refused with a clear ``ValueError``. The fail-closed branch
    prevents a malformed input from bypassing the rotation/shear gate
    and falling through to the no-georef path."""
    class _BogusAffine:
        b = "not a number"
        d = 0.0
    da = _float_da_small()
    da.attrs['transform'] = _BogusAffine()
    p = tmp_path / 'cog_bogus_affine_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'unconvertable' in msg or 'rotation/shear' in msg, msg


def test_axis_aligned_affine_attr_still_writes(tmp_path):
    """Sanity guard: an axis-aligned Affine (b=d=0) must keep working.
    Without this row the validation hook could regress every legitimate
    Affine call site by widening the rejection bucket."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da_small()
    da.attrs['transform'] = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)  # b=d=0
    p = tmp_path / 'cog_axis_aligned_affine_2301.tif'

    to_geotiff(da, str(p), cog=True)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Row 4: File-like / BytesIO destination with ``cog=True``
# ---------------------------------------------------------------------------

def test_bytesio_destination_with_cog_raises():
    """COG output needs a real filesystem path because the writer runs
    a second pass to populate overview offsets. ``to_geotiff`` rejects
    file-like destinations with ``cog=True`` up front."""
    da = _float_da_small()
    buf = io.BytesIO()

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, buf, cog=True)

    msg = str(exc.value)
    assert 'cog' in msg.lower(), msg
    assert 'file-like' in msg or 'string path' in msg, msg


def test_bytesio_destination_without_cog_still_works():
    """Sanity guard: BytesIO is fine for plain TIFF writes. Pinned so
    the COG-only rejection cannot regress into a blanket file-like
    refusal."""
    da = _float_da_small()
    buf = io.BytesIO()

    to_geotiff(da, buf, cog=False)
    assert buf.tell() > 0


# ---------------------------------------------------------------------------
# Row 5: CuPy / GPU-backed array with ``cog=True``
# ---------------------------------------------------------------------------

def test_cupy_input_with_cog_currently_succeeds(tmp_path):
    """The GPU writer currently produces a valid COG for CuPy input;
    GPU COG is documented as Experimental in the docstring tier map
    but is not refused at the entry point. This row pins the
    currently-succeeds behaviour so a future tier-promotion change
    does not silently break callers that already rely on the path.

    No production-side validation hook is added here because the
    constraint is 'do not change semantics on paths
    that currently succeed'."""
    if importlib.util.find_spec('cupy') is None:
        pytest.skip('cupy not installed')
    try:
        import cupy as cp
        if not cp.cuda.is_available():
            pytest.skip('CUDA device not available')
    except Exception as exc:
        pytest.skip(f'cupy import failed: {exc}')

    da = xr.DataArray(cp.zeros((8, 8), dtype=cp.float32), dims=('y', 'x'))
    p = tmp_path / 'cog_cupy_2301.tif'

    # No exception; produces a real file. If a future PR tightens the
    # GPU COG tier this assertion will start failing and the next
    # reviewer can decide whether to flip this to a ``pytest.raises``.
    to_geotiff(da, str(p), cog=True)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Row 6: Object-dtype DataArray
# ---------------------------------------------------------------------------

def test_object_dtype_with_cog_raises(tmp_path):
    """Object dtype is not a TIFF sample format. ``numpy_to_tiff_dtype``
    raises ``ValueError`` naming the dtype, so the writer surfaces a
    typed error rather than a deep struct-pack traceback."""
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=object), dims=('y', 'x'))
    p = tmp_path / 'cog_object_dtype_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'object' in msg.lower() or 'dtype' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Row 7: Conflicting ``crs=`` kwarg / array CRS
# ---------------------------------------------------------------------------

def test_conflicting_attrs_crs_and_crs_wkt_raises(tmp_path):
    """When ``attrs['crs']`` and ``attrs['crs_wkt']`` resolve to
    different CRSes via pyproj, the writer refuses with
    ``ConflictingCRSError``. This confirms the message stays
    actionable; it does not introduce a new check."""
    pytest.importorskip('pyproj')
    wkt_3857 = (
        'PROJCS["WGS 84 / Pseudo-Mercator",'
        'GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Mercator_1SP"],'
        'PARAMETER["central_meridian",0],'
        'PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1],'
        'AUTHORITY["EPSG","3857"]]'
    )
    da = _float_da_small()
    da.attrs['crs'] = 4326
    da.attrs['crs_wkt'] = wkt_3857
    p = tmp_path / 'cog_conflicting_crs_2301.tif'

    with pytest.raises(ConflictingCRSError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    # Message names both inputs and the resolution hint.
    assert "attrs['crs']" in msg, msg
    assert "attrs['crs_wkt']" in msg, msg
    # Caller-actionable: tells the user to reconcile the two attrs.
    assert 'Reconcile' in msg or 'reconcile' in msg, msg


def test_crs_kwarg_overrides_attrs_silently(tmp_path):
    """``crs=`` kwarg overrides the attrs disagreement. The
    ``_check_write_conflicting_crs`` short-circuit at the top of the
    check (``if context.get('crs_kwarg') is not None: return``) lets
    the write proceed even when the two attrs would otherwise
    disagree, so callers can intentionally use the kwarg to clobber
    stale attrs. Pinned here so a future 'stricter' rewrite of the
    conflict check that drops the short-circuit does not surprise
    those callers."""
    pytest.importorskip('pyproj')
    da = _float_da_small()
    da.attrs['crs'] = 4326
    # ``crs_wkt`` value is irrelevant: the check short-circuits on the
    # kwarg before pyproj parsing ever runs.
    da.attrs['crs_wkt'] = 'GEOGCS["foo"]'
    p = tmp_path / 'cog_crs_kwarg_override_2301.tif'

    to_geotiff(da, str(p), cog=True, crs=3857)
    assert p.exists()
    assert p.stat().st_size > 0


# -------------------------------------------------------------------------
# Section: COG parity rows
# -------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------


_HAS_DASK = importlib.util.find_spec("dask") is not None


def _require_dask() -> None:
    if not _HAS_DASK:
        pytest.skip(
            "dask is not installed; install the dask extra to exercise "
            "the COG dask-read row of the #2286 release gate."
        )


# Golden corpus COG fixture: tiled, internal overviews, written via
# GDAL's COG driver. Lives under ``golden_corpus/fixtures``.
_GOLDEN_COG_ID = "cog_internal_overview_uint16"


def _golden_cog_path() -> pathlib.Path:
    from xrspatial.geotiff.tests.golden_corpus import generate
    return (
        pathlib.Path(generate.__file__).resolve().parent
        / "fixtures"
        / f"{_GOLDEN_COG_ID}.tif"
    )


# ---------------------------------------------------------------------------
# Range-aware in-process HTTP server (mirrors the pattern used by
# test_cog_http_parallel_decode_2026_05_15.py and test_cog_http_concurrent.py).
# ---------------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Range",
                f"bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}",
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):  # silence test noise
        return


def _serve_payload(payload: bytes, monkeypatch):
    """Spin a range-aware server bound to localhost; return (httpd, port).

    The handler subclass is named with a uuid suffix so that the two
    fixtures in this module (and any future ones) don't share a
    qualname. Without the suffix, tracebacks reuse the same class
    identifier across fixture invocations and become harder to read.

    ``allow_reuse_address = True`` lets the OS reclaim the port
    quickly when the test tears down (avoiding TIME_WAIT-related
    binding races under parallel pytest runs). ``timeout=5`` on the
    server caps how long a stuck request can pin the daemon thread.
    """
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    handler_cls = type(
        f"RangeHandler2286_{uuid.uuid4().hex[:8]}",
        (_RangeHandler,),
        {"payload": payload},
    )

    class _ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        timeout = 5

    httpd = _ReusableTCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xrspatial_cog(tmp_path):
    """xrspatial writes a small lossless COG; yield (path, source_array, attrs).

    The source is a deterministic uint16 ramp so byte-exact comparison
    is meaningful. CRS / transform / nodata are stamped via the public
    ``to_geotiff`` API so the round trip exercises the user-visible
    surface, not a private writer entry point.
    """
    h, w = 64, 64
    # Use a +1 offset so pixel value 0 never appears -- the reader
    # masks nodata-valued pixels to NaN,
    # which upcasts integer rasters to float64. The fixture's payload
    # is a deterministic ramp regardless of the offset.
    data = (np.arange(h * w, dtype=np.uint16) + 1).reshape(h, w)
    # Build a DataArray with a real CRS and a regular grid so the
    # transform is non-degenerate. Pixel size 0.01 deg.
    y = np.linspace(45.0, 45.0 - 0.01 * (h - 1), h)
    x = np.linspace(-120.0, -120.0 + 0.01 * (w - 1), w)
    da = xr.DataArray(
        data, dims=["y", "x"],
        coords={"y": y, "x": x},
        # No ``nodata`` attr: the masked-nodata path upcasts integer
        # rasters to float64 and replaces sentinel pixels with NaN,
        # which would break the byte-exact uint16 comparison. The
        # nodata read contract is exercised separately under
        # ``test_nodata_lifecycle_parity_2211.py``.
        attrs={"crs": 4326},
        name="cog_2286",
    )
    path = str(tmp_path / "xrspatial_cog_2286.tif")
    to_geotiff(
        da, path,
        compression="deflate",
        tiled=True,
        tile_size=16,
        cog=True,
        overview_levels=[2],
    )
    return path, data, {"crs": 4326, "nodata": None}


@pytest.fixture
def golden_cog_http(monkeypatch):
    """Serve the golden COG fixture over a range-aware in-process HTTP server.

    Yields ``(url, expected_array)`` where ``expected_array`` is the
    pixels read via the local xrspatial reader (the ground truth for
    HTTP comparison). The fixture lives in the golden corpus and was
    written by GDAL's COG driver, so it stresses the third-party
    interop side of the COG read path.
    """
    path = _golden_cog_path()
    if not path.exists():
        pytest.skip(
            f"golden COG fixture {_GOLDEN_COG_ID!r} missing on disk; run "
            "`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            "to materialise the corpus (issue #1930)."
        )
    with open(path, "rb") as f:
        payload = f.read()
    httpd, port = _serve_payload(payload, monkeypatch)
    try:
        # Use a stable filename in the URL so the SSRF-hardened reader
        # has a sensible-looking path to log.
        yield f"http://127.0.0.1:{port}/{_GOLDEN_COG_ID}.tif", path
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Host-side numpy view (dask compute, cupy get) without leaking lazy state."""
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _assert_byte_exact(
    expected: np.ndarray, actual: np.ndarray, *, label: str,
) -> None:
    """Byte-exact equality on shape, dtype, and bytes. Every fixture is lossless."""
    assert expected.shape == actual.shape, (
        f"{label}: shape mismatch expected={expected.shape} actual={actual.shape}"
    )
    assert expected.dtype == actual.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} actual={actual.dtype}"
    )
    if not np.array_equal(expected, actual):
        diff = np.where(expected != actual)
        n = len(diff[0])
        raise AssertionError(
            f"{label}: byte-exact comparison failed; {n} pixel(s) differ"
        )


# Scope note: every fixture in this file is single-band 2D. The two
# helpers below hard-code that shape on purpose. If a future row adds
# a multi-band fixture, extend the helpers (or replace them with
# parametrised checks) rather than reusing them as-is.

def _assert_dim_names(da: xr.DataArray, *, label: str) -> None:
    """The 2D COG path must come back with ``(y, x)`` dim names."""
    assert da.dims == ("y", "x"), (
        f"{label}: dims must be ('y', 'x'), got {da.dims!r}"
    )


def _assert_band_count(arr: np.ndarray, *, label: str) -> None:
    """Single-band fixture; the returned array must be 2D."""
    assert arr.ndim == 2, (
        f"{label}: expected single-band 2D pixels, got ndim={arr.ndim} "
        f"shape={arr.shape}"
    )


def _assert_crs_present(da: xr.DataArray, *, label: str) -> None:
    """``crs`` (EPSG int or string) or ``crs_wkt`` must survive the read."""
    has_crs = "crs" in da.attrs and da.attrs["crs"] is not None
    has_wkt = "crs_wkt" in da.attrs and da.attrs["crs_wkt"]
    assert has_crs or has_wkt, (
        f"{label}: neither 'crs' nor 'crs_wkt' attr survived the read; "
        f"attrs={sorted(da.attrs)!r}"
    )


def _assert_crs_equals(da: xr.DataArray, expected_epsg: int, *, label: str) -> None:
    """Read-side CRS matches the writer's EPSG declaration."""
    crs = da.attrs.get("crs")
    assert crs == expected_epsg, (
        f"{label}: crs mismatch expected={expected_epsg!r} got={crs!r}"
    )


def _assert_transform(da: xr.DataArray, *, label: str) -> None:
    """Transform attr present and a finite 6-tuple."""
    t = da.attrs.get("transform")
    assert t is not None, f"{label}: transform attr missing"
    tup = tuple(float(v) for v in t)
    assert len(tup) == 6, f"{label}: transform must be a 6-tuple, got {tup}"
    assert all(np.isfinite(v) for v in tup), (
        f"{label}: transform has non-finite component: {tup}"
    )


def _assert_transform_equals(
    da: xr.DataArray, expected_t: tuple, *, label: str,
) -> None:
    """Transform 6-tuple matches an expected reference within a tight ULP."""
    t = da.attrs.get("transform")
    assert t is not None, f"{label}: transform attr missing"
    tup = tuple(float(v) for v in t)
    exp = tuple(float(v) for v in expected_t)
    assert len(tup) == 6 and len(exp) == 6
    for i, (a, b) in enumerate(zip(tup, exp)):
        assert abs(a - b) <= 1e-9, (
            f"{label}: transform[{i}] differs expected={b!r} got={a!r}"
        )


def _assert_nodata_equals(
    da: xr.DataArray, expected: float | int | None, *, label: str,
) -> None:
    """Assert nodata sentinel matches, including the no-nodata case.

    When ``expected`` is ``None`` we still check the read side: the
    reader must not fabricate a sentinel that the writer never stamped.
    The reader is allowed to expose the attr as ``None`` or omit it
    entirely; both count as "no nodata".
    """
    nd = da.attrs.get("nodata")
    if expected is None:
        assert nd is None, (
            f"{label}: writer stamped no nodata, but reader exposed "
            f"nodata={nd!r}"
        )
        return
    assert nd == expected, (
        f"{label}: nodata mismatch expected={expected!r} got={nd!r}"
    )


# ---------------------------------------------------------------------------
# Row 1: xrspatial write COG -> xrspatial eager read
# ---------------------------------------------------------------------------

def test_row1_xrspatial_cog_xrspatial_eager(xrspatial_cog):
    """xrspatial-written COG round-trips byte-exact through the eager reader."""
    path, expected, expected_attrs = xrspatial_cog
    da = open_geotiff(path)
    label = "row1_xrspatial_cog_eager"

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    _assert_dim_names(da, label=label)
    _assert_crs_equals(da, expected_attrs["crs"], label=label)
    _assert_transform(da, label=label)
    _assert_nodata_equals(da, expected_attrs["nodata"], label=label)
    assert da.dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={da.dtype}"
    )


# ---------------------------------------------------------------------------
# Row 2: xrspatial write COG -> xrspatial dask read
# ---------------------------------------------------------------------------

def test_row2_xrspatial_cog_xrspatial_dask(xrspatial_cog):
    """xrspatial-written COG round-trips byte-exact through the dask reader."""
    _require_dask()
    path, expected, expected_attrs = xrspatial_cog
    da = open_geotiff(path, chunks=16)
    label = "row2_xrspatial_cog_dask"

    # Verify we actually went through the dask path; a regression that
    # silently drops ``chunks=`` and falls back to eager would pass the
    # pixel check but exercise the wrong code path.
    assert hasattr(da.data, "dask"), (
        f"{label}: chunks=16 did not produce a dask-backed DataArray; "
        f"got data type {type(da.data).__name__}"
    )

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    _assert_dim_names(da, label=label)
    _assert_crs_equals(da, expected_attrs["crs"], label=label)
    _assert_transform(da, label=label)
    _assert_nodata_equals(da, expected_attrs["nodata"], label=label)
    assert da.dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={da.dtype}"
    )


# ---------------------------------------------------------------------------
# Row 3: xrspatial write COG -> rasterio read
# ---------------------------------------------------------------------------

def test_row3_xrspatial_cog_rasterio(xrspatial_cog):
    """rasterio reads an xrspatial-written COG and the pixel/metadata contract holds.

    Asserts the third-party reader sees the same pixels, dtype, CRS,
    transform, and nodata that xrspatial stamped on write. A regression
    that drops or mangles any of these would surface as a Tier-1
    interop break.
    """
    rasterio = pytest.importorskip(
        "rasterio",
        reason="rasterio is required for row 3 (issue #2294)",
    )
    path, expected, expected_attrs = xrspatial_cog
    label = "row3_xrspatial_cog_rasterio"

    with rasterio.open(path) as src:
        # Single-band fixture: read band 1.
        pixels = src.read(1)
        rio_crs = src.crs
        rio_transform = src.transform
        rio_nodata = src.nodata
        rio_count = src.count
        rio_dtype = np.dtype(src.dtypes[0])

    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    assert rio_count == 1, f"{label}: rasterio reports band count {rio_count}"
    assert rio_dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={rio_dtype}"
    )
    # rasterio CRS -> EPSG int when possible.
    epsg = rio_crs.to_epsg() if rio_crs is not None else None
    assert epsg == expected_attrs["crs"], (
        f"{label}: rasterio CRS EPSG mismatch "
        f"expected={expected_attrs['crs']!r} got={epsg!r}"
    )
    # rasterio Affine is 6-tuple compatible via ``.a, .b, .c, .d, .e, .f``.
    assert rio_transform is not None, f"{label}: rasterio transform missing"
    assert all(np.isfinite(v) for v in (
        rio_transform.a, rio_transform.b, rio_transform.c,
        rio_transform.d, rio_transform.e, rio_transform.f,
    )), f"{label}: rasterio transform has non-finite component"
    if expected_attrs["nodata"] is None:
        # The writer was not asked to stamp a nodata; rasterio should
        # report ``None`` too. Anything else means the writer leaked
        # a sentinel onto the file.
        assert rio_nodata is None, (
            f"{label}: writer stamped an unrequested nodata; "
            f"rasterio reports {rio_nodata!r}"
        )
    else:
        assert rio_nodata == expected_attrs["nodata"], (
            f"{label}: rasterio nodata mismatch "
            f"expected={expected_attrs['nodata']!r} got={rio_nodata!r}"
        )


# ---------------------------------------------------------------------------
# Row 4: golden/rasterio COG fixture -> xrspatial local read
# ---------------------------------------------------------------------------

def test_row4_golden_cog_xrspatial_local():
    """Read the GDAL-written golden COG fixture with xrspatial's local reader.

    Compares pixels byte-exact against a rasterio read of the same
    bytes -- the GDAL COG driver wrote the file, so rasterio is the
    canonical oracle here. Catches regressions that returned the right
    shape but mangled values (e.g. wrong endianness, predictor drift,
    overview IFD picked instead of full res).
    """
    rasterio = pytest.importorskip(
        "rasterio",
        reason="rasterio is required for row 4 oracle (issue #2294)",
    )
    path = _golden_cog_path()
    if not path.exists():
        pytest.skip(
            f"golden COG fixture {_GOLDEN_COG_ID!r} missing on disk; run "
            "`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            "(issue #1930)."
        )
    da = open_geotiff(str(path))
    label = "row4_golden_cog_xrspatial_local"

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_dim_names(da, label=label)
    # The golden fixture is uint16 per the manifest entry.
    assert da.dtype == np.dtype("uint16"), (
        f"{label}: dtype expected=uint16 got={da.dtype}"
    )
    _assert_crs_present(da, label=label)
    _assert_transform(da, label=label)

    # Pixel parity against the rasterio oracle. The fixture is lossless
    # deflate, so byte-exact is the right bar.
    with rasterio.open(str(path)) as src:
        expected = src.read(1)
    _assert_byte_exact(expected, pixels, label=label)


# ---------------------------------------------------------------------------
# Row 5: golden/rasterio COG fixture -> xrspatial HTTP range read
# ---------------------------------------------------------------------------

@requires_loopback
def test_row5_golden_cog_xrspatial_http(golden_cog_http):
    """xrspatial's HTTP range reader returns the same pixels as the local read.

    Exercises the cloud-source code path against the GDAL-written
    fixture. The reference is the local read of the same bytes, so any
    drift between the local and HTTP paths surfaces here.
    """
    url, local_path = golden_cog_http
    label = "row5_golden_cog_xrspatial_http"

    local_da = open_geotiff(str(local_path))
    http_da = open_geotiff(url)

    local_px = _materialise(local_da)
    http_px = _materialise(http_da)

    _assert_band_count(http_px, label=label)
    _assert_byte_exact(local_px, http_px, label=label)
    _assert_dim_names(http_da, label=label)
    assert http_da.dtype == local_da.dtype, (
        f"{label}: dtype mismatch local={local_da.dtype} http={http_da.dtype}"
    )
    # CRS and transform survive the cloud-source path.
    local_crs = local_da.attrs.get("crs")
    http_crs = http_da.attrs.get("crs")
    assert local_crs == http_crs, (
        f"{label}: crs mismatch local={local_crs!r} http={http_crs!r}"
    )
    local_t = local_da.attrs.get("transform")
    assert local_t is not None, f"{label}: local read missing transform"
    _assert_transform_equals(http_da, local_t, label=label)
    # nodata presence must agree (the fixture may or may not carry one;
    # both sides must agree either way).
    assert ("nodata" in local_da.attrs) == ("nodata" in http_da.attrs), (
        f"{label}: nodata presence differs "
        f"local={'nodata' in local_da.attrs} http={'nodata' in http_da.attrs}"
    )
    if "nodata" in local_da.attrs:
        assert local_da.attrs["nodata"] == http_da.attrs["nodata"], (
            f"{label}: nodata value differs "
            f"local={local_da.attrs['nodata']!r} "
            f"http={http_da.attrs['nodata']!r}"
        )


# ---------------------------------------------------------------------------
# Row 6: golden/rasterio COG fixture -> xrspatial dask HTTP range read
# ---------------------------------------------------------------------------

@requires_loopback
def test_row6_golden_cog_xrspatial_dask_http(golden_cog_http):
    """The dask HTTP path returns the same pixels as the local read.

    Combines the cloud-source and chunked-read code paths. A regression
    that silently drops ``chunks=`` over HTTP would compute correct
    pixels via the eager path; the storage-type assertion below guards
    against that.
    """
    _require_dask()
    url, local_path = golden_cog_http
    label = "row6_golden_cog_xrspatial_dask_http"

    local_da = open_geotiff(str(local_path))
    http_da = open_geotiff(url, chunks=16)

    assert hasattr(http_da.data, "dask"), (
        f"{label}: chunks=16 over HTTP did not produce a dask-backed "
        f"DataArray; got data type {type(http_da.data).__name__}"
    )

    local_px = _materialise(local_da)
    http_px = _materialise(http_da)

    _assert_band_count(http_px, label=label)
    _assert_byte_exact(local_px, http_px, label=label)
    _assert_dim_names(http_da, label=label)
    assert http_da.dtype == local_da.dtype, (
        f"{label}: dtype mismatch local={local_da.dtype} http={http_da.dtype}"
    )
    local_crs = local_da.attrs.get("crs")
    http_crs = http_da.attrs.get("crs")
    assert local_crs == http_crs, (
        f"{label}: crs mismatch local={local_crs!r} http={http_crs!r}"
    )
    local_t = local_da.attrs.get("transform")
    assert local_t is not None, f"{label}: local read missing transform"
    _assert_transform_equals(http_da, local_t, label=label)
    assert ("nodata" in local_da.attrs) == ("nodata" in http_da.attrs), (
        f"{label}: nodata presence differs "
        f"local={'nodata' in local_da.attrs} http={'nodata' in http_da.attrs}"
    )
    if "nodata" in local_da.attrs:
        assert local_da.attrs["nodata"] == http_da.attrs["nodata"], (
            f"{label}: nodata value differs "
            f"local={local_da.attrs['nodata']!r} "
            f"http={http_da.attrs['nodata']!r}"
        )


# -------------------------------------------------------------------------
# Section: COG: tile-layout pre-flight
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_da(shape=(64, 64)):
    """A small float32 DataArray suitable for COG writes."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=('y', 'x')
    )


# ---------------------------------------------------------------------------
# Public boundary: ``to_geotiff(cog=True, tiled=False)`` is refused.
# ---------------------------------------------------------------------------

def test_public_writer_rejects_cog_true_tiled_false(tmp_path):
    """The public entry point raises ``ValueError`` with a message that
    names the COG-spec constraint and both caller-side fixes."""
    da = _float_da()
    p = tmp_path / 'cog_tiled_false_2312.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, tiled=False)

    msg = str(exc.value)
    # The message must name the violated constraint.
    assert 'COG' in msg, msg
    assert 'tiled' in msg.lower(), msg
    # Both caller-side fixes must appear so the error is actionable.
    assert 'tiled=True' in msg, msg
    assert 'cog=False' in msg, msg


def test_public_writer_rejects_cog_true_tiled_false_with_tile_size(tmp_path):
    """Pinning the rejection survives a ``tile_size`` kwarg too.

    Previously, ``to_geotiff(..., cog=True, tiled=False,
    tile_size=128)`` emitted the "tile_size is ignored when tiled=False"
    warning and then wrote strips. The new gate has to fire before that
    warning so the caller never sees the misleading "tile_size is
    ignored" message under ``cog=True``.
    """
    da = _float_da()
    p = tmp_path / 'cog_tiled_false_with_tile_size_2312.tif'

    # ``pytest.warns(None)`` was removed; use the stdlib catch_warnings
    # recorder to assert the dead "tile_size is ignored" warning never
    # fires on the ``cog=True`` arm.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, str(p), cog=True, tiled=False, tile_size=128)

    msg = str(exc.value)
    assert 'COG' in msg, msg
    assert 'tiled=True' in msg, msg

    tile_size_warnings = [
        w for w in record
        if 'tile_size' in str(w.message)
        and 'is ignored when tiled=False' in str(w.message)
    ]
    assert not tile_size_warnings, [str(w.message) for w in tile_size_warnings]


# ---------------------------------------------------------------------------
# Defense in depth: ``_writer._write(cog=True, tiled=False)`` also raises.
# ---------------------------------------------------------------------------

def test_lowlevel_write_rejects_cog_true_tiled_false(tmp_path):
    """The array-level entry point ``_writer._write`` (re-exported as
    ``write``) carries its own gate so a caller that bypasses the public
    wrapper still gets the typed rejection.

    Without this, a direct caller could quietly produce the malformed
    strip-plus-overviews file the public boundary refuses.
    """
    arr = np.zeros((64, 64), dtype=np.float32)
    p = tmp_path / 'cog_tiled_false_lowlevel_2312.tif'

    with pytest.raises(ValueError) as exc:
        _array_write(
            arr,
            str(p),
            compression='deflate',
            tiled=False,
            cog=True,
        )

    msg = str(exc.value)
    assert 'COG' in msg, msg
    assert 'tiled=True' in msg, msg
    assert 'cog=False' in msg, msg


# ---------------------------------------------------------------------------
# Smoke test: the valid tiled COG path still works.
# ---------------------------------------------------------------------------

def test_tiled_cog_smoke_still_works(tmp_path):
    """A regression in the new gate that broke valid COG writes would
    be a worse outcome than the original bug. Pin the happy path
    end-to-end so the gate has to stay narrowly targeted at the
    ``cog=True, tiled=False`` combination it is meant to catch.
    """
    da = _float_da(shape=(128, 128))
    p = tmp_path / 'cog_tiled_smoke_2312.tif'

    rv = to_geotiff(da, str(p), cog=True, tiled=True, tile_size=64)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_tiled_cog_smoke_default_tiled(tmp_path):
    """``tiled`` defaults to ``True`` on ``to_geotiff``, so ``cog=True``
    on its own should also produce a valid COG. Pinned so a future
    change that flipped the default would not silently start hitting
    the new rejection gate.
    """
    da = _float_da(shape=(128, 128))
    p = tmp_path / 'cog_tiled_default_smoke_2312.tif'

    rv = to_geotiff(da, str(p), cog=True)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Negative control: ``cog=False, tiled=False`` is still a valid strip TIFF.
# ---------------------------------------------------------------------------

def test_strip_layout_without_cog_still_works(tmp_path):
    """``tiled=False`` without ``cog=True`` is the supported strip-TIFF
    path; the new gate must not regress it. Pinned so a stricter
    interpretation of ``cog=True implies tiled=True`` could not creep
    into the general ``tiled=False`` path.
    """
    da = _float_da(shape=(64, 64))
    p = tmp_path / 'strip_no_cog_2312.tif'

    rv = to_geotiff(da, str(p), cog=False, tiled=False)
    assert rv == str(p)
    assert p.exists()
    assert p.stat().st_size > 0


# -------------------------------------------------------------------------
# Section: COG: tile-size pre-flight
# -------------------------------------------------------------------------

@contextlib.contextmanager
def _alarm_timeout(seconds: int):
    """Raise TimeoutError after ``seconds`` to bound test failure modes.

    No-op on platforms that lack SIGALRM (Windows). The window is large
    enough that a healthy raise path finishes well before the alarm
    fires; if the fix regresses the writer hangs and the alarm fires.
    """
    if not hasattr(signal, 'SIGALRM') or os.name == 'nt':
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(
            f'test exceeded {seconds}s watchdog; the writer likely '
            f'regressed into the #2311 infinite-loop hang.'
        )

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Public boundary: ``to_geotiff(..., cog=True, tile_size<=0)`` must raise.
# Covers both tiled=True and tiled=False, plus 0 and a negative value, so
# the validator gate stays on regardless of layout flag.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('tiled', [True, False])
@pytest.mark.parametrize('tile_size', [-1, 0])
def test_to_geotiff_cog_non_positive_tile_size_raises(tmp_path, tiled, tile_size):
    """``cog=True`` with ``tile_size<=0`` raises ValueError up front,
    regardless of ``tiled``. Previously this hung the writer when
    ``tiled=False``."""
    da = _float_da()
    p = tmp_path / f'cog_tile_size_hang_2311_t{int(tiled)}_ts{tile_size}.tif'

    with _alarm_timeout(5), pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, tiled=tiled, tile_size=tile_size)

    msg = str(exc.value)
    assert 'tile_size' in msg, msg
    # The shared validator says "positive int" -- pin the substring so a
    # message rewrite still keeps the actionable wording.
    assert 'positive' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Sanity: ``cog=False`` with ``tiled=False`` still accepts an unused
# ``tile_size`` (the existing "ignored" warning shape) -- the new gate
# must not fire when neither path will consume the value.
# ---------------------------------------------------------------------------

def test_to_geotiff_non_cog_strip_does_not_validate_tile_size(tmp_path):
    """When neither tiled output nor COG overview generation will use
    ``tile_size``, the validator gate stays off. The pre-existing
    "tile_size ignored" warning still fires (it carries its own
    non-default-value check, not a positivity check), but no error
    is raised."""
    da = _float_da()
    p = tmp_path / 'cog_tile_size_hang_2311_no_cog_strip.tif'

    # A negative tile_size with cog=False AND tiled=False is accepted
    # (with the "ignored" warning) because nothing consumes the value.
    # Use ``filterwarnings`` to swallow the warning so the test only
    # asserts no raise / no hang.
    with _alarm_timeout(5), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        to_geotiff(da, str(p), cog=False, tiled=False, tile_size=-1)

    assert p.exists(), 'writer should have produced a strip-layout file'


# ---------------------------------------------------------------------------
# Defense in depth: drive the inner writer directly with a bad tile_size
# and assert the auto-overview loop raises instead of hanging. Guards
# against future internal callers that bypass ``to_geotiff``'s public
# validator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tile_size', [-1, 0])
def test_writer_auto_overview_loop_rejects_non_positive_tile_size(
        tmp_path, tile_size):
    """``_write(..., cog=True, overview_levels=None)`` raises ValueError
    when ``tile_size`` is not a positive int, instead of spinning in the
    halving loop. The public ``to_geotiff`` already validates earlier;
    this is the inner-writer safety net."""
    from xrspatial.geotiff._writer import _write

    # Minimal float32 array large enough for the auto-overview branch to
    # be entered. The exact pixel values do not matter -- the validator
    # check runs before any encoding work.
    data = np.zeros((64, 64), dtype=np.float32)
    out = tmp_path / f'cog_tile_size_hang_2311_inner_ts{tile_size}.tif'

    with _alarm_timeout(5), pytest.raises(ValueError) as exc:
        _write(data, str(out),
               compression='none',
               tiled=True,
               tile_size=tile_size,
               cog=True,
               overview_levels=None)

    assert 'tile_size' in str(exc.value), str(exc.value)


# ---------------------------------------------------------------------------
# Non-int tile_size values reach the same gate. The public
# ``_validate_tile_size`` (called from ``to_geotiff`` when tiled or cog is
# true) rejects None, float, and bool with typed errors; the
# defense-in-depth gate at the top of ``_write`` does the same for direct
# callers. Both layers should reject all three types.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bad_tile_size', [None, 128.0, True, False])
def test_to_geotiff_cog_non_int_tile_size_raises(tmp_path, bad_tile_size):
    """Non-int ``tile_size`` (None, float, bool) with ``cog=True`` is
    rejected at the public boundary, regardless of ``tiled``. Bool is
    explicitly listed because Python treats ``True``/``False`` as int
    subclasses."""
    da = _float_da()
    p = tmp_path / (
        f'cog_tile_size_hang_2311_nonint_{type(bad_tile_size).__name__}.tif')

    with _alarm_timeout(5), pytest.raises((ValueError, TypeError)) as exc:
        to_geotiff(da, str(p), cog=True, tiled=True, tile_size=bad_tile_size)

    assert 'tile_size' in str(exc.value), str(exc.value)


# ---------------------------------------------------------------------------
# Inner-loop guard coverage: confirm the auto-overview halving loop's own
# ``tile_size > 0`` pre-check is present in ``_write``'s compiled
# constants. Inspecting the constants pins the literal so a future
# refactor that removes the inner guard fails this test loudly even if
# the top-of-``_write`` gate still catches the bad input at runtime.
# (Reaching the inner guard through ``_write`` directly would require
# patching out the top gate, which is invasive; the constants check is
# the simplest reliable pin without rewriting production code.)
# ---------------------------------------------------------------------------

def test_inner_overview_loop_guard_message_is_pinned():
    """Pin the inner-overview ``tile_size`` guard literal so removing
    the loop-side defense fails this test even when the top gate at
    line 407 still raises for the same inputs."""
    from xrspatial.geotiff import _writer as wmod

    guard_msg = (
        'tile_size must be a positive int for COG overview '
        'generation, got tile_size=')
    consts = wmod._write.__code__.co_consts
    found = any(isinstance(c, str) and guard_msg in c for c in consts)
    assert found, (
        'inner-loop guard message not present in _write constants; the '
        'auto-overview guard introduced in #2311 may have been removed.')
