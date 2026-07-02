"""``color_ramp`` statistics ride along with the streaming dask write (#3597).

``to_geotiff(dask_data, path, color_ramp=...)`` used to execute the source
graph twice: once in ``_write_streaming`` for the pixels and once more in
``_finite_stats`` for the sidecar statistics. The fix threads a
``chunk_observer`` through the streaming writer so a ``StreamingStats``
accumulator folds in every buffer the write materialises anyway. These tests
pin the single-execution behaviour with a counting ``map_blocks`` layer and
check the accumulated statistics against ``_finite_stats`` / plain numpy on
the row-band, segmented wide-raster, and strip paths.
"""
import os
import threading
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._symbology import StreamingStats, _finite_stats

from .._helpers.markers import requires_gpu

pytest.importorskip("tifffile")


def _counting_da(base, chunks):
    """Dask-backed DataArray whose chunks count their own executions."""
    import dask.array as dsa

    counter = {"n": 0}
    lock = threading.Lock()

    def _count(block):
        if block.size:  # skip dask's zero-size meta-inference call
            with lock:
                counter["n"] += 1
        return block

    arr = dsa.from_array(base, chunks=chunks).map_blocks(
        _count, dtype=base.dtype)
    n = base.shape[0]
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": np.arange(n, dtype="float64"),
                "x": np.arange(base.shape[1], dtype="float64")},
        attrs={"crs": 4326},
    )
    return da, counter


def _aux_stats(path):
    """Parse ``<path>.aux.xml`` into a ``{STATISTICS_*: float}`` dict."""
    band = ET.parse(path + ".aux.xml").getroot().find(".//PAMRasterBand")
    return {mdi.get("key"): float(mdi.text)
            for mdi in band.findall("./Metadata/MDI")}


def _ref_stats(arr, nodata=None):
    """(min, max, mean, population std) over finite non-nodata values."""
    a = np.asarray(arr, dtype="float64")
    mask = np.isfinite(a)
    if nodata is not None:
        mask &= (a != nodata)
    v = a[mask]
    return float(v.min()), float(v.max()), float(v.mean()), float(v.std())


def _assert_aux_matches(path, ref):
    stats = _aux_stats(path)
    assert stats["STATISTICS_MINIMUM"] == pytest.approx(ref[0], rel=1e-9)
    assert stats["STATISTICS_MAXIMUM"] == pytest.approx(ref[1], rel=1e-9)
    assert stats["STATISTICS_MEAN"] == pytest.approx(ref[2], rel=1e-9)
    assert stats["STATISTICS_STDDEV"] == pytest.approx(ref[3], rel=1e-9)


_RNG = np.random.default_rng(3597)
_BASE = _RNG.uniform(-50.0, 150.0, (64, 64))
_BASE[3, 7] = np.nan
_BASE[40, 2] = np.nan


# --------------------------------------------------------------------------
# single execution of the source graph
# --------------------------------------------------------------------------

def test_streaming_color_ramp_executes_source_once(tmp_path):
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "once_3597.tif")
    to_geotiff(da, path, color_ramp="viridis")
    assert counter["n"] == 16  # was 32 before the fix
    _assert_aux_matches(path, _ref_stats(_BASE))
    assert os.path.exists(str(tmp_path / "once_3597.qml"))


def test_streaming_strip_path_executes_source_once(tmp_path):
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "strip_3597.tif")
    to_geotiff(da, path, tiled=False, color_ramp="viridis")
    assert counter["n"] == 16
    _assert_aux_matches(path, _ref_stats(_BASE))


def test_color_ramp_range_still_skips_stats(tmp_path):
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "rng_3597.tif")
    to_geotiff(da, path, color_ramp="viridis", color_ramp_range=(0.0, 10.0))
    assert counter["n"] == 16
    aux = open(path + ".aux.xml").read()
    assert "STATISTICS_MINIMUM" in aux and "STATISTICS_MEAN" not in aux


# --------------------------------------------------------------------------
# the segmented wide-raster path partitions pixels even when chunks recompute
# --------------------------------------------------------------------------

def test_segmented_wide_path_stats_exact(tmp_path):
    # Full-width source chunks + a buffer budget of two 16x16 float64 tile
    # columns force the column-segmented path, where a source chunk is
    # computed once per segment it spans. The observer is fed the segment
    # buffers (which partition the raster), so the statistics must stay
    # exact even though chunk executions exceed the chunk count.
    da, counter = _counting_da(_BASE, chunks=(16, 64))
    path = str(tmp_path / "wide_3597.tif")
    to_geotiff(da, path, tiled=True, tile_size=16,
               streaming_buffer_bytes=4096, color_ramp="viridis")
    assert counter["n"] > 4  # proves the segmented path actually engaged
    _assert_aux_matches(path, _ref_stats(_BASE))


# --------------------------------------------------------------------------
# semantics parity with _finite_stats
# --------------------------------------------------------------------------

def test_streaming_nodata_sentinel_excluded(tmp_path):
    import dask.array as dsa

    vals = _BASE.copy()
    vals[10:20, 10:20] = -9999.0
    da = xr.DataArray(
        dsa.from_array(vals, chunks=(16, 16)), dims=("y", "x"),
        coords={"y": np.arange(64.0), "x": np.arange(64.0)},
        attrs={"crs": 4326, "nodata": -9999.0},
    )
    path = str(tmp_path / "nd_3597.tif")
    to_geotiff(da, path, color_ramp="viridis")
    _assert_aux_matches(path, _ref_stats(vals, nodata=-9999.0))


def test_streaming_int_dtype_stats(tmp_path):
    import dask.array as dsa

    vals = _RNG.integers(-100, 500, (64, 64)).astype("int32")
    vals[0, :] = -32768
    da = xr.DataArray(
        dsa.from_array(vals, chunks=(16, 16)), dims=("y", "x"),
        coords={"y": np.arange(64.0), "x": np.arange(64.0)},
        attrs={"crs": 4326, "nodata": -32768},
    )
    path = str(tmp_path / "int_3597.tif")
    to_geotiff(da, path, color_ramp="viridis")
    _assert_aux_matches(path, _ref_stats(vals, nodata=-32768))


def test_all_nan_dask_writes_no_sidecars(tmp_path):
    import dask.array as dsa

    da = xr.DataArray(
        dsa.full((32, 32), np.nan, chunks=(16, 16)), dims=("y", "x"),
        coords={"y": np.arange(32.0), "x": np.arange(32.0)},
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "nan_3597.tif")
    to_geotiff(da, path, color_ramp="viridis")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".aux.xml")
    assert not os.path.exists(str(tmp_path / "nan_3597.qml"))


def test_multiband_dask_color_ramp_noop(tmp_path):
    import dask.array as dsa

    rgb = xr.DataArray(
        dsa.zeros((3, 32, 32), chunks=(3, 16, 16), dtype="float32"),
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": np.arange(32.0),
                "x": np.arange(32.0)},
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "rgb_3597.tif")
    to_geotiff(rgb, path, color_ramp="viridis")
    assert not os.path.exists(path + ".aux.xml")
    assert not os.path.exists(str(tmp_path / "rgb_3597.qml"))


# --------------------------------------------------------------------------
# StreamingStats unit behaviour
# --------------------------------------------------------------------------

def test_streaming_stats_matches_finite_stats_uneven_slabs():
    acc = StreamingStats()
    for r0, r1 in [(0, 5), (5, 6), (6, 40), (40, 64)]:
        acc.update(_BASE[r0:r1])
    da = xr.DataArray(_BASE, dims=("y", "x"))
    assert acc.result() == pytest.approx(_finite_stats(da, None), rel=1e-9)


def test_streaming_stats_nan_nodata_treated_as_unset():
    acc = StreamingStats(nodata=float("nan"))
    acc.update(_BASE)
    da = xr.DataArray(_BASE, dims=("y", "x"))
    assert acc.result() == pytest.approx(_finite_stats(da, None), rel=1e-9)


def test_streaming_stats_empty_returns_none():
    acc = StreamingStats()
    assert acc.result() is None
    acc.update(np.full((4, 4), np.nan))
    assert acc.result() is None


def test_streaming_stats_constant_buffers():
    acc = StreamingStats()
    acc.update(np.full((4, 4), 7.0))
    acc.update(np.full((2, 4), 7.0))
    assert acc.result() == pytest.approx((7.0, 7.0, 7.0, 0.0), abs=1e-12)


# --------------------------------------------------------------------------
# dask+cupy through the CPU streaming writer (gpu=False)
# --------------------------------------------------------------------------

@requires_gpu
def test_dask_cupy_streaming_executes_source_once(tmp_path):
    import cupy
    import dask.array as dsa

    counter = {"n": 0}
    lock = threading.Lock()

    def _count(block):
        if block.size:  # skip dask's zero-size meta-inference call
            with lock:
                counter["n"] += 1
        return block

    arr = dsa.from_array(cupy.asarray(_BASE), chunks=(16, 16)).map_blocks(
        _count, dtype=_BASE.dtype)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": np.arange(64.0), "x": np.arange(64.0)},
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "dgpu_3597.tif")
    to_geotiff(da, path, gpu=False, color_ramp="viridis")
    assert counter["n"] == 16
    _assert_aux_matches(path, _ref_stats(_BASE))
