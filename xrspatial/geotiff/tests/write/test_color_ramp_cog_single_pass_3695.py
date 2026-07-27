"""``color_ramp`` statistics on the eager COG write path (#3695).

#3597 stopped ``to_geotiff(dask_data, path, color_ramp=...)`` from executing
the source graph twice, but only on the streaming path. ``cog=True`` skips
the streaming writer (COG overviews need the full array), so it kept falling
through to ``_finite_stats`` on the still-lazy DataArray and ran the caller's
whole pipeline a second time.

The fix folds the already-materialised buffer into a ``StreamingStats``
accumulator, which ``write_symbology_sidecars`` prefers over the reduction.
These tests pin the single execution with a counting ``map_blocks`` layer and
check the resulting sidecar values against ``_finite_stats`` on every branch
the accumulator now covers.
"""
import os
import threading
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._symbology import _eager_finite_stats

from .._helpers.markers import requires_gpu

pytest.importorskip("tifffile")


def _counting_da(base, chunks, dims=("y", "x"), coords=None, attrs=None):
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
    if coords is None:
        coords = {"y": np.arange(base.shape[0], dtype="float64"),
                  "x": np.arange(base.shape[1], dtype="float64")}
    da = xr.DataArray(
        arr, dims=dims, coords=coords,
        attrs={"crs": 4326, **(attrs or {})},
    )
    return da, counter


def _aux_stats(path):
    """Parse ``<path>.aux.xml`` into a ``{STATISTICS_*: float}`` dict."""
    band = ET.parse(path + ".aux.xml").getroot().find(".//PAMRasterBand")
    return {mdi.get("key"): float(mdi.text)
            for mdi in band.findall("./Metadata/MDI")}


def _ref_stats(arr, nodata=None):
    """(min, max, mean, population std) over finite non-nodata values.

    Thin wrapper over the production reduction the accumulator replaces, so
    the assertions below compare against the real thing rather than a
    reimplementation. ``_eager_finite_stats`` is used directly because
    ``_finite_stats`` unwraps ``.data``, which on a bare numpy array is a
    memoryview.
    """
    return _eager_finite_stats(np.asarray(arr), nodata)


def _assert_aux_matches_rel(path, ref, rel):
    stats = _aux_stats(path)
    assert stats["STATISTICS_MINIMUM"] == pytest.approx(ref[0], rel=rel)
    assert stats["STATISTICS_MAXIMUM"] == pytest.approx(ref[1], rel=rel)
    assert stats["STATISTICS_MEAN"] == pytest.approx(ref[2], rel=rel)
    assert stats["STATISTICS_STDDEV"] == pytest.approx(ref[3], rel=rel)


def _assert_aux_matches(path, ref):
    _assert_aux_matches_rel(path, ref, rel=1e-9)


_RNG = np.random.default_rng(3695)
_BASE = _RNG.uniform(-50.0, 150.0, (64, 64))
_BASE[3, 7] = np.nan
_BASE[40, 2] = np.nan

_N_CHUNKS = 16  # 64x64 at chunks=(16, 16)


# --------------------------------------------------------------------------
# single execution of the source graph
# --------------------------------------------------------------------------

def test_cog_color_ramp_executes_source_once(tmp_path):
    """The regression: cog=True used to run the graph twice."""
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "cog_once_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    assert counter["n"] == _N_CHUNKS  # was 2 * _N_CHUNKS before the fix
    _assert_aux_matches(path, _ref_stats(_BASE))
    assert os.path.exists(str(tmp_path / "cog_once_3695.qml"))


def test_cog_without_color_ramp_still_single_pass(tmp_path):
    """The no-symbology COG write was already single-pass; keep it that way."""
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "cog_plain_3695.tif")
    to_geotiff(da, path, cog=True)
    assert counter["n"] == _N_CHUNKS
    assert not os.path.exists(path + ".aux.xml")


def test_streaming_path_unchanged(tmp_path):
    """cog=False keeps the #3597 chunk_observer behaviour."""
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "stream_3695.tif")
    to_geotiff(da, path, color_ramp="viridis")
    assert counter["n"] == _N_CHUNKS
    _assert_aux_matches(path, _ref_stats(_BASE))


# --------------------------------------------------------------------------
# the accumulated statistics must equal the reduction they replaced
# --------------------------------------------------------------------------

def test_cog_stats_match_finite_stats_across_chunkings(tmp_path):
    """Chan combine must be chunking-invariant, so all layouts agree."""
    ref = _ref_stats(_BASE)
    for i, chunks in enumerate([(16, 16), (64, 64), (8, 32), (7, 13)]):
        da, _ = _counting_da(_BASE, chunks=chunks)
        path = str(tmp_path / f"chunking_{i}_3695.tif")
        to_geotiff(da, path, cog=True, color_ramp="viridis")
        _assert_aux_matches(path, ref)


def test_cog_stats_exclude_nodata(tmp_path):
    base = _BASE.copy()
    base[10:14, 10:14] = -9999.0
    da, counter = _counting_da(base, chunks=(16, 16))
    path = str(tmp_path / "nodata_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis", nodata=-9999.0)
    assert counter["n"] == _N_CHUNKS
    _assert_aux_matches(path, _ref_stats(base, nodata=-9999.0))
    # The sentinel must not leak into the ramp bounds.
    assert _aux_stats(path)["STATISTICS_MINIMUM"] > -9999.0


def test_cog_integer_source_stats(tmp_path):
    base = _RNG.integers(0, 500, (64, 64)).astype("int32")
    da, counter = _counting_da(base, chunks=(16, 16))
    path = str(tmp_path / "int_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    assert counter["n"] == _N_CHUNKS
    _assert_aux_matches(path, _ref_stats(base))


def test_cog_integer_source_with_nodata(tmp_path):
    base = _RNG.integers(0, 500, (64, 64)).astype("int32")
    base[0, :] = -1
    da, counter = _counting_da(base, chunks=(16, 16))
    path = str(tmp_path / "int_nodata_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis", nodata=-1)
    assert counter["n"] == _N_CHUNKS
    _assert_aux_matches(path, _ref_stats(base, nodata=-1))


# --------------------------------------------------------------------------
# gating: the accumulator must not change who gets sidecars
# --------------------------------------------------------------------------

def test_cog_color_ramp_range_still_skips_stats(tmp_path):
    """The escape hatch writes bounds only and never builds the accumulator."""
    da, counter = _counting_da(_BASE, chunks=(16, 16))
    path = str(tmp_path / "cog_rng_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis",
               color_ramp_range=(0.0, 10.0))
    assert counter["n"] == _N_CHUNKS
    stats = _aux_stats(path)
    assert stats["STATISTICS_MINIMUM"] == pytest.approx(0.0)
    assert stats["STATISTICS_MAXIMUM"] == pytest.approx(10.0)
    assert "STATISTICS_MEAN" not in stats
    assert "STATISTICS_STDDEV" not in stats


def test_cog_multiband_gets_no_symbology(tmp_path):
    base = _RNG.uniform(0.0, 1.0, (32, 32, 3))
    da, counter = _counting_da(
        base, chunks=(16, 16, 3), dims=("y", "x", "band"),
        coords={"y": np.arange(32, dtype="float64"),
                "x": np.arange(32, dtype="float64"),
                "band": np.arange(3)})
    path = str(tmp_path / "multiband_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    assert counter["n"] == 4
    assert not os.path.exists(path + ".aux.xml")
    assert not os.path.exists(str(tmp_path / "multiband_3695.qml"))


def test_cog_single_band_3d_gets_symbology(tmp_path):
    """A 3D array with one band is still a single-band raster."""
    base = _BASE[:32, :32].reshape(32, 32, 1)
    da, counter = _counting_da(
        base, chunks=(16, 16, 1), dims=("y", "x", "band"),
        coords={"y": np.arange(32, dtype="float64"),
                "x": np.arange(32, dtype="float64"),
                "band": np.arange(1)})
    path = str(tmp_path / "single3d_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    assert counter["n"] == 4
    _assert_aux_matches(path, _ref_stats(base))


def test_cog_all_nan_writes_no_sidecar(tmp_path):
    base = np.full((32, 32), np.nan)
    da, counter = _counting_da(base, chunks=(16, 16))
    path = str(tmp_path / "allnan_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    assert counter["n"] == 4
    # No finite values -> no statistics, matching _finite_stats returning None.
    assert _ref_stats(base) is None
    assert not os.path.exists(path + ".aux.xml")
    assert not os.path.exists(str(tmp_path / "allnan_3695.qml"))


def test_cog_constant_raster_writes_stats_but_no_qml(tmp_path):
    """vmin == vmax is a degenerate ramp; stats still land."""
    base = np.full((32, 32), 7.5)
    da, _ = _counting_da(base, chunks=(16, 16))
    path = str(tmp_path / "const_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    stats = _aux_stats(path)
    assert stats["STATISTICS_MINIMUM"] == pytest.approx(7.5)
    assert stats["STATISTICS_MAXIMUM"] == pytest.approx(7.5)
    assert not os.path.exists(str(tmp_path / "const_3695.qml"))


def test_cog_float32_matches_streaming_write(tmp_path):
    """float32 statistics agree with the cog=False write on the same data.

    ``StreamingStats`` accumulates in float64 while ``_finite_stats``
    accumulates at the input's native width, so a float32 source's mean and
    stddev move by ~1e-7 relative against the old COG output. What matters is
    that the two dask write paths now agree with each other, which they did
    not before: cog=False has used the float64 accumulator since #3597.
    """
    base = _RNG.uniform(-1e4, 1e4, (64, 64)).astype("float32")
    base[5, 5] = np.nan

    cog_path = str(tmp_path / "f32_cog_3695.tif")
    stream_path = str(tmp_path / "f32_stream_3695.tif")
    da_cog, counter = _counting_da(base, chunks=(16, 16))
    da_stream, _ = _counting_da(base, chunks=(16, 16))
    to_geotiff(da_cog, cog_path, cog=True, color_ramp="viridis")
    to_geotiff(da_stream, stream_path, color_ramp="viridis")

    assert counter["n"] == _N_CHUNKS
    assert _aux_stats(cog_path) == _aux_stats(stream_path)
    # Still the same numbers as the native-width reduction to float32
    # resolution, so the ramp bounds and stretch are unchanged in practice.
    _assert_aux_matches_rel(cog_path, _ref_stats(base), rel=1e-6)


def test_eager_numpy_cog_unaffected(tmp_path):
    """A numpy source has no graph to re-execute; sidecars stay the same."""
    da = xr.DataArray(
        _BASE, dims=("y", "x"),
        coords={"y": np.arange(64, dtype="float64"),
                "x": np.arange(64, dtype="float64")},
        attrs={"crs": 4326})
    path = str(tmp_path / "numpy_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis")
    _assert_aux_matches(path, _ref_stats(_BASE))


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

@requires_gpu
def test_cog_dask_cupy_color_ramp_single_pass(tmp_path):
    """dask+cupy reaches the same eager COG fallthrough (gpu=False writer)."""
    import cupy
    import dask.array as dsa

    counter = {"n": 0}
    lock = threading.Lock()

    def _count(block):
        if block.size:
            with lock:
                counter["n"] += 1
        return block

    arr = dsa.from_array(cupy.asarray(_BASE), chunks=(16, 16)).map_blocks(
        _count, dtype=_BASE.dtype)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": np.arange(64, dtype="float64"),
                "x": np.arange(64, dtype="float64")},
        attrs={"crs": 4326})
    path = str(tmp_path / "cog_gpu_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis", gpu=False)
    assert counter["n"] == _N_CHUNKS
    _assert_aux_matches(path, _ref_stats(_BASE))


@requires_gpu
def test_cog_cupy_eager_color_ramp(tmp_path):
    """A plain cupy source has no graph; the accumulator must not fire."""
    import cupy

    da = xr.DataArray(
        cupy.asarray(_BASE), dims=("y", "x"),
        coords={"y": np.arange(64, dtype="float64"),
                "x": np.arange(64, dtype="float64")},
        attrs={"crs": 4326})
    path = str(tmp_path / "cupy_3695.tif")
    to_geotiff(da, path, cog=True, color_ramp="viridis", gpu=False)
    _assert_aux_matches(path, _ref_stats(_BASE))
