"""``to_geotiff(pack=True)`` must not execute a dask graph twice (#3235).

``_pack`` guards the no-sentinel integer restore with a NaN scan. The
scan used to run as an eager ``isnull().any()``, which executed the
whole upstream graph at call time; the streaming writer then executed
it again for the pixels, so every source chunk computed twice. The
guard is now mapped into the graph for dask-backed data
(``_pack_guard_no_nan``) and raises from the write's single compute.
numpy-backed data keeps the eager call-time check.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _pack, _pack_guard_no_nan

from .._helpers.markers import requires_gpu


def _write_packed_no_sentinel_tiff(path, data, *, scale="0.1", offset="5.0"):
    """Write an integer GeoTIFF carrying SCALE/OFFSET but no nodata."""
    h, w = data.shape
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(h, 0, -1) - 0.5, "x": np.arange(w) + 0.5},
        attrs={"crs": 4326,
               "gdal_metadata": {"SCALE": scale, "OFFSET": offset}},
    )
    to_geotiff(da, str(path))
    return str(path)


def _no_sentinel_unpacked_dask(values, chunks=2):
    """Build a dask-backed DataArray carrying unpack state but no nodata."""
    import dask.array as dask_array

    arr = dask_array.from_array(values, chunks=chunks)
    return xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"crs": 4326,
               "scale_factor": 0.1,
               "add_offset": 5.0,
               "mask_and_scale_dtype": "int16"},
    )


def test_pack_dask_no_sentinel_computes_each_chunk_once(tmp_path):
    """The NaN guard must not add a second full graph execution."""
    data = np.arange(64 * 64, dtype=np.int16).reshape(64, 64) % 3000
    src = _write_packed_no_sentinel_tiff(
        tmp_path / "src_no_sentinel_3235.tif", data)

    decoded = open_geotiff(src, unpack=True, chunks=16)
    assert hasattr(decoded.data, "dask")
    assert decoded.attrs.get("nodata") is None
    assert decoded.attrs.get("mask_and_scale_dtype") == "int16"

    # Fusion-proof execution counter: thread a counting identity block
    # through the graph so each chunk increments exactly once per
    # execution, whatever the scheduler fuses the decode tasks into.
    import threading
    lock = threading.Lock()
    counts = {"n": 0}

    def _count(block):
        with lock:
            counts["n"] += 1
        return block

    counted = decoded.copy(
        data=decoded.data.map_blocks(
            _count, dtype=decoded.dtype, meta=decoded.data._meta))
    n_chunks = counted.data.npartitions

    out = str(tmp_path / "out_no_sentinel_3235.tif")
    to_geotiff(counted, out, pack=True)

    # Pre-fix this was 2 * n_chunks: one execution for the eager
    # isnull().any() scan, one for the streaming write.
    assert counts["n"] == n_chunks

    back = open_geotiff(out)
    assert str(back.dtype) == "int16"
    np.testing.assert_array_equal(back.data, data)


def test_pack_dask_no_sentinel_round_trip_matches_numpy(tmp_path):
    """The lazy guard changes nothing about the packed pixels."""
    data = (np.arange(48, dtype=np.int16).reshape(6, 8) * 7) % 2000
    src = _write_packed_no_sentinel_tiff(
        tmp_path / "src_parity_3235.tif", data)

    eager = open_geotiff(src, unpack=True)
    lazy = open_geotiff(src, unpack=True, chunks=2)

    out_eager = str(tmp_path / "out_eager_3235.tif")
    out_lazy = str(tmp_path / "out_lazy_3235.tif")
    to_geotiff(eager, out_eager, pack=True)
    to_geotiff(lazy, out_lazy, pack=True)

    back_eager = open_geotiff(out_eager)
    back_lazy = open_geotiff(out_lazy)
    assert str(back_lazy.dtype) == str(back_eager.dtype) == "int16"
    np.testing.assert_array_equal(back_lazy.data, back_eager.data)
    np.testing.assert_array_equal(back_lazy.data, data)


def test_pack_dask_nan_no_sentinel_raises_at_compute(tmp_path):
    """NaN with no sentinel still refuses the integer restore on dask.

    The error now surfaces from the compute that consumes the packed
    array (the write) rather than at the ``to_geotiff`` call, but the
    message is unchanged.
    """
    values = np.full((4, 4), 10.0, dtype=np.float64)
    values[1, 2] = np.nan
    da = _no_sentinel_unpacked_dask(values)

    out = str(tmp_path / "out_nan_dask_3235.tif")
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype int16"):
        to_geotiff(da, out, pack=True)


def test_pack_dask_nan_no_sentinel_raises_through_vrt_write(tmp_path):
    """A ``.vrt`` destination surfaces the deferred guard the same way.

    The VRT path consumes the packed graph through ``_write_vrt_tiled``'s
    delayed per-tile writes (collect-and-reraise via ``_safe_write_tile``)
    rather than ``_write_streaming``; the original ValueError must come
    back verbatim and the staging dir must be cleaned up.
    """
    values = np.full((4, 4), 10.0, dtype=np.float64)
    values[3, 1] = np.nan
    da = _no_sentinel_unpacked_dask(values)

    out = str(tmp_path / "out_nan_dask_3235.vrt")
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype int16"):
        to_geotiff(da, out, pack=True)
    import os
    assert not os.path.exists(out)
    assert not os.path.isdir(str(tmp_path / "out_nan_dask_3235_tiles"))


def test_pack_dask_nan_no_sentinel_packed_graph_is_lazy():
    """``_pack`` itself must not compute dask input, even with NaN."""
    values = np.full((4, 4), 10.0, dtype=np.float64)
    values[1, 2] = np.nan
    da = _no_sentinel_unpacked_dask(values)

    packed = _pack(da)  # must not raise and must not compute
    assert hasattr(packed.data, "dask")
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype int16"):
        packed.compute()


def test_pack_numpy_nan_no_sentinel_raises_eagerly(tmp_path):
    """numpy-backed data keeps the call-time ValueError."""
    values = np.full((4, 4), 10.0, dtype=np.float64)
    values[0, 0] = np.nan
    da = xr.DataArray(
        values, dims=("y", "x"),
        attrs={"crs": 4326,
               "scale_factor": 0.1,
               "add_offset": 5.0,
               "mask_and_scale_dtype": "int16"},
    )
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype int16"):
        _pack(da)


def test_pack_guard_passes_clean_chunk_through():
    chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _pack_guard_no_nan(chunk, "int16")
    assert out is chunk


def test_pack_guard_raises_on_nan_chunk():
    chunk = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype uint8"):
        _pack_guard_no_nan(chunk, "uint8")


def test_pack_guard_skips_integer_chunk():
    chunk = np.array([[1, 2]], dtype=np.int32)
    assert _pack_guard_no_nan(chunk, "int32") is chunk


@requires_gpu
def test_pack_guard_handles_cupy_chunks():
    """The guard is backend-agnostic: cupy chunks pass and raise alike.

    Pinned at the unit level from when full dask+cupy pack round trips
    crashed upstream of this guard (fixed in #3112); kept as direct
    coverage of the cupy leg.
    """
    import cupy

    clean = cupy.asarray(np.array([[1.0, 2.0]]))
    out = _pack_guard_no_nan(clean, "int16")
    assert out is clean
    assert isinstance(out, cupy.ndarray)

    dirty = cupy.asarray(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError,
                       match="cannot restore integer dtype int16"):
        _pack_guard_no_nan(dirty, "int16")
