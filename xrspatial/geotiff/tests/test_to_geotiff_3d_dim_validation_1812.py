"""Regression tests for issue #1812.

``to_geotiff`` / ``write_geotiff_gpu`` used to silently mishandle 3D
DataArrays whose leading dim was not in
``_BAND_DIM_NAMES = ('band', 'bands', 'channel')``. The
``moveaxis(arr, 0, -1)`` that puts ``(band, y, x)`` into the on-disk
``(y, x, band)`` layout was skipped, the writer kept the leading axis
as the spatial ``y`` axis, and the result was a TIFF with silently
swapped axes -- on read-back, ``out[:, :, 0].sum() != arr[0].sum()``.

The fix raises ``ValueError`` at the entry point of all three writers
(eager, dask streaming, and GPU) when ``dims`` is not unambiguously
one of ``(band, y, x)`` or ``(y, x, band)``.
"""
from __future__ import annotations

import importlib.util

import dask.array as dsk
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


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


# Inputs that must be accepted (round-trip cleanly).
_HAPPY_3D_INPUTS = [
    pytest.param(("band", "y", "x"), (3, 4, 5), id="band-y-x"),
    pytest.param(("bands", "y", "x"), (3, 4, 5), id="bands-y-x"),
    pytest.param(("channel", "y", "x"), (3, 4, 5), id="channel-y-x"),
    pytest.param(("y", "x", "band"), (4, 5, 3), id="y-x-band"),
    pytest.param(("lat", "lon", "band"), (4, 5, 3), id="lat-lon-band"),
    pytest.param(("row", "col", "channel"), (4, 5, 3), id="row-col-channel"),
    pytest.param(("band", "lat", "lon"), (3, 4, 5), id="band-lat-lon-alias"),
]


def _make_da(dims, shape, dtype=np.uint8, backend="numpy"):
    if backend == "numpy":
        arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    elif backend == "dask":
        arr_np = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        arr = dsk.from_array(arr_np, chunks=2)
    elif backend == "cupy":
        import cupy

        arr = cupy.arange(int(np.prod(shape)),
                          dtype=cupy.dtype(dtype)).reshape(shape)
    else:
        raise ValueError(backend)
    return xr.DataArray(arr, dims=dims, attrs={"crs": "EPSG:4326"})


def test_repro_silent_corruption_now_raises(tmp_path):
    """The original repro now raises a clear ValueError.

    Post-#1972 the ``(time, y, x)`` layout produces the dedicated
    temporal-leading-dim message rather than the generic ambiguous-dims
    one, so accept either wording.
    """
    arr = np.zeros((2, 4, 5), dtype=np.uint8)
    arr[0] = 1
    arr[1] = 2
    da = xr.DataArray(arr, dims=("time", "y", "x"),
                      attrs={"crs": "EPSG:4326"})
    out_path = tmp_path / "tmp_1812_time_y_x.tif"
    with pytest.raises(ValueError, match="ambiguous dims|temporal leading dim"):
        to_geotiff(da, str(out_path), crs=4326)


@pytest.mark.parametrize("dims, shape", [
    pytest.param(("time", "y", "x"), (2, 4, 5), id="time-y-x"),
    pytest.param(("z", "y", "x"), (2, 4, 5), id="z-y-x"),
    pytest.param(("foo", "bar", "baz"), (2, 4, 5), id="foo-bar-baz"),
])
def test_eager_rejects_ambiguous_3d(tmp_path, dims, shape):
    """Eager numpy path raises ValueError on ambiguous 3D dim names."""
    da = _make_da(dims, shape, backend="numpy")
    out_path = tmp_path / f"tmp_1812_eager_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError, match="ambiguous dims|temporal leading dim"):
        to_geotiff(da, str(out_path), crs=4326)


@pytest.mark.parametrize("dims, shape", [
    pytest.param(("time", "y", "x"), (2, 4, 5), id="time-y-x"),
    pytest.param(("z", "y", "x"), (2, 4, 5), id="z-y-x"),
    pytest.param(("foo", "bar", "baz"), (2, 4, 5), id="foo-bar-baz"),
])
def test_dask_streaming_rejects_ambiguous_3d(tmp_path, dims, shape):
    """Dask-streaming branch raises ValueError on ambiguous 3D dim names."""
    da = _make_da(dims, shape, backend="dask")
    out_path = tmp_path / f"tmp_1812_dask_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError, match="ambiguous dims|temporal leading dim"):
        to_geotiff(da, str(out_path), crs=4326)


@_gpu_only
@pytest.mark.parametrize("dims, shape", [
    pytest.param(("time", "y", "x"), (2, 4, 5), id="time-y-x"),
    pytest.param(("foo", "bar", "baz"), (2, 4, 5), id="foo-bar-baz"),
])
def test_gpu_writer_rejects_ambiguous_3d(tmp_path, dims, shape):
    """GPU writer raises ValueError on ambiguous 3D dim names."""
    from xrspatial.geotiff import write_geotiff_gpu

    da = _make_da(dims, shape, backend="cupy")
    out_path = tmp_path / f"tmp_1812_gpu_{'_'.join(dims)}.tif"
    with pytest.raises(ValueError, match="ambiguous dims|temporal leading dim"):
        write_geotiff_gpu(da, str(out_path), crs=4326)


@pytest.mark.parametrize("dims, shape", _HAPPY_3D_INPUTS)
def test_happy_3d_round_trip(tmp_path, dims, shape):
    """Accepted 3D dim layouts still round-trip cleanly (eager + dask).

    Each slice along the band axis is filled with a distinct constant
    so a silent axis swap would change the per-slice sums.
    """
    # Build a per-slice-distinguishable array. ``arr_full[k]`` along the
    # band axis is filled with ``k + 1``.
    band_pos = next(i for i, d in enumerate(dims)
                    if d in ("band", "bands", "channel"))
    n_bands = shape[band_pos]
    spatial_shape = tuple(s for i, s in enumerate(shape) if i != band_pos)
    arr_np = np.empty(shape, dtype=np.uint8)
    for k in range(n_bands):
        slicer = [slice(None)] * 3
        slicer[band_pos] = k
        arr_np[tuple(slicer)] = k + 1

    # Eager round-trip
    da_eager = xr.DataArray(arr_np, dims=dims,
                            attrs={"crs": "EPSG:4326"})
    p_eager = tmp_path / f"tmp_1812_happy_eager_{'_'.join(dims)}.tif"
    to_geotiff(da_eager, str(p_eager), crs=4326)
    out_eager = open_geotiff(str(p_eager))
    # On-disk layout is always (y, x, band). Compare per-band sums.
    assert out_eager.shape == spatial_shape + (n_bands,)
    for k in range(n_bands):
        expected = (k + 1) * (spatial_shape[0] * spatial_shape[1])
        assert int(out_eager.values[:, :, k].sum()) == expected, (
            f"band {k} sum mismatch on eager round-trip of dims={dims}"
        )

    # Dask streaming round-trip
    da_dask = xr.DataArray(dsk.from_array(arr_np, chunks=2), dims=dims,
                           attrs={"crs": "EPSG:4326"})
    p_dask = tmp_path / f"tmp_1812_happy_dask_{'_'.join(dims)}.tif"
    to_geotiff(da_dask, str(p_dask), crs=4326)
    out_dask = open_geotiff(str(p_dask))
    assert out_dask.shape == spatial_shape + (n_bands,)
    for k in range(n_bands):
        expected = (k + 1) * (spatial_shape[0] * spatial_shape[1])
        assert int(out_dask.values[:, :, k].sum()) == expected, (
            f"band {k} sum mismatch on dask round-trip of dims={dims}"
        )


def test_2d_still_works(tmp_path):
    """2D inputs are unaffected by the new 3D validator."""
    arr = np.arange(20, dtype=np.uint8).reshape(4, 5)
    da = xr.DataArray(arr, dims=("y", "x"), attrs={"crs": "EPSG:4326"})
    p = tmp_path / "tmp_1812_2d.tif"
    to_geotiff(da, str(p), crs=4326)
    out = open_geotiff(str(p))
    assert out.shape == (4, 5)
    assert np.array_equal(out.values, arr)


def test_error_message_actionable(tmp_path):
    """The generic ValueError message tells the caller how to fix the input.

    Uses a non-temporal leading dim so the dedicated #1972 temporal path
    does not short-circuit, keeping the assertions on the generic
    "(band, y, x)" / "(y, x, band)" / "#1812" wording intact.
    """
    arr = np.zeros((2, 4, 5), dtype=np.uint8)
    da = xr.DataArray(arr, dims=("z", "y", "x"),
                      attrs={"crs": "EPSG:4326"})
    p = tmp_path / "tmp_1812_msg.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(p), crs=4326)
    msg = str(excinfo.value)
    # Mentions the offending dim layout
    assert "('z', 'y', 'x')" in msg
    # Mentions the accepted alternatives
    assert "(band, y, x)" in msg
    assert "(y, x, band)" in msg
    # Points the user at a concrete remediation
    assert "transpose" in msg.lower() or "rename" in msg.lower()
    # References the issue
    assert "#1812" in msg


@_gpu_only
def test_gpu_writer_happy_path_still_works(tmp_path):
    """GPU writer's existing happy paths (band-first and band-last) survive."""
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu

    arr_bf = cupy.arange(3 * 4 * 5, dtype=cupy.uint8).reshape(3, 4, 5)
    da_bf = xr.DataArray(arr_bf, dims=("band", "y", "x"),
                         attrs={"crs": "EPSG:4326"})
    p_bf = tmp_path / "tmp_1812_gpu_bf.tif"
    write_geotiff_gpu(da_bf, str(p_bf), crs=4326)
    out_bf = open_geotiff(str(p_bf))
    assert out_bf.shape == (4, 5, 3)

    arr_bl = cupy.arange(4 * 5 * 3, dtype=cupy.uint8).reshape(4, 5, 3)
    da_bl = xr.DataArray(arr_bl, dims=("y", "x", "band"),
                         attrs={"crs": "EPSG:4326"})
    p_bl = tmp_path / "tmp_1812_gpu_bl.tif"
    write_geotiff_gpu(da_bl, str(p_bl), crs=4326)
    out_bl = open_geotiff(str(p_bl))
    assert out_bl.shape == (4, 5, 3)
