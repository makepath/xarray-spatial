"""Combined dask+cupy GPU pipeline integration tests.

The ``requires_gpu`` marker comes from ``_helpers/markers.py``.
"""
from __future__ import annotations

import numpy as np

from .._helpers.markers import requires_gpu

pytestmark = requires_gpu


# ----------------------------------------------------------
# Section: dask_cupy_combined
# ----------------------------------------------------------
def _assert_dask_cupy_dask_cupy_combined(da_arr, expected_chunks, expected_dtype):
    """Common shape/type checks for a dask-wrapped cupy DataArray.

    Returns the computed DataArray so callers can reuse it for pixel
    comparison without paying for a second ``.compute()``.
    """
    import cupy
    import dask.array as da_mod

    raw = da_arr.data
    assert isinstance(raw, da_mod.Array), (
        f"expected dask Array, got {type(raw).__name__}"
    )

    # _meta carries the underlying array type for distributed Dask
    # graph optimisation. If this is numpy, downstream operations may
    # silently transfer GPU data back to CPU.
    meta = raw._meta
    assert isinstance(meta, cupy.ndarray), (
        f"expected cupy._meta, got {type(meta).__module__}."
        f"{type(meta).__name__}"
    )

    # Chunk shape must match what the caller asked for.
    assert raw.chunks == expected_chunks, (
        f"chunks {raw.chunks} != expected {expected_chunks}"
    )

    assert raw.dtype == expected_dtype, (
        f"dtype {raw.dtype} != expected {expected_dtype}"
    )

    # After compute the result is still a cupy array, not numpy.
    computed = da_arr.compute()
    assert isinstance(computed.data, cupy.ndarray), (
        f"compute() returned {type(computed.data).__name__} "
        "(should stay on device)"
    )
    return computed


def test_open_geotiff_gpu_chunks_int_round_trip(tmp_path):
    """`open_geotiff(gpu=True, chunks=N)` returns dask+cupy with int chunk."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(7)
    arr = rng.randint(0, 10_000, (256, 256)).astype(np.float32)
    path = str(tmp_path / "single_band.tif")
    to_geotiff(arr, path, compression="deflate", tiled=True, tile_size=64)

    eager = np.asarray(open_geotiff(path).data)

    da_arr = open_geotiff(path, gpu=True, chunks=64)

    computed = _assert_dask_cupy_dask_cupy_combined(
        da_arr,
        expected_chunks=((64, 64, 64, 64), (64, 64, 64, 64)),
        expected_dtype=np.dtype(np.float32),
    )

    got = computed.data.get()
    np.testing.assert_array_equal(got, eager)


def test_read_geotiff_gpu_chunks_tuple_round_trip(tmp_path):
    """`read_geotiff_gpu(chunks=(rh, cw))` accepts tuple chunk specs."""
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu, to_geotiff

    rng = np.random.RandomState(11)
    arr = rng.randint(0, 60_000, (192, 256)).astype(np.uint16)
    path = str(tmp_path / "tuple_chunks.tif")
    to_geotiff(arr, path, compression="lzw", tiled=True, tile_size=64)

    eager = np.asarray(open_geotiff(path).data)

    da_arr = read_geotiff_gpu(path, chunks=(96, 128))

    computed = _assert_dask_cupy_dask_cupy_combined(
        da_arr,
        expected_chunks=((96, 96), (128, 128)),
        expected_dtype=np.dtype(np.uint16),
    )

    got = computed.data.get()
    np.testing.assert_array_equal(got, eager)


def test_open_geotiff_gpu_chunks_multiband(tmp_path):
    """Combined backend round-trips a 3-band tiled raster.

    Multi-band exercises the planar-config branch in `read_geotiff_gpu`
    that the chunks=None path also walks; without this, a planar-related
    refactor could leave the chunked path with a stale shape.
    """
    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(13)
    arr = rng.randint(0, 256, (128, 192, 3)).astype(np.uint8)
    path = str(tmp_path / "rgb.tif")
    to_geotiff(arr, path, compression="deflate", tiled=True, tile_size=64)

    eager = np.asarray(open_geotiff(path).data)

    da_arr = open_geotiff(path, gpu=True, chunks=64)

    # Multi-band wraps as ('y', 'x', 'band') and chunking only applies to
    # spatial axes; the band axis becomes a single chunk.
    computed = _assert_dask_cupy_dask_cupy_combined(
        da_arr,
        expected_chunks=((64, 64), (64, 64, 64), (3,)),
        expected_dtype=np.dtype(np.uint8),
    )

    got = computed.data.get()
    np.testing.assert_array_equal(got, eager)


def test_open_geotiff_gpu_chunks_partial_last_chunk(tmp_path):
    """Image dimensions not a multiple of `chunks=` keeps the partial chunk."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = np.arange(100 * 150, dtype=np.float32).reshape(100, 150)
    path = str(tmp_path / "partial.tif")
    to_geotiff(arr, path, compression="none", tiled=True, tile_size=32)

    eager = np.asarray(open_geotiff(path).data)

    da_arr = open_geotiff(path, gpu=True, chunks=64)

    computed = _assert_dask_cupy_dask_cupy_combined(
        da_arr,
        expected_chunks=((64, 36), (64, 64, 22)),
        expected_dtype=np.dtype(np.float32),
    )

    got = computed.data.get()
    np.testing.assert_array_equal(got, eager)


def test_open_geotiff_gpu_chunks_preserves_geo_attrs(tmp_path):
    """CRS + transform attrs survive the dask wrap on the gpu+chunks path."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(17)
    arr = rng.rand(128, 128).astype(np.float32)
    path = str(tmp_path / "geo.tif")
    to_geotiff(arr, path, crs=4326, compression="deflate",
               tiled=True, tile_size=64)

    eager = open_geotiff(path)
    da_arr = open_geotiff(path, gpu=True, chunks=64)

    assert da_arr.attrs.get("crs") == eager.attrs.get("crs")
    # Transform tuple should round-trip identically through the gpu+dask
    # path; a missing or modified transform here would break downstream
    # raster math that depends on georeferencing.
    eager_t = eager.attrs.get("transform")
    dask_t = da_arr.attrs.get("transform")
    assert eager_t == dask_t, (
        f"transform mismatch: dask+cupy={dask_t}, eager={eager_t}"
    )

    # Coords align.
    np.testing.assert_array_equal(
        np.asarray(da_arr.coords["y"]), np.asarray(eager.coords["y"]),
    )
    np.testing.assert_array_equal(
        np.asarray(da_arr.coords["x"]), np.asarray(eager.coords["x"]),
    )
