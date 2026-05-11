"""Backend / parameter coverage for the VRT read path.

The non-VRT read backends (``open_geotiff`` / ``read_geotiff_dask`` /
``read_geotiff_gpu``) all have dedicated multi-backend coverage; the
VRT route through ``read_vrt`` historically lacked it. The eager
numpy path has dense coverage, but the GPU and dask+GPU paths the
``read_vrt`` body explicitly handles (the ``if gpu: cupy.asarray``
and trailing ``result.chunk(...)`` blocks) were only reachable
indirectly via ``open_geotiff('.vrt', gpu=True)`` / ``..., chunks=N)``
and went untested.

The error-rejection branches for file-like sources combined with
``gpu=True`` / ``chunks=N`` on ``open_geotiff`` were likewise covered
only by inspection.

Test coverage gap sweep 2026-05-11 (pass 3): close the VRT backend
coverage gap and the file-like-rejection parameter gaps.
"""
from __future__ import annotations

import importlib.util
import io
import os

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_vrt, to_geotiff
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


# ---------------------------------------------------------------------------
# GPU gating: matches the ``_gpu_available`` / ``_HAS_GPU`` predicate that
# the rest of the geotiff test suite (e.g. test_backend_kwarg_parity_1561,
# test_attrs_parity_1548) uses, so future GPU tests stay greppable.
# ---------------------------------------------------------------------------

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_tile_vrt(tmp_path):
    """A trivial single-tile VRT plus its source array.

    Float32 source so the VRT band advertises Float32 and the eager
    numpy read returns float32 (lets dtype-cast tests assert a real
    type change).
    """
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    tile_path = str(tmp_path / 'tile.tif')
    to_geotiff(arr, tile_path)
    vrt_path = str(tmp_path / 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    return vrt_path, arr


# ---------------------------------------------------------------------------
# Cat 1: read_vrt backend coverage (GPU + dask+GPU)
# ---------------------------------------------------------------------------

@_gpu_only
class TestReadVrtGpuBackend:
    """``read_vrt(gpu=True)`` returns a CuPy-backed DataArray.

    The eager VRT decode runs on the CPU (the VRT internal reader
    walks SimpleSources and assembles via windowed reads) then the
    final ``if gpu: arr = cupy.asarray(arr)`` block transfers to GPU.
    A regression that dropped the transfer block would have shipped
    a numpy DataArray instead of a CuPy one; this test pins that.
    """

    def test_read_vrt_gpu_returns_cupy(self, single_tile_vrt):
        import cupy

        vrt_path, arr = single_tile_vrt
        da = read_vrt(vrt_path, gpu=True)
        assert isinstance(da.data, cupy.ndarray), (
            f"expected cupy.ndarray, got {type(da.data).__name__}"
        )
        np.testing.assert_array_equal(da.data.get(), arr)

    def test_read_vrt_gpu_chunks_returns_dask_cupy(self, single_tile_vrt):
        """``read_vrt(gpu=True, chunks=N)`` is the documented dask+cupy
        VRT entry point. The trailing ``result.chunk(...)`` block has
        to wrap the cupy backing without falling back to numpy.
        """
        import cupy
        import dask.array as da_mod

        vrt_path, arr = single_tile_vrt
        result = read_vrt(vrt_path, gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array), (
            f"expected dask Array, got {type(result.data).__name__}"
        )
        # _meta tells distributed Dask the underlying array is cupy.
        # A numpy meta here would cause optimizers to silently move
        # data back to host.
        assert isinstance(result.data._meta, cupy.ndarray), (
            f"expected cupy._meta, got "
            f"{type(result.data._meta).__module__}."
            f"{type(result.data._meta).__name__}"
        )
        # Chunks honour the spatial spec; the band axis (absent here)
        # would chunk as a single block.
        assert result.data.chunks == ((2, 2), (2, 2))

        computed = result.compute()
        assert isinstance(computed.data, cupy.ndarray)
        np.testing.assert_array_equal(computed.data.get(), arr)

    def test_open_geotiff_vrt_gpu_routes_through(self, single_tile_vrt):
        """``open_geotiff('.vrt', gpu=True)`` dispatches to ``read_vrt``
        and surfaces the cupy data unchanged. The dispatcher branch
        is one line in ``open_geotiff`` but a refactor that dropped
        ``gpu=gpu`` from the forwarded kwargs would silently produce
        a numpy DataArray.
        """
        import cupy

        vrt_path, arr = single_tile_vrt
        da = open_geotiff(vrt_path, gpu=True)
        assert isinstance(da.data, cupy.ndarray)
        np.testing.assert_array_equal(da.data.get(), arr)

    def test_open_geotiff_vrt_gpu_chunks(self, single_tile_vrt):
        """``open_geotiff('.vrt', gpu=True, chunks=N)`` is the combined
        dask+cupy entry point. Same dispatch test as the gpu-only
        variant but also pins the chunk forwarding.
        """
        import cupy
        import dask.array as da_mod

        vrt_path, arr = single_tile_vrt
        result = open_geotiff(vrt_path, gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert isinstance(result.data._meta, cupy.ndarray)
        assert result.data.chunks == ((2, 2), (2, 2))

        computed = result.compute()
        np.testing.assert_array_equal(computed.data.get(), arr)


# ---------------------------------------------------------------------------
# Cat 4: read_vrt parameter coverage (dtype / name)
# ---------------------------------------------------------------------------

class TestReadVrtDtypeKwarg:
    """``read_vrt(dtype=...)`` casts after decode and validates the cast."""

    def test_safe_widening_cast(self, single_tile_vrt):
        """float32 -> float64 is permitted; values survive bit-for-bit."""
        vrt_path, arr = single_tile_vrt
        da = read_vrt(vrt_path, dtype='float64')
        assert da.dtype == np.float64
        np.testing.assert_array_equal(da.values, arr.astype(np.float64))

    def test_float_to_int_rejected(self, single_tile_vrt):
        """Float-to-int is lossy and refused with a descriptive error.
        Mirrors ``open_geotiff(dtype=...)`` behaviour so callers see the
        same gate on both entry points.
        """
        vrt_path, _ = single_tile_vrt
        with pytest.raises(ValueError, match="Cannot cast float"):
            read_vrt(vrt_path, dtype='int32')


class TestReadVrtNameKwarg:
    """``read_vrt(name='custom')`` overrides the file-stem derivation."""

    def test_explicit_name_used(self, single_tile_vrt):
        vrt_path, _ = single_tile_vrt
        da = read_vrt(vrt_path, name='custom_name')
        assert da.name == 'custom_name'

    def test_default_name_from_stem(self, single_tile_vrt):
        vrt_path, _ = single_tile_vrt
        da = read_vrt(vrt_path)
        # mosaic.vrt -> mosaic
        assert da.name == os.path.splitext(os.path.basename(vrt_path))[0]


# ---------------------------------------------------------------------------
# Cat 4: open_geotiff file-like + backend kwarg rejection
# ---------------------------------------------------------------------------

class TestOpenGeotiffFileLikeKwargRejection:
    """File-like sources reject ``gpu=True`` and ``chunks=N`` up front.

    The check sits in ``open_geotiff`` (not the underlying readers)
    because both downstream paths re-open the source by path from
    worker tasks. A buffer passed through would either raise deep
    inside dask graph construction or silently behave as if the
    buffer were a string path.
    """

    @staticmethod
    def _buf_with_tiff(tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'src.tif')
        to_geotiff(arr, path)
        with open(path, 'rb') as fh:
            return io.BytesIO(fh.read())

    def test_gpu_with_file_like_raises(self, tmp_path):
        buf = self._buf_with_tiff(tmp_path)
        with pytest.raises(ValueError, match="gpu=True is not supported"):
            open_geotiff(buf, gpu=True)

    def test_chunks_with_file_like_raises(self, tmp_path):
        buf = self._buf_with_tiff(tmp_path)
        with pytest.raises(ValueError, match="chunks=.*file-like"):
            open_geotiff(buf, chunks=64)

    def test_chunks_with_pathlib_path_still_works(self, tmp_path):
        """Sanity-check: pathlib.Path is not file-like and must keep
        working through the dask path. Otherwise the file-like gate
        would also lock out Path inputs.
        """
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = tmp_path / 'sample.tif'
        to_geotiff(arr, str(path))

        import dask.array as da_mod
        result = open_geotiff(path, chunks=2)
        assert isinstance(result.data, da_mod.Array)
        np.testing.assert_array_equal(np.asarray(result.data), arr)
