"""Regression tests for issue #1776.

Issue #1752 added ``tile_size`` validation to ``to_geotiff`` and
``chunks`` validation to ``read_geotiff_dask``. The matching kwargs on
three sibling entry points were left unchecked:

* ``write_geotiff_gpu(tile_size=)`` reached ``gpu_compress_tiles`` and
  raised ``ZeroDivisionError`` for ``tile_size=0``, ``struct.error``
  for negative values, and ``TypeError`` for floats -- none naming the
  parameter.
* ``read_geotiff_gpu(chunks=)`` and ``read_vrt(chunks=)`` reached the
  dask chunking code with ``ZeroDivisionError`` for ``chunks=0`` and
  silently accepted negative values, producing a corrupt chunk grid.

The fix factors the shared validators ``_validate_tile_size_arg`` and
``_validate_chunks_arg`` and calls them up front from each entry point,
so all four read paths and both write paths emit the same error format.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    read_geotiff_gpu,
    read_vrt,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


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


def _make_tif(tmp_path) -> str:
    """Write a 10x10 float32 GeoTIFF and return its path."""
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(10), 'x': np.arange(10)},
        attrs={'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)},
    )
    path = os.path.join(str(tmp_path), 'src_1776.tif')
    to_geotiff(da, path)
    return path


def _make_vrt(tmp_path) -> str:
    """Write a 10x10 GeoTIFF plus a single-source VRT and return the .vrt path."""
    tif = _make_tif(tmp_path)
    vrt = os.path.join(str(tmp_path), 'src_1776.vrt')
    write_vrt(vrt, [tif])
    return vrt


# -- write_geotiff_gpu tile_size ------------------------------------------


@_gpu_only
class TestWriteGeotiffGpuTileSize:
    """Mirror ``test_size_param_validation_1752`` for ``write_geotiff_gpu``."""

    @pytest.fixture
    def gpu_da(self):
        import cupy
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        return xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])

    def test_tile_size_zero_raises(self, gpu_da, tmp_path):
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            write_geotiff_gpu(gpu_da, out, tile_size=0)

    def test_tile_size_negative_raises(self, gpu_da, tmp_path):
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            write_geotiff_gpu(gpu_da, out, tile_size=-1)

    def test_tile_size_float_raises(self, gpu_da, tmp_path):
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            write_geotiff_gpu(gpu_da, out, tile_size=256.0)

    def test_tile_size_bool_true_raises(self, gpu_da, tmp_path):
        """``tile_size=True`` is an int subclass that used to silently write
        a 1x1-tile file. Reject it with a clear ValueError."""
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            write_geotiff_gpu(gpu_da, out, tile_size=True)

    def test_tile_size_bool_false_raises(self, gpu_da, tmp_path):
        """``tile_size=False`` was the worst case: ``False == 0`` slipped
        through the integer check and hit ZeroDivisionError downstream."""
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            write_geotiff_gpu(gpu_da, out, tile_size=False)

    def test_tile_size_positive_works(self, gpu_da, tmp_path):
        """tile_size=4 (small but valid) still round-trips."""
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        write_geotiff_gpu(gpu_da, out, tile_size=4)
        assert os.path.exists(out)

    def test_tile_size_numpy_int_scalar_works(self, gpu_da, tmp_path):
        """``np.int64(N)`` is accepted -- matches the
        ``isinstance(x, (int, np.integer))`` rule on ``to_geotiff``."""
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        write_geotiff_gpu(gpu_da, out, tile_size=np.int64(256))
        assert os.path.exists(out)


# -- read_geotiff_gpu chunks ----------------------------------------------


@_gpu_only
class TestReadGeotiffGpuChunks:
    """Mirror ``test_size_param_validation_1752`` for ``read_geotiff_gpu``."""

    def test_chunks_zero_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=0)

    def test_chunks_negative_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=-1)

    def test_chunks_tuple_zero_row_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=(0, 256))

    def test_chunks_tuple_negative_col_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=(256, -1))

    def test_chunks_tuple_wrong_length_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=(64, 64, 64))

    def test_chunks_bool_raises(self, tmp_path):
        """``chunks=True``/``False`` are int subclasses that used to slip
        through. Reject them with the same error as a non-int scalar."""
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=True)

    def test_chunks_non_int_raises(self, tmp_path):
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks='256')

    def test_chunks_tuple_float_raises(self, tmp_path):
        """Tuple entries that aren't int should reject too."""
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_gpu(path, chunks=(64, 64.5))

    def test_positive_int_chunks_works(self, tmp_path):
        path = _make_tif(tmp_path)
        r = read_geotiff_gpu(path, chunks=64)
        assert r.shape == (10, 10)

    def test_positive_tuple_chunks_works(self, tmp_path):
        path = _make_tif(tmp_path)
        r = read_geotiff_gpu(path, chunks=(4, 8))
        assert r.shape == (10, 10)

    def test_numpy_int_scalar_chunks_works(self, tmp_path):
        """``np.int64(N)`` scalar chunk size is accepted -- regression pin
        on the integer-scalar branch of the validator."""
        path = _make_tif(tmp_path)
        r = read_geotiff_gpu(path, chunks=np.int64(64))
        assert r.shape == (10, 10)


# -- read_vrt chunks ------------------------------------------------------


class TestReadVrtChunks:
    """Same matrix as ``TestReadGeotiffGpuChunks`` but for the VRT entry
    point. Runs without CUDA because ``read_vrt(chunks=)`` returns a
    Dask-backed numpy DataArray; no GPU is required."""

    def test_chunks_zero_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=0)

    def test_chunks_negative_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=-1)

    def test_chunks_tuple_zero_row_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=(0, 256))

    def test_chunks_tuple_negative_col_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=(256, -1))

    def test_chunks_tuple_wrong_length_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=(64, 64, 64))

    def test_chunks_bool_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=True)

    def test_chunks_non_int_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks='256')

    def test_chunks_tuple_float_raises(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            read_vrt(vrt, chunks=(64, 64.5))

    def test_positive_int_chunks_works(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        r = read_vrt(vrt, chunks=64)
        assert r.shape == (10, 10)

    def test_positive_tuple_chunks_works(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        r = read_vrt(vrt, chunks=(4, 8))
        assert r.shape == (10, 10)

    def test_numpy_int_scalar_chunks_works(self, tmp_path):
        vrt = _make_vrt(tmp_path)
        r = read_vrt(vrt, chunks=np.int64(64))
        assert r.shape == (10, 10)


# -- open_geotiff(gpu=True, chunks=) routes through read_geotiff_gpu -------


@_gpu_only
class TestOpenGeotiffGpuChunksDispatch:
    """``open_geotiff(gpu=True, chunks=X)`` routes through
    ``read_geotiff_gpu``; pin that the validation propagates through the
    dispatcher so callers using the auto-dispatch entry point still see a
    parameter-named ValueError instead of a deep ZeroDivisionError."""

    def test_open_geotiff_gpu_chunks_zero_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=0)

    def test_open_geotiff_gpu_chunks_negative_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=-1)

    def test_open_geotiff_gpu_chunks_tuple_zero_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=(0, 256))


# -- to_geotiff(gpu=True, tile_size=) was already validated up-front ------


class TestToGeotiffGpuTileSizeAlreadyChecked:
    """``to_geotiff(gpu=True, tile_size=0)`` was already validated by
    #1752's CPU-side check before dispatching to ``write_geotiff_gpu``.
    Pin this so the GPU-writer's new validation does not regress the
    error-message format observed via the auto-dispatch entry point."""

    @_gpu_only
    def test_to_geotiff_gpu_tile_size_zero_raises_same_message(
            self, tmp_path):
        import cupy
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
        out = os.path.join(str(tmp_path), 'out.tif')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, gpu=True, tile_size=0)


# -- ensure no double-validation surprises ---------------------------------


@_gpu_only
class TestNoDoubleValidationSideEffects:
    """The factored ``_validate_chunks_arg`` returns the coerced int when
    given an ``np.integer`` scalar. Tests pin that the GPU read path
    still works end-to-end with that coercion path."""

    def test_read_geotiff_gpu_chunks_numpy_int_no_side_effect(
            self, tmp_path):
        path = _make_tif(tmp_path)
        r = read_geotiff_gpu(path, chunks=np.int64(64))
        assert r.shape == (10, 10)
        # Materialise to confirm the lazy graph is well-formed under
        # cupy + dask.
        out = r.data
        if hasattr(out, 'compute'):
            out.compute()
