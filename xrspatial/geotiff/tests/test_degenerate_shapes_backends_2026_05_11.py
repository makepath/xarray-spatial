"""Backend coverage for degenerate raster shapes and special float values.

The eager numpy path (``test_edge_cases.py``) covers 1x1, 1xN, and Nx1
rasters end-to-end, plus all-NaN, all-Inf, and -Inf inputs. The GPU
(``open_geotiff(gpu=True)``), dask+numpy (``open_geotiff(chunks=N)``),
and dask+cupy (``open_geotiff(gpu=True, chunks=N)``) backends had no
matching coverage. A regression that broke any backend for a 1-pixel
row, 1-column strip, or all-NaN input would not surface until a user
hit the production-traffic mosaic that triggered it.

Test coverage gap sweep 2026-05-11 (pass 5) closes the Cat 3 (geometric
edge case) and Cat 2 (NaN / Inf) gaps for the non-eager backends:

* 1x1, 1xN, Nx1 reads on every backend (Cat 3 HIGH).
* 1x1 and 1xN writes through ``write_geotiff_gpu`` (Cat 3 HIGH for
  the GPU writer's degenerate-shape path).
* All-NaN read on GPU and dask backends (Cat 2 MEDIUM).
* Inf / -Inf read on GPU and dask backends (Cat 2 MEDIUM).
* NaN sentinel mask on dask read path (Cat 2 MEDIUM; the eager path
  has it via ``test_dask_int_nodata_chunks_1597`` for integer nodata
  but no float-NaN equivalent).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    to_geotiff,
    write_geotiff_gpu,
)


# ---------------------------------------------------------------------------
# GPU gating: matches the predicate the rest of the geotiff test suite uses.
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


# ===========================================================================
# Cat 3: 1x1 single-pixel read across backends
# ===========================================================================

class TestSinglePixelRead:
    """1x1 rasters round-trip through every read backend.

    The eager numpy path covers this in ``test_edge_cases.py``. A 1x1
    raster has degenerate stride and tile geometry; the dask path's
    chunk-tile alignment and the GPU path's grid sizing both have
    code that assumes >1 pixel without crashing -- but a future
    refactor could regress either.
    """

    @pytest.fixture
    def single_pixel_path(self, tmp_path):
        arr = np.array([[42.0]], dtype=np.float32)
        p = tmp_path / "single_pixel.tif"
        to_geotiff(arr, str(p))
        return str(p), arr

    def test_dask_numpy_backend(self, single_pixel_path):
        path, arr = single_pixel_path
        # chunks larger than the raster is the documented behaviour
        # (dask collapses to a single chunk that matches the data).
        result = open_geotiff(path, chunks=64)
        assert result.shape == (1, 1)
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)

    def test_read_geotiff_dask_direct(self, single_pixel_path):
        """The explicit ``read_geotiff_dask`` entry point matches dispatch."""
        path, arr = single_pixel_path
        result = read_geotiff_dask(path, chunks=8)
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result.compute().values, arr)

    @_gpu_only
    def test_gpu_backend(self, single_pixel_path):
        path, arr = single_pixel_path
        result = open_geotiff(path, gpu=True)
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_read_geotiff_gpu_direct(self, single_pixel_path):
        """The explicit ``read_geotiff_gpu`` entry point matches dispatch."""
        path, arr = single_pixel_path
        result = read_geotiff_gpu(path)
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_dask_cupy_backend(self, single_pixel_path):
        """dask+cupy must also handle a 1x1 raster.

        The dask graph here has exactly one block (chunks larger than
        the raster) and that block carries a cupy buffer.
        """
        import cupy
        path, arr = single_pixel_path
        result = open_geotiff(path, gpu=True, chunks=64)
        assert result.shape == (1, 1)
        computed = result.compute()
        assert isinstance(computed.data, cupy.ndarray)
        np.testing.assert_array_equal(computed.data.get(), arr)


# ===========================================================================
# Cat 3: 1xN single-row read across backends
# ===========================================================================

class TestSingleRowRead:
    """1xN rasters round-trip through every read backend.

    Single-row tiles trigger the strip-fallback path in the GPU decoder
    when there is no tiled layout, and a 1-row chunk in the dask graph.
    """

    @pytest.fixture
    def single_row_path(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(1, 10)
        p = tmp_path / "single_row.tif"
        to_geotiff(arr, str(p))
        return str(p), arr

    def test_dask_numpy_backend(self, single_row_path):
        path, arr = single_row_path
        result = open_geotiff(path, chunks=4)
        assert result.shape == (1, 10)
        np.testing.assert_array_equal(result.compute().values, arr)

    @_gpu_only
    def test_gpu_backend(self, single_row_path):
        path, arr = single_row_path
        result = open_geotiff(path, gpu=True)
        assert result.shape == (1, 10)
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_dask_cupy_backend(self, single_row_path):
        import cupy
        path, arr = single_row_path
        result = open_geotiff(path, gpu=True, chunks=4)
        assert result.shape == (1, 10)
        computed = result.compute()
        assert isinstance(computed.data, cupy.ndarray)
        np.testing.assert_array_equal(computed.data.get(), arr)


# ===========================================================================
# Cat 3: Nx1 single-column read across backends
# ===========================================================================

class TestSingleColumnRead:
    """Nx1 rasters round-trip through every read backend.

    Single-column tiles are the mirror case of single-row, and exercise
    the row-major iteration order in the dask block-builder and the
    GPU's window-band slice path.
    """

    @pytest.fixture
    def single_column_path(self, tmp_path):
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)
        p = tmp_path / "single_column.tif"
        to_geotiff(arr, str(p))
        return str(p), arr

    def test_dask_numpy_backend(self, single_column_path):
        path, arr = single_column_path
        result = open_geotiff(path, chunks=4)
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result.compute().values, arr)

    @_gpu_only
    def test_gpu_backend(self, single_column_path):
        path, arr = single_column_path
        result = open_geotiff(path, gpu=True)
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_dask_cupy_backend(self, single_column_path):
        import cupy
        path, arr = single_column_path
        result = open_geotiff(path, gpu=True, chunks=4)
        assert result.shape == (10, 1)
        computed = result.compute()
        assert isinstance(computed.data, cupy.ndarray)
        np.testing.assert_array_equal(computed.data.get(), arr)


# ===========================================================================
# Cat 3: 1x1 and 1xN writes through write_geotiff_gpu
# ===========================================================================

@_gpu_only
class TestGpuWriterDegenerateShapes:
    """``write_geotiff_gpu`` must accept 1-pixel and 1-row inputs.

    The GPU writer's tile-encoding path uses an internal grid sizing
    helper that fell back to host code for shapes smaller than the
    default tile. The fallback exists but had no regression test that
    would catch a future "fast-path only" refactor.
    """

    def test_single_pixel_round_trip(self, tmp_path):
        import cupy
        arr = cupy.array([[42.0]], dtype=cupy.float32)
        da_gpu = xr.DataArray(arr, dims=["y", "x"])
        p = str(tmp_path / "gpu_1x1.tif")
        write_geotiff_gpu(da_gpu, p)

        result = open_geotiff(p)
        assert result.shape == (1, 1)
        assert result.values[0, 0] == 42.0

    def test_single_row_round_trip(self, tmp_path):
        import cupy
        arr_np = np.arange(10, dtype=np.float32).reshape(1, 10)
        arr = cupy.asarray(arr_np)
        da_gpu = xr.DataArray(arr, dims=["y", "x"])
        p = str(tmp_path / "gpu_1xN.tif")
        write_geotiff_gpu(da_gpu, p)

        result = open_geotiff(p)
        assert result.shape == (1, 10)
        np.testing.assert_array_equal(result.values, arr_np)

    def test_single_column_round_trip(self, tmp_path):
        import cupy
        arr_np = np.arange(10, dtype=np.float32).reshape(10, 1)
        arr = cupy.asarray(arr_np)
        da_gpu = xr.DataArray(arr, dims=["y", "x"])
        p = str(tmp_path / "gpu_Nx1.tif")
        write_geotiff_gpu(da_gpu, p)

        result = open_geotiff(p)
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result.values, arr_np)


# ===========================================================================
# Cat 2: all-NaN / Inf reads on GPU and dask backends
# ===========================================================================

class TestAllNanRead:
    """All-NaN raster (boundary of the algorithm) reads cleanly on every
    backend.

    The eager path covers this in ``test_edge_cases.TestWriteSpecialValues``.
    Without a matching GPU/dask test, a regression in the GPU nodata
    masker or dask graph builder would only surface in production.
    """

    @pytest.fixture
    def all_nan_path(self, tmp_path):
        arr = np.full((8, 8), np.nan, dtype=np.float32)
        p = tmp_path / "all_nan.tif"
        to_geotiff(arr, str(p), nodata=float("nan"))
        return str(p), arr

    def test_dask_numpy_backend(self, all_nan_path):
        path, _ = all_nan_path
        result = open_geotiff(path, chunks=4)
        computed = result.compute()
        assert np.all(np.isnan(computed.values))

    @_gpu_only
    def test_gpu_backend(self, all_nan_path):
        path, _ = all_nan_path
        result = open_geotiff(path, gpu=True)
        assert np.all(np.isnan(result.data.get()))

    @_gpu_only
    def test_dask_cupy_backend(self, all_nan_path):
        path, _ = all_nan_path
        result = open_geotiff(path, gpu=True, chunks=4)
        computed = result.compute()
        assert np.all(np.isnan(computed.data.get()))


class TestInfRead:
    """+Inf and -Inf are valid float values in TIFF; they must survive
    every read backend without being masked or clipped.

    The eager path's ``test_edge_cases.TestWriteSpecialValues::test_nan_and_inf``
    is a write-then-CPU-read test. The GPU and dask backends were
    unexercised on Inf input.
    """

    @pytest.fixture
    def inf_path(self, tmp_path):
        arr = np.array(
            [
                [np.inf, -np.inf, 1.0, 2.0],
                [3.0, np.inf, -np.inf, 4.0],
                [-np.inf, 5.0, 6.0, np.inf],
                [7.0, 8.0, np.inf, 9.0],
            ],
            dtype=np.float32,
        )
        p = tmp_path / "inf.tif"
        # Do not set nodata: we want Inf to survive, not be remapped.
        to_geotiff(arr, str(p))
        return str(p), arr

    def test_dask_numpy_backend(self, inf_path):
        path, arr = inf_path
        result = open_geotiff(path, chunks=2).compute()
        assert np.isposinf(result.values[0, 0])
        assert np.isneginf(result.values[0, 1])
        np.testing.assert_array_equal(result.values, arr)

    @_gpu_only
    def test_gpu_backend(self, inf_path):
        path, arr = inf_path
        result = open_geotiff(path, gpu=True)
        host = result.data.get()
        assert np.isposinf(host[0, 0])
        assert np.isneginf(host[0, 1])
        np.testing.assert_array_equal(host, arr)

    @_gpu_only
    def test_dask_cupy_backend(self, inf_path):
        path, arr = inf_path
        result = open_geotiff(path, gpu=True, chunks=2)
        host = result.compute().data.get()
        assert np.isposinf(host[0, 0])
        assert np.isneginf(host[0, 1])
        np.testing.assert_array_equal(host, arr)


class TestNanSentinelDaskRead:
    """Float raster with ``nodata=NaN`` sentinel reads consistently
    across backends.

    The integer-sentinel equivalent is pinned by issue #1597. The
    float path has no such per-chunk dtype divergence (the input is
    already float), but the dask graph still has to forward the
    sentinel substitution. A regression in the float branch of
    ``_delayed_read_window`` would silently break this.
    """

    @pytest.fixture
    def nan_sentinel_path(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        arr[2:4, 2:4] = -9999.0
        arr[6, 0] = -9999.0
        p = tmp_path / "nan_sentinel_float.tif"
        to_geotiff(arr, str(p), nodata=-9999.0)
        return str(p), arr

    def test_eager_path_baseline(self, nan_sentinel_path):
        """Baseline: eager path replaces the sentinel with NaN."""
        path, _ = nan_sentinel_path
        result = open_geotiff(path)
        assert np.isnan(result.values[2, 2])
        assert np.isnan(result.values[6, 0])
        assert result.values[0, 0] == 0.0  # non-sentinel survives

    def test_dask_numpy_matches_eager(self, nan_sentinel_path):
        """dask compute reproduces the eager mask exactly."""
        path, _ = nan_sentinel_path
        eager = open_geotiff(path)
        dk = open_geotiff(path, chunks=4).compute()
        np.testing.assert_array_equal(np.isnan(dk.values), np.isnan(eager.values))
        finite = ~np.isnan(eager.values)
        np.testing.assert_array_equal(dk.values[finite], eager.values[finite])

    def test_dask_numpy_chunks_smaller_than_sentinel_block(self, nan_sentinel_path):
        """Sentinels split across two chunks still mask correctly.

        The 2x2 sentinel block at rows 2-3 cols 2-3 lands in a single
        chunk for chunks=4 (rows 0-3) but straddles a chunk boundary
        for chunks=2 (rows 2-3 split between chunks 1 and 2). This
        exercises the per-block sentinel comparison.
        """
        path, _ = nan_sentinel_path
        dk = open_geotiff(path, chunks=2).compute()
        assert np.isnan(dk.values[2, 2])
        assert np.isnan(dk.values[3, 3])
        assert np.isnan(dk.values[6, 0])
