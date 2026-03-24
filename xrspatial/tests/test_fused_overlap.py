"""Tests for fused_overlap and helpers."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import _normalize_depth, _pad_nan


class TestNormalizeDepth:
    def test_int_input(self):
        assert _normalize_depth(2, ndim=2) == {0: 2, 1: 2}

    def test_tuple_input(self):
        assert _normalize_depth((3, 1), ndim=2) == {0: 3, 1: 1}

    def test_dict_input(self):
        assert _normalize_depth({0: 2, 1: 4}, ndim=2) == {0: 2, 1: 4}

    def test_dict_missing_axis_raises(self):
        with pytest.raises(ValueError, match="missing axes"):
            _normalize_depth({0: 1}, ndim=2)

    def test_dict_extra_axis_raises(self):
        with pytest.raises(ValueError, match="extra axes"):
            _normalize_depth({0: 1, 1: 1, 2: 1}, ndim=2)

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_depth(-1, ndim=2)

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _normalize_depth((1, 2, 3), ndim=2)


class TestPadNan:
    def test_2d_pads_with_nan(self):
        data = np.ones((4, 4), dtype=np.float32)
        result = _pad_nan(data, depth=(1, 1))
        assert result.shape == (6, 6)
        assert np.isnan(result[0, 0])
        np.testing.assert_array_equal(result[1:-1, 1:-1], data)

    def test_asymmetric_depth(self):
        data = np.ones((4, 4), dtype=np.float32)
        result = _pad_nan(data, depth=(2, 1))
        assert result.shape == (8, 6)

    def test_integer_dtype_promotes_to_float(self):
        data = np.ones((4, 4), dtype=np.int32)
        result = _pad_nan(data, depth=(1, 1))
        assert np.issubdtype(result.dtype, np.floating)


da = pytest.importorskip("dask.array")


def _increment_interior(chunk):
    """Stage func: adds 1 to every cell. Returns interior only."""
    return chunk[1:-1, 1:-1] + 1


def _double_interior(chunk):
    """Stage func: doubles every cell. Returns interior only."""
    return chunk[1:-1, 1:-1] * 2


def _make_dask_raster(shape=(64, 64), chunks=16, dtype=np.float32):
    data = da.from_array(
        np.random.RandomState(42).rand(*shape).astype(dtype), chunks=chunks
    )
    return xr.DataArray(data, dims=['y', 'x'])


class TestFusedOverlapDask:
    def test_single_stage_matches_map_overlap(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        fused = fused_overlap(raster, (_increment_interior, 1))
        ref = raster.data.map_overlap(
            _increment_interior, depth=1, boundary=np.nan, trim=False,
            meta=np.array(()),
        )
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_two_stages_match_sequential(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        fused = fused_overlap(raster, (_increment_interior, 1), (_double_interior, 1))
        step1 = raster.data.map_overlap(_increment_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        ref = step1.map_overlap(_double_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_three_stages(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        fused = fused_overlap(raster, (_increment_interior, 1), (_double_interior, 1), (_increment_interior, 1))
        step1 = raster.data.map_overlap(_increment_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        step2 = step1.map_overlap(_double_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        ref = step2.map_overlap(_increment_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        # Allow small float32 rounding differences at chunk boundaries
        np.testing.assert_allclose(fused.values, ref.compute(), atol=1e-6, rtol=1e-6)

    def test_nonsquare_depth(self):
        from xrspatial.utils import fused_overlap
        def _stage_2_1(chunk):
            return chunk[2:-2, 1:-1] + 1
        raster = _make_dask_raster(shape=(64, 64), chunks=32)
        fused = fused_overlap(raster, (_stage_2_1, (2, 1)))
        ref = raster.data.map_overlap(_stage_2_1, depth=(2, 1), boundary=np.nan, trim=False, meta=np.array(()))
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_returns_dataarray(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        result = fused_overlap(raster, (_increment_interior, 1))
        assert isinstance(result, xr.DataArray)

    def test_fewer_graph_layers_than_sequential(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        fused = fused_overlap(raster, (_increment_interior, 1), (_double_interior, 1))
        step1 = raster.data.map_overlap(_increment_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        sequential = step1.map_overlap(_double_interior, depth=1, boundary=np.nan, trim=False, meta=np.array(()))
        assert len(dict(fused.data.__dask_graph__())) < len(dict(sequential.__dask_graph__()))


class TestFusedOverlapNumpy:
    def test_numpy_fallback_matches_dask(self):
        from xrspatial.utils import fused_overlap
        np_raster = xr.DataArray(np.random.RandomState(42).rand(64, 64).astype(np.float32), dims=['y', 'x'])
        dask_raster = np_raster.chunk(16)
        np_result = fused_overlap(np_raster, (_increment_interior, 1), (_double_interior, 1))
        dask_result = fused_overlap(dask_raster, (_increment_interior, 1), (_double_interior, 1))
        np.testing.assert_array_equal(np_result.values[2:-2, 2:-2], dask_result.values[2:-2, 2:-2])


class TestFusedOverlapValidation:
    def test_rejects_non_nan_boundary(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="boundary.*nan"):
            fused_overlap(raster, (_increment_interior, 1), boundary='nearest')

    def test_rejects_empty_stages(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="at least one stage"):
            fused_overlap(raster)

    def test_rejects_non_dataarray(self):
        from xrspatial.utils import fused_overlap
        with pytest.raises(TypeError):
            fused_overlap(np.zeros((10, 10)), (_increment_interior, 1))

    def test_rejects_chunks_smaller_than_total_depth(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster(shape=(32, 32), chunks=4)
        def _big_depth(chunk):
            return chunk[5:-5, 5:-5] + 1
        with pytest.raises(ValueError, match="[Cc]hunk size"):
            fused_overlap(raster, (_big_depth, 5))

    def test_small_chunks_barely_above_total_depth(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster(shape=(24, 24), chunks=6)
        result = fused_overlap(raster, (_increment_interior, 1), (_double_interior, 1))
        assert result.shape == (24, 24)


class TestFusedOverlapAccessor:
    def test_accessor_delegates(self):
        import xrspatial  # noqa: F401
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        direct = fused_overlap(raster, (_increment_interior, 1))
        via_acc = raster.xrs.fused_overlap((_increment_interior, 1))
        np.testing.assert_array_equal(direct.values, via_acc.values)
