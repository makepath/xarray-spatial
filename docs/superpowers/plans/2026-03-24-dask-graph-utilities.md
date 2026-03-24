# Dask Graph Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `fused_overlap` and `multi_overlap` utilities that reduce dask graph size by fusing multiple `map_overlap` calls into single passes.

**Architecture:** Two standalone functions in `xrspatial/utils.py`. `fused_overlap` composes sequential overlap stages into one `map_overlap` call with summed depth. `multi_overlap` runs a multi-output kernel via `da.overlap.overlap()` + `da.map_blocks()` with `new_axis=0`. Both are exposed on the DataArray `.xrs` accessor.

**Tech Stack:** dask.array (map_overlap, overlap.overlap, map_blocks), numpy, xarray

**Spec:** `docs/superpowers/specs/2026-03-24-dask-graph-utilities-design.md`

---

## File structure

| File | Role |
|------|------|
| `xrspatial/utils.py` | Add `_normalize_depth`, `_pad_nan`, `fused_overlap`, `multi_overlap` after existing `rechunk_no_shuffle` (line 1102) |
| `xrspatial/accessor.py` | Add `fused_overlap`, `multi_overlap` to DataArray accessor after `rechunk_no_shuffle` (line 501) |
| `xrspatial/__init__.py` | Export both functions |
| `xrspatial/tests/test_fused_overlap.py` | New test file |
| `xrspatial/tests/test_multi_overlap.py` | New test file |
| `xrspatial/tests/test_accessor.py` | Add to expected methods list (line 93) |
| `docs/source/reference/utilities.rst` | Add API entries |
| `README.md` | Add rows to Utilities table (line 289) |
| `examples/user_guide/37_Fused_Overlap.ipynb` | New notebook |

---

### Task 1: `_normalize_depth` and `_pad_nan` helpers

**Files:**
- Modify: `xrspatial/utils.py` (append after line 1102)
- Test: `xrspatial/tests/test_fused_overlap.py` (new)

- [ ] **Step 1: Write failing tests for `_normalize_depth`**

Create `xrspatial/tests/test_fused_overlap.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_fused_overlap.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `_normalize_depth` and `_pad_nan`**

Append to `xrspatial/utils.py` after line 1102:

```python
def _normalize_depth(depth, ndim):
    """Normalize depth to a dict {axis: int}.

    Accepts int, tuple, or dict.  Validates completeness and
    non-negativity.
    """
    if isinstance(depth, dict):
        expected = set(range(ndim))
        got = set(depth.keys())
        missing = expected - got
        extra = got - expected
        if missing:
            raise ValueError(
                f"_normalize_depth: missing axes {sorted(missing)} "
                f"for ndim={ndim}"
            )
        if extra:
            raise ValueError(
                f"_normalize_depth: extra axes {sorted(extra)} "
                f"for ndim={ndim}"
            )
        for v in depth.values():
            if v < 0:
                raise ValueError(
                    f"_normalize_depth: depth must be non-negative, got {v}"
                )
        return dict(depth)

    if isinstance(depth, int):
        if depth < 0:
            raise ValueError(
                f"_normalize_depth: depth must be non-negative, got {depth}"
            )
        return {ax: depth for ax in range(ndim)}

    if isinstance(depth, tuple):
        if len(depth) != ndim:
            raise ValueError(
                f"_normalize_depth: tuple length {len(depth)} != ndim {ndim}"
            )
        for v in depth:
            if v < 0:
                raise ValueError(
                    f"_normalize_depth: depth must be non-negative, got {v}"
                )
        return {ax: d for ax, d in enumerate(depth)}

    raise TypeError(
        f"_normalize_depth: expected int, tuple, or dict, got {type(depth).__name__}"
    )


def _pad_nan(data, depth):
    """Pad a 2-D numpy or cupy array with NaN on each side.

    Parameters
    ----------
    data : numpy or cupy array
    depth : tuple of int
        ``(d0, d1)`` cells to pad per axis.
    """
    pad_width = tuple((d, d) for d in depth)
    if is_cupy_array(data):
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(cupy.float64)
        out = cupy.pad(data, pad_width, mode='constant',
                       constant_values=np.nan)
    else:
        # Promote integer dtypes so NaN fill works
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64)
        out = np.pad(data, pad_width, mode='constant',
                     constant_values=np.nan)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_fused_overlap.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/utils.py xrspatial/tests/test_fused_overlap.py
git commit -m "Add _normalize_depth and _pad_nan helpers"
```

---

### Task 2: `fused_overlap` implementation

**Files:**
- Modify: `xrspatial/utils.py` (append after `_pad_nan`)
- Modify: `xrspatial/tests/test_fused_overlap.py` (add tests)

- [ ] **Step 1: Write failing tests for `fused_overlap`**

Append to `xrspatial/tests/test_fused_overlap.py`:

```python
da = pytest.importorskip("dask.array")


def _increment_interior(chunk):
    """Stage func: adds 1 to every cell. Returns interior only."""
    # depth=1 means chunk is (H+2, W+2), interior is (H, W)
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

        # Sequential reference
        ref = raster.data.map_overlap(
            _increment_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_two_stages_match_sequential(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()

        fused = fused_overlap(
            raster,
            (_increment_interior, 1),
            (_double_interior, 1),
        )

        # Sequential reference
        step1 = raster.data.map_overlap(
            _increment_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        ref = step1.map_overlap(
            _double_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_three_stages(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()

        fused = fused_overlap(
            raster,
            (_increment_interior, 1),
            (_double_interior, 1),
            (_increment_interior, 1),
        )

        step1 = raster.data.map_overlap(
            _increment_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        step2 = step1.map_overlap(
            _double_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        ref = step2.map_overlap(
            _increment_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_nonsquare_depth(self):
        from xrspatial.utils import fused_overlap

        def _stage_2_1(chunk):
            return chunk[2:-2, 1:-1] + 1

        raster = _make_dask_raster(shape=(64, 64), chunks=32)
        fused = fused_overlap(raster, (_stage_2_1, (2, 1)))

        ref = raster.data.map_overlap(
            _stage_2_1, depth=(2, 1), boundary=np.nan,
            meta=np.array(()),
        )
        np.testing.assert_array_equal(fused.values, ref.compute())

    def test_returns_dataarray(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        result = fused_overlap(raster, (_increment_interior, 1))
        assert isinstance(result, xr.DataArray)

    def test_fewer_graph_layers_than_sequential(self):
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()

        fused = fused_overlap(
            raster,
            (_increment_interior, 1),
            (_double_interior, 1),
        )

        step1 = raster.data.map_overlap(
            _increment_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        sequential = step1.map_overlap(
            _double_interior, depth=1, boundary=np.nan,
            meta=np.array(()),
        )
        assert len(dict(fused.data.__dask_graph__())) < len(
            dict(sequential.__dask_graph__())
        )


class TestFusedOverlapNumpy:
    def test_numpy_fallback_matches_dask(self):
        from xrspatial.utils import fused_overlap
        np_raster = xr.DataArray(
            np.random.RandomState(42).rand(64, 64).astype(np.float32),
            dims=['y', 'x'],
        )
        dask_raster = np_raster.chunk(16)

        np_result = fused_overlap(
            np_raster,
            (_increment_interior, 1),
            (_double_interior, 1),
        )
        dask_result = fused_overlap(
            dask_raster,
            (_increment_interior, 1),
            (_double_interior, 1),
        )
        # Compare interior (edges may differ due to NaN propagation,
        # but interior cells should match)
        np.testing.assert_array_equal(
            np_result.values[2:-2, 2:-2],
            dask_result.values[2:-2, 2:-2],
        )


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
        # total_depth = 5, chunks = 4
        def _big_depth(chunk):
            return chunk[5:-5, 5:-5] + 1
        with pytest.raises(ValueError, match="[Cc]hunk size"):
            fused_overlap(raster, (_big_depth, 5))

    def test_small_chunks_barely_above_total_depth(self):
        """Chunks just barely larger than total_depth should work."""
        from xrspatial.utils import fused_overlap
        # chunks=6, total_depth=2 (two stages of depth 1)
        raster = _make_dask_raster(shape=(24, 24), chunks=6)
        result = fused_overlap(
            raster,
            (_increment_interior, 1),
            (_double_interior, 1),
        )
        assert result.shape == (24, 24)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_fused_overlap.py::TestFusedOverlapDask -v`
Expected: FAIL with ImportError (`fused_overlap` not found)

- [ ] **Step 3: Implement `fused_overlap`**

Append to `xrspatial/utils.py` after `_pad_nan`:

```python
def fused_overlap(agg, *stages, boundary='nan'):
    """Run multiple overlap operations in a single map_overlap call.

    Each stage is a ``(func, depth)`` pair. ``func`` receives a padded
    chunk and returns the unpadded interior result.  Stages are fused
    into one ``map_overlap`` call with the sum of all depths, producing
    one blockwise graph layer instead of N.

    Parameters
    ----------
    agg : xr.DataArray
        Input raster.
    *stages : tuple of (callable, depth)
        Each ``func`` takes array ``(H+2*d, W+2*d)`` -> ``(H, W)``.
        ``depth`` is int, tuple, or dict.
    boundary : str
        Must be ``'nan'``.

    Returns
    -------
    xr.DataArray
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"fused_overlap(): expected xr.DataArray, "
            f"got {type(agg).__name__}"
        )
    if not stages:
        raise ValueError("fused_overlap(): need at least one stage")
    if boundary != 'nan':
        raise ValueError(
            f"fused_overlap(): boundary must be 'nan', got {boundary!r}"
        )

    ndim = agg.ndim

    # Normalize and sum depths
    stage_depths = [_normalize_depth(d, ndim) for _, d in stages]
    total_depth = {ax: sum(sd[ax] for sd in stage_depths)
                   for ax in range(ndim)}

    # --- non-dask path ---
    if not has_dask_array() or not isinstance(agg.data, da.Array):
        result = agg.data
        for i, (func, _) in enumerate(stages):
            depth_tuple = tuple(stage_depths[i][ax] for ax in range(ndim))
            padded = _pad_nan(result, depth_tuple)
            result = func(padded)
        return agg.copy(data=result)

    # --- dask path ---
    # Validate chunk sizes
    for ax, d in total_depth.items():
        for cs in agg.chunks[ax]:
            if cs < d:
                raise ValueError(
                    f"Chunk size {cs} on axis {ax} is smaller than "
                    f"total depth {d}. Rechunk first."
                )

    funcs = [f for f, _ in stages]

    def _fused_wrapper(block):
        result = block
        for func in funcs:
            result = func(result)
            # result is now the interior; it still has enough valid
            # overlap for the remaining stages
        # re-pad to original block shape so map_overlap can crop
        pad_width = tuple((total_depth[ax], total_depth[ax])
                          for ax in range(block.ndim))
        if is_cupy_array(result):
            padded = cupy.pad(result, pad_width, mode='constant',
                              constant_values=np.nan)
        else:
            padded = np.pad(result, pad_width, mode='constant',
                            constant_values=np.nan)
        return padded

    out = agg.data.map_overlap(
        _fused_wrapper,
        depth=total_depth,
        boundary=np.nan,
        meta=np.array(()),
    )

    return agg.copy(data=out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_fused_overlap.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/utils.py xrspatial/tests/test_fused_overlap.py
git commit -m "Add fused_overlap utility"
```

---

### Task 3: `multi_overlap` implementation

**Files:**
- Modify: `xrspatial/utils.py` (append after `fused_overlap`)
- Create: `xrspatial/tests/test_multi_overlap.py`

- [ ] **Step 1: Write failing tests for `multi_overlap`**

Create `xrspatial/tests/test_multi_overlap.py`:

```python
"""Tests for multi_overlap."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import multi_overlap

da = pytest.importorskip("dask.array")


def _triple_kernel(chunk):
    """Return 3 bands from a padded (H+2, W+2) chunk."""
    interior = chunk[1:-1, 1:-1]
    return np.stack([interior + 1, interior * 2, interior - 1], axis=0)


def _make_dask_raster(shape=(64, 64), chunks=16, dtype=np.float32):
    data = da.from_array(
        np.random.RandomState(99).rand(*shape).astype(dtype), chunks=chunks
    )
    return xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.arange(shape[0]), 'x': np.arange(shape[1])},
        attrs={'crs': 'EPSG:4326'},
    )


class TestMultiOverlapDask:
    def test_matches_sequential_stack(self):
        raster = _make_dask_raster()

        multi = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

        # Sequential reference: run kernel 3 times extracting one band each
        def _band_i(chunk, i=0):
            return _triple_kernel(chunk)[i]

        bands = []
        for i in range(3):
            from functools import partial
            b = raster.data.map_overlap(
                partial(_band_i, i=i), depth=1, boundary=np.nan,
                meta=np.array(()),
            )
            bands.append(b)
        ref = da.stack(bands, axis=0).compute()

        np.testing.assert_array_equal(multi.values, ref)

    def test_output_shape(self):
        raster = _make_dask_raster(shape=(32, 32), chunks=16)
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.shape == (3, 32, 32)

    def test_returns_dataarray_with_band_dim(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert isinstance(result, xr.DataArray)
        assert result.dims[0] == 'band'
        assert result.dims[1] == 'y'
        assert result.dims[2] == 'x'

    def test_preserves_coords_and_attrs(self):
        raster = _make_dask_raster()
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.attrs == raster.attrs
        xr.testing.assert_equal(
            result.coords['x'], raster.coords['x']
        )

    def test_explicit_dtype(self):
        raster = _make_dask_raster()
        result = multi_overlap(
            raster, _triple_kernel, n_outputs=3, depth=1,
            dtype=np.float64,
        )
        assert result.dtype == np.float64

    def test_fewer_graph_tasks_than_sequential(self):
        raster = _make_dask_raster()
        multi = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

        from functools import partial
        def _band_i(chunk, i=0):
            return _triple_kernel(chunk)[i]

        bands = []
        for i in range(3):
            b = raster.data.map_overlap(
                partial(_band_i, i=i), depth=1, boundary=np.nan,
                meta=np.array(()),
            )
            bands.append(b)
        sequential = da.stack(bands, axis=0)

        assert len(dict(multi.data.__dask_graph__())) < len(
            dict(sequential.__dask_graph__())
        )

    def test_single_output_matches_map_overlap(self):
        def _single_kernel(chunk):
            return (chunk[1:-1, 1:-1] + 1)[np.newaxis, :]

        raster = _make_dask_raster()
        multi = multi_overlap(raster, _single_kernel, n_outputs=1, depth=1)

        def _ref_func(chunk):
            return chunk[1:-1, 1:-1] + 1
        ref = raster.data.map_overlap(
            _ref_func, depth=1, boundary=np.nan, meta=np.array(()),
        )
        np.testing.assert_array_equal(multi.values[0], ref.compute())

    def test_dtype_inference_defaults_to_input(self):
        raster = _make_dask_raster(dtype=np.float32)
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert result.dtype == np.float32

    def test_non_nan_boundary(self):
        """multi_overlap supports all boundary modes."""
        raster = _make_dask_raster()
        result = multi_overlap(
            raster, _triple_kernel, n_outputs=3, depth=1,
            boundary='nearest',
        )
        assert result.shape == (3, 64, 64)
        assert not np.any(np.isnan(result.values))


class TestMultiOverlapNumpy:
    def test_numpy_fallback(self):
        raster = xr.DataArray(
            np.random.RandomState(99).rand(32, 32).astype(np.float32),
            dims=['y', 'x'],
        )
        result = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3, 32, 32)


class TestMultiOverlapValidation:
    def test_rejects_non_2d(self):
        raster = xr.DataArray(da.zeros((4, 32, 32), chunks=16), dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2-D"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)

    def test_rejects_n_outputs_zero(self):
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="n_outputs.*>= 1"):
            multi_overlap(raster, _triple_kernel, n_outputs=0, depth=1)

    def test_rejects_depth_zero(self):
        raster = _make_dask_raster()
        with pytest.raises(ValueError, match="depth.*>= 1"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=0)

    def test_rejects_chunks_smaller_than_depth(self):
        raster = _make_dask_raster(shape=(32, 32), chunks=4)
        with pytest.raises(ValueError, match="[Cc]hunk size"):
            multi_overlap(raster, _triple_kernel, n_outputs=3, depth=5)

    def test_rejects_non_dataarray(self):
        with pytest.raises(TypeError):
            multi_overlap(np.zeros((10, 10)), _triple_kernel, 3, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_multi_overlap.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `multi_overlap`**

Append to `xrspatial/utils.py` after `fused_overlap`:

```python
def multi_overlap(agg, func, n_outputs, depth, boundary='nan', dtype=None):
    """Run a multi-output kernel via a single overlap + map_blocks call.

    ``func`` receives a padded 2-D chunk and returns
    ``(n_outputs, H, W)`` — the unpadded interior for each output band.
    The result is a 3-D DataArray with a leading ``band`` dimension.

    Parameters
    ----------
    agg : xr.DataArray
        2-D input raster.
    func : callable
        ``(H+2*dy, W+2*dx) -> (n_outputs, H, W)``
    n_outputs : int
        Number of output bands (>= 1).
    depth : int or tuple of int
        Per-axis overlap (>= 1 on each axis).
    boundary : str
        Boundary mode: 'nan', 'nearest', 'reflect', or 'wrap'.
    dtype : numpy dtype, optional
        Output dtype.  Defaults to input dtype.

    Returns
    -------
    xr.DataArray
        Shape ``(n_outputs, H, W)`` with ``band`` leading dimension.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"multi_overlap(): expected xr.DataArray, "
            f"got {type(agg).__name__}"
        )
    if agg.ndim != 2:
        raise ValueError(
            f"multi_overlap(): input must be 2-D, got {agg.ndim}-D"
        )
    if n_outputs < 1:
        raise ValueError(
            f"multi_overlap(): n_outputs must be >= 1, got {n_outputs}"
        )

    _validate_boundary(boundary)

    depth_dict = _normalize_depth(depth, agg.ndim)
    for ax, d in depth_dict.items():
        if d < 1:
            raise ValueError(
                f"multi_overlap(): depth must be >= 1, got {d} on axis {ax}"
            )

    dtype = dtype or agg.dtype

    # --- non-dask path ---
    if not has_dask_array() or not isinstance(agg.data, da.Array):
        if boundary == 'nan':
            depth_tuple = tuple(depth_dict[ax] for ax in range(agg.ndim))
            padded = _pad_nan(agg.data, depth_tuple)
        else:
            depth_tuple = tuple(depth_dict[ax] for ax in range(agg.ndim))
            padded = _pad_array(agg.data, depth_tuple, boundary)
        result_data = func(padded).astype(dtype)
        return xr.DataArray(
            result_data,
            dims=['band'] + list(agg.dims),
            coords=agg.coords,
            attrs=agg.attrs,
        )

    # --- dask path ---
    import dask.array.overlap as _dask_overlap

    _validate_boundary(boundary)
    boundary_val = _boundary_to_dask(boundary, is_cupy=is_cupy_backed(agg))

    # Validate chunk sizes
    for ax, d in depth_dict.items():
        for cs in agg.chunks[ax]:
            if cs < d:
                raise ValueError(
                    f"Chunk size {cs} on axis {ax} is smaller than "
                    f"depth {d}. Rechunk first."
                )

    # Step 1: pad with overlap
    padded = _dask_overlap.overlap(
        agg.data, depth=depth_dict, boundary=boundary_val
    )

    # Step 2: map_blocks — func returns (n_outputs, H, W) per block
    out = da.map_blocks(
        func,
        padded,
        dtype=dtype,
        new_axis=0,
        chunks=((n_outputs,),) + agg.data.chunks,
    )

    return xr.DataArray(
        out,
        dims=['band'] + list(agg.dims),
        coords=agg.coords,
        attrs=agg.attrs,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_multi_overlap.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/utils.py xrspatial/tests/test_multi_overlap.py
git commit -m "Add multi_overlap utility"
```

---

### Task 4: Accessor and exports

**Files:**
- Modify: `xrspatial/accessor.py` (line 501, after `rechunk_no_shuffle`)
- Modify: `xrspatial/__init__.py` (after `rechunk_no_shuffle` import)
- Modify: `xrspatial/tests/test_accessor.py` (line 93, expected methods list)

- [ ] **Step 1: Write failing accessor tests**

Append to `xrspatial/tests/test_fused_overlap.py`:

```python
class TestFusedOverlapAccessor:
    def test_accessor_delegates(self):
        import xrspatial  # noqa: F401
        from xrspatial.utils import fused_overlap
        raster = _make_dask_raster()
        direct = fused_overlap(raster, (_increment_interior, 1))
        via_acc = raster.xrs.fused_overlap((_increment_interior, 1))
        np.testing.assert_array_equal(direct.values, via_acc.values)
```

Append to `xrspatial/tests/test_multi_overlap.py`:

```python
class TestMultiOverlapAccessor:
    def test_accessor_delegates(self):
        import xrspatial  # noqa: F401
        raster = _make_dask_raster()
        direct = multi_overlap(raster, _triple_kernel, n_outputs=3, depth=1)
        via_acc = raster.xrs.multi_overlap(_triple_kernel, n_outputs=3, depth=1)
        np.testing.assert_array_equal(direct.values, via_acc.values)
```

- [ ] **Step 2: Run accessor tests to verify they fail**

Run: `pytest xrspatial/tests/test_fused_overlap.py::TestFusedOverlapAccessor -v`
Expected: FAIL (method not found)

- [ ] **Step 3: Add accessor methods**

In `xrspatial/accessor.py`, after the `rechunk_no_shuffle` method (line 501), add:

```python
    def fused_overlap(self, *stages, **kwargs):
        from .utils import fused_overlap
        return fused_overlap(self._obj, *stages, **kwargs)

    def multi_overlap(self, func, n_outputs, **kwargs):
        from .utils import multi_overlap
        return multi_overlap(self._obj, func, n_outputs, **kwargs)
```

- [ ] **Step 4: Add exports to `__init__.py`**

After the existing `rechunk_no_shuffle` import line, add:

```python
from xrspatial.utils import fused_overlap  # noqa
from xrspatial.utils import multi_overlap  # noqa
```

- [ ] **Step 5: Update expected methods list in test_accessor.py**

In `xrspatial/tests/test_accessor.py`, find the `test_dataarray_accessor_has_expected_methods` function's expected list (around line 93). Add `'fused_overlap'` and `'multi_overlap'` to the list.

- [ ] **Step 6: Run all tests**

Run: `pytest xrspatial/tests/test_fused_overlap.py xrspatial/tests/test_multi_overlap.py xrspatial/tests/test_accessor.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add xrspatial/accessor.py xrspatial/__init__.py xrspatial/tests/test_accessor.py xrspatial/tests/test_fused_overlap.py xrspatial/tests/test_multi_overlap.py
git commit -m "Add fused_overlap and multi_overlap to accessor and exports"
```

---

### Task 5: Documentation and README

**Files:**
- Modify: `docs/source/reference/utilities.rst`
- Modify: `README.md` (line 289, Utilities table)

- [ ] **Step 1: Add API entries to utilities.rst**

In `docs/source/reference/utilities.rst`, before the `Rechunking` section, add:

```rst
Overlap Fusion
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.utils.fused_overlap
    xrspatial.utils.multi_overlap
```

- [ ] **Step 2: Add README rows**

In `README.md`, in the Utilities table (after the `rechunk_no_shuffle` row around line 289), add:

```markdown
| [fused_overlap](xrspatial/utils.py) | Fuse sequential map_overlap calls into a single pass | Custom | 🔄 | ✅️ | 🔄 | ✅️ |
| [multi_overlap](xrspatial/utils.py) | Run multi-output kernel in a single overlap pass | Custom | 🔄 | ✅️ | 🔄 | ✅️ |
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/reference/utilities.rst README.md
git commit -m "Add fused_overlap and multi_overlap to docs and README"
```

---

### Task 6: User guide notebook

**Files:**
- Create: `examples/user_guide/37_Fused_Overlap.ipynb`

- [ ] **Step 1: Create the notebook**

Create `examples/user_guide/37_Fused_Overlap.ipynb` with these cells:

**Cell 1 (markdown):**
```
# Fusing Overlap Operations

When you chain spatial operations like erode then dilate, each one adds a
blockwise layer to the dask graph. `fused_overlap` runs them in a single
`map_overlap` call, and `multi_overlap` does the same for kernels that
produce multiple output bands.
```

**Cell 2 (code):**
```python
import numpy as np
import dask.array as da
import xarray as xr
import xrspatial
from xrspatial.utils import fused_overlap, multi_overlap
```

**Cell 3 (markdown):**
```
## fused_overlap: chained operations in one pass

Define two stage functions. Each takes a padded chunk and returns the
unpadded interior.
```

**Cell 4 (code):**
```python
def smooth_interior(chunk):
    """3x3 mean filter. Takes (H+2, W+2), returns (H, W)."""
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(chunk, (3, 3))
    return np.nanmean(windows, axis=(-2, -1))

def threshold_interior(chunk):
    """Binary threshold. Takes (H+2, W+2), returns (H, W)."""
    interior = chunk[1:-1, 1:-1]
    return (interior > 0.5).astype(np.float32)

np.random.seed(42)
raw = np.random.rand(512, 512).astype(np.float32)
dem = xr.DataArray(da.from_array(raw, chunks=128), dims=['y', 'x'])
```

**Cell 5 (code):**
```python
# Fused: one map_overlap call
fused = fused_overlap(dem, (smooth_interior, 1), (threshold_interior, 1))

# Sequential: two map_overlap calls
step1 = dem.data.map_overlap(smooth_interior, depth=1, boundary=np.nan, meta=np.array(()))
sequential = step1.map_overlap(threshold_interior, depth=1, boundary=np.nan, meta=np.array(()))

print(f'Fused graph:      {len(dict(fused.data.__dask_graph__())):,} tasks')
print(f'Sequential graph: {len(dict(sequential.__dask_graph__())):,} tasks')
```

**Cell 6 (markdown):**
```
## multi_overlap: N outputs in one pass
```

**Cell 7 (code):**
```python
def gradient_kernel(chunk):
    """Compute dx and dy gradients. Takes (H+2, W+2), returns (2, H, W)."""
    dx = (chunk[1:-1, 2:] - chunk[1:-1, :-2]) / 2.0
    dy = (chunk[2:, 1:-1] - chunk[:-2, 1:-1]) / 2.0
    return np.stack([dx, dy], axis=0)

result = multi_overlap(dem, gradient_kernel, n_outputs=2, depth=1)
print(f'Output shape: {result.shape}')
print(f'Dimensions:   {result.dims}')
print(f'Graph tasks:  {len(dict(result.data.__dask_graph__())):,}')
```

**Cell 8 (code):**
```python
# Accessor syntax works too
fused_acc = dem.xrs.fused_overlap((smooth_interior, 1), (threshold_interior, 1))
multi_acc = dem.xrs.multi_overlap(gradient_kernel, n_outputs=2, depth=1)
print('Accessor: OK')
```

- [ ] **Step 2: Commit**

```bash
git add examples/user_guide/37_Fused_Overlap.ipynb
git commit -m "Add fused_overlap and multi_overlap user guide notebook"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run all new tests**

```bash
pytest xrspatial/tests/test_fused_overlap.py xrspatial/tests/test_multi_overlap.py xrspatial/tests/test_accessor.py -v
```

Expected: all PASS

- [ ] **Step 2: Run existing test suite to check for regressions**

```bash
pytest xrspatial/tests/test_rechunk_no_shuffle.py -v
```

Expected: all PASS (no regressions in utils.py)
