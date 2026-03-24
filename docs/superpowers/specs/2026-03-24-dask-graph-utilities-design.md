# Dask Graph Utilities: fused_overlap and multi_overlap

**Date:** 2026-03-24
**Status:** Draft
**Issue:** TBD (to be created during implementation)

## Problem

Several xrspatial operations run multiple `map_overlap` passes over the same data when a single pass would produce the same result with a smaller dask graph:

- **Morphological opening/closing** chains erode + dilate as two separate `map_overlap` calls (2 blockwise layers).
- **Flow direction MFD** runs the same 3x3 kernel 8 times to extract 8 output bands, then stacks them (8 blockwise layers + 1 stack).
- **GLCM texture** does the same per-metric extraction pattern.

Each extra `map_overlap` call adds a blockwise layer to the dask graph. For large rasters with many chunks, this inflates task counts and scheduler overhead.

**Not in scope:** Iterative operations like diffusion (N steps of depth-1) are a poor fit for fusion because `total_depth = N`, and for large N the overlap region dominates chunk data. Those are better handled by the existing iterative approach.

## Solution

Two new utilities in `xrspatial/utils.py`, both exposed on the DataArray `.xrs` accessor.

---

### 1. `fused_overlap`

Fuses a sequence of overlap operations into a single `map_overlap` call with combined depth.

**Stage function contract:** Each stage function takes a padded array and returns the **unpadded interior result only**. That is, given input of shape `(H + 2*dy, W + 2*dx)`, the function returns shape `(H, W)`. This is different from `map_overlap`'s built-in convention (same-shape return). Existing chunk functions that follow the same-shape convention need a one-line adapter: `lambda chunk: func(chunk)[dy:-dy, dx:-dx]` (where `dy`, `dx` are the per-axis depths).

**Boundary restriction:** Only `boundary='nan'` is supported. For non-NaN boundary modes (`nearest`, `reflect`, `wrap`), the fused result would differ from sequential execution at chunk/array edges because boundary fill happens once on the original data rather than after each stage. Restricting to NaN avoids this correctness gap. NaN boundaries cover the vast majority of spatial raster operations in this codebase.

**Signature:**

```python
def fused_overlap(agg, *stages, boundary='nan'):
    """Run multiple overlap operations in a single map_overlap call.

    Parameters
    ----------
    agg : xr.DataArray
        Input raster.  If not dask-backed, stages are applied
        sequentially with numpy/cupy padding.
    *stages : tuple of (func, depth)
        Each stage is a ``(callable, depth)`` pair.  ``func`` takes a
        padded numpy/cupy array of shape ``(H + 2*d, W + 2*d)`` and
        returns the interior result of shape ``(H, W)``.  ``depth``
        is an int or tuple of ints (per-axis overlap).
    boundary : str
        Must be ``'nan'``.

    Returns
    -------
    xr.DataArray
        Result of applying all stages in sequence.

    Raises
    ------
    ValueError
        If no stages are provided, boundary is not 'nan', or any
        chunk dimension is smaller than total_depth.
    """
```

**Usage:**

```python
from xrspatial.utils import fused_overlap

# morphological opening in one pass instead of two
result = fused_overlap(
    data,
    (erode_interior, (1, 1)),
    (dilate_interior, (1, 1)),
    boundary='nan',
)

# via accessor
result = data.xrs.fused_overlap(
    (erode_interior, (1, 1)),
    (dilate_interior, (1, 1)),
    boundary='nan',
)
```

**How it works (dask path):**

1. Normalize each stage's depth to a per-axis dict via `_normalize_depth`.
2. Compute `total_depth` by summing depths across all stages.
3. Validate that every chunk dimension exceeds `total_depth`.
4. Build a wrapper function that operates on the chunk padded with `total_depth`:
   - Stage 0 receives the full `(H + 2*T, W + 2*T)` block, returns `(H + 2*(T - d0), W + 2*(T - d0))` interior.
   - Stage 1 receives that `(H + 2*(T - d0), W + 2*(T - d0))` block (which has exactly `T - d0` cells of valid overlap remaining). It returns `(H + 2*(T - d0 - d1), W + 2*(T - d0 - d1))`.
   - This continues until the final stage, which has exactly `d_last` overlap and returns `(H, W)`.
   - The wrapper then re-pads the `(H, W)` result back to `(H + 2*T, W + 2*T)` using NaN fill so that `map_overlap` can crop its expected `total_depth`.
5. Call `data.map_overlap(wrapper, depth=total_depth, boundary=np.nan)` once.

**Worked example (two stages, each depth 1, total_depth 2):**

```
map_overlap gives wrapper a chunk of shape (H+4, W+4).

Stage 0: receives (H+4, W+4), returns interior (H+2, W+2).
  - The (H+2, W+2) block has 1 cell of valid overlap on each side.

Stage 1: receives (H+2, W+2), returns interior (H, W).

Wrapper: pads (H, W) back to (H+4, W+4) with NaN fill.
map_overlap crops total_depth=2 from each side -> final (H, W). Correct.
```

**Non-dask fallback:** For each stage in sequence: pad with `np.pad(..., mode='constant', constant_values=np.nan)` (or `cupy.pad` for cupy arrays), apply `func`, take interior. Note: the existing `_pad_array` helper does not support `boundary='nan'` directly, so the implementation must use `np.pad` with constant NaN fill for this path.

**Result:** N stages produce 1 blockwise layer instead of N.

---

### 2. `multi_overlap`

Runs a multi-output kernel via a single overlap + map_blocks call.

**Signature:**

```python
def multi_overlap(agg, func, n_outputs, depth, boundary='nan', dtype=None):
    """Run a multi-output kernel via a single map_overlap call.

    Parameters
    ----------
    agg : xr.DataArray
        2-D input raster.
    func : callable
        Takes a padded numpy/cupy chunk of shape
        ``(H + 2*dy, W + 2*dx)`` and returns an array of shape
        ``(n_outputs, H, W)`` -- the interior result per output band.
    n_outputs : int
        Number of output bands (must be >= 1).
    depth : int or tuple of int
        Per-axis overlap.  Must be >= 1 on each spatial axis.
    boundary : str
        Boundary mode: 'nan', 'nearest', 'reflect', or 'wrap'.
    dtype : numpy dtype, optional
        Output dtype.  If None, uses the input dtype.

    Returns
    -------
    xr.DataArray
        3-D DataArray of shape ``(n_outputs, H, W)`` with a leading
        ``band`` dimension.

    Raises
    ------
    ValueError
        If n_outputs < 1, depth < 1, input is not 2-D, or any chunk
        dimension is smaller than depth.
    """
```

**Usage:**

```python
from xrspatial.utils import multi_overlap

# flow direction MFD: one pass instead of 8
bands = multi_overlap(data, mfd_kernel, n_outputs=8, depth=(1, 1))

# via accessor
bands = data.xrs.multi_overlap(mfd_kernel, n_outputs=8, depth=(1, 1))
```

**How it works (dask path):**

```python
import dask.array.overlap as _overlap

def multi_overlap(agg, func, n_outputs, depth, boundary='nan', dtype=None):
    depth_dict = _normalize_depth(depth, agg.ndim)
    boundary_val = _boundary_to_dask(boundary)
    dtype = dtype or agg.dtype

    # Validate depth >= 1 on each axis
    for ax, d in depth_dict.items():
        if d < 1:
            raise ValueError(f"depth must be >= 1, got {d} on axis {ax}")

    # Validate chunk sizes exceed depth
    for ax, d in depth_dict.items():
        for cs in agg.chunks[ax]:
            if cs < d:
                raise ValueError(
                    f"Chunk size {cs} on axis {ax} is smaller than "
                    f"depth {d}. Rechunk first."
                )

    # Step 1: pad the dask array with overlap
    padded = _overlap.overlap(agg.data, depth=depth_dict, boundary=boundary_val)

    # Step 2: map_blocks with new output axis
    def _wrapper(block):
        # func returns (n_outputs, H, W) from padded (H+2dy, W+2dx)
        return func(block)

    out = da.map_blocks(
        _wrapper,
        padded,
        dtype=dtype,
        new_axis=0,
        chunks=((n_outputs,),) + agg.data.chunks,
    )

    # Step 3: wrap in DataArray with band dimension
    result = xr.DataArray(
        out,
        dims=['band'] + list(agg.dims),
        coords=agg.coords,
        attrs=agg.attrs,
    )
    return result
```

This produces 1 overlap layer + 1 map_blocks layer = 2 layers total, versus N+1 for the current N-separate-calls + stack approach.

**Non-dask fallback:** Pad the numpy/cupy array (using `_pad_array` for non-NaN boundaries, or `np.pad`/`cupy.pad` with `constant_values=np.nan` for NaN boundary), call `func` (returns `(n_outputs, H, W)`), wrap in DataArray.

---

## Helpers

### `_normalize_depth(depth, ndim)`

Accepts `int`, `tuple`, or `dict` and returns a dict `{axis: int}` for all axes. Follows dask's conventions:

- `int` -> same depth on all axes
- `tuple` -> one depth per axis
- `dict` -> passed through, validated that all axes `0..ndim-1` are present, all values are non-negative ints, and no extra axes exist

---

## Accessor integration

Both functions go on `XrsSpatialDataArrayAccessor` only. Not on the Dataset accessor -- these are chunk-level operations that don't generalize to "apply to every variable."

```python
# DataArray accessor
def fused_overlap(self, *stages, **kwargs):
    from .utils import fused_overlap
    return fused_overlap(self._obj, *stages, **kwargs)

def multi_overlap(self, func, n_outputs, **kwargs):
    from .utils import multi_overlap
    return multi_overlap(self._obj, func, n_outputs, **kwargs)
```

## Backend support

- **numpy:** Direct application with `_pad_array` for overlap simulation.
- **dask+numpy:** Primary target. One `map_overlap` or `overlap` + `map_blocks` call.
- **cupy:** Works if the user's `func` handles cupy arrays. `_pad_array` already supports cupy.
- **dask+cupy:** Same as dask+numpy, with `is_cupy=True` passed to `_boundary_to_dask`.

No `ArrayTypeFunctionMapping` needed. These are dask wrappers, not spatial operations.

## What this does NOT include

- **Non-NaN boundaries for `fused_overlap`.** Sequential boundary fill between stages gives different results than a single outer fill. NaN is the only mode where fusion is equivalent to sequential execution.
- **Diffusion / high-iteration fusion.** When `total_depth = N` for large N, the overlap dominates chunk data. The existing iterative approach is better for those cases. Practical limit: 2-5 stages.
- **Auto-rechunk between stages.** Separate concern -- `rechunk_no_shuffle` exists for that.
- **Dataset accessor methods.** These are per-array operations.
- **Refactoring existing call sites** (MFD, GLCM, morphology) to use the new utilities. That's follow-up work after the utilities ship.

## File changes

| File | Change |
|------|--------|
| `xrspatial/utils.py` | Add `fused_overlap`, `multi_overlap`, `_normalize_depth` helper |
| `xrspatial/accessor.py` | Add `fused_overlap`, `multi_overlap` to DataArray accessor |
| `xrspatial/__init__.py` | Export both functions |
| `xrspatial/tests/test_fused_overlap.py` | New test file |
| `xrspatial/tests/test_multi_overlap.py` | New test file |
| `xrspatial/tests/test_accessor.py` | Add to expected methods list |
| `docs/source/reference/utilities.rst` | Add API entries |
| `README.md` | Add rows to Utilities table |
| `examples/user_guide/37_Fused_Overlap.ipynb` | New notebook |

## Testing strategy

**fused_overlap:**
- Single stage produces same result as plain `map_overlap`
- Two stages (erode + dilate) matches sequential `map_overlap` calls
- Three+ stages work correctly
- Depth accumulation is correct (depth 1+1 = 2 total overlap)
- Non-square depth (e.g. `(2, 1)`) works
- Small chunks (barely larger than total_depth) produce correct results
- Rejects non-NaN boundary modes with clear error
- Rejects chunks smaller than total_depth with clear error
- Non-dask fallback (numpy) works
- Non-dask fallback (cupy) works
- Accessor delegates correctly
- Input validation (empty stages, non-DataArray, etc.)

**multi_overlap:**
- Single output matches plain `map_overlap`
- N outputs match N separate `map_overlap` calls + `da.stack`
- Output is an xr.DataArray with `band` leading dimension
- Output shape is `(n_outputs, H, W)`
- Values are identical to the sequential approach
- dtype inference works (None -> input dtype)
- Explicit dtype parameter is respected
- Rejects n_outputs < 1, depth < 1, non-2D input
- Rejects chunks smaller than depth
- Non-dask fallback (numpy) works
- Non-dask fallback (cupy) works
- Accessor delegates correctly
- Coords and attrs are preserved
