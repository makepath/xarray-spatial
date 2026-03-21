# Backend Parity: Cross-Backend Consistency Audit

Verify that all implemented backends produce consistent results for a given
function or set of functions. The prompt is: $ARGUMENTS

---

## Step 1 -- Identify targets

1. If $ARGUMENTS names specific functions (e.g. `slope`, `aspect`), use those.
2. If $ARGUMENTS names a category (e.g. `hydrology`, `surface`, `focal`), read
   `README.md` to find all functions in that category.
3. If $ARGUMENTS is empty or says "all", scan the full feature matrix in `README.md`
   and test every function that claims support for 2+ backends.
4. For each function, read its source file and find the `ArrayTypeFunctionMapping`
   call to determine which backends are actually implemented (not just what the
   README claims).

## Step 2 -- Build test inputs

For each target function, create test rasters at three scales:

| Name    | Size    | Purpose                                         |
|---------|---------|--------------------------------------------------|
| tiny    | 8x6     | Fast, easy to inspect cell-by-cell               |
| medium  | 64x64   | Catches chunk-boundary artifacts in dask          |
| large   | 256x256 | Stress test, exposes numerical accumulation drift |

For each size, generate two variants:
- **Clean:** no NaN, realistic value range for the function
  (e.g. 0-5000m for elevation, 0-1 for NDVI inputs)
- **Dirty:** 5-10% random NaN, some extreme values near dtype limits

Use `np.random.default_rng(42)` for reproducibility. For functions that require
specific input structure (e.g. `flow_direction` needs a DEM with drainage, not
random noise), use the project's `perlin` module or a synthetic cone/valley.

Also test with at least two dtypes: `float32` and `float64`.

## Step 3 -- Run every backend

For each function, input variant, and dtype:

1. **NumPy:** `create_test_raster(data, backend='numpy')` -- always the baseline.
2. **Dask+NumPy:** test with two chunk configurations:
   - `chunks=(size//2, size//2)` -- even split
   - `chunks=(size//3, size//3)` -- ragged remainder
3. **CuPy:** `create_test_raster(data, backend='cupy')` -- skip if CUDA unavailable.
4. **Dask+CuPy:** `create_test_raster(data, backend='dask+cupy')` -- skip if CUDA
   unavailable.

If the function has parameter variants (e.g. `boundary`, `method`), test the
default parameters first. If $ARGUMENTS includes "thorough", also sweep all
parameter combinations.

## Step 4 -- Pairwise comparison

For every non-NumPy result, compare against the NumPy baseline. Extract data using
the project conventions:
- Dask: `.data.compute()`
- CuPy: `.data.get()`
- Dask+CuPy: `.data.compute().get()`

For each pair, compute and record:

### 4a. Value agreement
```python
abs_diff = np.abs(result - baseline)
max_abs = np.nanmax(abs_diff)
rel_diff = abs_diff / (np.abs(baseline) + 1e-30)  # avoid div-by-zero
max_rel = np.nanmax(rel_diff)
mean_abs = np.nanmean(abs_diff)
```

### 4b. NaN mask agreement
```python
nan_match = np.array_equal(np.isnan(result), np.isnan(baseline))
nan_only_in_result = np.sum(np.isnan(result) & ~np.isnan(baseline))
nan_only_in_baseline = np.sum(np.isnan(baseline) & ~np.isnan(result))
```

### 4c. Metadata preservation
Using `general_output_checks` from `general_checks.py`:
- Output type matches input type (DataArray backed by the same array type)
- Shape, dims, coords, attrs preserved

### 4d. Pass/fail thresholds

| Comparison            | rtol     | atol     |
|-----------------------|----------|----------|
| NumPy vs Dask+NumPy   | 1e-5     | 0        |
| NumPy vs CuPy         | 1e-6     | 1e-6     |
| NumPy vs Dask+CuPy    | 1e-6     | 1e-6     |

A comparison **fails** if `max_abs > atol` AND `max_rel > rtol`, or if NaN masks
disagree.

## Step 5 -- Chunk boundary analysis

Dask backends are the most likely source of parity issues due to `map_overlap`
boundary handling. For any Dask comparison that fails or is borderline:

1. Identify which cells diverge from the NumPy result.
2. Map those cells to chunk boundaries (cells within `depth` pixels of a chunk edge).
3. Report what percentage of divergent cells are at chunk boundaries vs interior.
4. If all divergence is at boundaries, the issue is likely in the `map_overlap`
   `depth` or `boundary` parameter. Say so explicitly.

## Step 6 -- Generate the report

```
## Backend Parity Report

### Functions tested
| Function            | Backends implemented       | Source file              |
|---------------------|---------------------------|--------------------------|
| slope               | numpy, cupy, dask, dask+cupy | xrspatial/slope.py    |
| ...                 | ...                        | ...                      |

### Parity Matrix

#### <function_name>
| Comparison            | Input       | Dtype   | Max |Δ|   | Max |Δ/ref| | NaN match | Metadata | Status |
|-----------------------|-------------|---------|----------|------------|-----------|----------|--------|
| NumPy vs Dask+NumPy   | tiny clean  | float32 | ...      | ...        | yes       | ok       | PASS   |
| NumPy vs Dask+NumPy   | medium dirty| float64 | ...      | ...        | yes       | ok       | PASS   |
| NumPy vs CuPy         | tiny clean  | float32 | ...      | ...        | no (3)    | ok       | FAIL   |
| ...                   | ...         | ...     | ...      | ...        | ...       | ...      | ...    |

### Failures
For each FAIL row:
- Which cells diverged
- Whether divergence correlates with chunk boundaries (Dask) or specific
  input values (CuPy)
- Likely root cause
- Suggested fix

### Summary
- Functions tested: N
- Total comparisons: N
- Passed: N
- Failed: N
- Skipped (no CUDA): N
```

---

## General rules

- Do not modify any source or test files. This command is read-only.
- Use `create_test_raster` from `general_checks.py` for all raster construction.
- Any temporary files must include the function name for uniqueness.
- If CUDA is unavailable, skip CuPy and Dask+CuPy gracefully. Report them
  as SKIPPED, not FAIL.
- If $ARGUMENTS includes "fix", still do not auto-fix. Report the issue and ask.
- If a function is not in `ArrayTypeFunctionMapping` (e.g. it only has a numpy
  path), note it as "single-backend only" and skip parity checks for it.
- If $ARGUMENTS includes a specific tolerance (e.g. `rtol=1e-3`), override the
  defaults in the threshold table.
