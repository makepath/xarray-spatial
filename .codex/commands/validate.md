# Validate: Numerical Accuracy and Backend Parity Check

Take a function name (or detect the changed function from the current branch diff)
and verify its numerical accuracy against reference implementations and across all
four backends. The prompt is: $ARGUMENTS

---

## Step 1 -- Identify the target

1. If $ARGUMENTS names a specific function (e.g. `slope`, `flow_accumulation`),
   use that.
2. If $ARGUMENTS is empty or says "auto", run `git diff origin/main --name-only`
   to find changed source files under `xrspatial/`. Identify which public functions
   were added or modified. If multiple functions changed, validate each one.
3. Read the function's source to understand:
   - Which backends are implemented (check the `ArrayTypeFunctionMapping` call)
   - What parameters it accepts (boundary modes, method variants, etc.)
   - What the expected output range and dtype should be
   - Whether it's a neighborhood operation (uses `map_overlap`) or a per-cell operation

## Step 2 -- Select or build reference data

Build **three** test datasets, each serving a different purpose:

### 2a. Analytical known-answer dataset
Create a small synthetic raster where the correct answer can be computed by hand
or from a closed-form formula. Examples:

- **Slope/aspect:** a perfect plane tilted at a known angle (e.g. `z = 2x + 3y`
  gives slope = arctan(sqrt(13)) for planar method)
- **Flow direction:** a simple cone or V-shaped valley where flow paths are obvious
- **Focal:** a raster with a single non-zero cell surrounded by zeros
- **Multispectral indices:** bands with known ratios so NDVI/NDWI etc. are trivially
  verifiable

Compute the expected result array by hand (or with basic numpy math) and store it
as a numpy array. This is the **ground truth** for this dataset.

### 2b. QGIS / rasterio / scipy reference dataset
Check whether the function's existing test file already has a reference fixture
(like `qgis_slope` in `test_slope.py`). If so, reuse it.

If no reference exists, attempt to compute one:
1. Check if `rasterio` is installed (`python -c "import rasterio"`). If available,
   write the test raster to a temporary GeoTIFF (unique name including the function
   name, e.g. `tmp_validate_slope.tif`) and run the equivalent rasterio/GDAL operation.
2. If rasterio is not available, check for `scipy.ndimage` equivalents (e.g.
   `generic_filter`, `uniform_filter`, `sobel`).
3. If neither is available, skip this dataset and note it in the report.

### 2c. Realistic stress dataset
Generate a larger raster (at least 256x256) with terrain-like features using the
project's `perlin` module or `np.random.default_rng(42)`. Include:
- NaN patches (5-10% of cells) to test NaN propagation
- A mix of flat and steep areas
- Edge values near dtype limits for the tested dtypes

This dataset is for backend parity and performance, not absolute accuracy.

## Step 3 -- Run across all backends

For each dataset and each parameter combination (e.g. boundary modes, method
variants), run the function on every implemented backend:

1. **NumPy** -- always available, treat as the baseline
2. **Dask+NumPy** -- use `create_test_raster(data, backend='dask+numpy')` with
   at least two different chunk sizes:
   - Chunks that evenly divide the array
   - Ragged chunks (array size not divisible by chunk size)
3. **CuPy** -- skip with a note if CUDA is not available
4. **Dask+CuPy** -- skip with a note if CUDA is not available

Use the helpers from `general_checks.py`:
- `create_test_raster()` to build DataArrays for each backend
- For CuPy results, extract with `.data.get()`
- For Dask results, extract with `.data.compute()`

## Step 4 -- Compare results

Run four categories of comparison, reporting pass/fail and numeric details for each:

### 4a. Ground truth comparison (dataset 2a)
Compare the NumPy backend result against the hand-computed expected array.
```python
np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-10, equal_nan=True)
```
If this fails, the algorithm itself has a bug. Report the max absolute error,
max relative error, and the cell location(s) where divergence is worst.

### 4b. Reference implementation comparison (dataset 2b)
Compare the NumPy result against the rasterio/scipy/QGIS reference.
Use `rtol=1e-5` (matching the project's existing QGIS tolerance convention).
Exclude edge cells if the implementations handle boundaries differently (document
which edges were excluded and why).

### 4c. Backend parity (all datasets)
Compare every non-NumPy backend against the NumPy result:

| Comparison            | Default tolerance          |
|-----------------------|---------------------------|
| NumPy vs Dask+NumPy   | `rtol=1e-5`               |
| NumPy vs CuPy         | `atol=1e-6, rtol=1e-6`    |
| NumPy vs Dask+CuPy    | `atol=1e-6, rtol=1e-6`    |

For each comparison, report:
- Max absolute difference
- Max relative difference
- Whether NaN locations match exactly (`np.isnan` masks must be identical)
- Whether output shape, dims, coords, and attrs are preserved (use
  `general_output_checks`)

### 4d. Edge case and invariant checks
Run these regardless of which function is being validated:

- **NaN propagation:** cells neighboring NaN input should behave correctly for the
  function (NaN output for most neighborhood ops with `boundary='nan'`)
- **Constant surface:** if the input is uniform (e.g. all 42.0), the output should
  be zero for derivative operations (slope, curvature) or uniform for pass-through
  operations
- **Single-cell raster:** 1x1 input should not crash (may return NaN)
- **Dtype preservation:** run with float32 and float64 inputs; verify the output
  dtype matches expectations
- **Boundary modes:** if the function accepts a `boundary` parameter, test all
  valid modes (`nan`, `nearest`, `reflect`, `wrap`) and verify:
  - Shape is preserved
  - Non-nan modes produce no NaN output when source has no NaN
  - NumPy and Dask results agree for each mode

## Step 5 -- Generate the report

Print a structured report with these sections:

```
## Validation Report: <function_name>

### Target
- Function: <name>
- Source: <file_path>
- Backends implemented: <list>
- Parameter variants tested: <list>

### Datasets
| Dataset          | Shape   | Dtype   | NaN% | Notes                    |
|------------------|---------|---------|------|--------------------------|
| Analytical       | ...     | ...     | ...  | <description>            |
| Reference (src)  | ...     | ...     | ...  | <reference tool used>    |
| Stress           | ...     | ...     | ...  | <generation method>      |

### Results

#### Ground Truth (analytical dataset)
- Status: PASS / FAIL
- Max absolute error: ...
- Max relative error: ...
- Worst cell: (row, col) expected=... got=...

#### Reference Implementation
- Reference: <rasterio / scipy / QGIS fixture / skipped>
- Status: PASS / FAIL / SKIPPED
- Max absolute error: ...
- Notes: <edge exclusions, known differences>

#### Backend Parity
| Comparison              | Dataset     | Max |Δ|    | Max |Δ/ref| | NaN match | Status |
|-------------------------|-------------|-----------|-------------|-----------|--------|
| NumPy vs Dask+NumPy     | analytical  | ...       | ...         | yes/no    | ...    |
| NumPy vs Dask+NumPy     | stress      | ...       | ...         | yes/no    | ...    |
| NumPy vs CuPy           | analytical  | ...       | ...         | yes/no    | ...    |
| ...                     | ...         | ...       | ...         | ...       | ...    |

#### Edge Cases
| Check              | Status | Notes                               |
|--------------------|--------|-------------------------------------|
| NaN propagation    | ...    |                                     |
| Constant surface   | ...    |                                     |
| Single-cell        | ...    |                                     |
| Dtype float32      | ...    |                                     |
| Dtype float64      | ...    |                                     |
| Boundary modes     | ...    | <modes tested>                      |

### Verdict
- Overall: PASS / FAIL
- <1-3 sentence summary of findings>
- <action items if anything failed>
```

## Step 6 -- Suggest fixes (if failures found)

If any check failed:
1. Identify the root cause (algorithm bug, boundary handling, dtype casting,
   chunking artifact, GPU precision, etc.)
2. Describe the fix concisely.
3. Ask the user whether they want you to apply the fix now.

Do NOT apply fixes automatically. The purpose of `/validate` is to report, not to
change code.

---

## General rules

- Run all comparisons in a Python script or inline pytest, not by eyeballing
  print output. Use `np.testing.assert_allclose` for numeric checks.
- Any temporary files (GeoTIFFs, intermediate arrays) must use unique names
  including the function name (e.g. `tmp_validate_slope_256x256.tif`). Clean them
  up at the end.
- If CUDA is not available, skip GPU backends gracefully and note it in the report.
  Never fail the validation just because a backend is unavailable.
- If $ARGUMENTS specifies a tolerance override (e.g. "validate slope rtol=1e-3"),
  use the provided tolerances instead of the defaults.
- If $ARGUMENTS specifies "quick", skip the stress dataset and boundary mode sweep
  to give a faster result.
- Do not modify any source or test files. This command is read-only analysis.
- If the function has a `method` parameter (e.g. `slope(method='geodesic')`),
  validate each method variant separately.
