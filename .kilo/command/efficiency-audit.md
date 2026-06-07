# Efficiency Audit: Compute Waste and Anti-Pattern Detection

Analyze source code for performance anti-patterns specific to the NumPy / CuPy /
Dask / Numba stack. The prompt is: {{ARGUMENTS}}

---

## Step 0 -- Determine mode

Check {{ARGUMENTS}} for a mode keyword:

- **`compare`**: Skip straight to Step 7 (post-fix comparison). Requires a saved
  baseline file from a previous run.
- **`no-bench`**: Run the static audit only (Steps 1-6), skip benchmarking entirely.
- **Otherwise** (default): Run the full audit with baseline benchmarks.

## Step 1 -- Scope the audit

1. If {{ARGUMENTS}} names specific files or functions, audit only those.
2. If {{ARGUMENTS}} names a category (e.g. `hydrology`, `surface`), identify all
   source files in that category from the README feature matrix.
3. If {{ARGUMENTS}} is empty or says "all", audit every `.py` file under `xrspatial/`
   (excluding `tests/`, `datasets/`, and `__pycache__/`).
4. Read each file in scope.

## Step 2 -- Static analysis: Dask anti-patterns

Search for these patterns in each file. For every hit, record the file, line
number, the offending code, and the severity (HIGH / MEDIUM / LOW).

### 2a. Premature materialization (HIGH)
- **`.values` on a Dask-backed DataArray or CuPy array:** forces a full compute
  or GPU-to-CPU transfer. Search for `.values` usage outside of tests.
- **`.compute()` inside a loop or repeated call:** materializes the full graph
  each iteration instead of building a lazy pipeline.
- **`np.array()` or `np.asarray()` wrapping a Dask or CuPy array:** silent
  materialization.

### 2b. Chunking issues (MEDIUM)
- **`da.stack()` without a following `.rechunk()`:** creates size-1 chunks on the
  new axis, causing extreme task-graph overhead.
- **`map_overlap` with depth >= chunk_size / 2:** overlap regions dominate the
  chunk, wasting memory and compute. Flag if depth is not obviously small relative
  to expected chunk sizes.
- **Missing `boundary` argument in `map_overlap`:** defaults may not match the
  function's intended boundary handling.

### 2c. Redundant computation (MEDIUM)
- **Calling the same function twice on the same input** without caching the result
  (e.g. computing slope inside aspect when aspect already computes slope internally).
- **Building large intermediate arrays** that could be fused into the kernel
  (e.g. allocating a full-size output array, then filling it cell by cell in Numba
  instead of writing directly).

## Step 3 -- Static analysis: GPU anti-patterns

### 3a. Register pressure (HIGH)
- **CUDA kernels with many float64 local variables:** count the number of named
  float64 locals in each `@cuda.jit` kernel. Flag kernels with more than 20
  float64 locals (likely to spill to slow local memory).
- **Thread blocks larger than 16x16 on register-heavy kernels:** check the
  `cuda_args()` call or any custom dims function. If the kernel has high register
  count and uses 32x32 blocks, flag it.

### 3b. Unnecessary transfers (HIGH)
- **`.data.get()` followed by CuPy operations:** data round-trips GPU -> CPU -> GPU.
- **`cupy.asarray(numpy_array)` inside a hot path:** repeated CPU -> GPU transfers
  that could be hoisted outside the loop.
- **Mixing NumPy and CuPy operations** in the same function without an obvious
  reason (e.g. `np.where` on a CuPy array silently converts to NumPy).

### 3c. Kernel launch overhead (LOW)
- **Per-cell kernel launches:** launching a CUDA kernel inside a Python loop over
  cells instead of processing the full grid in one kernel launch.
- **Small array kernel launches:** calling a CUDA kernel on arrays smaller than
  the thread block (overhead dominates).

## Step 4 -- Static analysis: Numba anti-patterns

### 4a. JIT compilation issues (MEDIUM)
- **Missing `@ngjit` or `@jit(nopython=True)`:** pure-Python loops over arrays
  without JIT compilation. Search for nested `for` loops operating on `.data`
  arrays without a Numba decorator.
- **Object-mode fallback:** `@jit` without `nopython=True` may silently fall back
  to object mode. Only `@ngjit` or `@jit(nopython=True)` guarantees compilation.
- **Type instability:** mixing int and float in Numba functions (e.g. initializing
  with `0` then assigning a float) can cause unnecessary casts.

### 4b. Memory layout (LOW)
- **Column-major iteration on row-major arrays:** Numba loops that iterate
  `for col ... for row` on C-contiguous arrays (cache-unfriendly access pattern).
  The inner loop should iterate over the last axis (columns for row-major).

## Step 5 -- Static analysis: General Python anti-patterns

### 5a. Unnecessary copies (MEDIUM)
- **`.copy()` on arrays that are never mutated:** wasted allocation.
- **`np.zeros_like()` + fill loop:** when `np.empty()` + fill or direct
  computation would avoid zero-initialization overhead.

### 5b. Inefficient I/O patterns (LOW)
- **Reading the same file multiple times** in a function.
- **Writing intermediate results to disk** when they could stay in memory.

## Step 6 -- Baseline benchmarks

**Skip this step if mode is `no-bench` or `compare`.**

For each public function in the audited scope, capture rough baseline timings.
This does not use ASV; it runs quick inline timings so the user gets a
before-snapshot without heavyweight setup.

### 6a. Build a benchmark script

Create a temporary script at `/tmp/efficiency_audit_bench_<scope_hash>.py` (use a
short hash of the audited file list to keep the name unique). The script should:

1. Import the public functions found in the audited files.
2. Generate a test array using the same helper pattern as
   `benchmarks/benchmarks/common.py`:
   ```python
   import numpy as np, xarray as xr
   ny, nx = 512, 512  # moderate size -- fast but meaningful
   x = np.linspace(-180, 180, nx)
   y = np.linspace(-90, 90, ny)
   x2, y2 = np.meshgrid(x, y)
   z = 100.0 * np.exp(-x2**2 / 5e5 - y2**2 / 2e5)
   z += np.random.default_rng(71942).normal(0, 2, (ny, nx))
   raster = xr.DataArray(z, dims=['y', 'x'])
   ```
   Adjust as needed (e.g. add coords for geodesic functions, integer data for
   zonal, etc.).
3. For each function, time it with `timeit.repeat(number=1, repeat=3)` and take
   the **median** of the repeats. One iteration is enough -- we want a rough
   ballpark, not precise statistics.
4. Print results as JSON to stdout:
   ```json
   {
     "scope": ["slope.py", "aspect.py"],
     "array_shape": [512, 512],
     "backend": "numpy",
     "timings": {
       "slope": {"median_ms": 12.3, "runs": [12.1, 12.3, 13.0]},
       "aspect": {"median_ms": 8.7, "runs": [8.5, 8.7, 9.1]}
     }
   }
   ```

### 6b. Run the benchmark script

Execute the script and capture stdout. If a function errors (e.g. missing
optional dependency), record `"error": "<message>"` instead of timings and
continue with the rest.

### 6c. Save the baseline

Write the JSON output to `.efficiency-audit-baseline.json` in the project root.
This file is gitignored-by-convention (do not add it to git). Tell the user the
baseline has been saved and what it contains.

If a baseline file already exists, back it up to
`.efficiency-audit-baseline.prev.json` before overwriting.

## Step 7 -- Generate the report

```
## Efficiency Audit Report

### Scope
- Files audited: N
- Functions audited: N

### Findings

#### HIGH severity
| # | File:Line          | Pattern                    | Description                           | Fix                              |
|---|--------------------|---------------------------|---------------------------------------|----------------------------------|
| 1 | slope.py:142       | Premature materialization  | `.values` on dask input in _run_dask  | Use `.data.compute()` instead    |
| 2 | geodesic.py:87     | Register pressure          | 24 float64 locals in _gpu kernel      | Split kernel or use 16x16 blocks |
| ...| ...               | ...                        | ...                                   | ...                              |

#### MEDIUM severity
| # | File:Line          | Pattern                    | Description                           | Fix                              |
|---|--------------------|---------------------------|---------------------------------------|----------------------------------|
| ...| ...               | ...                        | ...                                   | ...                              |

#### LOW severity
| # | File:Line          | Pattern                    | Description                           | Fix                              |
|---|--------------------|---------------------------|---------------------------------------|----------------------------------|
| ...| ...               | ...                        | ...                                   | ...                              |

### Baseline Timings (512x512, numpy)
| Function   | Median (ms) | Runs (ms)           |
|------------|-------------|---------------------|
| slope      | 12.3        | 12.1, 12.3, 13.0   |
| aspect     | 8.7         | 8.5, 8.7, 9.1      |
| ...        | ...         | ...                 |

(If any function errored, show "ERROR: <reason>" in the Median column.)

### Summary
- HIGH: N findings
- MEDIUM: N findings
- LOW: N findings
- Clean files (no issues): <list>

### Recommendations
<Prioritized list of the top 3-5 changes that would have the most impact,
with estimated effort (one-liner / small PR / larger refactor)>
```

## Step 8 -- Post-fix comparison (mode=`compare`)

**Only run this step when {{ARGUMENTS}} contains `compare`.**

1. Read `.efficiency-audit-baseline.json` from the project root. If it does not
   exist, tell the user to run the audit without `compare` first to capture a
   baseline, and stop.
2. Regenerate the benchmark script from Step 6a using the `scope` and
   `array_shape` recorded in the baseline file (so the comparison is apples to
   apples).
3. Run the benchmark script (Step 6b) and capture the new timings.
4. For each function, compute the ratio: `new_median / old_median`.

Generate a comparison report:

```
## Efficiency Audit: Post-Fix Comparison

### Baseline
- Captured: <baseline file mtime or "unknown">
- Array shape: <from baseline>
- Backend: <from baseline>

### Results

| Function   | Before (ms) | After (ms) | Ratio | Verdict      |
|------------|-------------|------------|-------|--------------|
| slope      | 12.3        | 7.1        | 0.58x | IMPROVED     |
| aspect     | 8.7         | 8.5        | 0.98x | UNCHANGED    |
| ...        | ...         | ...        | ...   | ...          |

Thresholds: IMPROVED < 0.8x, REGRESSION > 1.2x, else UNCHANGED.

### Net impact
- Functions improved: N
- Functions regressed: N
- Functions unchanged: N
- Overall: <one-line summary, e.g. "2 of 3 functions faster, no regressions">
```

5. Save the new timings to `.efficiency-audit-after.json` for reference.

---

## General rules

- Do not modify source, test, or benchmark files. Temporary scripts go in `/tmp/`.
- Only flag patterns that are actually present in the code. Do not report
  hypothetical issues or patterns that "could" occur.
- Include the exact file path and line number for every finding so the user
  can navigate directly to the issue.
- False positives are worse than missed issues. If you are not confident a
  pattern is actually harmful in context (e.g. `.values` used intentionally
  on a known-numpy array), do not flag it.
- If {{ARGUMENTS}} includes "fix", still do not auto-fix. Report and ask.
- If {{ARGUMENTS}} includes a severity filter (e.g. "high only"), only report
  findings at that severity level.
- If {{ARGUMENTS}} includes "diff" or "changed", restrict the audit to files
  changed on the current branch vs origin/main.
- Baseline benchmark scripts are disposable. Clean up `/tmp/` scripts after
  capturing results.
- The 512x512 array size is a default. If {{ARGUMENTS}} includes a size like
  `1024x1024` or `small`, adjust accordingly. "small" = 128x128, "large" = 2048x2048.
