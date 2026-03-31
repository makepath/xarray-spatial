# Performance Sweep: Parallel Triage and Fix Workflow

Audit xrspatial modules for performance bottlenecks, OOM risk under 30TB dask
workloads, and backend-specific anti-patterns. Dispatches parallel subagents
for fast triage, then generates a ralph-loop to benchmark and fix HIGH-severity
issues.

Optional arguments: $ARGUMENTS
(e.g. `--top 5`, `--exclude slope,aspect`, `--only-io`, `--reset-state`)

---

## Step 0 -- Determine mode and parse arguments

Parse $ARGUMENTS for these flags (multiple may combine):

| Flag | Effect |
|------|--------|
| `--top N` | Limit Phase 1 to the top N scored modules (default: all) |
| `--exclude mod1,mod2` | Remove named modules from scope |
| `--only-terrain` | Restrict to: slope, aspect, curvature, terrain, terrain_metrics, hillshade, sky_view_factor |
| `--only-focal` | Restrict to: focal, convolution, morphology, bilateral, edge_detection, glcm |
| `--only-hydro` | Restrict to: flood, cost_distance, geodesic, surface_distance, viewshed, erosion, diffusion |
| `--only-io` | Restrict to: geotiff, reproject, rasterize, polygonize |
| `--reset-state` | Delete `.claude/performance-sweep-state.json` and treat all modules as never-inspected |
| `--skip-phase1` | Skip triage; reuse last state file; go straight to ralph-loop generation for unresolved HIGH items |
| `--report-only` | Run Phase 1 triage but do not generate a ralph-loop command |
| `--size small` | Phase 2 benchmarks use 128x128 arrays |
| `--size large` | Phase 2 benchmarks use 2048x2048 arrays |
| `--high-only` | Only report HIGH severity findings in the triage output |

If `--skip-phase1` is set, jump to Step 6 (ralph-loop generation).
Otherwise proceed to Step 1.

## Step 1 -- Discover modules in scope

Enumerate all candidate modules. For each, record its file path(s):

**Single-file modules:** Every `.py` file directly under `xrspatial/`, excluding
`__init__.py`, `_version.py`, `__main__.py`, `utils.py`, `accessor.py`,
`preview.py`, `dataset_support.py`, `diagnostics.py`, `analytics.py`.

**Subpackage modules:** The `geotiff/` and `reproject/` directories under
`xrspatial/`. Treat each subpackage as a single audit unit. List all `.py`
files within each (excluding `__init__.py`).

Apply `--only-*` and `--exclude` filters from Step 0 to narrow the list.

Store the filtered module list in memory (do NOT write intermediate files).

## Step 2 -- Gather metadata and score each module

For every module in scope, collect:

| Field | How |
|-------|-----|
| **last_modified** | `git log -1 --format=%aI -- <path>` (for subpackages, use the most recent file) |
| **total_commits** | `git log --oneline -- <path> \| wc -l` |
| **loc** | `wc -l < <path>` (for subpackages, sum all files) |
| **has_dask_backend** | grep the file(s) for `_run_dask`, `map_overlap`, `map_blocks` |
| **has_cuda_backend** | grep the file(s) for `@cuda.jit`, `import cupy` |
| **is_io_module** | module is geotiff or reproject |
| **has_existing_bench** | a file matching the module name exists in `benchmarks/benchmarks/` |

### Load inspection state

Read `.claude/performance-sweep-state.json`. If it does not exist, treat every
module as never-inspected. If `--reset-state` was set, delete the file first.

State file schema:

```json
{
  "last_triage": "ISO-DATE",
  "modules": {
    "slope": {
      "last_inspected": "ISO-DATE",
      "oom_verdict": "SAFE",
      "bottleneck": "compute-bound",
      "high_count": 0,
      "issue": null
    }
  }
}
```

### Compute scores

```
days_since_inspected = (today - last_perf_inspected).days   # 9999 if never
days_since_modified  = (today - last_modified).days

score = (days_since_inspected * 3)
      + (loc * 0.1)
      + (total_commits * 0.5)
      + (has_dask_backend * 200)
      + (has_cuda_backend * 150)
      + (is_io_module * 300)
      - (days_since_modified * 0.2)
      - (has_existing_bench * 100)
```

Sort modules by score descending. If `--top N` is set, keep only the top N.

## Step 3 -- Dispatch parallel subagents for static triage

For each module in the scored list, dispatch a subagent using the Agent tool.
Launch ALL subagents in a single message (parallel dispatch). Each subagent
receives the prompt below, with `MODULE_NAME` and `MODULE_FILES` substituted.

**Subagent prompt template:**

~~~
You are auditing the xrspatial module "MODULE_NAME" for performance issues.

Read these files: MODULE_FILES

Perform ALL of the following analyses and return your findings as a single
JSON object. Do NOT modify any files. This is read-only analysis.

### 1. Dask Path Analysis

Trace every dask code path (_run_dask, _run_dask_cupy, or any function that
receives dask-backed DataArrays). Flag these patterns with severity:

- HIGH: `.values` on a dask-backed DataArray or CuPy array (premature materialization)
- HIGH: `.compute()` inside a loop (materializes full graph each iteration)
- HIGH: `np.array()` or `np.asarray()` wrapping a dask or CuPy array
- MEDIUM: `da.stack()` without a following `.rechunk()`
- MEDIUM: `map_overlap` with depth >= chunk_size / 4
- MEDIUM: Missing `boundary` argument in `map_overlap`
- MEDIUM: Same function called twice on same input without caching
- MEDIUM: Python `for` loop iterating over dask chunks (serializes the graph)

If the module has NO dask code path, note "no dask backend" and skip.

### 2. 30TB / 16GB OOM Verdict

For each dask code path found in section 1:

**Part A — Static trace:** Follow the code end-to-end. Answer: does peak
memory scale with total array size, or with chunk size? If any operation
forces full materialization, the verdict is WILL OOM.

**Part B — Task graph simulation:** Write and run a Python script (in /tmp/
with a unique name including "MODULE_NAME") that:

```python
import dask.array as da
import xarray as xr
import json, sys

arr = da.zeros((2560, 2560), chunks=(256, 256), dtype='float64')
raster = xr.DataArray(arr, dims=['y', 'x'])

# Add coords if the function needs them (geodesic, slope with CRS, etc.)
# raster = raster.assign_coords(x=np.linspace(-180, 180, 2560),
#                                y=np.linspace(-90, 90, 2560))

try:
    result = MODULE_FUNCTION(raster, **DEFAULT_ARGS)
    graph = result.__dask_graph__()
    task_count = len(graph)
    tasks_per_chunk = task_count / 100.0

    # Check for fan-out: any task key that depends on more than 4 other tasks
    deps = dict(graph)
    max_fan_in = 0
    for key, val in deps.items():
        if hasattr(val, '__dask_graph__'):
            sub = val.__dask_graph__()
            max_fan_in = max(max_fan_in, len(sub))

    print(json.dumps({
        "success": True,
        "task_count": task_count,
        "tasks_per_chunk": round(tasks_per_chunk, 2),
        "max_fan_in": max_fan_in,
        "extrapolation_30tb": "~{} tasks at 57M chunks".format(
            int(tasks_per_chunk * 57_000_000))
    }))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
```

Adapt the function call and imports for the specific module. Run the script
and capture its JSON output. If it errors, record the error and rely on
Part A alone.

**Verdict:** One of:
- `SAFE` — memory bounded by chunk size, graph scales linearly
- `RISKY` — bounded but tight (e.g. large overlap depth, 3D intermediates)
- `WILL OOM` — forces full materialization or unbounded memory growth

### 3. GPU Transfer Analysis

Scan for CuPy/CUDA code paths. Flag:

- HIGH: `.data.get()` followed by CuPy operations (GPU-CPU-GPU round-trip)
- HIGH: `cupy.asarray()` inside a loop (repeated CPU-GPU transfers)
- MEDIUM: Mixing NumPy and CuPy ops in same function without clear reason
- MEDIUM: Register pressure — count float64 local variables in `@cuda.jit`
  kernels; flag if >20
- MEDIUM: Thread blocks >16x16 on kernels with >20 float64 locals

If the module has NO GPU code path, note "no GPU backend" and skip.

### 4. Memory Allocation Patterns

- MEDIUM: Unnecessary `.copy()` on arrays never mutated downstream
- MEDIUM: Large temporary arrays that could be fused into the kernel
- LOW: `np.zeros_like()` + fill loop where `np.empty()` would suffice

### 5. Numba Anti-Patterns

- MEDIUM: Missing `@ngjit` on nested for-loops over `.data` arrays
- MEDIUM: `@jit` without `nopython=True` (object-mode fallback risk)
- LOW: Type instability — initializing with int then assigning float
- LOW: Column-major iteration on row-major arrays (inner loop should be last axis)

### 6. Bottleneck Classification

Based on your analysis, classify the module as ONE of:
- `IO-bound` — dominated by disk reads/writes or serialization
- `memory-bound` — peak allocation is the limiting factor
- `compute-bound` — CPU/GPU time dominates, memory is fine
- `graph-bound` — dask task graph overhead dominates

### Output Format

Return EXACTLY this JSON structure (no extra text before or after):

```json
{
  "module": "MODULE_NAME",
  "files_read": ["list of files you read"],
  "findings": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "category": "dask_materialization|dask_chunking|gpu_transfer|register_pressure|memory_allocation|numba_antipattern",
      "file": "filename.py",
      "line": 123,
      "description": "what the issue is",
      "fix": "how to fix it",
      "backends_affected": ["dask+numpy", "dask+cupy", "cupy", "numpy"]
    }
  ],
  "oom_verdict": {
    "dask_numpy": "SAFE|RISKY|WILL OOM",
    "dask_cupy": "SAFE|RISKY|WILL OOM",
    "reasoning": "one-sentence explanation",
    "estimated_peak_per_chunk_mb": 0.5,
    "task_count": 3721,
    "tasks_per_chunk": 37.21,
    "graph_simulation_ran": true
  },
  "bottleneck": "compute-bound|memory-bound|IO-bound|graph-bound",
  "bottleneck_reasoning": "one-sentence explanation"
}
```

IMPORTANT: Only flag patterns that are ACTUALLY present in the code. Do not
report hypothetical issues. False positives are worse than missed issues.
If a pattern like `.values` is used on a known-numpy-only code path, do not
flag it.
~~~

Wait for all subagents to return before proceeding to Step 4.

## Step 4 -- Merge results and print the triage report

Parse the JSON returned by each subagent. If a subagent returned malformed
output, record the module as "audit failed" with a note.

### 4a. Print the Module Risk Ranking Table

Sort modules by score descending. Print:

```
## Performance Sweep — Static Triage Report

### Module Risk Ranking
| Rank | Module          | Score  | OOM Verdict     | Bottleneck    | HIGH | MED | LOW |
|------|-----------------|--------|-----------------|---------------|------|-----|-----|
| 1    | geotiff         | 31200  | WILL OOM (d+np) | IO-bound      | 3    | 1   | 0   |
| 2    | viewshed        | 30050  | RISKY (d+np)    | memory-bound  | 2    | 2   | 1   |
| ...  | ...             | ...    | ...             | ...           | ...  | ... | ... |
```

If `--high-only` is set, only count HIGH findings and omit modules with zero HIGH.

### 4b. Print the 30TB / 16GB Verdict Summary

Group modules by OOM verdict:

```
### 30TB on Disk / 16GB RAM — Out-of-Memory Analysis

#### WILL OOM (fix required)
- **module_name**: reasoning from subagent

#### RISKY (bounded but tight)
- **module_name**: reasoning from subagent

#### SAFE (memory bounded by chunk size)
- module_name, module_name, module_name, ...
```

### 4c. Print Detailed Findings

For each module that has findings, print a severity-grouped table:

```
### module_name (bottleneck: compute-bound, OOM: SAFE)

| # | Severity | File:Line      | Category                | Description                  | Fix                           |
|---|----------|----------------|-------------------------|------------------------------|-------------------------------|
| 1 | HIGH     | slope.py:142   | dask_materialization    | .values on dask input        | Use .data or stay lazy        |
| 2 | MEDIUM   | slope.py:88    | dask_chunking           | map_overlap depth too large  | Reduce depth or warn users    |
```

### 4d. Print Actionable Rockout Commands

For each HIGH-severity finding, print a ready-to-paste `/rockout` command:

```
### Ready-to-Run Fixes (HIGH severity only)

1. **geotiff** — eager .values materialization (WILL OOM)
   /rockout "Fix eager .values materialization in geotiff reader.
   The dask read path at reader.py:87 calls .values which forces
   the full array into memory. For 30TB inputs this will OOM on
   a 16GB machine. Must stay lazy through the entire read path."

2. **cost_distance** — iterative solver unbounded memory (WILL OOM)
   /rockout "Fix cost_distance iterative solver to work within
   bounded memory. Currently materializes the full distance matrix
   each iteration. Must use chunked iteration for 30TB dask inputs."
```

Construct each `/rockout` command from the finding's description and fix fields.
Include the OOM verdict and bottleneck classification in the prompt text so
rockout has full context.

## Step 5 -- Update state file

Write `.claude/performance-sweep-state.json` with the triage results:

```json
{
  "last_triage": "<current ISO datetime>",
  "modules": {
    "<module_name>": {
      "last_inspected": "<current ISO datetime>",
      "oom_verdict": "<SAFE|RISKY|WILL OOM>",
      "bottleneck": "<IO-bound|memory-bound|compute-bound|graph-bound>",
      "high_count": "<number of HIGH findings>",
      "issue": null
    }
  }
}
```

If the file already exists, merge — update entries for modules that were
just audited, keep entries for modules not in this run's scope.

If `--report-only` is set, stop here. Do not proceed to Step 6.

## Step 6 -- Generate the ralph-loop command

Collect all modules from Step 4 (or from the state file if `--skip-phase1`)
that have at least one HIGH-severity finding and no `issue` recorded in the
state file (i.e. not yet fixed).

Sort them by: WILL OOM first, then RISKY, then by HIGH count descending.

Determine the benchmark array size from arguments:
- `--size small` → 128x128
- `--size large` → 2048x2048
- default → 512x512

### 6a. Print the ranked target list

```
### Phase 2 Targets (HIGH severity, unfixed)
| # | Module        | HIGH Count | OOM Verdict | Bottleneck   |
|---|---------------|------------|-------------|--------------|
| 1 | geotiff       | 3          | WILL OOM    | IO-bound     |
| 2 | cost_distance | 1          | WILL OOM    | memory-bound |
| 3 | viewshed      | 2          | RISKY       | memory-bound |
```

If no modules qualify, print:
"No HIGH-severity findings to fix. Run `/sweep-performance` without
`--skip-phase1` to refresh the triage."
Then stop.

### 6b. Print the ralph-loop command

Using the target list, generate and print:

````
/ralph-loop "Performance sweep Phase 2: benchmark and fix HIGH-severity findings.

**Target modules in priority order:**
1. <module> (<N> HIGH findings, <OOM verdict>) -- <one-line summary of worst finding>
2. <module> ...
...

**For each module, in order:**

1. Write a benchmark script at /tmp/perf_sweep_bench_<module>.py that:
   - Imports the module's public functions
   - Creates a test array (<SIZE>x<SIZE>, float64)
   - For EACH available backend (numpy, dask+numpy; cupy and dask+cupy only if available):
     a. Wrap the array in the appropriate DataArray type
     b. Measure wall time: timeit.repeat(number=1, repeat=3), take median
     c. Measure Python memory: tracemalloc.start() / tracemalloc.get_traced_memory()[1] for peak
     d. Measure process memory: resource.getrusage(RUSAGE_SELF).ru_maxrss before and after
     e. For CuPy backends: cupy.get_default_memory_pool().used_bytes() before and after
   - Print results as JSON to stdout

2. Run the benchmark script and capture results.

3. Confirm the HIGH finding from Phase 1:
   - If the dask backend uses significantly more memory than expected for
     the chunk size, or wall time shows a materialization stall: CONFIRMED.
   - If the benchmark shows no anomaly: downgrade to MEDIUM in state file,
     print 'False positive — skipping' and move to the next module.

4. If confirmed: run /rockout to fix the issue end-to-end (issue, worktree,
   implementation, tests, docs). Include the benchmark numbers in the
   issue body for context.

5. After rockout completes: rerun the same benchmark script. Print a
   before/after comparison:
   | Backend    | Metric       | Before | After  | Ratio | Verdict    |
   |------------|-------------|--------|--------|-------|------------|
   | numpy      | wall_ms     | 45.2   | 12.1   | 0.27x | IMPROVED   |
   | dask+numpy | peak_rss_mb | 892    | 34     | 0.04x | IMPROVED   |
   Thresholds: IMPROVED < 0.8x, REGRESSION > 1.2x, else UNCHANGED.

6. Update .claude/performance-sweep-state.json with the issue number.

7. Output <promise>ITERATION DONE</promise>

If all targets have been addressed or confirmed as false positives:
<promise>ALL PERFORMANCE ISSUES FIXED</promise>." --max-iterations <N+2> --completion-promise "ALL PERFORMANCE ISSUES FIXED"
````

Set `--max-iterations` to the number of target modules plus 2 (buffer for
retries).

### 6c. Print reminder text

```
Phase 1 triage complete. To proceed with fixes:
  Copy the ralph-loop command above and paste it.

Other options:
  Fix one manually:    copy any /rockout command from the report above
  Rerun triage only:   /sweep-performance --report-only
  Skip Phase 1:        /sweep-performance --skip-phase1 (reuses last triage)
  Reset all tracking:  /sweep-performance --reset-state
```

---

## General Rules

- Phase 1 subagents do NOT modify any source, test, or benchmark files.
  Read-only analysis only.
- Phase 2 ralph-loop modifies code only through `/rockout`.
- Temporary benchmark scripts and graph simulation scripts go in `/tmp/`
  with unique names including the module name (e.g. `/tmp/perf_sweep_bench_slope.py`,
  `/tmp/perf_sweep_graph_slope.py`). Clean them up after capturing results.
- Only flag patterns that are ACTUALLY present in the code. Do not report
  hypothetical issues or patterns that "could" occur.
- Include the exact file path and line number for every finding so the user
  can navigate directly to the issue.
- False positives are worse than missed issues. If you are not confident a
  pattern is actually harmful in context (e.g. `.values` used intentionally
  on a known-numpy array), do not flag it.
- The 30TB simulation constructs the dask task graph only; it NEVER calls
  `.compute()`.
- State file (`.claude/performance-sweep-state.json`) is gitignored by
  convention — do not add it to git.
- If $ARGUMENTS is empty, use defaults: audit all modules, benchmark at
  512x512, generate ralph-loop for HIGH items.
- For subpackage modules (geotiff, reproject), the subagent should read ALL
  `.py` files in the subpackage directory, not just `__init__.py`.
- When generating `/rockout` commands, include the OOM verdict, bottleneck
  classification, and affected backends in the prompt text so rockout has
  full performance context.
