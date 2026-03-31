# Sweep-Performance: Parallel Performance Triage and Fix Workflow

**Date:** 2026-03-31
**Status:** Draft

## Overview

A `/sweep-performance` slash command that audits every xrspatial module for
performance bottlenecks, OOM risk under large-scale dask workloads, and
backend-specific anti-patterns. Uses parallel subagents for fast static triage,
then a sequential ralph-loop to benchmark and fix confirmed HIGH-severity
issues.

The central question for every dask backend: "If the data on disk was 30TB
and the machine only had 16GB of RAM, would this tool cause an out-of-memory
error?"

## Scope

All `.py` modules under `xrspatial/` plus the `geotiff/` and `reproject/`
subpackages. Excludes `__init__.py`, `_version.py`, `__main__.py`, `utils.py`,
`accessor.py`, `preview.py`, `dataset_support.py`, `diagnostics.py`,
`analytics.py`.

## Architecture

Two phases in a single invocation:

```
/sweep-performance
    |
    +-- Phase 1: Parallel Static Triage
    |   |-- Score & rank modules (git metadata + complexity heuristics)
    |   |-- Dispatch one subagent per module
    |   |   |-- Static analysis (dask, GPU, memory, Numba patterns)
    |   |   |-- 30TB/16GB OOM simulation (task graph construction, no compute)
    |   |   +-- Return structured JSON findings
    |   |-- Merge results into ranked report
    |   +-- Update state file
    |
    +-- Phase 2: Ralph-Loop (HIGH severity only)
        |-- Generate /ralph-loop command targeting HIGH modules
        |-- Each iteration:
        |   |-- Real benchmarks (wall time, tracemalloc, RSS, CuPy pool)
        |   |-- Confirm finding is not false positive
        |   |-- /rockout to fix
        |   |-- Post-fix benchmark comparison
        |   +-- Update state file
        +-- User pastes command to start
```

---

## Phase 1: Module Scoring

For every module in scope, collect via git:

| Field              | Source                                                    |
|--------------------|-----------------------------------------------------------|
| `last_modified`    | `git log -1 --format=%aI -- <path>`                      |
| `total_commits`    | `git log --oneline -- <path> \| wc -l`                   |
| `loc`              | `wc -l < <path>`                                         |
| `has_dask_backend` | grep for `_run_dask`, `map_overlap`, `map_blocks`         |
| `has_cuda_backend` | grep for `@cuda.jit`, `import cupy`                       |
| `is_io_module`     | module is in geotiff/ or reproject/                       |
| `has_existing_bench` | matching file exists in `benchmarks/benchmarks/`        |

### Scoring Formula

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

Rationale:
- Never-inspected modules dominate (9999 * 3 = ~30,000).
- Dask and CUDA backends boosted: that is where OOM and perf bugs live.
- I/O modules get the highest boost: most relevant for 30TB question.
- Larger modules more likely to contain issues.
- Existing ASV benchmarks slightly deprioritize (perf already considered).

---

## Phase 1: Subagent Static Analysis

One subagent per module. Each performs the checks below and returns a
structured JSON blob.

### Dask Path Analysis

- `.values` on dask-backed DataArray (premature materialization) — **HIGH**
- `.compute()` inside a loop — **HIGH**
- `np.array()` / `np.asarray()` wrapping dask or CuPy array — **HIGH**
- `da.stack()` without `.rechunk()` — **MEDIUM**
- `map_overlap` with depth >= chunk_size / 4 — **MEDIUM**
- Missing `boundary` argument in `map_overlap` — **MEDIUM**
- Redundant computation (same function called twice on same input) — **MEDIUM**
- Python loops over dask chunks (serializes the graph) — **MEDIUM**

### 30TB / 16GB OOM Verdict

Two-part analysis for each dask code path:

**Part 1 — Static trace.** Follow the dask code path and answer: does peak
memory scale with total array size, or with chunk size? If any step forces
full materialization, verdict is WILL OOM.

**Part 2 — Task graph simulation.** Write and execute a script that:

```python
import dask.array as da
import xarray as xr

# Use a representative grid (2560x2560, 10x10 = 100 chunks) to inspect
# graph structure. The pattern is identical at any scale — what matters
# is whether the graph fans out, materializes, or stays chunk-local.
arr = da.zeros((2560, 2560), chunks=(256, 256), dtype='float64')
raster = xr.DataArray(arr, dims=['y', 'x'])

# Call the function lazily
result = module_function(raster, **default_args)

# Inspect the graph without executing
graph = result.__dask_graph__()
task_count = len(graph)
tasks_per_chunk = task_count / 100  # normalize to per-chunk

# Check for fan-out patterns or full-materialization nodes
# Extrapolate to 30TB: ~57 million chunks at 256x256 float64
# If tasks_per_chunk is constant => graph scales linearly => SAFE
# If any node depends on all chunks => full materialization => WILL OOM
```

The script constructs the graph only, never calls `.compute()`. Reports:
- Task count and tasks-per-chunk ratio
- Estimated peak memory per chunk (MB)
- Whether the graph contains fan-out or materialization nodes
- Extrapolation to 30TB: linear graph growth (SAFE) vs fan-out (WILL OOM)

**Verdict**: `SAFE`, `RISKY` (bounded but tight), or `WILL OOM` (unbounded
or materializes).

### GPU Transfer Analysis

- `.data.get()` followed by CuPy ops (GPU-CPU-GPU round-trip) — **HIGH**
- `cupy.asarray()` inside a hot loop — **HIGH**
- Mixing NumPy/CuPy ops without reason — **MEDIUM**
- Register pressure: >20 float64 locals in `@cuda.jit` kernel — **MEDIUM**
- Thread blocks >16x16 on register-heavy kernels — **MEDIUM**

### Memory Allocation Patterns

- Unnecessary `.copy()` on arrays never mutated — **MEDIUM**
- `np.zeros_like()` + fill loop (could be `np.empty()`) — **LOW**
- Large temporary arrays that could be fused into the kernel — **MEDIUM**

### Numba Anti-Patterns

- Missing `@ngjit` on nested for-loops over `.data` arrays — **MEDIUM**
- `@jit` without `nopython=True` (object-mode fallback risk) — **MEDIUM**
- Type instability (int/float mixing in Numba functions) — **LOW**
- Column-major iteration on row-major arrays (cache-unfriendly) — **LOW**

### Bottleneck Classification

Based on static analysis, classify the module as one of:
- **IO-bound** — dominated by disk reads/writes or serialization
- **Memory-bound** — peak allocation is the limiting factor
- **Compute-bound** — CPU/GPU time dominates, memory is fine
- **Graph-bound** — dask task graph overhead dominates (too many small tasks)

### Subagent Output Schema

```json
{
  "module": "slope",
  "files_read": ["xrspatial/slope.py"],
  "findings": [
    {
      "severity": "HIGH",
      "category": "dask_materialization",
      "file": "slope.py",
      "line": 142,
      "description": ".values on dask input in _run_dask",
      "fix": "Use .data.compute() or restructure to stay lazy",
      "backends_affected": ["dask+numpy", "dask+cupy"]
    }
  ],
  "oom_verdict": {
    "dask_numpy": "SAFE",
    "dask_cupy": "SAFE",
    "reasoning": "map_overlap with depth=1, memory bounded by chunk size",
    "estimated_peak_per_chunk_mb": 0.5,
    "task_count": 3721,
    "graph_simulation_ran": true
  },
  "bottleneck": "compute-bound",
  "bottleneck_reasoning": "3x3 kernel with Numba JIT, no I/O, small overlap"
}
```

---

## Phase 1: Merged Report

After all subagents return, print a consolidated report.

### Module Risk Ranking Table

```
| Rank | Module        | Score | OOM Verdict     | Bottleneck   | HIGH | MED | LOW |
|------|---------------|-------|-----------------|--------------|------|-----|-----|
| 1    | geotiff       | 31200 | WILL OOM (d+np) | IO-bound     | 3    | 1   | 0   |
| 2    | viewshed      | 30050 | RISKY (d+np)    | memory-bound | 2    | 2   | 1   |
| ...  | ...           | ...   | ...             | ...          | ...  | ... | ... |
```

### 30TB / 16GB Verdict Summary

Grouped by verdict:

- **WILL OOM (fix required):** list modules with reasoning
- **RISKY (bounded but tight):** list modules with reasoning
- **SAFE (memory bounded by chunk size):** list modules

### Detailed Findings

Per-module table of all findings grouped by severity (file:line, pattern,
description, fix).

### Actionable Rockout Commands

For each HIGH-severity finding, a ready-to-paste `/rockout` command.

### State File Update

Write `.claude/performance-sweep-state.json`:

```json
{
  "last_triage": "2026-03-31T14:00:00Z",
  "modules": {
    "slope": {
      "last_inspected": "2026-03-31T14:00:00Z",
      "oom_verdict": "SAFE",
      "bottleneck": "compute-bound",
      "high_count": 0,
      "issue": null
    }
  }
}
```

---

## Phase 2: Ralph-Loop for HIGH Severity Fixes

Collect all modules with at least one HIGH-severity finding. Generate a
`/ralph-loop` command targeting them in priority order.

### Each Iteration

1. **Benchmark** the module on a moderate array (512x512 default) across all
   available backends. Measure four metrics per backend per function:
   - Wall time: `timeit.repeat(number=1, repeat=3)`, median
   - Python memory: `tracemalloc.get_traced_memory()` peak
   - Process memory: `resource.getrusage(RUSAGE_SELF).ru_maxrss` delta
   - GPU memory (if CuPy): `cupy.get_default_memory_pool().used_bytes()` delta

2. **Confirm the static finding** from Phase 1 is real. If the benchmark
   shows the issue does not manifest (false positive), downgrade to MEDIUM
   in the report and skip to next module.

3. **Classify the bottleneck** with measured data:
   - IO-bound: wall time dominated by read/write, low CPU
   - Memory-bound: peak RSS much larger than expected for chunk size
   - Compute-bound: CPU pegged, memory stable
   - Graph-bound: dask task count extremely high, scheduler overhead visible

4. **Run `/rockout`** to fix the confirmed issue (GitHub issue, worktree,
   implementation, tests, docs).

5. **Post-fix benchmark** — rerun the same benchmark. Report before/after
   delta.

6. **Update state** — record the fix in
   `.claude/performance-sweep-state.json` with issue number.

7. Output `<promise>ITERATION DONE</promise>`.

### Generated Command Shape

```
/ralph-loop "Performance sweep Phase 2: benchmark and fix HIGH-severity findings.

**Target modules in priority order:**
1. geotiff (3 HIGH findings, WILL OOM) -- eager .values materialization
2. cost_distance (1 HIGH finding, WILL OOM) -- iterative solver unbounded memory

**For each module:**
1. Write and run a benchmark script measuring wall time, peak memory
   (tracemalloc + RSS + CuPy pool) across all available backends
2. Confirm the HIGH finding from Phase 1 triage is real
3. If confirmed: run /rockout to fix it end-to-end
4. After rockout: rerun benchmark, report before/after delta
5. Update .claude/performance-sweep-state.json
6. Output <promise>ITERATION DONE</promise>

If all targets addressed: <promise>ALL PERFORMANCE ISSUES FIXED</promise>."
--max-iterations {N+2} --completion-promise "ALL PERFORMANCE ISSUES FIXED"
```

### Reminder Text

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

## Arguments

| Argument           | Effect                                                     |
|--------------------|------------------------------------------------------------|
| `--top N`          | Limit Phase 1 subagents to top N scored modules (default: all) |
| `--exclude m1,m2`  | Remove named modules from scope                           |
| `--only-terrain`   | slope, aspect, curvature, terrain, terrain_metrics, hillshade, sky_view_factor |
| `--only-focal`     | focal, convolution, morphology, bilateral, edge_detection, glcm |
| `--only-hydro`     | flood, cost_distance, geodesic, surface_distance, viewshed, erosion, diffusion |
| `--only-io`        | geotiff, reproject, rasterize, polygonize                  |
| `--reset-state`    | Delete state file and start fresh                          |
| `--skip-phase1`    | Reuse last triage state, go straight to ralph-loop generation |
| `--report-only`    | Run Phase 1 only, no ralph-loop command                    |
| `--size small`     | Benchmark at 128x128                                       |
| `--size large`     | Benchmark at 2048x2048                                     |
| `--high-only`      | Only report HIGH severity findings                         |

Default (no arguments): audit all modules, benchmark at 512x512, generate
ralph-loop for HIGH items.

---

## General Rules

- Phase 1 subagents do NOT modify source files. Read-only analysis.
- Phase 2 ralph-loop modifies code only through `/rockout`.
- Temporary benchmark scripts go in `/tmp/` with unique names.
- Only flag patterns actually present in the code; no hypothetical issues.
- Include exact file path and line number for every finding.
- False positives are worse than missed issues.
- The 30TB simulation constructs the dask graph only; it never calls `.compute()`.
- State file (`.claude/performance-sweep-state.json`) is gitignored by convention.
