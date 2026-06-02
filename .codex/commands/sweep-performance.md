# Performance Sweep: Dispatch subagents to audit and fix performance issues

Audit xrspatial modules for performance bottlenecks, OOM risk under 30TB dask
workloads, and backend-specific anti-patterns. Subagents fix HIGH and
MEDIUM-severity findings via /rockout in the same agent that did the audit,
in parallel.

Optional arguments: $ARGUMENTS
(e.g. `--top 5`, `--exclude slope,aspect`, `--only-io`, `--reset-state`)

---

## Step 0 -- Parse arguments

Parse $ARGUMENTS for these flags (multiple may combine):

| Flag | Effect |
|------|--------|
| `--top N` | Audit only the top N scored modules (default: 3) |
| `--exclude mod1,mod2` | Remove named modules from scope |
| `--only-terrain` | Restrict to: slope, aspect, curvature, terrain, terrain_metrics, hillshade, sky_view_factor |
| `--only-focal` | Restrict to: focal, convolution, morphology, bilateral, edge_detection, glcm |
| `--only-hydro` | Restrict to: flood, cost_distance, geodesic, surface_distance, viewshed, erosion, diffusion |
| `--only-io` | Restrict to: geotiff, reproject, rasterize, polygonize |
| `--reset-state` | Delete `.codex/sweep-performance-state.csv` and treat all modules as never-inspected |
| `--no-fix` | Audit only; subagents do not run /rockout. Useful for re-triage without producing PRs. |
| `--high-only` | Drop modules whose state row shows zero HIGH findings from the last triage within the past 30 days. |

## Step 0.5 -- Detect CUDA availability

After parsing arguments and before discovering modules, probe the host
for CUDA:

```bash
python -c "from numba import cuda; print(cuda.is_available())" 2>/dev/null
```

Capture the result as `CUDA_AVAILABLE` (`true` if the command prints `True`,
`false` otherwise — including import failure). Interpolate this flag into
each subagent prompt below so the agent knows whether to run cupy and
dask+cupy paths or limit itself to static review of the GPU code.

## Step 1 -- Discover modules in scope

Enumerate all candidate modules. For each, record its file path(s):

**Single-file modules:** Every `.py` file directly under `xrspatial/`, excluding
`__init__.py`, `_version.py`, `__main__.py`, `utils.py`, `accessor.py`,
`preview.py`, `dataset_support.py`, `diagnostics.py`, `analytics.py`.

**Subpackage modules:** The `geotiff/`, `reproject/`, and `hydro/` directories
under `xrspatial/`. Treat each subpackage as a single audit unit. List all
`.py` files within each (excluding `__init__.py`).

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

Read `.codex/sweep-performance-state.csv`. If it does not exist, treat every
module as never-inspected. If `--reset-state` was set, delete the file first.

State file schema (one row per module):

```
module,last_inspected,oom_verdict,bottleneck,high_count,issue,notes
slope,2026-04-15,SAFE,compute-bound,0,,"optional single-line notes"
```

- `oom_verdict` is one of `SAFE`, `RISKY`, `WILL OOM`, or `N/A`.
- `bottleneck` is one of `IO-bound`, `memory-bound`, `compute-bound`, `graph-bound`.
- `issue` is normally an integer, but may be a string token like
  `false-positive`, `fixed-in-tree`, or empty.
- `notes` is CSV-quoted; newlines must be flattened to spaces on write so
  every module stays exactly one line.

The file is registered with `merge=union` in `.gitattributes`, so two
parallel sweeps touching different modules auto-merge without conflict.
A transient duplicate-row state can occur after a merge if both branches
modified the same module; the read-update-write cycle in the agent prompt
keys rows by `module` and last-write-wins, so the next write cleans up.

### Compute scores

```
days_since_inspected = (today - last_inspected).days   # 9999 if never
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

Sort modules by score descending. Apply `--top N` (default 3).

If `--high-only` is set, drop any module whose state row shows
`high_count == 0` AND `last_inspected` is within the last 30 days. The
filter only looks at past triage results — it cannot predict findings on a
never-inspected module.

## Step 3 -- Print the ranked table and launch subagents

### 3a. Print the ranked table

Print a markdown table showing ALL scored modules (not just selected ones),
sorted by score descending:

```
| Rank | Module          | Score  | Last Inspected | Dask | CUDA | IO  | LOC  |
|------|-----------------|--------|----------------|------|------|-----|------|
| 1    | geotiff         | 30600  | never          | yes  | no   | yes | 1400 |
| 2    | viewshed        | 30050  | never          | yes  | yes  | no  | 800  |
| ...  | ...             | ...    | ...            | ...  | ...  | ... | ...  |
```

### 3b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, paths, and metadata):

~~~
You are auditing the xrspatial module "{module}" for performance issues.

This module has {commits} commits and {loc} lines of code.

Read these files: {module_files}

Also read xrspatial/utils.py for _validate_raster() behavior, and
xrspatial/tests/general_checks.py for cross-backend test helpers.

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- For Cat 3 (GPU transfer) and Cat 6 (OOM verdict), validate findings
  by actually running the cupy and dask+cupy paths. Construct a small
  cupy-backed DataArray and execute the function end-to-end. Time the
  result and confirm there is no host-device round trip.
- For register-pressure findings, compile the kernel with
  `numba.cuda.compile_ptx` or run it on a small input and report the
  observed register count rather than guessing from source.
- A /rockout fix that touches CUDA code must include a cupy run in its
  verification step before opening the PR.

If CUDA_AVAILABLE is false:
- Inspect the cupy / dask+cupy paths by reading the source only.
- Skip executing CUDA kernels and skip cupy benchmarking. Add the
  token `cuda-unavailable` to the `notes` column of the state CSV so
  a future re-run on a GPU host knows to re-validate the GPU paths.

**Your task:**

1. Read all listed files thoroughly, including the matching test file(s)
   under xrspatial/tests/.

2. Audit for these 6 categories. For each, look for the specific patterns
   described. Only flag issues ACTUALLY present in the code.

   **Cat 1 — Dask materialization**
   - HIGH: `.values` on a dask-backed DataArray or CuPy array
   - HIGH: `.compute()` inside a loop
   - HIGH: `np.array()` or `np.asarray()` wrapping a dask or CuPy array
   - MEDIUM: `da.stack()` without a following `.rechunk()`

   **Cat 2 — Dask chunking and overlap**
   - MEDIUM: `map_overlap` with depth >= chunk_size / 4
   - MEDIUM: Missing `boundary` argument in `map_overlap`
   - MEDIUM: Same function called twice on same input without caching
   - MEDIUM: Python `for` loop iterating over dask chunks

   **Cat 3 — GPU transfer**
   - HIGH: `.data.get()` followed by CuPy operations (GPU→CPU→GPU round-trip)
   - HIGH: `cupy.asarray()` inside a loop
   - MEDIUM: Mixing NumPy and CuPy ops in same function without clear reason
   - MEDIUM: Register pressure — count float64 local variables in `@cuda.jit`
     kernels; flag if >20
   - MEDIUM: Thread blocks >16x16 on kernels with >20 float64 locals

   **Cat 4 — Memory allocation**
   - MEDIUM: Unnecessary `.copy()` on arrays never mutated downstream
   - MEDIUM: Large temporary arrays that could be fused into the kernel
   - LOW: `np.zeros_like()` + fill loop where `np.empty()` would suffice

   **Cat 5 — Numba anti-patterns**
   - MEDIUM: Missing `@ngjit` on nested for-loops over `.data` arrays
   - MEDIUM: `@jit` without `nopython=True`
   - LOW: Type instability — initializing with int then assigning float
   - LOW: Column-major iteration on row-major arrays (inner loop should be
     last axis)

   **Cat 6 — 30TB / 16GB OOM verdict**
   For each dask code path, follow it end-to-end. Decide whether peak memory
   scales with chunk size or with the full array. Optionally write a small
   script under `/tmp/` (with a unique name including the module name) that
   constructs the dask task graph and reports task count and fan-in:

   ```python
   import dask.array as da
   import xarray as xr
   import json

   arr = da.zeros((2560, 2560), chunks=(256, 256), dtype='float64')
   raster = xr.DataArray(arr, dims=['y', 'x'])
   # add coords if needed
   try:
       result = MODULE_FUNCTION(raster, **DEFAULT_ARGS)
       graph = result.__dask_graph__()
       task_count = len(graph)
       print(json.dumps({
           "success": True,
           "task_count": task_count,
           "tasks_per_chunk": round(task_count / 100.0, 2),
       }))
   except Exception as e:
       print(json.dumps({"success": False, "error": str(e)}))
   ```

   The script must NEVER call `.compute()` — graph construction only.

   Verdict: one of `SAFE`, `RISKY`, `WILL OOM`, or `N/A` (no dask backend).

3. Classify the module's bottleneck as ONE of:
   `IO-bound`, `memory-bound`, `compute-bound`, `graph-bound`.

4. For each real issue found, assign a severity (CRITICAL/HIGH/MEDIUM/LOW)
   and note the exact file and line number.

5. If any CRITICAL, HIGH, or MEDIUM issue is found, run /rockout to fix it
   end-to-end (GitHub issue, worktree branch, fix, tests, and PR). Include
   the OOM verdict, bottleneck classification, and affected backends in the
   rockout prompt so it has full performance context. For LOW issues,
   document them but do not fix.

   Skip step 5 entirely if `--no-fix` was passed to the parent sweep.

6. After finishing (whether you found issues or not), update the inspection
   state file `.codex/sweep-performance-state.csv`. Header:

   `module,last_inspected,oom_verdict,bottleneck,high_count,issue,notes`

   Use this Python pattern to read, update, and write it (do NOT hand-edit
   the file -- always go through csv.DictReader / csv.DictWriter so quoting
   stays consistent):

   ```python
   import csv
   from pathlib import Path

   path = Path(".codex/sweep-performance-state.csv")
   header = ["module", "last_inspected", "oom_verdict", "bottleneck",
             "high_count", "issue", "notes"]

   rows = {}
   if path.exists():
       with path.open() as f:
           for r in csv.DictReader(f):
               rows[r["module"]] = r  # last write wins on dupes

   rows["{module}"] = {
       "module": "{module}",
       "last_inspected": "<today's ISO date, e.g. 2026-04-29>",
       "oom_verdict": "<SAFE|RISKY|WILL OOM|N/A>",
       "bottleneck": "<IO-bound|memory-bound|compute-bound|graph-bound>",
       "high_count": "<integer, count of HIGH findings>",
       "issue": "<issue number from rockout, or empty string>",
       "notes": "<single-line notes (replace any newlines with spaces), or empty>",
   }

   def _oneline(v):
       # merge=union is line-based: a newline inside a quoted field splits
       # the record on parallel-agent merges. Force one physical line per
       # record by collapsing embedded newlines to " | ".
       return "" if v is None else str(v).replace("\r\n", " | ").replace("\r", " | ").replace("\n", " | ")

   with path.open("w", newline="") as f:
       w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
       w.writeheader()
       for m in sorted(rows):
           w.writerow({k: _oneline(v) for k, v in rows[m].items()})
   ```

   Use empty strings (not `null`) for missing values. Set `issue` to the
   issue number when one was filed, otherwise leave it empty.

   Then `git add .codex/sweep-performance-state.csv` and commit it to the
   worktree branch so the state update is included in the PR.

Important:
- Only flag patterns ACTUALLY present in the code. False positives are worse
  than missed issues.
- Read the tests for this module before flagging a pattern as harmful — the
  test may codify the current behavior intentionally.
- For CUDA code, verify register pressure and bounds before flagging.
- Do NOT flag the use of numba @jit itself as a performance issue. Focus on
  what the JIT code does, not that it uses JIT.
- For the hydro subpackage: focus on one representative variant (d8) in
  detail, then note which dinf/mfd files share the same pattern. Do not read
  all 29 files line by line.
- This repo uses ArrayTypeFunctionMapping to dispatch across numpy/cupy/dask
  backends. Check all backend paths, not just numpy.
- Do NOT call `.compute()` in any analysis script. Graph construction only.
~~~

### 3c. Print a status line

After dispatching, print:

```
Launched {N} performance audit agents: {module1}, {module2}, {module3}
```

## Step 4 -- State updates

State is updated by the subagents themselves (see agent prompt step 6).
After completion, verify state with:

```
column -t -s, .codex/sweep-performance-state.csv | less
```

To reset all tracking: `/sweep-performance --reset-state`

---

## General Rules

- Do NOT modify any source files from the parent. Subagents handle fixes via
  /rockout.
- Keep the parent output concise — the ranked table and dispatch line are
  the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no
  exclusions.
- State file (`.codex/sweep-performance-state.csv`) is tracked in git, with
  `merge=union` set in `.gitattributes` so parallel sweeps touching
  different modules auto-merge. Subagents must `git add` and commit it so
  the state update lands in the PR.
- For subpackage modules (geotiff, reproject, hydro), the subagent reads ALL
  `.py` files in the subpackage directory, not just `__init__.py`.
- Only flag patterns that are ACTUALLY present in the code. Do not report
  hypothetical issues or patterns that "could" occur with imaginary inputs.
- False positives are worse than missed issues. When in doubt, skip.
- The 30TB graph simulation NEVER calls `.compute()` — it constructs the
  dask graph and inspects it.
