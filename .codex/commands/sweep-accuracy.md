# Accuracy Sweep: Dispatch subagents to audit modules for numerical accuracy issues

Audit xrspatial modules for numerical accuracy issues: floating point
precision loss, incorrect NaN propagation, off-by-one errors in neighborhood
operations, missing or wrong Earth curvature corrections, and backend
inconsistencies (numpy vs cupy vs dask results differ). Subagents fix
findings via /rockout.

Optional arguments: $ARGUMENTS
(e.g. `--top 3`, `--exclude slope,aspect`, `--only-terrain`, `--reset-state`)

---

## Step 0 -- Detect CUDA availability

Before discovering modules, probe the host for CUDA:

```bash
python -c "from numba import cuda; print(cuda.is_available())" 2>/dev/null
```

Capture the result as `CUDA_AVAILABLE` (`true` if the command prints `True`,
`false` otherwise — including import failure). Interpolate this flag into
each subagent prompt below so the agent knows whether to run cupy and
dask+cupy paths or limit itself to static review of the GPU code.

## Step 1 -- Gather module metadata via git

Enumerate candidate modules:

**Single-file modules:** Every `.py` file directly under `xrspatial/`, excluding
`__init__.py`, `_version.py`, `__main__.py`, `utils.py`, `accessor.py`,
`preview.py`, `dataset_support.py`, `diagnostics.py`, `analytics.py`.

**Subpackage modules:** `geotiff/`, `reproject/`, and `hydro/` directories under
`xrspatial/`. Treat each as a single audit unit. List all `.py` files within
each (excluding `__init__.py`).

For every module, collect:

| Field | How |
|-------|-----|
| **last_modified** | `git log -1 --format=%aI -- <path>` (for subpackages, most recent file) |
| **total_commits** | `git log --oneline -- <path> \| wc -l` |
| **loc** | `wc -l < <path>` (for subpackages, sum all files) |
| **recent_accuracy_commits** | `git log --oneline --grep='accuracy\|precision\|numerical\|geodesic' -- <path>` |

Store results in memory -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read `.codex/sweep-accuracy-state.csv`.

If it does not exist, treat every module as never-inspected.

If `$ARGUMENTS` contains `--reset-state`, delete the file and treat
everything as never-inspected.

State file schema (one row per module):

```
module,last_inspected,issue,severity_max,categories_found,notes
slope,2026-03-28,1042,HIGH,1;3,"optional single-line notes"
```

- `categories_found` is a semicolon-separated integer list (empty when null).
- `notes` is CSV-quoted; newlines must be flattened to spaces on write so
  every module stays exactly one line.

The file is registered with `merge=union` in `.gitattributes`, so two
parallel sweeps touching different modules auto-merge without conflict.
A transient duplicate-row state can occur after a merge if both branches
modified the same module; the read-update-write cycle in step 5 keys rows
by `module` and last-write-wins, so the next write cleans up.

## Step 3 -- Score each module

```
days_since_inspected = (today - last_inspected).days   # 9999 if never
days_since_modified  = (today - last_modified).days
has_recent_accuracy_work = 1 if recent_accuracy_commits is non-empty, else 0

score = (days_since_inspected * 3)
      + (total_commits * 0.5)
      - (days_since_modified * 0.2)
      - (has_recent_accuracy_work * 500)
      + (loc * 0.05)
```

Rationale:
- Modules never inspected dominate (9999 * 3)
- More commits = more complex = more likely to have accuracy bugs
- Recently modified modules slightly deprioritized (someone just touched them)
- Modules with existing accuracy work heavily deprioritized
- Larger files have more surface area (0.05 per line)

## Step 4 -- Apply filters from $ARGUMENTS

- `--top N` -- only audit the top N modules (default: 3)
- `--exclude mod1,mod2` -- remove named modules from the list
- `--only-terrain` -- restrict to: slope, aspect, curvature, terrain,
  terrain_metrics, hillshade, sky_view_factor
- `--only-focal` -- restrict to: focal, convolution, morphology, bilateral,
  edge_detection, glcm
- `--only-hydro` -- restrict to: flood, cost_distance, geodesic,
  surface_distance, viewshed, erosion, diffusion, hydro (subpackage)
- `--only-io` -- restrict to: geotiff, reproject, rasterize, polygonize

## Step 5 -- Print the ranked table and launch subagents

### 5a. Print the ranked table

Print a markdown table showing ALL scored modules (not just selected ones),
sorted by score descending:

```
| Rank | Module          | Score  | Last Inspected | Last Modified | Commits | LOC  |
|------|-----------------|--------|----------------|---------------|---------|------|
| 1    | viewshed        | 30012  | never          | 45 days ago   | 23      | 800  |
| 2    | flood           | 29998  | never          | 120 days ago  | 18      | 600  |
| ...  | ...             | ...    | ...            | ...           | ...     | ...  |
```

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, paths, and metadata):

```
You are auditing the xrspatial module "{module}" for numerical accuracy issues.

This module has {commits} commits and {loc} lines of code.

Read these files: {module_files}

Also read xrspatial/utils.py to understand _validate_raster() behavior and
xrspatial/tests/general_checks.py for the cross-backend comparison helpers.

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- When auditing the cupy / dask+cupy backends, actually run the matching
  tests in xrspatial/tests/ against those backends. The cross-backend
  helpers in general_checks.py already dispatch to all four backends —
  invoke them directly so cupy and dask+cupy paths execute, not just
  numpy.
- For CUDA-specific findings (kernel correctness, NaN propagation in
  device code, backend divergence), validate by running the kernel on
  a small input rather than reasoning from source alone.
- A /rockout fix that touches CUDA code must include a cupy run in its
  verification step before opening the PR.

If CUDA_AVAILABLE is false:
- Read the cupy / dask+cupy paths and flag patterns by inspection only.
- Skip executing tests on those backends. Add the token
  `cuda-unavailable` to the `notes` column of the state CSV so a future
  re-run on a GPU host knows to re-validate the GPU paths.

**Your task:**

1. Read all listed files thoroughly, including the matching test file(s)
   under xrspatial/tests/ so you understand expected behavior.

2. Audit for these 5 accuracy categories. For each, look for the specific
   patterns described. Only flag issues ACTUALLY present in the code.

   **Cat 1 — Floating Point Precision Loss**
   - Accumulation loops that sum many small values into a large running
     total without Kahan summation or compensated accumulation
   - float32 used where float64 is required for stable intermediate results
     (e.g. large grids, long gradients, iterative solvers)
   - Subtraction of nearly-equal large quantities (catastrophic cancellation)
   - Division by small numbers without a stability floor
   Severity: HIGH if the result is visibly wrong on realistic inputs;
   MEDIUM if only observable on adversarial inputs

   **Cat 2 — NaN / Inf Propagation Errors**
   - NaN input silently produces a finite output (masked, skipped, or
     treated as zero without being documented)
   - NaN check using `==` instead of `!= x` for NaN detection in numba
   - Neighborhood operations that ignore NaN pixels but do not update the
     normalization denominator, biasing the result
   - Inf / -Inf inputs treated as numbers in comparisons without guards
   - Divide-by-zero producing Inf that then corrupts downstream accumulation
   Severity: HIGH if NaN input yields a wrong but finite output;
   MEDIUM if the behavior is documented but still surprising

   **Cat 3 — Off-by-One Errors in Neighborhood Operations**
   - Loop bounds that exclude the last row/column (e.g. `range(H-1)` where
     `range(H)` is intended)
   - `map_overlap` depth that is smaller than the actual stencil radius
   - Boundary handling that duplicates or skips edge pixels
   - Asymmetric kernel indexing (one-sided rather than centered)
   - CUDA kernel bounds guard that is `i > H` instead of `i >= H`
   Severity: HIGH if it causes a silent wrong result at all chunk boundaries;
   MEDIUM if it only affects a single-pixel edge

   **Cat 4 — Missing or Wrong Earth Curvature / Projection Corrections**
   - Geodesic calculations that assume a flat projection without curvature
     correction (see slope.py, aspect.py, geodesic.py for the reference
     pattern: `u += (e² + n²) / (2R)`)
   - Haversine / great-circle distance using the wrong Earth radius
     constant, or using a spherical approximation where WGS84 is needed
   - Mixing projected and geographic coordinates in the same calculation
     without a transform
   - Using cell size in degrees as if it were meters
   Severity: HIGH if the correction is missing entirely on a public API;
   MEDIUM if the correction is present but uses a questionable constant

   **Cat 5 — Backend Inconsistency (numpy vs cupy vs dask)**
   - numpy and cupy paths use different algorithms that can diverge on
     identical inputs (e.g. different boundary handling, different NaN
     semantics, different numerical precision)
   - dask path silently falls back to materializing the full array
   - dask `map_overlap` chunk function returns a different shape than the
     input, corrupting the reassembled array
   - A backend raises on valid input that another backend accepts
   - Result dtype differs across backends without documentation
   Severity: HIGH if numerically different results on the same input;
   MEDIUM if only metadata (dtype, coords) differs

3. For each real issue found, assign a severity (CRITICAL/HIGH/MEDIUM/LOW)
   and note the exact file and line number.

4. If any CRITICAL, HIGH, or MEDIUM issue is found, run /rockout to fix it
   end-to-end (GitHub issue, worktree branch, fix, tests, and PR).
   For LOW issues, document them but do not fix.

5. After finishing (whether you found issues or not), update the inspection
   state file .codex/sweep-accuracy-state.csv. The file is row-per-module
   CSV with header:

   `module,last_inspected,issue,severity_max,categories_found,notes`

   Use this Python pattern to read, update, and write it (do NOT hand-edit
   the file -- always go through csv.DictReader / csv.DictWriter so quoting
   stays consistent):

   ```python
   import csv
   from pathlib import Path

   path = Path(".codex/sweep-accuracy-state.csv")
   header = ["module", "last_inspected", "issue", "severity_max",
             "categories_found", "notes"]

   rows = {}
   if path.exists():
       with path.open() as f:
           for r in csv.DictReader(f):
               rows[r["module"]] = r  # last write wins on dupes

   rows["{module}"] = {
       "module": "{module}",
       "last_inspected": "<today's ISO date, e.g. 2026-04-27>",
       "issue": "<issue number from rockout, or empty string>",
       "severity_max": "<HIGH|MEDIUM|LOW, or empty>",
       "categories_found": "<semicolon-joined ints, e.g. 1;3, or empty>",
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

   Then `git add .codex/sweep-accuracy-state.csv` and commit it to the
   worktree branch so the state update is included in the PR.

Important:
- Only flag real accuracy issues. False positives waste time.
- Read the tests for this module to understand expected behavior before
  flagging a result as wrong -- the test may codify the current behavior.
- For backend comparisons, check that the cross-backend tests in
  xrspatial/tests/general_checks.py actually exercise the code path you
  are suspicious of; missing test coverage is itself a finding.
- Do NOT flag the use of numba @jit itself as an accuracy issue. Focus on
  what the JIT code does, not that it uses JIT.
- For the hydro subpackage: focus on one representative variant (d8) in
  detail, then note which dinf/mfd files share the same pattern. Do not
  read all 29 files line by line.
- This repo uses ArrayTypeFunctionMapping to dispatch across numpy/cupy/dask
  backends. Check all backend paths, not just numpy.
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} accuracy audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

State is updated by the subagents themselves (see agent prompt step 5).
After completion, verify state with:

```
column -t -s, .codex/sweep-accuracy-state.csv | less
```

To reset all tracking: `/sweep-accuracy --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via /rockout.
- Keep the output concise -- the table and agent dispatch are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no exclusions.
- State file (`.codex/sweep-accuracy-state.csv`) is tracked in git, with
  `merge=union` set in `.gitattributes` so parallel sweeps touching
  different modules auto-merge. Subagents must `git add` and commit it so
  the state update lands in the PR.
- For subpackage modules (geotiff, reproject, hydro), the subagent should read
  ALL `.py` files in the subpackage directory, not just `__init__.py`.
- Only flag patterns that are ACTUALLY present in the code. Do not report
  hypothetical issues or patterns that "could" occur with imaginary inputs.
- False positives are worse than missed issues. When in doubt, skip.
