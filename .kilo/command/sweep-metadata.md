# Metadata Propagation Sweep: Dispatch subagents to audit modules for metadata preservation

Audit xrspatial modules for metadata propagation bugs: attrs (especially
`res`, `crs`, `transform`, `nodatavals`, `_FillValue`), coords (x/y values
and dims), and dim names. Spatial libs lose CRS/transform silently and the
result looks correct but is wrong. The sky_view_factor cellsize bug
(#1407) was exactly this class of issue. Subagents fix CRITICAL, HIGH, and
MEDIUM findings via rockout.

Optional arguments: {{ARGUMENTS}}
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
| **public_funcs** | count of functions defined at module level (heuristic: `^def [a-z]` not starting with `_`) |

Store results in memory -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read `.kilo/worktrees/sweep-metadata-state.csv`.

If it does not exist, treat every module as never-inspected.

If `{{ARGUMENTS}}` contains `--reset-state`, delete the file and treat
everything as never-inspected.

State file schema (one row per module):

```
module,last_inspected,issue,severity_max,categories_found,notes
slope,2026-05-01,1042,HIGH,1;3,"optional single-line notes"
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

score = (days_since_inspected * 3)
      + (public_funcs * 5)
      + (total_commits * 0.3)
      - (days_since_modified * 0.2)
      + (loc * 0.05)
```

Rationale:
- Modules never inspected dominate (9999 * 3)
- More public functions = more API surface that could lose metadata
- More commits = more refactor risk for metadata propagation
- Recently modified modules slightly deprioritized
- Larger files have more surface area

## Step 4 -- Apply filters from {{ARGUMENTS}}

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

Print a markdown table showing ALL scored modules sorted by score descending.

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, paths, and metadata):

```
You are auditing the xrspatial module "{module}" for metadata propagation issues.

This module has {commits} commits and {loc} lines of code.

Read these files: {module_files}

Also read xrspatial/utils.py to understand:
- _validate_raster() behavior — what does it accept/reject?
- get_dataarray_resolution() — what attrs does it pull from?
- ngjit / ArrayTypeFunctionMapping dispatch helpers

Read xrspatial/tests/general_checks.py for cross-backend test helpers.

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- For Cat 1 (attrs), Cat 2 (coords), Cat 3 (dims), Cat 4 (dtype/nodata),
  and Cat 5 (backend-inconsistent metadata), construct cupy and
  dask+cupy DataArrays and run the function end-to-end. Check
  attrs/coords/dims on the actual returned object — do not infer from
  source.
- A rockout fix that touches metadata-emitting code must verify all
  four backends (numpy, cupy, dask+numpy, dask+cupy) before opening
  the PR.

If CUDA_AVAILABLE is false:
- Inspect the cupy / dask+cupy paths by reading the source only.
- Skip executing tests on those backends. Add the token
  `cuda-unavailable` to the `notes` column of the state CSV so a
  future re-run on a GPU host knows to re-validate the GPU paths.

**Your task:**

1. Read all listed files thoroughly, including the matching test file(s)
   under xrspatial/tests/ so you understand expected behavior. Pay
   particular attention to whether tests assert on attrs/coords/dims of
   the returned DataArray.

2. Audit for these 5 metadata-propagation categories. Only flag issues
   ACTUALLY present in the code.

   **Cat 1 — attrs preservation**
   - HIGH: result DataArray has empty attrs even though input had attrs
     (`return xr.DataArray(out_data, dims=...)` instead of `dims=in.dims,
     attrs=in.attrs`)
   - HIGH: function silently drops `res`, `crs`, `transform`, or
     `nodatavals` from input attrs
   - HIGH: function reads `attrs['res']` for math but does not re-emit it
     on output (downstream callers see no res, recompute from coords,
     get different answer)
   - MEDIUM: function copies attrs but adds an inferred attr that
     overwrites a user-provided value (e.g. always sets `nodatavals` to
     `[np.nan]` even if input had `[-9999]`)
   - MEDIUM: attrs propagated for the eager path but lost on the dask path
     (or vice versa)
   Severity: HIGH if downstream spatial computation is affected (slope of
   a no-CRS raster gives wrong cell-size answers); MEDIUM otherwise

   **Cat 2 — coords preservation**
   - HIGH: result has integer-index coords (0,1,2,...) when input had
     georeferenced coords (lon/lat or projected x/y)
   - HIGH: coordinate values are stale by half-a-pixel after resampling
     (centre vs corner convention drift)
   - HIGH: coord dtype changes (float64 → float32) silently between input
     and output
   - MEDIUM: extra coords from input (e.g. `time`, `band`) are dropped on
     output even though they should pass through
   - MEDIUM: coord names renamed without the function documenting why
     (`x` → `lon`, `y` → `lat`, etc.)
   Severity: HIGH if downstream coord-based math (clipping, interp) breaks

   **Cat 3 — dim names and order**
   - HIGH: output dim order differs from input dim order without
     documentation (e.g. input `(y, x)`, output `(x, y)`)
   - HIGH: output has fewer/more dims than input without the function
     docstring saying so (e.g. reduces over `y` but doesn't reflect that
     in the dim list)
   - MEDIUM: function assumes hardcoded dim names (`y`, `x`) and silently
     mis-aligns when input uses (`lat`, `lon`) or (`row`, `col`)
   - MEDIUM: dask backend preserves dims, numpy backend does not (or vice
     versa)
   Severity: HIGH if it breaks chained xarray operations

   **Cat 4 — dtype and nodata semantics**
   - HIGH: function reads `attrs['nodatavals']` for input mask but does
     not propagate it to output (so a chained call sees the old nodata,
     possibly wrong)
   - HIGH: output dtype hardcoded to float64 even when input was uint8
     (memory blowup; downstream stats wrong)
   - MEDIUM: NaN used as the nodata sentinel internally but output dtype
     is integer (NaN cannot represent — silent conversion to MIN_INT or 0)
   - MEDIUM: `_FillValue` attr present on input but not on output
   Severity: HIGH if nodata mask is silently flipped or dtype change
   causes wrong arithmetic downstream

   **Cat 5 — backend-inconsistent metadata**
   - HIGH: numpy and cupy backends emit attrs differently (e.g. numpy
     keeps `crs`, cupy drops it, or numpy emits `_FillValue`, cupy emits
     `nodatavals`)
   - HIGH: dask path's metadata is computed from chunk-local stats not
     global stats (e.g. `attrs['min']` is per-chunk min, not global min)
   - MEDIUM: only one of the four backends (numpy / cupy / dask+numpy /
     dask+cupy) preserves attrs
   - MEDIUM: result name (`.name`) inconsistent across backends
   Severity: HIGH if a chained pipeline silently produces different
   numbers depending on which backend is active

3. For each real issue found, assign a severity (CRITICAL/HIGH/MEDIUM/LOW)
   and note the exact file and line number.

4. If any CRITICAL, HIGH, or MEDIUM issue is found, run rockout to fix it
   end-to-end (GitHub issue, worktree branch, fix, tests, and PR). For
   LOW issues, document them but do not fix.

5. After finishing (whether you found issues or not), update the inspection
   state file .kilo/worktrees/sweep-metadata-state.csv. Header:

   `module,last_inspected,issue,severity_max,categories_found,notes`

   Use this Python pattern (do NOT hand-edit the file):

   ```python
   import csv
   from pathlib import Path

   path = Path(".kilo/worktrees/sweep-metadata-state.csv")
   header = ["module", "last_inspected", "issue", "severity_max",
             "categories_found", "notes"]

   rows = {}
   if path.exists():
       with path.open() as f:
           for r in csv.DictReader(f):
               rows[r["module"]] = r

   rows["{module}"] = {
       "module": "{module}",
       "last_inspected": "<today's ISO date, e.g. 2026-05-03>",
       "issue": "<issue number from rockout, or empty>",
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

   Use empty strings (not `null`) for missing values.

   Then `git add .kilo/worktrees/sweep-metadata-state.csv` and commit it to the
   worktree branch so the state update lands in the PR.

Important:
- Only flag real metadata propagation issues. False positives waste time.
- Read the tests for this module before flagging — the test may codify
  the current behavior intentionally (e.g. an aggregation that genuinely
  drops a dim).
- Verify by reading the function end-to-end: does the input DataArray's
  attrs/coords/dims get propagated to the returned DataArray?
- For ALL backends, not just numpy. Check numpy / cupy / dask+numpy /
  dask+cupy paths.
- Do NOT flag the use of numba @jit itself.
- For the hydro subpackage: focus on one representative variant (d8) in
  detail, then note which dinf/mfd files share the same pattern.
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} metadata propagation audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

State is updated by the subagents themselves. After completion, verify with:

```
column -t -s, .kilo/worktrees/sweep-metadata-state.csv | less
```

To reset all tracking: `sweep-metadata --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via rockout.
- Keep the parent output concise — the ranked table and dispatch line are
  the deliverables.
- If {{ARGUMENTS}} is empty, use defaults: top 3, no category filter, no
  exclusions.
- State file (`.kilo/worktrees/sweep-metadata-state.csv`) is tracked in git, with
  `merge=union` set in `.gitattributes` so parallel sweeps touching
  different modules auto-merge.
- For subpackage modules (geotiff, reproject, hydro), the subagent should
  read ALL `.py` files in the subpackage directory, not just `__init__.py`.
- Only flag patterns that are ACTUALLY present in the code.
- False positives are worse than missed issues. When in doubt, skip.
