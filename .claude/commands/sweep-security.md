# Security Sweep: Dispatch subagents to audit modules for security vulnerabilities

Audit xrspatial modules for security issues specific to numeric/GPU raster
libraries: unbounded allocations, integer overflow, NaN logic bombs, GPU
kernel bounds, file path injection, and dtype confusion. Subagents fix
CRITICAL, HIGH, and MEDIUM severity issues via /rockout.

Optional arguments: $ARGUMENTS
(e.g. `--top 3`, `--exclude slope,aspect`, `--only-io`, `--reset-state`)

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

## Step 1 -- Gather module metadata via git and grep

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
| **has_cuda_kernels** | grep file(s) for `@cuda.jit` |
| **has_file_io** | grep file(s) for `open(`, `mkstemp`, `os.path`, `pathlib` |
| **has_numba_jit** | grep file(s) for `@ngjit`, `@njit`, `@jit`, `numba.jit` |
| **allocates_from_dims** | grep file(s) for `np.empty(height`, `np.zeros(height`, `np.empty(H`, `np.empty(h `, `cp.empty(`, and width variants |
| **has_shared_memory** | grep file(s) for `cuda.shared.array` |

Store results in memory -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read `.claude/sweep-security-state.csv`.

If it does not exist, treat every module as never-inspected.

If `$ARGUMENTS` contains `--reset-state`, delete the file and treat
everything as never-inspected.

State file schema (one row per module):

```
module,last_inspected,issue,severity_max,categories_found,followup_issues,notes
cost_distance,2026-04-10,1150,HIGH,1;2,,"optional single-line notes"
```

- `categories_found` and `followup_issues` are semicolon-separated integer
  lists (empty when null).
- `notes` is CSV-quoted; newlines must be flattened to spaces on write so
  every module stays exactly one line.

The file uses git's default 3-way text merge (no `merge=union`; see
issue #2754). Two parallel sweeps that touch the CSV surface a normal
merge conflict rather than silently unioning duplicate rows. Resolve a
conflict by keeping one row per `module` (latest `last_inspected` wins),
a single header, and one physical line per record -- or just re-run the
read-update-write cycle in step 5, which rewrites the whole canonical
file.

## Step 3 -- Score each module

```
days_since_inspected = (today - last_inspected).days   # 9999 if never
days_since_modified  = (today - last_modified).days

score = (days_since_inspected * 3)
      + (has_file_io * 400)
      + (allocates_from_dims * 300)
      + (has_cuda_kernels * 250)
      + (has_shared_memory * 200)
      + (has_numba_jit * 100)
      + (loc * 0.05)
      - (days_since_modified * 0.2)
```

Rationale:
- File I/O is the only external-escape vector (400)
- Unbounded allocation is a DoS vector across all backends (300)
- CUDA bugs cause silent memory corruption (250)
- Shared memory overflow is a CUDA sub-risk (200)
- Numba JIT is ubiquitous -- lower weight avoids noise (100)
- Larger files have more surface area (0.05 per line)
- Recently modified code slightly deprioritized

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
| Rank | Module          | Score  | Last Inspected | CUDA | FileIO | Alloc | Numba | LOC  |
|------|-----------------|--------|----------------|------|--------|-------|-------|------|
| 1    | geotiff         | 30600  | never          | yes  | yes    | no    | yes   | 1400 |
| 2    | hydro           | 30300  | never          | yes  | no     | yes   | yes   | 8200 |
| ...  | ...             | ...    | ...            | ...  | ...    | ...   | ...   | ...  |
```

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, paths, and metadata):

```
You are auditing the xrspatial module "{module}" for security vulnerabilities.

This module has {commits} commits and {loc} lines of code.

Read these files: {module_files}

Also read xrspatial/utils.py to understand _validate_raster() behavior.

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- For Cat 4 (GPU kernel bounds), validate suspected missing bounds
  guards by running the kernel on adversarial input shapes (1x1, Nx1,
  large prime dimensions) and confirm no out-of-bounds access. Use
  `compute-sanitizer` if installed; otherwise rely on test runs that
  exercise edge sizes.
- For Cat 1 (unbounded allocation) on cupy paths, confirm the
  allocation actually executes on the GPU and observe peak memory via
  `cupy.cuda.runtime.memGetInfo()` rather than reasoning from source.
- A /rockout fix that touches CUDA code must include a cupy run in its
  verification step before opening the PR.

If CUDA_AVAILABLE is false:
- Inspect the cupy / dask+cupy paths and CUDA kernels by reading the
  source only.
- Skip executing CUDA kernels. Add the token `cuda-unavailable` to the
  `notes` column of the state CSV so a future re-run on a GPU host
  knows to re-validate the GPU paths.

**Your task:**

1. Read all listed files thoroughly.

2. Audit for these 6 security categories. For each, look for the specific
   patterns described. Only flag issues ACTUALLY present in the code.

   **Cat 1 — Unbounded Allocation / Denial of Service**
   - np.empty(), np.zeros(), np.full() where size comes from array dimensions
     (height*width, H*W, nrows*ncols) without a configurable max or memory check
   - CuPy equivalents (cp.empty, cp.zeros)
   - Queue/heap arrays sized at height*width without bounds validation
   Severity: HIGH if no memory guard exists; MEDIUM if a partial guard exists

   **Cat 2 — Integer Overflow in Index Math**
   - height*width multiplication in int32 (overflows silently at ~46340x46340)
   - Flat index calculations (r*width + c) in numba JIT without overflow check
   - Queue index variables in int32 that could overflow for large arrays
   Severity: HIGH for int32 overflow in production paths; MEDIUM for int64
   overflow only possible with unrealistic dimensions (>3 billion pixels)

   **Cat 3 — NaN/Inf as Logic Errors**
   - Division without zero-check in numba kernels
   - log/sqrt of potentially negative values without guard
   - Accumulation loops that could hit Inf (summing many large values)
   - Missing NaN propagation: NaN input silently produces finite output
   - Incorrect NaN check: using == instead of != for NaN detection in numba
   Severity: HIGH if in flood routing, erosion, viewshed, or cost_distance
   (safety-critical modules); MEDIUM otherwise

   **Cat 4 — GPU Kernel Bounds Safety**
   - CUDA kernels missing `if i >= H or j >= W: return` bounds guard
   - cuda.shared.array with fixed size that could overflow with adversarial
     input parameters
   - Missing cuda.syncthreads() after shared memory writes before reads
   - Thread block dimensions that could cause register spill or launch failure
   Severity: CRITICAL if bounds guard is missing (out-of-bounds GPU write);
   HIGH for shared memory overflow or missing syncthreads

   **Cat 5 — File Path Injection**
   - File paths constructed from user strings without os.path.realpath() or
     os.path.abspath() canonicalization
   - Path traversal via ../ not prevented
   - Temporary file creation in user-controlled directories
   Severity: CRITICAL if user-provided path is used without any
   canonicalization; HIGH if partial canonicalization is bypassable

   **Cat 6 — Dtype Confusion**
   - Public API functions that do NOT call _validate_raster() on their inputs
   - Numba kernels that assume float64 but could receive float32 or int arrays
   - Operations where dtype mismatch causes silent wrong results (not an error)
   - CuPy/NumPy backend inconsistency in dtype handling
   Severity: HIGH if wrong results are silent; MEDIUM if an error occurs but
   the error message is misleading

3. For each real issue found, assign a severity (CRITICAL/HIGH/MEDIUM/LOW)
   and note the exact file and line number.

4. If any CRITICAL, HIGH, or MEDIUM issue is found, run /rockout to fix it
   end-to-end (GitHub issue, worktree branch, fix, tests, and PR).
   For LOW issues, document them but do not fix.

5. After finishing (whether you found issues or not), update the inspection
   state file .claude/sweep-security-state.csv. The file is row-per-module
   CSV with header:

   `module,last_inspected,issue,severity_max,categories_found,followup_issues,notes`

   Use this Python pattern to read, update, and write it (do NOT hand-edit
   the file -- always go through csv.DictReader / csv.DictWriter so quoting
   stays consistent):

   ```python
   import csv
   from pathlib import Path

   path = Path(".claude/sweep-security-state.csv")
   header = ["module", "last_inspected", "issue", "severity_max",
             "categories_found", "followup_issues", "notes"]

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
       "categories_found": "<semicolon-joined ints, e.g. 1;2, or empty>",
       "followup_issues": "<semicolon-joined ints, or empty>",
       "notes": "<single-line notes (replace any newlines with spaces), or empty>",
   }

   def _oneline(v):
       # Git merges these CSVs line by line, so a newline inside a quoted
       # field splits the record on a merge. Force one physical line per
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

   Then `git add .claude/sweep-security-state.csv` and commit it to the
   worktree branch so the state update is included in the PR.

Important:
- Only flag real, exploitable issues. False positives waste time.
- Read the tests for this module to understand expected behavior.
- For CUDA code, verify bounds guards are truly missing -- many kernels already
  have `if i >= H or j >= W: return`.
- Do NOT flag the use of numba @jit itself as a security issue. Focus on what
  the JIT code does, not that it uses JIT.
- For the hydro subpackage: focus on one representative variant (d8) in detail,
  then note which dinf/mfd files share the same pattern. Do not read all 29
  files line by line.
- This repo uses ArrayTypeFunctionMapping to dispatch across numpy/cupy/dask
  backends. Check all backend paths, not just numpy.
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} security audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

State is updated by the subagents themselves (see agent prompt step 5).
After completion, verify state with:

```
column -t -s, .claude/sweep-security-state.csv | less
```

To reset all tracking: `/sweep-security --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via /rockout.
- Keep the output concise -- the table and agent dispatch are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no exclusions.
- State file (`.claude/sweep-security-state.csv`) is tracked in git and uses
  git's default 3-way text merge (no `merge=union`; see issue #2754), so a
  concurrent change surfaces a conflict instead of silently unioning
  duplicate rows. Subagents must `git add` and commit it so the state
  update lands in the PR.
- For subpackage modules (geotiff, reproject, hydro), the subagent should read
  ALL `.py` files in the subpackage directory, not just `__init__.py`.
- Only flag patterns that are ACTUALLY present in the code. Do not report
  hypothetical issues or patterns that "could" occur with imaginary inputs.
- False positives are worse than missed issues. When in doubt, skip.
