# Security Sweep: Dispatch subagents to audit modules for security vulnerabilities

Audit xrspatial modules for security issues specific to numeric/GPU raster
libraries: unbounded allocations, integer overflow, NaN logic bombs, GPU
kernel bounds, file path injection, and dtype confusion. Subagents fix
CRITICAL/HIGH issues via /rockout.

Optional arguments: $ARGUMENTS
(e.g. `--top 3`, `--exclude slope,aspect`, `--only-io`, `--reset-state`)

---

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

Read `.claude/security-sweep-state.json`.

If it does not exist, treat every module as never-inspected.

If `$ARGUMENTS` contains `--reset-state`, delete the file and treat
everything as never-inspected.

State file schema:

```json
{
  "inspections": {
    "cost_distance": {
      "last_inspected": "2026-04-10T14:00:00Z",
      "issue": 1150,
      "severity_max": "HIGH",
      "categories_found": [1, 2]
    }
  }
}
```

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

4. If any CRITICAL or HIGH issue is found, run /rockout to fix it end-to-end
   (GitHub issue, worktree branch, fix, tests, and PR).
   For MEDIUM/LOW issues, document them but do not fix.

5. After finishing (whether you found issues or not), update the inspection
   state file .claude/security-sweep-state.json by reading its current
   contents and adding/updating the entry for "{module}" with:
   - "last_inspected": today's ISO date
   - "issue": the issue number from rockout (or null if clean / MEDIUM-only)
   - "severity_max": highest severity found (or null if clean)
   - "categories_found": list of category numbers that had findings (e.g. [1, 2])

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
cat .claude/security-sweep-state.json
```

To reset all tracking: `/sweep-security --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via /rockout.
- Keep the output concise -- the table and agent dispatch are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no exclusions.
- State file (`.claude/security-sweep-state.json`) is gitignored by convention --
  do not add it to git.
- For subpackage modules (geotiff, reproject, hydro), the subagent should read
  ALL `.py` files in the subpackage directory, not just `__init__.py`.
- Only flag patterns that are ACTUALLY present in the code. Do not report
  hypothetical issues or patterns that "could" occur with imaginary inputs.
- False positives are worse than missed issues. When in doubt, skip.
