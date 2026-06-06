# API Consistency Sweep: Dispatch subagents to audit parameter naming and signature drift

Audit xrspatial modules for API consistency issues across analogous public
functions: parameter naming drift (`cellsize` vs `cell_size` vs `res`,
`agg` vs `raster` vs `data`), inconsistent return-type shapes, missing or
mismatched type hints, docstring/signature divergence. Cheap to find; makes
the library feel polished and predictable. Subagents fix CRITICAL, HIGH,
and MEDIUM findings via rockout — but flag deprecation impact in the
issue since renames are breaking changes.

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
`xrspatial/`. Treat each as a single audit unit.

For every module, collect:

| Field | How |
|-------|-----|
| **last_modified** | `git log -1 --format=%aI -- <path>` |
| **total_commits** | `git log --oneline -- <path> \| wc -l` |
| **loc** | `wc -l < <path>` |
| **public_funcs** | count of functions at module level (heuristic: `^def [a-z]`) |

Store results in memory -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read `.kilo/worktrees/sweep-api-consistency-state.csv`.

If it does not exist, treat every module as never-inspected. If
`{{ARGUMENTS}}` contains `--reset-state`, delete the file first.

State file schema (one row per module):

```
module,last_inspected,issue,severity_max,categories_found,notes
slope,2026-05-01,1042,HIGH,1;3,"optional single-line notes"
```

The file is registered with `merge=union` in `.gitattributes`.

## Step 3 -- Score each module

```
days_since_inspected = (today - last_inspected).days   # 9999 if never
days_since_modified  = (today - last_modified).days

score = (days_since_inspected * 3)
      + (public_funcs * 8)
      + (total_commits * 0.3)
      - (days_since_modified * 0.1)
      + (loc * 0.03)
```

Rationale:
- Public function count weighted heavily — consistency issues are
  cross-function comparisons, so more functions = more comparison surface
- Modules never inspected dominate
- Recently modified slightly deprioritized

## Step 4 -- Apply filters from {{ARGUMENTS}}

Same filter set as other sweeps: `--top N`, `--exclude`, `--only-terrain`,
`--only-focal`, `--only-hydro`, `--only-io`, `--reset-state`.

## Step 5 -- Print the ranked table and launch subagents

### 5a. Print the ranked table

Print a markdown table showing ALL scored modules sorted by score descending.

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained:

```
You are auditing the xrspatial module "{module}" for API consistency issues.

This module has {commits} commits and {loc} lines of code.

Read these files: {module_files}

Also read xrspatial/__init__.py to see what is publicly re-exported, and
xrspatial/utils.py for shared helpers.

For comparison, read 2-3 sibling modules (analogous functions). Examples:
- For aspect: also read slope.py and curvature.py
- For erosion: also read morphology.py
- For glcm: also read focal.py and convolution.py
The point is to compare parameter naming and return shapes against
modules with similar function families.

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- When checking signature parity, also import the cupy backend variants
  and confirm they accept the same kwargs. Run a quick smoke test on a
  cupy DataArray for each public function so signature drift between
  numpy and cupy paths surfaces.
- A rockout fix that touches public signatures must verify both numpy
  and cupy entry points before opening the PR.

If CUDA_AVAILABLE is false:
- Inspect the cupy backend signatures by reading the source only.
- Add the token `cuda-unavailable` to the `notes` column of the state
  CSV so a future re-run on a GPU host knows to re-validate the cupy
  signatures.

**Your task:**

1. Read all listed files thoroughly. For each public function, build a
   small mental table of (function name, signature, return type).

2. Audit for these 5 API-consistency categories. Only flag issues ACTUALLY
   present.

   **Cat 1 — Parameter naming drift**
   - HIGH: same concept named differently across analogous public
     functions in this module or in sibling modules. Common offenders:
     `cellsize` vs `cell_size` vs `res` vs `resolution`
     `agg` vs `raster` vs `data` vs `array`
     `x` vs `xs` vs `x_coords`
     `nodata` vs `_FillValue` vs `nodata_value`
     `cmap` vs `color_map` vs `colormap`
     `kernel` vs `weights` vs `mask`
   - MEDIUM: same concept named consistently inside this module but
     different from sibling modules
   - MEDIUM: positional-vs-keyword convention drift (sibling functions
     accept the same arg, one as positional, one as keyword-only)
   Severity: HIGH if both names exist in the public API at the same time
   (real user-facing inconsistency); MEDIUM otherwise

   **Cat 2 — Return shape drift**
   - HIGH: analogous functions return different types (one returns
     DataArray, sibling returns Dataset for the same conceptual op)
   - HIGH: tuple-return vs single-return drift (one function returns
     `(slope, aspect)`, analog returns `slope` only — caller cannot
     interchange)
   - MEDIUM: result coord/attr conventions differ (one function emits
     `attrs['units']`, sibling does not)
   - MEDIUM: in-place vs returned-copy semantics drift
   Severity: HIGH if it breaks substitutability between sibling functions

   **Cat 3 — Type hints and docstrings**
   - MEDIUM: missing type hints on a public function while sibling
     functions in this module have them
   - MEDIUM: type hint says `xr.DataArray` but the docstring example
     passes a numpy array (or vice versa) — docs/types disagree
   - MEDIUM: docstring lists a parameter that does not exist in the
     signature (or omits one that does)
   - MEDIUM: docstring says "Returns: DataArray" but the function returns
     a tuple
   - LOW: docstring style drift (numpy-style vs google-style mix)
   Severity: MEDIUM (these are documentation bugs that mislead users)

   **Cat 4 — Default value inconsistency**
   - HIGH: same parameter has different defaults in analogous functions
     (e.g. `kernel_size=3` in one function, `kernel_size=5` in sibling,
     no documented reason)
   - MEDIUM: default uses a mutable type (`def f(x=[])`) — Python anti-pattern
   - MEDIUM: default `None` plus internal substitution where a literal
     default would be clearer and equally correct
   Severity: HIGH if user-surprise is likely (silent behavior change
   when switching between sibling functions)

   **Cat 5 — Public API surface drift**
   - HIGH: function is called by tests and notebooks but is not in
     `xrspatial/__init__.py` or in the module's `__all__` (orphan API)
   - HIGH: function in `__all__` but undocumented in the docstring
   - MEDIUM: deprecated alias still exported with no `DeprecationWarning`
   - MEDIUM: private-looking name (`_foo`) but is referenced in tests as
     if public
   - LOW: `from .module import *` patterns that bring inconsistent
     symbols into the public namespace
   Severity: HIGH for orphan APIs (users find them, depend on them, then
   break when they vanish)

3. For each real issue, assign severity + file:line.

4. If any CRITICAL, HIGH, or MEDIUM issue is found, run rockout to fix it.
   IMPORTANT: parameter renames are breaking changes — for HIGH
   parameter-rename fixes, the rockout PR must add a deprecation
   shim (accept both old and new names; emit DeprecationWarning on the
   old name; update docs). Document this in the issue body. For LOW
   issues, document but do not fix.

5. Update .kilo/worktrees/sweep-api-consistency-state.csv using csv.DictReader/Writer:

   ```python
   import csv
   from pathlib import Path

   path = Path(".kilo/worktrees/sweep-api-consistency-state.csv")
   header = ["module", "last_inspected", "issue", "severity_max",
             "categories_found", "notes"]

   rows = {}
   if path.exists():
       with path.open() as f:
           for r in csv.DictReader(f):
               rows[r["module"]] = r

   rows["{module}"] = {
       "module": "{module}",
       "last_inspected": "<today's ISO date>",
       "issue": "<issue number or empty>",
       "severity_max": "<HIGH|MEDIUM|LOW or empty>",
       "categories_found": "<semicolon-joined ints or empty>",
       "notes": "<single-line notes or empty>",
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

   Then `git add` and commit.

Important:
- Only flag real consistency issues. The lib has 40+ modules — do not
  list every minor naming difference; focus on user-facing surprise.
- Compare against 2-3 sibling modules. Cross-cutting concerns (e.g.
  cellsize naming convention) often span the whole library; if a rename
  is safe in one module but breaks 20 others, surface that as a notes
  comment, do not file a per-module issue.
- For the hydro subpackage: pick one variant (d8) and check whether
  dinf/mfd siblings agree.
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} API consistency audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

To reset: `sweep-api-consistency --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes.
- Keep the output concise.
- If {{ARGUMENTS}} is empty, use defaults: top 3, no category filter, no
  exclusions.
- State file (`.kilo/worktrees/sweep-api-consistency-state.csv`) is tracked in
  git with `merge=union`.
- Renames are breaking. The fix path is a deprecation shim, not a
  hard rename, unless the function has a clearly orphan/private status.
- False positives are worse than missed issues.
