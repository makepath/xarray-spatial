# Test Coverage Gap Sweep: Dispatch subagents to audit backend and edge-case test coverage

Audit xrspatial modules for test coverage gaps: missing backend coverage
(numpy / cupy / dask+numpy / dask+cupy), missing edge cases (NaN, Inf,
empty input, single-pixel, all-equal input), missing parameter-coverage
tests. Closes the gaps that the accuracy sweep keeps finding bugs in.
Subagents fix CRITICAL, HIGH, and MEDIUM findings via rockout — fixes
here are *adding tests*, not changing source code.

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
each subagent prompt below so the agent knows whether new tests can be
executed against cupy / dask+cupy backends or only added with a `pytest.skip`
guard for environments without CUDA.

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
| **test_loc** | `wc -l < xrspatial/tests/test_<module>.py` (or 0 if absent) |
| **public_funcs** | count of `^def [a-z]` in module |

Store results in memory.

## Step 2 -- Load inspection state

Read `.kilo/worktrees/sweep-test-coverage-state.csv`.

If absent, treat every module as never-inspected. If `{{ARGUMENTS}}` has
`--reset-state`, delete the file first.

State file schema:

```
module,last_inspected,issue,severity_max,categories_found,notes
slope,2026-05-01,1042,HIGH,1;3,"optional single-line notes"
```

`merge=union` is set in `.gitattributes`.

## Step 3 -- Score each module

```
days_since_inspected = (today - last_inspected).days
days_since_modified  = (today - last_modified).days

# Coverage ratio: low test_loc relative to source = higher score
coverage_deficit = max(0, loc - test_loc) / max(loc, 1)

score = (days_since_inspected * 3)
      + (public_funcs * 5)
      + (coverage_deficit * 200)
      + (total_commits * 0.3)
      - (days_since_modified * 0.1)
      + (loc * 0.03)
```

Rationale:
- Modules never inspected dominate
- Coverage deficit (test_loc << source_loc) is a strong signal
- Public functions weighted: each public function is an independent
  test surface
- Recently modified slightly deprioritized

## Step 4 -- Apply filters from {{ARGUMENTS}}

Same filter set as other sweeps: `--top N`, `--exclude`, `--only-terrain`,
`--only-focal`, `--only-hydro`, `--only-io`, `--reset-state`.

## Step 5 -- Print the ranked table and launch subagents

### 5a. Print the ranked table

Show all scored modules sorted by score descending. Include a `Coverage`
column (`test_loc / source_loc` ratio).

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel
using `isolation: "worktree"` and `mode: "auto"`. All N must be in a
single message.

Each agent's prompt must be self-contained:

```
You are auditing the xrspatial module "{module}" for test coverage gaps.

This module has {commits} commits, {loc} lines of source, and {test_loc}
lines of tests.

Read these files:
- {module_files}
- xrspatial/tests/test_{module}.py (if it exists)
- xrspatial/tests/general_checks.py (cross-backend test helpers)
- xrspatial/utils.py (ArrayTypeFunctionMapping, _validate_raster)
- xrspatial/conftest.py (shared fixtures)

CUDA available on this host: {cuda_available}

If CUDA_AVAILABLE is true:
- New cupy / dask+cupy tests must execute locally before rockout opens
  a PR. Use the cross-backend helpers in general_checks.py so the new
  test exercises all four backends on a CUDA host.
- Verify the test actually fails before the fix and passes after — do
  not commit a test that was never observed running on a GPU.

If CUDA_AVAILABLE is false:
- New cupy / dask+cupy tests are still added (CI runs them on a GPU
  host) but must be guarded with the project's existing GPU-skip
  decorator so local runs without CUDA do not error. Note that the
  test was not executed locally.
- Add the token `cuda-unavailable` to the `notes` column of the state
  CSV so a future re-run on a GPU host knows to re-validate that the
  newly added cupy tests pass.

**Your task:**

1. Read the module and its tests thoroughly. Build a mental matrix:
   for each public function, which backends and which edge cases are
   currently tested?

2. Audit for these 5 coverage-gap categories. Only flag gaps ACTUALLY
   present (the test file does not exercise the path).

   **Cat 1 — Backend coverage**
   - HIGH: function has a numpy path that is tested, but the cupy /
     dask+numpy / dask+cupy paths are not exercised at all
   - HIGH: dispatch table (ArrayTypeFunctionMapping) registers a backend
     but no test invokes it
   - MEDIUM: cross-backend equivalence not asserted (test_numpy_equals_cupy,
     test_numpy_equals_dask, test_numpy_equals_dask_cupy missing)
   - MEDIUM: only the eager path tested with realistic input shapes; the
     dask path tested only on a 4x4 toy
   Severity: HIGH if a real bug could ship undetected (the GLCM bug
   #1408 was caught precisely because backend coverage existed)

   **Cat 2 — NaN / Inf / nodata edge cases**
   - HIGH: function operates on raster data but no test passes a NaN
     input
   - HIGH: NaN appears in tests only as a non-edge cell, never at the
     boundary or in a position that interacts with the kernel
   - HIGH: Inf / -Inf inputs not tested at all (often surfaces silent
     failure modes)
   - MEDIUM: all-NaN input not tested (boundary of the algorithm)
   - MEDIUM: NaN input dtype is float; but integer dtype with the
     module's documented sentinel is not tested
   Severity: HIGH if NaN-related bugs in this module class have shipped
   before (see flood, glcm, sky_view_factor) — they have

   **Cat 3 — Geometric edge cases**
   - HIGH: 1x1 single-pixel raster not tested
   - HIGH: Nx1 or 1xN strip not tested (kernel boundary degeneracies)
   - MEDIUM: empty raster (0 rows or 0 cols) not tested
   - MEDIUM: all-equal-value raster not tested (zero variance, zero
     gradient → divide-by-zero opportunity)
   - MEDIUM: very large raster not benchmarked (no asv coverage)
   - LOW: raster with non-square cells (different cellsize_x and
     cellsize_y) not tested
   Severity: HIGH for 1x1 / Nx1 — these reveal kernel-bound bugs

   **Cat 4 — Parameter coverage**
   - HIGH: a parameter with multiple modes (e.g. `boundary='reflect'`,
     `'edge'`, `'wrap'`, `'nan'`) has only the default mode tested
   - HIGH: a `bool` flag has only one branch tested
   - MEDIUM: a numeric parameter has only one value tested (e.g.
     `kernel_size` only tested at 3, never at 5 or 7)
   - MEDIUM: error paths not tested (does invalid input raise the
     expected exception?)
   - LOW: kwargs documented in docstring but no test passes them
   Severity: HIGH if the untested mode is what advanced users rely on

   **Cat 5 — Metadata preservation tests**
   - HIGH: no test asserts that input attrs (`res`, `crs`, `transform`)
     are preserved in the output (this is the metadata-propagation
     sweep's smoke detector)
   - HIGH: no test asserts that input coords are preserved
   - MEDIUM: no test asserts that input dim names propagate (function
     would silently rename `lat`/`lon` → `y`/`x`)
   - MEDIUM: no test for the eager-vs-dask attrs equivalence
   Severity: HIGH if this module reads attrs for math (cellsize,
   resolution) — its result correctness depends on these being correct

3. For each real gap, assign severity + which test should be added.

4. If any CRITICAL, HIGH, or MEDIUM gap is found, run rockout to add
   tests. The fix in this sweep is *test-only* — do not modify source
   unless a test surfaces a bug, in which case file a separate accuracy
   issue. For LOW gaps, document but do not add tests.

5. Update .kilo/worktrees/sweep-test-coverage-state.csv:

   ```python
   import csv
   from pathlib import Path

   path = Path(".kilo/worktrees/sweep-test-coverage-state.csv")
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
       "issue": "<issue or empty>",
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
- The "fix" for this sweep is *adding tests*. If adding a test surfaces
  a bug in the source code, do NOT bundle the source fix — file a
  separate accuracy / performance / metadata issue and link it from the
  test PR.
- Only flag real gaps. If a test exists but is sloppy, that is not a
  coverage gap — that's a test quality issue out of scope here.
- Some functions genuinely do not need NaN coverage (procedural noise
  generators that take no raster input). Use judgment.
- For the hydro subpackage: focus on one representative variant (d8) and
  note dinf/mfd parity in the audit notes.
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} test coverage audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

To reset: `sweep-test-coverage --reset-state`

---

## General Rules

- Do NOT modify any source files. Subagents add tests via rockout.
- Keep parent output concise.
- Default: top 3, no filter.
- State file `.kilo/worktrees/sweep-test-coverage-state.csv` is tracked in git
  with `merge=union`.
- The "fix" is *tests, not source*. If a test reveals a bug, file a
  separate issue — do not change source in this sweep's PRs.
- False positives are worse than missed issues.
