# Style Sweep: Dispatch subagents to audit modules for PEP8 and coding-style issues

Audit xrspatial modules for Python style issues that the project's own
tooling already knows how to detect: PEP8 violations (flake8 E/W codes),
unused imports and dead locals (flake8 F codes), import-ordering drift
(isort), and bug-prone style anti-patterns (bare except, mutable defaults,
shadowed builtins). The project configures flake8 (`max-line-length=100`)
and isort (`line_length=100`) in `setup.cfg` but does not gate them in CI,
so drift is invisible. Subagents fix HIGH and MEDIUM findings via /rockout;
LOW findings are recorded but not auto-fixed to avoid nitpick PRs.

Optional arguments: $ARGUMENTS
(e.g. `--top 3`, `--exclude slope,aspect`, `--only-terrain`, `--reset-state`)

---

## Step 1 -- Gather module metadata via git, grep, and flake8

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
| **public_funcs** | count of functions at module level (heuristic: `^def [a-z]`) |
| **flake8_baseline** | `flake8 <module_files> 2>&1 \| wc -l` — observed lint count using the existing `setup.cfg` `[flake8]` config |

Store results in memory -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read `.claude/sweep-style-state.csv`.

If it does not exist, treat every module as never-inspected.

If `$ARGUMENTS` contains `--reset-state`, delete the file and treat
everything as never-inspected.

State file schema (one row per module):

```
module,last_inspected,issue,severity_max,categories_found,notes
slope,2026-05-01,1042,MEDIUM,1;4,"optional single-line notes"
```

- `categories_found` is a semicolon-separated integer list (empty when null).
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
      + (flake8_baseline * 25)
      + (loc * 0.05)
      + (total_commits * 0.2)
      - (days_since_modified * 0.1)
```

Rationale:
- Never-inspected modules dominate (9999 * 3)
- `flake8_baseline` is the measured truth — observed lint count, not a
  proxy. A module with 40 existing violations should outrank a clean
  module of similar size.
- Larger files have more surface area (0.05 per line)
- Churn correlates with style drift across many small commits (0.2)
- Recently modified modules slightly deprioritized to avoid stomping on
  in-flight work

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
- `--reset-state` -- delete the state file before scoring

## Step 5 -- Print the ranked table and launch subagents

### 5a. Print the ranked table

Print a markdown table showing ALL scored modules (not just selected ones),
sorted by score descending:

```
| Rank | Module          | Score  | Last Inspected | flake8 | LOC  | Commits |
|------|-----------------|--------|----------------|--------|------|---------|
| 1    | geotiff         | 31050  | never          | 42     | 1400 | 85      |
| 2    | hydro           | 30900  | never          | 28     | 8200 | 64      |
| ...  | ...             | ...    | ...            | ...    | ...  | ...     |
```

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, paths, and metadata):

```
You are auditing the xrspatial module "{module}" for Python style issues.

This module has {commits} commits, {loc} lines of code, and an observed
flake8 baseline of {flake8_baseline} violations.

Read these files: {module_files}

Also read setup.cfg to confirm the project's flake8 and isort config
(max-line-length=100, line_length=100, exclude .git/.asv/__pycache__).

**Your task:**

1. Run the project's own style tooling against the module files:

   ```
   flake8 {module_files}
   isort --check-only --diff {module_files}
   ```

   These tools are authoritative — every issue they report is in scope.

2. Classify each reported issue into one of these 5 categories. Only flag
   issues ACTUALLY reported by the tools or grep — do not invent style
   nitpicks the linters do not flag.

   **Cat 1 — flake8 E-codes (PEP8 errors)**
   - E1xx indentation, E2xx whitespace, E3xx blank lines, E5xx line length,
     E7xx statement-level (e.g. E711 comparison to None, E712 to True/False,
     E721 type comparison, E741 ambiguous name)
   Severity: MEDIUM (real PEP8 violations against the configured style)

   **Cat 2 — flake8 W-codes (PEP8 warnings)**
   - W191 indentation contains tabs, W291/W293 trailing whitespace, W391
     blank line at end of file, W605 invalid escape sequence
   Severity: LOW unless W605 (invalid escape — can mask intent), in which
   case bump to MEDIUM and add to Cat 5 as well

   **Cat 3 — flake8 F-codes (pyflakes: bug-masking lint)**
   - F401 unused import, F811 redefinition of unused name, F821 undefined
     name, F841 local assigned but unused, F823 local used before assignment
   Severity: HIGH — these frequently hide refactor leftovers and real
   bugs (F821 is always HIGH; F401 on a module shipped to users can mean
   a removed re-export)

   **Cat 4 — Import ordering (isort)**
   - Any diff produced by `isort --check-only --diff` against the
     configured `line_length=100`
   Severity: MEDIUM

   **Cat 5 — Bug-prone style anti-patterns**
   Grep for and review:
   - Bare `except:` (without an exception type) — `grep -nE '^\s*except\s*:' <files>`
   - Mutable default args — `grep -nE 'def [^(]+\([^)]*=\s*(\[|\{)' <files>`
   - `== None`, `!= None`, `== True`, `== False` — already caught by flake8
     E711/E712 but list separately here so the rockout PR addresses them
     together as a behavioural class
   - Shadowing builtins as variable or parameter names: `list`, `dict`,
     `set`, `id`, `type`, `input`, `filter`, `map`, `next`, `iter`
   Severity: HIGH — these are the only style findings that change runtime
   behaviour (bare except swallows KeyboardInterrupt; mutable defaults
   are shared across calls; shadowed builtins corrupt the namespace).

3. For each real issue found, assign a severity (HIGH/MEDIUM/LOW) and note
   the exact file and line number. Group same-category issues into a single
   finding when they're trivially related (e.g. 12 trailing-whitespace
   lines = one Cat 2 finding, not twelve).

4. If any HIGH or MEDIUM issue is found, run /rockout to fix it end-to-end
   (GitHub issue, worktree branch, fix, tests, and PR). One /rockout per
   module — the PR should bundle all HIGH+MEDIUM findings for that module
   into a single coherent style cleanup.

   For LOW findings (W-codes, single-line E501 on a long URL, cosmetic
   E2xx that don't reduce readability), document them in the state CSV
   notes column but do NOT open a PR. Per-line nitpick PRs are net
   negative.

   The /rockout PR description should:
   - List which categories were addressed (e.g. "Cat 3 (F401, F841), Cat 4
     (isort), Cat 5 (bare except)")
   - Confirm no behavioural change is intended for Cat 1/2/4 fixes
   - Call out any Cat 3/5 fix that does change behaviour (e.g. removing
     an unused import that was actually re-exporting a symbol)

5. After finishing (whether you found issues or not), update the inspection
   state file `.claude/sweep-style-state.csv`. The file is row-per-module
   CSV with header:

   `module,last_inspected,issue,severity_max,categories_found,notes`

   Use this Python pattern to read, update, and write it (do NOT hand-edit
   the file -- always go through csv.DictReader / csv.DictWriter so quoting
   stays consistent):

   ```python
   import csv
   from pathlib import Path

   path = Path(".claude/sweep-style-state.csv")
   header = ["module", "last_inspected", "issue", "severity_max",
             "categories_found", "notes"]

   rows = {}
   if path.exists():
       with path.open() as f:
           for r in csv.DictReader(f):
               rows[r["module"]] = r  # last write wins on dupes

   rows["{module}"] = {
       "module": "{module}",
       "last_inspected": "<today's ISO date, e.g. 2026-05-21>",
       "issue": "<issue number from rockout, or empty string>",
       "severity_max": "<HIGH|MEDIUM|LOW, or empty>",
       "categories_found": "<semicolon-joined ints, e.g. 1;4, or empty>",
       "notes": "<single-line notes (replace any newlines with spaces), or empty>",
   }

   with path.open("w", newline="") as f:
       w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
       w.writeheader()
       for m in sorted(rows):
           w.writerow(rows[m])
   ```

   Use empty strings (not `null`) for missing values. Set `issue` to the
   issue number when one was filed, otherwise leave it empty.

   Then `git add .claude/sweep-style-state.csv` and commit it to the
   worktree branch so the state update is included in the PR.

Important:
- Only flag issues the tools actually report (flake8, isort) or that grep
  confirms for Cat 5. Style is subjective; the project has already drawn
  the line at the configured `setup.cfg` settings.
- Do NOT run black, ruff format, autopep8, or any other auto-formatter.
  The project has not adopted a formatter and choosing one is a policy
  decision, not a sweep finding. Limit fixes to what flake8 + isort + the
  Cat 5 grep flag.
- Do NOT widen the flake8 config to silence findings. If a finding is a
  false positive (e.g. E501 on a URL where wrapping hurts readability),
  add a per-line `# noqa: E501` rather than changing the global config.
- For the hydro subpackage: run flake8 + isort across all `.py` files in
  the subpackage and treat them as one audit unit. Issues in dinf/mfd
  variants that mirror d8 should be fixed together in the same /rockout PR.
- This repo uses ArrayTypeFunctionMapping to dispatch across numpy/cupy/dask
  backends. Style fixes are static and apply uniformly across backend
  paths — no separate backend verification is needed (unlike security or
  accuracy sweeps).
```

### 5c. Print a status line

After dispatching, print:

```
Launched {N} style audit agents: {module1}, {module2}, {module3}
```

## Step 6 -- State updates

State is updated by the subagents themselves (see agent prompt step 5).
After completion, verify state with:

```
column -t -s, .claude/sweep-style-state.csv | less
```

To reset all tracking: `/sweep-style --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via /rockout.
- Keep the output concise -- the table and agent dispatch are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no exclusions.
- State file (`.claude/sweep-style-state.csv`) is tracked in git and uses
  git's default 3-way text merge (no `merge=union`; see issue #2754), so a
  concurrent change surfaces a conflict instead of silently unioning
  duplicate rows. Subagents must `git add` and commit it so the state
  update lands in the PR.
- For subpackage modules (geotiff, reproject, hydro), the subagent should run
  flake8 + isort across ALL `.py` files in the subpackage directory, not
  just `__init__.py`.
- Only flag what the tools and grep actually report. Style is configured by
  `setup.cfg`; the sweep's job is enforcement, not policy.
- False positives are worse than missed issues. When a flake8 finding is a
  legitimate exception (long URL, generated lookup table), the fix is a
  `# noqa` on that line — not a config widening, not a silent suppression.
