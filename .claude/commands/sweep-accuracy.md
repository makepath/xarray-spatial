# Accuracy Sweep: Dispatch subagents to audit under-inspected modules

Analyze xrspatial modules by recency and inspection history, then launch
subagents to audit the highest-priority modules in parallel.

Optional arguments: $ARGUMENTS
(e.g. `--top 3`, `--exclude slope,aspect`, `--only-terrain`, `--reset-state`)

---

## Step 1 -- Gather module metadata via git

For every `.py` file directly under `xrspatial/` (skip `__init__.py`,
`_version.py`, `__main__.py`, `utils.py`, `accessor.py`, `preview.py`,
`dataset_support.py`, `diagnostics.py`, `analytics.py`), collect:

| Field | How |
|-------|-----|
| **last_modified** | `git log -1 --format=%aI -- xrspatial/<module>.py` |
| **first_commit** | `git log --diff-filter=A --format=%aI -- xrspatial/<module>.py` |
| **total_commits** | `git log --oneline -- xrspatial/<module>.py \| wc -l` |
| **recent_accuracy_commits** | `git log --oneline --grep='accuracy\|precision\|numerical\|geodesic' -- xrspatial/<module>.py` |

Store results in a temporary variable -- do NOT write intermediate files.

## Step 2 -- Load inspection state

Read the state file at `.claude/accuracy-sweep-state.json`.

If it does not exist, treat every module as never-inspected.

If `$ARGUMENTS` contains `--reset-state`, delete the file and treat
everything as never-inspected.

The state file schema:

```json
{
  "inspections": {
    "slope": { "last_inspected": "2026-03-28T14:00:00Z", "issue": 1042 },
    "aspect": { "last_inspected": "2026-03-28T15:30:00Z", "issue": 1043 }
  }
}
```

## Step 3 -- Score each module

Compute a priority score for each module. Higher = more urgent.

```
days_since_inspected = (today - last_inspected).days   # 9999 if never inspected
days_since_modified  = (today - last_modified).days
total_commits        = from Step 1
has_recent_accuracy_work = 1 if recent_accuracy_commits is non-empty, else 0

score = (days_since_inspected * 3)
      + (total_commits * 0.5)
      - (days_since_modified * 0.2)
      - (has_recent_accuracy_work * 500)
```

Rationale:
- Modules never inspected dominate (9999 * 3)
- More commits = more complex = more likely to have bugs
- Recently modified modules slightly deprioritized (someone just touched them)
- Modules with existing accuracy work heavily deprioritized

## Step 4 -- Apply filters from $ARGUMENTS

- `--top N` -- only include the top N modules (default: 3)
- `--exclude mod1,mod2` -- remove named modules from the list
- `--only-terrain` -- restrict to slope, aspect, curvature, terrain,
  terrain_metrics, hillshade, sky_view_factor
- `--only-focal` -- restrict to focal, convolution, morphology, bilateral,
  edge_detection, glcm
- `--only-hydro` -- restrict to flood, cost_distance, geodesic,
  surface_distance, viewshed, erosion, diffusion

## Step 5 -- Print the ranked table and launch subagents

### 5a. Print the ranked table

Print a markdown table showing ALL scored modules (not just the selected ones),
sorted by score descending:

```
| Rank | Module          | Score  | Last Inspected | Last Modified | Commits |
|------|-----------------|--------|----------------|---------------|---------|
| 1    | viewshed        | 30012  | never          | 45 days ago   | 23      |
| 2    | flood           | 29998  | never          | 120 days ago  | 18      |
| ...  | ...             | ...    | ...            | ...           | ...     |
```

### 5b. Launch subagents for the top N modules

For each of the top N modules (default 3), launch an Agent in parallel using
`isolation: "worktree"` and `mode: "auto"`. All N agents must be dispatched
in a single message so they run concurrently.

Each agent's prompt must be self-contained and follow this template (adapt
the module name, path, and metadata):

```
You are auditing xrspatial/{module}.py for numerical accuracy issues.

This module has {commits} commits and has never been inspected for accuracy.

**Your task:**

1. Read xrspatial/{module}.py thoroughly.

2. Identify any potential accuracy issues:
   - Floating point precision loss (e.g. catastrophic cancellation, summing
     small numbers into large accumulators, unnecessary float32 intermediates)
   - Incorrect NaN/nodata propagation (NaN inputs silently producing finite
     outputs, or valid inputs wrongly becoming NaN)
   - Off-by-one errors in neighborhood/window operations
   - Missing or wrong Earth curvature corrections (for geographic CRS ops)
   - Backend inconsistencies (numpy vs cupy vs dask producing different results
     for the same input)
   - Incorrect algorithm implementation vs. reference literature

3. For each real issue found, run /rockout to fix it end-to-end (creates a
   GitHub issue, worktree branch, fix, tests, and PR).

4. If you find NO accuracy issues after thorough review, report that the module
   is clean -- no action needed.

5. After finishing (whether you found issues or not), update the inspection
   state file .claude/accuracy-sweep-state.json by reading its current contents
   and adding/updating the entry for "{module}" with today's ISO date and the
   issue number (or null if no issue was created).

Important:
- Be rigorous. Only flag real bugs that produce wrong numerical results, not
  style issues or theoretical concerns.
- Read the tests in xrspatial/tests/ for this module to understand expected
  behavior before flagging issues.
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
After completion, the user can verify state with:

```
cat .claude/accuracy-sweep-state.json
```

To reset all tracking: `/sweep-accuracy --reset-state`

---

## General Rules

- Do NOT modify any source files directly. Subagents handle fixes via /rockout.
- Keep the output concise -- the table and agent dispatch are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 3, no category filter, no exclusions.
