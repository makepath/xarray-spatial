# Accuracy Sweep: Generate a Ralph Loop targeting under-inspected modules

Analyze xrspatial modules by recency and inspection history, then print a
ready-to-run `/ralph-loop` command that targets the highest-priority modules.

Optional arguments: $ARGUMENTS
(e.g. `--top 5`, `--exclude slope,aspect`, `--only-terrain`, `--reset-state`)

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

- `--top N` -- only include the top N modules (default: 5)
- `--exclude mod1,mod2` -- remove named modules from the list
- `--only-terrain` -- restrict to slope, aspect, curvature, terrain,
  terrain_metrics, hillshade, sky_view_factor
- `--only-focal` -- restrict to focal, convolution, morphology, bilateral,
  edge_detection, glcm
- `--only-hydro` -- restrict to flood, cost_distance, geodesic,
  surface_distance, viewshed, erosion, diffusion

## Step 5 -- Print the results

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

### 5b. Print the generated ralph-loop command

Using the top N modules from the ranked list, generate and print a command
like this (adapt the module list to actual results):

````
/ralph-loop "Survey xarray-spatial modules for numerical accuracy issues.

**Target these modules in priority order:**
1. viewshed (xrspatial/viewshed.py) -- never inspected, 23 commits
2. flood (xrspatial/flood.py) -- never inspected, 18 commits
3. focal (xrspatial/focal.py) -- never inspected, 31 commits
4. erosion (xrspatial/erosion.py) -- never inspected, 12 commits
5. classify (xrspatial/classify.py) -- never inspected, 9 commits

**For each module, in order:**
1. Read the source and identify potential accuracy issues:
   - Floating point precision loss
   - Incorrect NaN propagation
   - Off-by-one errors in neighborhood operations
   - Missing or wrong Earth curvature corrections
   - Backend inconsistencies (numpy vs cupy vs dask results differ)
2. Run /rockout to fix the issue end-to-end (issue, worktree, fix, tests, docs)
3. After completing rockout for ONE module, output <promise>ITERATION DONE</promise>

If you find no accuracy issues in the current target module, skip it and move
to the next one.

If all target modules have been addressed or have no issues, output
<promise>ALL ACCURACY ISSUES FIXED</promise>." --max-iterations {N} --completion-promise "ALL ACCURACY ISSUES FIXED"
````

Set `--max-iterations` to the number of target modules + 2 (buffer for retries).

### 5c. Print a reminder

```
To run this sweep:  copy the command above and paste it.
To update state after a manual rockout:  edit .claude/accuracy-sweep-state.json
To reset all tracking:  /accuracy-sweep --reset-state
```

## Step 6 -- Update state (ONLY when called from inside a ralph-loop)

This step is informational. The accuracy-sweep command itself does NOT update
the state file. State is updated when `/rockout` completes -- the rockout
workflow should append to `.claude/accuracy-sweep-state.json` after creating
the issue.

To enable this, print a note reminding the user that after each rockout
iteration completes, they can manually record the inspection:

```json
// Add to .claude/accuracy-sweep-state.json after each rockout:
{ "module_name": { "last_inspected": "ISO-DATE", "issue": ISSUE_NUMBER } }
```

---

## General Rules

- Do NOT modify any source files. This command is read-only analysis.
- Do NOT create GitHub issues. This command only generates the ralph-loop command.
- Keep the output concise -- the table and command are the deliverables.
- If $ARGUMENTS is empty, use defaults: top 5, no category filter, no exclusions.
