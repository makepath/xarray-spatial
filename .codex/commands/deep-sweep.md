# Deep Sweep: Run every sweep-* command focused on a single module

Pick one xrspatial module and dispatch every `/sweep-*` command at it in
parallel. Each sub-sweep follows the audit template embedded in its own
`.codex/commands/sweep-*.md` file, runs `/rockout` for HIGH/MEDIUM findings
when the sweep specifies it, and updates its own
`.codex/sweep-{type}-state.csv` row for the target module.

New sweeps are picked up automatically. Drop a
`.codex/commands/sweep-XYZ.md` into the commands directory and the next
`/deep-sweep` run will dispatch it alongside the others.

Required first argument: the module name (e.g. `geotiff`, `slope`, `hydro`).
Optional flags: $ARGUMENTS
(e.g. `geotiff --only-sweep security,performance`,
`viewshed --exclude-sweep test-coverage`,
`slope --no-fix`,
`reproject --reset-state`)

---

## Step 0 -- Parse arguments and snapshot main-checkout state

The first positional token in `$ARGUMENTS` is the module name. It is
required. If `$ARGUMENTS` is empty or starts with a flag, stop and ask the
user which module to deep-sweep.

Capture the main checkout's branch as `DEEP_SWEEP_START_BRANCH` so Step
5.5 can verify the sweeps left it untouched:

```bash
DEEP_SWEEP_START_BRANCH="$(git -C $(git rev-parse --show-toplevel) branch --show-current)"
```

If the main checkout has uncommitted changes when /deep-sweep starts,
note them. Step 5.5 will diff against this snapshot, not the empty
state, so existing dirtiness is not mistaken for a sweep breach.

Then parse flags (multiple may combine):

| Flag | Effect |
|------|--------|
| `--only-sweep s1,s2` | Only dispatch the named sweeps. Names are the suffix after `sweep-` (e.g. `security`, `performance`, `api-consistency`). |
| `--exclude-sweep s1,s2` | Skip the named sweeps. |
| `--no-fix` | Pass `--no-fix` semantics to every dispatched sweep: subagent audits only, no `/rockout`, no PR. State CSV is still updated. |
| `--reset-state` | Before dispatching, delete the target module's row from every `.codex/sweep-*-state.csv` so the audit is treated as never-inspected. Do NOT delete other modules' rows. |

## Step 1 -- Validate the module

Determine the module's files under `xrspatial/`:

- If `xrspatial/{module}.py` exists, the module is a single file at that path.
- Else if `xrspatial/{module}/` is a directory, the module is a subpackage.
  List all `.py` files under it (excluding `__init__.py`).
- Otherwise, stop and report that `{module}` was not found, listing the
  available top-level `.py` files and subpackage directories under
  `xrspatial/` so the user can correct the name.

Skip names that the individual sweeps already exclude from their discovery:
`__init__`, `_version`, `__main__`, `utils`, `accessor`, `preview`,
`dataset_support`, `diagnostics`, `analytics`. If the user passes one of
these, stop and explain that these modules are not in scope for the
per-module sweeps.

## Step 2 -- Discover sweep commands

List all files matching `.codex/commands/sweep-*.md`. For each, the sweep
name is the basename without `sweep-` prefix and `.md` suffix
(e.g. `.codex/commands/sweep-security.md` → `security`). Build the list
in sorted order so the dispatch table is deterministic.

Apply `--only-sweep` / `--exclude-sweep` filters. If the resulting list is
empty, stop and report which filters eliminated everything.

For each remaining sweep, record:
- `sweep_name` (e.g. `security`)
- `sweep_file` (path to the `.md`)
- `state_file` (`.codex/sweep-{sweep_name}-state.csv`)

## Step 3 -- Gather shared module metadata

Collect once and pass to every subagent (each sweep file lists the metadata
it needs; the union below covers all current sweeps):

| Field | How |
|-------|-----|
| **module_files** | from Step 1 |
| **last_modified** | `git log -1 --format=%aI -- <path>` (for subpackages, most recent file) |
| **total_commits** | `git log --oneline -- <path> \| wc -l` |
| **loc** | `wc -l < <path>` (for subpackages, sum all files) |
| **has_cuda_kernels** | grep file(s) for `@cuda.jit` |
| **has_file_io** | grep file(s) for `open(`, `mkstemp`, `os.path`, `pathlib` |
| **has_numba_jit** | grep file(s) for `@ngjit`, `@njit`, `@jit`, `numba.jit` |
| **allocates_from_dims** | grep file(s) for `np.empty(height`, `np.zeros(height`, `np.empty(H`, `cp.empty(`, and width variants |
| **has_shared_memory** | grep file(s) for `cuda.shared.array` |
| **has_dask_backend** | grep file(s) for `_run_dask`, `map_overlap`, `map_blocks` |
| **has_cuda_backend** | grep file(s) for `@cuda.jit`, `import cupy` |

Also detect CUDA availability once:

```bash
python -c "from numba import cuda; print(cuda.is_available())" 2>/dev/null
```

Capture as `CUDA_AVAILABLE` (`true` / `false`).

## Step 4 -- Handle `--reset-state`

If `--reset-state` was passed, for each state file in scope:

```python
import csv
from pathlib import Path

path = Path("{state_file}")
if not path.exists():
    continue
with path.open() as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    rows = [r for r in reader if r["module"] != "{module}"]
def _oneline(v):
    # merge=union is line-based: a newline inside a quoted field splits
    # the record on parallel-agent merges. Force one physical line per
    # record by collapsing embedded newlines to " | ".
    return "" if v is None else str(v).replace("\r\n", " | ").replace("\r", " | ").replace("\n", " | ")

with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({k: _oneline(v) for k, v in r.items()})
```

This removes only the target module's row from each state file, leaving
other modules' history intact. Do this before dispatching the subagents so
they each see a clean slate for this module.

## Step 5 -- Dispatch one subagent per sweep, in parallel

Print a short dispatch table:

```
Deep-sweeping module "{module}" across {N} sweeps:
  - security       → .codex/sweep-security-state.csv
  - performance    → .codex/sweep-performance-state.csv
  - accuracy       → .codex/sweep-accuracy-state.csv
  ...
```

Then in a **single message**, launch one Agent per sweep with
`isolation: "worktree"` and `mode: "auto"` so they run concurrently in
separate worktrees. Use the prompt template below for every agent,
substituting `{sweep_name}`, `{sweep_file}`, `{state_file}`, `{module}`,
`{module_files}`, `{loc}`, `{commits}`, `{cuda_available}`, `{today}`, and
the boolean metadata flags. The `{today}` value is critical: it's woven
into the deterministic branch name `deep-sweep-{sweep_name}-{module}-{today}`
that each sibling rebases its worktree onto, and the parent later checks
those names for uniqueness.

### Subagent prompt template

```
You are running ONE specific sweep -- "{sweep_name}" -- against a single
xrspatial module: "{module}".

The parent command (/deep-sweep) has already chosen this module and is
dispatching every sweep against it in parallel. Your job is to behave
exactly as the embedded subagent prompt in
.codex/commands/sweep-{sweep_name}.md would, but skip module discovery
and scoring -- the module is already chosen.

## WORKTREE ISOLATION CONTRACT (read first, enforce throughout)

You were dispatched with `isolation: "worktree"`. That means a dedicated
git worktree was created for you, and your CWD at launch IS that
worktree directory. Several parallel siblings are running the other
sweeps against the same module right now. If you operate outside your
worktree, you will collide with them and your commits will land on the
wrong branch.

**Step ISO-1 (run BEFORE anything else, before reading any sweep file):**

```bash
DEEP_SWEEP_WT="$(pwd)"
DEEP_SWEEP_TOP="$(git rev-parse --show-toplevel)"
DEEP_SWEEP_BRANCH="$(git branch --show-current)"
echo "wt=$DEEP_SWEEP_WT top=$DEEP_SWEEP_TOP branch=$DEEP_SWEEP_BRANCH"
```

Assert ALL of the following. If any fails, STOP immediately, do NOT
make any commits, and report exactly `WORKTREE_ISOLATION_FAILED:
<reason>` back to the parent:

- `$DEEP_SWEEP_WT` equals `$DEEP_SWEEP_TOP` (you are at the worktree
  root, not in a subdirectory of some other checkout).
- `$DEEP_SWEEP_TOP` contains the segment `.codex/worktrees/agent-`
  (you are inside an isolated worktree, not the user's main checkout).
- `$DEEP_SWEEP_BRANCH` is NOT `main` and NOT `master`.
- `$DEEP_SWEEP_BRANCH` does NOT already match a branch created by
  another deep-sweep sibling. Specifically, reject branches matching
  `deep-sweep-*-{module}-*` whose `{sweep_name}` segment is NOT
  "{sweep_name}". (If you find yourself on a sibling's branch, the
  Agent harness has handed you the wrong worktree -- bail out.)

**Step ISO-2 (immediately after ISO-1, before any audit work):**

Rename your branch to a deterministic, sweep-specific name so /rockout
calls and state-CSV commits cannot collide with siblings:

```bash
DEEP_SWEEP_TARGET_BRANCH="deep-sweep-{sweep_name}-{module}-{today}"
if [ "$DEEP_SWEEP_BRANCH" != "$DEEP_SWEEP_TARGET_BRANCH" ]; then
  git branch -m "$DEEP_SWEEP_TARGET_BRANCH"
  DEEP_SWEEP_BRANCH="$DEEP_SWEEP_TARGET_BRANCH"
fi
```

From this point on, every git operation (add, commit, push,
checkout, rebase) MUST be executed from `$DEEP_SWEEP_WT`. Do NOT use
absolute paths into the user's main checkout. Do NOT `cd` away from
`$DEEP_SWEEP_WT`. If a tool resolves an absolute path back to the
main checkout (e.g. `/home/.../xarray-spatial-contrib/...`), pass the
worktree-relative path instead.

**Step ISO-3 (before EVERY commit you make, parent or /rockout-driven):**

Re-check that you are still on the right branch in the right
directory. /rockout in particular may switch branches; if so, it
must do so from within `$DEEP_SWEEP_WT` and the new branch name
must start with `deep-sweep-{sweep_name}-{module}-` (use
`--branch-prefix` or equivalent if /rockout exposes one; otherwise
create your /rockout branches manually from
`$DEEP_SWEEP_TARGET_BRANCH` rather than letting /rockout pick a
plain `issue-NNNN` name that could collide):

```bash
[ "$(pwd)" = "$DEEP_SWEEP_WT" ] || { echo "CWD drift"; exit 1; }
case "$(git branch --show-current)" in
  deep-sweep-{sweep_name}-{module}-*) : ;;
  *) echo "branch drift: $(git branch --show-current)"; exit 1 ;;
esac
```

A failed re-check is an isolation breach. Stop, do not commit, and
report back.

**Step ISO-4 (when filing PRs):**

If /rockout produces one or more PRs, every PR must be pushed from a
branch matching `deep-sweep-{sweep_name}-{module}-*`. Do NOT push to
`main`. Do NOT push to a sibling's branch name. If the sweep template
mandates one PR per finding (e.g. security: one fix per PR), use
suffixes like `deep-sweep-{sweep_name}-{module}-{today}-01`,
`-02`, etc., all branched off `$DEEP_SWEEP_TARGET_BRANCH`.

## Bootstrapping steps (after ISO-1 / ISO-2 pass)

1. Read the sweep definition: {sweep_file}

   Inside it, locate the "subagent prompt template" (a fenced block under
   a heading like "Step 5b" or "Step 3b" titled "Launch subagents"). That
   block is what an individual sweep dispatches to its own audit workers.
   You are going to act as that worker for module "{module}".

2. Pre-collected metadata for "{module}":

   - module_files       : {module_files}
   - loc                : {loc}
   - total_commits      : {commits}
   - last_modified      : {last_modified}
   - has_cuda_kernels   : {has_cuda_kernels}
   - has_file_io        : {has_file_io}
   - has_numba_jit      : {has_numba_jit}
   - allocates_from_dims: {allocates_from_dims}
   - has_shared_memory  : {has_shared_memory}
   - has_dask_backend   : {has_dask_backend}
   - has_cuda_backend   : {has_cuda_backend}
   - CUDA_AVAILABLE     : {cuda_available}

   Use only the fields the sweep's template actually references. Ignore
   ones it does not mention.

3. Follow the sweep's embedded subagent prompt verbatim against this
   module. That means:

   - Read every file the template tells you to read (module files, utils,
     tests, general_checks.py, etc.).
   - Run every audit category the template lists. Only flag issues
     ACTUALLY present in the code -- false positives are worse than
     missed issues.
   - If the template instructs the worker to run /rockout for
     HIGH/MEDIUM findings, do so {fix_mode_note}, observing the
     worktree-isolation contract above (ISO-3 / ISO-4).
   - Update the sweep's state CSV ({state_file}) using the read-update-
     write Python pattern the template specifies. Key by module name;
     last write wins on duplicates. Use today's ISO date
     ({today}) for last_inspected. Use empty strings (not "null") for
     missing fields.
   - `git add {state_file}` and commit it on YOUR worktree branch
     (`$DEEP_SWEEP_TARGET_BRANCH`) so the state update lands in any
     resulting PR. Run ISO-3's re-check immediately before the commit.
     If you did not file a PR, still commit the state update on the
     worktree branch -- the parent will surface the branch path in its
     summary.

4. The sweep file may have its own CUDA-availability conditional (run
   GPU paths vs. static review only). Honour it using CUDA_AVAILABLE
   above. If CUDA is unavailable and the sweep specifies adding a
   "cuda-unavailable" token to notes, do so.

**Hard rules (override any conflicting hint in the template):**

- Operate ONLY on module "{module}". Do not score, rank, or audit any
  other module. Do not re-discover the module list.
- Do not modify other modules' rows in {state_file}. Only your own
  module's row is touched.
- Do not call `.compute()` in any dask graph-construction probe.
- If the sweep template would normally launch its own sub-subagents,
  do NOT recurse -- you ARE the worker. Inline the work it would
  delegate.
- All commits and pushes happen from `$DEEP_SWEEP_WT` on a branch
  starting with `deep-sweep-{sweep_name}-{module}-`. Never on `main`,
  never in the user's main checkout, never on a sibling sweep's branch.
- {fix_mode_rule}

**Final report (mandatory):**

When you finish, report a short summary including, in addition to the
audit content, an isolation footer with the literal values of
`$DEEP_SWEEP_WT`, `$DEEP_SWEEP_TARGET_BRANCH`, and the SHA of the
state-CSV commit. The parent uses these to verify the contract held:

```
Findings: <N CRITICAL>, <N HIGH>, <N MEDIUM>, <N LOW>
/rockout: <not-run | PRs: #NNNN, #NNNN>
Isolation:
  worktree: <$DEEP_SWEEP_WT>
  branch:   <$DEEP_SWEEP_TARGET_BRANCH>
  state-commit: <SHA>
```
```

Where `{fix_mode_note}` and `{fix_mode_rule}` are:

- If `--no-fix` was NOT passed:
  - `{fix_mode_note}` = `end-to-end (GitHub issue, worktree branch, fix, tests, PR)`
  - `{fix_mode_rule}` = `Run /rockout for HIGH/MEDIUM/CRITICAL findings as the sweep template specifies. LOW findings: document, do not fix.`
- If `--no-fix` WAS passed:
  - `{fix_mode_note}` = `-- skipped, --no-fix is set`
  - `{fix_mode_rule}` = `Do NOT run /rockout. Document findings in the state CSV's notes field and your summary. This run is audit-only.`

And `{today}` is the current date in ISO 8601 (use the `currentDate`
context value if available; otherwise `date +%Y-%m-%d`).

## Step 5.5 -- Verify the worktree-isolation contract held

Before printing the user-facing results table, parse each agent's
returned summary for its "Isolation" footer (worktree path, branch
name, state-commit SHA). Then verify:

1. **No `WORKTREE_ISOLATION_FAILED` markers.** If any agent returned
   that token, mark its row `ISOLATION FAILED` in the results table
   and surface the agent's full final message verbatim. Do not treat
   its findings as merged-ready.
2. **Branch uniqueness.** Every agent must be on a distinct branch.
   Expected pattern: `deep-sweep-{sweep_name}-{module}-{today}`
   (with optional `-NN` suffix for /rockout fan-out). Reject any
   duplicates and any branch equal to `main` / `master`.
3. **Worktree distinctness.** Every agent's reported worktree path
   must be unique and must contain `.codex/worktrees/agent-`.
4. **Main checkout untouched.** Run:

   ```bash
   git -C $(git rev-parse --show-toplevel) rev-parse --abbrev-ref HEAD
   git -C $(git rev-parse --show-toplevel) status --porcelain
   ```

   The main checkout's HEAD branch must be unchanged from what it was
   before /deep-sweep started (capture it in Step 0 as
   `DEEP_SWEEP_START_BRANCH`). The porcelain output should contain no
   commits or modifications introduced by sweep agents (a still-untracked
   `.codex/commands/*.md` from the current session is fine; new commits
   on the current branch from a sweep agent are NOT).

If any of (1)-(4) fails, print a clearly-labeled
`### Isolation contract breached` section ABOVE the results table,
listing every breach and which agent caused it, so the user can decide
whether to keep the produced PRs or unwind them. Do not silently
proceed.

## Step 6 -- Wait, collect, and print the summary

All Agent calls run in the foreground in parallel. Once they return, print
a single results table:

```
| Sweep           | Findings        | /rockout PR | State row written |
|-----------------|-----------------|-------------|-------------------|
| security        | 0 HIGH, 1 MED   | #1567       | yes               |
| performance     | 2 HIGH          | #1568       | yes               |
| accuracy        | clean           | --          | yes               |
| api-consistency | 1 HIGH          | #1569       | yes               |
| metadata        | 0               | --          | yes               |
| test-coverage   | 3 MED           | #1570       | yes               |
```

Pull the values from each agent's returned summary. If an agent failed,
mark that row with `ERROR` in the findings column and surface the agent's
final message verbatim below the table so the user can decide whether to
re-run that single sweep manually (`/sweep-{sweep_name}`).

Finally, list the worktree branches each agent left behind so the user can
inspect or push them.

---

## General rules

- Never modify source files from the parent. All edits happen inside
  per-sweep worktrees via the subagents.
- The deliverable from the parent is: validated module, dispatch table,
  parallel agents, results table. Keep parent output concise.
- Each sweep's state CSV is registered with `merge=union` in
  `.gitattributes`, so the N concurrent state updates auto-merge cleanly
  even though they all touch the same module's row in different worktrees
  -- the last write per row wins, which is the read-update-write semantics
  the sweep templates already use.
- If a sweep template later changes its state-file schema or its audit
  categories, deep-sweep picks up the change automatically the next time
  it runs, because each subagent re-reads its sweep file on dispatch.
- If $ARGUMENTS provides a module that has no entry in any state file
  (never inspected before), that is fine -- the subagents will create the
  first row.
- /deep-sweep is not for triaging the whole codebase. For that, run the
  individual `/sweep-*` commands; they score and pick the highest-priority
  modules. Use /deep-sweep when you already know which module needs a
  full-spectrum audit.
