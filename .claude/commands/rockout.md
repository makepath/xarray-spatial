# Rockout: End-to-End Issue-to-Implementation Workflow

Take the user's prompt describing an enhancement, bug, or suggestion and drive it
through all ten steps below. The prompt is: $ARGUMENTS

---

## Step 1 -- Create a GitHub Issue

1. Decide the issue type from the prompt:
   - **enhancement** -- new feature or improvement
   - **bug** -- something broken
   - **suggestion / proposal** -- idea that needs design discussion
2. Pick labels from the repo's existing set. Always include the type label
   (`enhancement`, `bug`, or `proposal`). Add topical labels when they fit
   (e.g. `gpu`, `performance`, `focal tools`, `hydrology`, etc.).
3. Draft the title and body. Use the repo's issue templates as structure guides
   (skip the "Author of Proposal" field -- GitHub already shows the author):
   - Enhancement/proposal: follow `.github/ISSUE_TEMPLATE/feature-proposal.md`
   - Bug: follow `.github/ISSUE_TEMPLATE/bug_report.md`
4. **Run the body text through the `/humanizer` skill** before creating the issue
   to strip AI writing patterns.
5. Create the issue with `gh issue create` using the drafted title, body, and labels.
6. Capture the new issue number for later steps.

## Step 2 -- Create a Git Worktree (Isolation Contract)

The user's main checkout MUST remain on `main` for the entire rockout
run. All implementation, tests, docs, commits, and the PR push happen
inside a dedicated worktree on a feature branch. If you ever commit
from the main checkout, you have breached this contract.

1. From the main checkout, create a new branch and worktree using the
   issue number:
   ```bash
   git worktree add .claude/worktrees/issue-<NUMBER> -b issue-<NUMBER>
   ```

2. Capture the worktree path and verify isolation before doing
   anything else. Run this exact block and check every assertion:
   ```bash
   ROCKOUT_WT="$(git -C .claude/worktrees/issue-<NUMBER> rev-parse --show-toplevel)"
   ROCKOUT_MAIN="$(git rev-parse --show-toplevel)"
   ROCKOUT_BRANCH="$(git -C "$ROCKOUT_WT" branch --show-current)"
   echo "wt=$ROCKOUT_WT main=$ROCKOUT_MAIN branch=$ROCKOUT_BRANCH"
   ```

   Assert ALL of the following. If any fails, STOP, do NOT touch
   files or make commits, and report the failure to the user:
   - `$ROCKOUT_WT` ends in `.claude/worktrees/issue-<NUMBER>`.
   - `$ROCKOUT_WT` is NOT equal to `$ROCKOUT_MAIN` (you are not in
     the main checkout).
   - `$ROCKOUT_BRANCH` is `issue-<NUMBER>` (not `main`, not `master`).
   - `git -C "$ROCKOUT_MAIN" branch --show-current` is still `main`
     (or `master`) -- the main checkout's branch did NOT change.

3. `cd "$ROCKOUT_WT"` so subsequent Bash calls run inside the
   worktree by default.

4. For every Read / Edit / Write tool call from this point on, use
   paths anchored at `$ROCKOUT_WT` (or worktree-relative paths after
   the `cd`). NEVER pass an absolute path that resolves to
   `$ROCKOUT_MAIN/...` -- that bypasses the worktree and writes into
   the user's main checkout.

5. Before EVERY `git commit` you run (in any step below), re-check:
   ```bash
   [ "$(pwd)" = "$ROCKOUT_WT" ] || { echo "CWD drift"; exit 1; }
   [ "$(git branch --show-current)" = "issue-<NUMBER>" ] || { echo "branch drift"; exit 1; }
   ```
   A failed re-check is an isolation breach. Stop and report it.

## Step 3 -- Implement the Change

1. Read the relevant source files to understand the existing code.
2. Follow the project's backend-dispatch pattern (`ArrayTypeFunctionMapping`)
   when adding or modifying spatial operations.
3. Support all four backends where feasible: numpy, cupy, dask+numpy, dask+cupy.
4. Use `@ngjit` for CPU kernels and `@cuda.jit` for GPU kernels.
5. For dask support, use `map_overlap` with `depth` and `boundary=np.nan`
   when the operation needs neighborhood access.
6. Keep changes focused -- don't refactor surrounding code unnecessarily.
7. Review the implementation for OOM risks, especially dask code paths.
   Watch for patterns that accidentally materialize full arrays (e.g.
   calling `.values` or `.compute()` inside a loop, building large
   intermediate numpy arrays from dask inputs, unbounded `map_overlap`
   depth relative to chunk size). Prefer lazy operations that keep data
   chunked until final output.

## Step 4 -- Add Test Coverage

1. Add or update tests in `xrspatial/tests/`.
2. Use the project's cross-backend test helpers from `general_checks.py`.
3. Use existing fixtures from `conftest.py` (`elevation_raster`, `random_data`, etc.).
4. Any temporary files must have unique names. Include the issue number in
   the filename (e.g. `tmp_940_result.tif`) to avoid collisions with
   parallel test runs or other worktrees.
5. Cover:
   - Correctness against known values or reference implementations
   - Edge cases (NaN handling, empty input, single-cell rasters)
   - All supported backends when the implementation spans multiple backends
6. Run the tests with `pytest` to verify they pass before moving on.

## Step 5 -- Update Documentation

1. Check `docs/source/reference/` for the relevant `.rst` file.
2. Add or update the API entry for any new public functions.
3. If a new module was created, add a new `.rst` file and include it in the
   appropriate `toctree`.

## Step 6 -- Create a User Guide Notebook

**Skip this step** if the change is a pure bug fix with no new user-facing API.

Run the `/user-guide-notebook` skill to create the notebook. It handles structure,
plotting conventions, GIS alert boxes, preview images, and humanizer passes.

## Step 7 -- Update the README Feature Matrix

1. Open `README.md` and find the appropriate category section in the feature matrix.
2. Add a new row for any new function, following the existing format:
   ```
   | [Name](xrspatial/module.py) | Description | ✅️ | ✅️ | ✅️ | ✅️ |
   ```
   Use ✅️ for native backends, 🔄 for CPU-fallback, and leave blank for unsupported.
3. If the change modifies backend support for an existing function, update the
   corresponding checkmarks.

**Skip this step** if no new functions were added and no backend support changed.

## Step 8 -- Open the Pull Request

1. Push the branch to the remote with upstream tracking:
   ```
   git push -u origin issue-<NUMBER>
   ```
2. Draft a PR title and body. The body should:
   - Reference the issue with `Closes #<NUMBER>`.
   - Summarize the change in 1-3 bullets.
   - Note backend coverage (numpy / cupy / dask+numpy / dask+cupy).
   - Include a short test plan checklist.
3. **Run the PR body through the `/humanizer` skill** before opening the PR.
4. Open the PR:
   ```
   gh pr create --title "<title>" --body "$(cat <<'EOF'
   <body>
   EOF
   )"
   ```
5. Capture the PR number for the next step.

**Do NOT wait for CI to finish before moving on to Step 9.** Push the PR
and proceed to the review immediately. CI runs asynchronously and the
review-pr / follow-up loop runs in parallel. If CI surfaces a failure
later, address it as a separate follow-up commit on the same branch --
do not block the review pass on green CI.

## Step 9 -- Run the Domain-Aware PR Review

1. Invoke the `/review-pr` command against the PR number from Step 8:
   ```
   /review-pr <PR_NUMBER>
   ```
2. Do not pass "post" -- keep the review local so the rockout workflow can act
   on the findings before any of it lands as a public comment.
3. Capture the structured output. It will list findings grouped as:
   - **Blockers** -- must fix before merge
   - **Suggestions** -- should fix, not blocking
   - **Nits** -- optional improvements
4. Run this step regardless of CI status. Do not poll `gh pr checks` or
   wait for workflows to finish before invoking `/review-pr`.

## Step 10 -- Follow Up on Review Findings

Address every Blocker, then work through Suggestions and Nits in that order.

1. For each finding:
   - Read the referenced file at the cited line.
   - Decide one of: **fix**, **defer with reason**, or **dismiss with reason**.
   - Blockers must be either fixed or explicitly deferred with a written
     justification -- do not silently skip them.
   - Suggestions and nits may be dismissed when the cost outweighs the value,
     but record the reason.
2. Group related fixes into focused commits referencing the issue number
   (e.g. `Address review nits: fix NaN propagation in dask path (#<NUMBER>)`).
3. After applying fixes:
   - Re-run the tests touched by the changes.
   - Push the new commits to the PR branch.
4. Re-run `/review-pr <PR_NUMBER>` once after the follow-up commits to confirm
   the prior findings are resolved and no new ones surfaced. Stop iterating
   once only dismissed-with-reason items remain.
5. Summarize the disposition of each original finding (fixed / deferred /
   dismissed) in the final rockout summary so the trail is visible.

**Skip this step** only if Step 9 returned no Blockers, Suggestions, or Nits.

---

## General Rules

- Work entirely within the worktree created in Step 2. The main
  checkout MUST stay on `main` for the duration of the run -- never
  `git checkout`, `git switch`, `git commit`, `git add`, or edit a
  file inside `$ROCKOUT_MAIN`. Run the Step 2.5 pre-commit re-check
  before every commit.
- Commit progress after each major step with a clear commit message referencing
  the issue number (e.g. `Add flood velocity function (#42)`).
- Run `/humanizer` on any text destined for GitHub (issue body, PR description,
  commit messages) to remove AI writing artifacts.
- If any step is not applicable (e.g. no docs update needed for a typo fix),
  note why and skip it.
- At the end, print a summary of what was done and where the worktree lives.
