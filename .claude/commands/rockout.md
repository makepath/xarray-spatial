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

**Do NOT edit `CHANGELOG.md`.** Multiple rockout agents run in parallel and
every one of them touching `CHANGELOG.md` produces merge conflicts. Leave the
changelog alone -- it is updated separately at release time.

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

## Step 9 -- Run the Domain-Aware PR Review and Post It as a GitHub Review

Every rockout PR MUST receive a review posted to GitHub as a proper review
(not a plain issue comment), regardless of how clean the change looks. The
review is the audit trail.

1. Invoke the `/review-pr` command against the PR number from Step 8:
   ```
   /review-pr <PR_NUMBER>
   ```
2. Do not pass "post" -- keep `/review-pr` from posting on its own. Rockout
   will post the review explicitly in step 5 below so it lands as a GitHub
   review event, not a free-form comment.
3. Capture the structured output. It will list findings grouped as:
   - **Blockers** -- must fix before merge
   - **Suggestions** -- should fix, not blocking
   - **Nits** -- optional improvements
4. Run this step regardless of CI status. Do not poll `gh pr checks` or
   wait for workflows to finish before invoking `/review-pr`.
5. Post the captured review body to GitHub as a review event of type
   `COMMENT` so it shows up under the PR's Reviews tab (not just the
   Conversation tab). Use a heredoc to preserve formatting:
   ```bash
   gh pr review <PR_NUMBER> --comment --body "$(cat <<'EOF'
   <humanized review body from /review-pr>
   EOF
   )"
   ```
   - Use `--comment`, never `--approve` or `--request-changes`. Rockout
     does not have authority to approve its own work or block it.
   - If the review body is empty (no findings at all), still post a short
     review of type `--comment` summarizing that no issues were found, so
     every rockout PR has a visible review entry.
   - Confirm via `gh pr view <PR_NUMBER> --json reviews` that a review of
     state `COMMENTED` now exists on the PR before moving on.

## Step 10 -- Follow Up on Review Findings

Treat the review output as expert input. The reviewer is another LLM
running a checklist -- it catches real issues but occasionally misreads
context or invents problems. Your default disposition is **fix it**.
Deferral and dismissal are exceptions that require justification, not
the easy path.

**Default to fixing.** If a finding describes a real problem and the
fix is a reasonable size (typically anything that can be done in the
current session without expanding the PR's scope by more than ~50% or
pulling in unrelated subsystems), fix it now in this PR. Do not defer
work just because it is slightly more effort than the original change.
Suggestions and Nits in particular should be applied unless you have a
concrete reason not to -- "the PR already works" is not a reason.

Address every Blocker first, then work through Suggestions and Nits in
that order. Treat Suggestions and Nits as work to be done, not
optional polish.

1. For each finding:
   - Read the referenced file at the cited line and understand the
     surrounding context before deciding anything.
   - Verify the finding describes a real problem. If the reviewer
     misread the code, the cited line does not exist, or the
     "issue" is actually intended behavior, mark it **dismissed**
     and record the reason -- do not fix phantom bugs.
   - For Blockers: fix unless you can demonstrate the reviewer was
     wrong. Deferral is not an option for Blockers -- either fix or
     dismiss with a clear written explanation of the reviewer error.
   - For Suggestions: **fix by default.** Apply the change unless it
     conflicts with project conventions, would regress something else,
     or the work would substantially exceed the original PR's scope.
     A suggestion that takes a few edits and a test run is "reasonable
     size" -- do it. Do not dismiss with vague rationales like "out of
     scope" or "can be a follow-up" when the change fits in this PR.
   - For Nits: **fix by default.** Apply the change unless it is purely
     stylistic preference that conflicts with surrounding code. Nits
     are cheap; the cost of leaving them is reviewer fatigue on the
     next pass. Do not dismiss a nit just because it is a nit.
   - Deferral to a follow-up issue is only appropriate when the fix
     genuinely cannot fit in this PR -- e.g. it requires a separate
     design decision, touches an unrelated subsystem, or would more
     than roughly double the diff. When deferring, file a follow-up
     issue with `gh issue create` and link it in the summary.
   - In all cases, record the reason for dismiss / defer so the
     summary captures the reasoning, not just the verdict.
2. Group related fixes into focused commits referencing the issue number
   (e.g. `Address review nits: fix NaN propagation in dask path (#<NUMBER>)`).
3. After applying fixes:
   - Re-run the tests touched by the changes.
   - Push the new commits to the PR branch.
4. Re-run `/review-pr <PR_NUMBER>` once after the follow-up commits, and
   post the follow-up review the same way as step 9.5 above
   (`gh pr review <PR_NUMBER> --comment --body ...`). Stop iterating once
   only dismissed-with-reason items remain.
5. Summarize the disposition of each original finding (fixed / deferred /
   dismissed, with the reason for dismissals or deferrals) in the final
   rockout summary so the trail is visible. If the fixed count is low
   relative to the total findings, the summary should explain why --
   the expectation is that most findings get fixed in-PR.

**Do not skip this step.** Even if Step 9 returned no Blockers,
Suggestions, or Nits, the review of type `COMMENTED` from step 9.5 must
still be posted so every rockout PR carries a visible review entry.

## Step 11 -- Resolve Merge Conflicts With `main`

After review follow-ups are done, sync the branch with `main` and resolve
any conflicts before letting CI have the final word. Stay inside the
worktree from Step 2 -- do NOT switch the main checkout.

1. Confirm you are still in `$ROCKOUT_WT` on branch `issue-<NUMBER>`:
   ```bash
   [ "$(pwd)" = "$ROCKOUT_WT" ] || { echo "CWD drift"; exit 1; }
   [ "$(git branch --show-current)" = "issue-<NUMBER>" ] || { echo "branch drift"; exit 1; }
   ```
2. Fetch the latest `main` and check whether the branch is behind:
   ```bash
   git fetch origin main
   git log --oneline HEAD..origin/main | head
   ```
   If there are no new commits on `main`, skip to Step 12.
3. Merge `origin/main` into the feature branch (prefer merge over rebase
   so the PR history stays stable for reviewers):
   ```bash
   git merge --no-edit origin/main
   ```
4. If the merge reports conflicts:
   - Run `git status` and list every conflicted path.
   - For each conflicted file, read both sides, understand the intent,
     and edit the file to a resolution that preserves the feature work
     AND the incoming changes from `main`. Do NOT blindly accept one
     side with `git checkout --ours/--theirs` unless you have read the
     file and confirmed the other side is irrelevant.
   - After editing, `git add <file>` for each resolved path.
   - When all conflicts are resolved, finalize with `git commit` (no
     `-m` flag needed -- git will use the prepared merge message).
5. Re-run the test suite touched by the change to confirm the merge did
   not break behaviour. If tests fail because of the merge, fix the
   root cause; do not paper over with skips.
6. Push the merge commit to the PR branch:
   ```bash
   git push origin issue-<NUMBER>
   ```
7. Confirm via `gh pr view <PR_NUMBER> --json mergeable,mergeStateStatus`
   that the PR is no longer in a conflicted state before moving on.

If the merge produces no conflicts and no test fallout, this step is a
fast no-op. Run it anyway -- the goal is to know the PR is mergeable
before CI failures get evaluated in Step 12.

## Step 12 -- Fix CI Failures

CI runs asynchronously after the push in Step 8 (and again after the
follow-up pushes in Steps 10 and 11). This is the final gate: drive every
required check to green before declaring the rockout done.

1. Poll the PR's check status until every check has completed (success
   or failure -- not pending):
   ```bash
   gh pr checks <PR_NUMBER>
   ```
   If checks are still running, wait and re-poll. Do not declare done
   while any required check is pending.
2. For each failing check:
   - Pull the failing job's logs:
     ```bash
     gh run view --log-failed --job <JOB_ID>
     ```
     or open the run via `gh pr checks <PR_NUMBER> --watch` and drill
     into the failing job.
   - Read the actual failure (test name, traceback, lint rule, etc.).
     Do not guess from the check name.
   - Classify the failure:
     - **Real defect in the change** -- fix the code, add or update a
       test if coverage was missing, commit the fix.
     - **Pre-existing flake unrelated to the change** -- rerun the
       failed job once with `gh run rerun <RUN_ID> --failed`. If it
       passes, note it in the summary and move on. If it fails again
       in the same way, treat it as a real failure and fix it.
     - **Environment / infra issue** (cache miss, runner outage, token
       expiry) -- rerun the failed job. If it keeps failing for the
       same infra reason after one rerun, surface it to the user
       rather than hacking around it.
3. For real defects, follow the same isolation rules as earlier steps:
   work inside `$ROCKOUT_WT` on `issue-<NUMBER>`, commit with a message
   referencing the issue (e.g. `Fix dask path NaN handling for CI (#<NUMBER>)`),
   and push to the PR branch.
4. After each push, repeat from step 1 until every required check is
   green. Do not merge or hand off while any required check is red.
5. If a check is genuinely not relevant to the change and cannot be
   made green (e.g. an unrelated workflow that is broken on `main`),
   record the reason in the final summary and flag it to the user --
   do not silently ignore red checks.
6. Once all required checks are green, run the Step 11 conflict re-check
   one more time (`gh pr view <PR_NUMBER> --json mergeable,mergeStateStatus`)
   to confirm nothing landed on `main` while CI was running that would
   re-conflict the branch.

The rockout run is only complete when:
- Every required CI check on the PR is green (or explicitly justified).
- The PR reports `mergeable` with no conflicts against `main`.
- The Step 9 / Step 10 review trail is posted.

---

## General Rules

- Work entirely within the worktree created in Step 2. The main
  checkout MUST stay on `main` for the duration of the run -- never
  `git checkout`, `git switch`, `git commit`, `git add`, or edit a
  file inside `$ROCKOUT_MAIN`. Run the Step 2.5 pre-commit re-check
  before every commit.
- Commit progress after each major step with a clear commit message referencing
  the issue number (e.g. `Add flood velocity function (#42)`).
- Never modify `CHANGELOG.md` during a rockout run. Parallel agents all editing
  it cause merge conflicts; the changelog is maintained separately at release time.
- Run `/humanizer` on any text destined for GitHub (issue body, PR description,
  commit messages) to remove AI writing artifacts.
- If any step is not applicable (e.g. no docs update needed for a typo fix),
  note why and skip it.
- At the end, print a summary of what was done and where the worktree lives.
