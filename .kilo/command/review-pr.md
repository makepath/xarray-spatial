# Review PR: Domain-Aware Pull Request Review

Review a pull request with checks specific to a geospatial raster library built on
NumPy, Dask, CuPy, and Numba. The prompt is: {{ARGUMENTS}}

---

## Step 1 -- Load the PR

1. If {{ARGUMENTS}} contains a PR number (e.g. `123`), fetch it:
   ```bash
   gh pr view <number> --json title,body,files,commits,baseRefName,headRefName
   ```
2. If {{ARGUMENTS}} is empty, check whether the current branch has an open PR:
   ```bash
   gh pr view --json title,body,files,commits,baseRefName,headRefName
   ```
3. If neither works, tell the user to provide a PR number and stop.
4. Get the full diff:
   ```bash
   gh pr diff <number>
   ```

## Step 1.5 -- Materialize the PR in a worktree

The user's main checkout MUST stay on `main`. Read the PR's files
from a worktree on the PR's head branch so the review sees the
actual PR state, not whatever happens to be checked out in the
main directory.

First, detect whether we are already inside a worktree on the PR's
head branch (this is the common case when `/review-pr` is invoked
from `/rockout` Step 9):

```bash
REVIEW_PR_NUM=<number>
REVIEW_HEAD_BRANCH="$(gh pr view "$REVIEW_PR_NUM" --json headRefName -q .headRefName)"
REVIEW_CUR_BRANCH="$(git branch --show-current)"
REVIEW_CUR_TOP="$(git rev-parse --show-toplevel)"
```

- If `$REVIEW_CUR_BRANCH` equals `$REVIEW_HEAD_BRANCH` AND
  `$REVIEW_CUR_TOP` contains the segment `.kilo/worktrees/`,
  we are already in the right worktree. Set
  `REVIEW_WT="$REVIEW_CUR_TOP"` and skip to step 4 below. Do NOT
  create another worktree -- a second `git worktree add` on the
  same branch will fail.

- Otherwise, create a dedicated review worktree:

  1. From any path, resolve the main checkout (use `--git-common-dir`
     to find the shared repo even if we are inside another worktree):
     ```bash
     REVIEW_MAIN="$(git rev-parse --path-format=absolute --git-common-dir)"
     REVIEW_MAIN="${REVIEW_MAIN%/.git}"
     git -C "$REVIEW_MAIN" fetch origin "pull/$REVIEW_PR_NUM/head:pr-$REVIEW_PR_NUM-review"
     git -C "$REVIEW_MAIN" worktree add \
       ".kilo/worktrees/pr-$REVIEW_PR_NUM-review" "pr-$REVIEW_PR_NUM-review"
     REVIEW_WT="$REVIEW_MAIN/.kilo/worktrees/pr-$REVIEW_PR_NUM-review"
     REVIEW_WT_CREATED=1
     ```

  2. Verify isolation -- assert ALL of the following. If any fails,
     STOP and report it:
     - `$REVIEW_WT` exists and is NOT equal to `$REVIEW_MAIN`.
     - `git -C "$REVIEW_WT" branch --show-current` is
       `pr-$REVIEW_PR_NUM-review`.
     - `git -C "$REVIEW_MAIN" branch --show-current` is still
       `main` (or `master`).

3. `cd "$REVIEW_WT"` so subsequent reads happen inside the worktree.

4. Read every changed file in full (not just the diff) from
   `$REVIEW_WT`. Use paths anchored at `$REVIEW_WT` for all Read
   tool calls -- never read the same file from the main checkout;
   that path reflects `main` and will mislead the review.

5. The review is read-only -- do NOT make commits in this worktree.
   When the review is done (after Step 8), clean up only if Step
   1.5 created the worktree:
   ```bash
   if [ "${REVIEW_WT_CREATED:-0}" = "1" ]; then
     cd "$REVIEW_MAIN"
     git worktree remove ".kilo/worktrees/pr-$REVIEW_PR_NUM-review"
     git branch -D "pr-$REVIEW_PR_NUM-review"
   fi
   ```

## Step 2 -- Correctness review

Check the changed code for numerical and algorithmic correctness:

### 2a. Algorithm accuracy
- Does the implementation match the cited algorithm or paper? If a paper or
  standard is referenced (in comments, docstring, or PR body), verify the
  formulas match.
- Are there off-by-one errors in neighborhood indexing (common in 3x3 kernels)?
- Is the output in the correct units and range? (e.g. slope in degrees 0-90,
  aspect in degrees 0-360, NDVI in -1 to 1)

### 2b. Floating point concerns
- Are there divisions that could produce inf or NaN on valid input?
- Is there catastrophic cancellation risk (subtracting nearly equal large numbers)?
- Does the code handle the float32 vs float64 distinction correctly? (e.g. using
  float64 intermediates for accumulation, returning the expected output dtype)

### 2c. NaN handling
- Does the function propagate NaN correctly for its semantics?
- For neighborhood operations with `boundary='nan'`: do edge cells become NaN?
- Are NaN checks using `np.isnan` (not `== np.nan`)?

### 2d. Edge cases
- Empty input, single-row, single-column, 1x1 rasters
- All-NaN input
- Constant-value input (derivative operations should return zero)
- Very large or very small values

## Step 3 -- Backend completeness review

### 3a. Dispatch registration
- Does the `ArrayTypeFunctionMapping` include all four backends?
- If a backend is intentionally omitted, is there a comment explaining why?
- Does the public function's docstring mention which backends are supported?

### 3b. Dask correctness
- Does `map_overlap` use the correct `depth` for the kernel size?
  (depth should be `kernel_radius`, e.g. 1 for a 3x3 kernel)
- Is the `boundary` parameter forwarded correctly from the public API to
  `map_overlap`?
- Does the chunk function return the same shape as its input?
- For 3D stacked arrays: is `.rechunk({0: N})` called after `da.stack()`?

### 3c. CuPy correctness
- Does the CUDA kernel handle array bounds correctly (guard against
  out-of-bounds thread indices)?
- Is the thread block size appropriate for the kernel's register usage?
- Are results extracted with `.data.get()`, not `.values`?

## Step 4 -- Performance review

### 4a. Anti-patterns
Run the same checks as `/efficiency-audit` but scoped to only the changed files.
Specifically check for:
- Premature materialization (`.values`, `.compute()` in loops)
- Unnecessary copies
- GPU register pressure in new CUDA kernels
- Missing `@ngjit` on CPU loops

### 4b. Benchmark coverage
- Does a benchmark exist in `benchmarks/benchmarks/` for the changed function?
- If this PR adds a new function, does it also add a benchmark?
- If the PR modifies performance-critical code, should the "performance" label
  be added?

## Step 5 -- Test coverage review

### 5a. Test existence
- Are there tests for the changed code?
- Do tests cover all implemented backends (using the helpers from
  `general_checks.py`)?

### 5b. Test quality
- Do tests compare against known reference values (QGIS, analytical, etc.),
  not just "does it run without crashing"?
- Are edge cases tested (NaN, constant surface, boundary modes)?
- Do dask tests use multiple chunk sizes (including ragged chunks)?
- Are temporary files uniquely named?

### 5c. Missing tests
- List any code paths or parameter combinations that have no test coverage.

## Step 6 -- Documentation and API review

### 6a. Docstrings
- Does every new public function have a docstring with Parameters, Returns,
  and a short description?
- Are parameter types and defaults documented?

### 6b. README feature matrix
- If a new function was added, is it in the README feature matrix?
- Are the backend checkmarks accurate?

### 6c. API consistency
- Does the function signature follow the project's conventions?
  (e.g. `agg` for input DataArray, `name` for output name, `boundary` for
  boundary mode)
- Does it return an `xr.DataArray` with coords, dims, and attrs preserved?

## Step 7 -- Generate the review

Format the review as a structured comment suitable for posting on the PR.
Organize findings by severity:

```
## PR Review: <title>

### Blockers (must fix before merge)
- [ ] <finding with file:line reference>

### Suggestions (should fix, not blocking)
- [ ] <finding with file:line reference>

### Nits (optional improvements)
- [ ] <finding with file:line reference>

### What looks good
- <positive observations, kept brief>

### Checklist
- [ ] Algorithm matches reference/paper
- [ ] All implemented backends produce consistent results
- [ ] NaN handling is correct
- [ ] Edge cases are covered by tests
- [ ] Dask chunk boundaries handled correctly
- [ ] No premature materialization or unnecessary copies
- [ ] Benchmark exists or is not needed
- [ ] README feature matrix updated (if applicable)
- [ ] Docstrings present and accurate
```

After generating the review, run it through [TOOL: humanize] before
showing it to the user or posting it to GitHub.

## Step 8 -- Post (if requested)

If {{ARGUMENTS}} includes "post" or "comment":
1. Post the review as a PR comment using `gh pr comment <number> --body "..."`.
2. Confirm the comment was posted successfully.

If {{ARGUMENTS}} does not include "post", show the review to the user and ask
whether they want it posted.

---

## General rules

- Do not approve or request changes on the PR via GitHub's review system. Only
  post comments.
- Read the full context of changed files, not just the diff. Many bugs are only
  visible when you understand the surrounding code.
- Be specific. Every finding must include a file path and line number. Vague
  feedback ("consider improving performance") is not useful.
- Do not suggest changes to code that was not modified in the PR unless the
  existing code has a clear bug that the PR makes worse.
- False positives erode trust. If you are uncertain whether something is a
  problem, say so explicitly rather than presenting it as a definite issue.
- Run [TOOL: humanize] on the final review text before posting or displaying.
- If {{ARGUMENTS}} includes "quick", skip Steps 4 and 6 (performance and docs)
  and focus only on correctness, backend parity, and test coverage.
