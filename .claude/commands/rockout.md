# Rockout: End-to-End Issue-to-Implementation Workflow

Take the user's prompt describing an enhancement, bug, or suggestion and drive it
through all seven steps below. The prompt is: $ARGUMENTS

---

## Step 1 -- Create a GitHub Issue

1. Decide the issue type from the prompt:
   - **enhancement** -- new feature or improvement
   - **bug** -- something broken
   - **suggestion / proposal** -- idea that needs design discussion
2. Pick labels from the repo's existing set. Always include the type label
   (`enhancement`, `bug`, or `proposal`). Add topical labels when they fit
   (e.g. `gpu`, `performance`, `focal tools`, `hydrology`, etc.).
3. Draft the title and body. Use the repo's issue templates as structure guides:
   - Enhancement/proposal: follow `.github/ISSUE_TEMPLATE/feature-proposal.md`
   - Bug: follow `.github/ISSUE_TEMPLATE/bug_report.md`
4. **Run the body text through the `/humanizer` skill** before creating the issue
   to strip AI writing patterns.
5. Create the issue with `gh issue create` using the drafted title, body, and labels.
6. Capture the new issue number for later steps.

## Step 2 -- Create a Git Worktree

1. Create a new branch and worktree using the issue number:
   ```
   git worktree add .claude/worktrees/issue-<NUMBER> -b issue-<NUMBER>
   ```
2. Switch the working directory to the new worktree for all remaining steps.

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

The project has an `examples/user_guide/` directory with numbered notebooks.

1. Determine the next available notebook number by listing the directory.
2. Create a new `.ipynb` notebook following the established pattern:
   - Markdown cell with title and explanation of the feature
   - Import cell
   - Synthetic data generation with visualization
   - Demonstrate each mode/option of the feature
   - Show a practical use case or comparison
3. Use `matplotlib` for plots, consistent with existing notebooks.
4. Keep the notebook self-contained (no external data dependencies).

**Skip this step** if the change is a pure bug fix with no new user-facing API.

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

---

## General Rules

- Work entirely within the worktree created in Step 2.
- Commit progress after each major step with a clear commit message referencing
  the issue number (e.g. `Add flood velocity function (#42)`).
- Run `/humanizer` on any text destined for GitHub (issue body, PR description,
  commit messages) to remove AI writing artifacts.
- If any step is not applicable (e.g. no docs update needed for a typo fix),
  note why and skip it.
- At the end, print a summary of what was done and where the worktree lives.
