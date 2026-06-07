# Bench: Local Performance Comparison

Run ASV benchmarks for the current branch against main and report regressions
and improvements. The prompt is: {{ARGUMENTS}}

---

## Step 1 -- Identify what changed

1. If {{ARGUMENTS}} names specific benchmark classes or functions (e.g. `Slope`,
   `flow_accumulation`), use those directly.
2. If {{ARGUMENTS}} is empty or says "auto", run `git diff origin/main --name-only`
   to find changed source files under `xrspatial/`. Map each changed file to the
   corresponding benchmark module in `benchmarks/benchmarks/`. Use the filename
   and imports to match (e.g. changes to `slope.py` map to `benchmarks/benchmarks/slope.py`).
3. If no benchmark exists for the changed code, note this in the report and
   suggest whether one should be added.

## Step 2 -- Check prerequisites

1. Verify ASV is installed: `python -c "import asv"`. If missing, tell the user
   to install it (`pip install asv`) and stop.
2. Verify the benchmarks directory exists at `benchmarks/`.
3. Read `benchmarks/asv.conf.json` to confirm the project name and branch settings.
4. Check whether the ASV machine file exists (`.asv/machine.json`). If not, run
   `cd benchmarks && asv machine --yes` to initialize it.

## Step 3 -- Run the comparison

Run ASV in continuous-comparison mode from the `benchmarks/` directory:

```bash
cd benchmarks && asv continuous origin/main HEAD -b "<regex>" -e
```

Where `<regex>` is a pattern matching the benchmark classes identified in Step 1
(e.g. `Slope|Aspect` or `FlowAccumulation`). The `-e` flag shows stderr on failure.

If {{ARGUMENTS}} contains "quick", add `--quick` to run each benchmark only once
(faster but noisier).

If {{ARGUMENTS}} contains "full", omit the `-b` filter to run all benchmarks.

## Step 4 -- Parse and interpret results

ASV continuous outputs lines like:
```
BENCHMARKS NOT SIGNIFICANTLY CHANGED.
```
or:
```
REGRESSION: benchmarks.slope.Slope.time_numpy  3.45ms -> 5.67ms  (1.64x)
IMPROVED:   benchmarks.slope.Slope.time_dask   8.12ms -> 4.23ms  (0.52x)
```

Parse the output and classify each result:

| Category     | Criteria                    |
|--------------|-----------------------------|
| REGRESSION   | Ratio > 1.2x (matches CI)   |
| IMPROVED     | Ratio < 0.8x                |
| UNCHANGED    | Between 0.8x and 1.2x       |

## Step 5 -- Generate the report

```
## Benchmark Report: <branch> vs main

### Changed files
- <list of changed source files>

### Benchmarks run
- <list of benchmark classes/functions matched>

### Results

| Benchmark                          | main    | HEAD      | Ratio | Status     |
|------------------------------------|-----------|-----------|-------|------------|
| slope.Slope.time_numpy             | 3.45 ms   | 3.51 ms   | 1.02x | UNCHANGED  |
| slope.Slope.time_dask_numpy        | 8.12 ms   | 4.23 ms   | 0.52x | IMPROVED   |
| ...                                | ...       | ...       | ...   | ...        |

### Regressions
<details for each regression: which benchmark, how much slower, likely cause>

### Improvements
<details for each improvement>

### Missing benchmarks
<list any changed functions that have no benchmark coverage>

### Recommendation
- [ ] Safe to merge (no regressions)
- [ ] Add "performance" label to PR (regressions found, CI will recheck)
- [ ] Consider adding benchmarks for: <uncovered functions>
```

## Step 6 -- Suggest benchmark additions (if gaps found)

If Step 1 found changed functions with no benchmark coverage:

1. Read an existing benchmark file in `benchmarks/benchmarks/` that covers a
   similar function (same category or same backend pattern).
2. Describe what a new benchmark should test:
   - Which function and parameter variants
   - Suggested array sizes (match `common.py` conventions)
   - Which backends to benchmark (numpy at minimum, dask if applicable)
3. Ask the user whether they want you to write the benchmark file.

Do NOT write benchmark files automatically. Report the gap and propose, then wait.

---

## General rules

- Always run benchmarks from the `benchmarks/` directory, not the project root.
- The regression threshold is 1.2x, matching `.github/workflows/benchmarks.yml`.
  Do not change this unless {{ARGUMENTS}} overrides it.
- If ASV setup or machine detection fails, report the error clearly and suggest
  the fix. Do not retry in a loop.
- If benchmarks take longer than 5 minutes per class, note the elapsed time so
  the user can plan accordingly.
- Do not modify any source, test, or benchmark files. This command is read-only
  analysis (unless the user explicitly asks for a benchmark to be written in
  response to Step 6).
- If {{ARGUMENTS}} says "compare <branch1> <branch2>", run
  `asv continuous <branch1> <branch2>` instead of the default origin/main vs HEAD.
