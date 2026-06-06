# Ready to Merge: Surface PRs Safe to Merge

Scan the open pull requests and report the ones that are ready to merge. A PR is
ready when it has been reviewed, its review blockers are resolved, it has no
merge conflict with `main`, and CI is green. A failing Read the Docs build is
tolerated, because RTD flakes under rate limiting and that failure does not
reflect the change. The prompt is: {{ARGUMENTS}}

This command is read-only. It reports findings. It does not apply labels, post
comments, approve, or merge anything.

If `{{ARGUMENTS}}` names a label, author, or PR numbers, narrow the scan to those.
Otherwise scan every open non-draft PR.

---

## Step 1 -- List the open PRs

```bash
gh pr list --state open --limit 100 \
  --json number,title,url,isDraft,headRefName,reviews,mergeable,mergeStateStatus
```

Drop any PR where `isDraft` is true -- a draft is never ready to merge. Record
the remaining PRs as the candidate set.

Run the cheap, deterministic gates (Steps 2-4) on every candidate first. Only the
PRs that clear all three reach the expensive review re-run in Step 5.

## Step 2 -- Reviewed gate

A PR qualifies as reviewed when it has at least one review of any state -- an
`APPROVED` review or a `COMMENTED` review both count. Many PRs here carry a
`COMMENTED` review from automated tooling rather than a formal approval, so do
not require `reviewDecision == APPROVED`.

From the Step 1 JSON, a PR passes this gate when its `reviews` array is
non-empty. A PR with zero reviews is excluded with reason `not reviewed`.

If a PR's reviews are all `COMMENTED` with none `APPROVED`, it still passes the
gate, but flag it in the Step 6 report as `(no approving review)`. A rockout PR
carries a `COMMENTED` review posted by automation, so "reviewed" here can mean
"a bot looked", not "a human approved". Surfacing that lets the reader decide
whether an independent approval is needed before merging.

## Step 3 -- Merge-conflict gate

GitHub computes `mergeable` lazily, so the Step 1 list often reports
`"mergeable":"UNKNOWN"`. Do not trust `UNKNOWN`. For each candidate still in the
running, re-fetch until the value settles:

```bash
gh pr view <number> --json mergeable,mergeStateStatus
```

If it is still `UNKNOWN`, wait a few seconds and re-fetch (GitHub starts the
computation when first asked). Once it settles:

- `mergeable == "MERGEABLE"` -- passes this gate.
- `mergeable == "CONFLICTING"` -- excluded with reason `merge conflict with main`.
- `mergeStateStatus == "DIRTY"` also indicates a conflict.

`mergeStateStatus == "BEHIND"` (branch behind `main` but no conflict) does not by
itself disqualify a PR -- note it but let the PR through this gate.

## Step 4 -- CI gate, with the Read the Docs exception

Pull the check rollup for each candidate as JSON so you read a stable `bucket`
field instead of parsing the human-readable table:

```bash
gh pr checks <number> --json name,state,bucket
```

Each check has a `bucket` of `pass`, `fail`, `pending`, or `skipping`. The
`--json` form exits 0 even when checks fail, so read its output directly.
Classify the PR from the buckets:

- **Any check has bucket `pending`** -- the PR is not ready *yet*. Exclude it
  with reason `CI still running` rather than treating it as a failure.
- **A check has bucket `fail`** -- look at the check `name`:
  - The Read the Docs check is named `docs/readthedocs.org:xarray-spatial`. A
    failure on this check alone is tolerated (RTD rate-limit flakiness). It does
    not disqualify the PR. This name is the only RTD assumption in the command;
    if the RTD project slug ever changes, a real RTD failure would start
    disqualifying PRs (a stricter failure mode, never a silent pass), so update
    the name here if that happens.
  - Any other failing check disqualifies the PR. Exclude it with reason
    `CI failure: <check name>`.
- **Every check is bucket `pass` or `skipping`** (or the only `fail` is the RTD
  check) -- passes this gate.

Only a `fail` bucket on a non-RTD check, or a `pending` bucket, holds a PR back.

## Step 5 -- Blockers-addressed gate (review re-run)

For each PR that cleared Steps 2-4, re-run the domain-aware review to confirm no
unresolved blockers remain:

```
review-pr <number>
```

Do not pass `post` -- this is an inspection, not a review to publish. Read the
structured output:

- **Zero Blockers** -- the PR passes this gate and is ready to merge. Report any
  remaining Suggestions or Nits as informational so a human can weigh them, but
  they do not hold the PR back (they are advisory, not merge blockers).
- **One or more Blockers** -- excluded with reason
  `open review blockers (N)`, and list the blocker titles so the author knows
  what to fix.

This step is the slow one -- each re-run spends tokens and time. That is the
cost of trusting the "blockers addressed" signal rather than guessing from
metadata alone. Run it only on the PRs that survived the cheap gates.

## Step 6 -- Report

Print two sections.

**Ready to merge** -- a markdown list, one line per qualifying PR, each linking
to the PR:

```
## Ready to merge

- [#2746 aspect: test degenerate shapes ...](https://github.com/xarray-contrib/xarray-spatial/pull/2746)
- [#2738 Add dask+cupy test coverage ...](https://github.com/xarray-contrib/xarray-spatial/pull/2738)
```

If a ready PR has a tolerated RTD failure, no approving review, or outstanding
advisory suggestions/nits, append a short parenthetical so the human is not
surprised (e.g. `(RTD build failing -- ignored)`, `(no approving review)`, or
`(2 advisory nits)`).

**Excluded** -- a markdown list of every other open PR with the specific reason
it did not qualify, so the gap to ready is obvious:

```
## Excluded

- [#2745 Guard degenerate-axis resolution ...](...) -- CI failure: run (windows-latest, 3.14)
- [#2737 Style cleanup in focal.py ...](...) -- not reviewed
- [#2729 proximity: style cleanup ...](...) -- merge conflict with main
- [#2719 proximity: add return annotations ...](...) -- open review blockers (1): missing dask coverage
```

If no PR qualifies, say so plainly and show the Excluded list -- that list is the
to-do list for getting PRs merge-ready.

Do not apply the `ready to merge` label, comment on any PR, or merge anything.
The output is a report for a human to act on.
