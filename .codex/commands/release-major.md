# Release: Major

Cut a major release (X.Y.Z -> X+1.0.0). Follow every step below in order.

$ARGUMENTS

---

## Step 1 -- Determine the new version

1. Run `git tag --sort=-v:refname | head -5` to find the latest tag.
2. Parse the current version (format `vX.Y.Z`).
3. Increment the **major** component and reset minor+patch: `X.Y.Z` -> `(X+1).0.0`.
4. Store the new version string (without `v` prefix) for later steps.

## Step 2 -- Create a release branch

```bash
git checkout main && git pull
git checkout -b release/vX.Y.Z
```

## Step 3 -- Update CHANGELOG.md

1. Run `git log --pretty=format:"- %s" <latest_tag>..HEAD` to collect
   changes since the last release.
2. Add a new section at the top of CHANGELOG.md (below the header line)
   matching the existing format:
   ```
   ### Version X.Y.Z - YYYY-MM-DD

   #### New Features
   - feature description (#PR)

   #### Bug Fixes & Improvements
   - fix description (#PR)
   ```
3. Use today's date.  Categorize entries under "New Features" and/or
   "Bug Fixes & Improvements" as appropriate.
4. Run `/humanizer` on the changelog text before writing it.

## Step 4 -- Commit and push

```bash
git add CHANGELOG.md
git commit -m "Update CHANGELOG for vX.Y.Z release"
git push -u origin release/vX.Y.Z
```

## Step 5 -- Verify CI

1. Run `gh pr create --title "Release vX.Y.Z" --body "Changelog update for vX.Y.Z major release."` to open a PR against main.
2. Wait for CI:
   ```bash
   gh pr checks <PR_NUMBER> --watch
   ```
3. If CI fails, fix the issue, amend or add a commit, push, and re-check.

## Step 6 -- Merge the release branch

```bash
gh pr merge <PR_NUMBER> --merge --delete-branch
```

## Step 7 -- Tag the release

```bash
git checkout main && git pull
git tag -a vX.Y.Z -m "Version X.Y.Z"
git push origin vX.Y.Z
```

Do **not** sign the tag (`-s` flag omitted).

## Step 8 -- Create a GitHub release

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <(changelog_excerpt)
```

Use the CHANGELOG section for this version as the release notes body.
Run `/humanizer` on the notes before creating the release.

## Step 9 -- Verify PyPI

1. The `pypi-publish.yml` workflow triggers automatically on tag push.
2. Watch the workflow:
   ```bash
   gh run list --workflow=pypi-publish.yml --limit 1
   gh run watch <RUN_ID>
   ```
3. Confirm the new version appears:
   ```bash
   pip index versions xarray-spatial 2>/dev/null || echo "Check https://pypi.org/project/xarray-spatial/"
   ```

## Step 10 -- Summary

Print the new version, links to the PR, GitHub release, and PyPI page.

---

## General rules

- Run `/humanizer` on all text destined for GitHub: PR title/body, release
  notes, commit messages, and any comments left on issues or PRs.
- Any temporary files created during the release (build artifacts, scratch
  files) must use unique names including the version number to avoid
  collisions (e.g. `changelog-draft-1.0.0.md`).
