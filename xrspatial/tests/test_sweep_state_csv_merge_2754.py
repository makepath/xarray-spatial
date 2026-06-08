"""Regression test for issue #2754.

The sweep state CSVs under ``.claude/`` (and ``.codex/``) are keyed by module
-- one row per module -- not an append-only log.  They were once registered
with ``merge=union`` in ``.gitattributes``, which concatenates both sides of a
conflicting hunk.  When a branch was based on a stale copy, union silently
produced duplicate header lines and duplicate module rows instead of a
conflict (the file went from 9 lines to 43 on one merge).

This test does not exercise xrspatial at all; it pins the repo-tooling
contract: a concurrent edit to a sweep state CSV must surface a git merge
*conflict*, never a silent union.  If someone re-adds ``merge=union`` for these
paths, the union path below stops conflicting and this test fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# Path to the real .gitattributes shipped with the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GITATTRIBUTES = _REPO_ROOT / ".gitattributes"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _has_git() -> bool:
    try:
        subprocess.run(
            ["git", "--version"], capture_output=True, check=True
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


# One row per module, single physical line per record -- the canonical shape
# the sweep writers produce.
_BASE = (
    "module,last_inspected,notes\n"
    'aspect,2026-01-01,"a"\n'
    'slope,2026-01-01,"s"\n'
)
# Branch A updates the slope row.
_SIDE_A = (
    "module,last_inspected,notes\n"
    'aspect,2026-01-01,"a"\n'
    'slope,2026-06-01,"s2"\n'
)
# Branch B, based on the same base, adds a curvature row.
_SIDE_B = (
    "module,last_inspected,notes\n"
    'aspect,2026-01-01,"a"\n'
    'curvature,2026-05-01,"c"\n'
    'slope,2026-01-01,"s"\n'
)

_STATE_REL = ".claude/sweep-test-coverage-state.csv"


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo_2754"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    # Copy the project's real .gitattributes so we test the shipped rules.
    # The sweep globs use merge=text, which is git's built-in 3-way text
    # driver; this sandbox repo defines no custom driver of that name.
    (repo / ".gitattributes").write_text(_GITATTRIBUTES.read_text())
    state = repo / _STATE_REL
    state.parent.mkdir(parents=True)
    state.write_text(_BASE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


@pytest.mark.skipif(not _has_git(), reason="git executable not available")
def test_gitattributes_sweep_csv_not_union():
    """The .gitattributes must not set merge=union for sweep state CSVs."""
    text = _GITATTRIBUTES.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "sweep-" in stripped and "state.csv" in stripped:
            assert "merge=union" not in stripped, (
                f"sweep state CSV must not use merge=union (#2754): {stripped!r}"
            )


@pytest.mark.skipif(not _has_git(), reason="git executable not available")
def test_stale_branch_merge_conflicts_not_union(tmp_path):
    """A stale-branch merge of the keyed CSV must conflict, not union.

    Reproduces the #2754 scenario: two branches edit the same single-line
    state file from a common base.  Under the corrected attributes git's
    default 3-way text merge reports a conflict.  Under the old ``merge=union``
    driver it silently produced a file with a duplicated ``slope`` row and no
    conflict.
    """
    repo = _make_repo(tmp_path)
    state = repo / _STATE_REL
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "checkout", "-q", "-b", "branchA", base)
    state.write_text(_SIDE_A)
    _git(repo, "commit", "-qam", "A")

    _git(repo, "checkout", "-q", "-b", "branchB", base)
    state.write_text(_SIDE_B)
    _git(repo, "commit", "-qam", "B")

    _git(repo, "checkout", "-q", "branchA")
    merge = _git(repo, "merge", "--no-edit", "branchB")

    # The merge must fail with a conflict (non-zero exit), not auto-resolve.
    assert merge.returncode != 0, (
        "merge unexpectedly succeeded -- merge=union may have crept back in.\n"
        f"resulting file:\n{state.read_text()}"
    )
    merged = state.read_text()
    assert "<<<<<<<" in merged, "expected conflict markers in the merged file"
    # The union failure mode duplicated the header line; a conflict keeps one.
    # (A slope-row count would not distinguish the two: both the union output
    # and the conflict output contain two "slope," lines.)
    assert merged.count("module,last_inspected,notes") == 1, (
        "header line duplicated -- this is the union corruption from #2754"
    )
    # Clean up the conflicted merge state.
    _git(repo, "merge", "--abort")
