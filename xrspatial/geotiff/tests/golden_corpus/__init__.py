"""Golden corpus for xrspatial geotiff parity tests (issue #1930).

This package hosts the corpus fixtures, manifest, and oracle harness used
by Phase 3 backend cells. The pieces are split across PRs so they can
land independently:

* Phase 1 PR 1 -- manifest + deterministic generator. The contract for
  what a fixture is.
* Phase 1 PR 2 -- the oracle harness (``_oracle``).
* Phase 2     -- the seed fixtures themselves.
* Phase 3     -- the per-backend test cells that call the oracle.
"""
