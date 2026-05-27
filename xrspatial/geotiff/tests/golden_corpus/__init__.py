"""Golden corpus for xrspatial geotiff parity tests.

This package hosts the corpus fixtures, manifest, and oracle harness the
per-backend test cells run against. The pieces are:

* the manifest plus the deterministic generator that turns it into
  fixtures -- the contract for what a fixture is;
* the oracle harness (``_oracle``);
* the seed fixtures themselves;
* the per-backend test cells that call the oracle.
"""
