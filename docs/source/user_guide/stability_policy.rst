..  _user_guide.stability_policy:

Stability policy and LTS commitment
===================================

This page states which parts of xarray-spatial are covered by long-term
support (LTS) in the 1.0.0 series, and what "stable" means here.

What LTS covers
---------------

The high-level public spellings are the LTS surface: the names exported from
``xrspatial/__init__.py`` and the ``.xrs`` accessor on
``xarray.DataArray`` and ``xarray.Dataset``. Within the 1.0.x series these
names, their call signatures, and their documented return contracts do not
change in backward-incompatible ways. New optional keyword arguments may be
added; existing behavior for documented inputs is preserved.

What "stable" requires
----------------------

Following the feature-tier work in #2415, a function reaching the ``stable``
tier needs a documented contract and a release gate, not green cross-backend
tests alone. Concretely:

- a documented input/output contract (dtypes, dims, attrs, nodata handling),
- no open CRITICAL or HIGH ``/deep-sweep`` findings in accuracy or security,
- a notebook example.

Tiers below stable
------------------

``advanced`` and ``experimental`` functions work and are tested but carry
caveats or make no cross-backend parity claim. GPU and some Dask paths
usually stay at these tiers. The README feature matrix is the per-backend
source of truth for each function's tier.
