# Geotiff golden corpus

Tracking issue: [#1930](https://github.com/xarray-contrib/xarray-spatial/issues/1930).
This is Phase 1 PR 1 of that plan: corpus layout + manifest schema only.
No actual `.tif` files live in this directory yet (Phase 2). No backends
are wired (Phase 3).

## Layout

```
golden_corpus/
  manifest.yaml   # the contract: every dimension a fixture can have
  generate.py     # deterministic generator that rebuilds .tif files
  fixtures/       # output dir (populated by Phase 2 PRs, gitignored)
  README.md       # this file
```

## Regenerating the corpus

From the repository root, with the `tests` extras and `rasterio` +
`pyyaml` installed:

```
python -m xrspatial.geotiff.tests.golden_corpus.generate
```

Useful flags:

* `--dry-run` validates the manifest and prints the planned outputs.
  This is what the smoke test runs; you can run it locally without
  `rasterio` installed.
* `--only <id>` rebuilds one fixture by id. May be repeated.
* `--output-dir <path>` writes somewhere other than `./fixtures`.

The generator is deterministic: fixed seeds, no timestamps, sorted
iteration, and mtimes normalised to a fixed epoch.

## Adding a fixture

1. Add a new entry to `fixtures:` in `manifest.yaml`. Re-use the
   `defaults` block for everything you do not need to override.
2. Run the generator (or `--dry-run` first to catch schema errors).
3. Commit the manifest change. Generated `.tif` files are committed
   alongside the manifest entry in the same PR.

The schema is documented in the comments at the top of `manifest.yaml`
and enforced by `generate.validate()`.

## What is deliberately not in this PR

* Real fixture files. Phase 2 PRs add them in batches (tiled/stripped,
  dtypes, compression, nodata, overviews, CRS, GDAL_METADATA).
* The oracle harness (`_oracle.py`) that compares a candidate xarray
  DataArray to the rasterio baseline. That is Phase 1 PR 2 and is in
  flight in parallel.
* Backend wiring (eager, dask, gpu, vrt, http). Phase 3.

## Dependencies

`rasterio` and `pyyaml` are runtime dependencies of `generate.py`. They
are not in `setup.cfg`'s `install_requires` and not yet in the `tests`
extra. The smoke test uses `pytest.importorskip` for both so a minimal
test environment still passes. When Phase 2 starts seeding real
fixtures, the test extra should be amended to make these required.
