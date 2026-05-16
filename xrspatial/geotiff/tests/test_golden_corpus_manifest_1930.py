"""Smoke test for the golden corpus manifest + generator (issue #1930).

This is Phase 1 PR 1 of the plan in #1930: the manifest schema and the
deterministic generator. The test verifies that:

* the manifest parses,
* every fixture entry has the required schema fields after defaults are
  merged in,
* duplicate ids are rejected,
* dry-run mode runs end-to-end without writing files,
* fixture ids round-trip into the planned output paths in sorted order,
* the validator catches the kinds of schema errors a contributor is
  likely to hit (bad enum value, missing required field, conflicting
  CRS spec).

No real fixture files are produced or asserted here; that is Phase 2.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

# pyyaml is needed to read manifest.yaml. It is not in the package's
# install_requires (and not yet in the `tests` extra; see the README in
# the golden_corpus directory). importorskip keeps minimal test
# environments green until the test extra is amended in Phase 2.
pytest.importorskip("yaml")

generate = importlib.import_module(
    "xrspatial.geotiff.tests.golden_corpus.generate"
)


def test_manifest_parses():
    """The shipped manifest loads as a dict with the expected top-level keys."""
    manifest = generate.load_manifest()
    assert isinstance(manifest, dict)
    assert manifest.get("version") == 1
    assert "fixtures" in manifest
    assert isinstance(manifest["fixtures"], list)


def test_manifest_validates_and_has_at_least_one_entry():
    """Every shipped fixture passes the validator with defaults applied."""
    manifest = generate.load_manifest()
    resolved = generate.validate(manifest)
    assert len(resolved) >= 1
    for entry in resolved:
        for field in generate.REQUIRED_FIELDS:
            assert field in entry, f"{entry.get('id')!r} missing {field}"


def test_manifest_ids_are_unique():
    """Validator rejects duplicate fixture ids."""
    manifest = generate.load_manifest()
    fixtures = manifest["fixtures"]
    if len(fixtures) < 1:
        pytest.skip("no fixtures to duplicate")
    bad = dict(manifest)
    bad["fixtures"] = list(fixtures) + [dict(fixtures[0])]
    with pytest.raises(generate.ManifestError, match="duplicate"):
        generate.validate(bad)


def test_dry_run_does_not_write(tmp_path):
    """Dry-run returns planned paths in sorted-id order without writing anything."""
    out = tmp_path / "fixtures_dryrun"
    paths = generate.generate(output_dir=out, dry_run=True)
    assert paths, "expected at least one planned fixture"
    assert paths == sorted(paths, key=lambda p: p.name)
    assert not out.exists(), "dry-run must not create the output directory"
    for p in paths:
        assert isinstance(p, pathlib.Path)
        assert p.suffix == ".tif"
        assert p.parent == out


def test_unknown_only_id_errors():
    """--only with an unknown id is reported as a manifest error."""
    with pytest.raises(generate.ManifestError, match="unknown fixture ids"):
        generate.generate(only=["definitely-not-a-real-fixture-id"], dry_run=True)


@pytest.mark.parametrize(
    "mutate, match",
    [
        # Bad enum
        (lambda e: e.update(layout="checkered"), "layout must be one of"),
        # Missing required
        (lambda e: e.pop("dtype"), "missing required fields"),
        # Bad predictor
        (lambda e: e.update(predictor=99), "predictor must be one of"),
        # Tiled without tile_size
        (
            lambda e: (e.update(layout="tiled"), e.pop("tile_size", None)),
            "tile_size",
        ),
        # CRS with conflicting keys
        (
            lambda e: e.update(crs={"epsg": 4326, "wkt": "GEOGCS[..]"}),
            "exactly one of",
        ),
        # Bad nodata
        (lambda e: e.update(nodata="bananas"), "nodata must be"),
        # Predictor 3 on integer dtype
        (
            lambda e: e.update(predictor=3, dtype="uint16"),
            r"predictor 3 \(floating-point\) requires a float dtype",
        ),
        # Predictor 2 on float dtype
        (
            lambda e: e.update(predictor=2, dtype="float32"),
            r"predictor 2 \(horizontal\) requires an integer dtype",
        ),
        # Non-bool external_overview
        (
            lambda e: e.update(external_overview="yes"),
            "external_overview must be a bool",
        ),
    ],
)
def test_validator_rejects_bad_entries(mutate, match):
    """Schema errors a contributor is likely to hit are caught with a clear message."""
    manifest = generate.load_manifest()
    fixtures = manifest["fixtures"]
    assert fixtures, "manifest must ship at least one example fixture"

    # Start from the merged form so defaults are present, then mutate.
    defaults = manifest.get("defaults") or {}
    entry = dict(defaults)
    entry.update(fixtures[0])
    mutate(entry)

    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(generate.ManifestError, match=match):
        generate.validate(bad)


def test_required_fields_constant_covers_dimensions():
    """REQUIRED_FIELDS keeps the issue-1930 dimensions in lockstep with the schema.

    If you add a new dimension to the schema, add the field name here so
    no fixture can be added that quietly omits it.
    """
    expected_subset = {
        "id",
        "dtype",
        "layout",
        "compression",
        "predictor",
        "photometric",
        "byte_order",
        "planar_config",
    }
    assert expected_subset.issubset(set(generate.REQUIRED_FIELDS))
