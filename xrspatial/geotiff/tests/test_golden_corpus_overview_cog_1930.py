"""Smoke tests for the Phase 2 PR 7 overview / COG fixtures (issue #1930).

The three fixtures land here:

* ``overview_internal_uint16`` -- internal-IFD overviews at [2, 4]
* ``overview_external_ovr_uint16`` -- sidecar `.ovr` overviews at [2, 4]
* ``cog_internal_overview_uint16`` -- COG layout (tiled + IFD-ordered)

Each fixture is rebuilt by the deterministic generator and shipped in
``golden_corpus/fixtures``. These tests assert the shape of what is on
disk and run the Phase 1 oracle against the base (level-0) image.

Oracle gap (intentional, tracked separately): the Phase 1 oracle in
``_oracle.compare_to_oracle`` reads only the base IFD via
``rasterio.open(...).read()``. It does not inspect overview IFDs or the
sidecar `.ovr`. A future PR (post Phase 1 PR 2) will add an
overview-aware comparison; until then, the smoke tests below pin the
on-disk shape and the base-image parity check is what runs through the
oracle.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import Affine  # noqa: E402

from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

INTERNAL_ID = "overview_internal_uint16"
EXTERNAL_ID = "overview_external_ovr_uint16"
COG_ID = "cog_internal_overview_uint16"


def _fixture_path(fixture_id: str) -> pathlib.Path:
    p = FIXTURES_DIR / f"{fixture_id}.tif"
    if not p.exists():
        pytest.skip(
            f"fixture {fixture_id} not generated; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )
    return p


def _manifest_entry(fixture_id: str) -> dict:
    manifest = generate.load_manifest()
    resolved = generate.validate(manifest)
    for entry in resolved:
        if entry["id"] == fixture_id:
            return entry
    raise AssertionError(f"fixture {fixture_id!r} missing from manifest")


def _candidate_from_rasterio(path: pathlib.Path) -> xr.DataArray:
    """Build a candidate DataArray by reading ``path`` via rasterio.

    Phase 3 will swap this for real xrspatial backends; here we use the
    rasterio read so the oracle has something concrete to compare. The
    test asserts the level-0 image only.
    """
    with rasterio.open(path) as src:
        data = src.read(1)
        t = src.transform
        crs_epsg = src.crs.to_epsg() if src.crs else None
        nodata = src.nodata
        height, width = data.shape
        pw = float(t.a)
        ph = float(t.e)
        ox = float(t.c)
        oy = float(t.f)
        x = ox + (np.arange(width) + 0.5) * pw
        y = oy + (np.arange(height) + 0.5) * ph
        attrs: dict = {"transform": (pw, 0.0, ox, 0.0, ph, oy)}
        if crs_epsg is not None:
            attrs["crs"] = crs_epsg
        if nodata is not None:
            attrs["nodata"] = nodata
        return xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x},
                            attrs=attrs)


# ---------------------------------------------------------------------------
# Internal overviews
# ---------------------------------------------------------------------------

def test_internal_overview_fixture_has_overviews():
    """Internal overview fixture exposes [2, 4] on its first band."""
    path = _fixture_path(INTERNAL_ID)
    with rasterio.open(path) as src:
        assert src.overviews(1) == [2, 4]
        assert src.count == 1
        assert src.dtypes[0] == "uint16"
    assert not path.with_suffix(path.suffix + ".ovr").exists(), (
        "internal-overview fixture must not ship a sidecar"
    )


def test_internal_overview_fixture_matches_oracle():
    """Level-0 image agrees with the rasterio reference read."""
    path = _fixture_path(INTERNAL_ID)
    cand = _candidate_from_rasterio(path)
    compare_to_oracle(path, cand)


def test_internal_overview_fixture_size_budget():
    path = _fixture_path(INTERNAL_ID)
    assert path.stat().st_size < 12 * 1024, (
        f"{path.name} exceeds the 12 KB per-fixture budget "
        f"({path.stat().st_size} bytes)"
    )


# ---------------------------------------------------------------------------
# External `.ovr` sidecar
# ---------------------------------------------------------------------------

def test_external_overview_sidecar_present():
    """External overview fixture ships with a `.tif.ovr` sidecar."""
    path = _fixture_path(EXTERNAL_ID)
    ovr = path.with_suffix(path.suffix + ".ovr")
    assert ovr.exists(), f"expected sidecar at {ovr}"


def test_external_overview_fixture_reports_overviews():
    """rasterio surfaces the sidecar overviews on the main file."""
    path = _fixture_path(EXTERNAL_ID)
    with rasterio.open(path) as src:
        assert src.overviews(1) == [2, 4]


def test_external_overview_fixture_matches_oracle():
    path = _fixture_path(EXTERNAL_ID)
    cand = _candidate_from_rasterio(path)
    compare_to_oracle(path, cand)


def test_external_overview_fixture_size_budget():
    path = _fixture_path(EXTERNAL_ID)
    ovr = path.with_suffix(path.suffix + ".ovr")
    for p in (path, ovr):
        assert p.stat().st_size < 12 * 1024, (
            f"{p.name} exceeds the 12 KB per-fixture budget "
            f"({p.stat().st_size} bytes)"
        )


# ---------------------------------------------------------------------------
# COG
# ---------------------------------------------------------------------------

def test_cog_fixture_is_tiled_with_overviews():
    """COG fixture is tiled and carries internal overviews."""
    path = _fixture_path(COG_ID)
    with rasterio.open(path) as src:
        # Tiled is detectable from block_shapes: a tiled raster has square
        # blocks that are not the full image width. We avoid the
        # deprecated ``src.is_tiled`` property here.
        block = src.block_shapes[0]
        assert block[0] == block[1] and block[0] < src.width, (
            f"COG fixture must be tiled with square blocks, got {block}"
        )
        assert src.overviews(1) == [2, 4]
    # No external sidecar should accompany the COG.
    assert not path.with_suffix(path.suffix + ".ovr").exists(), (
        "COG fixture must not ship an external .ovr sidecar"
    )


def test_cog_fixture_carries_cog_layout_marker():
    """The COG driver writes a ``LAYOUT=COG`` marker into IMAGE_STRUCTURE.

    The COG spec mandates IFD ordering (base image before overviews) and a
    leading ghost-IFD layout block. Rather than re-parse the TIFF header,
    we trust GDAL's own marker, which is the public artefact rasterio
    exposes. Phase 3 backends do the equivalent check before claiming a
    file is COG-shaped.
    """
    path = _fixture_path(COG_ID)
    with rasterio.open(path) as src:
        # rasterio reports the base image dimensions on open, not an overview.
        assert src.width == 64 and src.height == 64
        tags = src.tags(ns="IMAGE_STRUCTURE")
        assert tags.get("LAYOUT") == "COG", (
            f"expected IMAGE_STRUCTURE LAYOUT=COG, got tags={tags!r}"
        )


def test_cog_fixture_matches_oracle():
    path = _fixture_path(COG_ID)
    cand = _candidate_from_rasterio(path)
    compare_to_oracle(path, cand)


def test_cog_fixture_size_budget():
    path = _fixture_path(COG_ID)
    assert path.stat().st_size < 12 * 1024, (
        f"{path.name} exceeds the 12 KB per-fixture budget "
        f"({path.stat().st_size} bytes)"
    )


# ---------------------------------------------------------------------------
# Manifest schema coverage
# ---------------------------------------------------------------------------

def test_manifest_carries_three_overview_fixtures():
    """All three fixture ids are declared in the manifest and validate."""
    manifest = generate.load_manifest()
    resolved = generate.validate(manifest)
    ids = {e["id"] for e in resolved}
    assert {INTERNAL_ID, EXTERNAL_ID, COG_ID}.issubset(ids)

    internal = _manifest_entry(INTERNAL_ID)
    external = _manifest_entry(EXTERNAL_ID)
    cog = _manifest_entry(COG_ID)

    assert internal["overviews"] == [2, 4]
    assert internal.get("external_overview", False) is False
    assert internal.get("cog", False) is False

    assert external["overviews"] == [2, 4]
    assert external["external_overview"] is True
    assert external.get("cog", False) is False

    assert cog["overviews"] == [2, 4]
    assert cog["cog"] is True
    assert cog["layout"] == "tiled"


def test_validator_rejects_non_bool_cog_flag():
    """``cog`` must be a bool; non-bool entries raise ManifestError."""
    manifest = generate.load_manifest()
    defaults = manifest.get("defaults") or {}
    entry = dict(defaults)
    entry.update(manifest["fixtures"][0])
    entry["cog"] = "yes"
    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(generate.ManifestError, match="cog must be a bool"):
        generate.validate(bad)


def test_validator_rejects_cog_with_external_overview():
    """COG layout forbids external overviews (cogeo.org/spec)."""
    manifest = generate.load_manifest()
    defaults = manifest.get("defaults") or {}
    entry = dict(defaults)
    entry.update(manifest["fixtures"][0])
    entry["cog"] = True
    entry["external_overview"] = True
    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(
        generate.ManifestError,
        match="cog=true is incompatible with external_overview",
    ):
        generate.validate(bad)
