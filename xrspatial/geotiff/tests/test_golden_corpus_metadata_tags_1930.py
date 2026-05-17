"""Smoke tests for the GDAL_METADATA + extra_tags golden-corpus fixtures.

Phase 2 PR 9 of issue #1930 adds two fixtures:

* ``gdal_metadata_namespaced_uint16`` -- a stripped uint16 raster with
  GDAL_METADATA entries spread across the default domain, the
  ``IMAGE_STRUCTURE`` and ``SUBDATASETS`` GDAL domains, and a custom user
  domain. This exercises the cross-domain key handling that the xrspatial
  reader has to navigate.
* ``extra_tags_uint16`` -- a stripped uint16 raster with arbitrary TIFF
  tags (``ImageDescription``, ``Software``, ``Artist``, and a private
  numeric tag ``65000``) written as real IFD entries. The generator routes
  these through a tifffile post-pass because rasterio cannot emit private
  numeric tags via its writer.

These tests confirm the fixtures land on disk in the expected shape and
that an xarray DataArray built from a plain rasterio read still satisfies
``compare_to_oracle`` -- i.e. the metadata layer does not break the parity
contract.

The full canonical-attrs contract for GDAL_METADATA / pass-through tags
lands in issue #1984; this PR does not assert it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")
tifffile = pytest.importorskip("tifffile")

from xrspatial.geotiff.tests.golden_corpus import generate as _generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = (
    Path(_generate.__file__).resolve().parent / "fixtures"
)


def _read_as_dataarray(path: Path) -> xr.DataArray:
    """Build an xrspatial-shaped DataArray from a rasterio read.

    Mirrors what a numpy backend would produce: 2-D for single-band rasters,
    pixel-centre y/x coords, and an ``attrs`` dict carrying the canonical
    keys the oracle inspects (``transform``, ``crs``, ``nodata``).
    """
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        width, height = src.width, src.height

    pw, ph = float(transform.a), float(transform.e)
    ox, oy = float(transform.c), float(transform.f)
    x = ox + (np.arange(width) + 0.5) * pw
    y = oy + (np.arange(height) + 0.5) * ph

    attrs: dict = {"transform": (pw, 0.0, ox, 0.0, ph, oy)}
    if crs is not None:
        epsg = crs.to_epsg()
        if epsg is not None:
            attrs["crs"] = epsg
        else:
            attrs["crs_wkt"] = crs.to_wkt()
    if nodata is not None:
        attrs["nodata"] = nodata

    if data.shape[0] == 1:
        return xr.DataArray(
            data[0],
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs=attrs,
        )
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, data.shape[0] + 1), "y": y, "x": x},
        attrs=attrs,
    )


def _ensure_fixture(fixture_id: str) -> Path:
    """Regenerate ``fixture_id`` if its .tif file is missing."""
    path = _FIXTURE_DIR / f"{fixture_id}.tif"
    if not path.exists():
        _generate.generate(only=[fixture_id], output_dir=_FIXTURE_DIR)
    return path


# ---------------------------------------------------------------------------
# gdal_metadata_namespaced_uint16
# ---------------------------------------------------------------------------

def test_gdal_metadata_namespaced_default_domain_present() -> None:
    """Default-domain GDAL_METADATA entries survive the write."""
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    with rasterio.open(path) as src:
        tags = src.tags()  # default domain
    assert tags.get("UNITS") == "meters"
    assert tags.get("AREA_OR_POINT") == "Area"


def test_gdal_metadata_namespaced_image_structure_domain_present() -> None:
    """IMAGE_STRUCTURE is a GDAL-managed domain. We do not own the values
    GDAL writes there (it overrides INTERLEAVE / COMPRESSION from the
    actual codec / planar config), but the domain must be non-empty so the
    reader's cross-domain enumeration has a real entry to walk.
    """
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    with rasterio.open(path) as src:
        struct = src.tags(ns="IMAGE_STRUCTURE")
    assert struct, (
        f"IMAGE_STRUCTURE domain must be non-empty, got {struct!r}"
    )
    # INTERLEAVE is always present for raster TIFFs.
    assert "INTERLEAVE" in struct


def test_gdal_metadata_namespaced_subdatasets_domain_present() -> None:
    """SUBDATASETS entries written by the generator round-trip via rasterio."""
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    with rasterio.open(path) as src:
        subs = src.tags(ns="SUBDATASETS")
    assert subs.get("SUBDATASET_1_NAME") == "fixture:band1"
    assert subs.get("SUBDATASET_1_DESC") == "single-band view"


def test_gdal_metadata_namespaced_custom_domain_present() -> None:
    """User-defined GDAL domains round-trip key-for-key."""
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    with rasterio.open(path) as src:
        custom = src.tags(ns="CUSTOM_DOMAIN")
    assert custom.get("owner") == "xrspatial-corpus"
    assert custom.get("purpose") == "cross-domain metadata smoke"


def test_gdal_metadata_namespaced_passes_oracle() -> None:
    """A plain rasterio-derived DataArray satisfies the oracle.

    The oracle inspects crs / transform / nodata / dtype + pixels; GDAL
    metadata is intentionally pass-through, so it must not break parity.
    """
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    cand = _read_as_dataarray(path)
    compare_to_oracle(path, cand)


# ---------------------------------------------------------------------------
# extra_tags_uint16
# ---------------------------------------------------------------------------

_EXPECTED_EXTRA_TAGS = {
    270: "xrspatial golden corpus fixture",  # ImageDescription
    305: "xrspatial-golden-corpus",          # Software
    315: "xarray-contrib",                    # Artist
    65000: "private-tag-payload",             # private numeric tag
}


def test_extra_tags_uint16_real_tiff_tags_present() -> None:
    """Every requested extra tag lands as a real IFD entry, not in GDAL_METADATA.

    Walks the IFD via tifffile because rasterio surfaces well-known tags
    under TIFFTAG_* but hides private numeric tags. The test checks the
    raw tag codes so a regression that silently rerouted entries through
    ``GDAL_METADATA`` XML would fail here.
    """
    path = _ensure_fixture("extra_tags_uint16")
    with tifffile.TiffFile(path) as t:
        page_tags = {tag.code: tag.value for tag in t.pages[0].tags}
    for code, expected in _EXPECTED_EXTRA_TAGS.items():
        assert code in page_tags, (
            f"expected TIFF tag {code} not present on the primary IFD; "
            f"got tag codes {sorted(page_tags)}"
        )
        assert page_tags[code] == expected, (
            f"tag {code} value mismatch: got {page_tags[code]!r}, "
            f"expected {expected!r}"
        )


def test_extra_tags_uint16_rasterio_surfaces_known_names() -> None:
    """The three well-known names show up in ``src.tags()`` (default domain).

    rasterio rewrites well-known TIFF tag codes into the ``TIFFTAG_*``
    namespace when it reads the default-domain tag dict; private numeric
    codes are not surfaced here and are checked by the tifffile test above.
    """
    path = _ensure_fixture("extra_tags_uint16")
    with rasterio.open(path) as src:
        tags = src.tags()
    assert tags.get("TIFFTAG_IMAGEDESCRIPTION") == _EXPECTED_EXTRA_TAGS[270]
    assert tags.get("TIFFTAG_SOFTWARE") == _EXPECTED_EXTRA_TAGS[305]
    assert tags.get("TIFFTAG_ARTIST") == _EXPECTED_EXTRA_TAGS[315]


def test_extra_tags_uint16_passes_oracle() -> None:
    """Adding TIFF tags does not perturb the oracle's pixel / georef checks."""
    path = _ensure_fixture("extra_tags_uint16")
    cand = _read_as_dataarray(path)
    compare_to_oracle(path, cand)


def test_extra_tags_uint16_size_under_4kb() -> None:
    """The fixture stays under the 4 KB per-file size budget."""
    path = _ensure_fixture("extra_tags_uint16")
    size = path.stat().st_size
    assert size < 4096, (
        f"extra_tags_uint16.tif grew past the 4 KB budget: {size} bytes"
    )


def test_gdal_metadata_namespaced_size_under_4kb() -> None:
    """The fixture stays under the 4 KB per-file size budget."""
    path = _ensure_fixture("gdal_metadata_namespaced_uint16")
    size = path.stat().st_size
    assert size < 4096, (
        f"gdal_metadata_namespaced_uint16.tif grew past the 4 KB budget: "
        f"{size} bytes"
    )


# ---------------------------------------------------------------------------
# Validator guard rails
# ---------------------------------------------------------------------------

def _minimal_entry(defaults: dict) -> dict:
    """Build a manifest entry minus extra_tags / gdal_metadata for guard tests."""
    base = dict(defaults)
    base.update(
        id="validator_smoke",
        description="validator smoke",
        width=16,
        height=16,
        dtype="uint16",
        blocksize=16,
    )
    return base


def test_validator_rejects_bool_extra_tags_key() -> None:
    """`bool` subclasses `int` so the isinstance check has to filter it out."""
    manifest = _generate.load_manifest()
    defaults = manifest.get("defaults") or {}
    entry = _minimal_entry(defaults)
    entry["extra_tags"] = {True: "boom"}
    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(_generate.ManifestError, match="strings or ints"):
        _generate.validate(bad)


def test_validator_rejects_non_string_gdal_metadata_domain() -> None:
    """Domain keys must be strings; the YAML loader can produce ints."""
    manifest = _generate.load_manifest()
    defaults = manifest.get("defaults") or {}
    entry = _minimal_entry(defaults)
    entry["gdal_metadata"] = {123: {"k": "v"}}
    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(_generate.ManifestError, match="domain keys must be strings"):
        _generate.validate(bad)
