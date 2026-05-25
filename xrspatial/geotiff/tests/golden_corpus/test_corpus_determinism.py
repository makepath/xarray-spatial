"""Corpus determinism gate (issue #1930, refined in #2299).

The golden corpus generator is built to be reproducible: fixed seeds,
sorted iteration, and an explicit ``os.utime`` pass that pins file
mtimes to a constant epoch. This test guards that property in CI so a
regression in the generator (or a manually-edited fixture on disk)
fails the build instead of silently drifting.

What this catches:

* a generator-side change that flips RNG ordering, drops the mtime
  normalisation, or otherwise breaks reproducibility -- the
  regenerated output diverges from the committed fixture;
* a fixture-on-disk drift where the manifest still says X but the
  committed ``.tif`` was edited (or stale) so it no longer matches
  what the manifest would produce.

How the comparison runs (issue #2299):

Most fixtures use a byte-level md5 check. That is the strictest signal
available and catches generator-side regressions immediately.

Two fixture classes are intrinsically coupled to the GDAL / libjpeg
build used to produce them: COG (``cog: true``) and JPEG
(``compression: jpeg``). The COG driver's overview pyramid layout and
libjpeg's encoder output both change across versions even when the
input pixels are identical, so a byte-md5 check against a fixture
committed by one toolchain will fail under another (e.g. a developer
laptop vs. the conda-forge CI lane). For those fixtures the test falls
back to a semantic comparison: open both files with rasterio and
verify dtype, shape, transform, CRS, nodata, and -- for lossless cells
-- pixel arrays. JPEG cells carry ``tolerance.lossy: true`` in the
manifest and skip the pixel comparison.

Fixtures the manifest declares but that are not committed on disk
(today: ``example_tiled_uint16_deflate_pred2``, kept as a
schema-illustrating example) are skipped here rather than failing,
mirroring how the per-backend tests handle the same case.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pytest

# rasterio / pyyaml are runtime deps of the generator. importorskip
# keeps minimal environments green by skipping the whole module when
# either is missing.
pytest.importorskip("yaml")
pytest.importorskip("rasterio")

import rasterio  # noqa: E402

from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402

FIXTURES_DIR = (
    pathlib.Path(__file__).resolve().parent / "fixtures"
)


def _md5(path: pathlib.Path) -> str:
    # ``usedforsecurity=False`` (Python 3.9+) keeps this working on
    # FIPS-strict runners where ``hashlib.md5()`` otherwise raises a
    # ValueError. Byte-identity comparison only, no security claim.
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_encoder_sensitive(entry: dict) -> bool:
    """Fixtures whose on-disk bytes depend on the encoder build.

    The COG driver and libjpeg both produce output that varies across
    versions even when the input pixels are identical. Byte-md5 across
    different builds is therefore unstable for those cells, while a
    semantic read still round-trips correctly.
    """
    return bool(entry.get("cog")) or entry.get("compression") == "jpeg"


def _nodata_equal(a, b) -> bool:
    """NaN-aware nodata comparison.

    Both None counts as equal. NaN sentinels compare equal even though
    ``float('nan') != float('nan')`` in plain Python.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    if np.isnan(af) and np.isnan(bf):
        return True
    return af == bf


def _assert_semantic_equal(
    committed: pathlib.Path,
    regenerated: pathlib.Path,
    entry: dict,
) -> None:
    """Compare two TIFFs by their rasterio read, not by file bytes.

    Used for ``cog`` and ``jpeg`` fixtures where the on-disk encoding
    is toolchain-coupled but the readable content is stable.
    Lossy cells (``tolerance.lossy: true`` in the manifest, today the
    JPEG-YCbCr entry) skip pixel equality and check only shape, dtype,
    georeferencing, and nodata.
    """
    lossy = bool(entry.get("tolerance", {}).get("lossy", False))
    with rasterio.open(committed) as ref, rasterio.open(regenerated) as cand:
        assert ref.count == cand.count, (
            f"band count differs: committed={ref.count}, "
            f"regenerated={cand.count}"
        )
        assert ref.dtypes == cand.dtypes, (
            f"dtypes differ: committed={ref.dtypes}, "
            f"regenerated={cand.dtypes}"
        )
        assert (ref.width, ref.height) == (cand.width, cand.height), (
            f"shape differs: committed={(ref.width, ref.height)}, "
            f"regenerated={(cand.width, cand.height)}"
        )
        assert tuple(ref.transform) == tuple(cand.transform), (
            f"transform differs: committed={tuple(ref.transform)}, "
            f"regenerated={tuple(cand.transform)}"
        )
        assert ref.crs == cand.crs, (
            f"CRS differs: committed={ref.crs!r}, regenerated={cand.crs!r}"
        )
        assert _nodata_equal(ref.nodata, cand.nodata), (
            f"nodata differs: committed={ref.nodata!r}, "
            f"regenerated={cand.nodata!r}"
        )
        assert ref.overviews(1) == cand.overviews(1), (
            f"overview decimation factors differ: "
            f"committed={ref.overviews(1)}, regenerated={cand.overviews(1)}"
        )
        if lossy:
            return
        ref_pixels = ref.read()
        cand_pixels = cand.read()
        assert np.array_equal(ref_pixels, cand_pixels, equal_nan=True), (
            f"pixel arrays differ for {entry['id']!r}; the generator output "
            f"no longer round-trips to the committed fixture's pixels"
        )


def _load_entries() -> list[dict]:
    """Return validated manifest entries (defaults merged), sorted by id."""
    return sorted(generate.validate(generate.load_manifest()), key=lambda e: e["id"])


# Cached at import so parametrize collection and the orphan-file test
# share one manifest load. Each call to ``validate()`` re-walks every
# entry, so collapsing the two callers cuts validation work in half.
_ENTRIES = _load_entries()
_ENTRY_BY_ID = {e["id"]: e for e in _ENTRIES}
_MANIFEST_IDS = [e["id"] for e in _ENTRIES]
_EXTERNAL_OVR_IDS = [e["id"] for e in _ENTRIES if e.get("external_overview")]


@pytest.fixture(scope="module")
def regenerated_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Regenerate the entire corpus into a module-scoped tmp dir.

    Module-scoped so the (few-second) write cost is paid once per
    test session rather than per parametrised case.
    """
    out = tmp_path_factory.mktemp("regen_corpus_1930")
    generate.generate(output_dir=out)
    return out


@pytest.mark.parametrize("fixture_id", _MANIFEST_IDS)
def test_fixture_bytes_are_deterministic(
    fixture_id: str, regenerated_dir: pathlib.Path
) -> None:
    """The committed ``.tif`` for each manifest id matches what the
    generator would produce today.

    For most fixtures the check is byte-md5: the strictest guard
    against generator drift. For ``cog`` and ``jpeg`` fixtures the
    GDAL / libjpeg encoder output is toolchain-coupled, so the check
    falls back to a semantic comparison via rasterio (see
    ``_assert_semantic_equal``). Issue #2299 has the rationale.

    Skip rather than fail when the committed file is missing -- that
    means the fixture is declared but intentionally not shipped
    (e.g. the schema-illustrating example fixture). The per-backend
    tests handle this the same way.
    """
    committed = FIXTURES_DIR / f"{fixture_id}.tif"
    if not committed.exists():
        pytest.skip(
            f"fixture {fixture_id!r} is in the manifest but not committed "
            f"on disk; nothing to compare"
        )
    regenerated = regenerated_dir / f"{fixture_id}.tif"
    assert regenerated.exists(), (
        f"generator did not produce {fixture_id!r}; check the generator "
        f"and manifest stayed in sync"
    )
    entry = _ENTRY_BY_ID[fixture_id]
    if _is_encoder_sensitive(entry):
        _assert_semantic_equal(committed, regenerated, entry)
        return
    committed_md5 = _md5(committed)
    regenerated_md5 = _md5(regenerated)
    assert committed_md5 == regenerated_md5, (
        f"fixture {fixture_id!r} drifted: committed md5 {committed_md5} "
        f"does not match regenerated md5 {regenerated_md5}. Either the "
        f"generator changed and the committed fixtures need re-running "
        f"(`python -m xrspatial.geotiff.tests.golden_corpus.generate`), "
        f"or the committed fixture was edited out of band."
    )


@pytest.mark.parametrize("fixture_id", _EXTERNAL_OVR_IDS or ["__none__"])
def test_external_overview_sidecar_is_deterministic(
    fixture_id: str, regenerated_dir: pathlib.Path
) -> None:
    """Fixtures with ``external_overview: true`` ship a sidecar
    ``<id>.tif.ovr`` next to the main ``.tif``. The sidecar bytes are
    part of the determinism contract too.

    Iterates over every manifest entry with ``external_overview=True``
    so a future fixture lands in this test automatically without a
    code change. When no such entry exists today the placeholder id
    short-circuits to a skip so pytest still reports a single
    informative case.
    """
    if fixture_id == "__none__":
        pytest.skip("manifest has no external_overview fixtures today")
    sidecar_name = f"{fixture_id}.tif.ovr"
    committed = FIXTURES_DIR / sidecar_name
    if not committed.exists():
        pytest.skip(
            f"sidecar {sidecar_name!r} is not committed; nothing to compare"
        )
    regenerated = regenerated_dir / sidecar_name
    assert regenerated.exists(), (
        f"generator did not produce {sidecar_name!r}; external_overview "
        f"path may be broken"
    )
    assert _md5(committed) == _md5(regenerated), (
        f"{sidecar_name!r} drifted from the committed bytes; rerun the "
        f"generator and recommit, or revert the on-disk edit"
    )


def test_no_orphan_fixtures_on_disk() -> None:
    """Every committed ``.tif`` (and ``.tif.ovr`` sidecar) corresponds
    to a manifest entry. Catches stale fixtures left behind after a
    manifest delete.
    """
    manifest_ids = set(_MANIFEST_IDS)
    orphans: list[str] = []
    for path in sorted(FIXTURES_DIR.glob("*.tif")):
        if path.stem not in manifest_ids:
            orphans.append(path.name)
    for path in sorted(FIXTURES_DIR.glob("*.tif.ovr")):
        # sidecar stem: strip ``.tif`` to recover the fixture id
        fid = path.name[: -len(".tif.ovr")]
        if fid not in manifest_ids:
            orphans.append(path.name)
    assert not orphans, (
        f"committed fixtures {orphans!r} have no matching manifest entry; "
        f"either re-add them to the manifest or remove the orphan files"
    )
