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


# Lossy cells (today: JPEG-YCbCr) can't compare pixels bit-exactly across
# libjpeg versions, but a per-band mean drift much beyond a few intensity
# units points at a real generator regression (wrong input array, swapped
# band order before encode) rather than codec noise. 4.0 on a 0-255 scale
# leaves several stops of headroom for libjpeg/YCbCr churn while still
# catching the kind of bug the rest of the test is meant to flag.
_LOSSY_PIXEL_MEAN_TOL = 4.0


def _assert_semantic_equal(
    committed: pathlib.Path,
    regenerated: pathlib.Path,
    entry: dict,
) -> None:
    """Compare two TIFFs by their rasterio read, not by file bytes.

    Used for ``cog`` and ``jpeg`` fixtures where the on-disk encoding
    is toolchain-coupled but the readable content is stable.
    Lossless cells assert bit-exact pixels at the base IFD and at
    every overview level the file declares. Lossy cells
    (``tolerance.lossy: true`` in the manifest, today the JPEG-YCbCr
    entry) drop to a coarse per-band mean tolerance instead of a
    bit-exact compare.
    """
    lossy = bool(entry.get("tolerance", {}).get("lossy", False))
    fid = entry["id"]
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
        ref_overviews = ref.overviews(1)
        assert ref_overviews == cand.overviews(1), (
            f"overview decimation factors differ: "
            f"committed={ref_overviews}, regenerated={cand.overviews(1)}"
        )

    if lossy:
        _assert_pixels_close_lossy(committed, regenerated, fid)
        return
    _assert_pixels_exact(committed, regenerated, fid)
    # Overview pixels are part of the determinism contract for fixtures
    # that ship them (the COG cell today). rasterio's OVERVIEW_LEVEL
    # is 0-indexed against the overview chain, hence range(len(...)).
    for level in range(len(ref_overviews)):
        _assert_overview_pixels_exact(committed, regenerated, level, fid)


def _read_all(path: pathlib.Path, *, overview_level: int | None = None) -> np.ndarray:
    """Open ``path`` with rasterio and return ``src.read()`` for the
    requested IFD. ``overview_level=None`` reads the base IFD.
    """
    if overview_level is None:
        with rasterio.open(path) as src:
            return src.read()
    with rasterio.open(path, OVERVIEW_LEVEL=overview_level) as src:
        return src.read()


def _assert_pixels_exact(
    committed: pathlib.Path, regenerated: pathlib.Path, fid: str,
) -> None:
    ref_pixels = _read_all(committed)
    cand_pixels = _read_all(regenerated)
    assert np.array_equal(ref_pixels, cand_pixels, equal_nan=True), (
        f"pixel arrays differ for {fid!r}; the generator output "
        f"no longer round-trips to the committed fixture's pixels"
    )


def _assert_overview_pixels_exact(
    committed: pathlib.Path,
    regenerated: pathlib.Path,
    overview_level: int,
    fid: str,
) -> None:
    ref_pixels = _read_all(committed, overview_level=overview_level)
    cand_pixels = _read_all(regenerated, overview_level=overview_level)
    assert np.array_equal(ref_pixels, cand_pixels, equal_nan=True), (
        f"overview level {overview_level} pixels differ for {fid!r}; "
        f"the generator's overview pyramid no longer matches the "
        f"committed fixture"
    )


def _assert_pixels_close_lossy(
    committed: pathlib.Path, regenerated: pathlib.Path, fid: str,
) -> None:
    """Coarse per-band mean comparison for lossy (JPEG) cells.

    Bit-exact comparison would re-introduce the libjpeg coupling this
    PR removed, but the per-band mean is stable enough across libjpeg
    versions to catch a real content regression (a swapped input
    array, a band-permutation bug) while tolerating ordinary codec
    drift.
    """
    ref_pixels = _read_all(committed).astype(np.float64)
    cand_pixels = _read_all(regenerated).astype(np.float64)
    # rasterio always returns (bands, H, W), so axis=(1, 2) collapses
    # to one mean per band.
    ref_means = ref_pixels.mean(axis=(1, 2))
    cand_means = cand_pixels.mean(axis=(1, 2))
    diff = np.abs(ref_means - cand_means)
    assert np.all(diff <= _LOSSY_PIXEL_MEAN_TOL), (
        f"per-band mean drift exceeds {_LOSSY_PIXEL_MEAN_TOL} for {fid!r}: "
        f"committed_means={ref_means.tolist()}, "
        f"regenerated_means={cand_means.tolist()}, "
        f"abs_diff={diff.tolist()}"
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
    out = tmp_path_factory.mktemp("regen_corpus_determinism")
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


def _write_doctored_copy(
    src: pathlib.Path, dst: pathlib.Path, *, delta: int = 1
) -> None:
    """Copy ``src`` to ``dst`` and flip one pixel by ``delta``.

    Used by the negative-path tests below: the resulting file has the
    same georeferencing and overview chain as the source but differs
    in pixel content, so a working semantic check must reject it.
    """
    with rasterio.open(src) as r:
        profile = r.profile
        data = r.read()
        overview_factors = r.overviews(1)
    data = data.copy()
    data[0, 0, 0] = (int(data[0, 0, 0]) + delta) & np.iinfo(data.dtype).max
    with rasterio.open(dst, "w", **profile) as w:
        w.write(data)
        if overview_factors:
            # Match the source's overview chain so the decimation check
            # passes and the comparison falls through to pixel reads.
            w.build_overviews(overview_factors)


def test_semantic_equal_rejects_lossless_pixel_drift(tmp_path) -> None:
    """A doctored lossless fixture with one flipped pixel must fail
    ``_assert_semantic_equal``. Locks the drift-detection path that the
    PR refactor depends on.
    """
    src = FIXTURES_DIR / "cog_internal_overview_uint16.tif"
    if not src.exists():
        pytest.skip("cog fixture not committed; cannot exercise drift path")
    doctored = tmp_path / "cog_doctored_2299.tif"
    _write_doctored_copy(src, doctored)
    entry = _ENTRY_BY_ID["cog_internal_overview_uint16"]
    with pytest.raises(AssertionError, match=r"pixels? .* differ"):
        _assert_semantic_equal(src, doctored, entry)


def test_semantic_equal_rejects_lossy_mean_drift(tmp_path) -> None:
    """A doctored lossy fixture with a large constant offset must fail
    the per-band mean check. Catches the case where the JPEG path
    would otherwise silently accept anything since pixel equality is
    skipped.
    """
    src = FIXTURES_DIR / "compression_jpeg_uint8_ycbcr.tif"
    if not src.exists():
        pytest.skip("jpeg fixture not committed; cannot exercise drift path")
    # Read the source, add a constant offset well past the mean
    # tolerance, then re-encode through the same profile so the
    # resulting file is still a valid JPEG-YCbCr TIFF.
    with rasterio.open(src) as r:
        profile = r.profile
        data = r.read()
    doctored = tmp_path / "jpeg_doctored_2299.tif"
    offset = int(_LOSSY_PIXEL_MEAN_TOL * 4) + 1
    shifted = np.clip(data.astype(np.int32) + offset, 0, 255).astype(data.dtype)
    with rasterio.open(doctored, "w", **profile) as w:
        w.write(shifted)
    entry = _ENTRY_BY_ID["compression_jpeg_uint8_ycbcr"]
    with pytest.raises(AssertionError, match="per-band mean drift"):
        _assert_semantic_equal(src, doctored, entry)


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
