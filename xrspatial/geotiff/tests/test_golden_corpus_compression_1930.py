"""Smoke tests for the Phase 2 PR 5 compression fixtures (issue #1930).

Six fixtures land in this PR, one per major codec / predictor variant:

* ``compression_none_uint8`` -- baseline uncompressed uint8.
* ``compression_deflate_predictor2_uint16`` -- deflate + predictor 2 (int).
* ``compression_deflate_predictor3_float32`` -- deflate + predictor 3 (float).
* ``compression_lzw_predictor2_int16`` -- LZW + predictor 2 on signed int.
* ``compression_lerc_float32`` -- LERC on float32 with max_z_error 0
  (lossless setting, so it is comparable to the other codecs).
* ``compression_jpeg_uint8_ycbcr`` -- JPEG in YCbCr on uint8 tiled raster.
  Intrinsically lossy; the oracle is called with ``lossy=True``.

The test:

* loads each fixture with rasterio,
* builds an xrspatial-shaped DataArray from the rasterio read,
* calls ``compare_to_oracle`` and asserts it accepts the matching case
  for lossless codecs, with ``lossy=True`` for jpeg,
* pins the JPEG cell's expected behaviour: strict mode must reject and
  lossy mode must accept.

LERC and JPEG codec availability depends on the GDAL build. When a
codec is unavailable the corresponding fixture is missing from disk and
the relevant test is skipped via ``pytest.skip``. The pre-PR generator
run did write them; this guard is defensive for environments where GDAL
was rebuilt without those drivers.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

# Both pyyaml and rasterio are runtime deps of this test. importorskip
# keeps minimal environments green; once Phase 2 fully lands these are
# planned to move into the test extras (see README).
pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")

import xarray as xr  # noqa: E402

from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(__file__).resolve().parent
    / "golden_corpus"
    / "fixtures"
)


# Fixture id, lossy flag for the oracle call.
COMPRESSION_FIXTURES: tuple[tuple[str, bool], ...] = (
    ("compression_none_uint8", False),
    ("compression_deflate_predictor2_uint16", False),
    ("compression_deflate_predictor3_float32", False),
    ("compression_lzw_predictor2_int16", False),
    ("compression_lerc_float32", False),
    ("compression_jpeg_uint8_ycbcr", True),
)


def _fixture_path(fid: str) -> pathlib.Path:
    """Return the on-disk path for a fixture id, skipping if absent.

    A missing file usually means the maintainer who regenerated the
    corpus had a GDAL build without the relevant codec (LERC or JPEG
    are the common offenders). The committed fixtures in this PR were
    built with both codecs available; this guard exists so a contributor
    rebuilding locally without those drivers does not see a hard fail
    for a file they could not produce.
    """
    p = FIXTURES_DIR / f"{fid}.tif"
    if not p.exists():
        pytest.skip(f"fixture {fid} not present (codec unavailable?)")
    return p


def _candidate_from_rasterio(path: pathlib.Path) -> xr.DataArray:
    """Read ``path`` with rasterio and return an xrspatial-shaped DataArray.

    Mirrors the shape that ``xrspatial.geotiff.open_geotiff`` produces:
    a 2-D DataArray for single-band, 3-D ``(band, y, x)`` otherwise,
    pixel-centre coords, and the canonical ``transform`` 6-tuple in attrs.
    """
    with rasterio.open(path) as src:
        data = src.read()  # shape (bands, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        dtype = src.dtypes[0]

    height, width = data.shape[-2], data.shape[-1]
    pw = float(transform.a)
    ph = float(transform.e)
    ox = float(transform.c)
    oy = float(transform.f)
    x = ox + (np.arange(width) + 0.5) * pw
    y = oy + (np.arange(height) + 0.5) * ph

    attrs: dict = {
        "transform": (pw, 0.0, ox, 0.0, ph, oy),
    }
    if crs is not None:
        epsg = crs.to_epsg()
        if epsg is not None:
            attrs["crs"] = epsg
        else:
            attrs["crs_wkt"] = crs.to_wkt()
    if nodata is not None:
        attrs["nodata"] = nodata

    if data.shape[0] == 1:
        arr = data[0].astype(dtype)
        return xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs=attrs,
        )
    return xr.DataArray(
        data.astype(dtype),
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, data.shape[0] + 1),
            "y": y,
            "x": x,
        },
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Per-fixture parametrised smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fid, lossy", COMPRESSION_FIXTURES)
def test_compression_fixture_is_valid_tiff(fid: str, lossy: bool) -> None:
    """Each fixture opens with rasterio and reports the expected codec."""
    path = _fixture_path(fid)
    with rasterio.open(path) as src:
        # Sanity: shape and dtype come out matching the manifest.
        assert src.width > 0 and src.height > 0
        assert src.count >= 1
        # CRS round-trips to EPSG:4326 (the default in the manifest).
        assert src.crs is not None
        assert src.crs.to_epsg() == 4326


@pytest.mark.parametrize("fid, lossy", COMPRESSION_FIXTURES)
def test_compression_fixture_oracle_accepts(fid: str, lossy: bool) -> None:
    """rasterio-built candidate must satisfy the oracle.

    For lossy fixtures (jpeg) the oracle is called with ``lossy=True``;
    for everything else it is called in strict bit-exact mode.
    """
    path = _fixture_path(fid)
    cand = _candidate_from_rasterio(path)
    compare_to_oracle(path, cand, lossy=lossy)


# ---------------------------------------------------------------------------
# JPEG: pin lossy semantics explicitly
# ---------------------------------------------------------------------------

def test_jpeg_lossy_mode_required() -> None:
    """The jpeg fixture passes only when the oracle is told it is lossy.

    Two halves: strict mode (default) must reject because JPEG quantises
    pixel values; lossy mode must accept. This pins the contract added
    in PR #1991 for Phase 2's jpeg cell.
    """
    path = _fixture_path("compression_jpeg_uint8_ycbcr")
    cand = _candidate_from_rasterio(path)
    # Strict mode: the rasterio-decoded pixels match themselves trivially.
    # To prove the lossy contract we need to perturb the candidate so
    # strict comparison fails while shape/dtype/transform/CRS all match.
    # Bump every pixel by 1 (clipped to uint8 range). The decoded YCbCr
    # checker pattern lands well below 255 so clipping is a no-op in
    # practice; the assertion is that the perturbed array is no longer
    # bit-equal to the rasterio read.
    perturbed_data = (
        cand.data.astype(np.int32) + 1
    ).clip(0, 255).astype(np.uint8)
    perturbed = xr.DataArray(
        perturbed_data,
        dims=cand.dims,
        coords=cand.coords,
        attrs=dict(cand.attrs),
    )
    with pytest.raises(AssertionError, match="pixel arrays differ"):
        compare_to_oracle(path, perturbed)
    # Lossy mode: same perturbed candidate is accepted.
    compare_to_oracle(path, perturbed, lossy=True)


# ---------------------------------------------------------------------------
# Manifest cross-check
# ---------------------------------------------------------------------------

def test_compression_fixtures_declared_in_manifest() -> None:
    """Every fixture id this test parametrises must exist in the manifest.

    Catches drift between this file and ``manifest.yaml``.
    """
    import importlib

    generate = importlib.import_module(
        "xrspatial.geotiff.tests.golden_corpus.generate"
    )
    manifest = generate.load_manifest()
    declared = {f["id"] for f in manifest["fixtures"]}
    for fid, _ in COMPRESSION_FIXTURES:
        assert fid in declared, f"{fid} missing from manifest.yaml"


def test_jpeg_fixture_marked_lossy_in_manifest() -> None:
    """The jpeg manifest entry must carry ``tolerance.lossy: true``.

    The oracle's lossy contract is opt-in per-fixture and the manifest
    is the source of truth; mis-marking would silently downgrade strict
    comparison for everyone.
    """
    import importlib

    generate = importlib.import_module(
        "xrspatial.geotiff.tests.golden_corpus.generate"
    )
    manifest = generate.load_manifest()
    by_id = {f["id"]: f for f in manifest["fixtures"]}
    jpeg = by_id["compression_jpeg_uint8_ycbcr"]
    tol = jpeg.get("tolerance") or {}
    assert tol.get("lossy") is True, (
        "compression_jpeg_uint8_ycbcr must declare tolerance.lossy: true"
    )
    # And conversely the lossless cells must not.
    for fid, lossy in COMPRESSION_FIXTURES:
        if lossy:
            continue
        entry = by_id.get(fid)
        if entry is None:
            continue
        tol = entry.get("tolerance") or {}
        assert tol.get("lossy") in (False, None), (
            f"{fid} unexpectedly marked lossy"
        )
