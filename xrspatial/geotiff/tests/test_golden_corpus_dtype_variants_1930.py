"""Smoke test for Phase 2 PR 4 dtype fixtures (issue #1930).

This test pins the eight dtype fixtures added in Phase 2 PR 4:
``dtype_int8``, ``dtype_uint8``, ``dtype_int16``, ``dtype_uint16``,
``dtype_int32``, ``dtype_uint32``, ``dtype_float32``, ``dtype_float64``.

It checks three things per fixture:

* the ``.tif`` file is present on disk under
  ``golden_corpus/fixtures/`` (i.e. the generator committed output);
* ``rasterio.open`` reports the dtype the manifest declared;
* the oracle (`compare_to_oracle`) accepts a candidate DataArray built
  straight from the rasterio read. This is a trivial identity check --
  no backend wiring lives here (that is Phase 3) -- but it confirms the
  oracle understands every dtype the corpus now ships.

For the integer dtypes the test also verifies that the four corner
pixels carry the dtype's min / max sentinels, since that is the whole
point of the ``noise_with_corners`` pixel pattern this PR adds.
"""
from __future__ import annotations

import importlib
import pathlib

import numpy as np
import pytest

pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")

import xarray as xr  # noqa: E402

from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)

generate = importlib.import_module(
    "xrspatial.geotiff.tests.golden_corpus.generate"
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

DTYPE_IDS = (
    ("dtype_int8", "int8"),
    ("dtype_uint8", "uint8"),
    ("dtype_int16", "int16"),
    ("dtype_uint16", "uint16"),
    ("dtype_int32", "int32"),
    ("dtype_uint32", "uint32"),
    ("dtype_float32", "float32"),
    ("dtype_float64", "float64"),
)


def _candidate_from_rasterio(src) -> xr.DataArray:
    """Build a DataArray that mirrors what a parity-correct reader would emit.

    The oracle compares attrs (transform / crs / nodata / dtype) and the
    pixel array. This helper round-trips the rasterio read into the same
    shape, so the oracle's accept/reject decision rests entirely on
    whether it can handle the dtype -- which is what this smoke test is
    pinning.
    """
    pixels = src.read(1)
    transform = src.transform
    attrs: dict = {
        "transform": (
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        ),
    }
    crs = src.crs
    if crs is not None:
        epsg = crs.to_epsg()
        if epsg is not None:
            attrs["crs"] = epsg
        else:
            attrs["crs_wkt"] = crs.to_wkt()
    if src.nodata is not None:
        attrs["nodata"] = src.nodata
    width = pixels.shape[-1]
    height = pixels.shape[-2]
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    return xr.DataArray(
        pixels,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs=attrs,
    )


@pytest.mark.parametrize("fixture_id, expected_dtype", DTYPE_IDS)
def test_dtype_fixture_exists_and_dtype_matches(
    fixture_id: str, expected_dtype: str,
) -> None:
    """The generator must have written the file with the manifest's dtype."""
    path = FIXTURES_DIR / f"{fixture_id}.tif"
    assert path.exists(), (
        f"missing fixture {path}; rerun "
        "`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
    )
    # 4 kB hard cap per PR 4 plan; the corpus must stay tiny in git.
    assert path.stat().st_size < 4 * 1024, (
        f"{path.name} is {path.stat().st_size} bytes; "
        "dtype fixtures must stay under 4 kB"
    )
    with rasterio.open(path) as src:
        assert src.dtypes[0] == expected_dtype, (
            f"{fixture_id}: rasterio dtype {src.dtypes[0]!r} != "
            f"expected {expected_dtype!r}"
        )


@pytest.mark.parametrize("fixture_id, expected_dtype", DTYPE_IDS)
def test_oracle_accepts_dtype_fixture(
    fixture_id: str, expected_dtype: str,
) -> None:
    """The oracle accepts a rasterio-read DataArray for every dtype.

    This is the dtype-level smoke check the plan asks for: it does not
    test any xrspatial backend (Phase 3) -- it confirms the oracle
    understands every dtype the corpus now ships.
    """
    path = FIXTURES_DIR / f"{fixture_id}.tif"
    with rasterio.open(path) as src:
        cand = _candidate_from_rasterio(src)
    assert np.dtype(cand.dtype) == np.dtype(expected_dtype)
    compare_to_oracle(path, cand)


@pytest.mark.parametrize(
    "fixture_id, expected_dtype",
    [(fid, dt) for fid, dt in DTYPE_IDS if not dt.startswith("float")],
)
def test_int_dtype_fixture_has_corner_sentinels(
    fixture_id: str, expected_dtype: str,
) -> None:
    """Integer fixtures plant min/max sentinels in the four corner pixels.

    Mirrors the ``noise_with_corners`` contract added to ``generate.py``
    in this PR. If somebody ever changes the corner-stamping logic this
    test will tell us.
    """
    info = np.iinfo(np.dtype(expected_dtype))
    path = FIXTURES_DIR / f"{fixture_id}.tif"
    with rasterio.open(path) as src:
        pixels = src.read(1)
    assert pixels[0, 0] == info.min, fixture_id
    assert pixels[0, -1] == info.max, fixture_id
    assert pixels[-1, 0] == info.max, fixture_id
    assert pixels[-1, -1] == info.min, fixture_id


def test_noise_with_corners_rejects_tiny_rasters() -> None:
    """``noise_with_corners`` needs >=2x2 so the four corners are distinct.

    The validator should refuse a 1x1 fixture with that pattern instead of
    silently collapsing all four corner stamps into the same pixel.
    """
    manifest = generate.load_manifest()
    defaults = manifest.get("defaults") or {}
    entry = dict(defaults)
    entry.update({
        "id": "tiny_corner_bad",
        "description": "1x1 noise_with_corners must be rejected.",
        "width": 1,
        "height": 1,
        "dtype": "uint8",
        "pixel_pattern": "noise_with_corners",
    })
    bad = {"version": 1, "defaults": {}, "fixtures": [entry]}
    with pytest.raises(generate.ManifestError, match="noise_with_corners"):
        generate.validate(bad)


def test_all_eight_dtype_fixtures_in_manifest() -> None:
    """The eight ids are present in the manifest with the expected dtypes."""
    manifest = generate.load_manifest()
    resolved = {e["id"]: e for e in generate.validate(manifest)}
    for fid, dt in DTYPE_IDS:
        assert fid in resolved, f"manifest missing {fid}"
        assert resolved[fid]["dtype"] == dt, (
            f"{fid}: manifest dtype {resolved[fid]['dtype']!r} != {dt!r}"
        )
