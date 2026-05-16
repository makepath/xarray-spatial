"""Smoke test for the layout x byte_order fixture batch (Phase 2 PR 3 of #1930).

Verifies that the four fixtures added by PR 3 exist on disk, parse as
TIFFs with the on-disk properties the manifest promises (layout flag and
byte-order magic), and round-trip through ``_oracle.compare_to_oracle``
as an identity check. The identity round-trip is a thin assertion that
the oracle accepts these fixtures; the real backend-vs-oracle parity
checks land in Phase 3.
"""

from __future__ import annotations

import importlib
import pathlib

import numpy as np
import pytest

pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")

from xrspatial.geotiff.tests.golden_corpus import _oracle

generate = importlib.import_module(
    "xrspatial.geotiff.tests.golden_corpus.generate"
)

FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

# (id, expected on-disk tiled flag, expected first two bytes)
LAYOUT_ENDIAN_FIXTURES = [
    ("stripped_le_uint16", False, b"II"),
    ("tiled_le_uint16", True, b"II"),
    ("stripped_be_uint16", False, b"MM"),
    ("tiled_be_uint16", True, b"MM"),
]


@pytest.mark.parametrize("fid,expected_tiled,expected_magic", LAYOUT_ENDIAN_FIXTURES)
def test_fixture_exists_and_is_tiff(fid, expected_tiled, expected_magic):
    """File is on disk, opens with rasterio, and has the promised on-disk layout/endianness."""
    path = FIXTURES_DIR / f"{fid}.tif"
    assert path.exists(), f"missing fixture {path}"
    with path.open("rb") as f:
        assert f.read(2) == expected_magic, (
            f"{fid}: byte-order magic mismatch (expected {expected_magic!r})"
        )
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint16"
        assert src.shape == (32, 32)
        assert src.is_tiled is expected_tiled, (
            f"{fid}: on-disk tiled flag mismatch"
        )


@pytest.mark.parametrize("fid,_tiled,_magic", LAYOUT_ENDIAN_FIXTURES)
def test_fixture_round_trips_through_oracle(fid, _tiled, _magic):
    """A rasterio-built DataArray of the fixture passes compare_to_oracle.

    This is an identity check: we feed the oracle the same bytes it reads
    internally. It asserts only that the oracle accepts the fixture shape
    (dtype, transform, CRS, nodata, pixels) for the four new variants;
    Phase 3 adds the real backend-vs-oracle parity tests.
    """
    xr = pytest.importorskip("xarray")
    path = FIXTURES_DIR / f"{fid}.tif"
    with rasterio.open(path) as src:
        pixels = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        # Pixel-centre coords matching what xrspatial would synthesise.
        ys = np.array(
            [transform.f + (i + 0.5) * transform.e for i in range(src.height)],
            dtype=np.float64,
        )
        xs = np.array(
            [transform.c + (i + 0.5) * transform.a for i in range(src.width)],
            dtype=np.float64,
        )

    da = xr.DataArray(
        pixels,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        attrs={
            "transform": tuple(transform)[:6],
            "crs": crs.to_epsg() if crs is not None else None,
            "nodata": nodata,
            "dtype": str(pixels.dtype),
        },
    )

    # No exception is the assertion.
    _oracle.compare_to_oracle(path, da)
