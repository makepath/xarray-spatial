"""Release gate: local GeoTIFF read (epic #2340).

This test pins the most basic promise the GeoTIFF module makes to a user:
``open_geotiff`` reads a local GeoTIFF and the result carries the pixels,
the CRS, the transform, and the nodata sentinel from the file.

Why a dedicated release gate
----------------------------
``reader.local_file`` is tagged ``stable`` in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES`. Per epic #2340 every stable
feature needs a release-gate test that fails loudly if the contract
breaks, so a release engineer can run ``pytest -m release_gate`` and
know the next release does not silently regress a stable promise.

This file is intentionally small. The surrounding test suite already
covers dtype variants, compression codecs, planar layouts, COG layouts,
fuzz cases, and golden-corpus parity. The release gate locks the single
contract a release note can quote without caveats:

* Pixel bytes survive the read.
* ``attrs['crs']`` round-trips as the source EPSG.
* ``attrs['transform']`` is the 6-tuple GeoTransform the file carried.
* ``attrs['nodata']`` reflects the on-disk sentinel.

Out of scope: alternative codecs (see ``test_release_gate_codecs.py``),
COG layouts (see ``test_release_gate_cog.py``), windowed reads (see
``test_release_gate_windowed_read.py``), and dask parity (see
``test_release_gate_dask_parity.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


# A tiny axis-aligned grid is enough to lock the contract. Using a
# distinctive pixel pattern (not a constant) means a single-axis drift
# in the writer or reader still fails the equality check.
_PIXELS = np.array(
    [
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ],
    dtype=np.float32,
)

# Web Mercator (EPSG:3857) is a common real-world CRS. The transform
# uses positive pixel width and negative pixel height so the y axis
# decreases with row index, which is the convention every reader in
# this project assumes for axis-aligned grids.
_EPSG = 3857
_ORIGIN_X = 500000.0
_ORIGIN_Y = 4000000.0
_PIXEL_W = 30.0
_PIXEL_H = -30.0
_EXPECTED_TRANSFORM = (_PIXEL_W, 0.0, _ORIGIN_X, 0.0, _PIXEL_H, _ORIGIN_Y)


def _write_known_good(path: str, *, nodata: float | None = None) -> None:
    """Write a known-good GeoTIFF with an explicit GeoTransform.

    Uses the lower-level :func:`xrspatial.geotiff._writer.write` so the
    transform is emitted from the explicit ``geo_transform`` argument
    rather than derived from xarray coords. The release gate locks the
    read side; the writer-side coord-to-transform derivation is covered
    elsewhere.
    """
    gt = GeoTransform(
        origin_x=_ORIGIN_X,
        origin_y=_ORIGIN_Y,
        pixel_width=_PIXEL_W,
        pixel_height=_PIXEL_H,
    )
    write(
        _PIXELS,
        path,
        geo_transform=gt,
        crs_epsg=_EPSG,
        nodata=nodata,
        compression="none",
        tiled=False,
    )


@pytest.mark.release_gate
def test_release_gate_local_read_pixels(tmp_path) -> None:
    """Pixel bytes survive the read."""
    path = str(tmp_path / "release_gate_local_read_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)

    assert da.dtype == np.float32, (
        f"release gate: local read promoted dtype to {da.dtype!r}; the "
        "release contract is that float32 stays float32 unless a "
        "nodata sentinel forces promotion"
    )
    np.testing.assert_array_equal(
        np.asarray(da.values),
        _PIXELS,
        err_msg=(
            "release gate: local read returned different pixels than the "
            "writer emitted; the byte-for-byte round trip is the most "
            "basic promise the release notes make"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_local_read_crs(tmp_path) -> None:
    """``attrs['crs']`` round-trips as the source EPSG."""
    path = str(tmp_path / "release_gate_local_read_crs_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)
    crs = da.attrs.get("crs")
    assert crs is not None, (
        "release gate: local read dropped ``attrs['crs']``; the release "
        "contract promises that an EPSG-coded source surfaces its CRS"
    )
    assert int(crs) == _EPSG, (
        f"release gate: ``attrs['crs']`` drifted from {_EPSG} to "
        f"{crs!r}; this changes the release notes contract for "
        "``reader.local_file``"
    )


@pytest.mark.release_gate
def test_release_gate_local_read_transform(tmp_path) -> None:
    """``attrs['transform']`` is the 6-tuple GeoTransform the file carried."""
    path = str(tmp_path / "release_gate_local_read_transform_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)
    transform = da.attrs.get("transform")
    assert transform is not None, (
        "release gate: local read dropped ``attrs['transform']``; the "
        "release contract promises a 6-tuple GeoTransform on every "
        "georeferenced read"
    )
    assert len(transform) == 6, (
        f"release gate: transform tuple is no longer length 6: "
        f"{transform!r}; release notes promise the rasterio-style 6-tuple"
    )
    for got, want in zip(transform, _EXPECTED_TRANSFORM):
        # Floats compared to float precision because the writer encodes
        # the transform as doubles in the GeoTIFF tags.
        assert got == pytest.approx(want, abs=1e-12, rel=1e-12), (
            f"release gate: transform tuple drifted: got {transform!r} "
            f"want {_EXPECTED_TRANSFORM!r}"
        )


@pytest.mark.release_gate
def test_release_gate_local_read_nodata(tmp_path) -> None:
    """``attrs['nodata']`` reflects the on-disk sentinel."""
    path = str(tmp_path / "release_gate_local_read_nodata_2340.tif")
    sentinel = -9999.0
    _write_known_good(path, nodata=sentinel)

    da = open_geotiff(path)
    nodata = da.attrs.get("nodata")
    assert nodata is not None, (
        "release gate: declared nodata sentinel was dropped on read; "
        "the release contract promises that a declared sentinel "
        "surfaces in ``attrs['nodata']``"
    )
    assert float(nodata) == pytest.approx(sentinel, abs=0.0), (
        f"release gate: ``attrs['nodata']`` drifted from {sentinel} to "
        f"{nodata!r}"
    )
