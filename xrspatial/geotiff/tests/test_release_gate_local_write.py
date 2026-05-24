"""Release gate: local GeoTIFF write (epic #2340).

``writer.local_file`` is tagged ``stable`` in
:data:`xrspatial.geotiff.SUPPORTED_FEATURES`. The release contract is:
``to_geotiff`` writes a file that ``open_geotiff`` reads back bit-exact,
with the CRS, transform, and nodata sentinel preserved.

This gate is small on purpose. The byte-equivalent pixel contract,
attrs canonicalisation, and dtype handling each have their own deep
test files (``test_round_trip_invariants.py``,
``test_attrs_contract_canonical_1984.py``, the matrix tests). The
release-gate test is the one-shot a release engineer can run to know
the most common public-API write -> read flow still works end-to-end.

Out of scope here:
* Compression codec coverage -- see ``test_release_gate_codecs.py``.
* COG layout -- see ``test_release_gate_cog.py``.
* Detailed attrs canonicalisation -- see
  ``test_release_gate_attrs_contract.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _make_data_array(*, nodata: float | None = None) -> xr.DataArray:
    """Build a small DataArray with explicit y/x coords.

    The release contract for ``to_geotiff`` is the public-API path: a
    user passes a DataArray with coords, gets back a file whose
    GeoTransform reproduces those coords. We keep the grid small (4x4)
    so the gate is fast even when run alongside the full release-gate
    suite.
    """
    pixels = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
    # Pixel-center y/x with width 30 m, origin (500000, 4000000),
    # descending y. The writer turns these into a GeoTransform with
    # origin at the top-left pixel corner.
    y = np.array([3999985.0, 3999955.0, 3999925.0, 3999895.0])
    x = np.array([500015.0, 500045.0, 500075.0, 500105.0])
    attrs: dict = {"crs": 32610}
    if nodata is not None:
        attrs["nodata"] = nodata
    return xr.DataArray(
        pixels,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs=attrs,
    )


@pytest.mark.release_gate
def test_release_gate_local_write_round_trips_pixels(tmp_path) -> None:
    """``to_geotiff`` writes a file that reads back bit-exact."""
    da = _make_data_array()
    path = str(tmp_path / "release_gate_local_write_pixels_2340.tif")
    to_geotiff(da, path, compression="none", tiled=False)

    out = open_geotiff(path)
    assert out.dtype == np.float32, (
        f"release gate: write -> read flipped dtype to {out.dtype!r}; "
        "the release contract promises float32 stays float32 absent a "
        "nodata sentinel"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        np.asarray(da.values),
        err_msg=(
            "release gate: write -> read changed pixel values; "
            "to_geotiff is promised to be lossless for the default "
            "'none' codec"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_local_write_preserves_crs(tmp_path) -> None:
    """The CRS survives the write -> read round trip."""
    da = _make_data_array()
    path = str(tmp_path / "release_gate_local_write_crs_2340.tif")
    to_geotiff(da, path, compression="none", tiled=False)

    out = open_geotiff(path)
    crs = out.attrs.get("crs")
    assert crs is not None, (
        "release gate: write -> read dropped ``attrs['crs']``; the "
        "release contract requires the CRS to survive"
    )
    assert int(crs) == 32610, (
        f"release gate: ``attrs['crs']`` drifted from 32610 to {crs!r}"
    )


@pytest.mark.release_gate
def test_release_gate_local_write_preserves_transform(tmp_path) -> None:
    """The GeoTransform survives the write -> read round trip."""
    da = _make_data_array()
    path = str(tmp_path / "release_gate_local_write_transform_2340.tif")
    to_geotiff(da, path, compression="none", tiled=False)

    out = open_geotiff(path)
    transform = out.attrs.get("transform")
    assert transform is not None, (
        "release gate: write -> read dropped ``attrs['transform']``; "
        "the release contract requires the GeoTransform to survive"
    )
    assert len(transform) == 6, (
        f"release gate: transform tuple is no longer length 6: "
        f"{transform!r}"
    )
    # Pixel width and pixel height must round-trip exactly; the origin
    # is the top-left corner derived from pixel-center coords plus a
    # half-pixel offset, so it is also a tight equality.
    assert transform[0] == pytest.approx(30.0, abs=1e-9), (
        f"release gate: pixel_width drifted: {transform!r}"
    )
    assert transform[4] == pytest.approx(-30.0, abs=1e-9), (
        f"release gate: pixel_height sign or magnitude drifted: "
        f"{transform!r}"
    )
    assert transform[1] == 0.0 and transform[3] == 0.0, (
        f"release gate: shear terms appeared in axis-aligned write: "
        f"{transform!r}"
    )


@pytest.mark.release_gate
def test_release_gate_local_write_preserves_nodata(tmp_path) -> None:
    """A declared nodata sentinel survives the write -> read round trip."""
    sentinel = -9999.0
    da = _make_data_array(nodata=sentinel)
    path = str(tmp_path / "release_gate_local_write_nodata_2340.tif")
    to_geotiff(da, path, compression="none", tiled=False, nodata=sentinel)

    out = open_geotiff(path)
    nodata = out.attrs.get("nodata")
    assert nodata is not None, (
        "release gate: declared nodata was dropped on write -> read; "
        "the release contract promises the sentinel survives"
    )
    assert float(nodata) == pytest.approx(sentinel, abs=0.0), (
        f"release gate: ``attrs['nodata']`` drifted from {sentinel} to "
        f"{nodata!r}"
    )
