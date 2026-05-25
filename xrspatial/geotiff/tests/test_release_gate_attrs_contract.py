"""Release gate: CRS / transform / nodata attrs contract (epic #2340).

The canonical attrs after a GeoTIFF read are tagged ``stable`` in the
release gate checklist. The contract: every georeferenced read produces
a DataArray whose ``attrs`` carry, at minimum, ``crs``, ``crs_wkt``,
``transform``, ``georef_status``, the contract version stamp, and (when
declared) ``nodata``. These attrs survive a write -> read round trip.

This file is the single-shot release gate. Deep canonicalisation,
alias handling, contract version bumps, and pass-through semantics are
each covered by their own ``test_attrs_contract_*_1984.py`` files; here
we lock the user-facing names and round-trip stability so the release
notes can quote the canonical attrs without caveats.

Out of scope:
* Alias handling (``test_attrs_contract_aliases_1984.py``).
* Attrs pass-through for user-supplied keys
  (``test_attrs_contract_passthrough_1984.py``).
* Contract version stamp bump policy
  (``test_attrs_contract_version_1984.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


# Keys that release notes are allowed to promise on every georeferenced
# read. Adding a new key to the canonical set is a contract-version
# bump (see issue #1984); removing one is a breaking change. Anything
# else in the attrs (``masked_nodata``, ``nodata_pixels_present``,
# ``raster_type``, etc.) is additive and not pinned here.
CANONICAL_KEYS = (
    "_xrspatial_geotiff_contract",
    "crs",
    "crs_wkt",
    "transform",
    "georef_status",
)


def _write_known_good(path: str, *, nodata: float | None = None) -> None:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    gt = GeoTransform(
        origin_x=500000.0,
        origin_y=4000000.0,
        pixel_width=30.0,
        pixel_height=-30.0,
    )
    write(
        arr,
        path,
        geo_transform=gt,
        crs_epsg=32610,
        nodata=nodata,
        compression="none",
        tiled=False,
    )


@pytest.mark.release_gate
def test_release_gate_attrs_canonical_keys_present(tmp_path) -> None:
    """A georeferenced read carries every canonical attrs key."""
    path = str(tmp_path / "release_gate_attrs_canonical_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)
    missing = [k for k in CANONICAL_KEYS if k not in da.attrs]
    assert not missing, (
        "release gate: canonical attrs keys missing from a georeferenced "
        f"read: {missing}; release notes promise every key in "
        f"{list(CANONICAL_KEYS)}"
    )


@pytest.mark.release_gate
def test_release_gate_attrs_georef_status_full(tmp_path) -> None:
    """A fully-georeferenced read reports ``georef_status='full'``."""
    path = str(tmp_path / "release_gate_attrs_georef_status_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)
    status = da.attrs.get("georef_status")
    assert status == "full", (
        f"release gate: a CRS+transform read should report "
        f"``georef_status='full'``; got {status!r}. The five canonical "
        "georef_status values are the contract downstream code branches on"
    )


@pytest.mark.release_gate
def test_release_gate_attrs_contract_version_is_int(tmp_path) -> None:
    """``attrs['_xrspatial_geotiff_contract']`` is an int.

    The contract version is the downstream signal for which attrs
    shape the array carries. A drift from int to string (or to a
    Python object) would silently break callers that compare versions.
    """
    path = str(tmp_path / "release_gate_attrs_contract_version_2340.tif")
    _write_known_good(path)

    da = open_geotiff(path)
    version = da.attrs.get("_xrspatial_geotiff_contract")
    assert isinstance(version, int), (
        f"release gate: contract version stamp is not int: type="
        f"{type(version).__name__}, value={version!r}"
    )
    assert version >= 1, (
        f"release gate: contract version stamp is non-positive: {version!r}"
    )


@pytest.mark.release_gate
def test_release_gate_attrs_round_trip_preserves_crs_transform_nodata(
    tmp_path,
) -> None:
    """Canonical attrs survive a full ``write -> read -> write -> read`` cycle."""
    src = str(tmp_path / "release_gate_attrs_rt_src_2340.tif")
    _write_known_good(src, nodata=-9999.0)

    first = open_geotiff(src)
    crs_first = int(first.attrs["crs"])
    transform_first = tuple(first.attrs["transform"])
    nodata_first = float(first.attrs["nodata"])

    # Round-trip through the public writer.
    rewrite = str(tmp_path / "release_gate_attrs_rt_rewrite_2340.tif")
    to_geotiff(first, rewrite, compression="none", tiled=False)

    second = open_geotiff(rewrite)
    assert int(second.attrs["crs"]) == crs_first, (
        f"release gate: CRS drifted across round-trip: {crs_first} -> "
        f"{second.attrs['crs']!r}"
    )
    transform_second = tuple(second.attrs["transform"])
    assert len(transform_second) == 6, (
        f"release gate: transform reshaped across round-trip: "
        f"{transform_second!r}"
    )
    for got, want in zip(transform_second, transform_first):
        assert got == pytest.approx(want, abs=1e-12, rel=1e-12), (
            f"release gate: transform drifted across round-trip: "
            f"{transform_first!r} -> {transform_second!r}"
        )
    assert float(second.attrs["nodata"]) == pytest.approx(
        nodata_first, abs=0.0
    ), (
        f"release gate: nodata drifted across round-trip: "
        f"{nodata_first} -> {second.attrs['nodata']!r}"
    )
