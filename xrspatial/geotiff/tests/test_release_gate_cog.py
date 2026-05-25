"""Release gate: COG write and read for stable lossless codecs (epic #2340).

The release contract tags ``writer.cog`` and ``reader.local_cog`` as
``stable`` in :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. The promise
is: ``to_geotiff(cog=True, compression=<stable lossless>)`` writes a
file that ``open_geotiff`` reads back bit-exact, with CRS, transform,
and (when declared) nodata preserved across every stable codec.

This gate parametrizes the codec axis so a single regression in any
stable codec on the COG path fails noisily. The COG layout itself
(IFD-first, tiled, internal overviews) is exhaustively pinned by
``test_cog_writer_compliance.py`` and ``test_cog_parity_2286.py``; the
release-gate gate is the small end-to-end shape every release needs.

Out of scope here:
* COG spec compliance details -- see ``test_cog_writer_compliance.py``.
* HTTP COG range reads -- ``reader.http_cog`` is ``advanced`` (not
  stable), so it is not part of this gate.
* BigTIFF COG -- ``writer.bigtiff_cog`` is ``advanced``.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

# Import the stable lossless set from the sibling release-gate file
# rather than redefining it. The cross-check against
# ``SUPPORTED_FEATURES`` lives in that file; reusing the same tuple
# here means a tier change in ``_attrs.py`` cannot leave the COG gate
# parametrized on a stale list.
from xrspatial.geotiff.tests.test_release_gate_codecs import (  # noqa: E402
    STABLE_LOSSLESS_CODECS,
)

# COG requires a tiled internal layout and benefits from a slightly
# larger raster than the plain-file gate so the writer can emit a real
# tile grid rather than a single 1-tile file. Sticking to 32x32 keeps
# the test fast (well under 1 ms for the codec loop) while still
# exercising multiple tiles.
_W = 32
_H = 32


def _make_data_array(*, nodata: float | None = None) -> xr.DataArray:
    pixels = np.arange(_H * _W, dtype=np.float32).reshape(_H, _W)
    # Pixel-center coords, 30 m pixels, top-left at (500000, 4000000).
    y = np.array(
        [4000000.0 - 15.0 - 30.0 * i for i in range(_H)],
        dtype=np.float64,
    )
    x = np.array(
        [500000.0 + 15.0 + 30.0 * i for i in range(_W)],
        dtype=np.float64,
    )
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
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_cog_round_trips_pixels(tmp_path, codec) -> None:
    """COG write -> read returns the same pixels under every stable codec."""
    da = _make_data_array()
    path = str(tmp_path / f"release_gate_cog_{codec}_pixels_2340.tif")
    to_geotiff(
        da,
        path,
        compression=codec,
        cog=True,
        tiled=True,
        tile_size=16,
    )

    out = open_geotiff(path)
    assert out.dtype == np.float32, (
        f"release gate: COG with codec {codec!r} promoted dtype to "
        f"{out.dtype!r}"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        np.asarray(da.values),
        err_msg=(
            f"release gate: COG with codec {codec!r} did not round-trip "
            "pixels byte-for-byte"
        ),
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_cog_preserves_crs_transform(tmp_path, codec) -> None:
    """CRS and transform survive the COG write -> read for every stable codec."""
    da = _make_data_array()
    path = str(tmp_path / f"release_gate_cog_{codec}_attrs_2340.tif")
    to_geotiff(
        da,
        path,
        compression=codec,
        cog=True,
        tiled=True,
        tile_size=16,
    )

    out = open_geotiff(path)
    crs = out.attrs.get("crs")
    assert crs is not None and int(crs) == 32610, (
        f"release gate: COG with codec {codec!r} dropped or drifted "
        f"``attrs['crs']``: got {crs!r}"
    )
    transform = out.attrs.get("transform")
    assert transform is not None and len(transform) == 6, (
        f"release gate: COG with codec {codec!r} dropped or reshaped "
        f"``attrs['transform']``: got {transform!r}"
    )
    assert transform[0] == pytest.approx(30.0, abs=1e-9), (
        f"release gate: COG pixel_width drifted under {codec!r}: "
        f"{transform!r}"
    )
    assert transform[4] == pytest.approx(-30.0, abs=1e-9), (
        f"release gate: COG pixel_height drifted under {codec!r}: "
        f"{transform!r}"
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_cog_preserves_nodata(tmp_path, codec) -> None:
    """A declared nodata sentinel survives COG write -> read under every codec."""
    sentinel = -9999.0
    da = _make_data_array(nodata=sentinel)
    path = str(tmp_path / f"release_gate_cog_{codec}_nodata_2340.tif")
    to_geotiff(
        da,
        path,
        compression=codec,
        nodata=sentinel,
        cog=True,
        tiled=True,
        tile_size=16,
    )

    out = open_geotiff(path)
    nodata = out.attrs.get("nodata")
    assert nodata is not None, (
        f"release gate: COG with codec {codec!r} dropped declared nodata"
    )
    assert float(nodata) == pytest.approx(sentinel, abs=0.0), (
        f"release gate: COG with codec {codec!r} drifted nodata from "
        f"{sentinel} to {nodata!r}"
    )
