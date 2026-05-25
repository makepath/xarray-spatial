"""Release gate: stable lossless codec round-trip (epic #2340).

The release contract for the GeoTIFF module names a specific set of
lossless codecs as ``stable``: ``none``, ``deflate``, ``lzw``,
``packbits``, ``zstd``. Every one of them must round-trip pixels
byte-for-byte through ``to_geotiff`` -> ``open_geotiff`` on both
integer and float dtypes.

This file is the per-codec gate: one parametrized test per dtype that
walks every stable codec. The fine-grained codec internals (LZW
dictionary edge cases, PackBits boundary cases, deflate stream framing,
etc.) live in their dedicated test files; here we only assert the
end-to-end public-API promise.

Out of scope: experimental codecs (``lerc``, ``jpeg2000``, ``j2k``,
``lz4``), the internal-only ``jpeg`` codec, and the COG layout gate
(see ``test_release_gate_cog.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import SUPPORTED_FEATURES, open_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


# The stable lossless codec set. Keep this list in lockstep with the
# ``codec.*`` entries tiered ``stable`` in
# :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. If a codec is promoted
# into or out of stable, add or remove it here -- the gate is meant
# to lock the public-facing list.
STABLE_LOSSLESS_CODECS = ("none", "deflate", "lzw", "packbits", "zstd")


def _gt() -> GeoTransform:
    return GeoTransform(
        origin_x=500000.0,
        origin_y=4000000.0,
        pixel_width=30.0,
        pixel_height=-30.0,
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_codec_round_trip_uint16(tmp_path, codec) -> None:
    """Integer pixel bytes survive every stable lossless codec."""
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = str(tmp_path / f"release_gate_codec_{codec}_uint16_2340.tif")
    write(
        arr,
        path,
        geo_transform=_gt(),
        crs_epsg=32610,
        compression=codec,
        tiled=False,
    )

    out = open_geotiff(path)
    assert out.dtype == np.uint16, (
        f"release gate: codec {codec!r} promoted uint16 to {out.dtype!r}; "
        "the lossless contract is that integer dtypes survive every "
        "stable codec"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        arr,
        err_msg=(
            f"release gate: codec {codec!r} did not round-trip uint16 "
            "pixels byte-for-byte; the release contract names this codec "
            "as lossless"
        ),
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_codec_round_trip_float32(tmp_path, codec) -> None:
    """Float pixel bytes survive every stable lossless codec."""
    # Use a deterministic but non-trivial pattern so a per-axis flip
    # or per-row stride bug still fails.
    arr = np.linspace(-100.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / f"release_gate_codec_{codec}_float32_2340.tif")
    write(
        arr,
        path,
        geo_transform=_gt(),
        crs_epsg=32610,
        compression=codec,
        tiled=False,
    )

    out = open_geotiff(path)
    assert out.dtype == np.float32, (
        f"release gate: codec {codec!r} promoted float32 to "
        f"{out.dtype!r}"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        arr,
        err_msg=(
            f"release gate: codec {codec!r} did not round-trip float32 "
            "pixels byte-for-byte; the release contract names this codec "
            "as lossless"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_codec_stable_set_matches_supported_features() -> None:
    """The stable codec list in this file matches ``SUPPORTED_FEATURES``.

    If a codec is promoted into ``stable`` (or demoted out) in
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` without updating this
    file, the release gate is out of sync with the runtime contract.
    Fail loudly here so the PR that changes the tier also updates the
    gate.
    """
    stable_from_constant = {
        key.split(".", 1)[1]
        for key, tier in SUPPORTED_FEATURES.items()
        if key.startswith("codec.") and tier == "stable"
    }
    assert stable_from_constant == set(STABLE_LOSSLESS_CODECS), (
        "release gate: STABLE_LOSSLESS_CODECS drifted from "
        "SUPPORTED_FEATURES; the gate and the runtime tier table must "
        "agree on which codecs are stable. "
        f"constant: {set(STABLE_LOSSLESS_CODECS)!r}; "
        f"SUPPORTED_FEATURES: {stable_from_constant!r}"
    )
