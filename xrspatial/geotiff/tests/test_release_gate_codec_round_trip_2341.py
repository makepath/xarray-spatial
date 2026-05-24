"""Release gate: stable-codec read/write/read round-trip (epic #2341).

PR 4 of 5 of epic #2341. The release contract names a specific set of
codecs as ``stable`` in :data:`xrspatial.geotiff.SUPPORTED_FEATURES`:
``none``, ``deflate``, ``lzw``, ``zstd``, ``packbits``. The release
notes promise that on any of these codecs, a round-trip preserves both
bit-exact pixels AND every canonical release attr key, on every dtype
the library promises to round-trip.

Existing tests split the contract:

* ``test_compression.py`` covers codec internals (LZW dictionary edge
  cases, PackBits boundary cases, deflate stream framing).
* ``test_supported_features_tiers_2137.py`` pins the
  ``SUPPORTED_FEATURES`` tier table.
* ``test_release_gate_codecs.py`` pins lossless pixel round-trip for
  two dtypes (``uint16``, ``float32``).

This file is the joint gate: the cartesian product of every stable
codec with every promised dtype, asserting both pixel equality AND
release-attr equality through a full read/write/read cycle.

Out of scope:

* Experimental codecs (``lerc``, ``jpeg2000``, ``j2k``, ``lz4``) --
  release tier is ``experimental``; covered by
  ``test_supported_features_tiers_2137.py``.
* Internal-only ``jpeg`` -- not part of the public surface.
* COG layout (``test_release_gate_cog.py``).
* Backend parity (``test_backend_parity_matrix.py``).
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import SUPPORTED_FEATURES, open_geotiff, to_geotiff
from xrspatial.geotiff._compression import (COMPRESSION_DEFLATE, COMPRESSION_LZW,
                                            COMPRESSION_NONE, COMPRESSION_PACKBITS,
                                            COMPRESSION_ZSTD)
from xrspatial.geotiff._header import parse_header, parse_ifd

# The stable lossless codec set. Kept in lockstep with the ``codec.*``
# entries tiered ``stable`` in
# :data:`xrspatial.geotiff.SUPPORTED_FEATURES`. The drift guard at the
# bottom of this file fails the build if the two sets disagree.
STABLE_CODECS = ("none", "deflate", "lzw", "zstd", "packbits")

# The dtype set the release contract promises to round-trip through
# every stable codec. ``int16`` and ``int32`` exercise the signed
# integer path; ``float32`` and ``float64`` exercise the IEEE float
# path with NaN as the nodata sentinel.
DTYPES = ("int16", "int32", "float32", "float64")

# TIFF tag value the on-disk file should carry for each stable codec
# name. The reader IFD parser exposes ``ifd.compression`` so we can
# assert the on-disk tag without depending on a high-level
# ``attrs['compression']`` key (none exists; see issue #2341).
_CODEC_TO_TIFF_TAG = {
    "none": COMPRESSION_NONE,
    "deflate": COMPRESSION_DEFLATE,
    "lzw": COMPRESSION_LZW,
    "zstd": COMPRESSION_ZSTD,
    "packbits": COMPRESSION_PACKBITS,
}

# Per-dtype integer nodata sentinel. Float dtypes use NaN. The
# integer sentinels are well outside the natural value range of the
# fixture below (small ascending integers) so the sentinel never
# collides with a real pixel.
_INT_NODATA = {
    "int16": np.int16(-32768),
    "int32": np.int32(-2147483648),
}

# Release-attr keys the cartesian-product gate asserts on. These come
# from the issue body (#2341) and from the canonical attrs the reader
# emits (see ``test_release_gate_attrs_contract.py``). ``raster_type``
# is included even though it is only emitted when the source was
# ``RasterPixelIsPoint``; we use a small fixture that defaults to
# ``'area'`` so it is normalized below in ``_canonical_attrs``.
_RELEASE_ATTR_KEYS = (
    "transform",
    "crs",
    "crs_wkt",
    "nodata",
    "masked_nodata",
    "georef_status",
    "raster_type",
)


def _make_input(dtype_name: str) -> xr.DataArray:
    """Build a 128x128 DataArray of the given dtype.

    Float arrays seed a NaN sentinel at (0, 0); integer arrays seed
    the per-dtype sentinel at (0, 0). The remaining pixels are a
    deterministic, non-trivial pattern so a per-axis flip or stride
    bug surfaces as a pixel mismatch.
    """
    dtype = np.dtype(dtype_name)
    height, width = 128, 128
    n = height * width
    if np.issubdtype(dtype, np.floating):
        arr = np.linspace(-100.0, 100.0, n, dtype=dtype).reshape(height, width)
        arr[0, 0] = np.nan
        nodata: float | int = float("nan")
    else:
        # Small positive ramp so the dtype min sentinel never collides
        # with a real pixel.
        arr = np.arange(n, dtype=dtype).reshape(height, width)
        sentinel = _INT_NODATA[dtype_name]
        arr[0, 0] = sentinel
        nodata = sentinel

    # 30 m pixels with a descending y axis (top-left at the highest y
    # coord). The writer turns these into a GeoTransform of
    # ``(30, 0, origin_x, 0, -30, origin_y)``.
    y = 4000000.0 - 30.0 * (np.arange(height) + 0.5)
    x = 500000.0 + 30.0 * (np.arange(width) + 0.5)
    attrs: dict = {"crs": 32610, "nodata": nodata}
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs=attrs,
    )


def _canonical_attrs(da: xr.DataArray) -> dict:
    """Project a DataArray's ``attrs`` onto the release-attr key set.

    ``raster_type`` is missing from ``attrs`` for the default ``area``
    raster (the writer only stamps ``'point'`` explicitly); normalize
    here so the cross-read comparison can treat the missing key as
    equivalent to ``'area'``.
    """
    out = {}
    for key in _RELEASE_ATTR_KEYS:
        if key == "raster_type":
            out[key] = da.attrs.get("raster_type", "area")
        else:
            out[key] = da.attrs.get(key)
    return out


def _read_tiff_compression_tag(path: str) -> int:
    """Read the on-disk TIFF Compression tag from the first IFD.

    The reader's high-level API does not surface ``attrs['compression']``
    (issue #2341 question). Inspect the IFD directly so the test pins
    the actual on-disk codec choice rather than relying on the
    DataArray attrs the reader emits.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    return ifd.compression


def _assert_pixels_equal(actual: np.ndarray, expected: np.ndarray,
                         *, codec: str, dtype_name: str) -> None:
    """NaN-aware byte-exact pixel comparison.

    The float path uses ``equal_nan=True`` so the NaN sentinel
    matches NaN-to-NaN. The integer path uses strict
    ``array_equal`` -- the sentinel is just another integer value
    and must round-trip bit-exact.
    """
    assert actual.shape == expected.shape, (
        f"release gate (#2341): codec {codec!r} dtype {dtype_name!r} "
        f"reshaped the array across the round-trip: "
        f"{expected.shape} -> {actual.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"release gate (#2341): codec {codec!r} promoted dtype "
        f"{dtype_name!r} to {actual.dtype!r} across the round-trip"
    )
    if np.issubdtype(expected.dtype, np.floating):
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    assert equal, (
        f"release gate (#2341): codec {codec!r} did not preserve "
        f"{dtype_name!r} pixels byte-for-byte; the release contract "
        f"names this codec as lossless for this dtype"
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("codec", STABLE_CODECS)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_release_gate_codec_round_trip(tmp_path, codec, dtype_name) -> None:
    """Stable codec * dtype: pixels and release attrs survive a full
    read/write/read cycle.

    Steps:

    1. Build an in-memory DataArray with a known transform, CRS, and
       nodata sentinel (NaN for float; per-dtype int min for int).
    2. Write via ``to_geotiff(path, compression=codec)``.
    3. Read back via ``open_geotiff(path)`` -- this is the canonical
       baseline. The reader fills in ``crs_wkt``,
       ``georef_status``, ``masked_nodata``, etc.
    4. Write the baseline DataArray to a second path under the same
       codec.
    5. Read the second path back; assert byte-exact pixels and every
       release-attr key matches the baseline.

    The two-pass shape is what makes this a *round-trip* gate
    rather than a single-pass write-and-read gate: the canonical
    attrs themselves have to survive the second cycle, not just the
    first.
    """
    # Unique tag per parametrized case so parallel pytest workers and
    # parallel rockout worktrees never collide on the same tmp file.
    nonce = uuid.uuid4().hex[:8]
    write_first = str(
        tmp_path
        / f"release_gate_2341_{codec}_{dtype_name}_first_{nonce}.tif"
    )
    write_second = str(
        tmp_path
        / f"release_gate_2341_{codec}_{dtype_name}_second_{nonce}.tif"
    )

    source = _make_input(dtype_name)

    # The masking behaviour differs by dtype: integer reads default to
    # masking the sentinel into NaN (which would change dtype and break
    # the byte-exact comparison), so we read integers with
    # ``mask_nodata=False`` to keep the sentinel as a real pixel.
    # Float reads round-trip NaN as NaN regardless of mask_nodata.
    mask_kwargs = ({}
                   if np.issubdtype(np.dtype(dtype_name), np.floating)
                   else {"mask_nodata": False})

    # Pass 1: write the in-memory source.
    to_geotiff(
        source,
        write_first,
        compression=codec,
        tiled=False,
        nodata=source.attrs["nodata"],
    )

    baseline = open_geotiff(write_first, **mask_kwargs)
    baseline_pixels = np.asarray(baseline.values)
    baseline_attrs = _canonical_attrs(baseline)

    # The on-disk TIFF Compression tag must reflect the requested codec.
    tag_first = _read_tiff_compression_tag(write_first)
    assert tag_first == _CODEC_TO_TIFF_TAG[codec], (
        f"release gate (#2341): codec {codec!r} encoded as TIFF tag "
        f"{tag_first} on first write; expected "
        f"{_CODEC_TO_TIFF_TAG[codec]} per the codec -> tag map"
    )

    # Pass 2: rewrite the baseline DataArray under the same codec.
    to_geotiff(
        baseline,
        write_second,
        compression=codec,
        tiled=False,
        nodata=baseline.attrs.get("nodata"),
    )

    second = open_geotiff(write_second, **mask_kwargs)
    second_pixels = np.asarray(second.values)
    second_attrs = _canonical_attrs(second)

    tag_second = _read_tiff_compression_tag(write_second)
    assert tag_second == _CODEC_TO_TIFF_TAG[codec], (
        f"release gate (#2341): codec {codec!r} encoded as TIFF tag "
        f"{tag_second} on the second write; expected "
        f"{_CODEC_TO_TIFF_TAG[codec]} per the codec -> tag map"
    )

    _assert_pixels_equal(
        second_pixels, baseline_pixels, codec=codec, dtype_name=dtype_name,
    )

    # Per-attribute comparison so a single failing key reports which
    # attr drifted instead of a wholesale dict-equality failure.
    for key in _RELEASE_ATTR_KEYS:
        want = baseline_attrs[key]
        got = second_attrs[key]
        if key == "nodata" and isinstance(want, float) and np.isnan(want):
            assert isinstance(got, float) and np.isnan(got), (
                f"release gate (#2341): codec {codec!r} dtype "
                f"{dtype_name!r} dropped NaN nodata across the "
                f"round-trip: got {got!r}"
            )
            continue
        if key == "transform":
            assert want is not None and got is not None, (
                f"release gate (#2341): codec {codec!r} dtype "
                f"{dtype_name!r} dropped ``attrs['transform']``: "
                f"{want!r} -> {got!r}"
            )
            assert tuple(got) == tuple(want), (
                f"release gate (#2341): codec {codec!r} dtype "
                f"{dtype_name!r} drifted ``attrs['transform']``: "
                f"{want!r} -> {got!r}"
            )
            continue
        assert got == want, (
            f"release gate (#2341): codec {codec!r} dtype {dtype_name!r} "
            f"drifted ``attrs[{key!r}]`` across the round-trip: "
            f"{want!r} -> {got!r}"
        )


@pytest.mark.release_gate
def test_release_gate_codec_round_trip_stable_set_matches_supported_features() -> None:
    """The codec list in this file matches ``SUPPORTED_FEATURES``.

    If a codec is promoted into ``stable`` (or demoted out) in
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` without updating
    this file, the cartesian-product gate is silently out of sync
    with the runtime tier table. Fail loudly here so the PR that
    changes the tier also updates the gate.
    """
    stable_from_constant = {
        key.split(".", 1)[1]
        for key, tier in SUPPORTED_FEATURES.items()
        if key.startswith("codec.") and tier == "stable"
    }
    assert stable_from_constant == set(STABLE_CODECS), (
        "release gate (#2341): STABLE_CODECS drifted from "
        "SUPPORTED_FEATURES; the gate and the runtime tier table "
        "must agree on which codecs are stable. "
        f"constant: {set(STABLE_CODECS)!r}; "
        f"SUPPORTED_FEATURES: {stable_from_constant!r}"
    )
