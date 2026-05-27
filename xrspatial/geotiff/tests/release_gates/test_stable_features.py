"""Single-file registry of every ``@pytest.mark.release_gate`` test.

The release process needs one audit point for "what does CI run when
validating release readiness". This module is that point: every test
that backs a row in ``docs/source/reference/release_gate_geotiff.rst``
lives here.

Organisation
============

Tests are grouped by feature area:

* ``Local read``       (``reader.local_file``, stable)
* ``Local write``      (``writer.local_file``, stable)
* ``Codecs``           (the stable lossless codec set)
* ``COG``              (``writer.cog`` / ``reader.local_cog``, stable)
* ``Windowed read``    (``reader.windowed``, stable)
* ``Dask parity``      (``reader.dask``, stable)
* ``Eager / dask full parity`` (four corpus scenarios)
* ``Attrs contract``   (canonical attrs round-trip, stable)
* ``Codec round-trip`` (cartesian stable codec x dtype)
* ``Overview / sidecar metadata`` (overview level attrs survive)
* ``Windowed reads -- shifted-transform parity``
* ``Negative cases``   (ambiguous metadata fails closed)
* ``Cross-cutting meta-gates`` (checklist parity)

Each section's helper functions are private to that section
(``_<section>_<name>``) so the consolidation does not introduce
cross-section coupling.

The ``release_gate`` marker is the single signal a release engineer
keys on:

.. code-block:: bash

    pytest xrspatial/geotiff/tests/release_gates/ -m release_gate
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

# Optional-dependency gates. Each section below decorates only the
# tests that actually touch the optional backend, instead of
# module-level ``importorskip`` which would skip the entire 159-test
# file (including pure-numpy gates) on a minimal install.
_HAS_DASK = importlib.util.find_spec("dask") is not None
_HAS_DASK_ARRAY = importlib.util.find_spec("dask.array") is not None
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None

_requires_dask = pytest.mark.skipif(
    not _HAS_DASK, reason="dask is required for this release gate",
)
_requires_rasterio = pytest.mark.skipif(
    not _HAS_RASTERIO,
    reason="rasterio is required for the overview / sidecar gate",
)
_requires_rasterio_and_dask = pytest.mark.skipif(
    not (_HAS_RASTERIO and _HAS_DASK_ARRAY),
    reason=(
        "rasterio and dask.array are required for the overview / sidecar gate"
    ),
)

from xrspatial.geotiff import (SUPPORTED_FEATURES, UnsafeURLError, open_geotiff,  # noqa: E402
                               read_geotiff_dask, to_geotiff)
from xrspatial.geotiff._compression import (COMPRESSION_DEFLATE, COMPRESSION_LZW,  # noqa: E402
                                            COMPRESSION_NONE, COMPRESSION_PACKBITS,
                                            COMPRESSION_ZSTD)
from xrspatial.geotiff._errors import GeoTIFFAmbiguousMetadataError  # noqa: E402
from xrspatial.geotiff._errors import RotatedTransformError  # noqa: E402
from xrspatial.geotiff._geotags import GeoTransform  # noqa: E402
from xrspatial.geotiff._header import parse_header, parse_ifd  # noqa: E402
from xrspatial.geotiff._writer import write  # noqa: E402
from xrspatial.geotiff.tests._helpers.markers import requires_gpu  # noqa: E402

# =========================================================================== #
# Shared codec constants                                                      #
# =========================================================================== #
#
# The stable lossless codec set is named here once so the codec, COG, and
# codec-round-trip sections below all share it. Keeps the cross-check
# against ``SUPPORTED_FEATURES`` in one place too.

STABLE_LOSSLESS_CODECS = ("none", "deflate", "lzw", "packbits", "zstd")

_CODEC_TO_TIFF_TAG = {
    "none": COMPRESSION_NONE,
    "deflate": COMPRESSION_DEFLATE,
    "lzw": COMPRESSION_LZW,
    "zstd": COMPRESSION_ZSTD,
    "packbits": COMPRESSION_PACKBITS,
}


# =========================================================================== #
# Section: Local read (reader.local_file, stable)                             #
# =========================================================================== #
#
# The most basic promise the GeoTIFF module makes to a user: ``open_geotiff``
# reads a local GeoTIFF and the result carries the pixels, the CRS, the
# transform, and the nodata sentinel from the file.

# A tiny axis-aligned grid is enough to lock the contract. The distinctive
# per-row pattern means a single-axis drift in the writer or reader still
# fails the equality check.
_LOCAL_READ_PIXELS = np.array(
    [
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ],
    dtype=np.float32,
)
_LOCAL_READ_EPSG = 3857
_LOCAL_READ_ORIGIN_X = 500000.0
_LOCAL_READ_ORIGIN_Y = 4000000.0
_LOCAL_READ_PIXEL_W = 30.0
_LOCAL_READ_PIXEL_H = -30.0
_LOCAL_READ_EXPECTED_TRANSFORM = (
    _LOCAL_READ_PIXEL_W, 0.0, _LOCAL_READ_ORIGIN_X,
    0.0, _LOCAL_READ_PIXEL_H, _LOCAL_READ_ORIGIN_Y,
)


def _local_read_write_known_good(
    path: str, *, nodata: float | None = None,
) -> None:
    """Write a known-good GeoTIFF using the explicit ``geo_transform`` path."""
    gt = GeoTransform(
        origin_x=_LOCAL_READ_ORIGIN_X,
        origin_y=_LOCAL_READ_ORIGIN_Y,
        pixel_width=_LOCAL_READ_PIXEL_W,
        pixel_height=_LOCAL_READ_PIXEL_H,
    )
    write(
        _LOCAL_READ_PIXELS,
        path,
        geo_transform=gt,
        crs_epsg=_LOCAL_READ_EPSG,
        nodata=nodata,
        compression="none",
        tiled=False,
    )


@pytest.mark.release_gate
def test_release_gate_local_read_pixels(tmp_path) -> None:
    """Pixel bytes survive the read."""
    path = str(tmp_path / "release_gate_local_read.tif")
    _local_read_write_known_good(path)

    da = open_geotiff(path)
    assert da.dtype == np.float32, (
        f"release gate: local read promoted dtype to {da.dtype!r}; the "
        "release contract is that float32 stays float32 unless a "
        "nodata sentinel forces promotion"
    )
    np.testing.assert_array_equal(
        np.asarray(da.values),
        _LOCAL_READ_PIXELS,
        err_msg=(
            "release gate: local read returned different pixels than the "
            "writer emitted; the byte-for-byte round trip is the most "
            "basic promise the release notes make"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_local_read_crs(tmp_path) -> None:
    """``attrs['crs']`` round-trips as the source EPSG."""
    path = str(tmp_path / "release_gate_local_read_crs.tif")
    _local_read_write_known_good(path)

    da = open_geotiff(path)
    crs = da.attrs.get("crs")
    assert crs is not None, (
        "release gate: local read dropped ``attrs['crs']``; the release "
        "contract promises that an EPSG-coded source surfaces its CRS"
    )
    assert int(crs) == _LOCAL_READ_EPSG, (
        f"release gate: ``attrs['crs']`` drifted from {_LOCAL_READ_EPSG} "
        f"to {crs!r}; this changes the release notes contract for "
        "``reader.local_file``"
    )


@pytest.mark.release_gate
def test_release_gate_local_read_transform(tmp_path) -> None:
    """``attrs['transform']`` is the 6-tuple GeoTransform the file carried."""
    path = str(tmp_path / "release_gate_local_read_transform.tif")
    _local_read_write_known_good(path)

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
    for got, want in zip(transform, _LOCAL_READ_EXPECTED_TRANSFORM):
        # Floats compared at float precision because the writer encodes the
        # transform as doubles in the GeoTIFF tags.
        assert got == pytest.approx(want, abs=1e-12, rel=1e-12), (
            f"release gate: transform tuple drifted: got {transform!r} "
            f"want {_LOCAL_READ_EXPECTED_TRANSFORM!r}"
        )


@pytest.mark.release_gate
def test_release_gate_local_read_nodata(tmp_path) -> None:
    """``attrs['nodata']`` reflects the on-disk sentinel."""
    path = str(tmp_path / "release_gate_local_read_nodata.tif")
    sentinel = -9999.0
    _local_read_write_known_good(path, nodata=sentinel)

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


# =========================================================================== #
# Section: Local write (writer.local_file, stable)                            #
# =========================================================================== #

def _local_write_make_data_array(
    *, nodata: float | None = None,
) -> xr.DataArray:
    """Build a small DataArray with explicit y/x coords for the writer gate."""
    pixels = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )
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
    da = _local_write_make_data_array()
    path = str(tmp_path / "release_gate_local_write_pixels.tif")
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
    da = _local_write_make_data_array()
    path = str(tmp_path / "release_gate_local_write_crs.tif")
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
    da = _local_write_make_data_array()
    path = str(tmp_path / "release_gate_local_write_transform.tif")
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
    da = _local_write_make_data_array(nodata=sentinel)
    path = str(tmp_path / "release_gate_local_write_nodata.tif")
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


# =========================================================================== #
# Section: Codecs -- stable lossless set                                      #
# =========================================================================== #
#
# The release contract names a specific set of codecs as ``stable``:
# ``none``, ``deflate``, ``lzw``, ``packbits``, ``zstd``. Every one must
# round-trip pixels byte-for-byte on integer and float dtypes.

def _codecs_gt() -> GeoTransform:
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
    path = str(tmp_path / f"release_gate_codec_{codec}_uint16.tif")
    write(
        arr,
        path,
        geo_transform=_codecs_gt(),
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
    arr = np.linspace(-100.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / f"release_gate_codec_{codec}_float32.tif")
    write(
        arr,
        path,
        geo_transform=_codecs_gt(),
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
    """The stable codec list matches ``SUPPORTED_FEATURES``.

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


# =========================================================================== #
# Section: COG (writer.cog / reader.local_cog, stable)                        #
# =========================================================================== #
#
# The release contract: ``to_geotiff(cog=True, compression=<stable
# lossless>)`` writes a file ``open_geotiff`` reads back bit-exact, with
# CRS, transform, and (when declared) nodata preserved across every
# stable codec.

_COG_W = 32
_COG_H = 32


def _cog_make_data_array(*, nodata: float | None = None) -> xr.DataArray:
    pixels = np.arange(_COG_H * _COG_W, dtype=np.float32).reshape(_COG_H, _COG_W)
    # Pixel-center coords, 30 m pixels, top-left at (500000, 4000000).
    y = np.array(
        [4000000.0 - 15.0 - 30.0 * i for i in range(_COG_H)],
        dtype=np.float64,
    )
    x = np.array(
        [500000.0 + 15.0 + 30.0 * i for i in range(_COG_W)],
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
    da = _cog_make_data_array()
    path = str(tmp_path / f"release_gate_cog_{codec}_pixels.tif")
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
    da = _cog_make_data_array()
    path = str(tmp_path / f"release_gate_cog_{codec}_attrs.tif")
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
    da = _cog_make_data_array(nodata=sentinel)
    path = str(tmp_path / f"release_gate_cog_{codec}_nodata.tif")
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


# =========================================================================== #
# Section: Windowed read (reader.windowed, stable)                            #
# =========================================================================== #
#
# A ``(row_start, col_start, row_stop, col_stop)`` window returns the
# exact subset of source pixels, preserves CRS, and shifts the
# transform origin to the window's top-left pixel corner.

_WINDOWED_H = 10
_WINDOWED_W = 10
_WINDOWED_PIXELS = (
    np.arange(_WINDOWED_H, dtype=np.int32).reshape(-1, 1) * 100
    + np.arange(_WINDOWED_W, dtype=np.int32).reshape(1, -1)
).astype(np.int32)
_WINDOWED_ORIGIN_X = 500000.0
_WINDOWED_ORIGIN_Y = 4000000.0
_WINDOWED_PIXEL_W = 30.0
_WINDOWED_PIXEL_H = -30.0


def _windowed_write_known_good(path: str) -> None:
    gt = GeoTransform(
        origin_x=_WINDOWED_ORIGIN_X,
        origin_y=_WINDOWED_ORIGIN_Y,
        pixel_width=_WINDOWED_PIXEL_W,
        pixel_height=_WINDOWED_PIXEL_H,
    )
    write(
        _WINDOWED_PIXELS,
        path,
        geo_transform=gt,
        crs_epsg=32610,
        compression="none",
        tiled=False,
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_returns_subset(tmp_path) -> None:
    """A windowed read returns exactly the requested subset."""
    path = str(tmp_path / "release_gate_windowed_read_subset.tif")
    _windowed_write_known_good(path)

    row_start, col_start = 2, 3
    row_stop, col_stop = 6, 8
    out = open_geotiff(path, window=(row_start, col_start, row_stop, col_stop))

    expected = _WINDOWED_PIXELS[row_start:row_stop, col_start:col_stop]
    assert out.shape == expected.shape, (
        f"release gate: windowed read shape {out.shape} does not match "
        f"the requested window shape {expected.shape}"
    )
    np.testing.assert_array_equal(
        np.asarray(out.values),
        expected,
        err_msg=(
            "release gate: windowed read returned different pixels than "
            "the same rows / cols of the source array; this would silently "
            "break every downstream caller that relies on window= for "
            "subsetting"
        ),
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_preserves_crs(tmp_path) -> None:
    """A windowed read carries ``attrs['crs']`` over from the source."""
    path = str(tmp_path / "release_gate_windowed_read_crs.tif")
    _windowed_write_known_good(path)

    out = open_geotiff(path, window=(1, 1, 5, 5))
    crs = out.attrs.get("crs")
    assert crs is not None and int(crs) == 32610, (
        f"release gate: windowed read dropped or drifted "
        f"``attrs['crs']``: got {crs!r}"
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_shifts_transform_origin(tmp_path) -> None:
    """The transform origin shifts to the window's top-left pixel.

    For a window starting at ``(row, col) = (2, 3)`` on a grid with pixel
    width ``+30`` and pixel height ``-30``, the new origin is
    ``(origin_x + 3 * 30, origin_y + 2 * -30)``.
    """
    path = str(tmp_path / "release_gate_windowed_read_transform.tif")
    _windowed_write_known_good(path)

    row_start, col_start = 2, 3
    out = open_geotiff(path, window=(row_start, col_start, 6, 8))
    transform = out.attrs.get("transform")
    assert transform is not None and len(transform) == 6, (
        f"release gate: windowed read dropped or reshaped transform: "
        f"{transform!r}"
    )
    assert transform[0] == pytest.approx(_WINDOWED_PIXEL_W, abs=1e-9), (
        f"release gate: windowed read changed pixel_width: {transform!r}"
    )
    assert transform[4] == pytest.approx(_WINDOWED_PIXEL_H, abs=1e-9), (
        f"release gate: windowed read changed pixel_height: {transform!r}"
    )
    expected_origin_x = _WINDOWED_ORIGIN_X + col_start * _WINDOWED_PIXEL_W
    expected_origin_y = _WINDOWED_ORIGIN_Y + row_start * _WINDOWED_PIXEL_H
    assert transform[2] == pytest.approx(expected_origin_x, abs=1e-9), (
        f"release gate: windowed read origin_x did not shift to the "
        f"window's left edge: got {transform[2]!r} expected "
        f"{expected_origin_x!r}"
    )
    assert transform[5] == pytest.approx(expected_origin_y, abs=1e-9), (
        f"release gate: windowed read origin_y did not shift to the "
        f"window's top edge: got {transform[5]!r} expected "
        f"{expected_origin_y!r}"
    )


@pytest.mark.release_gate
def test_release_gate_windowed_read_full_extent_matches_unwindowed(
    tmp_path,
) -> None:
    """``window=(0, 0, H, W)`` returns the same pixels as no window."""
    path = str(tmp_path / "release_gate_windowed_read_full.tif")
    _windowed_write_known_good(path)

    full = open_geotiff(path)
    windowed = open_geotiff(path, window=(0, 0, _WINDOWED_H, _WINDOWED_W))
    assert windowed.shape == full.shape, (
        f"release gate: full-extent window shape drift: "
        f"{windowed.shape} vs {full.shape}"
    )
    np.testing.assert_array_equal(
        np.asarray(windowed.values),
        np.asarray(full.values),
        err_msg=(
            "release gate: full-extent window returned different pixels "
            "than the unwindowed read"
        ),
    )


# =========================================================================== #
# Section: Dask parity (reader.dask, stable)                                  #
# =========================================================================== #
#
# Dask reads of a local GeoTIFF must return the same pixels and canonical
# attrs as the eager (numpy) read. The small one-shot gate below is the
# release-engineer-facing test; the wider parity matrix lives elsewhere.
# Per-test ``_requires_dask`` skip so a minimal install still runs the
# pure-numpy gates above.


def _dask_parity_write_known_good(path: str) -> np.ndarray:
    """Write a small tiled GeoTIFF and return the source array."""
    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
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
        compression="deflate",
        tiled=True,
        tile_size=16,
    )
    return arr


@pytest.mark.release_gate
@_requires_dask
def test_release_gate_dask_read_matches_eager_pixels(tmp_path) -> None:
    """The dask backend returns the same pixels as the eager backend."""
    path = str(tmp_path / "release_gate_dask_parity_pixels.tif")
    _dask_parity_write_known_good(path)

    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=8)

    lazy_values = np.asarray(lazy.values)
    eager_values = np.asarray(eager.values)
    np.testing.assert_array_equal(
        lazy_values,
        eager_values,
        err_msg=(
            "release gate: dask backend returned different pixels than "
            "the eager backend; the release contract promises dask read "
            "parity for the local-file stable path"
        ),
    )
    assert lazy.dtype == eager.dtype, (
        f"release gate: dask backend changed dtype from {eager.dtype!r} "
        f"to {lazy.dtype!r}"
    )
    assert lazy.shape == eager.shape, (
        f"release gate: dask backend changed shape from {eager.shape!r} "
        f"to {lazy.shape!r}"
    )


@pytest.mark.release_gate
@_requires_dask
def test_release_gate_dask_read_matches_eager_attrs(tmp_path) -> None:
    """The dask backend produces the same canonical attrs as eager."""
    path = str(tmp_path / "release_gate_dask_parity_attrs.tif")
    _dask_parity_write_known_good(path)

    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=8)

    canonical = ("crs", "transform", "georef_status")
    for key in canonical:
        assert key in eager.attrs, (
            f"release gate: eager read is missing canonical attr "
            f"{key!r}; cannot compare backends"
        )
        assert key in lazy.attrs, (
            f"release gate: dask read is missing canonical attr "
            f"{key!r}; the release contract requires backend parity on "
            "canonical attrs"
        )
        eager_v = eager.attrs[key]
        lazy_v = lazy.attrs[key]
        if key == "transform":
            assert len(eager_v) == len(lazy_v) == 6
            for a, b in zip(eager_v, lazy_v):
                assert a == pytest.approx(b, abs=1e-12, rel=1e-12), (
                    f"release gate: transform drifted across backends: "
                    f"eager={eager_v!r} lazy={lazy_v!r}"
                )
        else:
            assert eager_v == lazy_v, (
                f"release gate: ``attrs[{key!r}]`` drifted across "
                f"backends: eager={eager_v!r} lazy={lazy_v!r}"
            )


@pytest.mark.release_gate
@_requires_dask
def test_release_gate_dask_read_is_lazy(tmp_path) -> None:
    """A ``chunks=`` read produces a dask-backed DataArray.

    Without this assertion, a regression that silently materialised the
    dask path into numpy could pass the pixel-parity test above without
    anyone noticing. The dask backend's defining property is laziness;
    pin it.
    """
    import dask.array as da_mod

    path = str(tmp_path / "release_gate_dask_parity_lazy.tif")
    _dask_parity_write_known_good(path)

    lazy = open_geotiff(path, chunks=8)
    assert isinstance(lazy.data, da_mod.Array), (
        f"release gate: chunks= read returned a non-dask array of type "
        f"{type(lazy.data).__name__}; the release contract promises a "
        "dask-backed DataArray when chunks= is set"
    )


# =========================================================================== #
# Section: Eager / dask full parity                                           #
# =========================================================================== #
#
# Pixels matching while ``attrs``, ``coords``, or ``dims`` silently
# disagree between the eager and dask paths is the highest release risk
# for the GeoTIFF surface. This block reads each fixture in a small
# representative corpus once through ``open_geotiff`` and once through
# ``read_geotiff_dask``, then asserts full raster equivalence.

# Corpus fixtures live under ``golden_corpus/fixtures``.
_EAGER_DASK_FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "golden_corpus" / "fixtures"
)
_EAGER_DASK_CHUNK_SIZE = 32

# The seven release-attr keys the parity contract pins.
_EAGER_DASK_RELEASE_ATTR_KEYS: tuple[str, ...] = (
    "transform",
    "crs",
    "crs_wkt",
    "nodata",
    "masked_nodata",
    "georef_status",
    "raster_type",
)

_EAGER_DASK_CORPUS = [
    pytest.param(
        "nodata_int_sentinel_uint16", {}, id="int-dtype-nodata",
    ),
    pytest.param(
        "nodata_nan_float32", {}, id="float-dtype-nan-nodata",
    ),
    pytest.param(
        "nodata_miniswhite_uint8", {}, id="miniswhite",
    ),
    pytest.param(
        "nodata_int_sentinel_uint16",
        {"mask_nodata": False},
        id="masked-nodata-lifecycle",
    ),
]


def _eager_dask_materialise(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values)


def _eager_dask_assert_values_equal(
    eager: xr.DataArray, lazy: xr.DataArray,
) -> None:
    assert eager.dtype == lazy.dtype, (
        f"pixel dtype differs: eager={eager.dtype} lazy={lazy.dtype}"
    )
    eager_px = _eager_dask_materialise(eager)
    lazy_px = _eager_dask_materialise(lazy)
    assert eager_px.shape == lazy_px.shape, (
        f"pixel shape differs: eager={eager_px.shape} lazy={lazy_px.shape}"
    )
    equal_nan = eager_px.dtype.kind == "f"
    if not np.array_equal(eager_px, lazy_px, equal_nan=equal_nan):
        raise AssertionError(
            "pixel values differ between eager and dask reads "
            f"(dtype={eager_px.dtype}, equal_nan={equal_nan})"
        )


def _eager_dask_assert_dims_equal(
    eager: xr.DataArray, lazy: xr.DataArray,
) -> None:
    assert eager.dims == lazy.dims, (
        f"dims differ: eager={eager.dims!r} lazy={lazy.dims!r}"
    )


def _eager_dask_assert_coords_equal(
    eager: xr.DataArray, lazy: xr.DataArray,
) -> None:
    eager_coord_names = set(eager.coords)
    lazy_coord_names = set(lazy.coords)
    assert eager_coord_names == lazy_coord_names, (
        f"coord name set differs: "
        f"only-in-eager={sorted(eager_coord_names - lazy_coord_names)} "
        f"only-in-lazy={sorted(lazy_coord_names - eager_coord_names)}"
    )
    for axis in eager_coord_names:
        eager_c = np.asarray(eager.coords[axis].values)
        lazy_c = np.asarray(lazy.coords[axis].values)
        assert eager_c.dtype == lazy_c.dtype, (
            f"coord {axis!r} dtype differs: "
            f"eager={eager_c.dtype} lazy={lazy_c.dtype}"
        )
        assert eager_c.shape == lazy_c.shape, (
            f"coord {axis!r} shape differs: "
            f"eager={eager_c.shape} lazy={lazy_c.shape}"
        )
        assert eager_c.tobytes() == lazy_c.tobytes(), (
            f"coord {axis!r} bytes differ between eager and dask reads"
        )


def _eager_dask_is_nan_sentinel(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _eager_dask_attr_equal(a: Any, b: Any) -> bool:
    if _eager_dask_is_nan_sentinel(a) and _eager_dask_is_nan_sentinel(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and np.array_equal(a, b)
        )
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_eager_dask_attr_equal(x, y) for x, y in zip(a, b))
    return a == b


def _eager_dask_assert_release_attrs_equal(
    eager: xr.DataArray, lazy: xr.DataArray,
) -> None:
    for key in _EAGER_DASK_RELEASE_ATTR_KEYS:
        in_eager = key in eager.attrs
        in_lazy = key in lazy.attrs
        assert in_eager == in_lazy, (
            f"release attr {key!r} presence differs: "
            f"eager={in_eager} lazy={in_lazy}"
        )
        if not in_eager:
            continue
        eager_v = eager.attrs[key]
        lazy_v = lazy.attrs[key]
        assert _eager_dask_attr_equal(eager_v, lazy_v), (
            f"release attr {key!r} value differs: "
            f"eager={eager_v!r} lazy={lazy_v!r}"
        )


@pytest.mark.release_gate
@_requires_dask
@pytest.mark.parametrize("fixture_id, open_kwargs", _EAGER_DASK_CORPUS)
def test_release_gate_eager_dask_full_parity(
    fixture_id: str, open_kwargs: dict,
) -> None:
    """Eager and dask reads of the same file agree on the full contract."""
    path = _EAGER_DASK_FIXTURES_DIR / f"{fixture_id}.tif"
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    eager = open_geotiff(str(path), **open_kwargs)
    lazy = read_geotiff_dask(
        str(path), chunks=_EAGER_DASK_CHUNK_SIZE, **open_kwargs,
    )

    _eager_dask_assert_values_equal(eager, lazy)
    _eager_dask_assert_dims_equal(eager, lazy)
    _eager_dask_assert_coords_equal(eager, lazy)
    _eager_dask_assert_release_attrs_equal(eager, lazy)


@pytest.mark.release_gate
def test_release_gate_corpus_is_non_empty() -> None:
    """The corpus list must not silently shrink to zero rows."""
    assert len(_EAGER_DASK_CORPUS) == 4, (
        f"corpus row count drifted: expected 4 scenarios "
        f"(int-nodata, float-nan-nodata, miniswhite, "
        f"masked-nodata-lifecycle), got {len(_EAGER_DASK_CORPUS)}"
    )


# =========================================================================== #
# Section: Attrs contract (canonical attrs, stable)                           #
# =========================================================================== #
#
# Every georeferenced read produces a DataArray whose ``attrs`` carry, at
# minimum, ``crs``, ``crs_wkt``, ``transform``, ``georef_status``, the
# contract version stamp, and (when declared) ``nodata``. These attrs
# survive a write -> read round trip.

_ATTRS_CANONICAL_KEYS = (
    "_xrspatial_geotiff_contract",
    "crs",
    "crs_wkt",
    "transform",
    "georef_status",
)


def _attrs_write_known_good(
    path: str, *, nodata: float | None = None,
) -> None:
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
    path = str(tmp_path / "release_gate_attrs_canonical.tif")
    _attrs_write_known_good(path)

    da = open_geotiff(path)
    missing = [k for k in _ATTRS_CANONICAL_KEYS if k not in da.attrs]
    assert not missing, (
        "release gate: canonical attrs keys missing from a georeferenced "
        f"read: {missing}; release notes promise every key in "
        f"{list(_ATTRS_CANONICAL_KEYS)}"
    )


@pytest.mark.release_gate
def test_release_gate_attrs_georef_status_full(tmp_path) -> None:
    """A fully-georeferenced read reports ``georef_status='full'``."""
    path = str(tmp_path / "release_gate_attrs_georef_status.tif")
    _attrs_write_known_good(path)

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

    The contract version is the downstream signal for which attrs shape
    the array carries. A drift from int to string (or to a Python object)
    would silently break callers that compare versions.
    """
    path = str(tmp_path / "release_gate_attrs_contract_version.tif")
    _attrs_write_known_good(path)

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
    """Canonical attrs survive ``write -> read -> write -> read``."""
    src = str(tmp_path / "release_gate_attrs_rt_src.tif")
    _attrs_write_known_good(src, nodata=-9999.0)

    first = open_geotiff(src)
    crs_first = int(first.attrs["crs"])
    transform_first = tuple(first.attrs["transform"])
    nodata_first = float(first.attrs["nodata"])

    rewrite = str(tmp_path / "release_gate_attrs_rt_rewrite.tif")
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
        nodata_first, abs=0.0,
    ), (
        f"release gate: nodata drifted across round-trip: "
        f"{nodata_first} -> {second.attrs['nodata']!r}"
    )


# =========================================================================== #
# Section: Codec round-trip (cartesian stable codec x dtype)                  #
# =========================================================================== #
#
# Cartesian product of every stable codec with every promised dtype,
# asserting both pixel equality AND release-attr equality through a full
# read/write/read cycle.

_ROUND_TRIP_DTYPES = ("int16", "int32", "float32", "float64")

_ROUND_TRIP_INT_NODATA = {
    "int16": np.int16(-32768),
    "int32": np.int32(-2147483648),
}

_ROUND_TRIP_RELEASE_ATTR_KEYS = (
    "transform",
    "crs",
    "crs_wkt",
    "nodata",
    "masked_nodata",
    "georef_status",
    "raster_type",
)


def _round_trip_make_input(dtype_name: str) -> xr.DataArray:
    """Build a 128x128 DataArray of the given dtype."""
    dtype = np.dtype(dtype_name)
    height, width = 128, 128
    n = height * width
    if np.issubdtype(dtype, np.floating):
        arr = np.linspace(-100.0, 100.0, n, dtype=dtype).reshape(height, width)
        arr[0, 0] = np.nan
        nodata: float | int = float("nan")
    else:
        # Small positive ramp; the dtype min sentinel never collides with
        # a real pixel for the supported int dtypes.
        arr = np.arange(n, dtype=dtype).reshape(height, width)
        sentinel = _ROUND_TRIP_INT_NODATA[dtype_name]
        arr[0, 0] = sentinel
        nodata = sentinel

    y = 4000000.0 - 30.0 * (np.arange(height) + 0.5)
    x = 500000.0 + 30.0 * (np.arange(width) + 0.5)
    attrs: dict = {"crs": 32610, "nodata": nodata}
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs=attrs,
    )


def _round_trip_canonical_attrs(da: xr.DataArray) -> dict:
    """Project ``attrs`` onto the release-attr key set."""
    out = {}
    for key in _ROUND_TRIP_RELEASE_ATTR_KEYS:
        if key == "raster_type":
            out[key] = da.attrs.get("raster_type", "area")
        else:
            out[key] = da.attrs.get(key)
    return out


def _round_trip_read_tiff_compression_tag(path: str) -> int:
    """Read the on-disk TIFF Compression tag from the first IFD."""
    with open(path, "rb") as fh:
        data = fh.read()
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    return ifd.compression


def _round_trip_assert_pixels_equal(
    actual: np.ndarray, expected: np.ndarray, *,
    codec: str, dtype_name: str,
) -> None:
    """NaN-aware byte-exact pixel comparison."""
    assert actual.shape == expected.shape, (
        f"release gate: codec {codec!r} dtype {dtype_name!r} reshaped the "
        f"array across the round-trip: {expected.shape} -> {actual.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"release gate: codec {codec!r} promoted dtype {dtype_name!r} to "
        f"{actual.dtype!r} across the round-trip"
    )
    if np.issubdtype(expected.dtype, np.floating):
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        if np.issubdtype(expected.dtype, np.floating):
            mismatch_mask = ~(
                (actual == expected) | (np.isnan(actual) & np.isnan(expected))
            )
        else:
            mismatch_mask = actual != expected
        first = np.argwhere(mismatch_mask)
        first_idx = tuple(int(v) for v in first[0]) if first.size else None
        first_actual = (
            actual[first_idx] if first_idx is not None else None
        )
        first_expected = (
            expected[first_idx] if first_idx is not None else None
        )
        raise AssertionError(
            f"release gate: codec {codec!r} did not preserve "
            f"{dtype_name!r} pixels byte-for-byte; the release contract "
            f"names this codec as lossless for this dtype. First "
            f"divergence at index {first_idx!r}: actual="
            f"{first_actual!r}, expected={first_expected!r}"
        )


@pytest.mark.release_gate
@pytest.mark.parametrize("dtype_name", _ROUND_TRIP_DTYPES)
@pytest.mark.parametrize("codec", STABLE_LOSSLESS_CODECS)
def test_release_gate_codec_round_trip(
    tmp_path, codec, dtype_name,
) -> None:
    """Stable codec * dtype: pixels and release attrs survive read/write/read."""
    nonce = uuid.uuid4().hex[:8]
    write_first = str(
        tmp_path / f"release_gate_rt_{codec}_{dtype_name}_first_{nonce}.tif"
    )
    write_second = str(
        tmp_path / f"release_gate_rt_{codec}_{dtype_name}_second_{nonce}.tif"
    )

    source = _round_trip_make_input(dtype_name)
    is_float = np.issubdtype(np.dtype(dtype_name), np.floating)

    mask_kwargs: dict = {} if is_float else {"mask_nodata": False}

    pass_one_kwargs: dict = (
        {} if is_float else {"nodata": source.attrs["nodata"]}
    )
    to_geotiff(
        source,
        write_first,
        compression=codec,
        tiled=False,
        **pass_one_kwargs,
    )

    baseline = open_geotiff(write_first, **mask_kwargs)
    baseline_pixels = np.asarray(baseline.values)
    baseline_attrs = _round_trip_canonical_attrs(baseline)

    tag_first = _round_trip_read_tiff_compression_tag(write_first)
    assert tag_first == _CODEC_TO_TIFF_TAG[codec], (
        f"release gate: codec {codec!r} encoded as TIFF tag {tag_first} on "
        f"first write; expected {_CODEC_TO_TIFF_TAG[codec]} per the codec "
        f"-> tag map"
    )

    pass_two_kwargs: dict = (
        {} if is_float else {"nodata": baseline.attrs.get("nodata")}
    )
    to_geotiff(
        baseline,
        write_second,
        compression=codec,
        tiled=False,
        **pass_two_kwargs,
    )

    second = open_geotiff(write_second, **mask_kwargs)
    second_pixels = np.asarray(second.values)
    second_attrs = _round_trip_canonical_attrs(second)

    tag_second = _round_trip_read_tiff_compression_tag(write_second)
    assert tag_second == _CODEC_TO_TIFF_TAG[codec], (
        f"release gate: codec {codec!r} encoded as TIFF tag {tag_second} on "
        f"the second write; expected {_CODEC_TO_TIFF_TAG[codec]} per the "
        f"codec -> tag map"
    )

    _round_trip_assert_pixels_equal(
        second_pixels, baseline_pixels, codec=codec, dtype_name=dtype_name,
    )

    for key in _ROUND_TRIP_RELEASE_ATTR_KEYS:
        want = baseline_attrs[key]
        got = second_attrs[key]
        if key == "nodata" and isinstance(want, float) and np.isnan(want):
            assert isinstance(got, float) and np.isnan(got), (
                f"release gate: codec {codec!r} dtype {dtype_name!r} "
                f"dropped NaN nodata across the round-trip: got {got!r}"
            )
            continue
        if key == "transform":
            assert want is not None and got is not None, (
                f"release gate: codec {codec!r} dtype {dtype_name!r} "
                f"dropped ``attrs['transform']``: {want!r} -> {got!r}"
            )
            assert tuple(got) == tuple(want), (
                f"release gate: codec {codec!r} dtype {dtype_name!r} "
                f"drifted ``attrs['transform']``: {want!r} -> {got!r}"
            )
            continue
        assert got == want, (
            f"release gate: codec {codec!r} dtype {dtype_name!r} drifted "
            f"``attrs[{key!r}]`` across the round-trip: {want!r} -> {got!r}"
        )


@pytest.mark.release_gate
def test_release_gate_codec_round_trip_stable_set_matches_supported_features() -> None:
    """The codec list used by the round-trip gate matches ``SUPPORTED_FEATURES``."""
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


# =========================================================================== #
# Section: Overview / sidecar metadata survival                              #
# =========================================================================== #
#
# For an internal-overview COG and for a file whose overviews live in an
# external ``.ovr`` sidecar, per-level ``attrs`` agree on the canonical
# metadata set, and ``transform`` scales pixel size by the level factor
# while keeping the origin fixed.

# rasterio and dask.array are required only by this section; the
# imports live inside the helpers so a minimal install still runs the
# pure-numpy gates above. ``_requires_rasterio`` / ``_requires_rasterio_and_dask``
# skip the affected tests cleanly.

_OVERVIEW_BASE_SIZE = 64
_OVERVIEW_FACTORS = (2, 4)
_OVERVIEW_BASE_TRANSFORM = (1.0, 0.0, -120.0, 0.0, -1.0, 45.0)
_OVERVIEW_BASE_CRS = 4326
_OVERVIEW_NODATA = -9999.0

_OVERVIEW_EQUAL_KEYS = (
    "crs",
    "crs_wkt",
    "georef_status",
    "raster_type",
    "nodata",
    "masked_nodata",
)


def _overview_make_raster() -> xr.DataArray:
    arr = np.arange(
        _OVERVIEW_BASE_SIZE * _OVERVIEW_BASE_SIZE,
        dtype=np.float32,
    ).reshape(_OVERVIEW_BASE_SIZE, _OVERVIEW_BASE_SIZE)
    arr[0, 0] = np.nan
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        attrs={
            "transform": _OVERVIEW_BASE_TRANSFORM,
            "crs": _OVERVIEW_BASE_CRS,
        },
    )


def _overview_unique_tmp_path(tmp_path, label: str) -> str:
    return str(tmp_path / f"release_gate_overview_{label}_{uuid.uuid4().hex}.tif")


def _overview_write_internal_cog(path: str) -> None:
    """Write a COG with base + internal overviews at factors 2 and 4."""
    import rasterio  # gated by ``_requires_rasterio`` on every caller

    da = _overview_make_raster()
    to_geotiff(
        da, path,
        nodata=_OVERVIEW_NODATA,
        cog=True,
        compression="deflate",
        tiled=True,
        tile_size=16,
        overview_levels=list(_OVERVIEW_FACTORS),
        overview_resampling="nearest",
    )
    with rasterio.open(path) as ds:
        assert ds.overviews(1) == list(_OVERVIEW_FACTORS), (
            f"writer did not emit the requested overview IFDs: "
            f"got {ds.overviews(1)}, expected {list(_OVERVIEW_FACTORS)}"
        )


def _overview_write_external_sidecar(path: str) -> None:
    """Write a tiled TIFF + ``.ovr`` sidecar at factors 2 and 4."""
    import rasterio  # gated by ``_requires_rasterio`` on every caller
    from rasterio.enums import Resampling

    da = _overview_make_raster()
    to_geotiff(
        da, path,
        nodata=_OVERVIEW_NODATA,
        tiled=True,
        tile_size=16,
    )
    with rasterio.open(path) as ds:
        assert ds.overviews(1) == [], (
            "base file must have no internal overviews before sidecar build"
        )

    with rasterio.Env(TIFF_USE_OVR="YES", COMPRESS_OVERVIEW="DEFLATE"):
        with rasterio.open(path, "r+") as ds:
            ds.build_overviews(list(_OVERVIEW_FACTORS), Resampling.nearest)
    assert os.path.exists(path + ".ovr"), (
        "TIFF_USE_OVR=YES must produce a .ovr sidecar next to the base file"
    )


def _overview_assert_metadata_equal_across_levels(
    attrs_by_level: dict,
) -> None:
    base = attrs_by_level[0]
    for key in _OVERVIEW_EQUAL_KEYS:
        base_present = key in base
        base_val = base.get(key)
        for lvl, attrs in attrs_by_level.items():
            if lvl == 0:
                continue
            other_present = key in attrs
            other_val = attrs.get(key)
            assert other_present == base_present, (
                f"attrs[{key!r}] presence drifts: base={base_present}, "
                f"level={lvl}: {other_present}"
            )
            if base_present:
                assert other_val == base_val, (
                    f"attrs[{key!r}] differs across levels: "
                    f"base={base_val!r} level={lvl}: {other_val!r}"
                )


def _overview_assert_transform_scales(
    attrs_by_level: dict, factors: dict,
) -> None:
    base = attrs_by_level[0]["transform"]
    base_a, base_b, base_c, base_d, base_e, base_f = base
    for lvl, attrs in attrs_by_level.items():
        factor = factors[lvl]
        t = attrs["transform"]
        a, b, c, d, e, f = t
        assert a == pytest.approx(base_a * factor), (
            f"level {lvl}: pixel width did not scale by {factor}: "
            f"got {a}, expected {base_a * factor}"
        )
        assert e == pytest.approx(base_e * factor), (
            f"level {lvl}: pixel height did not scale by {factor}: "
            f"got {e}, expected {base_e * factor}"
        )
        assert c == pytest.approx(base_c), (
            f"level {lvl}: origin x drifted: got {c}, expected {base_c}"
        )
        assert f == pytest.approx(base_f), (
            f"level {lvl}: origin y drifted: got {f}, expected {base_f}"
        )
        assert b == pytest.approx(0.0) and d == pytest.approx(0.0), (
            f"level {lvl}: axis-aligned transform must not gain rotation "
            f"terms, got b={b}, d={d}"
        )


def _overview_read_levels_eager(path: str) -> dict:
    out = {0: open_geotiff(path)}
    for i, _ in enumerate(_OVERVIEW_FACTORS, start=1):
        out[i] = open_geotiff(path, overview_level=i)
    return out


def _overview_read_levels_dask(path: str) -> dict:
    out = {0: read_geotiff_dask(path, chunks=8)}
    for i, _ in enumerate(_OVERVIEW_FACTORS, start=1):
        out[i] = read_geotiff_dask(path, chunks=8, overview_level=i)
    return out


def _overview_factors_by_level() -> dict:
    factors = {0: 1}
    for i, f in enumerate(_OVERVIEW_FACTORS, start=1):
        factors[i] = f
    return factors


@pytest.mark.release_gate
@_requires_rasterio_and_dask
@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_release_gate_cog_internal_overview_metadata_survives(
    tmp_path, reader,
) -> None:
    """COG with internal overviews preserves the metadata contract."""
    path = _overview_unique_tmp_path(tmp_path, f"cog_meta_{reader}")
    _overview_write_internal_cog(path)

    if reader == "eager":
        levels = _overview_read_levels_eager(path)
    else:
        levels = _overview_read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _overview_assert_metadata_equal_across_levels(attrs_by_level)

    base = attrs_by_level[0]
    assert base.get("crs") == _OVERVIEW_BASE_CRS
    assert base.get("crs_wkt"), "crs_wkt must be set on a CRS-carrying COG"
    assert base.get("nodata") == _OVERVIEW_NODATA
    assert base.get("masked_nodata") is True
    assert base.get("georef_status") == "full"


@pytest.mark.release_gate
@_requires_rasterio_and_dask
@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_release_gate_cog_internal_overview_transform_scales(
    tmp_path, reader,
) -> None:
    """COG with internal overviews preserves origin and scales pixel size."""
    path = _overview_unique_tmp_path(tmp_path, f"cog_xform_{reader}")
    _overview_write_internal_cog(path)

    if reader == "eager":
        levels = _overview_read_levels_eager(path)
    else:
        levels = _overview_read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _overview_assert_transform_scales(
        attrs_by_level, _overview_factors_by_level(),
    )


@pytest.mark.release_gate
@_requires_rasterio
def test_release_gate_cog_internal_overview_shape_matches_factors(
    tmp_path,
) -> None:
    """Shapes follow the decimation factors so the test exercises real overview IFDs."""
    path = _overview_unique_tmp_path(tmp_path, "cog_shape")
    _overview_write_internal_cog(path)

    base = open_geotiff(path)
    assert base.shape == (_OVERVIEW_BASE_SIZE, _OVERVIEW_BASE_SIZE)
    for i, factor in enumerate(_OVERVIEW_FACTORS, start=1):
        da = open_geotiff(path, overview_level=i)
        expected = _OVERVIEW_BASE_SIZE // factor
        assert da.shape == (expected, expected), (
            f"overview_level={i} returned shape {da.shape}, "
            f"expected ({expected}, {expected})"
        )


@pytest.mark.release_gate
@_requires_rasterio_and_dask
@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_release_gate_sidecar_overview_metadata_survives(
    tmp_path, reader,
) -> None:
    """External `.ovr` sidecar preserves the metadata contract."""
    path = _overview_unique_tmp_path(tmp_path, f"sidecar_meta_{reader}")
    _overview_write_external_sidecar(path)

    if reader == "eager":
        levels = _overview_read_levels_eager(path)
    else:
        levels = _overview_read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _overview_assert_metadata_equal_across_levels(attrs_by_level)

    base = attrs_by_level[0]
    assert base.get("crs") == _OVERVIEW_BASE_CRS
    assert base.get("crs_wkt"), (
        "crs_wkt must be set when the base file carries an EPSG code"
    )
    assert base.get("nodata") == _OVERVIEW_NODATA
    assert base.get("masked_nodata") is True
    assert base.get("georef_status") == "full"


@pytest.mark.release_gate
@_requires_rasterio_and_dask
@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_release_gate_sidecar_overview_transform_scales(
    tmp_path, reader,
) -> None:
    """External `.ovr` sidecar scales pixel size by 2 per level, origin held."""
    path = _overview_unique_tmp_path(tmp_path, f"sidecar_xform_{reader}")
    _overview_write_external_sidecar(path)

    if reader == "eager":
        levels = _overview_read_levels_eager(path)
    else:
        levels = _overview_read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _overview_assert_transform_scales(
        attrs_by_level, _overview_factors_by_level(),
    )


@pytest.mark.release_gate
@_requires_rasterio
def test_release_gate_sidecar_overview_shape_matches_factors(
    tmp_path,
) -> None:
    """Sidecar reads return the right shape per level."""
    path = _overview_unique_tmp_path(tmp_path, "sidecar_shape")
    _overview_write_external_sidecar(path)

    base = open_geotiff(path)
    assert base.shape == (_OVERVIEW_BASE_SIZE, _OVERVIEW_BASE_SIZE)
    for i, factor in enumerate(_OVERVIEW_FACTORS, start=1):
        da = open_geotiff(path, overview_level=i)
        expected = _OVERVIEW_BASE_SIZE // factor
        assert da.shape == (expected, expected), (
            f"sidecar overview_level={i} returned shape {da.shape}, "
            f"expected ({expected}, {expected})"
        )


@pytest.mark.release_gate
@_requires_rasterio_and_dask
@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_release_gate_internal_vs_sidecar_metadata_agree(
    tmp_path, reader,
) -> None:
    """Internal COG and external sidecar agree on the metadata contract."""
    cog_path = _overview_unique_tmp_path(tmp_path, f"parity_cog_{reader}")
    sidecar_path = _overview_unique_tmp_path(
        tmp_path, f"parity_sidecar_{reader}",
    )
    _overview_write_internal_cog(cog_path)
    _overview_write_external_sidecar(sidecar_path)

    if reader == "eager":
        cog_levels = _overview_read_levels_eager(cog_path)
        sidecar_levels = _overview_read_levels_eager(sidecar_path)
    else:
        cog_levels = _overview_read_levels_dask(cog_path)
        sidecar_levels = _overview_read_levels_dask(sidecar_path)

    assert set(cog_levels) == set(sidecar_levels)
    for lvl in cog_levels:
        for key in _OVERVIEW_EQUAL_KEYS:
            cog_attrs = cog_levels[lvl].attrs
            sidecar_attrs = sidecar_levels[lvl].attrs
            assert (key in cog_attrs) == (key in sidecar_attrs), (
                f"level {lvl}: attrs[{key!r}] presence differs between "
                f"internal-COG and sidecar reads "
                f"(cog={key in cog_attrs}, sidecar={key in sidecar_attrs})"
            )
            if key in cog_attrs:
                assert cog_attrs[key] == sidecar_attrs[key], (
                    f"level {lvl}: attrs[{key!r}] differs between "
                    f"internal-COG and sidecar reads: "
                    f"cog={cog_attrs[key]!r}, "
                    f"sidecar={sidecar_attrs[key]!r}"
                )
        assert cog_levels[lvl].attrs["transform"] == pytest.approx(
            sidecar_levels[lvl].attrs["transform"]
        ), (
            f"level {lvl}: transform differs between internal-COG and "
            f"sidecar reads"
        )


# =========================================================================== #
# Section: Windowed reads -- shifted-transform parity                        #
# =========================================================================== #
#
# Windowed reads must return shapes matching the requested window,
# coords that are a bit-exact slice of the unwindowed read, an
# ``attrs['transform']`` shifted by ``Affine.translation(col_off,
# row_off)`` exactly, and the canonical non-transform release attrs
# unchanged.

_wsp_skip_no_tifffile = pytest.mark.skipif(
    not _HAS_TIFFFILE,
    reason="tifffile required for MinIsWhite fixture",
)

_WSP_FULL_H = 256
_WSP_FULL_W = 256

_WSP_WINDOWS = (
    pytest.param(
        (32, 64, 96, 192), id="aligned-row32-col64-h64-w128",
    ),
    pytest.param(
        (33, 65, 95, 191), id="chunk-misaligned-row33-col65-h62-w126",
    ),
)

_WSP_PIXEL_WIDTH = 30.0
_WSP_PIXEL_HEIGHT = -25.0
_WSP_ORIGIN_X = 500123.5
_WSP_ORIGIN_Y = 4001987.25

_WSP_NON_TRANSFORM_ATTRS_VALUE_EQUAL = (
    "crs",
    "crs_wkt",
    "nodata",
    "georef_status",
    "raster_type",
)
_WSP_NON_TRANSFORM_ATTRS_STRUCTURAL = ("masked_nodata",)


def _wsp_build_da(arr: np.ndarray, nodata=None) -> xr.DataArray:
    h, w = arr.shape
    x_centers = (
        _WSP_ORIGIN_X + _WSP_PIXEL_WIDTH * 0.5
        + _WSP_PIXEL_WIDTH * np.arange(w)
    )
    y_centers = (
        _WSP_ORIGIN_Y + _WSP_PIXEL_HEIGHT * 0.5
        + _WSP_PIXEL_HEIGHT * np.arange(h)
    )
    attrs = {"crs": 32610}
    if nodata is not None:
        attrs["nodata"] = nodata
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": y_centers.astype(np.float64),
                "x": x_centers.astype(np.float64)},
        attrs=attrs,
    )


def _wsp_write_int16_with_nodata(path: Path) -> None:
    rng = np.random.default_rng(2341)
    arr = rng.integers(
        -1000, 1000, size=(_WSP_FULL_H, _WSP_FULL_W), dtype=np.int16,
    )
    arr[10, 10] = -9999
    arr[200, 5] = -9999
    da = _wsp_build_da(arr, nodata=-9999)
    to_geotiff(da, str(path), compression="deflate", tiled=False)


def _wsp_write_float32_with_nan_nodata(path: Path) -> None:
    rng = np.random.default_rng(2342)
    arr = (rng.standard_normal((_WSP_FULL_H, _WSP_FULL_W)) * 100).astype(
        np.float32,
    )
    arr[40, 80] = np.nan
    da = _wsp_build_da(arr, nodata=np.float32("nan"))
    to_geotiff(
        da, str(path), compression="deflate", tiled=True, tile_size=64,
    )


def _wsp_write_float32_no_nodata(path: Path) -> None:
    rng = np.random.default_rng(2343)
    arr = (rng.standard_normal((_WSP_FULL_H, _WSP_FULL_W)) * 50).astype(
        np.float32,
    )
    da = _wsp_build_da(arr, nodata=None)
    to_geotiff(da, str(path), compression="none", tiled=False)


def _wsp_write_uint8_miniswhite(path: Path) -> None:
    """Write a MinIsWhite (photometric=0) uint8 stripped TIFF via tifffile."""
    import tifffile  # local import: gated by _wsp_skip_no_tifffile
    rng = np.random.default_rng(2344)
    arr = rng.integers(0, 256, size=(_WSP_FULL_H, _WSP_FULL_W), dtype=np.uint8)
    tifffile.imwrite(
        str(path), arr, photometric="miniswhite",
        compression="none", metadata=None,
    )


_WSP_CORPUS = (
    pytest.param(
        _wsp_write_int16_with_nodata, id="int16-deflate-stripped-nodata",
    ),
    pytest.param(
        _wsp_write_float32_with_nan_nodata,
        id="float32-deflate-tiled-nan-nodata",
    ),
    pytest.param(
        _wsp_write_float32_no_nodata, id="float32-none-stripped-no-nodata",
    ),
    pytest.param(
        _wsp_write_uint8_miniswhite,
        id="uint8-miniswhite-stripped",
        marks=_wsp_skip_no_tifffile,
    ),
)


@pytest.fixture
def wsp_corpus_file(tmp_path, request):
    """Write a single fixture file and return its on-disk path."""
    builder = request.param
    tag = uuid.uuid4().hex[:8]
    path = tmp_path / f"release_gate_wsp_{builder.__name__[1:]}_{tag}.tif"
    builder(path)
    return path


def _wsp_expected_window_transform(t_full, col_off, row_off):
    a, b, c, d, e, f = (float(x) for x in t_full)
    return (
        a,
        b,
        c + a * col_off + b * row_off,
        d,
        e,
        f + d * col_off + e * row_off,
    )


def _wsp_open_eager(path, *, window=None):
    return open_geotiff(str(path), window=window)


def _wsp_open_dask(path, *, window=None):
    return read_geotiff_dask(str(path), window=window, chunks=32)


_WSP_READERS = (
    pytest.param(_wsp_open_eager, id="eager"),
    pytest.param(_wsp_open_dask, id="dask", marks=_requires_dask),
)


def _wsp_assert_shape(out, *, expected_h, expected_w):
    assert out.shape == (expected_h, expected_w), (
        f"release gate: windowed read shape {out.shape} does not equal "
        f"the requested window shape {(expected_h, expected_w)}; a window "
        f"that returns the wrong shape is silently wrong, not noisily wrong"
    )


def _wsp_assert_coords_slice(
    windowed, full, *, row_off, col_off, height, width,
):
    full_y = np.asarray(full.coords["y"].values)
    full_x = np.asarray(full.coords["x"].values)
    win_y = np.asarray(windowed.coords["y"].values)
    win_x = np.asarray(windowed.coords["x"].values)
    np.testing.assert_array_equal(
        win_y,
        full_y[row_off:row_off + height],
        err_msg=(
            "release gate: windowed read y-coords are not a slice of the "
            "unwindowed read's y-coords; downstream callers that join on "
            "y will silently mismatch"
        ),
    )
    np.testing.assert_array_equal(
        win_x,
        full_x[col_off:col_off + width],
        err_msg=(
            "release gate: windowed read x-coords are not a slice of the "
            "unwindowed read's x-coords"
        ),
    )


def _wsp_assert_transform_shifted(windowed, full, *, col_off, row_off):
    if "transform" not in full.attrs:
        assert "transform" not in windowed.attrs, (
            f"release gate: source has no georef and the unwindowed read "
            f"emits no ``transform``, but the windowed read invented one: "
            f"{windowed.attrs.get('transform')!r}"
        )
        return
    t_full = tuple(full.attrs["transform"])
    assert "transform" in windowed.attrs, (
        f"release gate: unwindowed read carries ``transform`` "
        f"({t_full!r}) but the windowed read dropped it"
    )
    t_win = tuple(windowed.attrs["transform"])
    assert len(t_full) == 6, (
        f"release gate: full-read transform is not a 6-tuple: {t_full!r}"
    )
    assert len(t_win) == 6, (
        f"release gate: windowed-read transform is not a 6-tuple: {t_win!r}"
    )
    expected = _wsp_expected_window_transform(t_full, col_off, row_off)
    assert t_win == expected, (
        f"release gate: windowed transform does not equal "
        f"T_full * Affine.translation(col_off={col_off}, "
        f"row_off={row_off})\n"
        f"  T_full   = {t_full!r}\n"
        f"  T_window = {t_win!r}\n"
        f"  expected = {expected!r}"
    )


def _wsp_assert_canonical_attrs_unchanged(windowed, full):
    for key in _WSP_NON_TRANSFORM_ATTRS_VALUE_EQUAL:
        if key not in full.attrs:
            assert key not in windowed.attrs, (
                f"release gate: windowed read introduced attrs[{key!r}] "
                f"that the unwindowed read does not have"
            )
            continue
        assert key in windowed.attrs, (
            f"release gate: windowed read dropped attrs[{key!r}] that the "
            f"unwindowed read carries"
        )
        full_val = full.attrs[key]
        win_val = windowed.attrs[key]
        try:
            full_is_nan = bool(np.isnan(full_val))
        except (TypeError, ValueError):
            full_is_nan = False
        if full_is_nan:
            try:
                win_is_nan = bool(np.isnan(win_val))
            except (TypeError, ValueError):
                win_is_nan = False
            assert win_is_nan, (
                f"release gate: NaN-valued attrs[{key!r}] did not survive "
                f"the windowed read: full={full_val!r} window={win_val!r}"
            )
        else:
            assert win_val == full_val, (
                f"release gate: windowed read changed attrs[{key!r}]: "
                f"full={full_val!r} window={win_val!r}"
            )
    for key in _WSP_NON_TRANSFORM_ATTRS_STRUCTURAL:
        full_present = key in full.attrs
        win_present = key in windowed.attrs
        assert full_present == win_present, (
            f"release gate: windowed read changed presence of "
            f"attrs[{key!r}]: full_has={full_present} "
            f"window_has={win_present}"
        )
        if full_present:
            assert isinstance(windowed.attrs[key], bool), (
                f"release gate: attrs[{key!r}] is not a bool on the "
                f"windowed read: {windowed.attrs[key]!r}"
            )


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WSP_WINDOWS)
@pytest.mark.parametrize("wsp_corpus_file", _WSP_CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _WSP_READERS)
def test_release_gate_windowed_read_shape(wsp_corpus_file, reader, window):
    """The returned shape equals the window's ``(height, width)``."""
    row_off, col_off, row_stop, col_stop = window
    out = reader(wsp_corpus_file, window=window)
    _wsp_assert_shape(
        out,
        expected_h=row_stop - row_off,
        expected_w=col_stop - col_off,
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WSP_WINDOWS)
@pytest.mark.parametrize("wsp_corpus_file", _WSP_CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _WSP_READERS)
def test_release_gate_windowed_read_coords_slice(
    wsp_corpus_file, reader, window,
):
    """``coords['y'/'x']`` equals the matching slice of the full coords."""
    row_off, col_off, row_stop, col_stop = window
    full = reader(wsp_corpus_file)
    out = reader(wsp_corpus_file, window=window)
    _wsp_assert_coords_slice(
        out, full,
        row_off=row_off, col_off=col_off,
        height=row_stop - row_off, width=col_stop - col_off,
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WSP_WINDOWS)
@pytest.mark.parametrize("wsp_corpus_file", _WSP_CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _WSP_READERS)
def test_release_gate_windowed_read_transform_shifted(
    wsp_corpus_file, reader, window,
):
    """``attrs['transform']`` equals ``T_full * translation(col, row)``."""
    row_off, col_off, _row_stop, _col_stop = window
    full = reader(wsp_corpus_file)
    out = reader(wsp_corpus_file, window=window)
    _wsp_assert_transform_shifted(
        out, full, col_off=col_off, row_off=row_off,
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WSP_WINDOWS)
@pytest.mark.parametrize("wsp_corpus_file", _WSP_CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _WSP_READERS)
def test_release_gate_windowed_read_canonical_attrs_unchanged(
    wsp_corpus_file, reader, window,
):
    """The non-transform canonical attrs match the unwindowed read."""
    full = reader(wsp_corpus_file)
    out = reader(wsp_corpus_file, window=window)
    _wsp_assert_canonical_attrs_unchanged(out, full)


# =========================================================================== #
# Section: Negative cases -- ambiguous metadata fails closed                 #
# =========================================================================== #
#
# When metadata is ambiguous and the caller did NOT opt in via the
# documented flag, every promised read entry point raises a typed error
# whose message names the unlocking flag and points at the release-contract
# docs.

_NEG_RELEASE_CONTRACT_HINTS = (
    "release_gate_geotiff",
    "geotiff_release_contract",
    "#2341",
    "#1987",
    "#2342",
    "release contract",
)


def _neg_msg_cites_release_contract(msg: str) -> bool:
    return any(hint in msg for hint in _NEG_RELEASE_CONTRACT_HINTS)


def _neg_tmp(tmp_path, label: str, *, suffix: str = ".tif") -> str:
    return str(
        tmp_path / f"release_gate_neg_{label}_{uuid.uuid4().hex}{suffix}"
    )


# Hand-rolled rotated TIFF builder (case 3). The writer refuses rotated
# transforms at the boundary, so a round-trip through xrspatial cannot
# reproduce one. The 30-degree rotation matches
# ``test_allow_rotated_geotiff_2115.py`` so the gate rejects the same
# input shape that test pins behaviourally.

_NEG_TAG_MODEL_TRANSFORMATION = 34264
_NEG_COS30 = 0.8660254037844387
_NEG_SIN30 = 0.5
_NEG_ROTATED_M = (
    10.0 * _NEG_COS30, -10.0 * _NEG_SIN30, 0.0, 100.0,
    10.0 * _NEG_SIN30, 10.0 * _NEG_COS30, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _neg_write_rotated_tiff(path: str, arr: np.ndarray) -> None:
    """Emit a minimal little-endian TIFF with a rotated ModelTransformationTag."""
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries: list[tuple[int, int, int, int]] = []
    entries.append((256, 3, 1, w))            # ImageWidth
    entries.append((257, 3, 1, h))            # ImageLength
    entries.append((258, 3, 1, 16))           # BitsPerSample
    entries.append((259, 3, 1, 1))            # Compression none
    entries.append((262, 3, 1, 1))            # Photometric BlackIsZero
    entries.append((273, 4, 1, header_size))  # StripOffsets
    entries.append((277, 3, 1, 1))            # SamplesPerPixel
    entries.append((278, 3, 1, h))            # RowsPerStrip
    entries.append((279, 4, 1, strip_size))   # StripByteCounts
    entries.append((339, 3, 1, 1))            # SampleFormat uint
    entries.append((_NEG_TAG_MODEL_TRANSFORMATION, 12, 16, transform_off))

    entries.sort(key=lambda e: e[0])
    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)

    with open(path, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *_NEG_ROTATED_M))
        f.write(ifd_bytes)


def _neg_build_uint16_tiff_with_nodata(nodata_str: str, path: str) -> None:
    """Emit a 2x2 uint16 TIFF whose GDAL_NODATA tag holds ``nodata_str``."""
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_bytes = nodata_str.encode('ascii') + b'\x00'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag: int, val: int) -> None:
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag: int, val: int) -> None:
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_ascii(tag: int, data: bytes) -> None:
        tag_list.append((tag, 2, len(data), data))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 16)
    add_short(259, 1)
    add_short(262, 1)
    add_short(277, 1)
    add_short(278, height)
    add_short(339, 1)
    add_long(273, 0)
    add_long(279, len(pixels.tobytes()))
    add_ascii(42113, nodata_bytes)  # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])

    header_size = 8
    num_entries = len(tag_list)
    ifd_size = 2 + 12 * num_entries + 4
    ifd_off = header_size

    overflow = bytearray()
    overflow_start = header_size + ifd_size

    overflow_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_offsets[tag] = len(overflow)
            overflow.extend(raw)
            if len(overflow) % 2:
                overflow.append(0)
        else:
            overflow_offsets[tag] = None

    pixel_start = overflow_start + len(overflow)

    patched: list[tuple[int, int, int, bytes]] = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append(
                (tag, typ, count, struct.pack(f'{bo}I', pixel_start))
            )
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_off))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + overflow_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow)
    out.extend(pixels.tobytes())

    with open(path, 'wb') as f:
        f.write(out)


@pytest.mark.release_gate
@pytest.mark.xfail(
    reason=(
        "xrspatial.geotiff does not yet read ``.aux.xml`` PAM sidecars "
        "(no entry in SUPPORTED_FEATURES). When that support lands, the "
        "reader must fail closed on a CRS conflict between the header and "
        "the sidecar; this xfail flips to a pass at that point. Tracked "
        "alongside the PAM sidecar epic."
    ),
    strict=False,
)
def test_release_gate_negative_conflicting_aux_xml_crs(tmp_path) -> None:
    """The reader must not silently choose between header CRS and sidecar CRS."""
    path = _neg_tmp(tmp_path, "case1_aux_xml_crs")
    pixels = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write(
        pixels,
        path,
        geo_transform=GeoTransform(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=-1.0,
        ),
        crs_epsg=4326,
        compression="none",
        tiled=False,
    )
    sidecar = Path(path + ".aux.xml")
    sidecar.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PAMDataset>\n'
        '  <SRS dataAxisToSRSAxisMapping="1,2">EPSG:3857</SRS>\n'
        '</PAMDataset>\n',
        encoding="utf-8",
    )
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert "aux.xml" in msg or "sidecar" in msg or "PAM" in msg, (
        f"expected the error message to name the .aux.xml / PAM sidecar; "
        f"got: {msg!r}"
    )
    assert _neg_msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


@pytest.mark.release_gate
def test_release_gate_negative_integer_nodata_float_promoted(
    tmp_path,
) -> None:
    """The reader must not silently coerce a non-finite int-file nodata sentinel."""
    path = _neg_tmp(tmp_path, "case2_int_nodata_float_promoted")
    _neg_build_uint16_tiff_with_nodata("nan", path)
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert "nodata" in msg.lower(), (
        f"expected the error message to name nodata; got: {msg!r}"
    )
    assert _neg_msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


_NEG_ROTATED_PIXELS = np.arange(20, dtype='<u2').reshape(4, 5)


@pytest.fixture
def _neg_rotated_geotiff_path(tmp_path) -> str:
    """A throwaway rotated GeoTIFF that the rotated case's sub-tests share."""
    path = _neg_tmp(tmp_path, "case3_rotated")
    _neg_write_rotated_tiff(path, _NEG_ROTATED_PIXELS)
    return path


def _neg_assert_rotated_message(msg: str) -> None:
    """Shared assertions on the rotated error message."""
    assert "allow_rotated" in msg, (
        f"expected the error message to name the ``allow_rotated`` "
        f"opt-in; got: {msg!r}"
    )
    assert any(
        tier in msg for tier in ("advanced", "experimental", "stable")
    ), (
        f"expected the error message to name the feature tier; "
        f"got: {msg!r}"
    )
    assert _neg_msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


@pytest.mark.release_gate
def test_release_gate_negative_rotated_eager(
    _neg_rotated_geotiff_path,
) -> None:
    """Eager numpy path raises ``RotatedTransformError`` without the opt-in."""
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(_neg_rotated_geotiff_path)
    _neg_assert_rotated_message(str(excinfo.value))


@pytest.mark.release_gate
@_requires_dask
def test_release_gate_negative_rotated_dask(
    _neg_rotated_geotiff_path,
) -> None:
    """Dask path raises the same typed error, uniformly with the eager path."""
    with pytest.raises(RotatedTransformError) as excinfo:
        read_geotiff_dask(_neg_rotated_geotiff_path, chunks=2)
    _neg_assert_rotated_message(str(excinfo.value))


@pytest.mark.release_gate
def test_release_gate_negative_rotated_windowed(
    _neg_rotated_geotiff_path,
) -> None:
    """Windowed read raises the same typed error before pixel decode."""
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(_neg_rotated_geotiff_path, window=(0, 0, 2, 2))
    _neg_assert_rotated_message(str(excinfo.value))


@pytest.mark.release_gate
@requires_gpu
def test_release_gate_negative_rotated_gpu(
    _neg_rotated_geotiff_path,
) -> None:
    """GPU read raises the same typed error as the CPU paths.

    The rotated-transform refusal is upstream of the GPU decode path --
    the validator fires on the header read, before any pixel buffer
    reaches the GPU -- so the same typed error surfaces here regardless
    of the GPU tier.
    """
    with pytest.raises(RotatedTransformError) as excinfo:
        open_geotiff(_neg_rotated_geotiff_path, gpu=True)
    _neg_assert_rotated_message(str(excinfo.value))


@pytest.mark.release_gate
def test_release_gate_negative_mixed_tier_vrt_children(tmp_path) -> None:
    """The reader must refuse mixed-tier VRT children when stable-only is asked.

    When the caller asks for stable-only sources via
    ``stable_only=True`` and the source is a VRT, the read raises
    :class:`VRTStableSourcesOnlyError` (a
    :class:`GeoTIFFAmbiguousMetadataError` subclass) before any pixel
    decode. The message must name either ``stable_only`` or
    ``allow_experimental_codecs`` and cite the release-contract docs
    or the tracking issue.
    """
    path = _neg_tmp(tmp_path, "case4_mixed_tier_vrt", suffix=".vrt")
    Path(path).write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2"></VRTDataset>\n',
        encoding="utf-8",
    )
    with pytest.raises(GeoTIFFAmbiguousMetadataError) as excinfo:
        open_geotiff(path, stable_only=True)  # type: ignore[call-arg]
    msg = str(excinfo.value)
    assert "stable_only" in msg or "allow_experimental_codecs" in msg, (
        f"expected the error message to name the unlocking opt-in; "
        f"got: {msg!r}"
    )
    assert _neg_msg_cites_release_contract(msg), (
        f"expected the error message to cite the release-contract docs "
        f"or the tracking issue; got: {msg!r}"
    )


# =========================================================================== #
# Section: Cross-cutting meta-gates                                          #
# =========================================================================== #
#
# Checklist-parity gates: every file cited in the release-gate checklist
# exists on disk, every promised SUPPORTED_FEATURES key is mentioned in
# the prose, the HTTP SSRF presence gate stays loud, and every cited VRT
# test file actually carries test functions.

_META_HERE = Path(__file__).resolve()
_META_REPO_ROOT = _META_HERE.parents[4]
_META_CHECKLIST = (
    _META_REPO_ROOT
    / "docs" / "source" / "reference" / "release_gate_geotiff.rst"
)

# Match xrspatial/geotiff/tests/<name>.py inside the checklist body. The
# leaf may include slashes (e.g. ``release_gates/test_stable_features.py``).
_META_TEST_PATH_RE = re.compile(
    r"`{0,2}(xrspatial/geotiff/tests/[\w/]+\.py)`{0,2}",
)

# The new home of this file. The cited-test parser drops the self-reference
# so the gate cannot succeed only because it cites itself.
_META_SELF_REF = (
    "xrspatial/geotiff/tests/release_gates/test_stable_features.py"
)


def _meta_checklist_text() -> str:
    assert _META_CHECKLIST.is_file(), (
        f"release gate checklist missing: {_META_CHECKLIST}; the "
        "checklist must ship with the geotiff docs so release notes can "
        "cite it"
    )
    return _META_CHECKLIST.read_text(encoding="utf-8")


def _meta_cited_test_files() -> set[str]:
    text = _meta_checklist_text()
    matches = set(_META_TEST_PATH_RE.findall(text))
    matches.discard(_META_SELF_REF)
    return matches


@pytest.mark.release_gate
def test_release_gate_cites_only_existing_test_files() -> None:
    """Every test file cited in the release gate checklist exists on disk."""
    cited = _meta_cited_test_files()
    assert cited, (
        "release gate checklist cites zero test files; either the regex "
        "is wrong or the checklist is empty"
    )
    missing = sorted(p for p in cited if not (_META_REPO_ROOT / p).is_file())
    assert not missing, (
        "release gate checklist cites tests that do not exist on disk; "
        "rename the checklist row to a real file or restore the test: "
        f"{missing}"
    )
    # Tighten: every cited path must point at a ``test_*.py`` file, not at
    # ``conftest.py`` or a helper module.
    non_test = sorted(
        p for p in cited if not Path(p).name.startswith("test_")
    )
    assert not non_test, (
        "release gate checklist cites paths that do not start with "
        "``test_``; the checklist should point at regression tests, not "
        f"conftest or helper modules: {non_test}"
    )


_META_PROMISED_TIERS = {"stable", "advanced"}


def _meta_checklist_mentions(text: str, key: str) -> bool:
    """``key`` is something like ``reader.local_file``."""
    if key in text:
        return True
    return f"SUPPORTED_FEATURES['{key}']" in text


@pytest.mark.release_gate
def test_release_gate_lists_every_promised_supported_feature() -> None:
    """Every promised SUPPORTED_FEATURES entry appears in the checklist."""
    text = _meta_checklist_text()
    missing = []
    for key, tier in SUPPORTED_FEATURES.items():
        if tier not in _META_PROMISED_TIERS:
            continue
        if key.startswith("codec."):
            continue
        if not _meta_checklist_mentions(text, key):
            missing.append((key, tier))
    assert not missing, (
        "promised SUPPORTED_FEATURES entries are missing from the release "
        "gate checklist; add a row (or update SUPPORTED_FEATURES) so the "
        "release notes can quote the tier: "
        f"{missing}"
    )


@pytest.mark.release_gate
def test_release_gate_http_ssrf_rejects_loopback() -> None:
    """HTTP URLs targeting loopback hosts raise :class:`UnsafeURLError`."""
    with pytest.raises(UnsafeURLError):
        # No network call -- the SSRF check rejects before the socket
        # opens, so this test is offline-safe.
        open_geotiff("http://127.0.0.1/does-not-matter.tif")


@pytest.mark.release_gate
def test_release_gate_http_ssrf_rejects_loopback_uppercase_scheme() -> None:
    """Uppercase HTTP scheme must take the same SSRF path.

    The SSRF check is case-insensitive, so uppercase HTTP raises
    :class:`UnsafeURLError` like its lowercase sibling.
    """
    with pytest.raises(UnsafeURLError):
        open_geotiff("HTTP://127.0.0.1/does-not-matter.tif")


def _meta_vrt_section_test_files() -> set[str]:
    """Return the test files cited inside the VRT supported subset section."""
    text = _meta_checklist_text()
    start = text.find("VRT supported subset")
    assert start != -1, "checklist is missing the VRT supported subset section"
    end = text.find("Sidecar and overview interactions", start)
    if end == -1:
        end = len(text)
    section = text[start:end]
    return set(_META_TEST_PATH_RE.findall(section))


@pytest.mark.release_gate
def test_release_gate_vrt_rows_point_at_real_test_functions() -> None:
    """Every VRT row cites a file that contains at least one ``def test_``."""
    files = _meta_vrt_section_test_files()
    assert files, "no VRT test files cited in the checklist"
    empty = []
    for rel in sorted(files):
        if rel == _META_SELF_REF:
            continue
        path = _META_REPO_ROOT / rel
        if not path.is_file():
            # Caught by the cited-test-files gate; skip here.
            continue
        body = path.read_text(encoding="utf-8")
        if "def test_" not in body:
            empty.append(rel)
    assert not empty, (
        "VRT checklist rows cite files with no test functions; either the "
        "file was emptied or the row should be removed: "
        f"{empty}"
    )


# =========================================================================== #
# Section: COG stability contract parity (issue #2513)                        #
# =========================================================================== #
#
# Three surfaces have to agree on whether COG writes are stable:
#
# 1. The runtime tier registry ``SUPPORTED_FEATURES`` in ``_attrs.py``.
# 2. The release contract / reference docs in ``docs/source/reference/``.
# 3. The ``to_geotiff`` docstring's release-contract tier block and the
#    per-parameter ``[tier]`` marker on ``cog``.
#
# Issue #2513 documented that surface (3) declared ``cog=True`` as
# ``[advanced]`` while (1) and (2) said ``stable``. This gate locks the
# resolution: ``writer.cog`` is the stable COG layout contract;
# ``writer.overviews`` is the separately tracked advanced pyramid
# customisation surface. If a future change demotes ``writer.cog`` or
# re-introduces an ``[advanced]`` framing on the ``cog`` docstring, this
# gate fails so the drift is caught before release.


@pytest.mark.release_gate
def test_release_gate_writer_cog_stays_stable() -> None:
    """``SUPPORTED_FEATURES['writer.cog']`` is the stable COG layout entry."""
    assert SUPPORTED_FEATURES.get("writer.cog") == "stable", (
        "release gate: SUPPORTED_FEATURES['writer.cog'] is no longer "
        "'stable'. The release contract and the to_geotiff docstring "
        "promise stable COG writes; demoting the registry entry breaks "
        "that contract. See issue #2513."
    )


@pytest.mark.release_gate
def test_release_gate_writer_overviews_stays_advanced() -> None:
    """``writer.overviews`` is the advanced sub-behaviour of COG writes.

    Issue #2513: the stable COG layout (``writer.cog``) and the advanced
    overview-customisation surface (``writer.overviews``) are tracked as
    two separate registry entries so they can promote independently. If
    overview customisation gets promoted, the docstring and contract
    have to be updated together.
    """
    assert SUPPORTED_FEATURES.get("writer.overviews") == "advanced", (
        "release gate: SUPPORTED_FEATURES['writer.overviews'] is no "
        "longer 'advanced'. If overview customisation has been promoted "
        "(or demoted), update the to_geotiff docstring's "
        "[advanced]/[stable] markers on overview_levels and "
        "overview_resampling and the matching rows in "
        "docs/source/reference/geotiff_release_contract.md and "
        "docs/source/reference/geotiff.rst together. See issue #2513."
    )


@pytest.mark.release_gate
def test_release_gate_to_geotiff_docstring_marks_cog_stable() -> None:
    """The ``to_geotiff`` docstring marks ``cog=True`` as ``[stable]``.

    Tripwire for issue #2513: an earlier version of the docstring
    described ``cog=True`` as ``[advanced]``, contradicting the registry
    and the release contract. This assertion fails if the contradiction
    creeps back in. The check is deliberately strict on the wording so
    a copy-paste from another parameter cannot satisfy it accidentally.

    ``inspect.getdoc`` normalises docstring indentation across every
    supported Python (3.12 keeps the source-level indent on ``__doc__``;
    3.13+ strips the common leading whitespace at compile time). The
    regexes anchor at column 0 of the cleaned text so the assertion
    matches on every supported version.
    """
    doc = inspect.getdoc(to_geotiff) or ""
    # The function-level tier block must list cog=True under [stable].
    # Match the bullet body across line wraps without taking a hard
    # dependency on a single line layout.
    stable_bullet_re = re.compile(
        r"\*\s+\[stable\][^*]*?cog=True",
        re.DOTALL,
    )
    assert stable_bullet_re.search(doc), (
        "release gate: the to_geotiff docstring's [stable] tier bullet "
        "no longer mentions ``cog=True``. The COG layout is stable per "
        "SUPPORTED_FEATURES['writer.cog']; the docstring has to agree. "
        "See issue #2513."
    )

    # The per-parameter marker on the ``cog`` parameter must be [stable].
    # ``inspect.getdoc`` already removed the common indent, so the
    # parameter line is at column 0 and the body is indented one level.
    cog_param_re = re.compile(
        r"^cog : bool\n    \[(?P<tier>[\w-]+)\]",
        re.MULTILINE,
    )
    match = cog_param_re.search(doc)
    assert match is not None, (
        "release gate: cannot find the ``cog`` parameter docstring "
        "block in to_geotiff. The tier-marker regex needs updating, or "
        "the parameter was renamed."
    )
    assert match.group("tier") == "stable", (
        "release gate: to_geotiff's ``cog`` parameter is marked "
        f"[{match.group('tier')}] in its docstring. "
        "SUPPORTED_FEATURES['writer.cog'] is stable, so the docstring "
        "marker has to be [stable] too. See issue #2513."
    )


@pytest.mark.release_gate
def test_release_gate_to_geotiff_docstring_marks_overview_knobs_advanced() -> None:
    """``overview_levels`` and ``overview_resampling`` stay ``[advanced]``.

    The pyramid-customisation surface is the advanced sub-behaviour of
    COG writes (``SUPPORTED_FEATURES['writer.overviews'] == 'advanced'``).
    If those knobs ever get promoted, this gate fails together with the
    registry gate above so the change is forced through both surfaces.
    Uses ``inspect.getdoc`` for the same cross-version reason as the
    ``cog`` gate above.
    """
    doc = inspect.getdoc(to_geotiff) or ""
    for param in ("overview_levels", "overview_resampling"):
        param_re = re.compile(
            rf"^{param} : [^\n]+\n    \[(?P<tier>[\w-]+)\]",
            re.MULTILINE,
        )
        match = param_re.search(doc)
        assert match is not None, (
            f"release gate: cannot find the ``{param}`` parameter "
            "docstring block in to_geotiff. The tier-marker regex "
            "needs updating, or the parameter was renamed."
        )
        assert match.group("tier") == "advanced", (
            f"release gate: to_geotiff's ``{param}`` parameter is "
            f"marked [{match.group('tier')}] in its docstring. "
            "SUPPORTED_FEATURES['writer.overviews'] is advanced, so the "
            "docstring marker has to be [advanced] too. See issue #2513."
        )
