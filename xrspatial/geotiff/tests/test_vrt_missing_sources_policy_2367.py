"""Consolidated VRT ``missing_sources`` policy matrix (#2367, work item of #2342).

This file complements ``test_vrt_missing_sources_policy_1799.py`` and
``test_vrt_chunked_missing_sources_1799.py`` by covering the full
release contract in one place: every policy value (default,
``'raise'``, ``'warn'``, invalid) is exercised against both read paths
(eager ``read_vrt`` and dask ``open_geotiff(..., chunks=...)``), with
assertions on the exception or warning type, the message text, and the
actual output array values where applicable.

The existing 1799 / 1843 / 2265 tests pin individual cases. This file
keeps the four-by-two matrix together so a future kwarg refactor that
silently drops parity between the eager and chunked paths regresses a
single, focused test file.

Release contract (see ``_backends/vrt.py:206`` docstring):

* ``'raise'`` is the default since #1860.
* ``'raise'`` fails fast with ``FileNotFoundError`` naming the missing
  source path. The chunked path raises at build time (#2265) so a
  partial mosaic never surfaces silently from a delayed compute.
* ``'warn'`` is the explicit opt-in. It emits
  ``GeoTIFFFallbackWarning`` naming the missing source and returns the
  mosaic with NaN (or the band's nodata sentinel) in the corresponding
  region. ``attrs['vrt_holes']`` records the affected source(s).
* Any other value raises ``ValueError`` naming the bad kwarg.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    GeoTIFFFallbackWarning,
    open_geotiff,
    read_vrt,
    to_geotiff,
)


PRESENT_FILL = 7.0


def _build_partial_vrt(tmp_path) -> tuple[str, str, str]:
    """Build a 2-source VRT: left half is real, right half points at a
    non-existent file.

    Returns ``(vrt_path, present_src_path, missing_path)``. Filenames
    embed issue #2367 to keep parallel test runs from colliding on
    shared tmp roots.
    """
    src = os.path.join(tmp_path, "src_2367_present.tif")
    arr = np.full((4, 4), PRESENT_FILL, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)

    missing = os.path.join(tmp_path, "missing_2367.tif")
    vrt_path = os.path.join(tmp_path, "partial_2367.vrt")
    with open(vrt_path, "w") as f:
        f.write(
            '<VRTDataset rasterXSize="8" rasterYSize="4">\n'
            '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
            '<VRTRasterBand dataType="Float32" band="1">\n'
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{src}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '<DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{missing}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '<DstRect xOff="4" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
            '</VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
    return vrt_path, src, missing


# ---------------------------------------------------------------------------
# Reader-path fixtures. Each "reader" callable accepts ``(source,
# **kwargs)`` and returns a DataArray. The eager reader returns a numpy-
# backed array; the dask reader returns a chunked DataArray that still
# needs ``.compute()`` to materialise values.
# ---------------------------------------------------------------------------

def _eager_reader(source, **kwargs):
    return read_vrt(source, **kwargs)


def _dask_reader(source, **kwargs):
    # ``open_geotiff`` routes ``.vrt`` to ``read_vrt`` and forwards
    # ``chunks=`` / ``missing_sources=`` unchanged. Using a small chunk
    # size keeps the partial mosaic split across multiple tasks so the
    # lazy path is genuinely exercised.
    return open_geotiff(source, chunks=4, **kwargs)


READERS = [
    pytest.param(_eager_reader, id="eager_read_vrt"),
    pytest.param(_dask_reader, id="dask_open_geotiff_chunks"),
]


# ---------------------------------------------------------------------------
# Default policy: no kwarg -> raises.
# ---------------------------------------------------------------------------

class TestDefaultPolicyRaises:
    """No ``missing_sources`` kwarg -> ``FileNotFoundError`` naming the
    missing source. This is the public default since #1860 and the
    release matrix in #2342 calls it out as a hard contract."""

    @pytest.mark.parametrize("reader", READERS)
    def test_default_raises_filenotfound_naming_source(
        self, reader, tmp_path,
    ):
        vrt_path, _, missing = _build_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            reader(vrt_path)
        # The basename of the missing source must appear in the
        # message. The chunked path quotes the full path; the eager
        # path may quote just the source filename or the resolved
        # absolute path depending on which guard fires first. Match on
        # the basename to stay portable across both.
        assert "missing_2367.tif" in str(excinfo.value), (
            f"Default policy raise must name the missing source. "
            f"Got: {excinfo.value!r}"
        )


# ---------------------------------------------------------------------------
# Explicit raise: same shape as default.
# ---------------------------------------------------------------------------

class TestExplicitRaisePolicy:
    """``missing_sources='raise'`` passed explicitly must behave the
    same as the default. Pins that an explicit opt-in does not
    accidentally route through a separate code branch."""

    @pytest.mark.parametrize("reader", READERS)
    def test_explicit_raise_matches_default(self, reader, tmp_path):
        vrt_path, _, _ = _build_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            reader(vrt_path, missing_sources="raise")
        assert "missing_2367.tif" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Warn opt-in: warning class, message, and output values all pinned.
# ---------------------------------------------------------------------------

class TestWarnPolicyEmitsWarningAndFillsNodata:
    """``missing_sources='warn'`` is the lenient opt-in.

    Three things to lock in:

    1. The warning class is ``GeoTIFFFallbackWarning`` (not a bare
       ``UserWarning``) and the message names the missing source.
    2. ``attrs['vrt_holes']`` records the affected source.
    3. The returned array shows ``PRESENT_FILL`` on the present half
       and NaN on the missing half. The eager path materialises this
       immediately; the chunked path needs ``.compute()`` and emits the
       warning at compute time rather than build time, but the
       resulting array values must match.
    """

    def test_eager_warn_emits_and_fills(self, tmp_path):
        vrt_path, _, missing = _build_partial_vrt(str(tmp_path))
        with pytest.warns(GeoTIFFFallbackWarning) as record:
            da = read_vrt(vrt_path, missing_sources="warn")

        # Warning message names the missing source.
        msgs = [str(w.message) for w in record]
        assert any("missing_2367.tif" in m for m in msgs), (
            f"Expected GeoTIFFFallbackWarning naming the missing "
            f"source; got messages: {msgs!r}"
        )

        # vrt_holes attr is populated and points at the missing file.
        assert "vrt_holes" in da.attrs
        sources = [h["source"] for h in da.attrs["vrt_holes"]]
        assert any(s.endswith("missing_2367.tif") for s in sources)

        # Output values: present half == 7.0, missing half == NaN.
        out = np.asarray(da)
        np.testing.assert_array_equal(
            out[:, :4], np.full((4, 4), PRESENT_FILL, dtype=np.float32),
        )
        assert np.all(np.isnan(out[:, 4:])), (
            "Lenient policy must leave the missing region as NaN on "
            "float bands."
        )

    def test_dask_warn_emits_at_compute_and_fills(self, tmp_path):
        vrt_path, _, missing = _build_partial_vrt(str(tmp_path))
        # Build the lazy DataArray. The parse-time sweep populates
        # ``vrt_holes`` here without forcing a decode.
        da = open_geotiff(
            vrt_path, chunks=4, missing_sources="warn",
        )
        assert "vrt_holes" in da.attrs, (
            "Chunked warn path must populate vrt_holes at build so "
            "callers can branch on partial mosaics without computing."
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            computed = da.compute()

        msgs = [
            str(w.message) for w in caught
            if isinstance(w.message, GeoTIFFFallbackWarning)
        ]
        assert any("missing_2367.tif" in m for m in msgs), (
            f"Chunked warn path must emit GeoTIFFFallbackWarning at "
            f"compute naming the missing source; got: {msgs!r}"
        )

        out = np.asarray(computed)
        np.testing.assert_array_equal(
            out[:, :4], np.full((4, 4), PRESENT_FILL, dtype=np.float32),
        )
        assert np.all(np.isnan(out[:, 4:]))


# ---------------------------------------------------------------------------
# Invalid policy strings.
# ---------------------------------------------------------------------------

class TestInvalidPolicyRejected:
    """Garbage values for ``missing_sources`` raise ``ValueError`` at
    the public-API boundary. The message must name the bad value so
    typos like ``'raises'`` surface clearly.

    Sanity for the chunked path too: the same value-validation block
    runs before ``_read_vrt_chunked`` dispatches, so the eager and
    chunked invocations both reject identically."""

    @pytest.mark.parametrize("reader", READERS)
    @pytest.mark.parametrize(
        "bad_value", ["ignore", "RAISE", "raises", "", "warn ", "1"],
    )
    def test_invalid_policy_raises_value_error_naming_value(
        self, reader, bad_value, tmp_path,
    ):
        vrt_path, _, _ = _build_partial_vrt(str(tmp_path))
        with pytest.raises(ValueError) as excinfo:
            reader(vrt_path, missing_sources=bad_value)
        msg = str(excinfo.value)
        assert "missing_sources" in msg, (
            f"ValueError must name the kwarg; got {msg!r}"
        )
        # The current implementation quotes the bad value via repr().
        # Use repr() here so the assertion stays robust across the few
        # acceptable formats (single quotes, double quotes, empty
        # string repr).
        assert repr(bad_value) in msg, (
            f"ValueError must echo the bad value back to the caller; "
            f"got {msg!r}"
        )
