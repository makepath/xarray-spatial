"""VRT ``missing_sources`` policy matrix.

Consolidates the eager-only smoke checks that previously lived in
``test_vrt_missing_sources_policy_1799.py`` with the eager-plus-chunked
matrix from ``test_vrt_missing_sources_policy_2367.py``.

Release contract (see ``_backends/vrt.py:206`` docstring):

* ``'raise'`` is the default. The eager and chunked paths both fail
  fast with ``FileNotFoundError`` naming the missing source path so a
  partial mosaic never surfaces silently from a delayed compute.
* ``'warn'`` is the explicit opt-in. It emits ``GeoTIFFFallbackWarning``
  naming the missing source and returns the mosaic with NaN (or the
  band's nodata sentinel) in the corresponding region.
  ``attrs['vrt_holes']`` records the affected source(s).
* Any other value raises ``ValueError`` naming the bad kwarg and
  echoing the bad value via ``repr()``.

The companion file ``test_vrt_missing_sources_default_raise_1843.py``
stays in place for now: it exercises the internal
``xrspatial.geotiff._vrt.read_vrt`` entry point and the
``XRSPATIAL_GEOTIFF_STRICT=1`` env-var override, neither of which is in
this module's surface.
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


# ---------------------------------------------------------------------------
# VRT fixtures.
#
# Two shapes:
#
# * ``byte_missing_vrt`` -- a 2x2 ``Byte`` VRT whose only source does not
#   exist on disk. The smallest case that exercises the missing-source
#   guard. Inherited from the old _1799 smoke checks.
# * ``partial_float_vrt`` -- an 8x4 ``Float32`` VRT split across two
#   sources. The left half points at a real GeoTIFF written through
#   ``to_geotiff``; the right half points at a missing file. Exercises
#   the NaN-fill / vrt_holes contract that the chunked path also has to
#   honour at compute time.
# ---------------------------------------------------------------------------

def _write_byte_missing_vrt(tmp_path) -> str:
    """All-missing 2x2 Byte VRT. Returns the VRT path as ``str``."""
    vrt = tmp_path / "byte_missing.vrt"
    vrt.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    return str(vrt)


def _write_partial_float_vrt(tmp_path) -> tuple[str, str, str]:
    """Two-source partial mosaic.

    Returns ``(vrt_path, present_src_path, missing_path)`` as strings.
    """
    tmp_str = str(tmp_path)
    src = os.path.join(tmp_str, "src_present.tif")
    arr = np.full((4, 4), PRESENT_FILL, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)

    missing = os.path.join(tmp_str, "missing_source.tif")
    vrt_path = os.path.join(tmp_str, "partial.vrt")
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
# Reader-path parametrisation. Each ``reader`` callable takes ``(source,
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
    pytest.param(_eager_reader, id="eager"),
    pytest.param(_dask_reader, id="dask"),
]


# ---------------------------------------------------------------------------
# Default policy: no kwarg -> raises.
# ---------------------------------------------------------------------------

class TestDefaultPolicyRaises:
    """No ``missing_sources`` kwarg -> ``FileNotFoundError`` naming the
    missing source. The public default since the lenient-by-default
    behaviour was removed."""

    @pytest.mark.parametrize("reader", READERS)
    def test_default_raises_filenotfound_naming_source(
        self, reader, tmp_path,
    ):
        vrt_path, _, missing = _write_partial_float_vrt(tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            reader(vrt_path)
        # The basename of the missing source must appear in the
        # message. Different code paths quote the full path vs just the
        # filename; matching on the basename keeps this portable.
        assert "missing_source.tif" in str(excinfo.value), (
            f"Default policy raise must name the missing source. "
            f"Got: {excinfo.value!r}"
        )

    def test_eager_byte_default_raises(self, tmp_path):
        """Smoke check for the byte-band path with no real source on
        disk. Inherited from the old _1799 file."""
        vrt = _write_byte_missing_vrt(tmp_path)
        with pytest.raises((OSError, ValueError)):
            read_vrt(vrt)


# ---------------------------------------------------------------------------
# Explicit raise: same shape as default.
# ---------------------------------------------------------------------------

class TestExplicitRaisePolicy:
    """``missing_sources='raise'`` passed explicitly must behave the
    same as the default. Pins that an explicit opt-in does not
    accidentally route through a separate code branch."""

    @pytest.mark.parametrize("reader", READERS)
    def test_explicit_raise_matches_default(self, reader, tmp_path):
        vrt_path, _, _ = _write_partial_float_vrt(tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            reader(vrt_path, missing_sources="raise")
        assert "missing_source.tif" in str(excinfo.value)

    def test_eager_byte_explicit_raise(self, tmp_path):
        vrt = _write_byte_missing_vrt(tmp_path)
        with pytest.raises((OSError, ValueError)):
            read_vrt(vrt, missing_sources="raise")


# ---------------------------------------------------------------------------
# Warn opt-in: warning class, message, vrt_holes, and array values pinned.
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
        vrt_path, _, missing = _write_partial_float_vrt(tmp_path)
        with pytest.warns(
            GeoTIFFFallbackWarning, match="missing_source.tif",
        ):
            da = read_vrt(vrt_path, missing_sources="warn")

        assert "vrt_holes" in da.attrs
        sources = [h["source"] for h in da.attrs["vrt_holes"]]
        assert any(s.endswith("missing_source.tif") for s in sources)

        out = np.asarray(da)
        np.testing.assert_array_equal(
            out[:, :4], np.full((4, 4), PRESENT_FILL, dtype=np.float32),
        )
        assert np.all(np.isnan(out[:, 4:])), (
            "Lenient policy must leave the missing region as NaN on "
            "float bands."
        )

    def test_dask_warn_emits_at_compute_and_fills(self, tmp_path):
        vrt_path, _, missing = _write_partial_float_vrt(tmp_path)
        # The parse-time sweep populates ``vrt_holes`` at build so
        # callers can branch on partial mosaics without computing.
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
        assert any("missing_source.tif" in m for m in msgs), (
            f"Chunked warn path must emit GeoTIFFFallbackWarning at "
            f"compute naming the missing source; got: {msgs!r}"
        )

        out = np.asarray(computed)
        np.testing.assert_array_equal(
            out[:, :4], np.full((4, 4), PRESENT_FILL, dtype=np.float32),
        )
        assert np.all(np.isnan(out[:, 4:]))

    def test_eager_byte_warn_records_hole(self, tmp_path):
        """Byte-band warn path: warning fires and ``vrt_holes`` is
        populated even when there is no present half. Inherited from
        the old _1799 file."""
        vrt = _write_byte_missing_vrt(tmp_path)
        with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
            da = read_vrt(vrt, missing_sources="warn")
        assert "vrt_holes" in da.attrs
        assert da.attrs["vrt_holes"][0]["source"].endswith("missing.tif")


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
        vrt_path, _, _ = _write_partial_float_vrt(tmp_path)
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

    def test_eager_byte_invalid_policy(self, tmp_path):
        """Smoke check carried over from the old _1799 file. The
        parametrised matrix above covers more bad values across both
        reader paths; this stays as a literal copy of the original
        assertion so the byte-band code path stays exercised."""
        vrt = _write_byte_missing_vrt(tmp_path)
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt, missing_sources="ignore")
