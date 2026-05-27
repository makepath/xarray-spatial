"""VRT ``missing_sources`` policy matrix.

Covers the eager and chunked policy matrix.

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

Also covers:

* Internal ``_vrt.read_vrt`` entry point default-raise + explicit-warn
  + ``XRSPATIAL_GEOTIFF_STRICT=1`` override.
* Public ``read_vrt`` / ``open_geotiff('.vrt')`` default-raise +
  explicit-warn.
* Chunked-path missing-source policy: ``vrt_holes`` at build,
  raise-at-build, per-task compute warnings, window / band scoping,
  multi-source error preview.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, read_vrt, to_geotiff
from xrspatial.geotiff._vrt import read_vrt as _internal_read_vrt

PRESENT_FILL = 7.0


# ---------------------------------------------------------------------------
# VRT fixtures.
#
# Two shapes:
#
# * ``byte_missing_vrt`` -- a 2x2 ``Byte`` VRT whose only source does not
#   exist on disk. The smallest case that exercises the missing-source
#   guard. Inherited from the old eager-only smoke checks.
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
    src = str(tmp_path / "src_present.tif")
    arr = np.full((4, 4), PRESENT_FILL, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)

    missing = str(tmp_path / "missing_source.tif")
    vrt_path = tmp_path / "partial.vrt"
    vrt_path.write_text(
        '<VRTDataset rasterXSize="8" rasterYSize="4">\n'
        '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="4" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    return str(vrt_path), src, missing


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
        disk."""
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
        populated even when there is no present half."""
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
        """Byte-band smoke check. The parametrised matrix above covers
        more bad values across both reader paths; this stays as a
        literal copy of the original assertion so the byte-band code
        path stays exercised."""
        vrt = _write_byte_missing_vrt(tmp_path)
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt, missing_sources="ignore")


# ===========================================================================
# Internal ``_vrt.read_vrt`` entry point (was
# test_vrt_missing_sources_default_raise_1843.py).
#
# The public matrix above exercises the package-level ``read_vrt`` /
# ``open_geotiff`` surface. These cases pin the internal
# ``xrspatial.geotiff._vrt.read_vrt`` entry point directly, including the
# ``XRSPATIAL_GEOTIFF_STRICT=1`` module-wide override that wins over a
# per-call ``missing_sources='warn'``.
# ===========================================================================


def _write_internal_missing_source_vrt(path):
    """All-missing 2x2 Byte VRT for the internal-entry-point cases."""
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing_1843.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


class TestInternalEntryPointMissingSources:
    """``xrspatial.geotiff._vrt.read_vrt`` default + opt-in behaviour."""

    def test_internal_default_raises_on_unreadable_source(self, tmp_path):
        """Without an explicit ``missing_sources`` kwarg, an unreadable
        backing source must raise rather than silently zero-fill.

        Before the default flipped to ``'raise'`` a missing ``Byte`` tile
        produced a hole of zero pixels indistinguishable from real data
        unless the caller checked ``attrs['vrt_holes']``.
        """
        vrt = tmp_path / "tmp_1843_default_raise.vrt"
        _write_internal_missing_source_vrt(vrt)
        with pytest.raises((OSError, ValueError)):
            _internal_read_vrt(str(vrt))

    def test_internal_explicit_warn_preserves_lenient_behaviour(self, tmp_path):
        """``missing_sources='warn'`` is still the escape hatch for callers
        that want partial mosaics with ``parsed.holes`` populated."""
        vrt = tmp_path / "tmp_1843_explicit_warn.vrt"
        _write_internal_missing_source_vrt(vrt)
        with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
            arr, parsed = _internal_read_vrt(str(vrt), missing_sources='warn')
        assert arr.shape == (2, 2)
        assert len(parsed.holes) == 1
        assert parsed.holes[0]['source'].endswith('missing_1843.tif')

    def test_internal_strict_env_still_raises_under_warn(
        self, monkeypatch, tmp_path,
    ):
        """``XRSPATIAL_GEOTIFF_STRICT=1`` continues to force-raise even
        when the caller explicitly asks for the lenient ``'warn'`` policy.

        The strict env var is a module-wide override; it must still win
        over per-call ``missing_sources='warn'`` so CI runs with strict
        mode catch partial mosaics regardless of caller settings.
        """
        vrt = tmp_path / "tmp_1843_strict_env.vrt"
        _write_internal_missing_source_vrt(vrt)
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")
        with pytest.raises((OSError, ValueError)):
            _internal_read_vrt(str(vrt), missing_sources='warn')


# ===========================================================================
# Public default ``missing_sources='raise'`` on read_vrt + open_geotiff
#
# Pins that the public wrapper's default matches the internal
# ``_vrt.read_vrt`` default rather than silently overriding it with the
# old lenient ``'warn'`` behaviour.
# ===========================================================================


def _write_public_missing_source_vrt(path):
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing_1860.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


class TestPublicDefaultMissingSources:
    """Public ``read_vrt`` / ``open_geotiff('.vrt')`` default to ``'raise'``."""

    def test_public_read_vrt_default_raises(self, tmp_path):
        """Public ``read_vrt`` with no ``missing_sources`` kwarg must raise.

        The default is aligned to the internal ``_vrt.read_vrt`` default
        of ``'raise'`` so the unreadable source halts the call instead of
        returning a partial mosaic with ``attrs['vrt_holes']``.
        """
        vrt = tmp_path / "tmp_1860_public_default_raise.vrt"
        _write_public_missing_source_vrt(vrt)
        with pytest.raises((OSError, ValueError)):
            read_vrt(str(vrt))

    def test_open_geotiff_vrt_default_raises(self, tmp_path):
        """``open_geotiff(vrt_path)`` with no ``missing_sources`` kwarg must
        raise on an unreadable backing source.

        ``open_geotiff`` forwards ``missing_sources`` to ``read_vrt`` only
        when the caller passed it explicitly; otherwise the public
        ``read_vrt`` default applies.
        """
        vrt = tmp_path / "tmp_1860_open_geotiff_default_raise.vrt"
        _write_public_missing_source_vrt(vrt)
        with pytest.raises((OSError, ValueError)):
            open_geotiff(str(vrt))

    def test_public_read_vrt_explicit_warn_preserves_lenient_behaviour(
        self, tmp_path,
    ):
        """``missing_sources='warn'`` is still the escape hatch for partial
        mosaics on the public ``read_vrt`` API."""
        vrt = tmp_path / "tmp_1860_public_explicit_warn.vrt"
        _write_public_missing_source_vrt(vrt)
        with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
            da = read_vrt(str(vrt), missing_sources='warn')
        assert 'vrt_holes' in da.attrs
        assert da.attrs['vrt_holes'][0]['source'].endswith('missing_1860.tif')

    def test_open_geotiff_vrt_explicit_warn_preserves_lenient_behaviour(
        self, tmp_path,
    ):
        """``open_geotiff(vrt_path, missing_sources='warn')`` still produces
        a partial mosaic with the hole record on the DataArray attrs."""
        vrt = tmp_path / "tmp_1860_open_geotiff_explicit_warn.vrt"
        _write_public_missing_source_vrt(vrt)
        with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
            da = open_geotiff(str(vrt), missing_sources='warn')
        assert 'vrt_holes' in da.attrs
        assert da.attrs['vrt_holes'][0]['source'].endswith('missing_1860.tif')


# ===========================================================================
# Chunked-path missing-source policy (was
# test_vrt_chunked_missing_sources_1799.py).
#
# The eager path scans every source at decode time. The chunked path
# uses a parse-time ``os.path.exists`` sweep to populate ``vrt_holes`` at
# build, and threads ``missing_sources`` through to the per-chunk decode.
# ===========================================================================


def _chunked_make_partial_vrt(tmp_path) -> tuple[str, str]:
    """2-source VRT: present source on the left, missing on the right.

    Returns ``(vrt_path, present_src_path)``.
    """
    src = os.path.join(tmp_path, "src_present.tif")
    arr = np.full((4, 4), 7.0, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)

    missing = os.path.join(tmp_path, "missing.tif")
    vrt_path = os.path.join(tmp_path, "partial.vrt")
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
    return vrt_path, src


class TestChunkedMissingSourcesWarn:
    """``read_vrt(chunks=N, missing_sources='warn')`` records holes at build."""

    def test_vrt_holes_populated_at_build(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs, (
            "Chunked path must populate vrt_holes at build time so "
            "callers can detect partial mosaics without forcing a compute."
        )
        holes = result.attrs["vrt_holes"]
        assert len(holes) == 1
        assert set(holes[0].keys()) == {"source", "band", "dst_rect", "error"}
        assert holes[0]["source"].endswith("missing.tif")
        assert holes[0]["band"] == 1
        assert holes[0]["dst_rect"] == (4, 0, 4, 4)

    def test_compute_emits_per_task_warning(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
            computed = result.compute()
        messages = [str(w.message) for w in caught
                    if isinstance(w.message, GeoTIFFFallbackWarning)]
        assert any("missing.tif" in msg for msg in messages), (
            f"Expected GeoTIFFFallbackWarning naming the missing source "
            f"after compute, got messages: {messages!r}"
        )
        np.testing.assert_array_equal(
            np.asarray(computed)[:, :4], np.full((4, 4), 7.0, dtype=np.float32),
        )
        assert np.all(np.isnan(np.asarray(computed)[:, 4:]))

    def test_chunks_tuple_form(self, tmp_path):
        """Tuple ``chunks=(h, w)`` threads through identically."""
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=(2, 4), missing_sources="warn")
        assert "vrt_holes" in result.attrs
        assert len(result.attrs["vrt_holes"]) == 1


class TestChunkedMissingSourcesRaiseSmoke:
    """``read_vrt(chunks=N, missing_sources='raise')`` fails at build.

    The detailed raise-at-build matrix (window / band scoping, multi-source
    preview, strict env) lives in the 2265 section below; this keeps the
    1799 smoke assertions alongside the warn cases they were paired with.
    """

    def test_build_raises_immediately(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.tif"):
            read_vrt(vrt_path, chunks=4, missing_sources="raise")

    def test_build_raise_message_mentions_policy_kwarg(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            read_vrt(vrt_path, chunks=4, missing_sources="raise")
        assert "missing_sources='warn'" in str(excinfo.value)

    def test_window_past_missing_succeeds_under_raise(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=4, window=(0, 0, 4, 4),
            missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 7.0, dtype=np.float32),
        )

    def test_band_selection_single_band_still_raises(self, tmp_path):
        """Selecting band 0 (the only band) still touches the missing
        source so the build raises. Cross-band gating is exercised by the
        multiband cases in the 2265 section below."""
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, band=0, missing_sources="raise")


class TestChunkedMissingSourcesDefault:
    """The default ``missing_sources`` on chunked reads is ``'raise'``."""

    def test_chunked_default_raises_at_build(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.tif"):
            read_vrt(vrt_path, chunks=4)


class TestChunkedMissingSourcesValidation:
    """Invalid ``missing_sources`` policies are rejected at entry."""

    def test_invalid_policy_raises_at_build(self, tmp_path):
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt_path, chunks=4, missing_sources="ignore")

    def test_invalid_policy_raises_without_chunks_too(self, tmp_path):
        """The eager path also rejects the bad value; callers see the same
        error whether or not they pass ``chunks=``."""
        vrt_path, _ = _chunked_make_partial_vrt(str(tmp_path))
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt_path, missing_sources="ignore")


# ===========================================================================
# Chunked raise-at-build matrix (was
# test_vrt_chunked_missing_raise_at_build_2265.py).
#
# The chunked path now honours ``missing_sources='raise'`` at build time:
# the static ``os.path.exists`` sweep raises up front when a hole
# intersects the requested window / selected band, instead of only the
# per-chunk delayed decode raising at compute.
# ===========================================================================


def _raise_write_present_source(tmp_path: str, name: str, fill: float) -> str:
    """Write a 4x4 float32 GeoTIFF source for a multi-source VRT."""
    src = os.path.join(tmp_path, name)
    arr = np.full((4, 4), fill, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)
    return src


def _raise_make_horizontal_partial_vrt(tmp_path: str) -> str:
    """2-source VRT: ``[ present | missing ]`` laid out 4x8."""
    src = _raise_write_present_source(tmp_path, "src_2265_h_present.tif", 7.0)
    missing = os.path.join(tmp_path, "missing_2265_h.tif")
    vrt_path = os.path.join(tmp_path, "partial_2265_h.vrt")
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
    return vrt_path


def _raise_make_multiband_partial_vrt(tmp_path: str) -> str:
    """2-band VRT where band 1 has a missing source and band 2 is intact."""
    src_b1 = _raise_write_present_source(tmp_path, "src_2265_mb_b1.tif", 11.0)
    src_b2 = _raise_write_present_source(tmp_path, "src_2265_mb_b2.tif", 22.0)
    missing_b1 = os.path.join(tmp_path, "missing_2265_mb_b1.tif")
    vrt_path = os.path.join(tmp_path, "partial_2265_multiband.vrt")
    with open(vrt_path, "w") as f:
        f.write(
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
            '<VRTRasterBand dataType="Float32" band="1">\n'
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{src_b1}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '<DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{missing_b1}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '<DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
            '</VRTRasterBand>\n'
            '<VRTRasterBand dataType="Float32" band="2">\n'
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{src_b2}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '<DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
            '</VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
    return vrt_path


def _raise_make_multi_missing_vrt(tmp_path: str, n_missing: int) -> str:
    """VRT with ``n_missing`` missing sources tiling the destination."""
    vrt_path = os.path.join(tmp_path, f"partial_2265_multi_{n_missing}.vrt")
    width = 4 * n_missing
    src_xml = []
    for i in range(n_missing):
        missing = os.path.join(tmp_path, f"missing_2265_multi_{i}.tif")
        src_xml.append(
            '<SimpleSource>\n'
            f'<SourceFilename relativeToVRT="0">{missing}</SourceFilename>\n'
            '<SourceBand>1</SourceBand>\n'
            '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            f'<DstRect xOff="{i * 4}" yOff="0" xSize="4" ySize="4"/>\n'
            '</SimpleSource>\n'
        )
    with open(vrt_path, "w") as f:
        f.write(
            f'<VRTDataset rasterXSize="{width}" rasterYSize="4">\n'
            '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
            '<VRTRasterBand dataType="Float32" band="1">\n'
            + ''.join(src_xml) +
            '</VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
    return vrt_path


class TestRaiseAtBuild:
    """``missing_sources='raise'`` raises during construction, not compute."""

    def test_build_raises_immediately(self, tmp_path):
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing_2265_h"):
            read_vrt(vrt_path, chunks=4, missing_sources="raise")

    def test_default_raises_at_build(self, tmp_path):
        """The public default is ``'raise'`` so dropping the kwarg hits
        the same fast-fail path."""
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4)

    def test_error_message_mentions_opt_in(self, tmp_path):
        """The exception text tells the caller how to opt into the lenient
        path."""
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            read_vrt(vrt_path, chunks=4, missing_sources="raise")
        msg = str(excinfo.value)
        assert "missing_sources='warn'" in msg
        assert "partial mosaic" in msg


class TestRaiseAtBuildWindowScoping:
    """The raise honours the requested window."""

    def test_window_past_missing_does_not_raise(self, tmp_path):
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=4, window=(0, 0, 4, 4),
            missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 7.0, dtype=np.float32),
        )

    def test_window_intersecting_missing_raises(self, tmp_path):
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(
                vrt_path, chunks=4, window=(0, 4, 4, 8),
                missing_sources="raise",
            )


class TestRaiseAtBuildBandScoping:
    """The raise honours ``band=`` restriction."""

    def test_band_select_skips_other_bands_missing_source(self, tmp_path):
        """``band=1`` reads band 2 only; band 1's missing source is
        irrelevant to the graph, so the build must not raise."""
        vrt_path = _raise_make_multiband_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=4, band=1, missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 22.0, dtype=np.float32),
        )

    def test_band_select_on_missing_band_raises(self, tmp_path):
        vrt_path = _raise_make_multiband_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, band=0, missing_sources="raise")

    def test_no_band_restriction_raises(self, tmp_path):
        vrt_path = _raise_make_multiband_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, missing_sources="raise")


class TestRaiseAtBuildWarnPreserved:
    """``missing_sources='warn'`` keeps the record-and-warn behaviour."""

    def test_warn_records_holes_at_build(self, tmp_path):
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs
        assert len(result.attrs["vrt_holes"]) == 1
        assert result.attrs["vrt_holes"][0]["source"].endswith(
            "missing_2265_h.tif"
        )

    def test_warn_compute_emits_per_task_warning(self, tmp_path):
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
            computed = result.compute()
        messages = [str(w.message) for w in caught
                    if isinstance(w.message, GeoTIFFFallbackWarning)]
        assert any("missing_2265_h" in msg for msg in messages)
        np.testing.assert_array_equal(
            np.asarray(computed)[:, :4],
            np.full((4, 4), 7.0, dtype=np.float32),
        )
        assert np.all(np.isnan(np.asarray(computed)[:, 4:]))


class TestRaiseAtBuildMultipleMissingSources:
    """The error message previews multiple holes and reports the total."""

    def test_two_missing_sources_listed_with_count(self, tmp_path):
        """All missing sources fit in the preview (n=2 <= preview cap)."""
        vrt_path = _raise_make_multi_missing_vrt(str(tmp_path), n_missing=2)
        with pytest.raises(FileNotFoundError) as excinfo:
            read_vrt(vrt_path, chunks=4, missing_sources="raise")
        msg = str(excinfo.value)
        assert "missing_2265_multi_0" in msg
        assert "missing_2265_multi_1" in msg
        assert "2 missing source(s) total" in msg
        assert "more" not in msg.lower() or "and 0 more" not in msg

    def test_many_missing_sources_truncated_with_more_suffix(self, tmp_path):
        """Above the preview cap, the message says 'and N more'."""
        n = 5
        vrt_path = _raise_make_multi_missing_vrt(str(tmp_path), n_missing=n)
        with pytest.raises(FileNotFoundError) as excinfo:
            read_vrt(vrt_path, chunks=4, missing_sources="raise")
        msg = str(excinfo.value)
        assert "missing_2265_multi_0" in msg
        assert f"missing_2265_multi_{n - 1}" not in msg
        assert f"{n} missing source(s) total" in msg
        assert "and 2 more" in msg


class TestRaiseAtBuildStrictMode:
    """``XRSPATIAL_GEOTIFF_STRICT=1`` forces the raise even with ``'warn'``."""

    def test_strict_overrides_warn_kwarg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, missing_sources="warn")

    def test_strict_off_warn_still_warns(self, tmp_path, monkeypatch):
        """Without strict mode, ``'warn'`` keeps warning."""
        monkeypatch.delenv("XRSPATIAL_GEOTIFF_STRICT", raising=False)
        vrt_path = _raise_make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs
