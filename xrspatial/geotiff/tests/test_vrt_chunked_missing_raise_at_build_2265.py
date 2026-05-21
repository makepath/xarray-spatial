"""Issue #2265: chunked VRT ``missing_sources='raise'`` must raise at build.

The public docstring on ``read_vrt`` says ``missing_sources='raise'`` (the
public default since #1860) "fails immediately on an unreadable backing
source so a partial mosaic never surfaces silently". Before #2265 the
chunked path only honoured that contract at compute time: it ran a
static ``os.path.exists`` sweep at build, recorded misses into
``attrs['vrt_holes']``, and only the per-chunk delayed decode raised --
which meant a windowed downstream slice past the bad tile could ship a
partial mosaic silently. This module pins the "raise at build" behaviour
and the related scoping invariants:

* a missing source intersecting the requested window raises at build,
* a missing source outside the requested window does not raise,
* a missing source on a band the caller did not select does not raise,
* ``XRSPATIAL_GEOTIFF_STRICT=1`` forces the raise regardless of kwarg,
* ``missing_sources='warn'`` keeps the existing record-and-warn path.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, read_vrt, to_geotiff


def _write_present_source(tmp_path: str, name: str, fill: float) -> str:
    """Write a 4x4 float32 GeoTIFF source for use in a multi-source VRT."""
    src = os.path.join(tmp_path, name)
    arr = np.full((4, 4), fill, dtype=np.float32)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    to_geotiff(da, src)
    return src


def _make_horizontal_partial_vrt(tmp_path: str) -> str:
    """2-source VRT: present source on the left, missing source on the right.

    Layout (rows x cols = 4 x 8):
    ``[ present | missing ]``. Used for the basic
    ``raise at build`` and window-scoping assertions.
    """
    src = _write_present_source(tmp_path, "src_2265_h_present.tif", 7.0)
    missing = os.path.join(tmp_path, "missing_2265_h.tif")
    vrt_path = os.path.join(tmp_path, "partial_2265_h.vrt")
    with open(vrt_path, "w") as f:
        f.write(
            f'<VRTDataset rasterXSize="8" rasterYSize="4">\n'
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


def _make_multiband_partial_vrt(tmp_path: str) -> str:
    """2-band VRT where band 1 has a missing source and band 2 is intact.

    Both bands cover the full 4x4 extent with one source each. A
    ``band=1`` (0-based, the second band) read should not raise because
    the per-chunk decode never touches band 1's missing source. Reading
    without a band restriction or with ``band=0`` should raise.
    """
    src_b1 = _write_present_source(tmp_path, "src_2265_mb_b1.tif", 11.0)
    src_b2 = _write_present_source(tmp_path, "src_2265_mb_b2.tif", 22.0)
    missing_b1 = os.path.join(tmp_path, "missing_2265_mb_b1.tif")
    vrt_path = os.path.join(tmp_path, "partial_2265_multiband.vrt")
    with open(vrt_path, "w") as f:
        f.write(
            f'<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
            # Band 1: one present source + one missing source covering
            # the same extent. The missing source intersects every
            # chunk window so the build must raise when band 1 is in
            # scope.
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
            # Band 2: a single present source. ``band=1`` (0-based) on
            # the chunked read should pick this band only and skip
            # band 1's missing source.
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


class TestRaiseAtBuild:
    """``missing_sources='raise'`` raises during construction, not compute."""

    def test_build_raises_immediately(self, tmp_path):
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing_2265_h"):
            read_vrt(vrt_path, chunks=4, missing_sources="raise")

    def test_default_raises_at_build(self, tmp_path):
        """The public default is ``'raise'`` so dropping the kwarg
        must hit the same fast-fail path."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4)

    def test_error_message_mentions_opt_in(self, tmp_path):
        """The exception text should tell the caller how to opt into
        the lenient path. A regression that drops this guidance would
        leave callers debugging a bare ``FileNotFoundError`` without
        knowing the kwarg toggle exists."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            read_vrt(vrt_path, chunks=4, missing_sources="raise")
        msg = str(excinfo.value)
        assert "missing_sources='warn'" in msg
        assert "partial mosaic" in msg


class TestWindowScoping:
    """The raise honours the requested window."""

    def test_window_past_missing_does_not_raise(self, tmp_path):
        """A window that touches only the present source still builds
        and computes. Without this scoping the static raise would be
        overzealous compared to the eager path (which decodes only
        sources that intersect the window)."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=4, window=(0, 0, 4, 4),
            missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 7.0, dtype=np.float32),
        )

    def test_window_intersecting_missing_raises(self, tmp_path):
        """A window that overlaps the missing tile still raises at build."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(
                vrt_path, chunks=4, window=(0, 4, 4, 8),
                missing_sources="raise",
            )


class TestBandScoping:
    """The raise honours ``band=`` restriction."""

    def test_band_select_skips_other_bands_missing_source(self, tmp_path):
        """``band=1`` reads band 2 only; band 1's missing source is
        irrelevant to the graph, so the build must not raise."""
        vrt_path = _make_multiband_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=4, band=1, missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 22.0, dtype=np.float32),
        )

    def test_band_select_on_missing_band_raises(self, tmp_path):
        """``band=0`` selects the band with the missing source so the
        build must raise (mirror of the unselected-band test above)."""
        vrt_path = _make_multiband_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(
                vrt_path, chunks=4, band=0, missing_sources="raise",
            )

    def test_no_band_restriction_raises(self, tmp_path):
        """Without a ``band=`` restriction, both bands' sources are in
        scope and the missing source on band 1 raises at build."""
        vrt_path = _make_multiband_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, missing_sources="raise")


class TestWarnPreserved:
    """``missing_sources='warn'`` keeps the record-and-warn behaviour."""

    def test_warn_records_holes_at_build(self, tmp_path):
        """The lenient path must not regress to a build-time raise."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs
        assert len(result.attrs["vrt_holes"]) == 1
        assert result.attrs["vrt_holes"][0]["source"].endswith(
            "missing_2265_h.tif"
        )

    def test_warn_compute_emits_per_task_warning(self, tmp_path):
        """The compute step still warns per task on the lenient path."""
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
            computed = result.compute()
        messages = [str(w.message) for w in caught
                    if isinstance(w.message, GeoTIFFFallbackWarning)]
        assert any("missing_2265_h" in msg for msg in messages)
        # Present side decodes to 7.0; missing side decodes to NaN.
        np.testing.assert_array_equal(
            np.asarray(computed)[:, :4],
            np.full((4, 4), 7.0, dtype=np.float32),
        )
        assert np.all(np.isnan(np.asarray(computed)[:, 4:]))


class TestStrictMode:
    """``XRSPATIAL_GEOTIFF_STRICT=1`` forces the raise even with ``'warn'``."""

    def test_strict_overrides_warn_kwarg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            read_vrt(vrt_path, chunks=4, missing_sources="warn")

    def test_strict_off_warn_still_warns(self, tmp_path, monkeypatch):
        """Sanity: without strict mode, ``'warn'`` keeps warning."""
        monkeypatch.delenv("XRSPATIAL_GEOTIFF_STRICT", raising=False)
        vrt_path = _make_horizontal_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs
