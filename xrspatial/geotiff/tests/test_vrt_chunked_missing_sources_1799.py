"""Chunked-VRT coverage for ``missing_sources`` (issue #1799).

``test_vrt_missing_sources_policy_1799`` covers the eager (non-chunked)
``read_vrt`` path. The chunked path (``read_vrt(chunks=N)``, dispatching
through ``_read_vrt_chunked``) plumbs ``missing_sources`` separately:

* Parse-time approximation: a static ``os.path.exists`` sweep over every
  source populates ``attrs['vrt_holes']`` on the returned DataArray
  before any decode work starts (docstring in ``_backends/vrt.py:344``).
* Decode-time: each per-chunk task receives ``missing_sources`` and the
  internal reader applies the same warn/raise policy as the eager path.

A regression dropping either the parse-time sweep or the per-chunk
forward would silently change the contract:

* ``vrt_holes`` would disappear from the lazy build, breaking callers
  that branch on ``"vrt_holes" in da.attrs`` to detect partial mosaics
  before scheduling a compute (the contract documented in #1734).
* ``missing_sources='raise'`` could silently degrade to ``'warn'`` (or
  vice versa) on the chunked path while the eager path stays correct.

This module pins both invariants. Tests use a 2-source mosaic where one
source is missing on disk; the present source covers one chunk window
and the missing source covers another, so the warn/raise policy is
exercised against a non-trivial graph.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, read_vrt, to_geotiff


def _make_partial_vrt(tmp_path) -> tuple[str, str]:
    """Build a 2-source VRT with one present + one missing source.

    Returns ``(vrt_path, present_src_path)``. The VRT references the
    present source for the left half and a non-existent file for the
    right half, so chunked reads against the right half hit the
    missing-source decode path.
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
    return vrt_path, src


class TestChunkedMissingSourcesWarn:
    """``read_vrt(chunks=N, missing_sources='warn')`` records holes at build.

    The eager path scans every source at decode time. The chunked path
    cannot afford that sweep up front (it would defeat the lazy graph),
    so it uses ``os.path.exists`` to populate ``vrt_holes`` at build
    time. The compute step still emits per-task warnings for any
    missing source that survives.
    """

    def test_vrt_holes_populated_at_build(self, tmp_path):
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
        assert "vrt_holes" in result.attrs, (
            "Chunked path must populate vrt_holes at build time so "
            "callers can detect partial mosaics without forcing a "
            "compute (issue #1734)."
        )
        holes = result.attrs["vrt_holes"]
        assert len(holes) == 1
        assert holes[0]["source"].endswith("missing.tif")

    def test_compute_emits_per_task_warning(self, tmp_path):
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = read_vrt(vrt_path, chunks=4, missing_sources="warn")
            computed = result.compute()
        messages = [str(w.message) for w in caught
                    if isinstance(w.message, GeoTIFFFallbackWarning)]
        assert any("missing.tif" in msg for msg in messages), (
            f"Expected GeoTIFFFallbackWarning naming the missing "
            f"source after compute, got messages: {messages!r}"
        )
        # Present-source chunk decodes its 7.0 fill; missing-source
        # chunk decodes to the source dtype's default fill (typically
        # zero for float32). Pin the present-side decode value so a
        # regression in the lenient path that wiped both halves would
        # surface.
        np.testing.assert_array_equal(
            np.asarray(computed)[:, :4], np.full((4, 4), 7.0, dtype=np.float32),
        )

    def test_chunks_tuple_form(self, tmp_path):
        """Tuple ``chunks=(h, w)`` threads through identically."""
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        result = read_vrt(
            vrt_path, chunks=(2, 4), missing_sources="warn",
        )
        assert "vrt_holes" in result.attrs
        # 2 chunks vertically * 2 chunks horizontally = 4 tasks.
        # The missing source is in column 1 (cols 4-7); only the right
        # half should produce warning records, but vrt_holes is a
        # parse-time sweep so it records the source once regardless.
        assert len(result.attrs["vrt_holes"]) == 1


class TestChunkedMissingSourcesRaise:
    """``read_vrt(chunks=N, missing_sources='raise')`` fails on compute.

    The eager path raises at read time. The chunked path defers to
    compute because each chunk's decode is delayed; an upfront raise
    would force the parse-time sweep to decode every source, defeating
    the lazy graph. The contract: chunks intersecting a missing source
    raise on compute; chunks intersecting only present sources still
    succeed.
    """

    def test_compute_intersecting_missing_raises(self, tmp_path):
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4, missing_sources="raise")
        # Build does not raise (the graph is lazy).
        # Computing a chunk that intersects the missing source raises.
        with pytest.raises((OSError, FileNotFoundError, ValueError)):
            result.compute()

    def test_compute_present_only_chunk_succeeds(self, tmp_path):
        """A windowed compute against only the present source succeeds.

        ``read_vrt(window=...)`` restricts the chunked graph to the
        windowed extent; if the window misses the missing source, no
        chunk needs to decode it and compute succeeds even under
        ``missing_sources='raise'``. The contract: the raise policy is
        scoped to chunks that actually touch missing sources.
        """
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        # Window covers only the present source (cols 0-4).
        result = read_vrt(
            vrt_path, chunks=4, window=(0, 0, 4, 4),
            missing_sources="raise",
        )
        computed = result.compute()
        np.testing.assert_array_equal(
            np.asarray(computed), np.full((4, 4), 7.0, dtype=np.float32),
        )


class TestChunkedMissingSourcesDefault:
    """The default ``missing_sources`` on chunked reads is ``'raise'``.

    The public ``read_vrt`` default flipped to ``'raise'`` in #1843 /
    #1860. The chunked path goes through the same entry point so the
    default must agree. A regression flipping the chunked default to
    ``'warn'`` would silently produce partial mosaics for callers who
    don't pass the kwarg.
    """

    def test_chunked_default_raises_on_compute(self, tmp_path):
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        result = read_vrt(vrt_path, chunks=4)
        with pytest.raises((OSError, FileNotFoundError, ValueError)):
            result.compute()


class TestChunkedMissingSourcesValidation:
    """Invalid ``missing_sources`` policies are rejected at entry."""

    def test_invalid_policy_raises_at_build(self, tmp_path):
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt_path, chunks=4, missing_sources="ignore")

    def test_invalid_policy_raises_without_chunks_too(self, tmp_path):
        """Sanity: the eager path also rejects the bad value. Pinning
        cross-mode parity means callers see the same error whether or
        not they pass ``chunks=``."""
        vrt_path, _ = _make_partial_vrt(str(tmp_path))
        with pytest.raises(ValueError, match="missing_sources"):
            read_vrt(vrt_path, missing_sources="ignore")
