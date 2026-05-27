"""Default-rejection tests for the VRT ``stable_only=True`` gate.

Companion to ``test_release_gate_negative_mixed_tier_vrt_children`` in
``release_gates/test_stable_features.py``. These tests pin the
release-contract upgrade: a caller who asks for stable-tier sources
via ``stable_only=True`` on a VRT source must see a
typed :class:`VRTStableSourcesOnlyError` (a
:class:`GeoTIFFAmbiguousMetadataError` subclass) before any pixel
decode, and the message must name the file path and the
``allow_experimental_codecs`` opt-in unlock.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, VRTStableSourcesOnlyError,
                               open_geotiff, read_geotiff_dask, read_vrt)
from xrspatial.geotiff._errors import VRTUnsupportedError


_MINIMAL_VRT_XML = '<VRTDataset rasterXSize="2" rasterYSize="2"></VRTDataset>\n'


def _write_minimal_vrt(tmp_path: Path, name: str = "stable_only_2443") -> str:
    path = tmp_path / f"{name}.vrt"
    path.write_text(_MINIMAL_VRT_XML, encoding="utf-8")
    return str(path)


def test_open_geotiff_vrt_stable_only_rejected_by_default(tmp_path):
    """``open_geotiff(vrt, stable_only=True)`` raises the typed error."""
    path = _write_minimal_vrt(tmp_path, "open_geotiff_default")
    with pytest.raises(VRTStableSourcesOnlyError) as excinfo:
        open_geotiff(path, stable_only=True)
    msg = str(excinfo.value)
    assert path in msg, (
        f"expected the offending VRT path in the rejection message; "
        f"got: {msg!r}"
    )
    assert "stable_only" in msg
    assert "allow_experimental_codecs" in msg
    assert "release_gate_geotiff" in msg
    assert "#2342" in msg


def test_open_geotiff_vrt_stable_only_default_false_does_not_reject(tmp_path):
    """The default ``stable_only=False`` does not fire the new gate.

    The minimal-VRT body has no ``<VRTRasterBand>`` children so the
    read still raises the downstream :class:`VRTUnsupportedError`
    band-count check; pinning the exact class confirms the new gate is
    not stealing the raise site at the default flag value.
    """
    path = _write_minimal_vrt(tmp_path, "open_geotiff_default_false")
    with pytest.raises(VRTUnsupportedError):
        open_geotiff(path)


def test_open_geotiff_vrt_stable_only_with_experimental_unlock(tmp_path):
    """``allow_experimental_codecs=True`` is the documented unlock.

    When the caller passes both ``stable_only=True`` and
    ``allow_experimental_codecs=True`` the gate is a no-op (the per-source
    codec gate downstream handles the rest). The read still raises the
    downstream "no <VRTRasterBand>" :class:`VRTUnsupportedError` on
    this minimal-VRT fixture; pinning the exact downstream class keeps
    a future refactor from silently broadening the unlock past intent.
    """
    path = _write_minimal_vrt(tmp_path, "open_geotiff_unlock")
    with pytest.raises(VRTUnsupportedError):
        open_geotiff(
            path,
            stable_only=True,
            allow_experimental_codecs=True,
        )


def test_read_vrt_stable_only_rejected_by_default(tmp_path):
    """Direct ``read_vrt(stable_only=True)`` raises the typed error too."""
    path = _write_minimal_vrt(tmp_path, "read_vrt_direct")
    with pytest.raises(VRTStableSourcesOnlyError):
        read_vrt(path, stable_only=True)


def test_read_geotiff_dask_vrt_stable_only_rejected(tmp_path):
    """``read_geotiff_dask`` forwards the kwarg to ``read_vrt`` for VRT sources."""
    path = _write_minimal_vrt(tmp_path, "read_dask_vrt")
    with pytest.raises(VRTStableSourcesOnlyError):
        read_geotiff_dask(path, stable_only=True)


def test_vrt_stable_only_error_is_geotiff_ambiguous_metadata_error():
    """The typed error subclasses :class:`GeoTIFFAmbiguousMetadataError`.

    Callers that already ``except GeoTIFFAmbiguousMetadataError`` keep
    catching this case without an import-list change. The release-gate
    test in ``release_gates/test_stable_features.py`` relies on this
    inheritance to assert on the base class.
    """
    assert issubclass(VRTStableSourcesOnlyError, GeoTIFFAmbiguousMetadataError)


def test_read_vrt_stable_only_no_op_on_default(tmp_path):
    """``stable_only=False`` (the default) is a no-op on the direct VRT path.

    Same fixture as the rejection test, but the absence of the flag
    means the read proceeds to the existing band-count validator and
    raises :class:`VRTUnsupportedError`.
    """
    path = _write_minimal_vrt(tmp_path, "read_vrt_default")
    with pytest.raises(VRTUnsupportedError):
        read_vrt(path)
