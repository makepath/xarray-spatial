"""Regression tests for issue #1697.

``read_vrt(path, window=...)`` silently clamped invalid window
coordinates instead of raising ``ValueError``. The local TIFF path
(#1634) and the HTTP COG path (#1669) both reject the same bad
windows up front. VRT was missed when #1634 landed.

The hidden failure mode is the same one #1634 fixed: ``open_geotiff``
builds the y/x coord arrays from the caller-supplied window indices,
so a clamped read returns a smaller array than the coord arrays and
xarray raises a ``CoordinateValidationError`` deep inside its
constructor instead of a clear xrspatial-level error.

These tests pin the VRT path to the same contract as the local and
HTTP paths: out-of-bounds, zero-size, or inverted windows raise
``ValueError`` with a message of the same shape (one "extent" word
swap from ``source extent`` to ``VRT extent``).
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._vrt import read_vrt


def _unique_dir(tmp_path, label: str) -> str:
    """Return a sub-path under ``tmp_path`` with a uuid suffix so
    parallel test workers cannot collide on the same name."""
    d = tmp_path / f"vrt_1697_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_tif(path: str, size: int = 4) -> None:
    """Write a ``size``x``size`` float32 GeoTIFF the VRT can wrap."""
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    y = np.linspace(float(size) - 0.5, 0.5, size)
    x = np.linspace(0.5, float(size) - 0.5, size)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, compression='none')


def _write_vrt(vrt_path: str, source_filename: str, size: int = 4) -> None:
    """Write a single-band VRT of dimension ``size``x``size`` pointing
    at ``source_filename`` (relative to the VRT directory)."""
    xml = (
        f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_filename}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


@pytest.fixture
def vrt_4x4(tmp_path):
    """Return a path to a 4x4 single-band VRT wrapping a 4x4 TIFF."""
    d = _unique_dir(tmp_path, "fixture")
    tif = os.path.join(d, 'data.tif')
    _write_tif(tif, size=4)
    vrt = os.path.join(d, 'mosaic.vrt')
    _write_vrt(vrt, 'data.tif', size=4)
    return vrt


# ---------------------------------------------------------------------------
# Negative starts
# ---------------------------------------------------------------------------


def test_negative_r0_raises_value_error(vrt_4x4):
    """``r0 < 0`` raises ValueError instead of being clamped to 0."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        read_vrt(vrt_4x4, window=(-1, 0, 2, 2))


def test_negative_c0_raises_value_error(vrt_4x4):
    """``c0 < 0`` raises ValueError instead of being clamped to 0."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        read_vrt(vrt_4x4, window=(0, -1, 2, 2))


# ---------------------------------------------------------------------------
# Past-edge stops
# ---------------------------------------------------------------------------


def test_r1_past_bottom_edge_raises_value_error(vrt_4x4):
    """``r1 > vrt.height`` raises instead of being clamped to vrt.height."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        read_vrt(vrt_4x4, window=(0, 0, 5, 4))


def test_c1_past_right_edge_raises_value_error(vrt_4x4):
    """``c1 > vrt.width`` raises instead of being clamped to vrt.width."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        read_vrt(vrt_4x4, window=(0, 0, 4, 5))


def test_window_past_both_edges_raises_value_error(vrt_4x4):
    """Windows past both right and bottom edges raise the same error."""
    with pytest.raises(ValueError, match='outside the VRT extent'):
        read_vrt(vrt_4x4, window=(0, 0, 10, 10))


# ---------------------------------------------------------------------------
# Zero-size and inverted windows
# ---------------------------------------------------------------------------


def test_zero_size_row_window_raises_value_error(vrt_4x4):
    """``r0 == r1`` produces a zero-height window and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        read_vrt(vrt_4x4, window=(2, 0, 2, 4))


def test_zero_size_col_window_raises_value_error(vrt_4x4):
    """``c0 == c1`` produces a zero-width window and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        read_vrt(vrt_4x4, window=(0, 2, 4, 2))


def test_fully_zero_size_window_raises_value_error(vrt_4x4):
    """``r0 == r1 and c0 == c1`` raises (current code returned a (0, 0) array)."""
    with pytest.raises(ValueError, match='non-positive size'):
        read_vrt(vrt_4x4, window=(2, 2, 2, 2))


def test_inverted_row_window_raises_value_error(vrt_4x4):
    """``r0 > r1`` is degenerate and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        read_vrt(vrt_4x4, window=(3, 0, 1, 4))


def test_inverted_col_window_raises_value_error(vrt_4x4):
    """``c0 > c1`` is degenerate and must raise."""
    with pytest.raises(ValueError, match='non-positive size'):
        read_vrt(vrt_4x4, window=(0, 3, 4, 1))


# ---------------------------------------------------------------------------
# Valid in-bounds windows still work (don't over-reject)
# ---------------------------------------------------------------------------


def test_full_extent_window_still_works(vrt_4x4):
    """A window covering the full VRT extent still reads the full array."""
    arr, _ = read_vrt(vrt_4x4, window=(0, 0, 4, 4))
    assert arr.shape == (4, 4)


def test_interior_window_still_works(vrt_4x4):
    """An interior window returns the requested subset shape."""
    arr, _ = read_vrt(vrt_4x4, window=(1, 1, 3, 3))
    assert arr.shape == (2, 2)


def test_edge_aligned_window_still_works(vrt_4x4):
    """A window that touches but does not exceed the edges is accepted."""
    arr, _ = read_vrt(vrt_4x4, window=(2, 2, 4, 4))
    assert arr.shape == (2, 2)


def test_none_window_still_returns_full_array(vrt_4x4):
    """``window=None`` still returns the full VRT extent."""
    arr, _ = read_vrt(vrt_4x4)
    assert arr.shape == (4, 4)


# ---------------------------------------------------------------------------
# Cross-path parity: VRT and local-TIFF reject the same bad window
# ---------------------------------------------------------------------------


def test_vrt_and_local_paths_share_window_validation(tmp_path):
    """Same bad window rejected on both VRT and local-TIFF paths with the
    same error class and message shape (one word swap is fine)."""
    d = _unique_dir(tmp_path, "parity")
    tif = os.path.join(d, 'data.tif')
    _write_tif(tif, size=4)
    vrt = os.path.join(d, 'mosaic.vrt')
    _write_vrt(vrt, 'data.tif', size=4)

    bad_window = (-1, 0, 2, 2)

    with pytest.raises(ValueError) as vrt_exc:
        read_vrt(vrt, window=bad_window)
    with pytest.raises(ValueError) as local_exc:
        read_to_array(tif, window=bad_window)

    vrt_msg = str(vrt_exc.value)
    local_msg = str(local_exc.value)

    # Both messages must echo the offending window and the source dims.
    assert 'window=' in vrt_msg
    assert 'window=' in local_msg
    assert '4x4' in vrt_msg
    assert '4x4' in local_msg
    # And both must signal "out of bounds" via the same wording template,
    # with a single word swap on the extent label.
    assert 'extent' in vrt_msg
    assert 'extent' in local_msg
    assert 'non-positive size' in vrt_msg
    assert 'non-positive size' in local_msg
    # The VRT path says "VRT extent"; the local path says "source extent".
    assert 'VRT extent' in vrt_msg
    assert 'source extent' in local_msg
