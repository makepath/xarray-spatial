"""Regression tests for issue #1634.

``open_geotiff(path, window=...)`` on the eager (numpy) path produced a
confusing ``CoordinateValidationError`` when the requested window
extended past the source extent. ``read_to_array`` correctly clamped
the window to file bounds and returned a smaller array, but the eager
code path used the unclamped window indices to build the y/x
coordinate arrays. The resulting coord arrays had a different length
than the returned data, so xarray refused to construct the DataArray.

The dask path (``read_geotiff_dask``) already rejected out-of-bounds
windows with a clear ``ValueError`` since #1561. This test locks the
eager path into the same contract: out-of-bounds windows raise a clear
``ValueError`` with the same message format, regardless of which
backend the user requests via ``open_geotiff``.

The fix lives in ``xrspatial/geotiff/__init__.py``: the eager branch
now validates ``window`` up front against the source extent, mirroring
the dask path's validator.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _make_raster(tmp_path: str) -> str:
    """Write a deterministic 10x10 float32 GeoTIFF and return its path."""
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(10), 'x': np.arange(10)},
        attrs={'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)},
    )
    path = os.path.join(tmp_path, 'raster.tif')
    to_geotiff(da, path)
    return path


# -- Out-of-bounds windows on the eager path --------------------------------


def test_eager_negative_start_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent'):
        open_geotiff(path, window=(-5, -5, 5, 5))


def test_eager_past_right_edge_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent'):
        open_geotiff(path, window=(0, 5, 5, 15))


def test_eager_past_bottom_edge_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent'):
        open_geotiff(path, window=(5, 0, 15, 5))


def test_eager_past_both_edges_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent'):
        open_geotiff(path, window=(5, 5, 15, 15))


def test_eager_zero_size_window_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent|non-positive size'):
        open_geotiff(path, window=(3, 3, 3, 3))


def test_eager_inverted_window_raises_value_error(tmp_path):
    path = _make_raster(str(tmp_path))
    with pytest.raises(ValueError, match='outside the source extent|non-positive size'):
        open_geotiff(path, window=(5, 5, 3, 3))


# -- In-bounds windows still work on the eager path -------------------------


def test_eager_full_extent_window_returns_full_array(tmp_path):
    path = _make_raster(str(tmp_path))
    result = open_geotiff(path, window=(0, 0, 10, 10))
    assert result.shape == (10, 10)
    # Coord arrays should match the data dimensions
    assert result.coords['y'].size == 10
    assert result.coords['x'].size == 10


def test_eager_interior_window_returns_correct_subset(tmp_path):
    path = _make_raster(str(tmp_path))
    result = open_geotiff(path, window=(2, 3, 7, 8))
    assert result.shape == (5, 5)
    assert result.coords['y'].size == 5
    assert result.coords['x'].size == 5
    # Pixel values should match the subset of the original
    expected = np.arange(100, dtype=np.float32).reshape(10, 10)[2:7, 3:8]
    np.testing.assert_array_equal(result.values, expected)


def test_eager_edge_aligned_window_returns_correct_subset(tmp_path):
    path = _make_raster(str(tmp_path))
    # Window touches but does not exceed the edge
    result = open_geotiff(path, window=(0, 0, 10, 10))
    assert result.shape == (10, 10)


# -- Backend parity ---------------------------------------------------------


def test_eager_and_dask_paths_share_window_validation(tmp_path):
    """Both backends must raise ValueError on the same bad window."""
    path = _make_raster(str(tmp_path))
    bad_window = (5, 5, 15, 15)

    with pytest.raises(ValueError) as eager_exc:
        open_geotiff(path, window=bad_window)
    with pytest.raises(ValueError) as dask_exc:
        open_geotiff(path, window=bad_window, chunks=4)

    # Both errors should mention the source extent and the bad window
    assert 'outside the source extent' in str(eager_exc.value)
    assert 'outside the source extent' in str(dask_exc.value)
    # Both should reference the source dimensions (10x10) somewhere
    assert '10' in str(eager_exc.value)
    assert '10' in str(dask_exc.value)


def test_eager_and_dask_paths_share_window_message_format(tmp_path):
    """Eager and dask paths emit messages matching the same format."""
    path = _make_raster(str(tmp_path))
    bad_window = (-5, -5, 5, 5)

    with pytest.raises(ValueError) as eager_exc:
        open_geotiff(path, window=bad_window)
    with pytest.raises(ValueError) as dask_exc:
        open_geotiff(path, window=bad_window, chunks=4)

    # Both should be ValueError with the same template
    eager_msg = str(eager_exc.value)
    dask_msg = str(dask_exc.value)
    # The dask path's template:
    # "window={window} is outside the source extent ({h}x{w}) or has non-positive size."
    assert 'window=' in eager_msg
    assert 'window=' in dask_msg


# -- Issue #1634 specific repro --------------------------------------------


def test_issue_1634_reproducer_raises_clean_error(tmp_path):
    """The reproducer in the issue should raise ValueError, not
    CoordinateValidationError from xarray's internals.
    """
    path = _make_raster(str(tmp_path))
    # Reproducer from the issue
    try:
        result = open_geotiff(path, window=(5, 5, 15, 15))
        pytest.fail(f'expected ValueError, got result shape {result.shape}')
    except ValueError as e:
        # Must be a clear xrspatial-level error, not a deep xarray coord
        # validation error masking the real cause
        msg = str(e)
        assert 'window' in msg.lower()
        assert 'source extent' in msg.lower() or 'out' in msg.lower()
