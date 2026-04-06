"""Cross-validate xrspatial.sieve() against rasterio.features.sieve() (GDAL).

rasterio.features.sieve wraps GDAL's GDALSieveFilter, which is the accepted
reference for this operation.  These tests run both implementations on
identical inputs and compare outputs.

Requires rasterio; tests are skipped if it is not installed.

Known behavioral differences
-----------------------------
* **Input dtype**: rasterio requires integer arrays (int16/int32/uint8/uint16).
  xrspatial accepts float or int, converts to float64 internally.
* **Nodata**: rasterio uses a boolean ``mask`` (False = excluded).
  xrspatial uses NaN.
* **Tie-breaking**: when a small region has multiple neighbors of the same
  size, GDAL and xrspatial may pick different winners.  Tests that would
  depend on tie-breaking are designed to avoid ambiguity.
* **skip_values**: xrspatial-only feature, not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip("rasterio")
from rasterio.features import sieve as gdal_sieve

from xrspatial.sieve import sieve as xs_sieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_both(arr_int, threshold, connectivity=4):
    """Run both sieve implementations and return (xrspatial, gdal) arrays.

    Parameters
    ----------
    arr_int : ndarray
        2D integer array (shared input for both implementations).
    threshold : int
        Minimum region size to retain.
    connectivity : int
        4 or 8.

    Returns
    -------
    xs_out : ndarray of int32
        xrspatial result cast back to int32 for comparison.
    gdal_out : ndarray of int32
        rasterio/GDAL result.
    """
    # xrspatial path: wrap in DataArray (float64 internally)
    raster = xr.DataArray(arr_int.astype(np.float64), dims=["y", "x"])
    xs_result = xs_sieve(raster, threshold=threshold, neighborhood=connectivity)
    xs_out = xs_result.values.astype(np.int32)

    # GDAL path: needs int32
    gdal_src = arr_int.astype(np.int32).copy()
    gdal_out = gdal_sieve(gdal_src, size=threshold, connectivity=connectivity)

    return xs_out, gdal_out


def _run_both_with_nodata(arr_int, nodata_val, threshold, connectivity=4):
    """Run both implementations with nodata handling.

    For xrspatial: replace nodata_val with NaN.
    For GDAL: pass a mask where nodata pixels are False.

    Returns
    -------
    xs_out, gdal_out : ndarrays with nodata_val restored in nodata positions.
    """
    # xrspatial: use NaN for nodata
    arr_float = arr_int.astype(np.float64)
    arr_float[arr_int == nodata_val] = np.nan
    raster = xr.DataArray(arr_float, dims=["y", "x"])
    xs_result = xs_sieve(raster, threshold=threshold, neighborhood=connectivity)
    xs_vals = xs_result.values.copy()
    # Restore nodata positions to nodata_val for comparison
    xs_vals[np.isnan(xs_vals)] = nodata_val
    xs_out = xs_vals.astype(np.int32)

    # GDAL: use mask
    gdal_src = arr_int.astype(np.int32).copy()
    mask = (arr_int != nodata_val).astype(np.uint8)
    gdal_out = gdal_sieve(gdal_src, size=threshold, connectivity=connectivity, mask=mask)

    return xs_out, gdal_out


# ---------------------------------------------------------------------------
# Basic parity: noise removal
# ---------------------------------------------------------------------------


class TestBasicParity:
    """Simple cases where both implementations should agree exactly."""

    def test_single_pixel_noise(self):
        """One isolated pixel in a uniform background."""
        arr = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 3, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Both should have replaced the 3 with 1
        assert xs_out[2, 2] == 1

    def test_two_classes_no_noise(self):
        """Two large regions, nothing to sieve."""
        arr = np.zeros((6, 6), dtype=np.int32)
        arr[:, :3] = 1
        arr[:, 3:] = 2
        xs_out, gdal_out = _run_both(arr, threshold=5)
        np.testing.assert_array_equal(xs_out, gdal_out)
        np.testing.assert_array_equal(xs_out, arr)

    def test_multiple_noise_pixels(self):
        """Several isolated noise pixels in different locations."""
        arr = np.ones((8, 8), dtype=np.int32)
        arr[1, 1] = 5
        arr[1, 6] = 7
        arr[6, 1] = 3
        arr[6, 6] = 9
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        expected = np.ones((8, 8), dtype=np.int32)
        np.testing.assert_array_equal(xs_out, expected)

    def test_small_cluster_removal(self):
        """A 3-pixel cluster in a large background."""
        arr = np.ones((10, 10), dtype=np.int32)
        # 3-pixel L-shape of value 2
        arr[2, 2] = 2
        arr[2, 3] = 2
        arr[3, 2] = 2
        xs_out, gdal_out = _run_both(arr, threshold=4)
        np.testing.assert_array_equal(xs_out, gdal_out)
        assert xs_out[2, 2] == 1

    def test_threshold_exact_boundary(self):
        """Region exactly at threshold size should be retained."""
        arr = np.ones((6, 6), dtype=np.int32)
        # 4-pixel block of value 2
        arr[1, 1] = 2
        arr[1, 2] = 2
        arr[2, 1] = 2
        arr[2, 2] = 2
        xs_out, gdal_out = _run_both(arr, threshold=4)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Region of size 4 should survive threshold=4
        assert xs_out[1, 1] == 2


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


class TestThresholdSweep:
    """Run multiple thresholds on the same input."""

    @pytest.fixture()
    def classified_grid(self):
        """10x10 grid with a few distinct regions of known sizes."""
        arr = np.array(
            [
                [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 2, 3, 3, 2, 2, 2],
                [1, 1, 1, 1, 2, 3, 3, 2, 2, 2],
                [4, 4, 1, 1, 2, 2, 2, 2, 2, 2],
                [4, 4, 1, 1, 5, 5, 5, 2, 2, 2],
                [4, 4, 4, 1, 5, 5, 5, 2, 2, 2],
                [4, 4, 4, 4, 5, 5, 5, 2, 2, 2],
                [4, 4, 4, 4, 5, 5, 5, 6, 6, 2],
                [4, 4, 4, 4, 5, 5, 5, 6, 6, 2],
            ],
            dtype=np.int32,
        )
        # Region sizes: 1=16, 2=34, 3=4, 4=16, 5=12, 6=4
        return arr

    @pytest.mark.parametrize("threshold", [1, 3, 5, 10, 13])
    def test_threshold(self, classified_grid, threshold):
        xs_out, gdal_out = _run_both(classified_grid, threshold=threshold)
        np.testing.assert_array_equal(
            xs_out,
            gdal_out,
            err_msg=f"Mismatch at threshold={threshold}",
        )

    def test_threshold_50(self, classified_grid):
        """threshold=50 exceeds all region sizes -- nothing merges."""
        xs_out, gdal_out = _run_both(classified_grid, threshold=50)
        np.testing.assert_array_equal(
            xs_out,
            gdal_out,
            err_msg="Mismatch at threshold=50",
        )


# ---------------------------------------------------------------------------
# Connectivity modes
# ---------------------------------------------------------------------------


class TestConnectivity:
    """4-connectivity vs 8-connectivity parity."""

    def test_diagonal_pixel_4conn(self):
        """A pixel connected only diagonally is isolated under 4-conn."""
        arr = np.ones((5, 5), dtype=np.int32)
        arr[2, 2] = 2
        arr[1, 1] = 2  # diagonal to (2,2) but not edge-adjacent
        # With 4-conn: two separate 1-pixel regions of value 2
        xs_out, gdal_out = _run_both(arr, threshold=2, connectivity=4)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Both should be replaced
        assert xs_out[2, 2] == 1
        assert xs_out[1, 1] == 1

    def test_diagonal_pixel_8conn(self):
        """A pixel connected diagonally is part of the same region under 8-conn."""
        arr = np.ones((5, 5), dtype=np.int32)
        arr[2, 2] = 2
        arr[1, 1] = 2
        # With 8-conn: these two form a 2-pixel region
        xs_out, gdal_out = _run_both(arr, threshold=2, connectivity=8)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Region of size 2 should survive threshold=2
        assert xs_out[2, 2] == 2
        assert xs_out[1, 1] == 2

    def test_l_shape_4conn(self):
        """L-shaped region under 4-connectivity."""
        arr = np.ones((6, 6), dtype=np.int32)
        arr[1, 1] = 3
        arr[2, 1] = 3
        arr[3, 1] = 3
        arr[3, 2] = 3
        arr[3, 3] = 3
        # L-shape of 5 pixels, value 3
        xs_out, gdal_out = _run_both(arr, threshold=5, connectivity=4)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Size 5 = threshold, should survive
        assert xs_out[1, 1] == 3

    def test_l_shape_8conn(self):
        """L-shape under 8-connectivity should be the same region."""
        arr = np.ones((6, 6), dtype=np.int32)
        arr[1, 1] = 3
        arr[2, 1] = 3
        arr[3, 1] = 3
        arr[3, 2] = 3
        arr[3, 3] = 3
        xs_out, gdal_out = _run_both(arr, threshold=5, connectivity=8)
        np.testing.assert_array_equal(xs_out, gdal_out)
        assert xs_out[1, 1] == 3


# ---------------------------------------------------------------------------
# Nodata handling
# ---------------------------------------------------------------------------


class TestNodata:
    """Parity with nodata/mask regions."""

    def test_nodata_border(self):
        """Nodata ring around the edge, valid interior."""
        arr = np.full((8, 8), 0, dtype=np.int32)  # 0 = nodata
        arr[2:6, 2:6] = 1
        arr[3, 3] = 2  # single noise pixel in interior
        xs_out, gdal_out = _run_both_with_nodata(arr, nodata_val=0, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        assert xs_out[3, 3] == 1  # noise replaced

    def test_nodata_hole(self):
        """Nodata hole inside a region."""
        arr = np.ones((6, 6), dtype=np.int32)
        arr[2, 2] = 0  # nodata
        arr[2, 3] = 0  # nodata
        arr[4, 4] = 2  # single noise pixel
        xs_out, gdal_out = _run_both_with_nodata(arr, nodata_val=0, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        # Nodata pixels preserved
        assert xs_out[2, 2] == 0
        assert xs_out[2, 3] == 0
        # Noise replaced
        assert xs_out[4, 4] == 1

    def test_nodata_splits_region(self):
        """Nodata row splitting a region into two halves."""
        arr = np.ones((7, 4), dtype=np.int32)
        arr[3, :] = 0  # nodata row
        # Top half: 3*4=12 pixels of value 1
        # Bottom half: 3*4=12 pixels of value 1
        # These are separate regions because nodata splits them
        arr[1, 1] = 2  # noise in top half
        arr[5, 1] = 3  # noise in bottom half
        xs_out, gdal_out = _run_both_with_nodata(arr, nodata_val=0, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)


# ---------------------------------------------------------------------------
# Realistic classified raster
# ---------------------------------------------------------------------------


class TestRealisticRaster:
    """Larger synthetic rasters that mimic real classification output."""

    def test_quadrant_with_noise(self):
        """4-class quadrant raster with 5% salt-and-pepper noise."""
        rng = np.random.RandomState(1168)
        base = np.zeros((50, 50), dtype=np.int32)
        base[:25, :25] = 1
        base[:25, 25:] = 2
        base[25:, :25] = 3
        base[25:, 25:] = 4

        noise_mask = rng.random((50, 50)) < 0.05
        noise_vals = rng.choice([1, 2, 3, 4], size=(50, 50)).astype(np.int32)
        noisy = base.copy()
        noisy[noise_mask] = noise_vals[noise_mask]

        xs_out, gdal_out = _run_both(noisy, threshold=5)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_quadrant_with_noise_8conn(self):
        """Same as above but with 8-connectivity."""
        rng = np.random.RandomState(1168)
        base = np.zeros((50, 50), dtype=np.int32)
        base[:25, :25] = 1
        base[:25, 25:] = 2
        base[25:, :25] = 3
        base[25:, 25:] = 4

        noise_mask = rng.random((50, 50)) < 0.05
        noise_vals = rng.choice([1, 2, 3, 4], size=(50, 50)).astype(np.int32)
        noisy = base.copy()
        noisy[noise_mask] = noise_vals[noise_mask]

        xs_out, gdal_out = _run_both(noisy, threshold=5, connectivity=8)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_many_classes(self):
        """10-class raster with scattered noise.

        A few pixels at band boundaries may differ due to tie-breaking
        when two neighbor regions are close in size.  We allow up to
        0.5% mismatch and verify both outputs removed the noise.
        """
        rng = np.random.RandomState(42)
        arr = np.zeros((100, 50), dtype=np.int32)
        for i in range(10):
            arr[i * 10 : (i + 1) * 10, :] = i + 1

        noise_mask = rng.random((100, 50)) < 0.03
        noise_vals = rng.randint(1, 11, size=(100, 50)).astype(np.int32)
        arr[noise_mask] = noise_vals[noise_mask]

        xs_out, gdal_out = _run_both(arr, threshold=5)
        n_diff = np.sum(xs_out != gdal_out)
        assert n_diff <= 0.005 * arr.size, (
            f"{n_diff} pixels differ ({n_diff / arr.size:.1%}), "
            f"expected < 0.5% for tie-breaking differences only"
        )
        # Both should have eliminated all noise pixels interior to bands
        for i in range(10):
            band = slice(i * 10 + 2, (i + 1) * 10 - 2)
            assert np.all(xs_out[band, 2:-2] == i + 1)
            assert np.all(gdal_out[band, 2:-2] == i + 1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tricky inputs where implementations might diverge."""

    def test_single_value(self):
        """Uniform raster -- nothing to do."""
        arr = np.full((10, 10), 7, dtype=np.int32)
        xs_out, gdal_out = _run_both(arr, threshold=50)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_two_pixel_raster(self):
        """Smallest possible non-trivial raster -- both below threshold."""
        arr = np.array([[1, 2]], dtype=np.int32)
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_single_row(self):
        """1D-like raster (single row)."""
        arr = np.array([[1, 1, 1, 2, 1, 1, 1, 1]], dtype=np.int32)
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        assert xs_out[0, 3] == 1

    def test_single_column(self):
        """Single-column raster."""
        arr = np.array([[1], [1], [1], [2], [1], [1]], dtype=np.int32)
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
        assert xs_out[3, 0] == 1

    def test_checkerboard_4conn(self):
        """Checkerboard: all regions size 1, no neighbor >= threshold."""
        arr = np.zeros((8, 8), dtype=np.int32)
        arr[::2, ::2] = 1
        arr[1::2, 1::2] = 1
        arr[arr == 0] = 2
        xs_out, gdal_out = _run_both(arr, threshold=2, connectivity=4)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_stripes(self):
        """Alternating 1-pixel-wide stripes, all below threshold."""
        arr = np.zeros((6, 6), dtype=np.int32)
        arr[:, 0::2] = 1
        arr[:, 1::2] = 2
        xs_out, gdal_out = _run_both(arr, threshold=7, connectivity=4)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_large_threshold_collapses_all(self):
        """Threshold larger than any region -- nothing merges."""
        arr = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.int32,
        )
        xs_out, gdal_out = _run_both(arr, threshold=5)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_dtype_uint8(self):
        """uint8 input should work for both."""
        arr = np.ones((5, 5), dtype=np.uint8)
        arr[2, 2] = 3
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)

    def test_dtype_int16(self):
        """int16 input."""
        arr = np.ones((5, 5), dtype=np.int16)
        arr[2, 2] = -1
        xs_out, gdal_out = _run_both(arr, threshold=2)
        np.testing.assert_array_equal(xs_out, gdal_out)
