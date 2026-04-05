"""Tests for xrspatial.sieve."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.sieve import sieve
from xrspatial.tests.general_checks import create_test_raster, general_output_checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raster(arr, backend):
    return create_test_raster(arr, backend)


def _to_numpy(result):
    data = result.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_removes_single_pixel_noise(backend):
    """A single pixel of class 3 surrounded by class 1 should be replaced."""
    arr = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 3, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    expected = np.ones((5, 5), dtype=np.float64)
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    general_output_checks(raster, result, expected)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_preserves_large_regions(backend):
    """Regions at or above threshold stay unchanged."""
    arr = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=4)
    general_output_checks(raster, result, arr)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_merges_into_largest_neighbor(backend):
    """A small region should merge into its largest neighbor, not the nearest."""
    # Class 2 occupies 2 pixels, surrounded by class 1 (large) and class 3 (medium).
    # The 2 pixels are adjacent to both 1 and 3, but 1 is larger.
    arr = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 2, 3],
            [1, 1, 1, 3],
            [1, 1, 1, 3],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=3)
    data = _to_numpy(result)
    # The 2-pixel region of class 2 should become class 1 (9 pixels > 3 pixels of class 3)
    assert data[1, 1] == 1.0
    assert data[1, 2] == 1.0


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_four_connectivity(backend):
    """With 4-connectivity, diagonal pixels are separate regions."""
    arr = np.array(
        [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    # With 4-connectivity: each 1 and 2 forms its own 1-pixel region
    # except center which is 1 pixel.  All regions are size 1.
    # threshold=2 should merge them all.
    result = sieve(raster, threshold=2, neighborhood=4)
    data = _to_numpy(result)
    # All pixels should end up the same value (merged into one)
    assert len(np.unique(data)) == 1


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_eight_connectivity(backend):
    """With 8-connectivity, diagonal pixels form connected regions."""
    arr = np.array(
        [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    # With 8-connectivity: all 1s are diagonally connected (5 pixels),
    # all 2s are diagonally connected (4 pixels).
    # threshold=5 means only the 2-region (4 pixels) gets sieved.
    result = sieve(raster, threshold=5, neighborhood=8)
    data = _to_numpy(result)
    assert np.all(data == 1.0)


# ---------------------------------------------------------------------------
# skip_values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_skip_values(backend):
    """Regions of skipped values are never replaced, even if small."""
    arr = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    # Class 2 is 1 pixel, below threshold=5, but skip_values=[2]
    result = sieve(raster, threshold=5, skip_values=[2.0])
    data = _to_numpy(result)
    assert data[1, 1] == 2.0  # Preserved because of skip_values


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_skip_values_still_serve_as_merge_target(backend):
    """A skip-value region can absorb a neighboring small region."""
    arr = np.array(
        [
            [2, 2, 2, 2],
            [2, 3, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    # Class 2 is skipped, class 3 is 1 pixel.  3 should merge into 2.
    result = sieve(raster, threshold=2, skip_values=[2.0])
    data = _to_numpy(result)
    assert data[1, 1] == 2.0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_preserves_nan(backend):
    """NaN pixels remain NaN after sieving."""
    arr = np.array(
        [
            [1, 1, np.nan],
            [1, 1, np.nan],
            [1, 1, np.nan],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    data = _to_numpy(result)
    assert np.all(np.isnan(data[:, 2]))
    assert np.all(data[:, :2] == 1.0)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_all_nan(backend):
    """All-NaN input returns all-NaN output."""
    arr = np.full((3, 3), np.nan, dtype=np.float64)
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    data = _to_numpy(result)
    assert np.all(np.isnan(data))


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_small_region_adjacent_only_to_nan(backend):
    """A small region with no valid neighbors stays unchanged."""
    arr = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 5.0, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=10)
    data = _to_numpy(result)
    assert data[1, 1] == 5.0
    assert np.isnan(data[0, 0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_no_small_regions(backend):
    """When all regions are above threshold, output equals input."""
    arr = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    general_output_checks(raster, result, arr)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_threshold_one(backend):
    """threshold=1 means no region is below threshold; nothing changes."""
    arr = np.array(
        [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=1)
    general_output_checks(raster, result, arr)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_single_value_raster(backend):
    """Raster with one value everywhere should be unchanged."""
    arr = np.full((4, 4), 7.0, dtype=np.float64)
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=100)
    general_output_checks(raster, result, arr)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_integer_input(backend):
    """Integer-typed rasters should work (output is float64)."""
    arr = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    data = _to_numpy(result)
    # The single pixel of 2 should be merged into 1
    assert data[1, 1] == 1.0


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_cascading_merge(backend):
    """Small regions that chain-merge across multiple sizes."""
    # Region A (1 pixel, val=3) merges into B (2 pixels, val=2),
    # then B (now 3 pixels) merges into C (large, val=1) with threshold=4.
    arr = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 3, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=4)
    data = _to_numpy(result)
    # All small regions should ultimately merge into class 1
    expected = np.ones((4, 4), dtype=np.float64)
    np.testing.assert_array_equal(data, expected)


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_preserves_attrs(backend):
    """Output should preserve input dims, coords, and attrs."""
    arr = np.ones((4, 4), dtype=np.float64)
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2)
    general_output_checks(raster, result, verify_attrs=True)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_custom_name(backend):
    arr = np.ones((3, 3), dtype=np.float64)
    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2, name="filtered")
    assert result.name == "filtered"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_sieve_rejects_3d():
    arr = np.ones((3, 3, 3), dtype=np.float64)
    raster = xr.DataArray(arr, dims=["z", "y", "x"])
    with pytest.raises(ValueError, match="2D"):
        sieve(raster, threshold=2)


def test_sieve_rejects_bad_neighborhood():
    arr = np.ones((3, 3), dtype=np.float64)
    raster = xr.DataArray(arr, dims=["y", "x"])
    with pytest.raises(ValueError, match="neighborhood"):
        sieve(raster, threshold=2, neighborhood=6)


def test_sieve_rejects_bad_threshold():
    arr = np.ones((3, 3), dtype=np.float64)
    raster = xr.DataArray(arr, dims=["y", "x"])
    with pytest.raises(ValueError, match="threshold"):
        sieve(raster, threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        sieve(raster, threshold=-1)


def test_sieve_rejects_ndarray():
    with pytest.raises(TypeError, match="xarray.DataArray"):
        sieve(np.zeros((5, 5)), threshold=2)


# ---------------------------------------------------------------------------
# Dask memory guard
# ---------------------------------------------------------------------------


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_sieve_dask_memory_guard():
    """Should raise MemoryError before .compute() on huge arrays."""
    from unittest.mock import patch

    from xrspatial.sieve import _sieve_dask

    huge = da.zeros(
        (100_000, 100_000), chunks=(1000, 1000), dtype=np.float64
    )

    with patch(
        "xrspatial.sieve._available_memory_bytes", return_value=1 * 1024**3
    ):
        with pytest.raises(MemoryError, match="global operation"):
            _sieve_dask(huge, 10, 4, None)


# ---------------------------------------------------------------------------
# Numpy / dask consistency
# ---------------------------------------------------------------------------


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_sieve_numpy_dask_match():
    """Numpy and dask backends should produce identical results."""
    arr = np.array(
        [
            [1, 1, 2, 3, 3],
            [1, 4, 2, 3, 5],
            [1, 1, 2, 3, 3],
            [6, 6, 2, 2, 2],
            [6, 6, 6, 6, 6],
        ],
        dtype=np.float64,
    )
    np_raster = _make_raster(arr, "numpy")
    dk_raster = _make_raster(arr, "dask+numpy")

    np_result = _to_numpy(sieve(np_raster, threshold=3))
    dk_result = _to_numpy(sieve(dk_raster, threshold=3))

    np.testing.assert_array_equal(np_result, dk_result)


# ---------------------------------------------------------------------------
# Convergence warning
# ---------------------------------------------------------------------------


def test_sieve_convergence_warning():
    """Should warn when the iteration limit is reached."""
    from unittest.mock import patch

    # Create a raster where merging is artificially stalled by
    # patching _MAX_ITERATIONS to 0 so the loop never runs.
    arr = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    raster = _make_raster(arr, "numpy")

    with patch("xrspatial.sieve._MAX_ITERATIONS", 0):
        with pytest.warns(UserWarning, match="did not converge"):
            sieve(raster, threshold=2)


# ---------------------------------------------------------------------------
# Larger synthetic rasters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_noisy_classification(backend):
    """Sieve a noisy 100x100 classification with known outcome."""
    rng = np.random.RandomState(1162)
    # 4-class base raster in quadrants
    base = np.zeros((100, 100), dtype=np.float64)
    base[:50, :50] = 1
    base[:50, 50:] = 2
    base[50:, :50] = 3
    base[50:, 50:] = 4

    # Sprinkle 5 % salt-and-pepper noise
    noise_mask = rng.random((100, 100)) < 0.05
    noise_vals = rng.choice([1.0, 2.0, 3.0, 4.0], size=(100, 100))
    noisy = base.copy()
    noisy[noise_mask] = noise_vals[noise_mask]

    raster = _make_raster(noisy, backend)
    result = sieve(raster, threshold=10)
    data = _to_numpy(result)

    # After sieving, isolated noise pixels should be gone.
    # Each quadrant interior (excluding boundary) should be uniform.
    assert np.all(data[5:45, 5:45] == 1.0)
    assert np.all(data[5:45, 55:95] == 2.0)
    assert np.all(data[55:95, 5:45] == 3.0)
    assert np.all(data[55:95, 55:95] == 4.0)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sieve_many_small_regions(backend):
    """Checkerboard produces maximum region count; sieve should unify."""
    # 20x20 checkerboard: every pixel is its own 1-pixel region
    arr = np.zeros((20, 20), dtype=np.float64)
    arr[::2, ::2] = 1
    arr[1::2, 1::2] = 1
    arr[arr == 0] = 2

    raster = _make_raster(arr, backend)
    result = sieve(raster, threshold=2, neighborhood=4)
    data = _to_numpy(result)

    # With 4-connectivity every pixel is isolated (size 1).
    # threshold=2 forces all to merge.  Result should be uniform.
    assert len(np.unique(data)) == 1
