"""Dask polygonize parity tests for #2583.

The dask cross-chunk merge used to bucket chunk-boundary polygons by a
single representative float value, which produced two distinct bugs on
float rasters:

A. Geometry count diverged from the numpy reference because the value
   bucket compared pairwise against existing keys, so a transitive
   chain ``1.0 -> 1.000009 -> 1.000018`` (each pair within tolerance,
   end pair outside) did not merge across chunks even though numpy CCL
   merged it within a single chunk.

B. DN values for disconnected near-equal regions silently collapsed
   onto the bucket key, e.g. ``[1.0, 9.0, 1.000009]`` reported a third
   polygon with DN ``1.0`` instead of ``1.000009``.

Both modes are fixed by switching the cross-chunk merge to a
spatial-topology + value-closeness union-find -- only polygons that
share at least one integer-grid vertex AND have close values land in
the same group.  These tests pin numpy/dask parity for both symptoms
across dask+numpy and (where the backend is available) dask+cupy.
"""
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None

from ..polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_dask(arr, chunks):
    return xr.DataArray(da.from_array(arr, chunks=chunks))


def _to_dask_cupy(arr, chunks):
    return xr.DataArray(da.from_array(cupy.asarray(arr), chunks=chunks))


# Symptom A: chain of three near-equal floats where each adjacent pair
# is within the default tolerance but the end pair is outside.
_SYMPTOM_A = np.array([[1.0, 1.000009, 1.000018]], dtype=np.float64)

# Symptom B: two near-equal floats separated spatially by a third
# distinct value -- the close pair must stay distinct, not collapse.
_SYMPTOM_B = np.array([[1.0, 9.0, 1.000009]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Symptom A: transitive chain merging across chunks
# ---------------------------------------------------------------------------


@dask_array_available
class TestSymptomAChainMerging:
    """Dask must chain ``1.0 -> 1.000009 -> 1.000018`` across single-pixel
    chunks the same way numpy CCL chains it within a single chunk.
    """

    def test_dask_numpy_single_pixel_chunks(self):
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_A))
        v_dk, _ = polygonize(_to_dask(_SYMPTOM_A, chunks=(1, 1)))
        assert len(v_dk) == len(v_np) == 1
        assert_allclose(sorted(v_dk), sorted(v_np))

    def test_dask_numpy_two_pixel_chunks(self):
        # Chunk split between pixel 1 and pixel 2: same chain must merge.
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_A))
        v_dk, _ = polygonize(_to_dask(_SYMPTOM_A, chunks=(1, 2)))
        assert len(v_dk) == len(v_np) == 1
        assert_allclose(sorted(v_dk), sorted(v_np))


@cuda_and_cupy_available
@dask_array_available
class TestSymptomAChainMergingDaskCuPy:
    """Same chain merging guarantee on the dask+cupy backend."""

    def test_dask_cupy_single_pixel_chunks(self):
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_A))
        v_dc, _ = polygonize(_to_dask_cupy(_SYMPTOM_A, chunks=(1, 1)))
        assert len(v_dc) == len(v_np) == 1
        assert_allclose(sorted(v_dc), sorted(v_np))


# ---------------------------------------------------------------------------
# Symptom B: DN values preserved on disconnected close-valued regions
# ---------------------------------------------------------------------------


@dask_array_available
class TestSymptomBDNPreservation:
    """Two near-equal floats separated by a distinct value must keep
    their own DN values, not collapse onto a shared bucket key.
    """

    def test_dask_numpy_preserves_dn_for_disconnected_regions(self):
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_B))
        v_dk, _ = polygonize(_to_dask(_SYMPTOM_B, chunks=(1, 1)))
        assert len(v_dk) == len(v_np) == 3
        assert_allclose(sorted(v_dk), sorted(v_np))
        # Specifically pin that 1.000009 is reported, not collapsed to 1.0.
        assert any(abs(v - 1.000009) < 1e-12 for v in v_dk), (
            f"dask DN values lost 1.000009: {v_dk}"
        )

    def test_dask_numpy_disconnected_regions_two_pixel_chunks(self):
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_B))
        v_dk, _ = polygonize(_to_dask(_SYMPTOM_B, chunks=(1, 2)))
        assert len(v_dk) == len(v_np) == 3
        assert_allclose(sorted(v_dk), sorted(v_np))


@cuda_and_cupy_available
@dask_array_available
class TestSymptomBDNPreservationDaskCuPy:
    """Same DN-preservation guarantee on dask+cupy."""

    def test_dask_cupy_preserves_dn(self):
        v_np, _ = polygonize(xr.DataArray(_SYMPTOM_B))
        v_dc, _ = polygonize(_to_dask_cupy(_SYMPTOM_B, chunks=(1, 1)))
        assert len(v_dc) == len(v_np) == 3
        assert_allclose(sorted(v_dc), sorted(v_np))


# ---------------------------------------------------------------------------
# Cross-shape sanity: 2D arrays with the same patterns must also match
# ---------------------------------------------------------------------------


@dask_array_available
class TestNumpyDaskParity2D:
    """A handful of 2D shapes covering chunk-boundary chains and
    disconnected close-valued regions.  Each chunking pattern must
    match the single-chunk numpy reference for polygon count and DN
    values.
    """

    @pytest.mark.parametrize(
        "arr",
        [
            # Chain along a row across multiple chunks.
            np.array([[1.0, 1.000009, 1.000018, 1.000027]],
                     dtype=np.float64),
            # Chain along a column.
            np.array([[1.0], [1.000009], [1.000018]], dtype=np.float64),
            # Disconnected near-equal pixels at the corners.
            np.array([[1.0, 9.0],
                      [9.0, 1.000009]], dtype=np.float64),
            # Mixed: connected chain + disconnected near-equal.
            np.array([[1.0, 1.000009, 9.0, 1.000018]], dtype=np.float64),
        ],
    )
    @pytest.mark.parametrize("chunks", [(1, 1), (1, 2), (2, 1)])
    def test_parity(self, arr, chunks):
        if any(c > s for c, s in zip(chunks, arr.shape)):
            pytest.skip(f"chunks {chunks} larger than array {arr.shape}")
        v_np, _ = polygonize(xr.DataArray(arr))
        v_dk, _ = polygonize(_to_dask(arr, chunks=chunks))
        assert len(v_dk) == len(v_np), (
            f"polygon count mismatch with chunks={chunks}: "
            f"numpy={len(v_np)} dask={len(v_dk)}"
        )
        assert_allclose(sorted(v_dk), sorted(v_np))

    def test_distant_close_valued_regions_5x5(self):
        """Two close-valued regions in opposite corners of a 5x5 raster,
        separated by several chunks of a different value, must NOT
        merge into a single polygon -- adjacency probing must stay
        local, not global, even when values lie within tolerance.

        Guards against a future regression that drops the spatial
        adjacency check and re-collapses on value alone.
        """
        arr = np.array([
            [1.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 1.000009],
        ], dtype=np.float64)
        v_np, _ = polygonize(xr.DataArray(arr))
        # Numpy reference: 1.0 region, 1.000009 region, plus a single
        # 9.0 region with two holes (or two 9.0 regions depending on
        # connectivity).  What matters here is that the two 1.0-ish
        # values are reported distinctly with their original DNs.
        v_dk, _ = polygonize(_to_dask(arr, chunks=(2, 2)))
        assert len(v_dk) == len(v_np), (
            f"5x5 distant close-valued count mismatch: "
            f"numpy={len(v_np)} dask={len(v_dk)}"
        )
        assert_allclose(sorted(v_dk), sorted(v_np))
        assert any(abs(v - 1.000009) < 1e-12 for v in v_dk), (
            f"5x5 dask lost 1.000009 DN: {v_dk}"
        )
        assert any(abs(v - 1.0) < 1e-12 for v in v_dk), (
            f"5x5 dask lost 1.0 DN: {v_dk}"
        )
