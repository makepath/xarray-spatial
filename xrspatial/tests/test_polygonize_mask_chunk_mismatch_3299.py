"""Dask mask-rechunk coverage for polygonize (#3299).

``_polygonize_dask`` rechunks the mask when its chunk layout differs from
the raster's::

    if mask_data is not None and mask_data.chunks != dask_data.chunks:
        mask_data = mask_data.rechunk(dask_data.chunks)

The public API accepts any dask mask whose type and shape match the
raster -- chunk layout is not validated -- yet every pre-existing dask
test passes a mask whose chunks match the raster's exactly, so this
branch was never exercised.

If the rechunk were dropped or broken, ``mask_data.blocks[iy, ix]``
would walk a different chunk grid than the raster's.  When the two
grids happen to have the same number of blocks (raster chunks (5, 6)
vs mask chunks (6, 7) on a (15, 18) raster: both 3x3 grids), each
integer-raster chunk would be masked by a misaligned,
differently-shaped mask block -- silently excluding the wrong pixels.
Other layouts fail loudly (IndexError on ``blocks`` or a shape
mismatch inside the chunk function).

These tests pin mismatched-chunk mask layouts against the
matching-chunks reference on the same backend, for dask+numpy and
dask+cupy, integer and float rasters, both connectivities.
"""

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial import polygonize
from xrspatial.tests.general_checks import (cuda_and_cupy_available,
                                            dask_array_available)

_SHAPE = (15, 18)
_RASTER_CHUNKS = (5, 6)

# Mask chunk layouts that differ from the raster's:
# * (6, 7): same 3x3 block-grid shape as the raster -> the layout that
#   would silently mis-mask if the rechunk guard were removed.
# * (15, 18): single-chunk mask over a multi-chunk raster.
# * (4, 5): more mask blocks than raster blocks.
_MISMATCHED_MASK_CHUNKS = [(6, 7), (15, 18), (4, 5)]


def _make_data(dtype):
    rng = np.random.default_rng(61212)
    data = rng.integers(low=0, high=3, size=_SHAPE).astype(dtype)
    rng = np.random.default_rng(61213)
    mask = rng.uniform(0, 1, size=_SHAPE) < 0.85
    return data, mask


def _make_pair(data, mask, backend, mask_chunks):
    """Build (raster, mask) DataArrays on the given dask backend."""
    if backend == "dask_cupy":
        import cupy
        data = cupy.asarray(data)
        mask = cupy.asarray(mask)
    raster = xr.DataArray(da.from_array(data, chunks=_RASTER_CHUNKS))
    mask_da = xr.DataArray(da.from_array(mask, chunks=mask_chunks))
    return raster, mask_da


def _assert_results_equal(result_a, result_b):
    vals_a, polys_a = result_a
    vals_b, polys_b = result_b
    np.testing.assert_allclose(vals_a, vals_b)
    assert len(polys_a) == len(polys_b)
    for pa, pb in zip(polys_a, polys_b):
        assert len(pa) == len(pb)
        for ra, rb in zip(pa, pb):
            np.testing.assert_allclose(ra, rb)


_BACKEND_PARAMS = [
    pytest.param("dask", marks=[dask_array_available]),
    pytest.param("dask_cupy",
                 marks=[dask_array_available, cuda_and_cupy_available]),
]


@pytest.mark.parametrize("backend", _BACKEND_PARAMS)
@pytest.mark.parametrize("mask_chunks", _MISMATCHED_MASK_CHUNKS)
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("dtype", [np.int64, np.float64],
                         ids=["int_raster", "float_raster"])
def test_mask_chunk_mismatch_matches_aligned_mask(
        dtype, connectivity, mask_chunks, backend):
    """A mask with chunks differing from the raster's gives the same
    result as the identical mask chunked to match the raster."""
    data, mask = _make_data(dtype)

    raster_ref, mask_ref = _make_pair(data, mask, backend, _RASTER_CHUNKS)
    expected = polygonize(raster_ref, mask=mask_ref,
                          connectivity=connectivity)

    raster, mask_da = _make_pair(data, mask, backend, mask_chunks)
    assert mask_da.data.chunks != raster.data.chunks
    actual = polygonize(raster, mask=mask_da, connectivity=connectivity)

    _assert_results_equal(expected, actual)


@pytest.mark.parametrize("backend", _BACKEND_PARAMS)
def test_mask_chunk_mismatch_excludes_correct_pixels(backend):
    """Sanity anchor with exact geometry: a single-chunk mask over a
    multi-chunk raster excludes exactly the masked pixel."""
    data = np.ones(_SHAPE, dtype=np.int64)
    mask = np.ones(_SHAPE, dtype=bool)
    mask[7, 9] = False  # interior pixel away from chunk corners

    raster, mask_da = _make_pair(data, mask, backend, _SHAPE)
    values, polygons = polygonize(raster, mask=mask_da, connectivity=4)

    assert values == [1]
    assert len(polygons) == 1
    # One exterior ring plus one single-pixel hole at the masked cell.
    assert len(polygons[0]) == 2
    hole = polygons[0][1]
    xs = sorted(set(hole[:, 0]))
    ys = sorted(set(hole[:, 1]))
    assert xs == [9.0, 10.0]
    assert ys == [7.0, 8.0]
