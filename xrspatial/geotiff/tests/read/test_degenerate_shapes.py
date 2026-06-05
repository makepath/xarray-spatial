"""Degenerate-shape reads across the dask and GPU backends.

The eager numpy reader already covers 1x1 / 1xN / Nx1 sources (see
``tests/test_edge_cases.py::TestWriteBoundaryShapes``), and the dask
*streaming write* path covers writing degenerate dask rasters (see
``tests/integration/test_dask_pipeline.py``). What is missing is the
read side on the non-eager backends:

* the windowed dask reader (``open_geotiff(..., chunks=...)``) splitting
  a source with a single-pixel dimension into chunks, and
* the GPU reader (``open_geotiff(..., gpu=True)``) launching its decode
  kernels on a degenerate grid (grid-size-1 launches),
* and the ``dask+gpu`` combination.

These paths work today; this file pins them so a regression in the
window-clamp math or the GPU grid launch on a 1-pixel dimension cannot
ship undetected. Each cell asserts pixel parity against the eager
numpy read of the same on-disk file.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .._helpers.markers import requires_gpu


# ---------------------------------------------------------------------------
# Degenerate fixture set: every shape with at least one size-1 dimension.
# ---------------------------------------------------------------------------

_DEGENERATE_SHAPES = {
    "1x1": np.array([[42.0]], dtype=np.float32),
    "1xN": np.arange(10, dtype=np.float32).reshape(1, 10),
    "Nx1": np.arange(10, dtype=np.float32).reshape(10, 1),
}


def _write_degenerate(tmp_path, shape_id):
    """Write a degenerate-shape georeferenced TIFF and return its path.

    The transform is supplied explicitly via ``attrs['transform']``: the
    writer cannot infer a pixel size from a single-element coord axis, so
    a 1x1 / 1xN / Nx1 array with spatial coords on both axes needs the
    affine spelled out (rasterio 6-tuple ``(px, 0, ox, 0, py, oy)``).
    """
    arr = _DEGENERATE_SHAPES[shape_id]
    height, width = arr.shape
    da = xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "y": np.arange(height - 1, -1, -1, dtype=np.float64),
            "x": np.arange(width, dtype=np.float64),
        },
        attrs={
            "crs": 4326,
            # Unit pixels, origin at the (0, height) edge: x centres at
            # 0..width-1, y centres descending height-1..0.
            "transform": (1.0, 0.0, -0.5, 0.0, -1.0, height - 0.5),
        },
    )
    path = str(tmp_path / f"degenerate_{shape_id}.tif")
    to_geotiff(da, path, compression="none", tiled=False)
    return path, arr


def _materialise(da):
    """numpy view of a possibly dask/cupy-backed DataArray."""
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


# ---------------------------------------------------------------------------
# Dask read: windowed chunking on a single-pixel dimension.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_id", list(_DEGENERATE_SHAPES))
@pytest.mark.parametrize("chunks", [1, 3, 4])
def test_dask_read_degenerate_matches_eager(tmp_path, shape_id, chunks):
    """``open_geotiff(chunks=...)`` on a degenerate source equals the eager read.

    Exercises the dask window-clamp math when the chunk size meets, splits,
    or exceeds a single-pixel dimension. The eager reader does one full
    read and never hits this windowing path.
    """
    path, arr = _write_degenerate(tmp_path, shape_id)
    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=chunks)
    # Graph builds and shape is correct before compute.
    assert lazy.shape == arr.shape
    assert lazy.dims == ("y", "x")
    np.testing.assert_array_equal(_materialise(lazy), _materialise(eager))


@pytest.mark.parametrize("shape_id", list(_DEGENERATE_SHAPES))
def test_dask_read_degenerate_preserves_coords_and_crs(tmp_path, shape_id):
    """Degenerate dask read keeps x/y coords, transform, and CRS attrs."""
    path, _ = _write_degenerate(tmp_path, shape_id)
    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=4)
    np.testing.assert_array_equal(
        lazy.coords["x"].values, eager.coords["x"].values)
    np.testing.assert_array_equal(
        lazy.coords["y"].values, eager.coords["y"].values)
    assert lazy.attrs.get("transform") == eager.attrs.get("transform")
    assert lazy.attrs.get("crs") == eager.attrs.get("crs") == 4326


# ---------------------------------------------------------------------------
# GPU read: decode-kernel launch on a degenerate grid.
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("shape_id", list(_DEGENERATE_SHAPES))
def test_gpu_read_degenerate_matches_eager(tmp_path, shape_id):
    """``open_geotiff(gpu=True)`` on a degenerate source equals the eager read.

    A 1x1 / 1xN / Nx1 source launches the GPU decode path with a
    grid-size-1 dimension. Pins that the cupy result matches the numpy
    reference byte-for-byte.
    """
    path, arr = _write_degenerate(tmp_path, shape_id)
    eager = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)
    assert gpu.shape == arr.shape
    np.testing.assert_array_equal(_materialise(gpu), _materialise(eager))


@requires_gpu
@pytest.mark.parametrize("shape_id", list(_DEGENERATE_SHAPES))
def test_dask_gpu_read_degenerate_matches_eager(tmp_path, shape_id):
    """``open_geotiff(gpu=True, chunks=...)`` on a degenerate source.

    The out-of-core GPU path combines the dask windowing and the GPU
    decode launch; both run on a single-pixel dimension here.
    """
    path, arr = _write_degenerate(tmp_path, shape_id)
    eager = open_geotiff(path)
    dask_gpu = open_geotiff(path, gpu=True, chunks=4)
    assert dask_gpu.shape == arr.shape
    np.testing.assert_array_equal(
        _materialise(dask_gpu), _materialise(eager))
