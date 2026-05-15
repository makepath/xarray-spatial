"""Regression test for issue #1909.

The GDS chunked GPU read path (``_read_geotiff_gpu_chunked_gds``)
declares ``float64`` on the dask graph whenever the source file has an
integer nodata sentinel that round-trips through the source dtype, but
each chunk task returned the raw source dtype if no pixel hit the
sentinel. The result was a silent declared/actual dtype mismatch: the
dask array advertised float64 while ``.compute()`` produced uint16
buffers from chunks where the sentinel didn't appear.

The fix casts each chunk to the declared dtype before returning, gated
on ``arr.dtype != declared_dtype`` so the no-op case skips the
``astype(copy=True)`` allocation (same #1624 optimisation as the CPU
dask path).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    to_geotiff,
)

cupy = pytest.importorskip("cupy")


@pytest.fixture
def uint16_no_sentinel_path():
    """A uint16 GeoTIFF with declared nodata=9999 and no sentinel pixels."""
    arr = (np.arange(40, dtype=np.uint16) + 100).reshape(5, 8)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = Path(f.name)
    try:
        to_geotiff(arr, str(path), compression="none", nodata=9999)
        yield path
    finally:
        if path.exists():
            path.unlink()


def test_chunked_gpu_declared_dtype_matches_computed(uint16_no_sentinel_path):
    """Declared dask graph dtype must equal the computed chunk dtype."""
    da = read_geotiff_gpu(str(uint16_no_sentinel_path), chunks=4)
    declared = da.data.dtype
    computed = da.data.compute().dtype
    assert declared == computed, (
        f"GDS chunked path declared {declared} but computed {computed}; "
        f"this is the issue #1909 silent declared/actual dtype mismatch."
    )


def test_chunked_gpu_dtype_matches_cpu_dask(uint16_no_sentinel_path):
    """The dask+cupy declared dtype must match the dask+numpy path."""
    cpu = read_geotiff_dask(str(uint16_no_sentinel_path), chunks=4)
    gpu = read_geotiff_gpu(str(uint16_no_sentinel_path), chunks=4)
    assert cpu.data.dtype == gpu.data.dtype, (
        f"CPU dask declared {cpu.data.dtype} but GPU dask declared "
        f"{gpu.data.dtype}; backends must agree on graph dtype."
    )


def test_chunked_gpu_eager_paths_keep_source_dtype(uint16_no_sentinel_path):
    """Eager paths (no chunks) keep the source uint16 dtype.

    Pinned to lock the contract: the eager paths only promote when an
    actual sentinel pixel hits; the dask paths always promote when a
    sentinel is declared. Both contracts are valid; the fix only ties
    the GDS chunked path to the dask contract.
    """
    np_da = open_geotiff(str(uint16_no_sentinel_path))
    cp_da = read_geotiff_gpu(str(uint16_no_sentinel_path))
    assert np_da.dtype == np.uint16
    assert cp_da.dtype == np.uint16


def test_chunked_gpu_no_nodata_keeps_source_dtype(tmp_path):
    """No nodata declared => declared dtype stays at source dtype."""
    arr = (np.arange(40, dtype=np.uint16) + 100).reshape(5, 8)
    path = tmp_path / "no_nodata.tif"
    to_geotiff(arr, str(path), compression="none")
    da = read_geotiff_gpu(str(path), chunks=4)
    assert da.data.dtype == np.uint16
    assert da.data.compute().dtype == np.uint16


def test_chunked_gpu_explicit_dtype_kwarg_threads_through(tmp_path):
    """Explicit dtype= overrides nodata promotion and chunks land in it."""
    arr = (np.arange(40, dtype=np.uint16) + 100).reshape(5, 8)
    path = tmp_path / "explicit_dtype.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999)
    da = read_geotiff_gpu(str(path), chunks=4, dtype="float32")
    assert da.data.dtype == np.float32
    assert da.data.compute().dtype == np.float32


def test_chunked_gpu_sentinel_hit_still_promotes(tmp_path):
    """A chunk that hits the sentinel still NaN-masks and lands in float64."""
    arr = (np.arange(40, dtype=np.uint16) + 100).reshape(5, 8)
    arr[0, 0] = 9999
    path = tmp_path / "sentinel_hit.tif"
    to_geotiff(arr, str(path), compression="none", nodata=9999)
    da = read_geotiff_gpu(str(path), chunks=4)
    assert da.data.dtype == np.float64
    computed = da.data.compute()
    assert computed.dtype == np.float64
    assert np.isnan(computed[0, 0])
