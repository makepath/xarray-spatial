"""Parameter coverage for ``read_geotiff_gpu`` / ``read_geotiff_dask``.

The ``name=`` and ``max_pixels=`` kwargs flow through ``open_geotiff``'s
dispatch into the GPU and dask backends. The eager numpy path tests
both kwargs directly (e.g. ``test_cog::test_open_geotiff_custom_name``,
``test_security`` for ``max_pixels``). The dask backend covers
``max_pixels`` in ``test_backend_kwarg_parity_1561``. The remaining
gaps that this sweep (test coverage gap sweep 2026-05-11, pass 4)
closes are:

* ``read_geotiff_gpu(name=...)`` -- direct test on the GPU eager path
  and the dask+GPU path.
* ``read_geotiff_dask(name=...)`` -- direct test on the dask-on-CPU
  path.
* ``read_geotiff_gpu(max_pixels=...)`` -- both the accept and reject
  branches; the GPU pipeline calls ``_check_dimensions`` twice (once
  for the full raster, once per tile) and neither call had regression
  coverage.
* ``open_geotiff(chunks=..., name=...)`` /
  ``open_geotiff(gpu=True, name=...)`` /
  ``open_geotiff(gpu=True, chunks=..., name=...)`` -- the dispatcher
  forwards ``name=`` through three distinct branches and a silent
  drop would only show up in user code.

Adding these closes the MEDIUM Cat 4 (parameter coverage) gap that
was open after pass 3.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, read_geotiff_gpu, to_geotiff


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


@pytest.fixture
def small_tiff_path(tmp_path):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    p = tmp_path / "small.tif"
    to_geotiff(arr, str(p), tile_size=16)
    return str(p), arr


# ---------------------------------------------------------------------------
# read_geotiff_dask(name=...) -- direct
# ---------------------------------------------------------------------------


def test_read_geotiff_dask_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_dask(path, chunks=4, name="custom_dask")
    assert da.name == "custom_dask"
    np.testing.assert_array_equal(da.values, arr)


def test_read_geotiff_dask_default_name_from_path(small_tiff_path):
    path, _ = small_tiff_path
    da = read_geotiff_dask(path, chunks=4)
    # Default name is filename stem when no override is supplied.
    assert da.name == "small"


# ---------------------------------------------------------------------------
# read_geotiff_gpu(name=...) -- direct
# ---------------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_gpu(path, name="custom_gpu")
    assert da.name == "custom_gpu"
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_default_name_from_path(small_tiff_path):
    path, _ = small_tiff_path
    da = read_geotiff_gpu(path)
    assert da.name == "small"


@_gpu_only
def test_read_geotiff_gpu_chunks_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_gpu(path, chunks=4, name="custom_dask_gpu")
    assert da.name == "custom_dask_gpu"
    np.testing.assert_array_equal(da.data.compute().get(), arr)


# ---------------------------------------------------------------------------
# read_geotiff_gpu(max_pixels=...) -- accept + reject
# ---------------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_max_pixels_accepts_within_budget(small_tiff_path):
    path, arr = small_tiff_path
    # 8 * 8 = 64 pixels but per-tile dim safety check uses tile_size=16
    # (256 pixels per tile); 300 leaves room. The fixture's tile_size
    # was bumped to 16 to satisfy the TIFF 6 multiple-of-16 rule (#1767).
    da = read_geotiff_gpu(path, max_pixels=300)
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_max_pixels_rejects_oversized(small_tiff_path):
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        read_geotiff_gpu(path, max_pixels=10)


@_gpu_only
def test_read_geotiff_gpu_chunks_max_pixels_rejects_oversized(small_tiff_path):
    """Dask+GPU path also enforces ``max_pixels``."""
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        read_geotiff_gpu(path, chunks=4, max_pixels=10)


# ---------------------------------------------------------------------------
# open_geotiff dispatch: name= flows through every backend branch
# ---------------------------------------------------------------------------


def test_open_geotiff_chunks_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, chunks=4, name="dispatch_dask")
    assert da.name == "dispatch_dask"
    np.testing.assert_array_equal(da.values, arr)


@_gpu_only
def test_open_geotiff_gpu_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, name="dispatch_gpu")
    assert da.name == "dispatch_gpu"
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_chunks_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, chunks=4, name="dispatch_dask_gpu")
    assert da.name == "dispatch_dask_gpu"
    np.testing.assert_array_equal(da.data.compute().get(), arr)


# ---------------------------------------------------------------------------
# open_geotiff dispatch: max_pixels reject flows through GPU branch
# ---------------------------------------------------------------------------


@_gpu_only
def test_open_geotiff_gpu_max_pixels_rejects(small_tiff_path):
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        open_geotiff(path, gpu=True, max_pixels=10)
