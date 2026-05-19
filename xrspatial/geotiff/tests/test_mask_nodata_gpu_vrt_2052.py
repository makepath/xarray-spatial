"""Backend-parity coverage for ``mask_nodata=False`` (follow-up to #2052).

The ``mask_nodata`` kwarg added in #2052 is wired through every public
reader entry point in ``xrspatial.geotiff`` (``open_geotiff``,
``read_geotiff_gpu``, ``read_geotiff_dask``, ``read_vrt``), but the
original test module ``test_mask_nodata_kwarg_2052.py`` only exercises
the eager-numpy and dask+numpy branches. The GPU, dask+GPU, and VRT
(eager + dask) branches read the kwarg from distinct call sites:

* ``read_geotiff_gpu`` -- ``_backends/gpu.py:709`` (single-shot decode)
* ``read_geotiff_gpu(chunks=)`` -- ``_backends/gpu.py:991`` then
  ``_backends/gpu.py:1016`` (dask-graph builder forwarding)
* ``read_vrt`` -- ``_backends/vrt.py:320`` (eager) and
  ``_backends/vrt.py:408`` / ``_backends/vrt.py:588`` (dask + chunked
  branch builders)

A regression dropping ``mask_nodata`` from any of those call sites
would silently promote integer rasters to ``float64`` even when the
caller asked to keep the source dtype. The accuracy sweep would not
flag it (the values are still correct), the metadata sweep would not
flag it (``attrs['nodata']`` is still stamped), and the contract
docstring would still read as documented -- which makes the missing
backend coverage the only smoke detector.

Each test below uses the same uint16 fixture as the eager tests so
the cross-backend equivalence is bit-exact when ``mask_nodata=False``
is honoured.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
    write_vrt,
)
from xrspatial.geotiff._writer import write


def _cupy_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _cupy_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


# Sentinel pixel positions, hand-picked to ensure
# ``mask_nodata=True`` would have promoted the dtype.
_SENTINEL_POS = [(0, 0), (1, 2), (2, 3), (3, 0)]


@pytest.fixture
def uint16_with_matching_sentinel(tmp_path):
    """uint16 TIFF where nodata=0 and the array has zeros in it.

    Identical layout to the eager fixture in
    ``test_mask_nodata_kwarg_2052.py`` so the cross-backend
    comparison is straightforward.

    Written tiled + deflate so the GPU reader's pure nvCOMP-decode
    path (``_backends/gpu.py:709``) is exercised, not just the
    CPU-fallback path that piggybacks on ``_read_to_array``.
    """
    arr = np.array(
        [[0, 100, 200, 300],
         [400, 500, 0, 600],
         [700, 800, 900, 0],
         [0, 1100, 1200, 1300]],
        dtype=np.uint16,
    )
    path = str(tmp_path / "uint16_match_2052_gpu_vrt.tif")
    write(arr, path, nodata=0, compression="deflate", tiled=True,
          tile_size=256)
    return path, arr


@pytest.fixture
def uint16_vrt_with_matching_sentinel(tmp_path):
    """A single-source VRT wrapping the uint16 fixture above.

    Single-source VRTs are the simplest mosaic shape that still goes
    through the full VRT reader (XML parse, source resolution, nodata
    propagation, per-source dtype check). Multi-source VRTs share the
    same ``mask_nodata`` code path so a regression here would fire
    regardless of source count.
    """
    arr = np.array(
        [[0, 100, 200, 300],
         [400, 500, 0, 600],
         [700, 800, 900, 0],
         [0, 1100, 1200, 1300]],
        dtype=np.uint16,
    )
    tif_path = str(tmp_path / "uint16_match_2052_vrt_src.tif")
    write(arr, tif_path, nodata=0, compression="none", tiled=False)

    vrt_path = str(tmp_path / "uint16_match_2052.vrt")
    write_vrt(vrt_path, [tif_path])
    return vrt_path, arr


# ---- GPU (eager, cupy-backed) ----------------------------------------


@_gpu_only
def test_read_geotiff_gpu_mask_nodata_false_preserves_uint16(
        uint16_with_matching_sentinel):
    """``read_geotiff_gpu(mask_nodata=False)`` keeps the uint16 dtype."""
    path, arr = uint16_with_matching_sentinel

    da = read_geotiff_gpu(path, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)
    # ``attrs['nodata']`` still surfaces the declared sentinel so
    # downstream code can mask explicitly.
    assert da.attrs["nodata"] == 0


@_gpu_only
def test_read_geotiff_gpu_default_mask_nodata_true_still_promotes(
        uint16_with_matching_sentinel):
    """The GPU default is unchanged: ``mask_nodata=True`` promotes."""
    import cupy

    path, _ = uint16_with_matching_sentinel
    da = read_geotiff_gpu(path)

    assert da.dtype == np.float64
    nan_count = int(cupy.isnan(da.data).sum().get())
    assert nan_count == len(_SENTINEL_POS)


@_gpu_only
def test_open_geotiff_gpu_mask_nodata_false_threads_through(
        uint16_with_matching_sentinel):
    """``open_geotiff(gpu=True, mask_nodata=False)`` reaches the GPU
    reader's opt-out branch with the kwarg intact.

    Pins the dispatcher forwarding at ``__init__.py:530``; a regression
    dropping the kwarg from the ``gpu=True`` branch would silently
    fall back to the default ``mask_nodata=True`` behaviour.
    """
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, gpu=True, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_dtype_uint16_with_mask_nodata_false(
        uint16_with_matching_sentinel):
    """``dtype="uint16"`` works on the GPU branch when the opt-out is set.

    Mirrors ``test_mask_nodata_kwarg_2052.py::
    test_regression_dtype_uint16_was_unreachable`` for the GPU path.
    """
    path, arr = uint16_with_matching_sentinel
    da = read_geotiff_gpu(path, dtype="uint16", mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)


# ---- dask + GPU (cupy-backed chunked) --------------------------------


@_gpu_only
def test_read_geotiff_gpu_dask_mask_nodata_false_preserves_uint16(
        uint16_with_matching_sentinel):
    """``read_geotiff_gpu(chunks=..., mask_nodata=False)`` keeps uint16."""
    path, arr = uint16_with_matching_sentinel
    da = read_geotiff_gpu(path, chunks=2, mask_nodata=False)

    # Graph-level dtype must reflect the opt-out before any chunk runs.
    assert da.dtype == np.uint16

    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_dask_default_mask_nodata_true_still_promotes(
        uint16_with_matching_sentinel):
    """The dask+GPU default still promotes the graph dtype to float64.

    Pinning the default explicitly so a regression that mis-wires the
    opt-out (e.g. ``mask_nodata`` defaulting to False in the chunked
    branch) would flip this test red.
    """
    import cupy

    path, _ = uint16_with_matching_sentinel
    da = read_geotiff_gpu(path, chunks=2)

    assert da.dtype == np.float64
    computed = da.compute()
    nan_count = int(cupy.isnan(computed.data).sum().get())
    assert nan_count == len(_SENTINEL_POS)


@_gpu_only
def test_open_geotiff_dask_gpu_mask_nodata_false_threads_through(
        uint16_with_matching_sentinel):
    """``open_geotiff(gpu=True, chunks=N, mask_nodata=False)`` flows
    the kwarg all the way through the dispatcher + dask-graph builder
    to the cupy chunk reader."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, gpu=True, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    np.testing.assert_array_equal(computed.data.get(), arr)


# ---- VRT (eager) -----------------------------------------------------


def test_read_vrt_mask_nodata_false_preserves_uint16(
        uint16_vrt_with_matching_sentinel):
    """``read_vrt(mask_nodata=False)`` keeps the uint16 dtype on the
    eager VRT path."""
    vrt_path, arr = uint16_vrt_with_matching_sentinel
    da = read_vrt(vrt_path, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)
    assert da.attrs["nodata"] == 0


def test_read_vrt_default_mask_nodata_true_still_promotes(
        uint16_vrt_with_matching_sentinel):
    """The VRT default unchanged: ``mask_nodata=True`` promotes."""
    vrt_path, _ = uint16_vrt_with_matching_sentinel
    da = read_vrt(vrt_path)

    assert da.dtype == np.float64
    assert int(np.isnan(da.values).sum()) == len(_SENTINEL_POS)


def test_open_geotiff_vrt_mask_nodata_false_threads_through(
        uint16_vrt_with_matching_sentinel):
    """``open_geotiff(<vrt>, mask_nodata=False)`` reaches the VRT
    reader's opt-out branch via the dispatcher."""
    vrt_path, arr = uint16_vrt_with_matching_sentinel
    da = open_geotiff(vrt_path, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)


def test_read_vrt_dtype_uint16_with_mask_nodata_false(
        uint16_vrt_with_matching_sentinel):
    """``dtype="uint16"`` works on the VRT path with the opt-out."""
    vrt_path, arr = uint16_vrt_with_matching_sentinel
    da = read_vrt(vrt_path, dtype="uint16", mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)


# ---- VRT (dask, chunked numpy) ---------------------------------------


def test_read_vrt_chunked_mask_nodata_false_preserves_uint16(
        uint16_vrt_with_matching_sentinel):
    """``read_vrt(chunks=..., mask_nodata=False)`` keeps uint16 on
    the dask+VRT path. Pins both the graph dtype declaration and the
    per-chunk reader's mask gating."""
    vrt_path, arr = uint16_vrt_with_matching_sentinel
    da = read_vrt(vrt_path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(computed.values), arr)


def test_read_vrt_chunked_default_mask_nodata_true_still_promotes(
        uint16_vrt_with_matching_sentinel):
    """The chunked-VRT default still promotes to float64."""
    vrt_path, _ = uint16_vrt_with_matching_sentinel
    da = read_vrt(vrt_path, chunks=2)

    assert da.dtype == np.float64
    computed = da.compute()
    assert int(np.isnan(computed.values).sum()) == len(_SENTINEL_POS)


def test_open_geotiff_vrt_chunked_mask_nodata_false_threads_through(
        uint16_vrt_with_matching_sentinel):
    """``open_geotiff(<vrt>, chunks=N, mask_nodata=False)`` reaches
    the chunked-VRT opt-out branch via the dispatcher."""
    vrt_path, arr = uint16_vrt_with_matching_sentinel
    da = open_geotiff(vrt_path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    np.testing.assert_array_equal(np.asarray(computed.values), arr)


# ---- Cross-backend parity ------------------------------------------


def test_cross_backend_parity_eager_dask_numpy(
        uint16_with_matching_sentinel):
    """Eager-numpy and dask+numpy agree bit-exact with ``mask_nodata=False``.

    Companion to the GPU cross-backend test below. Lives here because
    the rest of the module already builds the fixture in the right
    shape and the test reads naturally next to the GPU one.
    """
    path, arr = uint16_with_matching_sentinel

    eager = open_geotiff(path, mask_nodata=False)
    dask_ = open_geotiff(path, chunks=2, mask_nodata=False).compute()

    assert eager.dtype == np.uint16
    assert dask_.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, dask_.values)
    np.testing.assert_array_equal(eager.values, arr)


@_gpu_only
def test_cross_backend_parity_eager_gpu(
        uint16_with_matching_sentinel):
    """Eager-numpy and eager-GPU agree bit-exact with ``mask_nodata=False``."""
    path, arr = uint16_with_matching_sentinel

    eager = open_geotiff(path, mask_nodata=False)
    gpu = open_geotiff(path, gpu=True, mask_nodata=False)

    assert eager.dtype == np.uint16
    assert gpu.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, gpu.data.get())
    np.testing.assert_array_equal(eager.values, arr)


@_gpu_only
def test_cross_backend_parity_eager_dask_gpu(
        uint16_with_matching_sentinel):
    """Eager-numpy and dask+GPU agree bit-exact with ``mask_nodata=False``."""
    path, arr = uint16_with_matching_sentinel

    eager = open_geotiff(path, mask_nodata=False)
    dgpu = open_geotiff(
        path, gpu=True, chunks=2, mask_nodata=False).compute()

    assert eager.dtype == np.uint16
    assert dgpu.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, dgpu.data.get())
    np.testing.assert_array_equal(eager.values, arr)


def test_cross_backend_parity_eager_vrt(
        uint16_with_matching_sentinel,
        uint16_vrt_with_matching_sentinel):
    """A single-source VRT and the underlying TIFF return identical
    arrays under ``mask_nodata=False``.

    Confirms the VRT reader's opt-out semantics match the direct-file
    reader's semantics, i.e. the kwarg is not just plumbed through
    but actually applied at the same step."""
    tif_path, arr = uint16_with_matching_sentinel
    vrt_path, _ = uint16_vrt_with_matching_sentinel

    eager = open_geotiff(tif_path, mask_nodata=False)
    vrt = open_geotiff(vrt_path, mask_nodata=False)

    assert eager.dtype == vrt.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, np.asarray(vrt.values))
    np.testing.assert_array_equal(eager.values, arr)


# ---- Direct backend entry-point coverage ----------------------------


def test_read_geotiff_dask_direct_mask_nodata_false(
        uint16_with_matching_sentinel):
    """The direct ``read_geotiff_dask`` entry point also accepts the
    kwarg (not just through the ``open_geotiff`` dispatcher)."""
    path, arr = uint16_with_matching_sentinel
    da = read_geotiff_dask(path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, arr)
