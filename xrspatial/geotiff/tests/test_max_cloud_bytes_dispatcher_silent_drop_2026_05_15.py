"""Dispatcher parameter coverage for ``open_geotiff(max_cloud_bytes=...)``.

``max_cloud_bytes`` was added in #1928 as an eager-path cloud-budget
guard, and re-ordered into the canonical reader signature in #1957
(commit 750dc20). The kwarg is only meaningful for the eager fsspec
read path inside ``_read_to_array``: the HTTP/COG path is range-based
and the local + file-like paths skip the budget check.

The dispatcher in ``open_geotiff`` (``xrspatial/geotiff/__init__.py``)
forwards the value to ``_read_to_array`` only on the eager non-VRT
branch. The GPU branch (``read_geotiff_gpu``), the dask branch
(``read_geotiff_dask``), and the VRT branch (``read_vrt``) all ignore
the kwarg silently -- the value is accepted at the signature but never
consumed downstream.

That is the same class of dispatcher silently-drops-backend-kwarg bug
that issues #1561 (``overview_level`` to dask), #1605 (``band`` to GPU),
#1685 (``overview_level`` to VRT), and #1810 (``missing_sources`` to
non-VRT) fixed for the other backend-only kwargs. Pass 14 + 15 of the
test-coverage sweep closed several adjacent parameter gaps but did not
pin this one.

The two sibling kwargs ``on_gpu_failure`` and ``missing_sources``
already raise ``ValueError`` when used on a path where they do not
apply (the dispatcher gates them on sentinel defaults at
``__init__.py:339`` and ``:355``). ``max_cloud_bytes`` defaults to the
``_MAX_CLOUD_BYTES_SENTINEL`` and would slot into the same pattern,
but the rejection guard is missing.

This module pins the current silent-drop behaviour. The fix surface is
expected to be either:

* Add ValueError guards mirroring ``on_gpu_failure`` /
  ``missing_sources`` (refuses the kwarg on gpu / chunks / VRT paths).
* Or thread ``max_cloud_bytes`` through every backend so it has effect
  everywhere (broader change because the GPU + dask paths would need
  to plumb the budget into their respective fsspec entry points).

Either fix would flip the four ``xfail(strict=True)`` tests below from
xpass to pass after the source change. The fifth class (positive
``test_eager_*`` pins) stays green either way so the canonical eager
path keeps its current contract.

Filed as issue #1974 (test-coverage sweep is test-only; the fix lives
in a separate PR). See test-coverage-state.csv pass 16.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    to_geotiff,
    write_vrt,
)


def _skip_if_no_cupy_cuda():
    """Skip the calling test if cupy is unavailable or CUDA is offline."""
    import importlib.util
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("cupy not available")
    try:
        import cupy
        if not cupy.cuda.is_available():
            pytest.skip("CUDA unavailable on host")
    except Exception:
        pytest.skip("cupy import failed")


def _build_local_tif(tmp_path, name='src.tif'):
    """Write a small valid GeoTIFF used as the dispatcher's source."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'transform': (1.0, 0, 0.0, 0, -1.0, 8.0),
        },
    )
    path = str(tmp_path / name)
    to_geotiff(da, path)
    return path


def _build_vrt(tmp_path):
    """Build a 1-source VRT mosaic referencing a small local GeoTIFF."""
    src = _build_local_tif(tmp_path, name='vrt_src.tif')
    vrt = str(tmp_path / 'mosaic.vrt')
    write_vrt(vrt, [src])
    return vrt, src


# ---------------------------------------------------------------------
# Positive pins: the kwarg is forwarded through the eager path.
# These stay green whether the dispatcher fix raises ValueError or
# threads the budget into every backend.
# ---------------------------------------------------------------------

class TestEagerLocalPathAcceptsMaxCloudBytes:
    """Local-file eager reads accept ``max_cloud_bytes`` as a no-op.

    The docstring on ``open_geotiff`` states the budget "Has no effect
    on local file or file-like sources." A tight budget on a local
    file still reads successfully.
    """

    def test_local_file_max_cloud_bytes_small_is_noop(self, tmp_path):
        path = _build_local_tif(tmp_path)
        # 8 bytes is far below the file size; local files skip the budget.
        out = open_geotiff(path, max_cloud_bytes=8)
        assert out.shape == (8, 8)
        assert out.dtype == np.float32

    def test_local_file_max_cloud_bytes_none_is_noop(self, tmp_path):
        path = _build_local_tif(tmp_path)
        out = open_geotiff(path, max_cloud_bytes=None)
        assert out.shape == (8, 8)

    def test_local_file_max_cloud_bytes_large_is_noop(self, tmp_path):
        path = _build_local_tif(tmp_path)
        out = open_geotiff(path, max_cloud_bytes=10 ** 9)
        assert out.shape == (8, 8)


class TestEagerFileLikeAcceptsMaxCloudBytes:
    """File-like sources accept ``max_cloud_bytes`` (documented no-op)."""

    def test_bytesio_max_cloud_bytes_small_is_noop(self, tmp_path):
        path = _build_local_tif(tmp_path)
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
        out = open_geotiff(buf, max_cloud_bytes=8)
        assert out.shape == (8, 8)


# ---------------------------------------------------------------------
# Silent-drop pins.  These mark the current buggy behaviour with
# ``xfail(strict=True)``: when the dispatcher fix lands (whichever
# direction), these flip from xpass to pass.  ``strict=True`` makes
# the xpass a test failure so the diff is visible at fix time.
# ---------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "open_geotiff silently drops max_cloud_bytes when gpu=True. "
        "Should raise ValueError mirroring on_gpu_failure (#1810 pattern), "
        "or thread the budget into the GPU fsspec entry point. "
        "See test-coverage sweep pass 16."
    ),
)
def test_dispatcher_gpu_path_rejects_max_cloud_bytes(tmp_path):
    """``gpu=True`` with ``max_cloud_bytes=...`` should not silently drop.

    The kwarg is only consumed on the eager non-VRT path; the GPU
    branch at ``__init__.py:410`` never references it. Caller has no
    way to learn the budget is being ignored.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open_geotiff silently drops max_cloud_bytes when chunks=N. "
        "Should raise ValueError mirroring on_gpu_failure (#1810 pattern), "
        "or thread the budget into read_geotiff_dask. "
        "See test-coverage sweep pass 16."
    ),
)
def test_dispatcher_dask_path_rejects_max_cloud_bytes(tmp_path):
    """``chunks=N`` with ``max_cloud_bytes=...`` should not silently drop.

    The kwarg is only consumed on the eager non-VRT path; the dask
    branch at ``__init__.py:422`` never references it.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, chunks=4)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open_geotiff silently drops max_cloud_bytes for .vrt sources. "
        "Should raise ValueError mirroring missing_sources (#1810 "
        "pattern), or thread the budget into read_vrt. "
        "See test-coverage sweep pass 16."
    ),
)
def test_dispatcher_vrt_path_rejects_max_cloud_bytes(tmp_path):
    """``.vrt`` source with ``max_cloud_bytes=...`` should not silently drop.

    The kwarg is only consumed on the eager non-VRT path; the VRT
    branch at ``__init__.py:362`` never references it.
    """
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=8)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open_geotiff silently drops max_cloud_bytes when "
        "gpu=True + chunks=N (dask+cupy dispatch). "
        "Should raise ValueError mirroring on_gpu_failure (#1810 pattern). "
        "See test-coverage sweep pass 16."
    ),
)
def test_dispatcher_dask_gpu_path_rejects_max_cloud_bytes(tmp_path):
    """``gpu=True + chunks=N`` should not silently drop max_cloud_bytes."""
    _skip_if_no_cupy_cuda()
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True, chunks=4)


# ---------------------------------------------------------------------
# Pinning the buggy CURRENT behaviour so the fix diff is observable.
# These tests pass today (the kwarg is silently dropped). When the
# dispatcher fix lands they will fail and must be replaced by the
# xfail tests above flipping green. They live alongside the xfails so
# the fix author sees both the "before" and "after" expectations.
# ---------------------------------------------------------------------

# remove with #1974
class TestCurrentSilentDropPins:
    """Pin the current silent-drop behaviour.

    These tests assert that the kwarg is silently accepted today on the
    non-eager paths. They are the "before" half of the fix-visibility
    contract documented at module top. After the dispatcher fix the
    xfail siblings above flip to pass; remove this class at that time.
    """

    def test_gpu_path_silently_accepts_today(self, tmp_path):
        _skip_if_no_cupy_cuda()
        path = _build_local_tif(tmp_path)
        # No raise today; the kwarg is silently dropped.
        out = open_geotiff(path, max_cloud_bytes=8, gpu=True)
        assert out.shape == (8, 8)

    def test_dask_path_silently_accepts_today(self, tmp_path):
        path = _build_local_tif(tmp_path)
        # No raise today; the kwarg is silently dropped.
        out = open_geotiff(path, max_cloud_bytes=8, chunks=4)
        assert out.shape == (8, 8)
        # Lazy result; confirm it computes.
        arr = out.values
        assert arr.shape == (8, 8)

    def test_vrt_path_silently_accepts_today(self, tmp_path):
        vrt, _src = _build_vrt(tmp_path)
        # No raise today; the kwarg is silently dropped.
        out = open_geotiff(vrt, max_cloud_bytes=8)
        assert out.shape == (8, 8)
