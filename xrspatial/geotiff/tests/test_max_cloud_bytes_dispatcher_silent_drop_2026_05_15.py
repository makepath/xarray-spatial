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
the kwarg.

That was the same class of dispatcher silently-drops-backend-kwarg bug
that issues #1561 (``overview_level`` to dask), #1605 (``band`` to GPU),
#1685 (``overview_level`` to VRT), and #1810 (``missing_sources`` to
non-VRT) fixed for the other backend-only kwargs. Issue #1974 closed
this last gap: the dispatcher now raises ``ValueError`` when
``max_cloud_bytes`` is supplied alongside ``gpu=True``, ``chunks=...``,
or a ``.vrt`` source.

The two sibling kwargs ``on_gpu_failure`` and ``missing_sources``
follow the same rejection pattern (see ``__init__.py:339`` and
``:355``).

Filed as issue #1974 (test-coverage sweep is test-only; the fix lives
in a separate PR). See test-coverage-state.csv pass 16.
"""
from __future__ import annotations

import io

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
# Rejection pins. After the #1974 dispatcher fix, supplying
# ``max_cloud_bytes`` alongside ``gpu=True``, ``chunks=...``, or a
# ``.vrt`` source raises ``ValueError`` rather than silently dropping
# the kwarg. Mirrors the ``on_gpu_failure`` / ``missing_sources``
# guards established by #1810.
# ---------------------------------------------------------------------

def test_dispatcher_gpu_path_rejects_max_cloud_bytes(tmp_path):
    """``gpu=True`` with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; the GPU
    branch never references it, so silently dropping the budget would
    leave callers without feedback. The dispatcher rejects up front
    before the GPU stack is even probed -- no cupy/CUDA is required.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True)


def test_dispatcher_dask_path_rejects_max_cloud_bytes(tmp_path):
    """``chunks=N`` with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; the dask
    branch (``read_geotiff_dask``) never references it.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, chunks=4)


def test_dispatcher_vrt_path_rejects_max_cloud_bytes(tmp_path):
    """``.vrt`` source with ``max_cloud_bytes=...`` raises ValueError.

    The kwarg is only consumed on the eager non-VRT path; ``read_vrt``
    never references it.
    """
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=8)


def test_dispatcher_dask_gpu_path_rejects_max_cloud_bytes(tmp_path):
    """``gpu=True + chunks=N`` rejects max_cloud_bytes.

    The GPU rejection fires first (it is checked before chunks), so
    cupy / CUDA need not be installed for this assertion.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=8, gpu=True, chunks=4)


# ---------------------------------------------------------------------
# Sentinel-default regression pins. The default value of
# ``max_cloud_bytes`` must remain the module sentinel, otherwise the
# guard above would fire on every call that omits the kwarg.
# ---------------------------------------------------------------------

def test_default_kwarg_does_not_trigger_guard_on_gpu_path(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the GPU branch.

    The dispatcher only rejects when the caller passed an explicit
    value; the sentinel default must pass through.
    """
    _skip_if_no_cupy_cuda()
    path = _build_local_tif(tmp_path)
    # No max_cloud_bytes -- should reach the GPU reader, not raise.
    out = open_geotiff(path, gpu=True)
    assert out.shape == (8, 8)


def test_default_kwarg_does_not_trigger_guard_on_dask_path(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the dask branch."""
    path = _build_local_tif(tmp_path)
    out = open_geotiff(path, chunks=4)
    assert out.shape == (8, 8)


def test_default_kwarg_does_not_trigger_guard_on_vrt_path(tmp_path):
    """Omitting ``max_cloud_bytes`` does not raise on the VRT branch."""
    vrt, _src = _build_vrt(tmp_path)
    out = open_geotiff(vrt)
    assert out.shape == (8, 8)


# ---------------------------------------------------------------------
# Explicit-``None`` pins. ``max_cloud_bytes=None`` is the documented
# "skip the check entirely" sentinel on the eager path (#1928). The
# rejection guard is sentinel-based, so an explicit ``None`` is treated
# as "caller supplied a value" and rejected on the non-eager branches
# -- consistent with how ``on_gpu_failure`` and ``missing_sources``
# treat explicit values. These tests pin that semantics.
# ---------------------------------------------------------------------

def test_explicit_none_max_cloud_bytes_rejected_on_gpu_path(tmp_path):
    """``gpu=True, max_cloud_bytes=None`` raises ValueError.

    ``None`` is the documented "disable the budget" value on the eager
    path. On the GPU path the budget is not consumed at all, so an
    explicit ``None`` still indicates the caller expected the kwarg to
    have an effect -- reject it for the same reason an explicit byte
    count is rejected.
    """
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=None, gpu=True)


def test_explicit_none_max_cloud_bytes_rejected_on_dask_path(tmp_path):
    """``chunks=N, max_cloud_bytes=None`` raises ValueError."""
    path = _build_local_tif(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(path, max_cloud_bytes=None, chunks=4)


def test_explicit_none_max_cloud_bytes_rejected_on_vrt_path(tmp_path):
    """``.vrt`` source + ``max_cloud_bytes=None`` raises ValueError."""
    vrt, _src = _build_vrt(tmp_path)
    with pytest.raises(ValueError, match=r"max_cloud_bytes"):
        open_geotiff(vrt, max_cloud_bytes=None)
