"""Regression test for #1938: writer entry points return the written path.

``write_vrt`` returned ``str`` while ``to_geotiff`` and
``write_geotiff_gpu`` returned ``None``. The drift broke ``mypy``
consumers who handle the three writers uniformly and made the
Sphinx-rendered docs surface inconsistent.

This module asserts:

1. ``to_geotiff`` returns the ``path`` argument for filesystem and
   file-like destinations.
2. ``write_geotiff_gpu``'s annotation matches the canonical ``path``
   return (the runtime check is gated on cupy + CUDA availability and
   skipped here so the CPU test suite stays green).
3. ``write_vrt`` keeps returning the path (already conformant).
4. The three entry points share the same ``Returns`` annotation in
   ``inspect.signature``.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def _small_da() -> xr.DataArray:
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4)[::-1].astype(np.float64),
                "x": np.arange(4).astype(np.float64)},
        attrs={"crs": 4326},
    )


def test_to_geotiff_returns_string_path(tmp_path):
    """``to_geotiff`` returns the str path passed in."""
    da = _small_da()
    out = tmp_path / "test_1938_str.tif"
    rv = to_geotiff(da, str(out))
    assert isinstance(rv, str), (
        f"to_geotiff(str) must return a str, got {type(rv).__name__}"
    )
    assert rv == str(out)
    assert os.path.exists(rv)


def test_to_geotiff_returns_file_like(tmp_path):
    """``to_geotiff`` returns the file-like object passed in."""
    da = _small_da()
    buf = io.BytesIO()
    rv = to_geotiff(da, buf)
    assert rv is buf, (
        f"to_geotiff(BytesIO) must return the same file-like, "
        f"got {type(rv).__name__}"
    )
    # The buffer was actually written to.
    assert buf.tell() > 0 or len(buf.getvalue()) > 0


def test_to_geotiff_cog_returns_path(tmp_path):
    """COG path also returns the str path."""
    da = _small_da()
    out = tmp_path / "test_1938_cog.tif"
    rv = to_geotiff(da, str(out), cog=True, tile_size=16)
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_to_geotiff_dask_streaming_returns_path(tmp_path):
    """Dask-streaming write path also returns the str path."""
    import dask.array as da_arr

    arr = da_arr.arange(256, dtype=np.float32, chunks=64).reshape(16, 16)
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(16)[::-1].astype(np.float64),
                "x": np.arange(16).astype(np.float64)},
        attrs={"crs": 4326},
    )
    out = tmp_path / "test_1938_dask.tif"
    rv = to_geotiff(da, str(out))
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_write_vrt_returns_string_path(tmp_path):
    """``write_vrt`` (already conformant) keeps returning the str path."""
    # Create a source tif first.
    src = tmp_path / "src.tif"
    to_geotiff(_small_da(), str(src))
    vrt_path = tmp_path / "out.vrt"
    rv = write_vrt(str(vrt_path), [str(src)])
    assert isinstance(rv, str)
    assert rv == str(vrt_path)
    assert os.path.exists(rv)


@_gpu_only
def test_write_geotiff_gpu_returns_string_path(tmp_path):
    """GPU writer returns the str path (only runs with cupy + CUDA)."""
    import cupy

    arr_cpu = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr_gpu = cupy.asarray(arr_cpu)
    da = xr.DataArray(
        arr_gpu,
        dims=("y", "x"),
        coords={"y": np.arange(4)[::-1].astype(np.float64),
                "x": np.arange(4).astype(np.float64)},
        attrs={"crs": 4326},
    )
    out = tmp_path / "test_1938_gpu.tif"
    rv = write_geotiff_gpu(da, str(out))
    assert isinstance(rv, str)
    assert rv == str(out)
    assert os.path.exists(rv)


def test_writer_signatures_declare_path_return():
    """All three writers annotate the same return type.

    The annotation is a string under ``from __future__ import annotations``;
    pin the literal so the three writers cannot drift apart silently.
    """
    expected = {
        to_geotiff: "str | BinaryIO",
        write_geotiff_gpu: "str | BinaryIO",
        write_vrt: "str",
    }
    for fn, expected_ann in expected.items():
        sig = inspect.signature(fn)
        assert sig.return_annotation == expected_ann, (
            f"{fn.__name__} return annotation drifted: expected "
            f"{expected_ann!r}, got {sig.return_annotation!r}"
        )


def test_writer_returns_are_not_none(tmp_path):
    """None of the public writers may go back to returning ``None``."""
    # Use the ``tmp_path`` fixture (not ``tempfile.TemporaryDirectory``)
    # because ``write_vrt`` reads each source through the module-level
    # ``_MmapCache`` in ``_reader.py``, which keeps the file handle and
    # mmap of ``src.tif`` open after ``_FileSource.close()`` so repeated
    # reads of the same file stay cheap. On Windows that cached handle
    # blocks ``os.unlink`` (WinError 32), so a synchronous
    # ``TemporaryDirectory`` teardown raises before the test returns.
    # ``tmp_path`` defers cleanup to pytest's session-end sweep, which
    # tolerates the still-open handle the same way the other tests in
    # this file already do.
    da = _small_da()
    out = str(tmp_path / "out.tif")
    rv = to_geotiff(da, out)
    assert rv is not None
    src = str(tmp_path / "src.tif")
    to_geotiff(da, src)
    vrt_rv = write_vrt(str(tmp_path / "m.vrt"), [src])
    assert vrt_rv is not None
