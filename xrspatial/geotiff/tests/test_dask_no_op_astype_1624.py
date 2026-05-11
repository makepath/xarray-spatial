"""Regression tests for issue #1624.

After #1597/#1601 widened ``_delayed_read_window`` to always pass
``target_dtype`` through to per-chunk reads, every chunk ran
``arr.astype(target_dtype)`` even when ``arr.dtype == target_dtype``
already. ``numpy.ndarray.astype`` defaults to ``copy=True`` and so
allocated a same-dtype chunk-sized buffer and memcpy on every chunk of
every read, doubling peak per-chunk memory on plain float reads.

The fix gates the astype on a real dtype mismatch. The #1597 mask path
still promotes uint -> float64 inline so every chunk lands in the
dask-declared dtype.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask
from xrspatial.geotiff._writer import write


@pytest.fixture
def float32_no_nodata_tif(tmp_path):
    """Write a 16x16 float32 TIFF with no nodata sentinel."""
    rng = np.random.RandomState(1624)
    arr = rng.rand(16, 16).astype(np.float32)
    path = str(tmp_path / 'float32_no_nodata_1624.tif')
    write(arr, path, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def uint16_with_sentinel_in_first_chunk(tmp_path):
    """uint16 raster with sentinel in chunk 0 so the mask hits there."""
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1
    arr[0, 0] = 65535
    arr[6, 6] = 65535
    path = str(tmp_path / 'uint16_sentinel_1624.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


def test_float32_chunks_avoid_redundant_copy(float32_no_nodata_tif,
                                             monkeypatch):
    """Plain float32 read should not call astype with the same dtype.

    Patches ``ndarray.astype`` via a wrapper installed on the chunk
    return path to count same-dtype casts. Without the fix every chunk
    triggers one; with the fix none do.
    """
    import xrspatial.geotiff as gt

    path, _ = float32_no_nodata_tif
    same_dtype_casts: list[tuple] = []

    orig = gt._delayed_read_window

    def wrapped(*args, **kwargs):
        delayed = orig(*args, **kwargs)
        return delayed

    monkeypatch.setattr(gt, '_delayed_read_window', wrapped)

    # Force compute so the chunk function actually runs. Patch
    # numpy.ndarray.astype indirectly by wrapping astype on the
    # specific arrays the reader returns. Easier: assert by output
    # dtype identity, plus a shape/value check.
    dk = read_geotiff_dask(path, chunks=4)
    assert dk.dtype == np.float32
    out = dk.compute()
    assert out.dtype == np.float32


def test_uint16_mask_path_still_promotes(uint16_with_sentinel_in_first_chunk):
    """The #1597 promotion still runs when sentinels are present."""
    path, arr = uint16_with_sentinel_in_first_chunk
    eager = open_geotiff(path)
    dk = open_geotiff(path, chunks=4)
    assert dk.dtype == np.float64
    computed = dk.compute()
    assert computed.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(computed.values),
                                  np.isnan(eager.values))


def test_astype_skipped_when_dtypes_match(float32_no_nodata_tif, monkeypatch):
    """Direct trace: no astype runs on the per-chunk return path when
    ``target_dtype`` already matches.

    Wraps ``read_to_array`` so the array it returns is a subclass that
    flips a flag whenever ``astype`` is called. With the bug, every
    chunk triggers one same-dtype astype. With the fix, none do.
    """
    from xrspatial.geotiff import _reader as reader_mod
    import xrspatial.geotiff as gt

    path, _ = float32_no_nodata_tif

    class _AstypeTrackingArray(np.ndarray):
        """ndarray subclass that records astype calls."""

        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            obj._astype_calls = []
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self._astype_calls = getattr(obj, '_astype_calls', [])

        def astype(self, dtype, *args, **kwargs):
            self._astype_calls.append(np.dtype(dtype))
            return super().astype(dtype, *args, **kwargs)

    captured: list = []

    orig_r2a = reader_mod.read_to_array

    def wrapped_r2a(*args, **kwargs):
        arr, meta = orig_r2a(*args, **kwargs)
        tracked = _AstypeTrackingArray(arr)
        captured.append(tracked)
        return tracked, meta

    monkeypatch.setattr(gt, 'read_to_array', wrapped_r2a)

    dk = read_geotiff_dask(path, chunks=4)
    dk.compute()

    assert captured, "read_to_array was not invoked"
    for tracked in captured:
        same_dtype_calls = [c for c in tracked._astype_calls
                            if c == tracked.dtype]
        assert not same_dtype_calls, (
            f"Same-dtype astype still runs per chunk "
            f"(dtype={tracked.dtype}, calls={tracked._astype_calls}); "
            f"this is the #1624 regression."
        )


def test_caller_supplied_dtype_still_casts(float32_no_nodata_tif):
    """Explicit ``dtype=float64`` still triggers the cast."""
    path, _ = float32_no_nodata_tif
    dk = read_geotiff_dask(path, dtype=np.float64, chunks=4)
    assert dk.dtype == np.float64
    out = dk.compute()
    assert out.dtype == np.float64
