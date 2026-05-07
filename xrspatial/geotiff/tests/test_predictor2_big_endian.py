"""Predictor=2 big-endian read tests.

PR #1498 reworked the predictor=2 (horizontal differencing) decode to run
sample-wise via a numpy view at the file's byte order. Numba's nopython
mode rejects arrays with a non-native byte order, so reading a big-endian
TIFF with uint16/uint32/uint64 + predictor=2 raised a ``TypingError``
("Unsupported array dtype: >u2"). The fix byte-swaps the underlying
buffer to native order around the kernel call and swaps it back, keeping
the on-disk representation intact for the downstream
``chunk.view(file_dtype)`` step in ``_decode_strip_or_tile``.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._reader import read_to_array

tifffile = pytest.importorskip("tifffile")


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32, np.int32])
def test_big_endian_predictor2_round_trip(tmp_path, dtype):
    """xrspatial reads a big-endian predictor=2 file exactly."""
    info = np.iinfo(dtype)
    arr = np.linspace(info.min // 2, info.max // 2, 64, dtype=np.int64)
    arr = arr.astype(dtype).reshape(8, 8)
    path = tmp_path / f"be_pred2_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_little_endian_predictor2_still_round_trips(tmp_path):
    """Sanity: the BE byteswap did not break the LE path."""
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) * 100
    path = tmp_path / "le_pred2_u16.tif"
    tifffile.imwrite(
        str(path), arr, byteorder="<", predictor=2, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)


def test_big_endian_predictor2_uint8_unaffected(tmp_path):
    """uint8 takes the byte-wise kernel path, which never needed swapping."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "be_pred2_u8.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2, compression="deflate")

    out, _ = read_to_array(str(path))
    np.testing.assert_array_equal(out, arr)
