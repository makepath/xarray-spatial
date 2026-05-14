"""Issue #1845: ``write_geotiff_gpu`` must reject ``compression='jpeg'``
by default.

Background
----------
``to_geotiff`` raises a ``ValueError`` for ``compression='jpeg'`` because
the encoder writes self-contained JFIF tiles without the TIFF JPEGTables
tag (347); the resulting files are unreadable by libtiff, GDAL, and
rasterio. The GPU writer sat in the same module and silently accepted
the same kwarg, producing the same broken format. The fix introduces an
``allow_internal_only_jpeg`` opt-in: callers who want the experimental
internal-reader-only path must ask for it explicitly, and they get a
``GeoTIFFFallbackWarning`` reminding them the file will not round-trip
through external readers.

These tests pin the rejection (CPU-only, no CUDA required) and the
opt-in warning behaviour. The internal-only encode path itself is
covered by the updated tests in
``test_gpu_writer_compression_modes_2026_05_11.py`` which run only when
CUDA is present.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, write_geotiff_gpu


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


def _make_rgb_uint8_da() -> xr.DataArray:
    """64x64x3 uint8 RGB raster suitable for the JPEG encode path."""
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=("y", "x", "band"),
        coords={
            "y": np.arange(64, dtype=np.float64),
            "x": np.arange(64, dtype=np.float64),
            "band": np.array([1, 2, 3], dtype=np.int32),
        },
    )


def test_write_geotiff_gpu_rejects_jpeg_without_opt_in(tmp_path):
    """``compression='jpeg'`` without the opt-in raises ``ValueError``.

    Mirrors the ``to_geotiff`` rejection so both writers in the same
    module agree about JPEG-in-TIFF interop. The check runs before any
    GPU work, so the test does not need CUDA.
    """
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected_issue_D_1845.tif")

    with pytest.raises(ValueError, match="JPEGTables"):
        write_geotiff_gpu(da, path, compression='jpeg')


def test_write_geotiff_gpu_rejects_jpeg_message_mentions_alternatives(tmp_path):
    """The rejection error mentions the same alternative codecs that
    ``to_geotiff`` recommends, so callers landing on either entry point
    learn what to switch to."""
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected_msg_issue_D_1845.tif")

    with pytest.raises(ValueError) as exc:
        write_geotiff_gpu(da, path, compression='jpeg')

    # The shared message wording from to_geotiff. If the two writers
    # drift apart, callers reading the error get inconsistent advice.
    msg = str(exc.value)
    assert "deflate" in msg
    assert "zstd" in msg


def test_write_geotiff_gpu_rejects_jpeg_case_insensitive(tmp_path):
    """Upper-case ``compression='JPEG'`` is rejected too.

    ``to_geotiff`` lower-cases the compression string before comparing,
    so the GPU writer must follow the same rule -- otherwise a caller
    who types ``'JPEG'`` slips past the gate."""
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected_upper_issue_D_1845.tif")

    with pytest.raises(ValueError, match="JPEGTables"):
        write_geotiff_gpu(da, path, compression='JPEG')


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_write_geotiff_gpu_jpeg_opt_in_emits_warning(tmp_path):
    """Setting ``allow_internal_only_jpeg=True`` proceeds with the JPEG
    encode and emits ``GeoTIFFFallbackWarning``.

    The warning is the only signal the caller gets that their file may
    not round-trip through GDAL or rasterio, so a missing warning here
    would let the footgun back in.
    """
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "opt_in_issue_D_1845.tif")

    with pytest.warns(GeoTIFFFallbackWarning, match="JPEGTables"):
        write_geotiff_gpu(
            da, path,
            compression='jpeg',
            allow_internal_only_jpeg=True,
        )

    # File was actually written, not just warned-about.
    import os
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_write_geotiff_gpu_non_jpeg_unaffected_by_flag(tmp_path):
    """Setting ``allow_internal_only_jpeg=True`` on a non-JPEG codec is a
    no-op (no warning, no error).

    The flag is JPEG-specific; other codecs must not pay any cost for
    it being present in the signature."""
    pytest.importorskip("cupy")
    if not _HAS_GPU:
        pytest.skip("cupy + CUDA required")

    da = _make_rgb_uint8_da()
    path = str(tmp_path / "non_jpeg_flag_issue_D_1845.tif")

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", GeoTIFFFallbackWarning)
        # Should not raise (no JPEG-related warning fires).
        write_geotiff_gpu(
            da, path,
            compression='zstd',
            allow_internal_only_jpeg=True,
        )
