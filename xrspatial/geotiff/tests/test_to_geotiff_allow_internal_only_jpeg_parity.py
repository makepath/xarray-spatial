"""API parity: ``to_geotiff`` must accept ``allow_internal_only_jpeg``.

Background
----------
Issue #1845 added the ``allow_internal_only_jpeg`` opt-in to
``write_geotiff_gpu`` so callers who explicitly want the experimental
internal-reader-only JPEG-in-TIFF encode path could reach it. The CPU
dispatcher ``to_geotiff`` kept rejecting ``compression='jpeg'``
unconditionally at the top of the function, which meant the
``to_geotiff(..., gpu=True)`` auto-dispatch path could never reach the
GPU writer's opt-in: the rejection fired before the GPU dispatch.

The two public writers therefore disagreed about the supported codec
set. Callers reading the GPU writer's docstring saw an opt-in flag they
could not actually invoke through the public dispatcher. The fix adds
the same kwarg to ``to_geotiff``, gates the up-front jpeg rejection on
it, and forwards it through the GPU dispatch path so the two entry
points expose a consistent surface.

These tests pin the parity: signature, rejection, opt-in warning, and
end-to-end write round-trip via the internal reader.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    GeoTIFFFallbackWarning,
    open_geotiff,
    to_geotiff,
    write_geotiff_gpu,
)


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


def test_to_geotiff_signature_has_allow_internal_only_jpeg():
    """``to_geotiff`` must expose ``allow_internal_only_jpeg`` so callers
    on the auto-dispatch path can reach the same opt-in
    ``write_geotiff_gpu`` exposes (issue #1845).

    Pinning the signature catches accidental removal during future
    refactors: if the kwarg disappears, the dispatcher silently drops
    back to the unconditional jpeg rejection and the parity regresses.
    """
    params = inspect.signature(to_geotiff).parameters
    assert "allow_internal_only_jpeg" in params, (
        "to_geotiff must accept allow_internal_only_jpeg for parity "
        "with write_geotiff_gpu (issue #1845)."
    )
    # Default matches write_geotiff_gpu so callers passing the kwarg to
    # either entry point see identical behaviour.
    assert params["allow_internal_only_jpeg"].default is False


def test_to_geotiff_rejects_jpeg_without_opt_in(tmp_path):
    """``to_geotiff(compression='jpeg')`` without the opt-in still
    raises ``ValueError``. The default rejection is the entire reason
    the flag exists; flipping the default would re-introduce the
    interop break #1845 fixed.
    """
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected.tif")
    with pytest.raises(ValueError, match="JPEGTables"):
        to_geotiff(da, path, compression='jpeg')


def test_to_geotiff_jpeg_rejection_message_mentions_opt_in(tmp_path):
    """The rejection message must point users at the opt-in flag so
    they can find the escape hatch. The GPU writer's matching error
    references the same flag; the two messages need to agree.
    """
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "rejected_msg.tif")
    with pytest.raises(ValueError) as exc:
        to_geotiff(da, path, compression='jpeg')
    msg = str(exc.value)
    assert "allow_internal_only_jpeg" in msg
    assert "1845" in msg


def test_to_geotiff_jpeg_opt_in_emits_warning_and_writes(tmp_path):
    """Setting ``allow_internal_only_jpeg=True`` lets the write
    proceed and emits ``GeoTIFFFallbackWarning`` so the caller knows
    the file may not round-trip through external readers.

    Round-trips through this library's reader (it decodes the JFIF
    streams directly via Pillow); the file is otherwise unreadable by
    libtiff / GDAL / rasterio, which is exactly what the warning warns
    about.
    """
    da = _make_rgb_uint8_da()
    path = str(tmp_path / "opt_in.tif")
    with pytest.warns(GeoTIFFFallbackWarning, match="JPEGTables"):
        to_geotiff(
            da, path,
            compression='jpeg',
            allow_internal_only_jpeg=True,
        )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    # Internal reader still decodes the file.
    decoded = open_geotiff(path)
    assert decoded.shape == da.shape


def test_to_geotiff_non_jpeg_unaffected_by_flag(tmp_path):
    """Passing ``allow_internal_only_jpeg=True`` on a non-JPEG codec is
    a no-op: no warning, no error. The flag must stay JPEG-specific so
    callers passing the kwarg defensively to either writer do not pay a
    cost when they pick a different codec.
    """
    import warnings as _warnings

    da = _make_rgb_uint8_da()
    path = str(tmp_path / "non_jpeg_flag.tif")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", GeoTIFFFallbackWarning)
        to_geotiff(
            da, path,
            compression='zstd',
            allow_internal_only_jpeg=True,
        )
    assert os.path.exists(path)


def test_to_geotiff_and_gpu_writer_share_kwarg_default():
    """Both writers must use the same default for the flag so a caller
    forwarding kwargs from one to the other sees identical behaviour.
    """
    eager_default = inspect.signature(
        to_geotiff).parameters["allow_internal_only_jpeg"].default
    gpu_default = inspect.signature(
        write_geotiff_gpu).parameters["allow_internal_only_jpeg"].default
    assert eager_default == gpu_default == False  # noqa: E712
