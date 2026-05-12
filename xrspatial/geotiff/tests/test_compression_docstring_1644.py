"""Regression test for #1644: ``write_geotiff_gpu`` compression docstring
parity vs ``to_geotiff``.

The api-consistency sweep on 2026-05-11 flagged that
``write_geotiff_gpu.__doc__`` listed only four codecs (``'zstd'``,
``'deflate'``, ``'jpeg'``, ``'none'``) under the ``compression``
parameter, while the implementation actually accepts every codec
``to_geotiff`` does. Codecs unsupported by nvCOMP fall through to the
CPU encoders (``lzw``, ``packbits``, ``lz4``, ``lerc``, ``jpeg2000`` /
``j2k``) so the output matches the CPU writer byte-for-byte. This
module pins the full codec list against future drift and confirms the
underlying entry point accepts the codec names that the docstring now
advertises.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import write_geotiff_gpu


def _gpu_available() -> bool:
    """True when cupy imports and CUDA is initialised."""
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


# The full set ``to_geotiff`` accepts, mirrored to ``write_geotiff_gpu``
# so both entry points stay in lockstep. Excludes ``jpeg`` because PR
# #1633 already pins that name and the ``to_geotiff`` runtime rejects
# it -- but it is still listed in the docstring as an accepted codec
# name, matching ``to_geotiff``'s wording.
_GPU_FALLBACK_CODECS = (
    "lzw", "packbits", "lz4", "lerc", "jpeg2000", "j2k",
)


def test_write_geotiff_gpu_docstring_lists_full_codec_set():
    """The ``compression`` docstring lists every codec ``to_geotiff`` accepts.

    Prior to #1644 the docstring listed only ``'zstd'``, ``'deflate'``,
    ``'jpeg'``, and ``'none'``, which made the GPU writer look much
    more restrictive than it actually is. The block below pins the
    canonical wording.
    """
    doc = write_geotiff_gpu.__doc__
    assert doc is not None, "write_geotiff_gpu lost its docstring"
    block_start = doc.index("compression : str")
    block_end = doc.index("compression_level", block_start)
    block = doc[block_start:block_end]
    # Every codec name in the canonical list must appear in the block.
    # Use single-quoted form because that is how the docstring writes them.
    for codec in (
        "'none'", "'deflate'", "'lzw'", "'jpeg'", "'packbits'",
        "'zstd'", "'lz4'", "'jpeg2000'", "'j2k'", "'lerc'",
    ):
        assert codec in block, (
            f"compression docstring missing {codec}; current block:\n{block}"
        )


@_gpu_only
@pytest.mark.parametrize("codec", _GPU_FALLBACK_CODECS)
def test_write_geotiff_gpu_accepts_cpu_fallback_codecs(tmp_path, codec):
    """Codecs without a GPU encoder still write successfully via CPU.

    Confirms the docstring's promise that the GPU writer accepts the
    same codec set as ``to_geotiff``. ``jpeg`` is exercised separately
    by ``test_features.py`` because the test data must be 8-bit
    integer. ``jpeg2000`` / ``j2k`` go through ``glymur`` which only
    accepts uint8/uint16 -- pick a uint16 source for those codecs so
    the encode path is the one users actually hit, not a dtype-rejected
    pre-check inside glymur.
    """
    import cupy

    if codec in ("jpeg2000", "j2k"):
        arr_cpu = np.random.RandomState(0).randint(
            0, 65535, size=(64, 64), dtype=np.uint16,
        )
    else:
        arr_cpu = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    da = xr.DataArray(
        cupy.asarray(arr_cpu), dims=["y", "x"],
        coords={"y": np.arange(64.0, 0, -1), "x": np.arange(64.0)},
        attrs={"crs": 4326,
               "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 64.0)},
    )
    path = str(tmp_path / f"out_{codec}.tif")
    write_geotiff_gpu(da, path, compression=codec)
    assert os.path.exists(path), (
        f"write_geotiff_gpu(compression={codec!r}) failed to write a file"
    )
    # File must be non-empty so we know the encode path actually ran
    assert os.path.getsize(path) > 0
