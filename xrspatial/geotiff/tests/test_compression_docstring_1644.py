"""Regression test for #1644: ``write_geotiff_gpu`` compression docstring
parity vs ``to_geotiff``.

The api-consistency sweep on 2026-05-11 flagged that
``write_geotiff_gpu.__doc__`` listed only four codecs (``'zstd'``,
``'deflate'``, ``'jpeg'``, ``'none'``) under the ``compression``
parameter, while the implementation actually accepts every codec
``to_geotiff`` does.

Routing for the additional codecs:

* ``'lzw'``, ``'packbits'``, ``'lz4'``, ``'lerc'`` -- not nvCOMP-
  accelerated and have no GPU library, so they fall through to the
  CPU encoder. Byte-for-byte identical to ``to_geotiff``.
* ``'jpeg2000'`` / ``'j2k'`` -- attempts an nvJPEG2K *GPU* encode
  first via ``_nvjpeg2k_batch_encode`` and falls back to the CPU
  ``glymur`` encoder only when libnvjpeg2k is unavailable. The two
  paths are NOT byte-stable against each other; this module pins the
  acceptance contract (the codec name is accepted and a file gets
  written), not output-byte parity with the CPU writer.
* ``'jpeg'`` -- accepted here even though ``to_geotiff`` rejects it
  (the CPU writer omits the JPEGTables tag, so its output doesn't
  round-trip through GDAL). The GPU path emits self-contained JFIF
  tiles. Covered separately by
  ``test_gpu_writer_compression_modes_2026_05_11.py``; this module
  excludes it from the parametrized fallback list because the test
  data needs to be uint8 with sensible pixel content.

This module pins the full codec list against future drift and confirms
the underlying entry point accepts the codec names that the docstring
now advertises.
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


# Codecs to exercise end-to-end through the GPU writer to confirm they
# accept the docstring's advertised names. Excludes ``jpeg`` because
# (a) ``to_geotiff`` rejects it at runtime and (b) the JPEG round-trip
# is covered with appropriate uint8 RGB data in
# ``test_gpu_writer_compression_modes_2026_05_11.py``; keeping it out of
# this parametrize avoids exercising the JPEG path on dtype/shape
# combinations that aren't representative.
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
    by ``test_gpu_writer_compression_modes_2026_05_11.py`` because the
    test data must be uint8 with sensible content. ``jpeg2000`` /
    ``j2k`` will attempt nvJPEG2K if available and fall back to
    ``glymur`` otherwise; either way the encoder needs uint8/uint16
    input, so pick a uint16 source for those codecs so the encode path
    is the one users actually hit, not a dtype-rejected pre-check.
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
