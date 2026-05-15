"""Regression tests for issue #1950.

``_try_nvcomp_batch_decompress`` used to compute its per-tile host
prefix-sum offsets via a Python ``for`` loop:

```
comp_sizes_list = [len(t) for t in raw_tiles]
comp_offsets_h = np.zeros(n_tiles, dtype=np.int64)
for i in range(1, n_tiles):
    comp_offsets_h[i] = comp_offsets_h[i - 1] + comp_sizes_list[i - 1]
```

The sibling batched-D2H helper ``_batched_d2h_to_bytes`` at ~L924 and
the compress-side prefix sum in ``_nvcomp_batch_compress`` at ~L2572
both use ``np.cumsum(sizes, out=offsets[1:])``. Aligning the
decompress side keeps the codebase consistent and trims interpreter
overhead.

Two guards here:

1. Correctness -- a tiny synthetic nvCOMP round-trip (when the lib is
   available) still decodes every tile correctly. Without nvCOMP the
   test exercises the same prefix-sum reshape via direct comparison
   against ``np.cumsum``.
2. Structural -- the source uses ``np.cumsum`` (not a Python
   ``range(1, n_tiles)`` loop) for the prefix sum.
"""
from __future__ import annotations

import importlib.util
import os
import re
import tempfile

import numpy as np
import pytest


def test_nvcomp_decompress_uses_cumsum_for_offsets_1950():
    """Source-level guard against reintroducing the Python for loop.

    The fix swaps the per-tile prefix-sum loop for ``np.cumsum``.
    This test fires if anyone reverts to the loop or otherwise breaks
    the alignment with ``_batched_d2h_to_bytes`` / ``_nvcomp_batch_compress``.
    """
    import pathlib

    src_path = pathlib.Path(__file__).parent.parent / "_gpu_decode.py"
    src = src_path.read_text()

    # Anchor on the exact decompress-side prefix-sum call. Regex is
    # tighter than a fixed character window and survives surrounding
    # code edits.
    cumsum_call = re.compile(
        r"np\.cumsum\(\s*comp_sizes_arr\[:-1\]\s*,\s*"
        r"out\s*=\s*comp_offsets_h\[1:\]\s*\)"
    )
    assert cumsum_call.search(src), (
        "decompress upload block should use "
        "``np.cumsum(comp_sizes_arr[:-1], out=comp_offsets_h[1:])`` for "
        "prefix-sum offsets, aligning with _batched_d2h_to_bytes "
        "(issue #1950)."
    )
    # The legacy Python loop would have written
    # ``comp_offsets_h[i] = comp_offsets_h[i - 1] + ...`` inside a
    # ``for i in range(1, n_tiles):`` block.
    legacy_loop = re.compile(
        r"for\s+i\s+in\s+range\(\s*1\s*,\s*n_tiles\s*\)\s*:\s*\n"
        r"\s*comp_offsets_h\[i\]"
    )
    assert not legacy_loop.search(src), (
        "decompress upload block should no longer compute prefix-sum "
        "offsets with a Python for loop (issue #1950)."
    )


def test_cumsum_matches_loop_prefix_sum_1950():
    """Equivalence between the vectorised cumsum and the prior loop.

    Numeric guard. Even though the two forms produce the same output
    by construction, a runtime check confirms the cumsum form does not
    drift away from the previous semantics across numpy versions.
    """
    rng = np.random.RandomState(1950)
    n = 1024
    sizes = rng.randint(100, 100_000, size=n).astype(np.int64)

    # Vectorised form (matches the fix).
    offsets_cumsum = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(sizes[:-1], out=offsets_cumsum[1:])

    # Reference: explicit Python prefix sum.
    offsets_loop = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        offsets_loop[i] = offsets_loop[i - 1] + sizes[i - 1]

    np.testing.assert_array_equal(offsets_cumsum, offsets_loop)


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None,
    reason="cupy required for nvCOMP path",
)
def test_nvcomp_batch_decompress_roundtrip_1950():
    """End-to-end check: a deflate-tiled raster still decodes correctly.

    Exercises ``_try_nvcomp_batch_decompress`` on a real file via the
    public ``read_geotiff_gpu`` entry point. If the prefix-sum
    refactor mis-stages a tile, the decoded buffer would not match
    the source, surfacing as a numerical regression here.

    Gated on ``XRSPATIAL_GEOTIFF_STRICT_GPU=1`` so the run-time check
    only fires in environments that actually carry nvCOMP. Without the
    flag the GPU read path silently falls back to a CPU codec when
    nvCOMP is missing, which bypasses the prefix-sum site entirely; a
    pass under that fallback would be misleading, so we skip instead.
    """
    if os.environ.get("XRSPATIAL_GEOTIFF_STRICT_GPU") != "1":
        pytest.skip(
            "set XRSPATIAL_GEOTIFF_STRICT_GPU=1 to exercise the nvCOMP "
            "prefix-sum site; without it the GPU path may fall back to "
            "a CPU codec and bypass this regression."
        )
    try:
        import cupy
    except ImportError:
        pytest.skip("cupy not importable")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA device not available")

    import xarray as xr
    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(1950)
    height, width = 1024, 1024
    arr = rng.rand(height, width).astype(np.float32)
    da = xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(height), "x": np.arange(width)},
        attrs={"crs": 4326},
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_1950_deflate.tif")
        to_geotiff(da, path, compression="deflate", tile_size=256)

        # Read back through the GPU pipeline.
        result = open_geotiff(path, gpu=True)
        assert result.shape == (height, width)
        decoded = cupy.asnumpy(result.data) if hasattr(
            result.data, "get") else np.asarray(result.data)

    np.testing.assert_allclose(decoded, arr, atol=0, rtol=0)
