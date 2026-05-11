"""Dask read of multi-band planar TIFF files.

``read_geotiff_dask`` advertises multi-band support through the ``n_bands``
branch in ``read_geotiff_dask`` -- when ``samples_per_pixel > 1`` the
returned ``DataArray`` is shaped ``(y, x, band)``. Until this module
landed, no test exercised that branch through the dask reader, so a
regression in the underlying ``read_to_array(window=...)`` call for
``PlanarConfiguration=2`` (planar=separate, the COG-friendly layout)
would ship undetected.

This module pins:

  * ``read_geotiff_dask`` returns the expected ``(y, x, band)`` shape
    and dtype for both planar=1 (contig, chunky) and planar=2
    (separate) source files.
  * The computed values match the original numpy buffer pixel-for-pixel
    after the lazy dask graph is materialised.
  * Chunks tuple (row_chunk, col_chunk) is honoured on the y/x axes
    while the band axis stays a single contiguous chunk.

Both stripped and tiled file layouts are covered because the
``_decode_strip_or_tile`` path branches on layout and the planar
handling differs in each branch.
"""
from __future__ import annotations

import numpy as np
import pytest


tifffile = pytest.importorskip("tifffile")
dask_array = pytest.importorskip("dask.array")


def _write_planar_tiff(path: str, data: np.ndarray, *,
                       planar: str, tiled: bool) -> None:
    """Write *data* shaped ``(bands, height, width)`` with chosen layout.

    tifffile expects ``(bands, h, w)`` for ``planarconfig='separate'`` and
    ``(h, w, bands)`` for ``planarconfig='contig'``. This helper centralises
    the transpose so the test bodies stay focused on the assertion.
    """
    kwargs: dict = {"photometric": "minisblack"}
    if data.shape[0] == 3:
        kwargs["photometric"] = "rgb"
    if tiled:
        kwargs["tile"] = (32, 32)
    if planar == "separate":
        kwargs["planarconfig"] = "separate"
        tifffile.imwrite(path, data, **kwargs)
    elif planar == "contig":
        kwargs["planarconfig"] = "contig"
        tifffile.imwrite(path, np.transpose(data, (1, 2, 0)), **kwargs)
    else:
        raise ValueError(f"unknown planar={planar!r}")


def _make_data(bands: int, height: int, width: int, dtype) -> np.ndarray:
    rng = np.random.RandomState(0xD45C + bands * 100 + height)
    info = np.iinfo(dtype)
    high = min(int(info.max), 60_000) + 1
    return rng.randint(0, high, size=(bands, height, width)).astype(dtype)


@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dask_planar_multiband_matches_numpy(
    tmp_path, planar, tiled, bands, dtype
):
    """``read_geotiff_dask`` returns ``(y, x, band)`` matching the source."""
    from xrspatial.geotiff import read_geotiff_dask

    height, width = 96, 128
    data = _make_data(bands, height, width, dtype)
    # On disk the file stores ``(bands, h, w)`` but the reader returns
    # the xarray convention ``(y, x, band)``.
    expected = np.transpose(data, (1, 2, 0))

    path = str(tmp_path
               / f"dask_planar_{planar}_{'tile' if tiled else 'strip'}_"
                 f"b{bands}_{np.dtype(dtype).name}.tif")
    _write_planar_tiff(path, data, planar=planar, tiled=tiled)

    da_arr = read_geotiff_dask(path, chunks=32)

    assert isinstance(da_arr.data, dask_array.Array), (
        f"expected dask Array, got {type(da_arr.data).__name__}"
    )
    assert da_arr.shape == (height, width, bands), (
        f"shape mismatch: {da_arr.shape} vs {(height, width, bands)}"
    )
    assert da_arr.dtype == np.dtype(dtype)
    assert list(da_arr.dims) == ["y", "x", "band"]

    materialised = da_arr.compute().values
    np.testing.assert_array_equal(materialised, expected)


def test_dask_planar_separate_chunks_tuple(tmp_path):
    """Tuple chunks ``(ch_h, ch_w)`` honoured; band axis stays single chunk."""
    from xrspatial.geotiff import read_geotiff_dask

    bands, height, width = 3, 80, 120
    data = _make_data(bands, height, width, np.uint8)
    expected = np.transpose(data, (1, 2, 0))

    path = str(tmp_path / "dask_planar_chunktuple.tif")
    _write_planar_tiff(path, data, planar="separate", tiled=True)

    da_arr = read_geotiff_dask(path, chunks=(40, 60))

    # ``read_geotiff_dask`` builds row-major chunks of (ch_h, ch_w, n_bands).
    # With height=80, width=120, chunks=(40, 60) the expected layout is
    # 2 row blocks x 2 col blocks x 1 band block.
    assert da_arr.data.chunksize[:2] == (40, 60)
    # The band axis is concatenated as one block (n_bands shape).
    assert da_arr.data.chunksize[2] == bands

    np.testing.assert_array_equal(da_arr.compute().values, expected)
