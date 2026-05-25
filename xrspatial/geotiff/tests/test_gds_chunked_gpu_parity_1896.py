"""Regression tests for issue #1896.

Pin two parity gaps in the GDS chunked GPU read path (#1813, #1895).

1. **Bool-band rejection.** ``_read_geotiff_gpu_chunked_gds`` used to
   validate ``band`` with a plain numeric range check. Because
   ``isinstance(True, int)`` is True in Python, ``band=True`` slipped
   past ``True < n_bands_out`` and silently selected band 1. The eager
   GPU path (#1786) and the dask path already rejected bools up front;
   the GDS chunked path now does too.

2. **LERC ``masked_fill`` forwarding.** ``_decode_window_gpu_direct``
   used to call ``gpu_decode_tiles_from_file`` and ``gpu_decode_tiles``
   without forwarding ``masked_fill``. On a LERC file with a per-pixel
   valid mask, the chunked path left invalid pixels at LERC's zero fill
   instead of restoring the sentinel before ``_apply_nodata_mask_gpu``
   ran. The eager GPU path resolved ``masked_fill`` once (#1529) and
   threaded it through both kernels; the chunked path now mirrors that.

These tests drive the helpers directly because ``read_geotiff_gpu(
chunks=...)`` only enters the GDS path when ``_gds_chunk_path_available``
returns True (requires KvikIO, a local tiled chunky file, no sparse
tiles, orientation == 1, photometric != 0). The mmap fallback inside
``gpu_decode_tiles_from_file`` keeps the test runnable without KvikIO
on CI.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _parse_for_gds(path: str):
    """Return ``(ifd, geo_info, header)`` for the GDS entry point."""
    from xrspatial.geotiff._geotags import extract_geo_info_with_overview_inheritance
    from xrspatial.geotiff._header import parse_all_ifds, parse_header, select_overview_ifd
    from xrspatial.geotiff._reader import _FileSource

    fs = _FileSource(path)
    try:
        raw = fs.read_all()
    finally:
        fs.close()
    header = parse_header(raw)
    ifds = parse_all_ifds(raw, header)
    ifd = select_overview_ifd(ifds, None)
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, raw, header.byte_order,
    )
    return ifd, geo_info, header


# ---------------------------------------------------------------------------
# Bool rejection on the GDS chunked path
# ---------------------------------------------------------------------------


@pytest.fixture
def multiband_tiff_1896(tmp_path):
    """4x6 three-band tiled tiff for the bool-rejection test."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'mb_gds_chunked_1896.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p)


@_gpu_only
def test_gds_chunked_band_true_rejected(multiband_tiff_1896):
    """``_read_geotiff_gpu_chunked_gds(band=True)`` raises ValueError."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=True,
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_false_rejected(multiband_tiff_1896):
    """``band=False`` is rejected the same way."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=False,
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_np_bool_rejected(multiband_tiff_1896):
    """``np.bool_`` is rejected too (not a subclass of ``bool``)."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=np.bool_(True),
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_int_still_works(multiband_tiff_1896):
    """Plain int ``band=1`` continues to select the right band."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds(multiband_tiff_1896)
    result = _read_geotiff_gpu_chunked_gds(
        multiband_tiff_1896, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=1,
        name=None, max_pixels=None,
    )
    expected = np.arange(72, dtype=np.float32).reshape(4, 6, 3)[:, :, 1]
    out = result.data.compute().get()
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# LERC masked_fill threaded through the GDS chunked path
# ---------------------------------------------------------------------------


lerc = pytest.importorskip("lerc")

from xrspatial.geotiff._compression import LERC_AVAILABLE  # noqa: E402

_lerc_gpu_only = pytest.mark.skipif(
    not (_HAS_GPU and LERC_AVAILABLE),
    reason="cupy + CUDA + lerc required",
)


@pytest.fixture
def lerc_writer_with_mask_1896(monkeypatch):
    """Inject a per-tile mask into the LERC writer (see #1529 / #1817).

    The xrspatial writer hard-codes ``hasMask=False`` in its
    ``lerc.encode`` call. Monkeypatch ``lerc_compress`` so the mask
    survives the encode and reappears at decode time -- the only way
    to exercise the GPU LERC mask path without an external mask-bearing
    fixture file.
    """
    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype('float32'), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = lerc.encode(arr, samples, has_mask, mask, max_z_error, 1)
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}")
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


@_lerc_gpu_only
def test_gds_chunked_lerc_mask_matches_eager(tmp_path, lerc_writer_with_mask_1896):
    """Chunked GDS path restores LERC-masked pixels to the nodata sentinel.

    Before #1896, ``_decode_window_gpu_direct`` dropped ``masked_fill``,
    so masked pixels read back at LERC's zero fill rather than NaN.
    The eager GPU path resolves and forwards ``masked_fill``; this
    test pins the chunked path to the same behaviour.
    """
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds
    from xrspatial.geotiff._writer import write

    arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    invalid_positions = {(0, 1), (3, 3), (7, 7)}

    def invalid_pred(a):
        m = np.zeros(a.shape[:2], dtype=bool)
        for r, c in invalid_positions:
            m[r, c] = True
        return m
    lerc_writer_with_mask_1896["invalid"] = invalid_pred

    path = str(tmp_path / "lerc_gds_chunked_1896.tif")
    write(arr, path, compression="lerc", tiled=True, tile_size=8,
          nodata=float("nan"))

    eager = read_geotiff_gpu(
        path, on_gpu_failure='strict',
        allow_experimental_codecs=True,
    ).data.get()

    ifd, geo_info, header = _parse_for_gds(path)
    chunked_da = _read_geotiff_gpu_chunked_gds(
        path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    chunked = chunked_da.data.compute().get()

    for (r, c) in invalid_positions:
        assert np.isnan(eager[r, c]), "eager path should NaN-mask invalid pixels"
        assert np.isnan(chunked[r, c]), (
            f"chunked GDS path left ({r},{c}) at LERC zero fill "
            f"{chunked[r, c]!r}; expected NaN")

    eager_valid = np.where(np.isnan(eager), 0.0, eager)
    chunked_valid = np.where(np.isnan(chunked), 0.0, chunked)
    np.testing.assert_array_equal(eager_valid, chunked_valid)


@_lerc_gpu_only
def test_gds_chunked_lerc_mask_sentinel_nodata(tmp_path,
                                               lerc_writer_with_mask_1896):
    """Sentinel nodata (-9999) on float LERC: chunked path matches eager."""
    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds
    from xrspatial.geotiff._writer import write

    arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    invalid_positions = {(0, 1), (5, 4)}

    def invalid_pred(a):
        m = np.zeros(a.shape[:2], dtype=bool)
        for r, c in invalid_positions:
            m[r, c] = True
        return m
    lerc_writer_with_mask_1896["invalid"] = invalid_pred

    path = str(tmp_path / "lerc_gds_chunked_sentinel_1896.tif")
    write(arr, path, compression="lerc", tiled=True, tile_size=8,
          nodata=-9999.0)

    eager = read_geotiff_gpu(
        path, on_gpu_failure='strict',
        allow_experimental_codecs=True,
    ).data.get()

    ifd, geo_info, header = _parse_for_gds(path)
    chunked_da = _read_geotiff_gpu_chunked_gds(
        path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    chunked = chunked_da.data.compute().get()

    for (r, c) in invalid_positions:
        assert np.isnan(eager[r, c])
        assert np.isnan(chunked[r, c])
    np.testing.assert_array_equal(
        np.where(np.isnan(eager), 0.0, eager),
        np.where(np.isnan(chunked), 0.0, chunked),
    )
