"""Regression tests for issue #1786.

Every non-VRT read path range-checks ``band`` but does not reject
``bool``. Because ``isinstance(True, int)`` is True in Python and
``True < N`` evaluates as ``1 < N``, ``band=True`` silently reads
band 1 and ``band=False`` reads band 0. The VRT path
(``_vrt.read_vrt``) already rejects bools up front (#1673 follow-up)
so the API contract is inconsistent across read paths.

These tests pin every read entry point -- ``read_to_array`` (local
and HTTP), ``open_geotiff``, ``read_geotiff_dask``,
``read_geotiff_gpu`` (when cupy is available), and ``read_vrt`` --
to the same rejection so all four backends agree: ``band`` must be
a non-negative int, never a bool.
"""
from __future__ import annotations

import importlib.util
import os
import uuid

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


@pytest.fixture
def multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff for the bool-rejection tests."""
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
    p = tmp_path / 'mb_1786.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


def _write_vrt_xml(vrt_path: str, source_filename: str, size_h: int,
                   size_w: int, n_bands: int) -> None:
    bands_xml = ""
    for b in range(1, n_bands + 1):
        bands_xml += (
            f'  <VRTRasterBand dataType="Float32" band="{b}">\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{source_filename}'
            '</SourceFilename>\n'
            f'      <SourceBand>{b}</SourceBand>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="{size_w}" '
            f'ySize="{size_h}"/>\n'
            f'      <DstRect xOff="0" yOff="0" xSize="{size_w}" '
            f'ySize="{size_h}"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
        )
    xml = (
        f'<VRTDataset rasterXSize="{size_w}" rasterYSize="{size_h}">\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        f'{bands_xml}'
        '</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


@pytest.fixture
def multiband_vrt_path(tmp_path, multiband_tiff_path):
    """A 3-band VRT wrapping the same multi-band TIFF used above."""
    src_tif, _ = multiband_tiff_path
    d = tmp_path / f'vrt_1786_{uuid.uuid4().hex[:8]}'
    d.mkdir()
    # The VRT needs the source TIFF inside (or under an allowed root)
    # for path-containment (#1671). Copy bytes rather than symlink so
    # the test does not depend on the platform's symlink behaviour.
    import shutil
    local_tif = d / 'data.tif'
    shutil.copy(src_tif, local_tif)
    vrt_path = d / 'mosaic.vrt'
    _write_vrt_xml(str(vrt_path), 'data.tif', size_h=4, size_w=6,
                   n_bands=3)
    return str(vrt_path)


# ---------------------------------------------------------------------------
# read_to_array (local eager path)
# ---------------------------------------------------------------------------


def test_read_to_array_band_true_rejected(multiband_tiff_path):
    """``band=True`` no longer silently reads band 1."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_to_array(path, band=True)


def test_read_to_array_band_false_rejected(multiband_tiff_path):
    """``band=False`` no longer silently reads band 0."""
    from xrspatial.geotiff._reader import read_to_array

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_to_array(path, band=False)


def test_read_to_array_band_zero_still_works(multiband_tiff_path):
    """``band=0`` is a plain int and still selects band 0."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path
    out, _ = read_to_array(path, band=0)
    np.testing.assert_array_equal(out, arr[:, :, 0])


def test_read_to_array_band_one_still_works(multiband_tiff_path):
    """``band=1`` is a plain int and still selects band 1."""
    from xrspatial.geotiff._reader import read_to_array

    path, arr = multiband_tiff_path
    out, _ = read_to_array(path, band=1)
    np.testing.assert_array_equal(out, arr[:, :, 1])


# ---------------------------------------------------------------------------
# open_geotiff (public dispatcher)
# ---------------------------------------------------------------------------


def test_open_geotiff_band_true_rejected(multiband_tiff_path):
    """The public ``open_geotiff`` entry point rejects ``band=True``."""
    from xrspatial.geotiff import open_geotiff

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        open_geotiff(path, band=True)


def test_open_geotiff_band_false_rejected(multiband_tiff_path):
    """``open_geotiff(..., band=False)`` is rejected the same way."""
    from xrspatial.geotiff import open_geotiff

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        open_geotiff(path, band=False)


# ---------------------------------------------------------------------------
# read_geotiff_dask (dask CPU path)
# ---------------------------------------------------------------------------


def test_read_geotiff_dask_band_true_rejected(multiband_tiff_path):
    """``read_geotiff_dask(..., band=True)`` is rejected before scheduling."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_geotiff_dask(path, chunks=4, band=True)


def test_read_geotiff_dask_band_false_rejected(multiband_tiff_path):
    """``read_geotiff_dask(..., band=False)`` raises the same way."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_geotiff_dask(path, chunks=4, band=False)


# ---------------------------------------------------------------------------
# read_geotiff_gpu (GPU path)
# ---------------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_band_true_rejected(multiband_tiff_path):
    """``read_geotiff_gpu(..., band=True)`` is rejected (cupy required)."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_geotiff_gpu(path, band=True)


@_gpu_only
def test_read_geotiff_gpu_band_false_rejected(multiband_tiff_path):
    """``read_geotiff_gpu(..., band=False)`` raises the same way."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = multiband_tiff_path
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_geotiff_gpu(path, band=False)


# ---------------------------------------------------------------------------
# read_vrt (regression: was already rejecting bool; should keep doing so)
# ---------------------------------------------------------------------------


def test_read_vrt_band_true_still_rejected(multiband_vrt_path):
    """VRT path's existing bool rejection remains in place."""
    from xrspatial.geotiff import read_vrt

    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_vrt(multiband_vrt_path, band=True)


def test_read_vrt_band_false_still_rejected(multiband_vrt_path):
    """VRT path rejects ``band=False`` as well."""
    from xrspatial.geotiff import read_vrt

    with pytest.raises(ValueError, match="band must be a non-negative int"):
        read_vrt(multiband_vrt_path, band=False)
