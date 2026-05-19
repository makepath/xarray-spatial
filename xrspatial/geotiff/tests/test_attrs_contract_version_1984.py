"""Contract-version marker tests for issue #1984 / #2016.

PR 3 of issue #1984 stamped every DataArray returned by an xrspatial
geotiff read path with ``attrs['_xrspatial_geotiff_contract']``. Issue
#2016 (removal phase of #1984) bumped the version to ``2`` and dropped
the 13 deprecated GeoKey-derived and matplotlib-colormap attrs.
Downstream code reads this marker to learn which attrs-contract
revision produced the array.

The stamp must appear on every backend:

* eager numpy (``open_geotiff``)
* dask + numpy (``open_geotiff(chunks=...)`` / ``read_geotiff_dask``)
* cupy / GPU (``open_geotiff(gpu=True)`` / ``read_geotiff_gpu``)
* dask + cupy (``open_geotiff(gpu=True, chunks=...)``)
* VRT eager (``read_vrt``)
* VRT dask chunked (``read_vrt(chunks=...)``)

The fixture style mirrors ``test_attrs_parity_1548.py``: build a small
on-disk TIFF (and a small VRT pointing at one) inside ``tmp_path``,
open it through each backend, and assert on the resulting attrs.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_vrt
from xrspatial.geotiff._attrs import _ATTRS_CONTRACT_VERSION

tifffile = pytest.importorskip("tifffile")


_CONTRACT_KEY = '_xrspatial_geotiff_contract'


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


def _write_small_tiff(path):
    """Write a small tiled float32 TIFF used by every read-path assertion."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        tile=(32, 32), compression='deflate', metadata=None,
    )
    return arr


def _write_minimal_vrt(vrt_path, source_name, *, height, width):
    """Write a VRT that references ``source_name`` as a single-band source."""
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_attrs_contract_version_constant_is_current():
    """Pin the integer value so a careless bump shows up here first.

    Contract v3 (issue #2136) added ``attrs['georef_status']`` to the
    canonical tier. Bumping past 3 should be paired with a docs update
    and a sibling test for the new key.
    """
    assert _ATTRS_CONTRACT_VERSION == 3


def test_eager_numpy_stamps_contract_version(tmp_path):
    path = str(tmp_path / "contract_v1_eager.tif")
    _write_small_tiff(path)

    da = open_geotiff(path)

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def test_dask_numpy_stamps_contract_version(tmp_path):
    path = str(tmp_path / "contract_v1_dask.tif")
    _write_small_tiff(path)

    da = open_geotiff(path, chunks=32)

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


@_gpu_only
def test_gpu_stamps_contract_version(tmp_path):
    path = str(tmp_path / "contract_v1_gpu.tif")
    _write_small_tiff(path)

    da = open_geotiff(path, gpu=True)

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


@_gpu_only
def test_dask_gpu_stamps_contract_version(tmp_path):
    path = str(tmp_path / "contract_v1_dask_gpu.tif")
    _write_small_tiff(path)

    da = open_geotiff(path, gpu=True, chunks=32)

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def test_vrt_eager_stamps_contract_version(tmp_path):
    src = tmp_path / "contract_v1_vrt_source.tif"
    _write_small_tiff(str(src))
    vrt = tmp_path / "contract_v1_vrt_eager.vrt"
    _write_minimal_vrt(vrt, os.path.basename(src), height=64, width=64)

    da = read_vrt(str(vrt))

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION


def test_vrt_chunked_stamps_contract_version(tmp_path):
    src = tmp_path / "contract_v1_vrt_chunked_source.tif"
    _write_small_tiff(str(src))
    vrt = tmp_path / "contract_v1_vrt_chunked.vrt"
    _write_minimal_vrt(vrt, os.path.basename(src), height=64, width=64)

    da = read_vrt(str(vrt), chunks=32)

    assert da.attrs[_CONTRACT_KEY] == _ATTRS_CONTRACT_VERSION
