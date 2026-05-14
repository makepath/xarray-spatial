"""Regression test for issue #1861: ``photometric`` dropped by VRT writer.

``to_geotiff(data, '.vrt', photometric=...)`` accepted the kwarg at the
public boundary but ``_write_vrt_tiled`` did not take ``photometric`` and
``_write_single_tile`` did not forward it to ``write(...)``. Per-tile
TIFFs were always tagged with the default Photometric=MinIsBlack (1) no
matter what the caller requested.

This test pins the fix: the kwarg now threads through to every per-tile
``write`` call so the on-disk Photometric tag matches the request.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._header import TAG_PHOTOMETRIC, parse_header, parse_ifd


def _read_primary_ifd(path: str):
    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    return parse_ifd(raw, hdr.first_ifd_offset, hdr)


def _tile_paths(vrt_path: str):
    stem = os.path.splitext(os.path.basename(vrt_path))[0]
    tiles_dir = os.path.join(
        os.path.dirname(os.path.abspath(vrt_path)),
        stem + '_tiles',
    )
    return sorted(glob.glob(os.path.join(tiles_dir, 'tile_*.tif')))


def test_vrt_writer_forwards_photometric_miniswhite_1861(tmp_path):
    """photometric='miniswhite' must tag every per-tile TIFF with
    PhotometricInterpretation = 0 (MinIsWhite)."""
    arr = np.zeros((48, 48), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    vrt_path = str(tmp_path / 'miniswhite_1861.vrt')

    to_geotiff(da, vrt_path, photometric='miniswhite', tile_size=16)

    tiles = _tile_paths(vrt_path)
    assert tiles, 'expected at least one per-tile TIFF under _tiles/'
    for tile in tiles:
        ifd = _read_primary_ifd(tile)
        assert ifd.get_value(TAG_PHOTOMETRIC) == 0, (
            f'tile {tile} has Photometric '
            f'{ifd.get_value(TAG_PHOTOMETRIC)}, expected 0 (MinIsWhite)'
        )


def test_vrt_writer_default_photometric_minisblack_1861(tmp_path):
    """Control: default photometric='auto' keeps per-tile TIFFs at
    PhotometricInterpretation = 1 (MinIsBlack)."""
    arr = np.zeros((48, 48), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    vrt_path = str(tmp_path / 'default_auto_1861.vrt')

    to_geotiff(da, vrt_path, tile_size=16)

    tiles = _tile_paths(vrt_path)
    assert tiles, 'expected at least one per-tile TIFF under _tiles/'
    for tile in tiles:
        ifd = _read_primary_ifd(tile)
        assert ifd.get_value(TAG_PHOTOMETRIC) == 1, (
            f'tile {tile} has Photometric '
            f'{ifd.get_value(TAG_PHOTOMETRIC)}, expected 1 (MinIsBlack)'
        )
