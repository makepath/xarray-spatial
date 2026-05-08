"""Regression tests for issue #1504: overview_level must skip mask IFDs.

GDAL COG variants can interleave NewSubfileType=4 (transparency mask) IFDs
with the overview pyramid. Indexing the raw IFD list by overview_level then
returns a 1-bit mask instead of a reduced-resolution overview. The reader
should filter out mask IFDs before resolving overview_level, and raise a
clear ValueError when the requested level is out of range.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._header import (
    IFD,
    parse_all_ifds,
    parse_header,
    select_overview_ifd,
)


def _write_tiff_with_mask(path, full_res, mask, overview):
    """Write a 3-IFD TIFF: full-res, mask (subfiletype=4), overview.

    All IFDs are tiled so that the xrspatial reader exercises its tiled-COG
    path. Tiles are 16x16 to keep the test files small.
    """
    with tifffile.TiffWriter(str(path)) as tw:
        # IFD 0: full resolution (subfiletype=0 implicit).
        tw.write(full_res, tile=(16, 16), photometric='minisblack')
        # IFD 1: transparency mask (subfiletype=4).
        tw.write(
            mask,
            tile=(16, 16),
            photometric='minisblack',
            subfiletype=4,
        )
        # IFD 2: reduced-resolution overview (subfiletype=1).
        tw.write(
            overview,
            tile=(16, 16),
            photometric='minisblack',
            subfiletype=1,
        )


def _write_normal_cog(path, full_res, overviews):
    """Write a typical COG: full-res then a chain of overviews (subfiletype=1)."""
    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(full_res, tile=(16, 16), photometric='minisblack')
        for ov in overviews:
            tw.write(
                ov,
                tile=(16, 16),
                photometric='minisblack',
                subfiletype=1,
            )


# ---------------------------------------------------------------------------
# select_overview_ifd unit tests (operate on parsed IFD lists directly)
# ---------------------------------------------------------------------------

class TestSelectOverviewIFD:
    def _ifds_for(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        return parse_all_ifds(data, parse_header(data))

    def test_skips_mask_ifd(self, tmp_path):
        path = tmp_path / 'with_mask.tif'
        full = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
        mask = np.zeros((64, 64), dtype=bool)
        ov = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32))
        _write_tiff_with_mask(path, full, mask, ov)

        ifds = self._ifds_for(path)
        assert len(ifds) == 3
        # Sanity: middle IFD really is the mask.
        assert ifds[1].subfile_type & 4 == 4
        assert ifds[1].is_mask

        # Level 0 is full-res (NOT the mask).
        sel0 = select_overview_ifd(ifds, 0)
        assert sel0.width == 64 and sel0.height == 64
        assert not sel0.is_mask

        # Level 1 is the overview, jumping over the mask IFD.
        sel1 = select_overview_ifd(ifds, 1)
        assert sel1.width == 32 and sel1.height == 32
        assert not sel1.is_mask

    def test_none_returns_full_res(self, tmp_path):
        path = tmp_path / 'plain.tif'
        full = np.zeros((32, 32), dtype=np.uint8)
        _write_normal_cog(path, full, [])
        ifds = self._ifds_for(path)
        assert select_overview_ifd(ifds, None).width == 32

    def test_out_of_range_raises(self, tmp_path):
        path = tmp_path / 'with_mask.tif'
        full = np.zeros((64, 64), dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        ov = np.zeros((32, 32), dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, ov)
        ifds = self._ifds_for(path)

        with pytest.raises(ValueError) as excinfo:
            select_overview_ifd(ifds, 99)
        msg = str(excinfo.value)
        assert 'overview_level=99' in msg
        # Useful diagnostic: tells the user how many real IFDs there are.
        assert '2 non-mask IFDs' in msg
        assert 'mask' in msg.lower()

    def test_negative_raises(self, tmp_path):
        path = tmp_path / 'plain.tif'
        full = np.zeros((32, 32), dtype=np.uint8)
        _write_normal_cog(path, full, [])
        ifds = self._ifds_for(path)
        with pytest.raises(ValueError, match='must be >= 0'):
            select_overview_ifd(ifds, -1)

    def test_normal_cog_works(self, tmp_path):
        path = tmp_path / 'normal_cog.tif'
        full = np.full((128, 128), 42, dtype=np.uint16)
        ovs = [
            np.full((64, 64), 43, dtype=np.uint16),
            np.full((32, 32), 44, dtype=np.uint16),
            np.full((16, 16), 45, dtype=np.uint16),
        ]
        _write_normal_cog(path, full, ovs)
        ifds = self._ifds_for(path)
        assert len(ifds) == 4

        for level, expected_w in [(0, 128), (1, 64), (2, 32), (3, 16)]:
            sel = select_overview_ifd(ifds, level)
            assert sel.width == expected_w


# ---------------------------------------------------------------------------
# End-to-end: open_geotiff(overview_level=...) on a file with a mask IFD
# ---------------------------------------------------------------------------

class TestOpenGeotiffSkipsMask:
    def test_overview_level_1_returns_overview_not_mask(self, tmp_path):
        path = tmp_path / 'gdal_style_cog.tif'
        # Distinct fill values per IFD so the test cannot be fooled by shape.
        full = np.full((64, 64), 100, dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        overview = np.full((32, 32), 200, dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, overview)

        # Sanity: full-res still works.
        da_full = open_geotiff(str(path), overview_level=0)
        assert da_full.shape == (64, 64)
        assert int(da_full.values[0, 0]) == 100
        assert da_full.dtype == np.uint16

        # The bug: overview_level=1 used to land on the mask IFD.
        da_ov = open_geotiff(str(path), overview_level=1)
        assert da_ov.shape == (32, 32), (
            'overview_level=1 returned wrong shape; likely picked the mask IFD')
        assert int(da_ov.values[0, 0]) == 200
        assert da_ov.dtype == np.uint16

    def test_out_of_range_raises_value_error(self, tmp_path):
        path = tmp_path / 'gdal_style_cog.tif'
        full = np.zeros((64, 64), dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        overview = np.zeros((32, 32), dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, overview)

        with pytest.raises(ValueError) as excinfo:
            open_geotiff(str(path), overview_level=99)
        msg = str(excinfo.value)
        assert 'overview_level=99' in msg
        assert '2 non-mask IFDs' in msg

    def test_normal_cog_unchanged(self, tmp_path):
        path = tmp_path / 'normal_cog.tif'
        full = np.full((128, 128), 1, dtype=np.uint16)
        ovs = [
            np.full((64, 64), 2, dtype=np.uint16),
            np.full((32, 32), 3, dtype=np.uint16),
            np.full((16, 16), 4, dtype=np.uint16),
        ]
        _write_normal_cog(path, full, ovs)

        for level, expected_shape, expected_val in [
            (0, (128, 128), 1),
            (1, (64, 64), 2),
            (2, (32, 32), 3),
            (3, (16, 16), 4),
        ]:
            da = open_geotiff(str(path), overview_level=level)
            assert da.shape == expected_shape
            assert int(da.values[0, 0]) == expected_val
