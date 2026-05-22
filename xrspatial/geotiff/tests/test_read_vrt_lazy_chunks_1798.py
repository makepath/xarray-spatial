"""read_vrt(chunks=...) should build lazy window tasks (#1798)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt, to_geotiff


def _write_vrt(vrt_path, source_name):
    vrt_path.write_text(
        '<VRTDataset rasterXSize="6" rasterYSize="4">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_read_vrt_chunks_matches_eager_values(tmp_path):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    src = tmp_path / "tmp_1798_source.tif"
    to_geotiff(arr, str(src), compression='none')
    vrt = tmp_path / "tmp_1798_source.vrt"
    _write_vrt(vrt, os.path.basename(src))

    eager = read_vrt(str(vrt))
    lazy = read_vrt(str(vrt), chunks=2)

    assert lazy.data.chunks == ((2, 2), (2, 2, 2))
    np.testing.assert_array_equal(lazy.compute().values, eager.values)


def test_read_vrt_chunks_does_not_read_sources_during_construction(tmp_path):
    """The chunked path must not eagerly decode sources at build.

    Construction does run a cheap ``os.path.exists`` sweep over each
    source (to populate ``vrt_holes`` and to fail-fast under the
    default ``missing_sources='raise'``), but it must not open or
    decode any source file. This test pairs the missing source with
    the lenient ``missing_sources='warn'`` opt-in so the build
    succeeds; the assertion is that no decode-time warnings (which
    would only fire if the source were actually read) leak out
    during construction.
    """
    vrt = tmp_path / "tmp_1798_missing_source.vrt"
    _write_vrt(vrt, "missing.tif")

    with warnings.catch_warnings(record=True) as caught:
        lazy = read_vrt(str(vrt), chunks=2, missing_sources="warn")

    # Build-time warnings from the decode codecs should be absent.
    # ``missing_sources='warn'`` does not warn at build time either; the
    # per-task ``GeoTIFFFallbackWarning`` only fires when a chunk
    # actually decodes the missing tile during ``compute()``.
    assert caught == []
    assert hasattr(lazy.data, 'compute')


def test_read_vrt_chunks_rejects_excessive_task_count(tmp_path):
    vrt = tmp_path / "tmp_1798_huge_extent.vrt"
    vrt.write_text(
        '<VRTDataset rasterXSize="100000" rasterYSize="100000">\n'
        '  <VRTRasterBand dataType="Byte" band="1"/>\n'
        '</VRTDataset>\n'
    )

    with pytest.raises(ValueError, match="task cap"):
        read_vrt(str(vrt), chunks=1, max_pixels=20_000_000_000)
