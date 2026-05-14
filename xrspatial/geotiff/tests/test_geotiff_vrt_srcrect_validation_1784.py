"""VRT ``SrcRect`` must reject negative sizes and offsets up front.

The ``DstRect`` validation added for issue #1737 only covers one half of the
SimpleSource rectangle pair.  A malformed ``<SrcRect xSize="-100"/>`` (or
negative offset) reaches ``read_to_array`` as a bad window, raises
``ValueError`` for the out-of-range window, and is then swallowed by the
lenient source-read ``try/except`` that is meant to handle *missing or
unreadable source files* -- not malformed XML rectangles.

Net effect before this fix: malformed XML becomes a single warning plus a
zero-filled hole in the mosaic.  In strict mode the same condition surfaces
the swallowed error inside the try.  Either way, the caller cannot tell the
malformed-VRT case from a legitimate missing tile.

Regression test for issue #1784: ``read_vrt`` should refuse the read with a
``ValueError`` that names the offending SrcRect field, in both lenient and
strict modes.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import read_vrt


def _write_source(td: str, name: str = 'src.tif') -> str:
    """Write a 10x10 uint8 source GeoTIFF and return its path."""
    src_path = os.path.join(td, name)
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none')
    return src_path


def _write_vrt(td: str, *,
               src_x_off: int = 0, src_y_off: int = 0,
               src_x_size: int = 10, src_y_size: int = 10,
               src_filename: str = 'src.tif',
               raster_x: int = 100, raster_y: int = 100) -> str:
    """Write a VRT with a single SimpleSource using the given SrcRect."""
    vrt_path = os.path.join(td, 'mosaic.vrt')
    vrt_xml = (
        f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{src_filename}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="{src_x_off}" yOff="{src_y_off}" '
        f'xSize="{src_x_size}" ySize="{src_y_size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_negative_srcrect_x_size_rejected():
    """Negative ``SrcRect xSize`` surfaces as ``ValueError`` rather than
    being swallowed by the missing-source fallback."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_x_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            read_vrt(vrt_path)


def test_negative_srcrect_y_size_rejected():
    """Negative ``SrcRect ySize`` surfaces as ``ValueError``."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_y_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            read_vrt(vrt_path)


def test_negative_srcrect_x_off_rejected():
    """Negative ``SrcRect xOff`` surfaces as ``ValueError``."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_x_off=-10)
        with pytest.raises(ValueError, match=r"SrcRect.*negative offset"):
            read_vrt(vrt_path)


def test_negative_srcrect_y_off_rejected():
    """Negative ``SrcRect yOff`` surfaces as ``ValueError``."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_y_off=-10)
        with pytest.raises(ValueError, match=r"SrcRect.*negative offset"):
            read_vrt(vrt_path)


def test_negative_srcrect_message_names_bad_values():
    """The error message must name the malformed field and its value so the
    caller can find the offending ``<SrcRect>`` in the VRT."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_x_size=-7, src_y_size=-3)
        with pytest.raises(ValueError) as excinfo:
            read_vrt(vrt_path)
        msg = str(excinfo.value)
        assert "SrcRect" in msg
        assert "-7" in msg
        assert "-3" in msg


def test_missing_source_still_takes_lenient_warning_path():
    """A *valid* SrcRect with a missing source file must still hit the
    lenient warning path -- the new SrcRect check must not swallow the
    missing-file case that PR #1675 narrowed.

    Issue #1843 flipped the default to ``missing_sources='raise'`` so
    this test now passes ``'warn'`` explicitly to exercise the opt-in
    lenient branch.
    """
    from xrspatial.geotiff import GeoTIFFFallbackWarning
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        # No source file written; SrcRect itself is well-formed.
        vrt_path = _write_vrt(td, src_filename='does_not_exist.tif')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            arr, _ = read_vrt(vrt_path, missing_sources='warn')
        # The lenient path must produce a fallback warning and a result
        # array (zero-filled at the hole), not raise.
        fallback = [w for w in caught
                    if issubclass(w.category, GeoTIFFFallbackWarning)]
        assert fallback, (
            "expected a GeoTIFFFallbackWarning for the missing source"
        )
        assert arr.shape == (100, 100)


def test_valid_srcrect_reads_normally():
    """A well-formed SrcRect with a real source must succeed -- no false
    positives on valid VRTs."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, raster_x=10, raster_y=10)
        arr, _ = read_vrt(vrt_path)
        assert arr.shape == (10, 10)
        # Source is all zeros and DstRect covers the full VRT raster, so
        # the entire output must be zero.
        assert np.all(arr == 0)


def test_negative_srcrect_raises_under_strict_mode(monkeypatch):
    """The check runs *before* the lenient try/except, so strict mode and
    lenient mode must both raise.  Pinning strict mode here prevents a
    regression where the check accidentally moves back inside the try."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _write_source(td)
        vrt_path = _write_vrt(td, src_x_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            read_vrt(vrt_path)
