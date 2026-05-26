"""GeoTIFF reader / writer input-validation surface.

Consolidated from the per-failure-mode top-level files listed in
``CLUSTER_AUDIT_INPUT_VALIDATION.md`` (cluster 6 of long-tail epic
#2424). Every test here pins how the public read / write entry points
reject malformed inputs, before any decode or schedule work runs.

Sections, by validation axis:

1. ``band`` type / bool rejection -- ``band`` must be a non-negative
   ``int`` / ``np.integer``; ``bool`` / ``np.bool_`` raise ``ValueError``
   and ``float`` / ``str`` raise ``TypeError``, across every read entry
   point (issues #1786 and #1910).
2. Size-parameter validation -- ``tile_size`` and ``read_geotiff_dask``
   ``chunks`` must be positive, and ``tile_size`` must be a multiple of
   16 when ``tiled=True`` (issues #1752 and #1767).
3. Source-dimension validation -- zero / negative ``ImageWidth`` /
   ``ImageLength`` / ``SamplesPerPixel`` are rejected on both stripped
   and tiled read paths, local and HTTP (issue #2053).
4. 3D writer-dim validation -- ``(y, x, <non-band>)`` DataArray inputs
   are rejected rather than silently written band-last (issue #2240).
5. Window-bounds validation -- out-of-bounds ``window`` raises a clear
   ``ValueError`` on both the eager and dask read paths (issue #1634).
6. Degenerate pixel-size fail-closed -- a 1xN / Nx1 write with no
   explicit transform and no opt-in raises rather than borrowing the
   other axis's pixel size (issue #2214).
"""
from __future__ import annotations

import io
import os
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (_header, open_geotiff, read_geotiff_dask, to_geotiff,
                               write_geotiff_gpu)
from xrspatial.geotiff._coords import coords_to_transform
from xrspatial.geotiff._dtypes import LONG, SHORT
from xrspatial.geotiff._header import (MAX_PIXEL_ARRAY_COUNT, TAG_BITS_PER_SAMPLE, TAG_COLORMAP,
                                       TAG_IMAGE_LENGTH, TAG_IMAGE_WIDTH, TAG_PLANAR_CONFIG,
                                       TAG_ROWS_PER_STRIP, TAG_SAMPLES_PER_PIXEL,
                                       TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS, TAG_TILE_LENGTH,
                                       TAG_TILE_OFFSETS, TAG_TILE_WIDTH, parse_header, parse_ifd)
from xrspatial.geotiff._reader import _check_source_dimensions, read_to_array
from xrspatial.geotiff._validation import _validate_3d_writer_dims

from .._helpers.markers import requires_gpu

# ===========================================================================
# Section 1: band type / bool rejection (#1786, #1910)
#
# Every non-VRT read path range-checks ``band`` but historically did not
# reject ``bool`` (``isinstance(True, int)`` is True, so ``band=True``
# silently read band 1) or non-integer numerics (``band=0.0`` slipped the
# range check). The VRT path already used the stricter
# ``isinstance(band, (int, np.integer))`` form, so the contract differed
# across backends. ``band`` must be a non-negative int: ``bool`` /
# ``np.bool_`` raise ``ValueError`` (the #1786 guard fires first for
# back-compat), and ``float`` / ``str`` raise ``TypeError`` (#1910).
# ===========================================================================


@pytest.fixture
def multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff for the band-validation tests."""
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
    p = tmp_path / 'multiband_input_validation.tif'
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
    import shutil
    import uuid

    src_tif, _ = multiband_tiff_path
    d = tmp_path / f'vrt_input_validation_{uuid.uuid4().hex[:8]}'
    d.mkdir()
    # The VRT needs the source TIFF inside (or under an allowed root)
    # for path-containment (#1671). Copy bytes rather than symlink so
    # the test does not depend on the platform's symlink behaviour.
    local_tif = d / 'data.tif'
    shutil.copy(src_tif, local_tif)
    vrt_path = d / 'mosaic.vrt'
    _write_vrt_xml(str(vrt_path), 'data.tif', size_h=4, size_w=6, n_bands=3)
    return str(vrt_path)


class TestBandBoolRejection:
    """``band=True`` / ``band=False`` (Python and numpy bools) raise
    ``ValueError`` on every read entry point so all four backends agree."""

    def test_read_to_array_band_true_rejected(self, multiband_tiff_path):
        """``band=True`` no longer silently reads band 1."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_to_array(path, band=True)

    def test_read_to_array_band_false_rejected(self, multiband_tiff_path):
        """``band=False`` no longer silently reads band 0."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_to_array(path, band=False)

    def test_open_geotiff_band_true_rejected(self, multiband_tiff_path):
        """The public ``open_geotiff`` entry point rejects ``band=True``."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            open_geotiff(path, band=True)

    def test_open_geotiff_band_false_rejected(self, multiband_tiff_path):
        """``open_geotiff(..., band=False)`` is rejected the same way."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            open_geotiff(path, band=False)

    def test_read_geotiff_dask_band_true_rejected(self, multiband_tiff_path):
        """``read_geotiff_dask(..., band=True)`` rejected before scheduling."""
        from xrspatial.geotiff import read_geotiff_dask

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_dask(path, chunks=4, band=True)

    def test_read_geotiff_dask_band_false_rejected(self, multiband_tiff_path):
        """``read_geotiff_dask(..., band=False)`` raises the same way."""
        from xrspatial.geotiff import read_geotiff_dask

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_dask(path, chunks=4, band=False)

    @requires_gpu
    def test_read_geotiff_gpu_band_true_rejected(self, multiband_tiff_path):
        """``read_geotiff_gpu(..., band=True)`` is rejected (cupy required)."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_gpu(path, band=True)

    @requires_gpu
    def test_read_geotiff_gpu_band_false_rejected(self, multiband_tiff_path):
        """``read_geotiff_gpu(..., band=False)`` raises the same way."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_gpu(path, band=False)

    def test_read_vrt_band_true_still_rejected(self, multiband_vrt_path):
        """VRT path's existing bool rejection remains in place."""
        from xrspatial.geotiff import read_vrt

        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_vrt(multiband_vrt_path, band=True)

    def test_read_vrt_band_false_still_rejected(self, multiband_vrt_path):
        """VRT path rejects ``band=False`` as well."""
        from xrspatial.geotiff import read_vrt

        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_vrt(multiband_vrt_path, band=False)

    # np.bool_ parity: ``isinstance(np.bool_(True), bool)`` is False so it
    # bypasses a plain ``isinstance(band, bool)`` guard and is then treated
    # as 1/0 by the range check. Every read path must reject it so the four
    # backends agree.

    def test_read_to_array_band_np_bool_rejected(self, multiband_tiff_path):
        """Local file path rejects ``band=np.bool_(True)``."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_to_array(path, band=np.bool_(True))

    def test_open_geotiff_band_np_bool_rejected(self, multiband_tiff_path):
        """``open_geotiff`` rejects ``band=np.bool_(False)``."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            open_geotiff(path, band=np.bool_(False))

    def test_read_geotiff_dask_band_np_bool_rejected(self, multiband_tiff_path):
        """``read_geotiff_dask`` rejects ``band=np.bool_(True)``."""
        from xrspatial.geotiff import read_geotiff_dask

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_dask(path, band=np.bool_(True))

    @requires_gpu
    def test_read_geotiff_gpu_band_np_bool_rejected(self, multiband_tiff_path):
        """``read_geotiff_gpu`` rejects ``band=np.bool_(True)``."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_geotiff_gpu(path, band=np.bool_(True))

    def test_read_vrt_band_np_bool_still_rejected(self, multiband_vrt_path):
        """VRT path already rejects ``np.bool_`` via its integer-type check."""
        from xrspatial.geotiff import read_vrt

        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_vrt(multiband_vrt_path, band=np.bool_(True))


class TestBandTypeRejection:
    """Non-integer ``band`` values (``float`` / ``str``) raise
    ``TypeError``; valid ``int`` / ``np.integer`` values still work."""

    def test_read_to_array_band_float_rejected(self, multiband_tiff_path):
        """``band=0.0`` no longer silently reads band 0."""
        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_to_array(path, band=0.0)

    def test_read_to_array_band_np_float_rejected(self, multiband_tiff_path):
        """``band=np.float32(0)`` is rejected as well."""
        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_to_array(path, band=np.float32(0))

    def test_read_to_array_band_str_rejected(self, multiband_tiff_path):
        """Strings are rejected too."""
        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_to_array(path, band="0")

    def test_read_to_array_band_int_still_works(self, multiband_tiff_path):
        """``band=1`` is a plain int and still selects band 1."""
        path, arr = multiband_tiff_path
        out, _ = read_to_array(path, band=1)
        np.testing.assert_array_equal(out, arr[:, :, 1])

    def test_read_to_array_band_zero_still_works(self, multiband_tiff_path):
        """``band=0`` is a plain int and still selects band 0."""
        path, arr = multiband_tiff_path
        out, _ = read_to_array(path, band=0)
        np.testing.assert_array_equal(out, arr[:, :, 0])

    def test_read_to_array_band_np_integer_still_works(self, multiband_tiff_path):
        """``np.int64(1)`` is accepted because it is an ``np.integer``."""
        path, arr = multiband_tiff_path
        out, _ = read_to_array(path, band=np.int64(1))
        np.testing.assert_array_equal(out, arr[:, :, 1])

    def test_read_to_array_band_bool_still_rejected(self, multiband_tiff_path):
        """The #1786 bool guard fires first and keeps the ValueError."""
        path, _ = multiband_tiff_path
        with pytest.raises(ValueError, match="band must be a non-negative int"):
            read_to_array(path, band=True)

    def test_open_geotiff_band_float_rejected(self, multiband_tiff_path):
        """``open_geotiff(..., band=0.0)`` raises ``TypeError``."""
        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            open_geotiff(path, band=0.0)

    def test_open_geotiff_band_str_rejected(self, multiband_tiff_path):
        """``open_geotiff(..., band='0')`` raises ``TypeError``."""
        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            open_geotiff(path, band="0")

    def test_read_geotiff_dask_band_float_rejected(self, multiband_tiff_path):
        """``read_geotiff_dask(..., band=0.0)`` rejected before scheduling."""
        from xrspatial.geotiff import read_geotiff_dask

        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_geotiff_dask(path, chunks=4, band=0.0)

    def test_read_geotiff_dask_band_str_rejected(self, multiband_tiff_path):
        """``read_geotiff_dask(..., band='0')`` raises ``TypeError``."""
        from xrspatial.geotiff import read_geotiff_dask

        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_geotiff_dask(path, chunks=4, band="0")

    def test_read_geotiff_dask_band_int_still_works(self, multiband_tiff_path):
        """``band=1`` still routes through and reads band 1."""
        from xrspatial.geotiff import read_geotiff_dask

        path, arr = multiband_tiff_path
        out = read_geotiff_dask(path, chunks=4, band=1)
        np.testing.assert_array_equal(out.values, arr[:, :, 1])

    @requires_gpu
    def test_read_geotiff_gpu_band_float_rejected(self, multiband_tiff_path):
        """``read_geotiff_gpu(..., band=0.0)`` raises ``TypeError``."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_geotiff_gpu(path, band=0.0)

    @requires_gpu
    def test_read_geotiff_gpu_band_str_rejected(self, multiband_tiff_path):
        """``read_geotiff_gpu(..., band='0')`` raises ``TypeError``."""
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = multiband_tiff_path
        with pytest.raises(TypeError, match="band must be a non-negative int"):
            read_geotiff_gpu(path, band="0")


# ===========================================================================
# Section 2: size-parameter validation (#1752, #1767)
#
# Two writer/reader size parameters used to flow through unchecked:
# ``to_geotiff(..., tiled=True, tile_size=0)`` reached the tiled writer
# where ``math.ceil(width / tile_size)`` raised a bare ZeroDivisionError,
# and ``read_geotiff_dask(chunks=0)`` propagated zero into dask's chunk
# math. Both now validate up front and raise ``ValueError`` naming the
# parameter (#1752). On top of positivity, ``tile_size`` must be a
# multiple of 16 when ``tiled=True`` per the TIFF 6 spec; the error
# suggests the nearest valid value(s) (#1767). ``write_geotiff_gpu`` is
# always tiled and shares the same check before any cupy import.
# ===========================================================================


def _make_raster(tmp_path: str) -> str:
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(10), 'x': np.arange(10)},
        attrs={'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)},
    )
    path = os.path.join(tmp_path, 'raster.tif')
    to_geotiff(da, path)
    return path


def _make_da(shape=(32, 32)):
    arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return xr.DataArray(arr, dims=['y', 'x'])


class TestTileSizePositive:
    """``to_geotiff(..., tiled=True, tile_size=...)`` must be a positive
    int (#1752)."""

    def test_to_geotiff_tile_size_zero_raises(self, tmp_path):
        da = _make_da((10, 10))
        out = os.path.join(str(tmp_path), 'out.tif')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, tiled=True, tile_size=0)

    def test_to_geotiff_tile_size_negative_raises(self, tmp_path):
        da = _make_da((10, 10))
        out = os.path.join(str(tmp_path), 'out.tif')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, tiled=True, tile_size=-1)

    def test_to_geotiff_tile_size_non_int_raises(self, tmp_path):
        da = _make_da((10, 10))
        out = os.path.join(str(tmp_path), 'out.tif')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, tiled=True, tile_size=256.0)

    def test_to_geotiff_tile_size_16_writes(self, tmp_path):
        # ``tile_size=16`` is the smallest TIFF-spec-legal tile size.
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        da = xr.DataArray(arr, dims=['y', 'x'])
        out = os.path.join(str(tmp_path), 'out.tif')
        to_geotiff(da, out, tiled=True, tile_size=16)
        assert os.path.exists(out)


class TestTileSizeMultipleOf16:
    """``tile_size`` must be a multiple of 16 when ``tiled=True`` (#1767)."""

    def test_tile_size_17_rejected(self, tmp_path):
        """``tile_size=17`` is not a multiple of 16 and must be rejected."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'tile_size_17.tif')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, out, tiled=True, tile_size=17)
        msg = str(exc.value)
        assert 'tile_size' in msg
        assert '17' in msg
        # Hint should suggest nearest valid choices (16 and 32).
        assert '16' in msg and '32' in msg

    def test_tile_size_1_rejected(self, tmp_path):
        """``tile_size=1`` is rejected because 1 is not a multiple of 16."""
        da = _make_da((16, 16))
        out = os.path.join(str(tmp_path), 'tile_size_1.tif')
        with pytest.raises(ValueError, match=r'tile_size.*multiple of 16'):
            to_geotiff(da, out, tiled=True, tile_size=1)

    def test_tile_size_default_256_works(self, tmp_path):
        """The default ``tile_size=256`` is a multiple of 16 and must work."""
        da = _make_da((256, 256))
        out = os.path.join(str(tmp_path), 'tile_size_256.tif')
        to_geotiff(da, out, tiled=True, tile_size=256)
        assert os.path.exists(out)

    def test_tile_size_512_works(self, tmp_path):
        da = _make_da((512, 512))
        out = os.path.join(str(tmp_path), 'tile_size_512.tif')
        to_geotiff(da, out, tiled=True, tile_size=512)
        assert os.path.exists(out)

    def test_tile_size_128_works(self, tmp_path):
        da = _make_da((128, 128))
        out = os.path.join(str(tmp_path), 'tile_size_128.tif')
        to_geotiff(da, out, tiled=True, tile_size=128)
        assert os.path.exists(out)

    def test_tile_size_16_works(self, tmp_path):
        """The smallest legal tile size is 16."""
        da = _make_da((32, 32))
        out = os.path.join(str(tmp_path), 'tile_size_16.tif')
        to_geotiff(da, out, tiled=True, tile_size=16)
        assert os.path.exists(out)

    def test_tile_size_17_with_tiled_false_passes(self, tmp_path):
        """``tiled=False`` ignores ``tile_size``; the multiple-of-16
        validation must not fire there."""
        import warnings

        da = _make_da()
        out = os.path.join(str(tmp_path), 'tile_size_17_strip.tif')
        # ``tiled=False`` emits a warning when a non-default tile_size is
        # passed; we only care that no ValueError fires.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            to_geotiff(da, out, tiled=False, tile_size=17)
        assert os.path.exists(out)

    def test_tile_size_24_message_suggests_16_and_32(self, tmp_path):
        """Error message names both nearest valid multiples (lower & upper)."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'tile_size_24.tif')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, out, tiled=True, tile_size=24)
        msg = str(exc.value)
        assert '16' in msg
        assert '32' in msg

    def test_tile_size_8_message_suggests_16_only(self, tmp_path):
        """For ``tile_size < 16`` only the upper neighbour (16) is valid."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'tile_size_8.tif')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, out, tiled=True, tile_size=8)
        msg = str(exc.value)
        assert '16' in msg
        # 0 is not a valid tile size and should not appear as a suggestion.
        assert 'tile_size=0' not in msg

    def test_write_geotiff_gpu_tile_size_17_rejected(self, tmp_path):
        """``write_geotiff_gpu`` shares the multiple-of-16 check with
        ``to_geotiff``. The validation runs before any cupy import, so the
        bad-tile-size path can be exercised on CPU-only runs."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'gpu_tile_size_17.tif')
        with pytest.raises(ValueError) as exc:
            write_geotiff_gpu(da, out, tile_size=17)
        msg = str(exc.value)
        assert 'tile_size' in msg
        assert '17' in msg
        # Hint should suggest nearest valid choices (16 and 32).
        assert '16' in msg and '32' in msg

    def test_write_geotiff_gpu_tile_size_zero_rejected(self, tmp_path):
        """``tile_size=0`` is rejected as non-positive before the
        multiple-of-16 branch fires."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'gpu_tile_size_0.tif')
        with pytest.raises(ValueError, match=r'tile_size.*positive'):
            write_geotiff_gpu(da, out, tile_size=0)

    def test_write_geotiff_gpu_tile_size_float_rejected(self, tmp_path):
        """``tile_size`` must be an int; floats are rejected by the shared
        helper before any GPU machinery is touched."""
        da = _make_da()
        out = os.path.join(str(tmp_path), 'gpu_tile_size_float.tif')
        with pytest.raises(ValueError, match=r'tile_size.*positive int'):
            write_geotiff_gpu(da, out, tile_size=256.0)


class TestReadDaskChunksValidation:
    """``read_geotiff_dask(chunks=...)`` must be a positive int or a
    length-2 tuple of positive ints (#1752)."""

    def test_chunks_zero_raises(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_dask(path, chunks=0)

    def test_chunks_negative_raises(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_dask(path, chunks=-1)

    def test_chunks_tuple_zero_row_raises(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_dask(path, chunks=(0, 256))

    def test_chunks_tuple_negative_col_raises(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_dask(path, chunks=(256, -1))

    def test_chunks_tuple_wrong_length_raises(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='chunks'):
            read_geotiff_dask(path, chunks=(64, 64, 64))

    def test_positive_int_chunks_works(self, tmp_path):
        path = _make_raster(str(tmp_path))
        arr = read_geotiff_dask(path, chunks=256)
        assert arr.shape == (10, 10)
        # Materialise to confirm the lazy graph is well-formed.
        np.asarray(arr)

    def test_positive_tuple_chunks_works(self, tmp_path):
        path = _make_raster(str(tmp_path))
        arr = read_geotiff_dask(path, chunks=(4, 8))
        assert arr.shape == (10, 10)
        np.asarray(arr)

    def test_numpy_int_scalar_chunks_works(self, tmp_path):
        # Numpy integer scalars (e.g. np.int64) should behave like plain
        # ``int`` for the scalar ``chunks`` form.
        path = _make_raster(str(tmp_path))
        arr = read_geotiff_dask(path, chunks=np.int64(256))
        assert arr.shape == (10, 10)
        np.asarray(arr)

    def test_numpy_int_tuple_chunks_works(self, tmp_path):
        path = _make_raster(str(tmp_path))
        arr = read_geotiff_dask(path, chunks=(np.int64(256), 256))
        assert arr.shape == (10, 10)
        np.asarray(arr)


# ===========================================================================
# Section 3: source-dimension validation (#2053, #1901)
#
# Two related defenses against malformed IFD geometry:
#
# * ``_check_source_dimensions`` rejects ``<= 0`` ImageWidth /
#   ImageLength / SamplesPerPixel before any window clamp, on both
#   stripped read paths (local ``_read_strips`` and HTTP
#   ``_fetch_decode_cog_http_strips``). Tiled paths already validate via
#   ``validate_tile_layout``; the tests pin that parity (#2053).
# * The pixel-array tags (Strip/Tile Offsets & ByteCounts, ColorMap) are
#   exempt from the generic ``MAX_IFD_ENTRY_COUNT`` cap, so their
#   ``count`` is instead bounded against the IFD geometry (and an
#   absolute ``MAX_PIXEL_ARRAY_COUNT`` when geometry is missing) to stop
#   a crafted ``count`` from driving a multi-GiB tuple allocation (#1901).
# ===========================================================================


def _find_ifd_entry_offset(buf: bytes, tag_id: int) -> int:
    """Return the byte offset of the IFD entry for ``tag_id``.

    Classic TIFF only. The IFD entry layout is 12 bytes:
    ``tag(2) + type(2) + count(4) + value/offset(4)``.
    """
    header = parse_header(buf)
    assert not header.is_bigtiff, "helper only handles classic TIFF"
    bo = header.byte_order
    ifd_off = header.first_ifd_offset
    num_entries = struct.unpack_from(f'{bo}H', buf, ifd_off)[0]
    entry_base = ifd_off + 2
    for i in range(num_entries):
        entry_off = entry_base + i * 12
        tag = struct.unpack_from(f'{bo}H', buf, entry_off)[0]
        if tag == tag_id:
            return entry_off
    raise KeyError(f"Tag {tag_id} not found in IFD")


def _patch_inline_long(buf: bytearray, tag_id: int, new_value: int) -> None:
    """Patch the inline LONG / SHORT value of an IFD entry to ``new_value``."""
    header = parse_header(bytes(buf))
    bo = header.byte_order
    entry_off = _find_ifd_entry_offset(bytes(buf), tag_id)
    type_id = struct.unpack_from(f'{bo}H', buf, entry_off + 2)[0]
    count = struct.unpack_from(f'{bo}I', buf, entry_off + 4)[0]
    assert count == 1, (
        f"helper only supports count=1 entries; got count={count} "
        f"for tag {tag_id}"
    )
    value_off = entry_off + 8
    if type_id == 4:  # LONG
        struct.pack_into(f'{bo}I', buf, value_off, new_value & 0xFFFFFFFF)
    elif type_id == 3:  # SHORT (2 bytes; upper 2 bytes of slot are padding)
        struct.pack_into(f'{bo}H', buf, value_off, new_value & 0xFFFF)
    else:
        raise AssertionError(
            f"unsupported type_id={type_id} for tag {tag_id}; helper handles "
            f"LONG and SHORT only"
        )


def _make_valid_stripped(tmp_path, *, height=16, width=8):
    """Write a small valid stripped TIFF and return its bytes + path."""
    arr = xr.DataArray(
        np.arange(height * width, dtype=np.uint8).reshape(height, width),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'valid_stripped.tif')
    to_geotiff(arr, path, compression='none', tiled=False)
    with open(path, 'rb') as f:
        return bytearray(f.read()), path


def _make_valid_tiled(tmp_path, *, height=32, width=32, tile_size=16):
    """Write a small valid tiled TIFF and return its bytes + path."""
    arr = xr.DataArray(
        np.arange(height * width, dtype=np.uint8).reshape(height, width),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'valid_tiled.tif')
    to_geotiff(arr, path, compression='none', tiled=True, tile_size=tile_size)
    with open(path, 'rb') as f:
        return bytearray(f.read()), path


class TestCheckSourceDimensions:
    """The validator must reject every flavor of non-positive input."""

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(0, 16, 1)

    def test_zero_height_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 0, 1)

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 16, 0)

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(-1, 16, 1)

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, -1, 1)

    def test_negative_samples_rejected(self):
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            _check_source_dimensions(16, 16, -1)

    def test_all_positive_passes(self):
        # No exception => pass
        _check_source_dimensions(1, 1, 1)
        _check_source_dimensions(1024, 1024, 3)

    def test_error_message_contains_each_value(self):
        with pytest.raises(ValueError) as excinfo:
            _check_source_dimensions(0, 5, 7)
        msg = str(excinfo.value)
        assert "ImageWidth=0" in msg
        assert "ImageLength=5" in msg
        assert "SamplesPerPixel=7" in msg


class TestStrippedZeroDimsRejected:
    """End-to-end: malformed stripped TIFFs are rejected by open_geotiff."""

    def test_zero_image_width_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_path = tmp_path / 'zero_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_zero_image_length_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_path = tmp_path / 'zero_height.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_zero_samples_per_pixel_rejected(self, tmp_path):
        buf, _ = _make_valid_stripped(tmp_path)
        # SamplesPerPixel is written as SHORT (type=3) by the writer.
        _patch_inline_long(buf, TAG_SAMPLES_PER_PIXEL, 0)
        bad_path = tmp_path / 'zero_samples.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            open_geotiff(str(bad_path))

    def test_negative_width_via_signed_cast_rejected(self, tmp_path):
        """A 32-bit pattern that looks like a negative signed int.

        Real-world TIFFs store ImageWidth as an unsigned LONG, so a
        "negative" value surfaces as a huge unsigned int. Either the
        strict ``<= 0`` check rejects it directly, or the upper-bound
        ``_check_dimensions`` rejects it as oversized. Either error is
        acceptable; the test pins that the file does not silently produce
        an empty array.
        """
        buf, _ = _make_valid_stripped(tmp_path)
        # 0xFFFFFFFF = -1 as int32, ~4.29B as uint32. Larger than
        # MAX_PIXELS_DEFAULT so the upper-bound check fires regardless.
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0xFFFFFFFF)
        bad_path = tmp_path / 'huge_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError):
            open_geotiff(str(bad_path))


class TestWindowedEmptyStillAllowed:
    """The new check sits *before* window clamping. A caller passing a
    window entirely outside the image still receives an empty result; the
    strict check only applies to source IFD dims.
    """

    def test_windowed_outside_image_returns_empty_not_error(self, tmp_path):
        buf, path = _make_valid_stripped(tmp_path, height=16, width=8)
        from xrspatial.geotiff._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
        from xrspatial.geotiff._header import parse_all_ifds
        from xrspatial.geotiff._reader import _read_strips

        data = bytes(buf)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)

        # Window starting at the image's bottom-right corner. After
        # clamping, the post-window dims along y are (0,).
        edge_window = (ifd.height, 0, ifd.height, ifd.width)
        arr = _read_strips(data, ifd, header, dtype, window=edge_window)
        # r0 = 16 (clamped), r1 = 16 -> out_h = 0; c spans 0..8 -> out_w = 8.
        assert arr.shape[0] == 0, (
            f"expected zero-height array from edge window, got shape "
            f"{arr.shape}"
        )


class TestTiledZeroDimsParityPinned:
    """``validate_tile_layout`` already rejects zero w/h on tiled files.
    This pins that behavior so any refactor of the tiled validator that
    drops the check would surface here, not in production.
    """

    def test_tiled_zero_width_rejected(self, tmp_path):
        buf, _ = _make_valid_tiled(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_path = tmp_path / 'tiled_zero_width.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid"):
            open_geotiff(str(bad_path))

    def test_tiled_zero_height_rejected(self, tmp_path):
        buf, _ = _make_valid_tiled(tmp_path)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_path = tmp_path / 'tiled_zero_height.tif'
        bad_path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="Invalid"):
            open_geotiff(str(bad_path))


class _StaticBytesHTTPSource:
    """Minimal ``_HTTPSource`` stand-in backed by a static buffer."""

    def __init__(self, buf: bytes):
        self._buf = buf
        self.read_all_called = False

    def read_range(self, start: int, length: int) -> bytes:
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        self.read_all_called = True
        return self._buf

    def read_ranges_coalesced(self, ranges, *, max_workers=8,
                              gap_threshold=0,
                              max_coalesced_range_bytes=None):
        return [self._buf[s:s + le] for (s, le) in ranges]

    def close(self):
        pass


class TestHTTPStrippedZeroDimsRejected:
    """A malformed stripped COG over HTTP must also reject."""

    def test_zero_image_width_over_http_rejected(self, tmp_path, monkeypatch):
        buf, _ = _make_valid_stripped(tmp_path, height=64, width=32)
        _patch_inline_long(buf, TAG_IMAGE_WIDTH, 0)
        bad_bytes = bytes(buf)

        from xrspatial.geotiff import _reader as reader_mod
        monkeypatch.setattr(
            reader_mod, '_HTTPSource',
            lambda url, **kw: _StaticBytesHTTPSource(bad_bytes))

        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            reader_mod._read_cog_http('http://mock/bad.tif')

    def test_zero_image_length_over_http_rejected(self, tmp_path, monkeypatch):
        buf, _ = _make_valid_stripped(tmp_path, height=64, width=32)
        _patch_inline_long(buf, TAG_IMAGE_LENGTH, 0)
        bad_bytes = bytes(buf)

        from xrspatial.geotiff import _reader as reader_mod
        monkeypatch.setattr(
            reader_mod, '_HTTPSource',
            lambda url, **kw: _StaticBytesHTTPSource(bad_bytes))

        with pytest.raises(ValueError, match="Invalid TIFF dimensions"):
            reader_mod._read_cog_http('http://mock/bad.tif')


def _short_bytes(v: int) -> bytes:
    return struct.pack('<H', v) + b'\x00\x00'


def _long_bytes(v: int) -> bytes:
    return struct.pack('<I', v)


def _build_classic_tiff(
    entries: list[tuple[int, int, int, bytes]],
    tail_padding: int = 0,
    external_payloads: list[tuple[int, bytes]] | None = None,
) -> bytes:
    bo = '<'
    n = len(entries)
    ifd_offset = 8
    ifd_size = 2 + n * 12 + 4
    end_of_ifd = ifd_offset + ifd_size
    file_size = end_of_ifd + tail_padding
    if external_payloads:
        for off, payload in external_payloads:
            file_size = max(file_size, off + len(payload))

    buf = bytearray(file_size)
    buf[0:2] = b'II'
    struct.pack_into(f'{bo}H', buf, 2, 42)
    struct.pack_into(f'{bo}I', buf, 4, ifd_offset)
    struct.pack_into(f'{bo}H', buf, ifd_offset, n)
    for i, (tag, type_id, count, value_bytes) in enumerate(entries):
        eo = ifd_offset + 2 + i * 12
        struct.pack_into(f'{bo}H', buf, eo, tag)
        struct.pack_into(f'{bo}H', buf, eo + 2, type_id)
        struct.pack_into(f'{bo}I', buf, eo + 4, count)
        assert len(value_bytes) == 4
        buf[eo + 8:eo + 12] = value_bytes
    struct.pack_into(f'{bo}I', buf, ifd_offset + 2 + n * 12, 0)
    if external_payloads:
        for off, payload in external_payloads:
            buf[off:off + len(payload)] = payload
    return bytes(buf)


class TestPixelArrayCountCap:
    """Pixel-array tag ``count`` is bounded against IFD geometry (and an
    absolute cap when geometry is missing) (#1901)."""

    def test_tile_offsets_count_exceeds_geometry_rejected(self):
        """TileOffsets ``count`` larger than tiles_across * tiles_down raises.

        1024x1024 image, 256x256 tiles -> 16 tiles. count=100 must raise.
        """
        payload_offset = 8 + 2 + 12 * 5 + 4
        bad_count = 100
        payload = b'\x00' * (bad_count * 4)
        entries = [
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
            (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
            (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 16"):
            parse_ifd(data, header.first_ifd_offset, header)

    def test_tile_offsets_count_matching_geometry_passes(self):
        """16 tiles in a 1024x1024 image with 256x256 tiles must parse."""
        payload_offset = 8 + 2 + 12 * 5 + 4
        good_count = 16
        payload = b'\x00' * (good_count * 4)
        entries = [
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
            (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
            (TAG_TILE_OFFSETS, LONG, good_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)
        assert ifd.entries[TAG_TILE_OFFSETS].count == good_count

    def test_strip_offsets_count_exceeds_geometry_rejected(self):
        """StripOffsets count larger than ceil(height / rows_per_strip) raises.

        256x256 with RowsPerStrip=64 -> 4 strips. count=200 must raise.
        """
        payload_offset = 8 + 2 + 12 * 4 + 4
        bad_count = 200
        payload = b'\x00' * (bad_count * 4)
        entries = [
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
            (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
            (TAG_STRIP_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 4"):
            parse_ifd(data, header.first_ifd_offset, header)

    def test_strip_byte_counts_planar_multiplies_by_samples(self):
        """PlanarConfig=2 multiplies expected strip count by samples_per_pixel.

        256x256 with RowsPerStrip=64 and 3 samples planar -> 12 entries.
        count=12 passes; count=13 raises.
        """
        payload_offset = 8 + 2 + 12 * 6 + 4
        payload = b'\x00' * (12 * 4)
        base_entries = [
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
            (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, _short_bytes(3)),
            (TAG_PLANAR_CONFIG, SHORT, 1, _short_bytes(2)),
        ]
        good = base_entries + [
            (TAG_STRIP_BYTE_COUNTS, LONG, 12, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            good, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)
        assert ifd.entries[TAG_STRIP_BYTE_COUNTS].count == 12

        bad = base_entries + [
            (TAG_STRIP_BYTE_COUNTS, LONG, 13, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            bad, external_payloads=[(payload_offset, b'\x00' * (13 * 4))],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 12"):
            parse_ifd(data, header.first_ifd_offset, header)

    def test_colormap_count_exceeds_bits_per_sample_rejected(self):
        """ColorMap count > 3 * 2^bits_per_sample raises.

        BitsPerSample=8 -> expected 3 * 256 = 768. count=2000 must raise.
        """
        payload_offset = 8 + 2 + 12 * 2 + 4
        bad_count = 2000
        payload = b'\x00' * (bad_count * 2)
        entries = [
            (TAG_BITS_PER_SAMPLE, SHORT, 1, _short_bytes(8)),
            (TAG_COLORMAP, SHORT, bad_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 768"):
            parse_ifd(data, header.first_ifd_offset, header)

    def test_colormap_count_at_expected_passes(self):
        """ColorMap with the exact expected count for BPS=8 must parse."""
        payload_offset = 8 + 2 + 12 * 2 + 4
        good_count = 3 * 256
        payload = b'\x00' * (good_count * 2)
        entries = [
            (TAG_BITS_PER_SAMPLE, SHORT, 1, _short_bytes(8)),
            (TAG_COLORMAP, SHORT, good_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)
        assert ifd.entries[TAG_COLORMAP].count == good_count

    def test_absolute_cap_fires_when_dimensions_missing(self):
        """With no geometry tags in the IFD, MAX_PIXEL_ARRAY_COUNT alone caps.

        Monkeypatched down to keep the test cheap.
        """
        cap = 100
        orig = _header.MAX_PIXEL_ARRAY_COUNT
        _header.MAX_PIXEL_ARRAY_COUNT = cap
        try:
            bad_count = cap + 1
            entries = [
                (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(0)),
            ]
            data = _build_classic_tiff(entries, tail_padding=512)
            header = parse_header(data)
            with pytest.raises(
                ValueError, match=r"exceeds MAX_PIXEL_ARRAY_COUNT=100"
            ):
                parse_ifd(data, header.first_ifd_offset, header)
        finally:
            _header.MAX_PIXEL_ARRAY_COUNT = orig

    def test_absolute_cap_constant_is_reasonable(self):
        """Sanity check: 100M elements is enough for any realistic image but
        far below the count required to drive a multi-GiB allocation."""
        # 1M x 1M image at 256-pixel tiles is ~16M tiles.
        assert MAX_PIXEL_ARRAY_COUNT >= 16_000_000
        # 100M PyLongs is roughly 3 GiB; refuse to allocate more than that.
        assert MAX_PIXEL_ARRAY_COUNT <= 1_000_000_000

    def test_dimensions_listed_after_pixel_array_tag_still_validate(self):
        """Pre-scan must collect dimensions even when the pixel-array tag
        appears earlier in tag-numeric order than they do.

        A malicious file could reorder entries; the parser pre-scan walks
        the whole entry table before validating counts.
        """
        payload_offset = 8 + 2 + 12 * 5 + 4
        bad_count = 100
        payload = b'\x00' * (bad_count * 4)
        # Same 1024x1024, 256x256 case (16 tiles), but TileOffsets first.
        entries = [
            (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
            (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 16"):
            parse_ifd(data, header.first_ifd_offset, header)

    def test_strip_byte_counts_chunky_uses_image_length_only(self):
        """PlanarConfig=1 (chunky) does NOT multiply expected strip count.

        256x256 with RowsPerStrip=64 and 3 samples chunky -> 4 entries.
        """
        payload_offset = 8 + 2 + 12 * 6 + 4
        good_count = 4
        payload = b'\x00' * (good_count * 4)
        entries = [
            (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
            (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
            (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, _short_bytes(3)),
            (TAG_PLANAR_CONFIG, SHORT, 1, _short_bytes(1)),
            (TAG_STRIP_OFFSETS, LONG, good_count, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            entries, external_payloads=[(payload_offset, payload)],
        )
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)
        assert ifd.entries[TAG_STRIP_OFFSETS].count == good_count

        # And chunky with count=5 raises.
        bad = entries[:-1] + [
            (TAG_STRIP_OFFSETS, LONG, 5, _long_bytes(payload_offset)),
        ]
        data = _build_classic_tiff(
            bad, external_payloads=[(payload_offset, b'\x00' * (5 * 4))],
        )
        header = parse_header(data)
        with pytest.raises(ValueError, match="exceeds expected value 4"):
            parse_ifd(data, header.first_ifd_offset, header)


# ===========================================================================
# Section 4: 3D writer-dim validation (#2240)
#
# ``_validate_3d_writer_dims`` used to accept any ``(y_alias, x_alias, *)``
# DataArray dim tuple whose trailing dim was not a recognized temporal
# name, so ``('y', 'x', 'z')`` / ``('lat', 'lon', 'scenario')`` slipped
# through and were silently written band-last. #2240 closes that escape
# hatch for DataArray inputs; raw-ndarray band-last writes (which never
# reach the validator) are unaffected.
# ===========================================================================


class TestValidate3DWriterDims:
    """Validator-level coverage for ``_validate_3d_writer_dims``."""

    @pytest.mark.parametrize(
        "trailing",
        ['z', 'level', 'scenario', 'depth', 'member', 'realization',
         'foo', 'bar', 'baz'],
    )
    def test_rejects_yx_non_band_trailing(self, trailing):
        """``(y, x, <non-band, non-temporal>)`` raises with a clear message."""
        with pytest.raises(ValueError, match="non-band trailing dim"):
            _validate_3d_writer_dims(('y', 'x', trailing))

    @pytest.mark.parametrize(
        "yx",
        [('y', 'x'), ('lat', 'lon'), ('latitude', 'longitude'),
         ('row', 'col')],
    )
    @pytest.mark.parametrize(
        "trailing",
        ['z', 'level', 'scenario'],
    )
    def test_rejects_yx_aliases_with_non_band_trailing(self, yx, trailing):
        """Non-band trailing dim is rejected for every recognized y/x alias."""
        with pytest.raises(ValueError, match="non-band trailing dim"):
            _validate_3d_writer_dims((yx[0], yx[1], trailing))

    def test_still_accepts_band_alias_trailing(self):
        """Recognized band aliases at the trailing position still succeed."""
        _validate_3d_writer_dims(('y', 'x', 'band'))
        _validate_3d_writer_dims(('y', 'x', 'bands'))
        _validate_3d_writer_dims(('y', 'x', 'channel'))

    def test_still_accepts_band_alias_leading(self):
        """``(band, y, x)`` and its aliases still succeed."""
        _validate_3d_writer_dims(('band', 'y', 'x'))
        _validate_3d_writer_dims(('bands', 'y', 'x'))
        _validate_3d_writer_dims(('channel', 'y', 'x'))

    def test_still_routes_temporal_to_temporal_message(self):
        """Temporal trailing dims still take the dedicated temporal error path.

        The #1972 message gives more specific remediation (``isel`` /
        ``mean`` along the time axis) than the #2240 generic non-band
        message, so the temporal-name branch must fire first.
        """
        with pytest.raises(ValueError, match="temporal trailing dim"):
            _validate_3d_writer_dims(('y', 'x', 'time'))
        with pytest.raises(ValueError, match="temporal trailing dim"):
            _validate_3d_writer_dims(('lat', 'lon', 'date'))

    def test_still_rejects_other_ambiguous_leading(self):
        """Generic ambiguous-dim message still fires for non-y/x leading dims."""
        with pytest.raises(ValueError, match="ambiguous dims"):
            _validate_3d_writer_dims(('foo', 'y', 'x'))
        with pytest.raises(ValueError, match="ambiguous dims"):
            _validate_3d_writer_dims(('scenario', 'y', 'x'))

    def test_2d_dims_unchanged(self):
        """2D dim tuples are still pass-through (validator only runs on 3D)."""
        _validate_3d_writer_dims(('y', 'x'))
        _validate_3d_writer_dims(('lat', 'lon'))


class TestValidate3DWriterEndToEnd:
    """End-to-end writer coverage for the #2240 tightening."""

    def test_to_geotiff_rejects_yxz_dataarray(self):
        """``(y, x, z)`` DataArray writes are rejected."""
        da = xr.DataArray(
            np.zeros((4, 4, 3), dtype=np.float32),
            coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                    'z': np.arange(3)},
            dims=('y', 'x', 'z'),
        )
        buf = io.BytesIO()
        with pytest.raises(ValueError, match="non-band trailing dim"):
            to_geotiff(da, buf)

    def test_to_geotiff_rejects_lat_lon_scenario_dataarray(self):
        """``(lat, lon, scenario)`` is rejected on the writer entry."""
        da = xr.DataArray(
            np.zeros((4, 4, 3), dtype=np.float32),
            coords={'lat': np.arange(4.0), 'lon': np.arange(4.0),
                    'scenario': np.arange(3)},
            dims=('lat', 'lon', 'scenario'),
        )
        buf = io.BytesIO()
        with pytest.raises(ValueError, match="non-band trailing dim"):
            to_geotiff(da, buf)

    def test_error_message_is_actionable(self):
        """The error names the offending dim and points at fixes."""
        da = xr.DataArray(
            np.zeros((4, 4, 3), dtype=np.float32),
            coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                    'scenario': np.arange(3)},
            dims=('y', 'x', 'scenario'),
        )
        buf = io.BytesIO()
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, buf)
        msg = str(excinfo.value)
        # Names the offending dim
        assert "'scenario'" in msg
        # Mentions accepted band aliases
        assert "band" in msg
        # Points at concrete remediations
        assert "isel(scenario=0)" in msg or "isel" in msg
        assert "raw ndarray" in msg.lower() or "ndarray" in msg.lower()
        # References the issue
        assert "#2240" in msg

    def test_to_geotiff_still_accepts_yx_band_dataarray(self, tmp_path):
        """``(y, x, band)`` DataArrays still round-trip cleanly."""
        arr = np.empty((4, 5, 3), dtype=np.uint8)
        for k in range(3):
            arr[:, :, k] = k + 1
        da = xr.DataArray(arr, dims=('y', 'x', 'band'),
                          attrs={'crs': 'EPSG:4326'})
        out = tmp_path / 'yx_band.tif'
        to_geotiff(da, str(out), crs=4326)
        rt = open_geotiff(str(out))
        assert rt.shape == (4, 5, 3)
        for k in range(3):
            assert int(rt.values[:, :, k].sum()) == (k + 1) * 20

    def test_to_geotiff_still_accepts_band_yx_dataarray(self, tmp_path):
        """``(band, y, x)`` DataArrays still round-trip cleanly."""
        arr = np.empty((3, 4, 5), dtype=np.uint8)
        for k in range(3):
            arr[k] = k + 1
        da = xr.DataArray(arr, dims=('band', 'y', 'x'),
                          attrs={'crs': 'EPSG:4326'})
        out = tmp_path / 'band_yx.tif'
        to_geotiff(da, str(out), crs=4326)
        rt = open_geotiff(str(out))
        assert rt.shape == (4, 5, 3)
        for k in range(3):
            assert int(rt.values[:, :, k].sum()) == (k + 1) * 20

    def test_raw_ndarray_band_last_still_writes(self, tmp_path):
        """Raw ndarray inputs with band-last layout are unaffected by #2240.

        The validator is only invoked from the ``isinstance(data,
        xr.DataArray)`` branch of every writer entry point, so a bare
        numpy array never goes through the dim check.
        """
        arr = np.empty((4, 5, 3), dtype=np.uint8)
        for k in range(3):
            arr[:, :, k] = k + 1
        out = tmp_path / 'raw_ndarray_band_last.tif'
        to_geotiff(arr, str(out), crs=4326)
        rt = open_geotiff(str(out))
        assert rt.shape == (4, 5, 3)
        for k in range(3):
            assert int(rt.values[:, :, k].sum()) == (k + 1) * 20

    def test_raw_ndarray_unusual_third_axis_still_writes(self, tmp_path):
        """Raw ndarray with no dim metadata is band-last by definition.

        Passing a bare ndarray bypasses the DataArray dim contract
        entirely; the writer treats the trailing axis as bands. The #2240
        tightening only constrains DataArray inputs.
        """
        arr = np.empty((4, 5, 3), dtype=np.float32)
        for k in range(3):
            arr[:, :, k] = float(k + 1)
        out = tmp_path / 'raw_ndarray_band_last_floats.tif'
        to_geotiff(arr, str(out), crs=4326)
        rt = open_geotiff(str(out))
        assert rt.shape == (4, 5, 3)
        for k in range(3):
            assert float(rt.values[:, :, k].sum()) == float(k + 1) * 20


# ===========================================================================
# Section 5: window-bounds validation (#1634)
#
# ``open_geotiff(path, window=...)`` on the eager (numpy) path used to
# produce a confusing ``CoordinateValidationError`` when the window ran
# past the source extent: ``read_to_array`` clamped the window and
# returned a smaller array, but the eager path built coord arrays from
# the unclamped indices. The eager branch now validates ``window`` up
# front, mirroring the dask path's validator (which has rejected
# out-of-bounds windows since #1561), so both backends share the
# contract.
# ===========================================================================


class TestWindowOutOfBoundsEager:
    """Out-of-bounds windows raise a clear ``ValueError`` on the eager path."""

    def test_negative_start_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='outside the source extent'):
            open_geotiff(path, window=(-5, -5, 5, 5))

    def test_past_right_edge_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='outside the source extent'):
            open_geotiff(path, window=(0, 5, 5, 15))

    def test_past_bottom_edge_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='outside the source extent'):
            open_geotiff(path, window=(5, 0, 15, 5))

    def test_past_both_edges_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(ValueError, match='outside the source extent'):
            open_geotiff(path, window=(5, 5, 15, 15))

    def test_zero_size_window_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(
            ValueError,
            match='outside the source extent|non-positive size',
        ):
            open_geotiff(path, window=(3, 3, 3, 3))

    def test_inverted_window_raises_value_error(self, tmp_path):
        path = _make_raster(str(tmp_path))
        with pytest.raises(
            ValueError,
            match='outside the source extent|non-positive size',
        ):
            open_geotiff(path, window=(5, 5, 3, 3))


class TestWindowInBoundsEager:
    """In-bounds windows still work on the eager path."""

    def test_full_extent_window_returns_full_array(self, tmp_path):
        path = _make_raster(str(tmp_path))
        result = open_geotiff(path, window=(0, 0, 10, 10))
        assert result.shape == (10, 10)
        assert result.coords['y'].size == 10
        assert result.coords['x'].size == 10

    def test_interior_window_returns_correct_subset(self, tmp_path):
        path = _make_raster(str(tmp_path))
        result = open_geotiff(path, window=(2, 3, 7, 8))
        assert result.shape == (5, 5)
        assert result.coords['y'].size == 5
        assert result.coords['x'].size == 5
        expected = np.arange(100, dtype=np.float32).reshape(10, 10)[2:7, 3:8]
        np.testing.assert_array_equal(result.values, expected)

    def test_edge_aligned_window_returns_correct_subset(self, tmp_path):
        path = _make_raster(str(tmp_path))
        # Window touches but does not exceed the edge.
        result = open_geotiff(path, window=(0, 0, 10, 10))
        assert result.shape == (10, 10)


class TestWindowBackendParity:
    """The eager and dask paths must share window validation."""

    def test_eager_and_dask_paths_share_window_validation(self, tmp_path):
        """Both backends must raise ValueError on the same bad window."""
        path = _make_raster(str(tmp_path))
        bad_window = (5, 5, 15, 15)

        with pytest.raises(ValueError) as eager_exc:
            open_geotiff(path, window=bad_window)
        with pytest.raises(ValueError) as dask_exc:
            open_geotiff(path, window=bad_window, chunks=4)

        assert 'outside the source extent' in str(eager_exc.value)
        assert 'outside the source extent' in str(dask_exc.value)
        # Both should reference the source dimensions (10x10) somewhere.
        assert '10' in str(eager_exc.value)
        assert '10' in str(dask_exc.value)

    def test_eager_and_dask_paths_share_window_message_format(self, tmp_path):
        """Eager and dask paths emit messages matching the same format."""
        path = _make_raster(str(tmp_path))
        bad_window = (-5, -5, 5, 5)

        with pytest.raises(ValueError) as eager_exc:
            open_geotiff(path, window=bad_window)
        with pytest.raises(ValueError) as dask_exc:
            open_geotiff(path, window=bad_window, chunks=4)

        eager_msg = str(eager_exc.value)
        dask_msg = str(dask_exc.value)
        assert 'window=' in eager_msg
        assert 'window=' in dask_msg

    def test_reproducer_raises_clean_error(self, tmp_path):
        """The reproducer should raise ValueError, not
        CoordinateValidationError from xarray's internals."""
        path = _make_raster(str(tmp_path))
        try:
            result = open_geotiff(path, window=(5, 5, 15, 15))
            pytest.fail(f'expected ValueError, got result shape {result.shape}')
        except ValueError as e:
            msg = str(e)
            assert 'window' in msg.lower()
            assert 'source extent' in msg.lower() or 'out' in msg.lower()


# ===========================================================================
# Section 6: degenerate pixel-size fail-closed (#2214)
#
# ``coords_to_transform`` used to borrow the non-degenerate axis's pixel
# size for a length-1 axis, so a 30m x 10m source served as a 1xN strip
# silently wrote 30m x 30m pixels. The default is now fail-closed: a 1xN
# / Nx1 DataArray with spatial coords, no ``attrs['transform']``, and no
# opt-in flag raises ``ValueError``. ``attrs['transform']`` supplies the
# true geometry; ``attrs['assume_square_pixels_for_degenerate_axis'] =
# True`` opts back into the borrow path. Every writer routes through
# ``_require_transform_for_georeferenced``, so the guard is correct by
# construction; the per-backend tests pin it against future refactors.
# ===========================================================================


# Source raster the bug reporter described: 30 m x pixels, 10 m y pixels.
PIXEL_X_TRUE = 30.0
PIXEL_Y_TRUE = 10.0
X0 = -120.0
Y0 = 45.0


def _strip_1xN_nonsquare() -> xr.DataArray:
    """A 1xN strip whose source raster has non-square pixels.

    The x coord spacing is 30 (readable from coords). The y axis is
    length 1, so the y pixel size of 10 cannot be recovered from coords.
    """
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(1, 8),
        dims=("y", "x"),
        coords={
            "x": X0 + np.arange(8, dtype="float64") * PIXEL_X_TRUE,
            "y": np.array([Y0], dtype="float64"),
        },
        attrs={"crs": 4326},
    )


def _strip_Nx1_nonsquare() -> xr.DataArray:
    """An Nx1 profile whose source raster has non-square pixels."""
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(8, 1),
        dims=("y", "x"),
        coords={
            "x": np.array([X0], dtype="float64"),
            "y": Y0 - np.arange(8, dtype="float64") * PIXEL_Y_TRUE,
        },
        attrs={"crs": 4326},
    )


class TestDegenerateWritesFailClosed:
    """A 1xN / Nx1 write with spatial coords must raise without opt-in."""

    def test_1xN_without_transform_or_optin_raises(self, tmp_path):
        da = _strip_1xN_nonsquare()
        p = str(tmp_path / "fail_1xN.tif")
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, p)
        msg = str(excinfo.value)
        # The error must name both escape hatches.
        assert "transform" in msg
        assert "assume_square_pixels_for_degenerate_axis" in msg

    def test_Nx1_without_transform_or_optin_raises(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        p = str(tmp_path / "fail_Nx1.tif")
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, p)
        msg = str(excinfo.value)
        assert "transform" in msg
        assert "assume_square_pixels_for_degenerate_axis" in msg


class TestDegenerateWritesWithExplicitTransform:
    """``attrs['transform']`` round-trips the supplied pixel size exactly."""

    def test_1xN_with_attrs_transform_round_trips_true_pixel_size(self, tmp_path):
        da = _strip_1xN_nonsquare()
        # rasterio 6-tuple: (a, b, c, d, e, f) = (px, 0, ox, 0, py, oy)
        true_transform = (
            PIXEL_X_TRUE, 0.0, X0 - PIXEL_X_TRUE * 0.5,
            0.0, -PIXEL_Y_TRUE, Y0 + PIXEL_Y_TRUE * 0.5,
        )
        da = da.copy()
        da.attrs = {**da.attrs, "transform": true_transform}

        p = str(tmp_path / "explicit_1xN.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # The non-degenerate axis (x) keeps its true 30 m step.
        x_step = float(r.coords["x"][1] - r.coords["x"][0])
        assert x_step == pytest.approx(PIXEL_X_TRUE)
        # The readback transform records the true 10 m y pixel.
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_Nx1_with_attrs_transform_round_trips_true_pixel_size(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        true_transform = (
            PIXEL_X_TRUE, 0.0, X0 - PIXEL_X_TRUE * 0.5,
            0.0, -PIXEL_Y_TRUE, Y0 + PIXEL_Y_TRUE * 0.5,
        )
        da = da.copy()
        da.attrs = {**da.attrs, "transform": true_transform}

        p = str(tmp_path / "explicit_Nx1.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        y_step = float(r.coords["y"][1] - r.coords["y"][0])
        # y decreases top-to-bottom by convention.
        assert y_step == pytest.approx(-PIXEL_Y_TRUE)
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)


class TestDegenerateWritesWithOptIn:
    """``attrs['assume_square_pixels_for_degenerate_axis'] = True`` opts in.

    Behaviour matches the pre-#2214 #1945 borrow path: the writer assumes
    the source raster is square and copies the non-degenerate axis's
    pixel size onto the degenerate axis. The opt-in must be the boolean
    ``True`` -- a stray truthy string must not enable the borrow.
    """

    def test_1xN_optin_borrows_from_x_axis(self, tmp_path):
        da = _strip_1xN_nonsquare()
        da = da.copy()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}

        p = str(tmp_path / "optin_1xN.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # The borrow path copies the magnitude of the x step onto the
        # y axis with the y-down sign convention (true x=30, true y=10
        # -> the file records y=-30). That is the documented opt-in cost.
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_X_TRUE)

    def test_Nx1_optin_borrows_from_y_axis(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        da = da.copy()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}

        p = str(tmp_path / "optin_Nx1.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        # Borrow path takes abs(y step) = 10 and copies it onto pixel_width.
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_Y_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_optin_must_be_boolean_True_not_truthy_string(self, tmp_path):
        """A stray ``'yes'`` value must not silently enable the borrow path."""
        da = _strip_1xN_nonsquare()
        da = da.copy()
        # 'yes' is truthy but is NOT the boolean True. The identity check
        # on ``_assume_square_for_degenerate`` rejects everything that
        # isn't ``is True`` so an accidental attrs value can't re-enable
        # the silent-invent path.
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": "yes"}

        p = str(tmp_path / "optin_bad.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


class TestMultiRowMultiColumnUnchanged:
    """The fix must not touch the regular (non-degenerate) write path."""

    def test_2x2_writes_without_optin(self, tmp_path):
        """A 2x2 raster reads its pixel size off the coords; no opt-in needed."""
        da = xr.DataArray(
            np.arange(4, dtype="float32").reshape(2, 2),
            dims=("y", "x"),
            coords={
                "x": np.array([X0, X0 + PIXEL_X_TRUE], dtype="float64"),
                "y": np.array([Y0, Y0 - PIXEL_Y_TRUE], dtype="float64"),
            },
            attrs={"crs": 4326},
        )
        p = str(tmp_path / "multi_2x2.tif")
        # No fail-closed: both axes have length >= 2.
        to_geotiff(da, p)

        r = open_geotiff(p)
        tx = r.attrs["transform"]
        # True (non-borrowed) pixel sizes on both axes.
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)

    def test_3x5_writes_without_optin(self, tmp_path):
        rng = np.random.RandomState(0)
        arr = rng.random((3, 5)).astype("float32")
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "x": X0 + np.arange(5, dtype="float64") * PIXEL_X_TRUE,
                "y": Y0 - np.arange(3, dtype="float64") * PIXEL_Y_TRUE,
            },
            attrs={"crs": 4326},
        )
        p = str(tmp_path / "multi_3x5.tif")
        to_geotiff(da, p)

        r = open_geotiff(p)
        tx = r.attrs["transform"]
        assert tx[0] == pytest.approx(PIXEL_X_TRUE)
        assert tx[4] == pytest.approx(-PIXEL_Y_TRUE)
        np.testing.assert_array_equal(np.asarray(r.values), arr)


class TestCoordsToTransformHelperContract:
    """Direct tests for ``coords_to_transform`` so the contract is pinned
    independent of any writer wrapping."""

    def test_degenerate_without_optin_returns_None(self):
        """The writer relies on ``None`` to trigger the fail-closed branch
        via ``require_transform_for_georeferenced``."""
        da = _strip_1xN_nonsquare()
        assert coords_to_transform(da) is None

    def test_degenerate_with_optin_returns_borrowed_transform(self):
        da = _strip_1xN_nonsquare()
        da.attrs = {**da.attrs,
                    "assume_square_pixels_for_degenerate_axis": True}
        t = coords_to_transform(da)
        assert t is not None
        assert t.pixel_width == pytest.approx(PIXEL_X_TRUE)
        # Borrowed -- not the true 10.0.
        assert t.pixel_height == pytest.approx(-PIXEL_X_TRUE)

    def test_multi_axis_ignores_optin_flag(self):
        """The opt-in flag is only consulted for the degenerate branch.
        A regular 2x2 write doesn't trip the borrow path even if the flag
        is set, so the writer can't accidentally start borrowing."""
        da = xr.DataArray(
            np.arange(4, dtype="float32").reshape(2, 2),
            dims=("y", "x"),
            coords={
                "x": np.array([X0, X0 + PIXEL_X_TRUE], dtype="float64"),
                "y": np.array([Y0, Y0 - PIXEL_Y_TRUE], dtype="float64"),
            },
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        assert t.pixel_width == pytest.approx(PIXEL_X_TRUE)
        assert t.pixel_height == pytest.approx(-PIXEL_Y_TRUE)


class TestDegenerateFailClosedAcrossBackends:
    """Every writer raises on a 1xN / Nx1 input without opt-in or transform."""

    def test_dask_numpy_1xN_raises(self, tmp_path):
        da = _strip_1xN_nonsquare().chunk({"x": 4, "y": 1})
        p = str(tmp_path / "dask_np_fail_1xN.tif")
        with pytest.raises(ValueError) as excinfo:
            to_geotiff(da, p)
        msg = str(excinfo.value)
        assert "transform" in msg
        assert "assume_square_pixels_for_degenerate_axis" in msg

    def test_dask_numpy_Nx1_raises(self, tmp_path):
        da = _strip_Nx1_nonsquare().chunk({"x": 1, "y": 4})
        p = str(tmp_path / "dask_np_fail_Nx1.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)

    def test_vrt_1xN_raises(self, tmp_path):
        """``to_geotiff(da, '*.vrt')`` dispatches through the VRT writer."""
        da = _strip_1xN_nonsquare()
        p = str(tmp_path / "vrt_fail_1xN.vrt")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)

    def test_vrt_Nx1_raises(self, tmp_path):
        da = _strip_Nx1_nonsquare()
        p = str(tmp_path / "vrt_fail_Nx1.vrt")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)

    @requires_gpu
    def test_gpu_1xN_raises(self, tmp_path):
        import cupy

        da_cpu = _strip_1xN_nonsquare()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "gpu_fail_1xN.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            write_geotiff_gpu(da_gpu, p)

    @requires_gpu
    def test_gpu_Nx1_raises(self, tmp_path):
        import cupy

        da_cpu = _strip_Nx1_nonsquare()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "gpu_fail_Nx1.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            write_geotiff_gpu(da_gpu, p)

    @requires_gpu
    def test_dask_cupy_1xN_raises(self, tmp_path):
        import cupy

        da_cpu = _strip_1xN_nonsquare()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        da_gpu = da_gpu.chunk({"x": 4, "y": 1})
        p = str(tmp_path / "dask_cupy_fail_1xN.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da_gpu, p)
