"""Regression tests for issue #2158.

Before the fix, ``read_vrt(vrt, mask_nodata=False)`` silently rewrote
float source sentinels (e.g. ``-9999.0``) to NaN inside the internal
VRT reader (``_vrt._read_data``), so the documented opt-out only
worked for integer sources. The public backend's ``mask_nodata=False``
branch skipped the integer post-decode helper but never reached the
inline float masking, which had already destroyed the literal
sentinel pixels by the time the buffer surfaced to the public layer.

The fix threads ``mask_nodata`` into the internal reader and gates
both inline masking branches (float source NaN substitution and
integer-source-feeding-float-VRT promotion) on the kwarg. The
``attrs['masked_nodata']`` stamp on the eager and chunked VRT paths
is also AND-ed with ``mask_nodata`` so the attr does not lie when
the inline masking is skipped.
"""
from __future__ import annotations

import numpy as np

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._writer import write


def _write_float32_with_sentinel(tmp_path, sentinel=-9999.0,
                                 filename='float_2158.tif'):
    """float32 GeoTIFF with a non-NaN sentinel and matching pixels.

    The middle row has a literal ``-9999.0`` so the inline masking
    actually has something to rewrite.
    """
    band = np.array([[1.0, 2.0, 3.0],
                     [4.0, sentinel, 6.0],
                     [7.0, sentinel, 9.0]], dtype=np.float32)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p, band


def _write_float64_with_fractional_sentinel(tmp_path, sentinel=-9999.25,
                                            filename='float64_2158.tif'):
    """float64 GeoTIFF with a fractional sentinel.

    Float32's exact-cast rounding would clobber a fractional value
    like ``-9999.25``; the float64 path is the only one where the
    sentinel survives lossless.
    """
    band = np.array([[1.0, 2.0],
                     [sentinel, 4.0]], dtype=np.float64)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p, band


def _build_vrt(tmp_path, source_path, vrt_dtype, nodata_value,
               filename='float_2158.vrt', shape=(3, 3)):
    """Hand-roll a single-source VRT pointing at the float source."""
    h, w = shape
    vrt_xml = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{vrt_dtype}" band="1">
    <NoDataValue>{nodata_value}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = str(tmp_path / filename)
    with open(p, 'w') as f:
        f.write(vrt_xml)
    return p


# ---------------------------------------------------------------------------
# Baseline: ``mask_nodata=True`` still rewrites the float sentinel.
# ---------------------------------------------------------------------------


def test_default_mask_nodata_true_rewrites_float_sentinel(tmp_path):
    """The default behaviour (mask_nodata=True) still substitutes NaN.

    Pins the existing contract so the fix below does not regress the
    masking happy path.
    """
    src, _ = _write_float32_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', -9999.0)

    r = read_vrt(vrt)

    assert r.dtype == np.float32
    # The two sentinel positions in the source array are at (1, 1)
    # and (2, 1). Both should be NaN after masking.
    assert np.isnan(r.values[1, 1])
    assert np.isnan(r.values[2, 1])
    # Non-sentinel pixels untouched.
    assert r.values[0, 0] == 1.0
    assert r.values[1, 0] == 4.0
    assert r.attrs.get('nodata') == -9999.0
    assert r.attrs.get('masked_nodata') is True


# ---------------------------------------------------------------------------
# The fix: ``mask_nodata=False`` preserves the literal float sentinel.
# ---------------------------------------------------------------------------


def test_eager_mask_nodata_false_preserves_float_sentinel(tmp_path):
    """Eager VRT path: ``mask_nodata=False`` keeps the literal sentinel.

    Before #2158 this assertion failed -- the sentinel pixels were
    silently rewritten to NaN inside ``_vrt._read_data`` regardless
    of the kwarg.
    """
    src, original = _write_float32_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', -9999.0)

    r = read_vrt(vrt, mask_nodata=False)

    assert r.dtype == np.float32
    # No NaNs anywhere; sentinel survives as the literal float value.
    assert not np.isnan(r.values).any()
    assert r.values[1, 1] == np.float32(-9999.0)
    assert r.values[2, 1] == np.float32(-9999.0)
    # The full array round-trips exactly when the opt-out is honored.
    np.testing.assert_array_equal(r.values, original)
    # The declared sentinel is still surfaced for downstream maskers.
    assert r.attrs.get('nodata') == -9999.0
    # The buffer is not NaN-aware: ``masked_nodata`` should reflect
    # that, not lie.
    assert r.attrs.get('masked_nodata') is False


def test_chunked_mask_nodata_false_preserves_float_sentinel(tmp_path):
    """Chunked VRT path: ``mask_nodata=False`` keeps the literal sentinel.

    The chunked path used to call ``_read_vrt_internal`` from
    ``_vrt_chunk_read`` without forwarding the kwarg, so per-chunk
    decodes silently rewrote float sentinels too. With #2158 the
    kwarg is forwarded into the internal reader and both paths agree.
    """
    src, original = _write_float32_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', -9999.0)

    r = read_vrt(vrt, chunks=2, mask_nodata=False)

    assert r.dtype == np.float32
    computed = r.compute()
    assert not np.isnan(computed.values).any()
    assert computed.values[1, 1] == np.float32(-9999.0)
    assert computed.values[2, 1] == np.float32(-9999.0)
    np.testing.assert_array_equal(computed.values, original)
    assert computed.attrs.get('nodata') == -9999.0
    assert computed.attrs.get('masked_nodata') is False


def test_eager_and_chunked_agree_under_mask_nodata_false(tmp_path):
    """Cross-path parity: eager and chunked produce the same buffer.

    Before #2158 the two paths could disagree because both rewrote
    the sentinel inline but at slightly different points in the
    pipeline. With the opt-out honored, both paths land on the
    untouched source array.
    """
    src, _ = _write_float32_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', -9999.0)

    eager = read_vrt(vrt, mask_nodata=False)
    chunked = read_vrt(vrt, chunks=2, mask_nodata=False).compute()

    np.testing.assert_array_equal(eager.values, chunked.values)
    assert eager.attrs.get('masked_nodata') == chunked.attrs.get(
        'masked_nodata')


# ---------------------------------------------------------------------------
# Fractional sentinel: float64 source with a non-integer nodata.
# ---------------------------------------------------------------------------


def test_mask_nodata_false_float64_fractional_sentinel(tmp_path):
    """A fractional sentinel survives the float64 opt-out path.

    Float32 would round ``-9999.25`` to the nearest representable
    value, so this corner is float64-only. With the opt-out honored
    the pixel keeps its exact bit pattern.
    """
    src, original = _write_float64_with_fractional_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float64', -9999.25,
                     filename='float64_2158.vrt', shape=(2, 2))

    r = read_vrt(vrt, mask_nodata=False)

    assert r.dtype == np.float64
    assert r.values[1, 0] == -9999.25
    np.testing.assert_array_equal(r.values, original)


# ---------------------------------------------------------------------------
# Bit-exact equivalence: masked NaN positions match unmasked sentinel positions.
# ---------------------------------------------------------------------------


def test_masked_vs_unmasked_differ_only_at_sentinels(tmp_path):
    """``mask_nodata=True`` and ``=False`` differ only where the sentinel hits.

    Every pixel that is NaN in the masked output equals the declared
    sentinel in the unmasked output, and every non-sentinel pixel is
    bit-identical between the two reads. This pins the contract that
    the opt-out is a pure passthrough on the non-sentinel positions.
    """
    src, _ = _write_float32_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', -9999.0)

    masked = read_vrt(vrt).values
    unmasked = read_vrt(vrt, mask_nodata=False).values

    nan_positions = np.isnan(masked)
    sentinel_positions = unmasked == np.float32(-9999.0)
    np.testing.assert_array_equal(nan_positions, sentinel_positions)
    # Non-sentinel pixels bit-identical between the two reads.
    np.testing.assert_array_equal(masked[~nan_positions],
                                  unmasked[~sentinel_positions])


# ---------------------------------------------------------------------------
# Integer source feeding a float-dataType VRT: the other branch the fix gated.
# ---------------------------------------------------------------------------


def _write_uint16_with_sentinel(tmp_path, sentinel=65535,
                                filename='uint16_2158.tif'):
    """uint16 GeoTIFF with a matching sentinel.

    Used to exercise the integer-source-feeding-float-VRT promotion at
    ``_vrt.py:1351-1390``. With ``mask_nodata=True`` the sentinel pixel
    surfaces as NaN in the float buffer; with ``mask_nodata=False`` the
    literal integer value flows through the int->float cast and lands
    as ``65535.0``.
    """
    band = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p, band


def test_int_source_float_vrt_mask_nodata_false_keeps_literal(tmp_path):
    """Integer source feeding a Float32 VRT preserves the literal sentinel.

    Pins the second branch of the inline masking that #2158 gated.
    Before the fix, ``_vrt._read_data`` ran the int->float-with-NaN
    promotion unconditionally, so even ``mask_nodata=False`` lost the
    sentinel. After the fix the integer source pixel survives the
    int->float cast as ``65535.0`` and ``masked_nodata`` reflects
    that no masking ran.
    """
    src, _ = _write_uint16_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', 65535,
                     filename='int_float_2158.vrt', shape=(2, 2))

    r = read_vrt(vrt, mask_nodata=False)

    assert r.dtype == np.float32
    # No NaN substitution -- the sentinel survives as a literal float.
    assert not np.isnan(r.values).any()
    assert r.values[1, 1] == np.float32(65535.0)
    assert r.values[0, 0] == 1.0
    assert r.attrs.get('nodata') == 65535.0
    # Buffer is not NaN-aware even though dtype is float.
    assert r.attrs.get('masked_nodata') is False


def test_int_source_float_vrt_default_still_promotes(tmp_path):
    """Default ``mask_nodata=True`` still NaN-masks the int->float promotion.

    Baseline that documents the pre-#2158 contract for the integer
    source path: the existing #1616 behavior is unchanged when the
    opt-out is not requested.
    """
    src, _ = _write_uint16_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', 65535,
                     filename='int_float_default_2158.vrt', shape=(2, 2))

    r = read_vrt(vrt)

    assert r.dtype == np.float32
    assert np.isnan(r.values[1, 1])
    assert r.values[0, 0] == 1.0
    assert r.attrs.get('nodata') == 65535.0
    assert r.attrs.get('masked_nodata') is True
