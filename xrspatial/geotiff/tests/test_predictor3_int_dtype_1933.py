"""Regression tests for issue #1933.

A malformed TIFF that claims ``Predictor=3`` (Floating-Point Predictor,
TIFF Technical Note 3) paired with an integer ``SampleFormat`` used to
decode silently to garbage. The byte-swizzle unshuffle ran on integer
bytes and produced values that look like valid integers, with no
warning. The writer side already rejects this combination in
``_writer._resolve_predictor``; these tests assert reader symmetry.

The fix lives in ``xrspatial.geotiff._validation`` (helper
``_validate_predictor_sample_format``) and is invoked at every IFD-read
site (eager numpy, dask, GPU tiled, GPU stripped).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._compression import COMPRESSION_NONE
from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
                                       TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_PREDICTOR,
                                       TAG_ROWS_PER_STRIP, TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
                                       TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS)
from xrspatial.geotiff._validation import _validate_predictor_sample_format
from xrspatial.geotiff._writer import _assemble_standard_layout, _write_stripped


def _build_predictor3_uint32_tiff(arr: np.ndarray) -> bytes:
    """Build a malformed TIFF: predictor=3 + uint32 (integer) sample format.

    Uses the in-repo ``_assemble_standard_layout`` so we can write tags
    the public writer would reject. Compression is COMPRESSION_NONE so
    the strip bytes are exactly the raw integer values; the bug is then
    visible by comparing the round-tripped values against the originals.
    """
    rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
    bits_per_sample, _ = numpy_to_tiff_dtype(arr.dtype)
    # SampleFormat=1 (UINT) is the *integer* tag that the malformed file
    # advertises; predictor=3 is the *float* predictor. Pair them.
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, arr.shape[1]),
        (TAG_IMAGE_LENGTH, LONG, 1, arr.shape[0]),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
        (TAG_COMPRESSION, SHORT, 1, COMPRESSION_NONE),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, 1),  # UINT
        (TAG_PREDICTOR, SHORT, 1, 3),  # floating-point predictor
        (TAG_ROWS_PER_STRIP, SHORT, 1, arr.shape[0]),
        (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    parts = [(arr, arr.shape[1], arr.shape[0], rel_off, bc, chunks)]
    return _assemble_standard_layout(8, [tags], parts, bigtiff=False)


class TestPredictor3IntegerSampleFormatRejected:
    """The validator rejects predictor=3 with non-float sample formats."""

    def test_helper_rejects_pred3_sf1_uint(self):
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 1)

    def test_helper_rejects_pred3_sf2_int(self):
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 2)

    def test_helper_rejects_pred3_sf4_undefined(self):
        # SampleFormat=4 (UNDEFINED) is treated as uint by the dtype map;
        # routing it through predictor=3 also produces garbage.
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format(3, 4)

    def test_helper_accepts_pred3_sf3_float(self):
        # The legitimate combination remains a no-op.
        _validate_predictor_sample_format(3, 3)

    def test_helper_accepts_pred1_with_any_sf(self):
        # Predictor=1 (none) is sample-format-agnostic.
        _validate_predictor_sample_format(1, 1)
        _validate_predictor_sample_format(1, 2)
        _validate_predictor_sample_format(1, 3)

    def test_helper_accepts_pred2_with_any_sf(self):
        # Predictor=2 (horizontal) is sample-format-agnostic.
        _validate_predictor_sample_format(2, 1)
        _validate_predictor_sample_format(2, 2)
        _validate_predictor_sample_format(2, 3)

    def test_helper_normalizes_tuple_predictor(self):
        # IFD.predictor delegates to get_value, which returns a tuple for
        # a malformed Predictor tag with count > 1. The validator must
        # still fire on the (3,) + non-float pair.
        with pytest.raises(ValueError, match="Predictor=3"):
            _validate_predictor_sample_format((3,), 1)
        # Empty tuple falls back to predictor=1 (none) -> no-op.
        _validate_predictor_sample_format((), 1)
        # Non-3 tuple predictor + non-float sample_format -> no-op.
        _validate_predictor_sample_format((2,), 1)


class TestEagerReadRejectsMalformedFile:
    """``open_geotiff`` raises on the malformed predictor=3 + uint32 file."""

    def test_open_geotiff_eager_raises(self, tmp_path):
        arr = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32.tif"
        path.write_bytes(_build_predictor3_uint32_tiff(arr))

        with pytest.raises(ValueError, match="Predictor=3"):
            open_geotiff(str(path))

    def test_open_geotiff_dask_raises(self, tmp_path):
        arr = np.array(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.uint32)
        path = tmp_path / "pred3_uint32_dask.tif"
        path.write_bytes(_build_predictor3_uint32_tiff(arr))

        from xrspatial.geotiff import read_geotiff_dask
        with pytest.raises(ValueError, match="Predictor=3"):
            read_geotiff_dask(str(path), chunks=64)


class TestValidPredictor3StillWorks:
    """The fix does not break legitimate predictor=3 float files."""

    def test_predictor3_float32_round_trip(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        pytest.importorskip("imagecodecs")

        arr = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        path = tmp_path / "pred3_float32.tif"
        tifffile.imwrite(
            str(path), arr, predictor=3, compression="deflate")

        result = open_geotiff(str(path))
        np.testing.assert_array_equal(result.values, arr)
