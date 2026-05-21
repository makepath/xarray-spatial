"""Unit tests for the read-finalization helpers in ``_attrs.py``.

Issue #2177 (PR B of #2162).

The two helpers ``_finalize_eager_read`` and ``_finalize_lazy_read_attrs``
capture the post-decode finalization steps duplicated across the four
read backends. This test file pins their behaviour in isolation: it
synthesises ``GeoInfo`` fixtures and exercises the helpers directly,
without going through the read backends (those are migrated in waves 2
and 3).

The four invariants covered here:

1. Eager helper populates the nodata lifecycle attrs (``nodata``,
   ``nodata_pixels_present``, ``nodata_dtype_cast``, ``masked_nodata``)
   and the ``georef_status`` attr across float and integer input dtypes.
2. ``mask_nodata=False`` leaves the buffer alone and surfaces
   ``nodata_pixels_present`` without rewriting.
3. Lazy helper produces the same attrs minus ``nodata_pixels_present``
   (the dask contract from #2135).
4. Both helpers validate ``geo_info`` first so a rejected file does not
   leak partial attrs onto the caller's dict.
5. Both helpers route ``mask_sentinel != nodata`` through the masking
   step (the GPU MinIsWhite inversion case).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._attrs import (
    _finalize_eager_read,
    _finalize_lazy_read_attrs,
)
from xrspatial.geotiff._errors import UnparseableCRSError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeTransform:
    """Stand-in for ``GeoInfo.transform``. Axis-aligned by default."""

    def __init__(self, origin_x=0.0, origin_y=10.0,
                 pixel_width=1.0, pixel_height=-1.0,
                 rotated_affine=None):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.rotated_affine = rotated_affine


class _FakeGeoInfo:
    """Minimal ``GeoInfo`` stand-in covering the fields the helpers read.

    Mirrors the shape used in ``test_geotiff_metadata_2139.py`` so the
    two test files stay in sync if the ``GeoInfo`` field set ever grows.
    """

    def __init__(
        self,
        *,
        transform=None,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=1,  # RASTER_PIXEL_IS_AREA
        has_georef=True,
        nodata=None,
        extra_tags=None,
        image_description=None,
        extra_samples=None,
        gdal_metadata=None,
        gdal_metadata_xml=None,
        x_resolution=None,
        y_resolution=None,
        resolution_unit=None,
    ):
        self.transform = transform if transform is not None else _FakeTransform()
        self.crs_epsg = crs_epsg
        self.crs_wkt = crs_wkt
        self.raster_type = raster_type
        self.has_georef = has_georef
        self.nodata = nodata
        self.extra_tags = extra_tags
        self.image_description = image_description
        self.extra_samples = extra_samples
        self.gdal_metadata = gdal_metadata
        self.gdal_metadata_xml = gdal_metadata_xml
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.resolution_unit = resolution_unit


def _default_geo_info(**overrides):
    """A minimal georeferenced ``_FakeGeoInfo`` for happy-path tests."""
    base = dict(crs_epsg=4326, crs_wkt='EPSG:4326', nodata=-9999)
    base.update(overrides)
    return _FakeGeoInfo(**base)


# ---------------------------------------------------------------------------
# Eager helper: attrs surface
# ---------------------------------------------------------------------------


def test_eager_float_input_sets_nodata_lifecycle_attrs():
    arr = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata'] == -9999
    assert da.attrs['masked_nodata'] is True
    assert da.attrs['nodata_pixels_present'] is True
    # No explicit dtype= cast was requested.
    assert 'nodata_dtype_cast' not in da.attrs
    # georef_status comes from _populate_attrs_from_geo_info.
    assert da.attrs['georef_status'] == 'full'
    # The sentinel pixel is now NaN.
    assert np.isnan(np.asarray(da.values)[0, 1])


def test_eager_int_input_promotes_and_sets_attrs():
    arr = np.array([[1, 0], [0, -1]], dtype=np.int16)
    gi = _default_geo_info(nodata=0)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=0,
        mask_sentinel=0,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.dtype.kind == 'f'
    assert da.attrs['nodata'] == 0
    assert da.attrs['masked_nodata'] is True
    assert da.attrs['nodata_pixels_present'] is True
    assert da.attrs['georef_status'] == 'full'


def test_eager_explicit_dtype_records_dtype_cast():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype='float64',
        window=None,
        name='t',
    )

    assert da.dtype == np.float64
    assert da.attrs['nodata_dtype_cast'] == 'float64'
    # No sentinel matched, so pixels_present is False rather than absent.
    assert da.attrs['nodata_pixels_present'] is False


def test_eager_no_sentinel_pixels_present_is_false():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata_pixels_present'] is False
    assert da.attrs['masked_nodata'] is True


# ---------------------------------------------------------------------------
# Eager helper: mask_nodata=False opt-out (issue #2052)
# ---------------------------------------------------------------------------


def test_eager_mask_nodata_false_skips_mask_keeps_attr_surface():
    arr = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=False,
        dtype=None,
        window=None,
        name='t',
    )

    # The sentinel pixel is preserved literally.
    assert da.values[0, 1] == -9999.0
    # ``masked_nodata=False`` per the #2092 contract.
    assert da.attrs['masked_nodata'] is False
    # ``nodata_pixels_present`` still surfaces so callers know a sentinel
    # pixel exists in the buffer (#2135).
    assert da.attrs['nodata_pixels_present'] is True


def test_eager_mask_nodata_false_no_sentinel_present():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=False,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata_pixels_present'] is False
    assert da.attrs['masked_nodata'] is False


# ---------------------------------------------------------------------------
# Eager helper: no declared sentinel
# ---------------------------------------------------------------------------


def test_eager_no_nodata_omits_nodata_attrs():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info(nodata=None)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=None,
        mask_sentinel=None,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert 'nodata' not in da.attrs
    assert 'masked_nodata' not in da.attrs
    assert 'nodata_pixels_present' not in da.attrs
    assert 'nodata_dtype_cast' not in da.attrs


# ---------------------------------------------------------------------------
# Eager helper: mask_sentinel != nodata (GPU MinIsWhite inversion, #1809)
# ---------------------------------------------------------------------------


def test_eager_mask_sentinel_differs_from_nodata():
    # MinIsWhite inverts the sentinel: nodata=0 on disk but the in-memory
    # buffer has been inverted, so the masking value is 255 (8-bit) not 0.
    arr = np.array([[100.0, 255.0], [50.0, 0.0]], dtype=np.float32)
    gi = _default_geo_info(nodata=0)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=0,           # canonical attrs['nodata'] keeps the on-disk sentinel
        mask_sentinel=255,  # actual value to match in the in-memory buffer
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    # attrs['nodata'] keeps the file sentinel so a round-trip write
    # rebuilds the original GDAL_NODATA tag.
    assert da.attrs['nodata'] == 0
    # The pixel matching mask_sentinel=255 became NaN; the literal-0
    # pixel was left alone because mask_sentinel != 0.
    arr_out = np.asarray(da.values)
    assert np.isnan(arr_out[0, 1])
    assert arr_out[1, 1] == 0.0
    assert da.attrs['nodata_pixels_present'] is True


# ---------------------------------------------------------------------------
# Eager helper: validate-first ordering (no partial attrs leak)
# ---------------------------------------------------------------------------


def test_eager_raises_on_unparseable_crs_without_partial_attrs():
    # Try to bypass pyproj entirely: if pyproj is missing, the check
    # short-circuits and the test would be a no-op. Skip in that case.
    pytest.importorskip('pyproj')

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    # An obviously-bad WKT string. The unparseable-CRS check raises
    # before any attrs population step runs.
    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    # ``attrs_in`` is mutated only via ``dict(attrs_in)`` copy, but the
    # important contract is "if validation raises, no DataArray is built
    # and the caller's seed dict is untouched." Pass a seed dict and
    # confirm it is unchanged afterwards.
    seed = {'sentinel_marker': True}
    with pytest.raises(UnparseableCRSError):
        _finalize_eager_read(
            arr,
            geo_info=gi,
            nodata=-9999,
            mask_sentinel=-9999,
            mask_nodata=True,
            dtype=None,
            window=None,
            name='t',
            attrs_in=seed,
        )
    # Seed dict was never written to; the validation failure raised
    # before any ``attrs[...]`` step ran. Check both the exact contents
    # AND the length so a future partial-leak that adds new keys is
    # caught even if the existing key still matches.
    assert seed == {'sentinel_marker': True}
    assert len(seed) == 1


def test_eager_allow_unparseable_crs_bypasses_check():
    pytest.importorskip('pyproj')

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    # Opt-in bypass: the unparseable WKT lands on the attrs unchanged.
    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
        allow_unparseable_crs=True,
    )
    assert da.attrs['crs_wkt'] == 'NOT-A-VALID-WKT-STRING'


# ---------------------------------------------------------------------------
# Eager helper: attrs_in seed forwarded onto the DataArray
# ---------------------------------------------------------------------------


def test_eager_attrs_in_seed_is_copied_onto_dataarray():
    arr = np.array([[1.0]], dtype=np.float32)
    gi = _default_geo_info()

    seed = {'extra_user_attr': 'kept'}
    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=None,
        mask_sentinel=None,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
        attrs_in=seed,
    )

    assert da.attrs['extra_user_attr'] == 'kept'
    # The seed dict was not mutated -- the helper copies before writing.
    assert seed == {'extra_user_attr': 'kept'}


# ---------------------------------------------------------------------------
# Lazy helper: attrs surface (pixels_present is None per #2135 dask contract)
# ---------------------------------------------------------------------------


def test_lazy_float_dtype_sets_masked_true_no_pixels_present_attr():
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    assert attrs['nodata'] == -9999
    assert attrs['masked_nodata'] is True
    assert attrs['nodata_dtype_cast'] == 'float64'
    # pixels_present stays absent on the lazy path (#2135 dask contract).
    assert 'nodata_pixels_present' not in attrs
    assert attrs['georef_status'] == 'full'


def test_lazy_int_graph_dtype_keeps_masked_false():
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='int16',
        caller_dtype=None,
        window=None,
    )

    # Integer graph -> the per-chunk mask cannot have run, so masked=False
    # mirrors the #2092 contract even though the caller asked for masking.
    assert attrs['masked_nodata'] is False
    # Post-#2206 split: no caller cast -> ``nodata_dtype_cast`` stays
    # absent. The auto-promoted graph dtype never leaks into the attr.
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_int_caller_cast_records_dtype():
    # Companion to the test above: when the caller explicitly passes
    # ``dtype='int16'`` (an actual cast request), the attr surfaces.
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='int16',
        caller_dtype='int16',
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert attrs['nodata_dtype_cast'] == 'int16'


def test_lazy_graph_and_caller_dtype_differ():
    # Pins which parameter drives which attr when they actually differ:
    # ``graph_dtype`` drives ``masked_nodata`` (float graph -> per-chunk
    # mask runs), ``caller_dtype`` drives ``nodata_dtype_cast`` (records
    # the user's cast request, not the graph dtype). An accidental swap
    # of the two arguments would flip both attrs at once.
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='int16',
        window=None,
    )

    assert attrs['masked_nodata'] is True
    assert attrs['nodata_dtype_cast'] == 'int16'


def test_lazy_mask_promoted_graph_no_caller_cast():
    # Mirrors the dask backend's int->float64 auto-promotion case: graph
    # dtype is float64 because masking forces it, but the caller did not
    # ask for any cast. ``masked_nodata`` follows the graph dtype;
    # ``nodata_dtype_cast`` follows caller intent (absent here).
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is True
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_mask_nodata_false_sets_masked_false():
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=False,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert 'nodata_pixels_present' not in attrs


def test_lazy_no_graph_dtype_resolves_to_masked_false():
    # ``graph_dtype=None`` means the caller didn't resolve a graph dtype
    # to compare against (matches the pre-#2135 dask paths in tests).
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype=None,
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_no_nodata_omits_nodata_attrs():
    gi = _default_geo_info(nodata=None)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=None,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    assert 'nodata' not in attrs
    assert 'masked_nodata' not in attrs
    assert 'nodata_pixels_present' not in attrs
    assert 'nodata_dtype_cast' not in attrs


# ---------------------------------------------------------------------------
# Lazy helper: validate-first ordering
# ---------------------------------------------------------------------------


def test_lazy_raises_on_unparseable_crs_without_partial_attrs():
    pytest.importorskip('pyproj')

    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')
    seed = {'sentinel_marker': True}

    with pytest.raises(UnparseableCRSError):
        _finalize_lazy_read_attrs(
            geo_info=gi,
            nodata=-9999,
            mask_nodata=True,
            graph_dtype='float64',
            caller_dtype='float64',
            window=None,
            attrs_in=seed,
        )
    assert seed == {'sentinel_marker': True}
    assert len(seed) == 1


def test_lazy_allow_unparseable_crs_bypasses_check():
    pytest.importorskip('pyproj')

    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
        allow_unparseable_crs=True,
    )
    assert attrs['crs_wkt'] == 'NOT-A-VALID-WKT-STRING'


# ---------------------------------------------------------------------------
# Both helpers: parity on the non-mask attrs surface
# ---------------------------------------------------------------------------


def test_lazy_and_eager_produce_same_georef_and_nodata_attrs():
    # Same input on both paths should produce the same attrs dict apart
    # from ``nodata_pixels_present`` (eager-only by design).
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype='float64',
        window=None,
        name='t',
    )
    lazy_attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    keys_to_check = {
        'crs', 'crs_wkt', 'transform', 'georef_status',
        'nodata', 'masked_nodata', 'nodata_dtype_cast',
        '_xrspatial_geotiff_contract',
    }
    for key in keys_to_check:
        assert key in da.attrs, key
        assert key in lazy_attrs, key
        assert da.attrs[key] == lazy_attrs[key], key
    # And the lazy path does not carry the eager-only attr.
    assert 'nodata_pixels_present' in da.attrs
    assert 'nodata_pixels_present' not in lazy_attrs


# ---------------------------------------------------------------------------
# Lazy helper: mask_sentinel != nodata is a no-op (helper takes attrs only)
# ---------------------------------------------------------------------------


def test_lazy_helper_signature_omits_mask_sentinel():
    # The lazy helper deliberately does not accept ``mask_sentinel`` --
    # the dask graph applies masking per-chunk, and the per-chunk task
    # closes over the sentinel value separately. Pin the signature here
    # so a future refactor that adds ``mask_sentinel`` triggers a review.
    import inspect

    sig = inspect.signature(_finalize_lazy_read_attrs)
    assert 'mask_sentinel' not in sig.parameters


def test_eager_helper_signature_includes_mask_sentinel():
    # Mirror of the lazy check: the eager helper does take
    # ``mask_sentinel`` because the three GPU eager sites derive it three
    # different ways.
    import inspect

    sig = inspect.signature(_finalize_eager_read)
    assert 'mask_sentinel' in sig.parameters
