"""Nodata propagation, attrs, lifecycle, and semantics on read.

Sections in source order:

* ``nodatavals`` / ``_FillValue`` alias resolution on write feeds the
  read-back nodata tag.
* ``attrs['masked_nodata']`` reflects whether masking actually ran.
* Additive ``nodata_pixels_present`` / ``nodata_dtype_cast`` lifecycle
  attrs.
* Non-finite / fractional integer sentinels are a no-op under the opt-in.
* The dropped defensive copies on the read path do not alias
  caller-visible buffers.
* ``band_nodata`` forwarding through the shared validation helpers.
* ``NodataLifecycle`` decision contract + cross-backend parity.
* The ``nodata`` vs ``masked_nodata`` split-attrs contract.

The GPU nodata-mask reader coverage (in-place mask + removal-pin) stays
at the top of the file.

GPU-only nodata cases live in
``xrspatial/geotiff/tests/gpu/test_reader.py``; the GPU parity tests that
live here are gated through the shared ``requires_gpu`` marker (aliased
``_gpu_only`` for brevity).
"""
from __future__ import annotations

import importlib.util
import inspect
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, InvalidIntegerNodataError,
                               open_geotiff, read_geotiff_dask, read_vrt, to_geotiff)
from xrspatial.geotiff._attrs import _finalize_lazy_read_attrs, _validate_read_geo_info
from xrspatial.geotiff._backends import _gpu_helpers
from xrspatial.geotiff._errors import MixedBandMetadataError
from xrspatial.geotiff._nodata import NodataLifecycle
from xrspatial.geotiff.tests.conftest import requires_gpu as _gpu_only


@_gpu_only
def test_apply_nodata_mask_gpu_float_masks_sentinel_to_nan():
    """Float path masks the sentinel to NaN and leaves other pixels alone."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
    )
    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    assert out.dtype == cupy.float32
    host = out.get()
    assert np.isnan(host[1, 0])
    assert host[0, 0] == 1.0
    assert host[0, 1] == 2.0
    assert host[1, 1] == 4.0


@_gpu_only
def test_apply_nodata_mask_gpu_float_in_place_no_copy():
    """Float path mutates the input buffer in place."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    assert out.data.ptr == input_ptr


@_gpu_only
def test_apply_nodata_mask_gpu_float_alloc_count_unchanged():
    """Float path does not pull a fresh chunk-sized buffer from the pool."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    isolated_pool = cupy.cuda.MemoryPool()
    prev_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(isolated_pool.malloc)
    try:
        arr_gpu = cupy.full((512, 512), -9999.0, dtype=cupy.float32)
        arr_gpu[0, 0] = 1.0  # plant a non-sentinel pixel

        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_before = isolated_pool.total_bytes()

        out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_after = isolated_pool.total_bytes()

        array_bytes = arr_gpu.nbytes
        growth = total_after - total_before
        assert growth < array_bytes, (
            f"unexpected allocation growth {growth} bytes >= "
            f"array_bytes {array_bytes}; in-place mutation regressed"
        )
        assert out.data.ptr == arr_gpu.data.ptr
    finally:
        cupy.cuda.set_allocator(prev_allocator)
        isolated_pool.free_all_blocks()


@_gpu_only
def test_apply_nodata_mask_gpu_int_promotes_and_masks():
    """Integer path still promotes to float64 and masks the sentinel."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1, 2], [3, 4]], dtype=np.uint16)
    )
    out = _apply_nodata_mask_gpu(arr_gpu, 3)
    assert out.dtype == cupy.float64
    host = out.get()
    assert np.isnan(host[1, 0])
    assert host[0, 0] == 1.0
    assert host[0, 1] == 2.0
    assert host[1, 1] == 4.0


@_gpu_only
def test_apply_nodata_mask_gpu_int_no_extra_buffer_after_astype():
    """Integer path: only the ``astype(float64)`` buffer is allocated."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    isolated_pool = cupy.cuda.MemoryPool()
    prev_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(isolated_pool.malloc)
    try:
        arr_gpu = cupy.full((512, 512), 3, dtype=cupy.uint16)
        arr_gpu[0, 0] = 1  # ensure non-sentinel pixel exists

        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_before = isolated_pool.total_bytes()

        out = _apply_nodata_mask_gpu(arr_gpu, 3)
        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_after = isolated_pool.total_bytes()

        float64_bytes = out.nbytes
        growth = total_after - total_before
        assert growth < 2 * float64_bytes, (
            f"unexpected allocation growth {growth} bytes >= "
            f"2 * float64_bytes {2 * float64_bytes}; pre-fix double-alloc"
        )
    finally:
        cupy.cuda.set_allocator(prev_allocator)
        isolated_pool.free_all_blocks()


@_gpu_only
def test_apply_nodata_mask_gpu_float_nan_sentinel_noop():
    """NaN nodata on a float array stays a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, float('nan'))
    assert out.data.ptr == input_ptr
    np.testing.assert_array_equal(out.get(), [[1.0, 2.0], [3.0, 4.0]])


@_gpu_only
def test_apply_nodata_mask_gpu_none_nodata_passthrough():
    """``nodata is None`` returns the input array untouched."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, None)
    assert out.data.ptr == input_ptr
    assert out.dtype == cupy.int32


# ---------------------------------------------------------------------------
# Helper removal pin
# ---------------------------------------------------------------------------


def test_apply_nodata_mask_gpu_with_presence_not_importable():
    """The dead sibling helper stays removed."""
    # Covers both module-attribute absence and the import-time surface.
    with pytest.raises(ImportError):
        from xrspatial.geotiff._backends._gpu_helpers import \
            _apply_nodata_mask_gpu_with_presence  # noqa: F401


def test_apply_nodata_mask_gpu_still_present():
    """``_apply_nodata_mask_gpu`` is still on the chunked GPU dask path."""
    assert hasattr(_gpu_helpers, '_apply_nodata_mask_gpu')
    assert callable(_gpu_helpers._apply_nodata_mask_gpu)


# ===========================================================================
# nodata attr aliases on write feed the read-back tag
# ===========================================================================

_SENTINEL_1582 = -9999.0


def _da_float_1582(arr, **attrs):
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(arr.shape[0], dtype=np.float64),
                "x": np.arange(arr.shape[1], dtype=np.float64)},
        attrs=attrs,
    )


@pytest.fixture
def arr_with_sentinel_1582():
    return np.array(
        [[1.0, 2.0, _SENTINEL_1582], [3.0, _SENTINEL_1582, 5.0]],
        dtype=np.float32,
    )


def test_nodatavals_tuple_resolves_to_nodata_tag(
        tmp_path, arr_with_sentinel_1582):
    """rioxarray-style ``nodatavals`` tuple lands as the file's nodata."""
    da = _da_float_1582(arr_with_sentinel_1582,
                        crs=4326, nodatavals=(_SENTINEL_1582,))
    out = str(tmp_path / "nodatavals_tuple.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL_1582


def test_nodatavals_list_resolves_to_nodata_tag(
        tmp_path, arr_with_sentinel_1582):
    """List variant of nodatavals (some readers return list, not tuple)."""
    da = _da_float_1582(arr_with_sentinel_1582,
                        crs=4326, nodatavals=[_SENTINEL_1582])
    out = str(tmp_path / "nodatavals_list.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL_1582


def test_nodatavals_scalar_resolves_to_nodata_tag(
        tmp_path, arr_with_sentinel_1582):
    """Single-band variant where the attr is a scalar, not a sequence."""
    da = _da_float_1582(arr_with_sentinel_1582,
                        crs=4326, nodatavals=_SENTINEL_1582)
    out = str(tmp_path / "nodatavals_scalar.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL_1582


def test_fill_value_resolves_to_nodata_tag(tmp_path, arr_with_sentinel_1582):
    """CF-style ``_FillValue`` lands as the file's nodata."""
    da = _da_float_1582(arr_with_sentinel_1582,
                        crs=4326, **{"_FillValue": _SENTINEL_1582})
    out = str(tmp_path / "fillvalue.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL_1582


def test_explicit_nodata_attr_wins_over_aliases(
        tmp_path, arr_with_sentinel_1582):
    """``attrs['nodata']`` is xrspatial's canonical key. Rather than
    silently letting the canonical value win and dropping the alias, the
    writer fails closed: a DataArray with disagreeing ``nodata`` and
    ``nodatavals`` refuses to write. The explicit ``nodata=`` writer
    kwarg overrides both attrs and bypasses the check."""
    from xrspatial.geotiff import ConflictingNodataError

    da = _da_float_1582(
        arr_with_sentinel_1582, crs=4326,
        nodata=-8888.0,
        nodatavals=(_SENTINEL_1582,),
        **{"_FillValue": -7777.0},
    )
    out = str(tmp_path / "explicit_wins.tif")

    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, out)

    # Explicit kwarg overrides both attrs.
    to_geotiff(da, out, nodata=-8888.0)
    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -8888.0


def test_kwarg_nodata_wins_over_attrs(tmp_path, arr_with_sentinel_1582):
    """The ``nodata=`` keyword overrides anything in attrs."""
    da = _da_float_1582(arr_with_sentinel_1582,
                        crs=4326, nodatavals=(_SENTINEL_1582,))
    out = str(tmp_path / "kwarg_wins.tif")
    to_geotiff(da, out, nodata=-1234.0)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -1234.0


def test_nan_nodatavals_does_not_emit_tag(tmp_path):
    """NaN in nodatavals means the float NaN is already the sentinel;
    no GDAL_NODATA tag should be written for that case (mirrors
    rioxarray behaviour, and keeps the file's metadata clean)."""
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    da = _da_float_1582(arr, crs=4326, nodatavals=(float("nan"),))
    out = str(tmp_path / "nan_nodatavals.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") is None


def test_no_nodata_attrs_means_no_tag(tmp_path, arr_with_sentinel_1582):
    """Sanity guard: a DataArray with no nodata-related attrs still
    writes without a GDAL_NODATA tag."""
    da = _da_float_1582(arr_with_sentinel_1582, crs=4326)
    out = str(tmp_path / "no_nodata.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") is None


@_gpu_only
@pytest.mark.parametrize("attr_key,attr_value", [
    ("nodatavals", (_SENTINEL_1582,)),
    ("nodatavals", [_SENTINEL_1582]),
    ("_FillValue", _SENTINEL_1582),
])
def test_gpu_writer_resolves_alias(tmp_path, arr_with_sentinel_1582,
                                   attr_key, attr_value):
    """The GPU write path (write_geotiff_gpu) honours the same aliases."""
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu

    da = xr.DataArray(
        cupy.asarray(arr_with_sentinel_1582),
        dims=["y", "x"],
        coords={"y": np.arange(2, dtype=np.float64),
                "x": np.arange(3, dtype=np.float64)},
        attrs={"crs": 4326, attr_key: attr_value},
    )
    out = str(tmp_path / f"gpu_{attr_key}.tif")
    write_geotiff_gpu(da, out, compression="none")

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL_1582


# ===========================================================================
# ``attrs['masked_nodata']`` reflects whether masking ran
# ===========================================================================


def _make_float_raster_with_nodata_2092(path):
    """Float32 raster with -9999 sentinel embedded in two cells."""
    da = xr.DataArray(
        np.array(
            [[1.0, 2.0, -9999.0],
             [4.0, -9999.0, 6.0]],
            dtype=np.float32,
        ),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999.0},
    )
    to_geotiff(da, path)
    return da


def test_eager_mask_nodata_false_reports_false(tmp_path):
    """Float file + nodata=-9999 + mask_nodata=False: buffer keeps
    literal sentinel pixels; attr must say False so downstream code
    knows the sentinel is still present."""
    path = str(tmp_path / "tmp_2092_eager_unmasked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = open_geotiff(path, mask_nodata=False)
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is False, (
        f"buffer holds literal -9999 pixels but attrs say "
        f"masked_nodata={out.attrs.get('masked_nodata')!r}")
    # And the literal sentinel really is in the buffer.
    assert -9999.0 in out.values


def test_eager_mask_nodata_true_reports_true(tmp_path):
    """Default behaviour: float file + nodata=-9999 + mask_nodata=True
    replaces -9999 with NaN. Attr must say True. Pin the canonical
    direction so the fix doesn't accidentally flip it everywhere."""
    path = str(tmp_path / "tmp_2092_eager_masked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = open_geotiff(path)  # mask_nodata defaults to True
    assert out.attrs.get('nodata') == -9999.0
    assert out.attrs.get('masked_nodata') is True
    # -9999 replaced with NaN.
    assert np.isnan(out.values).sum() == 2
    assert -9999.0 not in out.values[~np.isnan(out.values)]


def test_eager_int_file_mask_nodata_true_no_match_reports_false(tmp_path):
    """Int file + mask_nodata=True but sentinel is out of range, so
    no pixel matches and no cast to float happens. Buffer stays int
    with literal sentinel; attr must say False. The pre-fix rule
    already got this right (dtype is int -> False); regression
    check that the new rule doesn't break it."""
    # uint16 file with a -9999 declared sentinel that cannot match
    # any pixel (uint16 range is 0..65535).
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999},
    )
    path = str(tmp_path / "tmp_2092_int_oor_sentinel.tif")
    to_geotiff(da, path)

    out = open_geotiff(path)
    assert out.attrs.get('nodata') == -9999
    # No pixel matched, no cast, buffer stays uint16.
    assert out.dtype.kind == 'u'
    assert out.attrs.get('masked_nodata') is False


def test_eager_explicit_float_dtype_mask_off_reports_false(tmp_path):
    """Eager path: int file + mask_nodata=False + dtype=float64.
    The mask block at __init__.py is gated on ``mask_nodata``, then
    the explicit ``dtype=`` cast promotes the int buffer to float
    with literal sentinels still in it. The rule must report False
    (mask_nodata short-circuits the conjunction). Mirrors the dask
    edge in test_dask_explicit_float_dtype_mask_off_reports_false."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': 30},
    )
    path = str(tmp_path / "tmp_2092_eager_int_to_float_unmasked.tif")
    to_geotiff(da, path)

    out = open_geotiff(path, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in out.values


def test_dask_mask_nodata_false_reports_false(tmp_path):
    path = str(tmp_path / "tmp_2092_dask_unmasked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = read_geotiff_dask(path, chunks=2, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False
    computed = out.values
    assert -9999.0 in computed


def test_dask_mask_nodata_true_reports_true(tmp_path):
    path = str(tmp_path / "tmp_2092_dask_masked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = read_geotiff_dask(path, chunks=2)
    assert out.attrs.get('masked_nodata') is True
    computed = out.values
    assert np.isnan(computed).sum() == 2


def test_dask_explicit_float_dtype_mask_off_reports_false(tmp_path):
    """Edge: caller passes ``dtype=np.float64`` on an int file with
    ``mask_nodata=False``. Old rule would have said
    ``masked_nodata=True`` (final dtype is float) even though the
    buffer holds literal sentinel values. New rule correctly says
    False because the caller opted out of masking."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': 30},
    )
    path = str(tmp_path / "tmp_2092_dask_int_to_float_unmasked.tif")
    to_geotiff(da, path)

    out = read_geotiff_dask(
        path, chunks=2, mask_nodata=False, dtype=np.float64,
    )
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    computed = out.values
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in computed


def _write_int_vrt_2092(tmp_path, src_basename, vrt_basename, sentinel=30):
    """Build a single-band VRT pointing at an int16 source TIFF.

    The VRT XML follows the working pattern from
    ``test_vrt_multiband_int_nodata_1611``: ``GeoTransform`` plus
    explicit ``SrcRect`` and ``DstRect`` are required by the
    in-repo VRT reader; without them the reader returns a zero-fill
    buffer instead of decoding the source.
    """
    tifffile = pytest.importorskip("tifffile")
    src = str(tmp_path / src_basename)
    tifffile.imwrite(src, np.array(
        [[10, 20, 30], [40, 50, 60]], dtype=np.int16,
    ), metadata=None)
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Int16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(vrt, 'w') as fh:
        fh.write(vrt_xml)
    return vrt


def test_vrt_int_source_mask_nodata_false_reports_false(tmp_path):
    """VRT with int source + mask_nodata=False: buffer stays int with
    literal sentinel; attr must say False. (Float VRT sources get
    inline NaN-masking from the VRT internal reader regardless of
    mask_nodata, so the int case is where the kwarg actually changes
    behaviour. This is the bug-class test for the VRT path.)"""
    vrt = _write_int_vrt_2092(
        tmp_path,
        "tmp_2092_vrt_src.tif",
        "tmp_2092_vrt_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i', f"expected int dtype, got {out.dtype}"
    assert out.attrs.get('masked_nodata') is False
    # The literal sentinel is still in the buffer.
    assert 30 in out.values


def test_vrt_int_source_mask_nodata_true_reports_true(tmp_path):
    """VRT with int source + mask_nodata=True: helper promotes to
    float64 and replaces sentinel pixels with NaN. Attr says True.
    Baseline direction so the fix doesn't silently flip the canonical
    behaviour."""
    vrt = _write_int_vrt_2092(
        tmp_path,
        "tmp_2092_vrt_src2.tif",
        "tmp_2092_vrt_masked.vrt",
    )
    out = open_geotiff(vrt)  # mask_nodata defaults to True
    assert out.dtype == np.float64, (
        f"expected float64 promotion, got {out.dtype}")
    assert out.attrs.get('masked_nodata') is True
    assert np.isnan(out.values).sum() == 1  # the lone 30 cell


def test_vrt_int_source_mask_off_with_float_cast_reports_false(tmp_path):
    """VRT int source + mask_nodata=False + dtype=float64 cast.

    Initial PR landed a dtype-only VRT rule (``arr.dtype.kind == 'f'``)
    which mis-claimed ``masked_nodata=True`` here -- the integer mask
    helper is skipped under ``mask_nodata=False``, then the explicit
    dtype cast promotes the int buffer to float64 with literal
    sentinels still in it. The fix reads the pre-cast dtype, so
    pre_cast is int and the attr is False. Pins the follow-up fix."""
    vrt = _write_int_vrt_2092(
        tmp_path,
        "tmp_2092_vrt_src_cast.tif",
        "tmp_2092_vrt_unmasked_cast.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    # The literal 30 is still in the float buffer (cast, not masked).
    assert 30.0 in out.values


@_gpu_only
def test_gpu_mask_nodata_false_reports_false(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2092_gpu_unmasked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = read_geotiff_gpu(path, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False


@_gpu_only
def test_gpu_mask_nodata_true_reports_true(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2092_gpu_masked.tif")
    _make_float_raster_with_nodata_2092(path)

    out = read_geotiff_gpu(path)
    assert out.attrs.get('masked_nodata') is True


# ===========================================================================
# Additive nodata lifecycle attrs: pixels_present / dtype_cast
# ===========================================================================


def _make_float_raster_2135(path, sentinel=-9999.0, plant_sentinel=True):
    """Float32 raster: 2x3 with one (or zero) sentinel pixels."""
    if plant_sentinel:
        data = np.array(
            [[1.0, 2.0, sentinel], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    else:
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    da = xr.DataArray(
        data,
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': sentinel},
    )
    to_geotiff(da, path)
    return da


def _make_int_raster_2135(path, sentinel=30, plant_sentinel=True):
    """Int16 raster with sentinel optionally embedded."""
    if plant_sentinel:
        data = np.array([[10, 20, sentinel], [40, 50, 60]], dtype=np.int16)
    else:
        data = np.array([[10, 20, 25], [40, 50, 60]], dtype=np.int16)
    da = xr.DataArray(
        data,
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': sentinel},
    )
    to_geotiff(da, path)
    return da


def test_eager_float_sentinel_present_masked(tmp_path):
    """Float file + sentinel embedded + mask_nodata=True:
    nodata_pixels_present=True, nodata_dtype_cast absent."""
    path = str(tmp_path / "tmp_2135_eager_float_present.tif")
    _make_float_raster_2135(path)
    out = open_geotiff(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True
    assert 'nodata_dtype_cast' not in out.attrs


def test_eager_float_sentinel_absent_masked(tmp_path):
    """Float file + sentinel NOT embedded + mask_nodata=True:
    nodata_pixels_present=False."""
    path = str(tmp_path / "tmp_2135_eager_float_absent.tif")
    _make_float_raster_2135(path, plant_sentinel=False)
    out = open_geotiff(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is False


def test_eager_float_sentinel_present_unmasked(tmp_path):
    """Float file + sentinel embedded + mask_nodata=False:
    masking branch skipped but presence scan still runs."""
    path = str(tmp_path / "tmp_2135_eager_float_present_unmasked.tif")
    _make_float_raster_2135(path)
    out = open_geotiff(path, mask_nodata=False)
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_int_sentinel_present(tmp_path):
    """Int file + sentinel embedded + mask_nodata=True:
    promotion fires, nodata_pixels_present=True."""
    path = str(tmp_path / "tmp_2135_eager_int_present.tif")
    _make_int_raster_2135(path)
    out = open_geotiff(path)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_int_out_of_range_sentinel(tmp_path):
    """Int (uint16) file + sentinel out of range:
    no cast, nodata_pixels_present=False."""
    da = xr.DataArray(
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
        attrs={'nodata': -9999},
    )
    path = str(tmp_path / "tmp_2135_eager_int_oor.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert out.attrs.get('nodata') == -9999
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


def test_eager_int_sentinel_present_unmasked(tmp_path):
    """Matrix row (5): int file + sentinel embedded + mask_nodata=False +
    no dtype= kwarg. Buffer stays int with literal sentinel,
    nodata_pixels_present=True from the no-mask scan branch."""
    path = str(tmp_path / "tmp_2135_eager_int_present_unmasked.tif")
    _make_int_raster_2135(path)
    out = open_geotiff(path, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is True
    assert 'nodata_dtype_cast' not in out.attrs


def test_eager_dtype_cast_records_target(tmp_path):
    """``dtype=`` kwarg surfaces as nodata_dtype_cast."""
    path = str(tmp_path / "tmp_2135_eager_dtype_cast.tif")
    _make_int_raster_2135(path)
    out = open_geotiff(path, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    # Literal sentinel still in buffer (cast, not masked).
    assert 30.0 in out.values
    # Pixel-presence scan should still confirm the sentinel is there.
    assert out.attrs.get('nodata_pixels_present') is True


def test_eager_dtype_cast_absent_without_dtype_kwarg(tmp_path):
    """No ``dtype=`` kwarg: ``nodata_dtype_cast`` absent from attrs."""
    path = str(tmp_path / "tmp_2135_eager_no_dtype.tif")
    _make_float_raster_2135(path)
    out = open_geotiff(path)
    assert 'nodata_dtype_cast' not in out.attrs


def test_eager_no_declared_sentinel(tmp_path):
    """File without GDAL_NODATA: no nodata-related attrs surface."""
    da = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        coords={'y': np.array([0.5, 1.5]), 'x': np.array([0.5, 1.5, 2.5])},
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2135_eager_no_sentinel.tif")
    to_geotiff(da, path)
    out = open_geotiff(path)
    assert 'nodata' not in out.attrs
    assert 'masked_nodata' not in out.attrs
    assert 'nodata_pixels_present' not in out.attrs
    assert 'nodata_dtype_cast' not in out.attrs


def test_dask_leaves_pixels_present_unset(tmp_path):
    """Dask path: per-chunk reduction would force eager compute, so
    ``nodata_pixels_present`` stays unset by design."""
    path = str(tmp_path / "tmp_2135_dask_present.tif")
    _make_float_raster_2135(path)
    out = read_geotiff_dask(path, chunks=2)
    assert out.attrs.get('masked_nodata') is True
    assert 'nodata_pixels_present' not in out.attrs


def test_dask_dtype_cast_records_target(tmp_path):
    """Dask path emits ``nodata_dtype_cast`` when caller passes dtype=."""
    path = str(tmp_path / "tmp_2135_dask_cast.tif")
    _make_int_raster_2135(path)
    out = read_geotiff_dask(
        path, chunks=2, mask_nodata=False, dtype=np.float64,
    )
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert 'nodata_pixels_present' not in out.attrs


def test_dask_no_dtype_cast_attr_absent(tmp_path):
    """Dask path without dtype=: nodata_dtype_cast absent."""
    path = str(tmp_path / "tmp_2135_dask_no_cast.tif")
    _make_float_raster_2135(path)
    out = read_geotiff_dask(path, chunks=2)
    assert 'nodata_dtype_cast' not in out.attrs


def _write_int_vrt_2135(tmp_path, src_basename, vrt_basename, sentinel=30,
                        plant_sentinel=True):
    tifffile = pytest.importorskip("tifffile")
    src = str(tmp_path / src_basename)
    if plant_sentinel:
        data = np.array([[10, 20, sentinel], [40, 50, 60]], dtype=np.int16)
    else:
        data = np.array([[10, 20, 25], [40, 50, 60]], dtype=np.int16)
    tifffile.imwrite(src, data, metadata=None)
    vrt = str(tmp_path / vrt_basename)
    vrt_xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Int16" band="1">
    <NoDataValue>{sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(vrt, 'w') as fh:
        fh.write(vrt_xml)
    return vrt


def test_vrt_int_sentinel_present_masked(tmp_path):
    """VRT int source + sentinel embedded + mask_nodata=True:
    helper promotes to float, nodata_pixels_present=True."""
    vrt = _write_int_vrt_2135(
        tmp_path, "tmp_2135_vrt_src.tif", "tmp_2135_vrt_present.vrt",
    )
    out = open_geotiff(vrt)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


def test_vrt_int_sentinel_absent_masked(tmp_path):
    """VRT int source + sentinel NOT embedded + mask_nodata=True:
    helper does not promote, nodata_pixels_present=False."""
    vrt = _write_int_vrt_2135(
        tmp_path, "tmp_2135_vrt_src_absent.tif",
        "tmp_2135_vrt_absent.vrt",
        plant_sentinel=False,
    )
    out = open_geotiff(vrt)
    assert out.dtype.kind == 'i'  # no promotion
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


def test_vrt_int_unmasked_still_scans(tmp_path):
    """VRT int + mask_nodata=False: presence scan still runs."""
    vrt = _write_int_vrt_2135(
        tmp_path, "tmp_2135_vrt_src_unmasked.tif",
        "tmp_2135_vrt_unmasked.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False)
    assert out.dtype.kind == 'i'
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is True


def test_vrt_dtype_cast_records_target(tmp_path):
    """VRT + dtype=float64 + mask_nodata=False: cast attr surfaces."""
    vrt = _write_int_vrt_2135(
        tmp_path, "tmp_2135_vrt_src_cast.tif",
        "tmp_2135_vrt_cast.vrt",
    )
    out = open_geotiff(vrt, mask_nodata=False, dtype=np.float64)
    assert out.dtype == np.float64
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_dtype_cast') == 'float64'
    assert out.attrs.get('nodata_pixels_present') is True


@_gpu_only
def test_gpu_float_sentinel_present_masked(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_float_present.tif")
    _make_float_raster_2135(path)
    out = read_geotiff_gpu(path)
    assert out.attrs.get('masked_nodata') is True
    assert out.attrs.get('nodata_pixels_present') is True


@_gpu_only
def test_gpu_int_sentinel_absent(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_int_absent.tif")
    _make_int_raster_2135(path, plant_sentinel=False)
    out = read_geotiff_gpu(path)
    # No sentinel pixel: helper short-circuits, buffer stays int.
    assert out.attrs.get('masked_nodata') is False
    assert out.attrs.get('nodata_pixels_present') is False


@_gpu_only
def test_gpu_dtype_cast_records_target(tmp_path):
    from xrspatial.geotiff import read_geotiff_gpu

    path = str(tmp_path / "tmp_2135_gpu_cast.tif")
    _make_int_raster_2135(path)
    out = read_geotiff_gpu(path, mask_nodata=False, dtype=np.float64)
    assert out.attrs.get('nodata_dtype_cast') == 'float64'


# ===========================================================================
# Non-finite / fractional integer sentinels are a no-op under the opt-in
# ===========================================================================


def _build_uint16_tiff_1774(nodata_str: str, tmp_path) -> str:
    """Write a minimal 2x2 uint16 TIFF with GDAL_NODATA=<nodata_str>.

    Hand-rolled rather than going through ``to_geotiff`` so the GDAL_NODATA
    tag carries arbitrary string content (``"nan"``, ``"Inf"``, etc.). The
    writer would refuse those at the resolve-nodata step before the file
    ever lands on disk.
    """
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_bytes = nodata_str.encode('ascii') + b'\x00'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag: int, val: int) -> None:
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag: int, val: int) -> None:
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_ascii(tag: int, data: bytes) -> None:
        tag_list.append((tag, 2, len(data), data))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 16)   # BitsPerSample
    add_short(259, 1)    # Compression = none
    add_short(262, 1)    # Photometric = MinIsBlack
    add_short(277, 1)    # SamplesPerPixel
    add_short(278, height)  # RowsPerStrip
    add_long(273, 0)     # StripOffsets (patched after layout)
    add_long(279, len(pixels.tobytes()))  # StripByteCounts
    add_short(339, 1)    # SampleFormat = uint
    add_ascii(42113, nodata_bytes)  # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_base = ifd_start + ifd_size
    overflow_buf = bytearray()

    processed: list[tuple[int, int, int, bytes]] = []
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            ovf_pos = overflow_base + len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
            new_raw = struct.pack(f'{bo}I', ovf_pos)
        else:
            new_raw = raw
        processed.append((tag, typ, count, new_raw))

    pixel_start = overflow_base + len(overflow_buf)
    for i, (tag, typ, count, raw) in enumerate(processed):
        if tag == 273:
            processed[i] = (tag, typ, count,
                            struct.pack(f'{bo}I', pixel_start))

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in processed:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        out.extend(raw.ljust(4, b'\x00'))
    out.extend(struct.pack(f'{bo}I', 0))  # next IFD = 0
    out.extend(overflow_buf)
    out.extend(pixels.tobytes())

    path = str(tmp_path / f'uint16_nodata_{nodata_str.replace("-", "neg")}.tif')
    with open(path, 'wb') as f:
        f.write(bytes(out))
    return path


@pytest.mark.parametrize('nodata_str', ['nan', 'NaN', 'NAN'])
def test_open_geotiff_eager_int_nodata_nan(tmp_path, nodata_str):
    """Eager numpy path: NaN nodata on uint16 file is a no-op under the
    ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774(nodata_str, tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    # No pixel can match NaN, so the dtype stays uint16
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    # The raw sentinel survives on attrs so write round-trips keep the tag
    assert 'nodata' in da.attrs
    assert np.isnan(da.attrs['nodata'])


@pytest.mark.parametrize('nodata_str', ['inf', 'Inf', 'INF',
                                        '-inf', '-Inf', '-INF'])
def test_open_geotiff_eager_int_nodata_inf(tmp_path, nodata_str):
    """Eager numpy path: +/-Inf nodata on uint16 file is a no-op under
    the ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774(nodata_str, tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    assert 'nodata' in da.attrs
    assert np.isinf(da.attrs['nodata'])


def test_open_geotiff_eager_int_nodata_finite_still_masks(tmp_path):
    """Regression guard: in-range finite sentinel still masks correctly."""
    # 30 is one of the pixel values; using it as a sentinel masks one pixel.
    path = _build_uint16_tiff_1774('30', tmp_path)
    da = open_geotiff(path)
    # uint16 + in-range sentinel hit promotes to float64 with NaN
    assert da.dtype == np.float64
    assert np.isnan(da.values[1, 0])
    assert da.values[0, 0] == 10
    assert da.attrs['nodata'] == 30


def test_read_geotiff_dask_int_nodata_nan(tmp_path):
    """Dask path: NaN nodata on uint16 file is a no-op under the
    ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774('nan', tmp_path)
    da = read_geotiff_dask(path, chunks=2, allow_invalid_nodata=True)
    # effective_dtype stays uint16 because the sentinel is non-finite
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])
    assert 'nodata' in da.attrs
    assert np.isnan(da.attrs['nodata'])


def test_read_geotiff_dask_int_nodata_inf(tmp_path):
    """Dask path: Inf nodata on uint16 file is a no-op under the
    ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774('inf', tmp_path)
    da = read_geotiff_dask(path, chunks=2, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])
    assert np.isinf(da.attrs['nodata'])


@_gpu_only
def test_apply_nodata_mask_gpu_int_nan_noop():
    """GPU helper: NaN nodata on uint16 array is a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, float('nan'))
    # No promotion, same buffer back
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])


@_gpu_only
def test_apply_nodata_mask_gpu_int_inf_noop():
    """GPU helper: Inf nodata on uint16 array is a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, float('inf'))
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])


@_gpu_only
def test_apply_nodata_mask_gpu_int_finite_still_masks():
    """GPU helper regression guard: in-range finite sentinel still masks."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, 3)
    # 3 is in range and hits a pixel; promotes to float64 with NaN
    assert out.dtype == cupy.float64
    arr = out.get()
    assert np.isnan(arr[1, 0])
    assert arr[0, 0] == 1.0


@pytest.mark.parametrize('nodata_str', ['3.5', '29.5', '0.5'])
def test_open_geotiff_eager_int_nodata_fractional_noop(tmp_path, nodata_str):
    """Eager numpy path: fractional nodata on uint16 is a no-op under the
    ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774(nodata_str, tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])
    assert da.attrs['nodata'] == float(nodata_str)


def test_open_geotiff_eager_int_nodata_fractional_does_not_alias_truncation(
    tmp_path,
):
    """A ``"30.5"`` sentinel must not mask the real pixel value 30
    (which is in the test image). ``int(30.5)`` would truncate to 30
    without the integerness gate. Runs under the ``allow_invalid_nodata``
    opt-in.
    """
    path = _build_uint16_tiff_1774('30.5', tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    # pixel @[1,0] is 30; the fractional sentinel must NOT have masked it
    assert da.values[1, 0] == 30
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


def test_read_geotiff_dask_int_nodata_fractional_noop(tmp_path):
    """Dask path: fractional nodata on uint16 is a no-op under the
    ``allow_invalid_nodata`` opt-in.
    """
    path = _build_uint16_tiff_1774('30.5', tmp_path)
    da = read_geotiff_dask(path, chunks=2, allow_invalid_nodata=True)
    # effective_dtype stays uint16 because the sentinel is fractional
    assert da.dtype == np.uint16
    computed = da.compute().values
    assert computed[1, 0] == 30
    np.testing.assert_array_equal(computed, [[10, 20], [30, 40]])
    assert da.attrs['nodata'] == 30.5


@_gpu_only
def test_apply_nodata_mask_gpu_int_fractional_noop():
    """GPU helper: fractional nodata on uint16 is a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.uint16))
    out = _apply_nodata_mask_gpu(arr_gpu, 3.5)
    # 3.5 cannot match any uint16 pixel; ``int(3.5) == 3`` would have
    # truncated and masked the real pixel value 3.
    assert out.dtype == cupy.uint16
    np.testing.assert_array_equal(out.get(), [[1, 2], [3, 4]])


# ===========================================================================
# Dropped defensive copies on the read path do not alias buffers
# ===========================================================================


def _make_float_with_sentinel_1553(h=24, w=24, sentinel=-9999.0,
                                   dtype=np.float32):
    """Float array with a few sentinel pixels at known positions."""
    rng = np.random.default_rng(seed=1553)
    arr = rng.uniform(0.0, 100.0, size=(h, w)).astype(dtype)
    arr[0, 0] = sentinel
    arr[5, 7] = sentinel
    arr[-1, -1] = sentinel
    return arr


def _make_uint16_with_sentinel_1553(h=24, w=24, sentinel=65535):
    """uint16 array with a few sentinel pixels at known positions."""
    rng = np.random.default_rng(seed=1554)
    arr = rng.integers(0, 1000, size=(h, w), dtype=np.uint16)
    arr[0, 0] = sentinel
    arr[5, 7] = sentinel
    arr[-1, -1] = sentinel
    return arr


def test_float_sentinel_strip_tiff_read(tmp_path):
    """Strip-organized float32 TIFF with sentinel nodata reads correctly."""
    src = _make_float_with_sentinel_1553()
    path = str(tmp_path / 'issue_1553_float_strip.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=False,
               compression='deflate')

    out = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(out.data[finite], src[finite])
    assert out.attrs.get('nodata') == -9999.0


def test_float_sentinel_tiled_tiff_read(tmp_path):
    """Tiled float32 TIFF with sentinel nodata reads correctly."""
    src = _make_float_with_sentinel_1553(h=64, w=64)
    path = str(tmp_path / 'issue_1553_float_tiled.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(out.data[finite], src[finite])


def test_uint16_sentinel_tiled_tiff_read(tmp_path):
    """Tiled uint16 TIFF with sentinel nodata is promoted to float+NaN."""
    src = _make_uint16_with_sentinel_1553(h=48, w=48)
    path = str(tmp_path / 'issue_1553_uint16_tiled.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': 65535})
    to_geotiff(da, path, nodata=65535, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    expected_mask = (src == 65535)
    np.testing.assert_array_equal(np.isnan(out.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_array_equal(out.data[finite].astype(np.uint16),
                                  src[finite])


def test_repeat_reads_independent(tmp_path):
    """Repeated reads return independent arrays with the correct mask.

    If the dropped ``.copy()`` had been protecting against shared
    state across reads, the second read would either see corrupted
    values (NaN bleeding into other slots) or share a buffer with the
    first call. We mutate the first result and then re-read to verify
    the file's contents are not affected.
    """
    src = _make_float_with_sentinel_1553()
    path = str(tmp_path / 'issue_1553_repeat_reads.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    first = open_geotiff(path)
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(first.data), expected_mask)

    # Mutate the first read in place. If subsequent reads share state
    # with this buffer, the second read will look corrupted.
    first.data[1, 1] = np.nan
    first.data[2, 2] = 12345.0

    second = open_geotiff(path)
    np.testing.assert_array_equal(np.isnan(second.data), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(second.data[finite], src[finite])


def test_dask_chunked_float_sentinel_read(tmp_path):
    """Dask-chunked read of a float TIFF with sentinel nodata.

    Exercises ``_delayed_read_window`` -- the second site where the
    defensive copy was dropped. Each chunk runs the per-window nodata
    rewrite path in a worker task.
    """
    src = _make_float_with_sentinel_1553(h=64, w=64)
    path = str(tmp_path / 'issue_1553_dask_float.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path, chunks=16)
    materialised = out.compute().data
    expected_mask = (src == np.float32(-9999.0))
    np.testing.assert_array_equal(np.isnan(materialised), expected_mask)
    finite = ~expected_mask
    np.testing.assert_allclose(materialised[finite], src[finite])


def test_dask_chunked_uint16_sentinel_read(tmp_path):
    """Dask-chunked read of a uint16 TIFF promotes to float+NaN per chunk."""
    src = _make_uint16_with_sentinel_1553(h=64, w=64)
    path = str(tmp_path / 'issue_1553_dask_uint16.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': 65535})
    to_geotiff(da, path, nodata=65535, tiled=True, tile_size=16,
               compression='deflate')

    out = open_geotiff(path, chunks=16)
    materialised = out.compute().data
    assert materialised.dtype.kind == 'f'
    expected_mask = (src == 65535)
    np.testing.assert_array_equal(np.isnan(materialised), expected_mask)
    finite = ~expected_mask
    np.testing.assert_array_equal(materialised[finite].astype(np.uint16),
                                  src[finite])


def test_writer_does_not_mutate_caller_input(tmp_path):
    """``to_geotiff`` must not mutate the caller's input array.

    Covers the defensive copy at the ``to_geotiff`` entry; the sibling
    test below covers the second kept copy in ``_write_single_tile``
    (the .vrt tiled-output path).
    """
    src = np.array([
        [1.0, 2.0, np.nan],
        [np.nan, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)
    snapshot = src.copy()
    path = str(tmp_path / 'issue_1553_writer_no_mutate.tif')
    da = xr.DataArray(src, dims=('y', 'x'),
                      attrs={'crs': 4326, 'nodata': -9999.0})
    to_geotiff(da, path, nodata=-9999.0, compression='deflate')

    # Caller's buffer must still hold its original NaNs and finite
    # values; the writer must not have stamped the sentinel value into
    # the user's array in place.
    np.testing.assert_array_equal(np.isnan(src), np.isnan(snapshot))
    finite = ~np.isnan(snapshot)
    np.testing.assert_array_equal(src[finite], snapshot[finite])


def test_write_single_tile_does_not_mutate_caller_input(tmp_path):
    """``_write_single_tile`` must not mutate the caller's array either.

    The .vrt tiled-output path goes through ``_write_single_tile`` per
    chunk. That helper has its own defensive copy at the
    NaN -> sentinel rewrite. Direct invocation pins it: pass a numpy
    buffer with NaNs, request an integer nodata sentinel, then assert
    the source still has NaNs in the same places.
    """
    from xrspatial.geotiff import _write_single_tile

    src = np.array([
        [1.0, np.nan, 3.0],
        [4.0, 5.0, np.nan],
        [np.nan, 8.0, 9.0],
    ], dtype=np.float32)
    snapshot = src.copy()

    out_path = str(tmp_path / 'issue_1553_single_tile_no_mutate.tif')
    _write_single_tile(
        src, out_path,
        geo_transform=None, epsg=4326, wkt=None,
        nodata=-9999.0, compression='deflate', compression_level=None,
        tile_size=16, predictor=False, bigtiff=False,
    )

    # Source must still hold its original NaNs and finite values.
    np.testing.assert_array_equal(np.isnan(src), np.isnan(snapshot))
    finite = ~np.isnan(snapshot)
    np.testing.assert_array_equal(src[finite], snapshot[finite])


# ===========================================================================
# band_nodata forwarding through the shared validation helpers
# ===========================================================================


class _FakeTransform2210:
    def __init__(self, origin_x=0.0, origin_y=10.0,
                 pixel_width=1.0, pixel_height=-1.0,
                 rotated_affine=None):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.rotated_affine = rotated_affine


class _FakeGeoInfo2210:
    def __init__(self, *, crs_wkt='EPSG:4326', has_georef=True):
        self.transform = _FakeTransform2210()
        self.crs_epsg = 4326
        self.crs_wkt = crs_wkt
        self.raster_type = 1
        self.has_georef = has_georef
        self.nodata = None
        self.extra_tags = None
        self.image_description = None
        self.extra_samples = None
        self.gdal_metadata = None
        self.gdal_metadata_xml = None
        self.x_resolution = None
        self.y_resolution = None
        self.resolution_unit = None


def test_validate_helper_signature_includes_band_nodata_kwargs():
    sig = inspect.signature(_validate_read_geo_info)
    assert 'band_nodata' in sig.parameters
    assert 'band_nodata_values' in sig.parameters
    # Keyword-only so the wave-1 positional signature stays stable.
    assert sig.parameters['band_nodata'].kind == inspect.Parameter.KEYWORD_ONLY
    assert (sig.parameters['band_nodata_values'].kind
            == inspect.Parameter.KEYWORD_ONLY)


def test_lazy_finalize_signature_includes_band_nodata_kwargs():
    sig = inspect.signature(_finalize_lazy_read_attrs)
    assert 'band_nodata' in sig.parameters
    assert 'band_nodata_values' in sig.parameters
    assert sig.parameters['band_nodata'].kind == inspect.Parameter.KEYWORD_ONLY


def test_validate_helper_rejects_mixed_band_sentinels():
    gi = _FakeGeoInfo2210()

    with pytest.raises(MixedBandMetadataError):
        _validate_read_geo_info(
            gi,
            band_nodata=None,
            band_nodata_values=[-9999.0, 0.0],
        )


def test_validate_helper_accepts_single_sentinel_band_list():
    # All bands carry the same sentinel -- the check passes.
    gi = _FakeGeoInfo2210()

    _validate_read_geo_info(
        gi,
        band_nodata=None,
        band_nodata_values=[-9999.0, -9999.0, -9999.0],
    )


def test_validate_helper_band_nodata_first_opts_out():
    # ``band_nodata='first'`` keeps the legacy flatten-to-first-band
    # behaviour explicitly even when the bands disagree.
    gi = _FakeGeoInfo2210()

    _validate_read_geo_info(
        gi,
        band_nodata='first',
        band_nodata_values=[-9999.0, 0.0],
    )


def test_validate_helper_omits_band_kwargs_short_circuits():
    # Non-VRT callers omit the kwargs entirely; the check short-circuits
    # because ``band_nodata_values`` defaults to ``None`` (falsy).
    gi = _FakeGeoInfo2210()

    _validate_read_geo_info(gi)  # no kwargs -- no raise


def test_lazy_finalize_routes_band_nodata_through_validator():
    # Mixed-band sentinels routed through the lazy finalizer surface as
    # the same error class the VRT pre-read inline call raises, so the
    # helper-routed post-read check is not a no-op.
    gi = _FakeGeoInfo2210()

    with pytest.raises(MixedBandMetadataError):
        _finalize_lazy_read_attrs(
            geo_info=gi,
            nodata=None,
            mask_nodata=False,
            graph_dtype='float64',
            window=None,
            band_nodata=None,
            band_nodata_values=[-9999.0, 0.0],
        )


def test_lazy_finalize_band_nodata_first_opts_out():
    # Same as above with the explicit opt-in to legacy semantics. The
    # finalizer returns attrs without raising.
    gi = _FakeGeoInfo2210()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        window=None,
        band_nodata='first',
        band_nodata_values=[-9999.0, 0.0],
    )
    assert attrs['georef_status'] == 'full'


def test_lazy_finalize_without_band_kwargs_unchanged():
    # The default kwargs are backward-compatible: no mixed-band dispatch
    # is meaningful because ``band_nodata_values`` is None.
    gi = _FakeGeoInfo2210()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        window=None,
    )
    assert attrs['georef_status'] == 'full'
    assert attrs['nodata'] == -9999


# ===========================================================================
# NodataLifecycle decision contract + cross-backend parity
# ===========================================================================

class TestRawSentinelExposure:
    """``raw_sentinel`` is the declared value, not the inverted one."""

    def test_returns_declared_when_no_photometric(self):
        lc = NodataLifecycle(declared=-9999, dtype_in=np.dtype("int16"))
        assert lc.raw_sentinel == -9999

    def test_returns_declared_under_miniswhite(self):
        # Even when MinIsWhite would invert the effective sentinel, the
        # raw sentinel must remain the original (so attrs['nodata']
        # preserves the on-disk value for write round-trip).
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        assert lc.raw_sentinel == 10

    def test_none_passes_through(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.raw_sentinel is None


class TestEffectiveSentinelUnderMinIsWhite:
    """``effective_sentinel`` mirrors ``_reader._miniswhite_inverted_nodata``."""

    def test_uint_inverts_to_iinfo_max_minus_value(self):
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        # 255 - 10 = 245 (TIFF6 MinIsWhite semantics)
        assert lc.effective_sentinel == 245

    def test_float_inverts_to_negation(self):
        lc = NodataLifecycle(
            declared=3.5,
            photometric=0,
            dtype_in=np.dtype("float32"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == -3.5

    def test_nan_passes_unchanged(self):
        lc = NodataLifecycle(
            declared=float("nan"),
            photometric=0,
            dtype_in=np.dtype("float32"),
            samples_per_pixel=1,
        )
        assert np.isnan(lc.effective_sentinel)

    def test_multiband_photometric_zero_is_not_miniswhite(self):
        # Multi-band photometric=0 is a different TIFF case from
        # single-band MinIsWhite; the reader's gate is
        # ``photometric == 0 AND samples_per_pixel == 1``.
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=3,
        )
        assert lc.effective_sentinel == 10

    def test_photometric_one_minisblack_no_inversion(self):
        lc = NodataLifecycle(
            declared=10,
            photometric=1,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == 10

    def test_out_of_range_sentinel_stays_unchanged(self):
        # uint16 + nodata=-9999: cannot be represented, mirror reader.
        lc = NodataLifecycle(
            declared=-9999,
            photometric=0,
            dtype_in=np.dtype("uint16"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == -9999


class TestSentinelFitsBuffer:
    """``sentinel_fits_buffer`` collapses finite + integer + in-range gates."""

    def test_int_in_range(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is True

    def test_int_out_of_range(self):
        # uint16 + -9999: cannot match.
        lc = NodataLifecycle(declared=-9999, dtype_in=np.dtype("uint16"))
        assert lc.sentinel_fits_buffer is False

    def test_int_fractional(self):
        # 3.5 on uint16 -> int(3.5) would truncate to 3 and false-flag
        # a real pixel value; helper must reject it.
        lc = NodataLifecycle(declared=3.5, dtype_in=np.dtype("uint16"))
        assert lc.sentinel_fits_buffer is False

    def test_int_non_finite(self):
        lc = NodataLifecycle(declared=float("nan"), dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is False
        lc2 = NodataLifecycle(declared=float("inf"), dtype_in=np.dtype("uint8"))
        assert lc2.sentinel_fits_buffer is False

    def test_float_dtype_always_fits(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.sentinel_fits_buffer is True
        lc2 = NodataLifecycle(
            declared=float("nan"), dtype_in=np.dtype("float32"),
        )
        assert lc2.sentinel_fits_buffer is True

    def test_none_does_not_fit(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is False


class TestMaskingOccurredDecision:
    """``masking_occurred`` mirrors the call sites' attr-stamping rule."""

    def test_no_sentinel_returns_false(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=True) is False

    def test_mask_nodata_false_returns_false(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=False) is False

    def test_float_buffer_with_sentinel_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=True) is True

    def test_integer_buffer_returns_false(self):
        # Integer buffer surviving means no promotion -> no pixels masked.
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.masking_occurred(mask_nodata=True) is False

    def test_explicit_float_cast_returns_true(self):
        # Caller cast to float, so the final buffer is float and the
        # mask-occurred attr stays True.
        lc = NodataLifecycle(
            declared=0,
            dtype_in=np.dtype("uint8"),
            dtype_request=np.dtype("float64"),
        )
        assert lc.masking_occurred(mask_nodata=True) is True


class TestDtypeCastDecision:
    """``dtype_cast_occurred`` / ``cast_dtype_name`` reflect caller intent."""

    def test_no_request_returns_false(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.dtype_cast_occurred is False
        assert lc.cast_dtype_name is None

    def test_request_records_name(self):
        lc = NodataLifecycle(
            declared=0,
            dtype_in=np.dtype("uint8"),
            dtype_request="float64",
        )
        assert lc.dtype_cast_occurred is True
        assert lc.cast_dtype_name == "float64"


class TestWriterRestoreSentinelDecision:
    """``writer_restore_sentinel`` mirrors the in-line write-side gate."""

    def test_no_declared_returns_false(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is False

    def test_non_float_buffer_returns_false(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("uint8"),
        ) is False

    def test_nan_sentinel_returns_false(self):
        lc = NodataLifecycle(
            declared=float("nan"), dtype_in=np.dtype("float32"),
        )
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is False

    def test_finite_float_sentinel_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is True

    def test_masked_nodata_attr_false_returns_false(self):
        # Literal False on attrs['masked_nodata'] tells the writer the
        # read path did NOT mask, so the in-buffer NaN must NOT be
        # rewritten to the sentinel.
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=False,
        ) is False

    def test_masked_nodata_attr_true_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=True,
        ) is True

    def test_masked_nodata_attr_none_defaults_true(self):
        # Attr absent -> default (True).
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=None,
        ) is True

    def test_restore_sentinel_false_short_circuits(self):
        # ``restore_sentinel=False`` caller opt-out (the cached attr
        # answer from ``_should_restore_nan_sentinel``) must win over
        # the otherwise-True default path.
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            restore_sentinel=False,
        ) is False

    def test_restore_sentinel_true_is_default(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        # Explicit True matches the default (kwarg omitted).
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            restore_sentinel=True,
        ) is True


class TestInvertForMinIsWhiteLockstep:
    """``_invert_for_miniswhite`` mirrors ``_reader._miniswhite_inverted_nodata``.

    The two functions intentionally duplicate the inversion math so the
    lifecycle helper does not pull ``_reader`` into its dependency
    graph. Sweep a wide sentinel / dtype grid here so drift between the
    two surfaces fast.
    """

    def test_grid_matches_reader_helper(self):
        from xrspatial.geotiff._nodata import _invert_for_miniswhite
        from xrspatial.geotiff._reader import _miniswhite_inverted_nodata

        class _IFDStub:
            photometric = 0
            samples_per_pixel = 1

        sentinels = [
            0, 1, 10, 127, 128, 255,        # uint8 range
            256, 1000, 32767, 65535,        # uint16 range
            -1, -9999,                      # out of unsigned range
            0.0, 1.5, 3.5, -3.5,            # float values
            float("nan"), float("inf"),     # non-finite
        ]
        dtypes = [
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("int16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ]
        for s in sentinels:
            for d in dtypes:
                ref = _miniswhite_inverted_nodata(s, _IFDStub, d)
                got = _invert_for_miniswhite(s, d)
                if isinstance(ref, float) and np.isnan(ref):
                    assert isinstance(got, float) and np.isnan(got), (
                        f"sentinel={s} dtype={d}: ref=NaN, got={got!r}"
                    )
                else:
                    assert ref == got, (
                        f"sentinel={s} dtype={d}: ref={ref!r}, got={got!r}"
                    )


class TestPixelsPresentSlot:
    """``pixels_present`` is a settable slot; constructor leaves it None."""

    def test_default_none(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.pixels_present is None

    def test_settable(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        lc.pixels_present = True
        assert lc.pixels_present is True
        lc.pixels_present = False
        assert lc.pixels_present is False


# Cross-backend parity through the public API


def _make_int_raster_2211(path, sentinel=255, dtype=np.uint8, plant=True):
    """2x3 uint raster with the sentinel planted at (0, 2) when ``plant``."""
    if plant:
        data = np.array(
            [[1, 2, sentinel], [4, 5, 6]], dtype=dtype,
        )
    else:
        data = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=dtype,
        )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": sentinel},
    )
    to_geotiff(da, path, nodata=sentinel)
    return path


def _make_float_raster_2211(path, sentinel=-9999.0, plant=True):
    if plant:
        data = np.array(
            [[1.0, 2.0, sentinel], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    else:
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": sentinel},
    )
    to_geotiff(da, path, nodata=sentinel)
    return path


@pytest.fixture
def int_tif(tmp_path):
    return _make_int_raster_2211(str(tmp_path / "int_sentinel_2226.tif"))


@pytest.fixture
def float_tif(tmp_path):
    return _make_float_raster_2211(str(tmp_path / "float_sentinel_2226.tif"))


@pytest.fixture
def nan_tif(tmp_path):
    return _make_float_raster_2211(
        str(tmp_path / "nan_sentinel_2226.tif"),
        sentinel=float("nan"),
        plant=False,
    )


@pytest.fixture
def oor_tif(tmp_path):
    # uint16 + -9999: classic out-of-range sentinel that cannot match
    # any pixel; the reader still surfaces it on attrs['nodata'].
    path = str(tmp_path / "uint16_oor_2226.tif")
    return _make_int_raster_2211(
        path, sentinel=-9999, dtype=np.uint16, plant=False,
    )


class TestIntegerSentinelParity:
    def test_eager_masks_int_sentinel_to_nan(self, int_tif):
        da = open_geotiff(int_tif)
        # int sentinel auto-promotes to float when at least one pixel
        # matches, leaving NaN where the sentinel was.
        assert da.dtype.kind == "f"
        assert np.isnan(da.data[0, 2])
        assert da.attrs["nodata"] == 255
        assert da.attrs["masked_nodata"] is True

    def test_dask_matches_eager(self, int_tif):
        eager = open_geotiff(int_tif)
        lazy = read_geotiff_dask(int_tif, chunks=2)
        # Same on-disk sentinel propagated.
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]
        # Same lifecycle decision: dask graph promoted to float64.
        assert lazy.dtype.kind == "f"
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )

    @_gpu_only
    def test_gpu_matches_eager(self, int_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        eager = open_geotiff(int_tif)
        gpu = read_geotiff_gpu(int_tif)
        assert gpu.attrs["nodata"] == eager.attrs["nodata"]
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(gpu.data.get()),
        )


class TestFloatSentinelParity:
    def test_eager(self, float_tif):
        da = open_geotiff(float_tif)
        assert da.dtype == np.float32
        assert np.isnan(da.data[0, 2])
        assert da.attrs["nodata"] == -9999.0
        assert da.attrs["masked_nodata"] is True

    def test_dask(self, float_tif):
        eager = open_geotiff(float_tif)
        lazy = read_geotiff_dask(float_tif, chunks=2)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]

    @_gpu_only
    def test_gpu(self, float_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        eager = open_geotiff(float_tif)
        gpu = read_geotiff_gpu(float_tif)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(gpu.data.get()),
        )


class TestNaNSentinelParity:
    def test_eager(self, nan_tif):
        da = open_geotiff(nan_tif)
        # NaN sentinel: float buffer keeps NaN as the sentinel; no
        # integer-to-float promotion because the source was already
        # float, and mask sees no literal pixel match.
        assert da.dtype == np.float32
        assert np.isnan(da.attrs["nodata"])

    def test_dask_matches(self, nan_tif):
        lazy = read_geotiff_dask(nan_tif, chunks=2)
        out = lazy.compute()
        # Source had no NaN pixels planted, so the float buffer carries
        # the original values.
        assert out.dtype == np.float32
        assert np.isnan(out.attrs["nodata"])


class TestOutOfRangeSentinelParity:
    def test_eager_keeps_int_dtype_and_records_sentinel(self, oor_tif):
        da = open_geotiff(oor_tif)
        # uint16 + -9999 cannot match any pixel; buffer stays integer.
        assert da.dtype == np.uint16
        # Sentinel still recorded for write round-trip.
        assert int(da.attrs["nodata"]) == -9999

    def test_dask_matches(self, oor_tif):
        eager = open_geotiff(oor_tif)
        lazy = read_geotiff_dask(oor_tif, chunks=2)
        assert lazy.dtype == eager.dtype
        np.testing.assert_array_equal(
            eager.data, lazy.compute().data,
        )
        assert int(lazy.attrs["nodata"]) == int(eager.attrs["nodata"])


class TestMinIsWhiteSentinelInversion:
    """Direct lifecycle check + downstream usage on a hand-built file.

    Building a real MinIsWhite file through ``to_geotiff`` is supported
    (``photometric='miniswhite'``); we use that to drive an end-to-end
    parity check across eager + dask. The lifecycle helper's
    ``effective_sentinel`` is what the per-chunk masking compares
    against, so a planted sentinel pixel must mask correctly after
    inversion.
    """

    def _build(self, path, sentinel=10):
        data = np.array(
            [[1, 2, sentinel], [4, 5, 6]], dtype=np.uint8,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
            attrs={"nodata": sentinel},
        )
        to_geotiff(da, path, nodata=sentinel, photometric="miniswhite")

    def test_eager_masks_inverted_sentinel(self, tmp_path):
        path = str(tmp_path / "miw_2226.tif")
        self._build(path)
        da = open_geotiff(path)
        # The MinIsWhite writer pre-inverts both pixels AND the sentinel
        # (see ``_writer._invert_nodata_for_miniswhite``), so the on-disk
        # GDAL_NODATA tag stores 245 = 255 - 10. The reader's MinIsWhite
        # inversion runs on read; the reader's downstream mask compares
        # against the post-inversion sentinel (10 again) and rewrites the
        # planted pixel to NaN. The lifecycle helper's contract requires
        # the same effective_sentinel resolution.
        assert np.isnan(da.data[0, 2])
        # attrs['nodata'] carries the on-disk value (inverted by the
        # writer); this is the documented round-trip behaviour.
        assert int(da.attrs["nodata"]) == 245

    def test_dask_matches_eager(self, tmp_path):
        path = str(tmp_path / "miw_dask_2226.tif")
        self._build(path)
        eager = open_geotiff(path)
        lazy = read_geotiff_dask(path, chunks=2)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]

    def test_lifecycle_effective_sentinel_matches_reader_helper(self):
        from xrspatial.geotiff._reader import _miniswhite_inverted_nodata

        # Synthesize an IFD-like object for the legacy helper. Only
        # ``photometric`` and ``samples_per_pixel`` are read; build a
        # minimal stand-in rather than constructing a real IFD.
        class _IFDStub:
            photometric = 0
            samples_per_pixel = 1

        for sentinel, dtype in [
            (10, np.dtype("uint8")),
            (-9999, np.dtype("uint16")),  # out of range -> unchanged
            (3.5, np.dtype("float32")),
            (float("nan"), np.dtype("float32")),
        ]:
            ref = _miniswhite_inverted_nodata(sentinel, _IFDStub, dtype)
            lc = NodataLifecycle(
                declared=sentinel,
                photometric=0,
                dtype_in=dtype,
                samples_per_pixel=1,
            )
            if isinstance(ref, float) and np.isnan(ref):
                assert np.isnan(lc.effective_sentinel)
            else:
                assert lc.effective_sentinel == ref, (
                    f"sentinel={sentinel} dtype={dtype}: ref={ref}, "
                    f"lifecycle={lc.effective_sentinel}"
                )


class TestMaskNodataFalseParity:
    def test_eager_keeps_literal_sentinel(self, int_tif):
        da = open_geotiff(int_tif, mask_nodata=False)
        # Buffer keeps integer dtype + literal sentinel pixel (255).
        assert da.dtype == np.uint8
        assert int(da.data[0, 2]) == 255
        # Lifecycle attr says masking did NOT occur.
        assert da.attrs["masked_nodata"] is False

    def test_dask_keeps_literal_sentinel(self, int_tif):
        lazy = read_geotiff_dask(int_tif, chunks=2, mask_nodata=False)
        out = lazy.compute()
        assert out.dtype == np.uint8
        assert int(out.data[0, 2]) == 255
        assert lazy.attrs["masked_nodata"] is False

    @_gpu_only
    def test_gpu_keeps_literal_sentinel(self, int_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        gpu = read_geotiff_gpu(int_tif, mask_nodata=False)
        host = gpu.data.get()
        assert host.dtype == np.uint8
        assert int(host[0, 2]) == 255
        assert gpu.attrs["masked_nodata"] is False


class TestExplicitDtypeRequestParity:
    def test_eager_records_dtype_cast(self, int_tif):
        da = open_geotiff(int_tif, dtype="float64")
        assert da.dtype == np.float64
        assert da.attrs.get("nodata_dtype_cast") == "float64"

    def test_dask_records_dtype_cast(self, int_tif):
        lazy = read_geotiff_dask(int_tif, chunks=2, dtype="float64")
        assert lazy.dtype == np.float64
        assert lazy.attrs.get("nodata_dtype_cast") == "float64"


@pytest.fixture
def simple_vrt(tmp_path):
    """A trivial VRT wrapping a single GeoTIFF with a -9999 sentinel."""
    import os

    src = str(tmp_path / "vrt_src_2226.tif")
    data = np.array(
        [[1.0, 2.0, -9999.0], [4.0, 5.0, 6.0]], dtype=np.float32,
    )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": -9999.0},
    )
    to_geotiff(da, src, nodata=-9999.0)

    vrt = str(tmp_path / "vrt_eager_2226.vrt")
    xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>  0.0,  1.0,  0.0,  0.0,  0.0, 1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(src)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(vrt, "w") as f:
        f.write(xml)
    return vrt


class TestVRTEagerParity:
    def test_vrt_masks_sentinel_to_nan(self, simple_vrt):
        da = read_vrt(simple_vrt)
        assert da.dtype.kind == "f"
        assert np.isnan(da.data[0, 2])

    def test_vrt_mask_nodata_false_keeps_literal(self, simple_vrt):
        da = read_vrt(simple_vrt, mask_nodata=False)
        # Literal -9999 survives in the float buffer.
        assert da.data[0, 2] == -9999.0
        assert da.attrs.get("masked_nodata") is False


class TestWriterRestoreParity:
    """Round-trip a float raster with NaN pixels through the writer.

    The lifecycle's ``writer_restore_sentinel`` controls whether NaN
    pixels are rewritten to the integer / float sentinel on disk. The
    eager and (when available) GPU writer must agree.
    """

    def test_eager_restores_nan_to_sentinel(self, tmp_path):
        path = str(tmp_path / "restore_eager_2226.tif")
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
        )
        to_geotiff(da, path, nodata=-9999.0)
        # Read back with mask_nodata=False so we can see the literal
        # on-disk byte value the restore step planted.
        readback = open_geotiff(path, mask_nodata=False)
        assert readback.data[0, 1] == -9999.0

    def test_masked_nodata_false_attr_blocks_restore(self, tmp_path):
        # The reader stores masked_nodata=False to opt out of the
        # writer's NaN->sentinel rewrite. The lifecycle's
        # writer_restore_sentinel reads this through the
        # ``restore_sentinel`` kwarg the writer threads in.
        path = str(tmp_path / "no_restore_2226.tif")
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
            attrs={"nodata": -9999.0, "masked_nodata": False},
        )
        to_geotiff(da, path)
        readback = open_geotiff(path, mask_nodata=False)
        # Restore step skipped, so the NaN survives as on-disk NaN.
        assert np.isnan(readback.data[0, 1])

    @_gpu_only
    def test_gpu_writer_matches_eager(self, tmp_path):
        import cupy

        path = str(tmp_path / "restore_gpu_2226.tif")
        data = cupy.asarray(np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        ))
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
        )
        to_geotiff(da, path, nodata=-9999.0, gpu=True)
        readback = open_geotiff(path, mask_nodata=False)
        assert readback.data[0, 1] == -9999.0


# ===========================================================================
# nodata vs masked_nodata split-attrs contract
# ===========================================================================
#
# ``attrs['nodata']`` was historically overloaded as both "the file
# declared this sentinel" and "the reader already replaced sentinel
# pixels with NaN." Those are split into ``nodata`` (declared) and
# ``masked_nodata`` (whether the in-memory array was NaN-masked).
# The rasterio-backed helpers below import rasterio lazily so the rest
# of this module still collects when rasterio is absent.

_SENTINEL_1988 = -9999.0


def _write_float_tiff_1988(path: str, *, with_sentinel: bool) -> None:
    """Write a 4x4 float32 TIFF with or without a declared nodata sentinel."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    arr = np.array(
        [[1.0, 2.0, _SENTINEL_1988, 4.0],
         [5.0, _SENTINEL_1988, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    kw = dict(
        driver="GTiff", height=4, width=4, count=1, dtype="float32",
        transform=from_origin(0, 4, 1, 1), crs="EPSG:4326",
    )
    if with_sentinel:
        kw["nodata"] = _SENTINEL_1988
    with rasterio.open(path, "w", **kw) as ds:
        ds.write(arr, 1)


def _write_int_tiff_1988(path: str, *, with_sentinel_hit: bool) -> None:
    """Write a uint16 TIFF.

    ``with_sentinel_hit`` controls whether the sentinel matches any
    pixel: when True, the file contains pixels equal to the declared
    sentinel (so the reader will float-promote and mask); when False,
    the file declares a sentinel but no pixel matches (so the reader
    keeps the integer dtype and masked_nodata stays False).
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    if with_sentinel_hit:
        arr = np.array(
            [[10, 20, 65535, 40],
             [50, 65535, 70, 80],
             [90, 100, 110, 120],
             [130, 140, 150, 160]],
            dtype=np.uint16,
        )
    else:
        arr = np.array(
            [[10, 20, 30, 40],
             [50, 60, 70, 80],
             [90, 100, 110, 120],
             [130, 140, 150, 160]],
            dtype=np.uint16,
        )
    with rasterio.open(
        path, "w",
        driver="GTiff", height=4, width=4, count=1, dtype="uint16",
        transform=from_origin(0, 4, 1, 1), crs="EPSG:4326",
        nodata=65535,
    ) as ds:
        ds.write(arr, 1)


def _build_uint16_with_out_of_range_nodata_1988(path: str) -> None:
    """Write a uint16 TIFF whose declared nodata is out of dtype range.

    Mirrors the corpus used elsewhere in this module. The sentinel
    cannot match any pixel, so masking is a no-op and the array stays
    uint16. ``masked_nodata`` must be False so downstream code knows the
    literal value space is intact.
    """
    bo = '<'
    width, height = 2, 2
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    nodata_str = "-9999"  # out of range for uint16
    nodata_bytes = nodata_str.encode('ascii') + b'\x00'

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag: int, val: int) -> None:
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag: int, val: int) -> None:
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_ascii(tag: int, data: bytes) -> None:
        tag_list.append((tag, 2, len(data), data))

    add_short(256, width)
    add_short(257, height)
    add_short(258, 16)            # BitsPerSample
    add_short(259, 1)             # Compression (none)
    add_short(262, 1)             # PhotometricInterpretation (BlackIsZero)
    add_short(277, 1)             # SamplesPerPixel
    add_short(284, 1)             # PlanarConfiguration (chunky)
    add_short(339, 1)             # SampleFormat (unsigned)
    add_long(254, 0)              # NewSubfileType
    add_long(322, 256)            # TileWidth (placeholder; we use strips)

    # Strip the placeholder TileWidth: switch to strips.
    tag_list = [t for t in tag_list if t[0] != 322]

    rows_per_strip = height
    add_short(278, rows_per_strip)  # RowsPerStrip

    pixels_bytes = pixels.tobytes()
    add_long(279, len(pixels_bytes))  # StripByteCounts

    add_ascii(42113, nodata_bytes)   # GDAL_NODATA

    tag_list.sort(key=lambda t: t[0])

    header_size = 8
    ifd_entries = len(tag_list)
    # Strip offset placeholder; pixel data goes after the IFD.
    strip_offsets_tag = (273, 4, 1, b'\x00\x00\x00\x00')  # type=LONG
    tag_list.append(strip_offsets_tag)
    ifd_entries += 1
    tag_list.sort(key=lambda t: t[0])

    ifd_size = 2 + ifd_entries * 12 + 4

    # Layout: header (8) + IFD + tag-overflow data + pixels.
    ifd_offset = header_size
    overflow_offset = ifd_offset + ifd_size

    overflow_buffers: list[bytes] = []
    overflow_positions: dict[int, int] = {}

    new_entries: list[bytes] = []
    for (tag, type_, count, data) in tag_list:
        if len(data) > 4:
            overflow_positions[tag] = overflow_offset + sum(
                len(b) for b in overflow_buffers
            )
            overflow_buffers.append(data)
            value_field = struct.pack(f'{bo}I', overflow_positions[tag])
        else:
            value_field = data.ljust(4, b'\x00')
        new_entries.append(
            struct.pack(f'{bo}HHI', tag, type_, count) + value_field
        )

    pixel_offset = (overflow_offset
                    + sum(len(b) for b in overflow_buffers))

    # Patch StripOffsets value field.
    patched_entries = []
    for entry, (tag, *_) in zip(new_entries, tag_list):
        if tag == 273:
            entry = entry[:8] + struct.pack(f'{bo}I', pixel_offset)
        patched_entries.append(entry)

    with open(path, 'wb') as f:
        f.write(b'II' + struct.pack(f'{bo}HI', 42, ifd_offset))
        f.write(struct.pack(f'{bo}H', ifd_entries))
        for e in patched_entries:
            f.write(e)
        f.write(struct.pack(f'{bo}I', 0))  # next IFD = none
        for b in overflow_buffers:
            f.write(b)
        f.write(pixels_bytes)


class TestEagerNumpy:
    """``open_geotiff`` (eager numpy backend)."""

    def test_float_source_with_sentinel(self, tmp_path):
        """Float source + declared sentinel -> nodata set, masked_nodata=True."""
        path = str(tmp_path / "tnss1988_float_sentinel.tif")
        _write_float_tiff_1988(path, with_sentinel=True)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == _SENTINEL_1988
        assert da.attrs["masked_nodata"] is True
        # The literal sentinel must have been replaced with NaN.
        assert np.isnan(da.values).sum() == 2

    def test_float_source_without_sentinel(self, tmp_path):
        """Float source + no sentinel declared -> neither attr set."""
        path = str(tmp_path / "tnss1988_float_no_sentinel.tif")
        _write_float_tiff_1988(path, with_sentinel=False)
        da = open_geotiff(path)
        assert "nodata" not in da.attrs
        # ``masked_nodata`` is only meaningful when a sentinel was
        # declared; absence is the signal.
        assert "masked_nodata" not in da.attrs

    def test_int_source_with_sentinel_hit(self, tmp_path):
        """Int source + sentinel hit -> nodata set, masked_nodata=True (promoted)."""
        path = str(tmp_path / "tnss1988_int_hit.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=True)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == 65535
        # Eager numpy promotes integer to float64 on the first hit.
        assert da.dtype.kind == "f"
        assert da.attrs["masked_nodata"] is True
        assert np.isnan(da.values).sum() == 2

    def test_int_source_no_hit_keeps_sentinel(self, tmp_path):
        """Int source + sentinel declared but no hit -> nodata set, masked_nodata=False.

        The eager numpy path only promotes integer arrays to float on
        the first sentinel hit. When the sentinel is in-range but never
        matches a pixel, the array stays at the source integer dtype
        and ``masked_nodata`` is False so downstream code knows the
        literal sentinel is still in-band.
        """
        path = str(tmp_path / "tnss1988_int_no_hit.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=False)
        da = open_geotiff(path)
        assert da.attrs["nodata"] == 65535
        assert da.dtype.kind in ("u", "i")
        assert da.attrs["masked_nodata"] is False


class TestDaskNumpy:
    """``read_geotiff_dask`` (lazy dask + numpy backend)."""

    def test_float_source_with_sentinel(self, tmp_path):
        path = str(tmp_path / "tnss1988_dask_float_sentinel.tif")
        _write_float_tiff_1988(path, with_sentinel=True)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == _SENTINEL_1988
        assert da.attrs["masked_nodata"] is True

    def test_float_source_without_sentinel(self, tmp_path):
        path = str(tmp_path / "tnss1988_dask_float_no_sentinel.tif")
        _write_float_tiff_1988(path, with_sentinel=False)
        da = read_geotiff_dask(path, chunks=2)
        assert "nodata" not in da.attrs
        assert "masked_nodata" not in da.attrs

    def test_int_source_with_in_range_sentinel(self, tmp_path):
        """Dask declares float64 up front for any in-range integer sentinel.

        The dask backend cannot defer promotion to runtime the way the
        eager path does (each chunk reads independently and concat
        needs a fixed dtype). When any integer sentinel is in-range,
        the declared graph dtype is float64 and ``masked_nodata`` is
        True regardless of whether a chunk actually hits the sentinel.
        """
        path = str(tmp_path / "tnss1988_dask_int_in_range.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=False)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == 65535
        assert da.dtype.kind == "f"
        assert da.attrs["masked_nodata"] is True

    def test_int_source_with_out_of_range_sentinel(self, tmp_path):
        """Dask + out-of-range int sentinel -> graph stays int, masked_nodata=False.

        The dask ``effective_dtype`` branch only promotes to float64
        when the sentinel fits the source dtype range; an out-of-range
        sentinel (e.g. uint16 file with ``GDAL_NODATA="-9999"``) cannot
        match any pixel, so the declared graph dtype stays uint16 and
        ``masked_nodata`` must be False.
        """
        path = str(tmp_path / "tnss1988_dask_int_oor.tif")
        _build_uint16_with_out_of_range_nodata_1988(path)
        da = read_geotiff_dask(path, chunks=2)
        assert da.attrs["nodata"] == -9999
        assert da.dtype.kind == "u"
        assert da.attrs["masked_nodata"] is False


def _write_uint16_vrt_source_1988(tmp_path, *, sentinel_hit: bool,
                                  filename: str):
    """Write a 2x2 uint16 source raster with declared sentinel 65535."""
    from xrspatial.geotiff._writer import write
    if sentinel_hit:
        band = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    else:
        band = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=65535, compression="none", tiled=False)
    return p


def _build_vrt_1988(tmp_path, source_path, vrt_dtype, nodata_value,
                    filename="tnss1988.vrt"):
    """Hand-roll a 2x2 VRT pointing at ``source_path``."""
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{vrt_dtype}" band="1">
    <NoDataValue>{nodata_value}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = str(tmp_path / filename)
    with open(p, "w") as f:
        f.write(vrt_xml)
    return p


class TestVRTEager:
    """``read_vrt`` (eager path) honours the split-attrs contract."""

    def test_float32_vrt_int_source_with_hit(self, tmp_path):
        """Float-typed VRT over int source with sentinel hit -> masked_nodata=True."""
        src = _write_uint16_vrt_source_1988(
            tmp_path, sentinel_hit=True,
            filename="tnss1988_vrt_src_hit.tif",
        )
        vrt = _build_vrt_1988(tmp_path, src, "Float32", 65535,
                              filename="tnss1988_vrt_hit.vrt")
        r = read_vrt(vrt)
        assert r.attrs["nodata"] == 65535.0
        assert r.dtype.kind == "f"
        assert r.attrs["masked_nodata"] is True

    def test_uint16_vrt_int_source_no_hit(self, tmp_path):
        """Int-typed VRT over int source, no sentinel pixel -> masked_nodata=False.

        A ``dataType="UInt16"`` VRT with no scale/offset keeps the
        source integer dtype. With no sentinel pixel in the source, the
        eager path produces a uint16 array carrying the literal
        sentinel value space, so ``masked_nodata`` must be False.
        """
        src = _write_uint16_vrt_source_1988(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_src_nohit.tif",
        )
        vrt = _build_vrt_1988(tmp_path, src, "UInt16", 65535,
                              filename="tnss1988_vrt_nohit.vrt")
        r = read_vrt(vrt)
        assert r.attrs["nodata"] == 65535.0
        assert r.dtype.kind in ("u", "i")
        assert r.attrs["masked_nodata"] is False

    def test_vrt_no_nodata_emits_neither_attr(self, tmp_path):
        """VRT band with no ``<NoDataValue>`` -> neither attr set."""
        src = _write_uint16_vrt_source_1988(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_src_no_nd.tif",
        )
        # Build a VRT without a NoDataValue element.
        vrt_xml = """<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">""" + src + """</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
        vrt = str(tmp_path / "tnss1988_vrt_no_nd.vrt")
        with open(vrt, "w") as f:
            f.write(vrt_xml)
        r = read_vrt(vrt)
        assert "nodata" not in r.attrs
        assert "masked_nodata" not in r.attrs


class TestVRTChunked:
    """``read_vrt(..., chunks=N)`` honours the split-attrs contract."""

    def test_chunked_int_source_in_range_sentinel(self, tmp_path):
        """Chunked VRT declares float64 for in-range int sentinel -> masked_nodata=True."""
        src = _write_uint16_vrt_source_1988(
            tmp_path, sentinel_hit=False,
            filename="tnss1988_vrt_chunked_src.tif",
        )
        vrt = _build_vrt_1988(tmp_path, src, "UInt16", 65535,
                              filename="tnss1988_vrt_chunked.vrt")
        r = read_vrt(vrt, chunks=2)
        assert r.attrs["nodata"] == 65535.0
        # Chunked path promotes to float64 declared dtype.
        assert r.dtype == np.float64
        assert r.attrs["masked_nodata"] is True


@_gpu_only
class TestGPU:
    """``read_geotiff_gpu`` honours the split-attrs contract."""

    def test_int_source_with_hit(self, tmp_path):
        """Int source + sentinel hit on GPU -> masked_nodata=True (float)."""
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_int_hit.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=True)
        da = read_geotiff_gpu(path)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind == "f"
        assert da.attrs["masked_nodata"] is True

    def test_int_source_no_hit_keeps_sentinel(self, tmp_path):
        """Int source + sentinel no hit on GPU -> masked_nodata=False.

        Mirrors the eager-numpy contract: GPU masking only promotes int
        to float64 when at least one sentinel pixel is found.
        """
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_int_nohit.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=False)
        da = read_geotiff_gpu(path)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind in ("u", "i")
        assert da.attrs["masked_nodata"] is False

    def test_dask_gpu_in_range_sentinel(self, tmp_path):
        """Dask+GPU declares float64 graph for in-range int sentinel."""
        from xrspatial.geotiff import read_geotiff_gpu
        path = str(tmp_path / "tnss1988_gpu_dask_int.tif")
        _write_int_tiff_1988(path, with_sentinel_hit=False)
        da = read_geotiff_gpu(path, chunks=2)
        assert da.attrs["nodata"] == 65535
        assert np.dtype(str(da.dtype)).kind == "f"
        assert da.attrs["masked_nodata"] is True


def test_int_source_with_out_of_range_sentinel(tmp_path):
    """Out-of-range int sentinel -> nodata set, masked_nodata=False (eager).

    The sentinel cannot match any pixel so masking is a no-op and the
    array stays at the source integer dtype. ``masked_nodata`` must be
    False so downstream code knows the literal sentinel value is still
    a possible (but in this case unhit) pixel value in the array.
    """
    path = str(tmp_path / "tnss1988_int_oor.tif")
    _build_uint16_with_out_of_range_nodata_1988(path)
    da = open_geotiff(path)
    assert da.attrs["nodata"] == -9999
    assert da.dtype.kind == "u"
    assert da.attrs["masked_nodata"] is False


class TestSetNodataAttrsHelper:
    """Direct coverage of :func:`_set_nodata_attrs` in ``_attrs.py``.

    The helper takes ``masked`` (an explicit decision passed by the
    caller) rather than inferring from ``array_dtype``. These tests pin
    that contract.
    """

    def test_masked_true_marks_masked(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, -9999, masked=True)
        assert attrs == {"nodata": -9999, "masked_nodata": True}

    def test_masked_false_marks_unmasked(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, -9999, masked=False)
        assert attrs == {"nodata": -9999, "masked_nodata": False}

    def test_none_nodata_is_noop(self):
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, None, masked=False)
        assert attrs == {}
        _set_nodata_attrs(attrs, None, masked=True)
        assert attrs == {}

    def test_masked_coerced_to_bool(self):
        """Non-bool truthy/falsy values are coerced to bool so the
        attr is always a plain Python bool (downstream serialisers
        can't always handle numpy scalars or 0/1 ints)."""
        from xrspatial.geotiff._attrs import _set_nodata_attrs
        attrs: dict = {}
        _set_nodata_attrs(attrs, 0, masked=np.True_)
        assert attrs["masked_nodata"] is True
        assert type(attrs["masked_nodata"]) is bool
        attrs = {}
        _set_nodata_attrs(attrs, 0, masked=0)
        assert attrs["masked_nodata"] is False


class TestShouldRestoreNanSentinelHelper:
    """Direct coverage of :func:`_should_restore_nan_sentinel`."""

    def test_missing_attr_defaults_to_true(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel({}) is True
        # The default is backward-compatible for any DataArray that did
        # not pass through xrspatial's reader.
        assert _should_restore_nan_sentinel({"nodata": -9999}) is True

    def test_masked_nodata_true_returns_true(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        attrs = {"nodata": -9999, "masked_nodata": True}
        assert _should_restore_nan_sentinel(attrs) is True

    def test_masked_nodata_false_returns_false(self):
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        attrs = {"nodata": -9999, "masked_nodata": False}
        assert _should_restore_nan_sentinel(attrs) is False

    def test_none_attrs_defaults_to_true(self):
        """GPU writer's positional-cupy branch has no attrs to read."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel(None) is True

    def test_non_mapping_defaults_to_true(self):
        """A misuse that hands in a non-mapping must not crash the writer."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel
        assert _should_restore_nan_sentinel("not a dict") is True

    def test_stray_truthy_value_is_true(self):
        """Only literal ``False`` disables. Stray ``0`` / ``''`` stays True."""
        from xrspatial.geotiff._attrs import _should_restore_nan_sentinel

        # Anything other than literal False should keep the default
        # behaviour. ``0`` is falsy but is not the contract value.
        assert _should_restore_nan_sentinel({"masked_nodata": 0}) is True
        assert _should_restore_nan_sentinel({"masked_nodata": None}) is True


class TestWriterRoundTripEager:
    """Round-trip through ``to_geotiff`` to verify the writer respects
    ``attrs['masked_nodata']``."""

    def test_masked_nodata_true_restores_sentinel(self, tmp_path):
        """Reader-style attrs (masked_nodata=True): NaN -> sentinel on write."""
        rasterio = pytest.importorskip("rasterio")

        path = tmp_path / "test_1988_writer_masked.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": True,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # NaN pixels should have been rewritten to the sentinel.
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2

    def test_masked_nodata_false_preserves_nan(self, tmp_path):
        """``masked_nodata=False`` -> NaN survives, no silent sentinel rewrite."""
        rasterio = pytest.importorskip("rasterio")

        path = tmp_path / "test_1988_writer_unmasked.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            # The GDAL_NODATA tag is still set, regardless of the
            # in-memory masking state. The two attrs carry independent
            # meanings.
            assert ds.nodata == -9999.0
        # NaN pixels survive unchanged: the writer must NOT rewrite
        # them to the integer sentinel because the array did not pass
        # through the reader's sentinel-to-NaN promotion.
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0

    def test_missing_masked_nodata_attr_restores_sentinel(self, tmp_path):
        """External DataArrays without the attr keep the legacy behaviour."""
        rasterio = pytest.importorskip("rasterio")

        path = tmp_path / "test_1988_writer_no_attr.tif"
        arr = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        # No ``masked_nodata`` attr -> default True.
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
            },
        )
        to_geotiff(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # Legacy behaviour: missing attr = treat as masked.
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2

    def test_round_trip_preserves_masked_nodata_true(self, tmp_path):
        """Read sentinel TIFF -> attrs say masked=True -> write -> reread.

        Closes the loop: a file with a declared sentinel reads back
        with NaN-masked pixels and ``masked_nodata=True``. Writing it
        out then reading again must produce the same sentinel-tagged
        file (the writer correctly inverts the read-side promotion).
        """
        rasterio = pytest.importorskip("rasterio")

        src = tmp_path / "test_1988_round_trip_src.tif"
        _write_float_tiff_1988(str(src), with_sentinel=True)

        da = open_geotiff(str(src))
        assert da.attrs["masked_nodata"] is True
        # The reader promoted the sentinel value to NaN.
        arr_in = np.asarray(da.data)
        assert np.isnan(arr_in).sum() == 2

        dst = tmp_path / "test_1988_round_trip_dst.tif"
        to_geotiff(da, str(dst), compression="none")

        with rasterio.open(str(dst)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == _SENTINEL_1988
        # Sentinel values restored at the expected positions.
        assert (on_disk == _SENTINEL_1988).sum() == 2
        assert not np.isnan(on_disk).any()

    def test_dask_streaming_path_respects_flag(self, tmp_path):
        """Dask + tiled streaming write must honour the gate too."""
        rasterio = pytest.importorskip("rasterio")
        import dask.array as da_mod

        path = tmp_path / "test_1988_writer_dask.tif"
        # 32x32 with NaN sprinkled in -- the tiled streaming writer
        # requires tile_size to be a positive multiple of 16.
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        arr[12, 19] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        da = xr.DataArray(
            dask_arr,
            dims=("y", "x"),
            coords={
                "y": np.arange(32, 0, -1, dtype=np.float64),
                "x": np.arange(32, dtype=np.float64),
            },
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 32.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        to_geotiff(
            da, str(path), compression="none",
            tile_size=16, tiled=True,
        )

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        # NaN preserved through the streaming path.
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0


class TestWriteStreamingRestoreSentinelKwarg:
    """Direct coverage of ``restore_sentinel`` on the low-level
    ``write_streaming`` callable surface, where the gate actually
    suppresses an internal NaN-to-sentinel rewrite step.

    The non-streaming ``write`` function expects its caller (e.g.
    ``to_geotiff``) to have already performed the NaN-to-sentinel
    rewrite, so its own ``restore_sentinel`` flag only gates the
    overview-decimation rewrite (a no-op when ``cog=False``).
    """

    def test_streaming_restore_sentinel_true_rewrites(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import dask.array as da_mod

        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_true.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
            restore_sentinel=True,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert (on_disk == -9999.0).sum() == 1
        assert not np.isnan(on_disk).any()

    def test_streaming_restore_sentinel_false_preserves_nan(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import dask.array as da_mod

        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_false.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
            restore_sentinel=False,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        # GDAL_NODATA tag still set; only the array bytes change.
        assert ds.nodata == -9999.0
        assert np.isnan(on_disk).sum() == 1
        assert (on_disk == -9999.0).sum() == 0

    def test_streaming_default_is_true(self, tmp_path):
        """Default preserves the legacy behaviour."""
        rasterio = pytest.importorskip("rasterio")
        import dask.array as da_mod

        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_default.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=16,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert (on_disk == -9999.0).sum() == 1
        assert not np.isnan(on_disk).any()

    def test_streaming_strip_layout_restore_false_preserves_nan(self, tmp_path):
        """The strip-write branch in ``write_streaming`` must honour the gate."""
        rasterio = pytest.importorskip("rasterio")
        import dask.array as da_mod

        from xrspatial.geotiff._writer import write_streaming

        path = tmp_path / "test_1988_stream_strip_false.tif"
        arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
        arr[5, 7] = np.nan
        dask_arr = da_mod.from_array(arr, chunks=(16, 16))
        write_streaming(
            dask_arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=False,
            restore_sentinel=False,
        )
        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
        assert np.isnan(on_disk).sum() == 1
        assert (on_disk == -9999.0).sum() == 0


class TestWriteCOGOverviewGateInteraction:
    """The ``write()`` function's only gated branch is the overview-
    level NaN-to-sentinel rewrite (the user-data rewrite has already
    been done by ``to_geotiff`` upstream). Close that coverage gap
    with a direct ``write(..., cog=True, ...)`` exercise of the
    gated branch at ``_writer.py:1742-1749``."""

    def test_cog_overview_rewrite_runs_by_default(self, tmp_path):
        """Default ``restore_sentinel=True`` rewrites NaN in overviews."""
        rasterio = pytest.importorskip("rasterio")
        from xrspatial.geotiff._writer import write

        path = tmp_path / "test_1988_cog_default.tif"
        # 64x64 with a sentinel-valued patch that the float pyramid
        # will average down through several overview levels.
        arr = np.ones((64, 64), dtype=np.float32)
        arr[0:16, 0:16] = np.nan
        write(
            arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=32,
            cog=True,
            overview_levels=[2, 4],
        )
        with rasterio.open(str(path)) as ds:
            assert ds.nodata == -9999.0
            # Read overview level 1 (factor=2). Pure-NaN 2x2 blocks
            # reduce to NaN under nanmean; the gated branch rewrites
            # those back to the sentinel.
            ov = ds.read(1, out_shape=(32, 32))
        # Overview tiles covering the all-NaN corner should hold the
        # sentinel, not NaN.
        assert (ov[0:8, 0:8] == -9999.0).any() or np.isnan(ov[0:8, 0:8]).sum() == 0

    def test_cog_overview_rewrite_skipped_when_gated_off(self, tmp_path):
        """``restore_sentinel=False`` preserves NaN in overview pyramid."""
        rasterio = pytest.importorskip("rasterio")
        from xrspatial.geotiff._writer import write

        path = tmp_path / "test_1988_cog_gated.tif"
        arr = np.ones((64, 64), dtype=np.float32)
        arr[0:16, 0:16] = np.nan
        write(
            arr, str(path),
            nodata=-9999.0,
            compression='none',
            tiled=True,
            tile_size=32,
            cog=True,
            overview_levels=[2, 4],
            restore_sentinel=False,
        )
        with rasterio.open(str(path)) as ds:
            assert ds.nodata == -9999.0
            # Same overview read. With the gate off, the all-NaN 2x2
            # blocks stay NaN through the pyramid.
            ov = ds.read(1, out_shape=(32, 32))
        # NaN survives in the overview corner.
        assert np.isnan(ov[0:8, 0:8]).any()


@_gpu_only
class TestWriterGPU:
    """GPU writer also gates on ``attrs['masked_nodata']``."""

    def test_masked_nodata_false_preserves_nan_gpu(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import cupy

        from xrspatial.geotiff import write_geotiff_gpu

        path = tmp_path / "test_1988_writer_gpu_unmasked.tif"
        arr_np = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        arr = cupy.asarray(arr_np)
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": False,
            },
        )
        write_geotiff_gpu(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        assert np.isnan(on_disk).sum() == 2
        assert (on_disk == -9999.0).sum() == 0

    def test_masked_nodata_true_restores_sentinel_gpu(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import cupy

        from xrspatial.geotiff import write_geotiff_gpu

        path = tmp_path / "test_1988_writer_gpu_masked.tif"
        arr_np = np.array(
            [[1.0, 2.0, np.nan, 4.0],
             [5.0, np.nan, 7.0, 8.0]],
            dtype=np.float32,
        )
        arr = cupy.asarray(arr_np)
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            attrs={
                "crs": 4326,
                "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
                "nodata": -9999.0,
                "masked_nodata": True,
            },
        )
        write_geotiff_gpu(da, str(path), compression="none")

        with rasterio.open(str(path)) as ds:
            on_disk = ds.read(1)
            assert ds.nodata == -9999.0
        assert not np.isnan(on_disk).any()
        assert (on_disk == -9999.0).sum() == 2

# ===========================================================================
# Default rejection of non-finite / fractional int nodata (#2441)
# Source: test_invalid_int_nodata_rejection_2441.py
# ===========================================================================


_build_uint16_tiff = _build_uint16_tiff_1774


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


# ----------------------------------------------------------------------
# Default behaviour: reject non-finite int sentinels at the read boundary
# ----------------------------------------------------------------------


@pytest.mark.parametrize('nodata_str', ['nan', 'NaN', 'NAN',
                                        'inf', '-inf', 'Inf', '-Inf'])
def test_open_geotiff_eager_int_nodata_nonfinite_rejected_by_default(
    tmp_path, nodata_str,
):
    """Eager numpy path raises ``InvalidIntegerNodataError`` for non-finite
    ``GDAL_NODATA`` on integer sources.
    """
    path = _build_uint16_tiff(nodata_str, tmp_path)
    with pytest.raises(InvalidIntegerNodataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert 'nodata' in msg.lower()
    # Message names the offending sentinel kind and dtype so the user
    # can locate the bad source.
    assert 'non-finite' in msg
    assert 'uint16' in msg
    # The opt-in flag name appears in the message so the caller can
    # discover the escape hatch from the rejection itself.
    assert 'allow_invalid_nodata' in msg


@pytest.mark.parametrize('nodata_str', ['3.5', '29.5', '30.5', '0.25'])
def test_open_geotiff_eager_int_nodata_fractional_rejected_by_default(
    tmp_path, nodata_str,
):
    """Eager numpy path raises ``InvalidIntegerNodataError`` for fractional
    ``GDAL_NODATA`` on integer sources.
    """
    path = _build_uint16_tiff(nodata_str, tmp_path)
    with pytest.raises(InvalidIntegerNodataError) as excinfo:
        open_geotiff(path)
    msg = str(excinfo.value)
    assert 'nodata' in msg.lower()
    assert 'fractional' in msg
    assert 'uint16' in msg
    assert 'allow_invalid_nodata' in msg


def test_invalid_int_nodata_error_is_geotiff_ambiguous_metadata_error():
    """The new error subclasses ``GeoTIFFAmbiguousMetadataError`` so
    existing ``except GeoTIFFAmbiguousMetadataError`` callers catch it.
    """
    assert issubclass(InvalidIntegerNodataError,
                      GeoTIFFAmbiguousMetadataError)


def test_read_geotiff_dask_int_nodata_nan_rejected_by_default(tmp_path):
    """Dask path raises at graph-build time, before any chunk task fires."""
    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_dask(path, chunks=2)


def test_read_geotiff_dask_int_nodata_fractional_rejected_by_default(
    tmp_path,
):
    """Dask path raises at graph-build time for fractional int sentinels."""
    path = _build_uint16_tiff('30.5', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_dask(path, chunks=2)


# ----------------------------------------------------------------------
# Float sources are unaffected
# ----------------------------------------------------------------------


def test_open_geotiff_float_dtype_nan_nodata_still_allowed(tmp_path):
    """Float-dtype sources with NaN ``GDAL_NODATA`` are the normal case
    and must not raise. NaN matches NaN, masking proceeds.
    """
    import xarray as xr

    from xrspatial.geotiff import to_geotiff

    arr = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    da = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': [0.5, -0.5], 'x': [0.5, 1.5]},
        attrs={'crs': 4326},
    )
    path = str(tmp_path / 'float_nan_nodata_2441.tif')
    to_geotiff(da, path, nodata=float('nan'), compression='none', tiled=False)
    out = open_geotiff(path)
    assert out.dtype.kind == 'f'
    assert np.isnan(out.attrs['nodata'])


# ----------------------------------------------------------------------
# Finite, in-range integer sentinels are unaffected
# ----------------------------------------------------------------------


def test_open_geotiff_int_finite_nodata_unaffected(tmp_path):
    """Finite integer-valued sentinels still mask as before; the new
    validator must only reject non-finite / fractional sentinels.
    """
    path = _build_uint16_tiff('30', tmp_path)
    da = open_geotiff(path)
    # 30 matches a real pixel; the sentinel-to-NaN promotion fires.
    assert da.dtype == np.float64
    assert np.isnan(da.values[1, 0])
    assert da.attrs['nodata'] == 30


# ----------------------------------------------------------------------
# Opt-in restores the legacy no-op
# ----------------------------------------------------------------------


@pytest.mark.parametrize('nodata_str', ['nan', 'inf', '3.5'])
def test_open_geotiff_opt_in_restores_noop_eager(tmp_path, nodata_str):
    """``allow_invalid_nodata=True`` keeps the pre-2441 no-op behaviour."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = open_geotiff(path, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, [[10, 20], [30, 40]])


@pytest.mark.parametrize('nodata_str', ['nan', '30.5'])
def test_read_geotiff_dask_opt_in_restores_noop(tmp_path, nodata_str):
    """``allow_invalid_nodata=True`` keeps the pre-2441 no-op for dask."""
    path = _build_uint16_tiff(nodata_str, tmp_path)
    da = read_geotiff_dask(path, chunks=2, allow_invalid_nodata=True)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, [[10, 20], [30, 40]])


# ----------------------------------------------------------------------
# GPU path mirrors the CPU contract
# ----------------------------------------------------------------------


@_gpu_only
def test_read_geotiff_gpu_int_nodata_nan_rejected_by_default(tmp_path):
    """GPU read entry point raises before kicking off the device decode."""
    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_gpu(path)


@_gpu_only
def test_read_geotiff_gpu_int_nodata_opt_in_restores_noop(tmp_path):
    """GPU opt-in keeps the no-op (sentinel cannot match any uint16 pixel)."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    da = read_geotiff_gpu(path, allow_invalid_nodata=True)
    # Buffer stays uint16 on the device.
    assert da.dtype == cupy.uint16
    arr = da.data.get()
    np.testing.assert_array_equal(arr, [[10, 20], [30, 40]])


@_gpu_only
def test_read_geotiff_gpu_chunked_int_nodata_rejected_by_default(tmp_path):
    """dask+cupy backend rejects at metadata parse, before any chunk task
    is scheduled. Closes the four-backend matrix explicitly.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    path = _build_uint16_tiff('nan', tmp_path)
    with pytest.raises(InvalidIntegerNodataError):
        read_geotiff_gpu(path, chunks=2)
