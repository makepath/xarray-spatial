"""``to_geotiff(pack=True)`` -- the inverse of ``unpack=True`` (#3064).

A ``unpack=True`` read promotes an integer raster to float64,
applies the GDAL SCALE/OFFSET, and masks the nodata sentinel to NaN.
``pack=True`` reverses that on write: it un-scales, fills NaN back to the
sentinel, and restores the integer source dtype recorded on
``attrs['mask_and_scale_dtype']``. The SCALE/OFFSET tags are kept so the
re-packed file unpacks to the same values on the next ``mask_and_scale``
read rather than double-scaling.

``unpack`` reads run on numpy, dask, gpu, and dask+gpu since #3075 (VRT
still rejects it). The round-trip passes on numpy and dask; the gpu and
dask+gpu legs are pinned as strict xfail on #3112 until the cupy
``fillna`` crash in ``_pack`` is fixed.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .._helpers.markers import requires_gpu


def _write_int_tiff(path, data, *, nodata=None, scale=None, offset=None):
    """Write ``data`` (an integer ndarray) as a GeoTIFF, optionally carrying
    a nodata sentinel and GDAL SCALE/OFFSET packing metadata."""
    h, w = data.shape
    attrs = {"crs": 4326}
    if nodata is not None:
        attrs["nodata"] = nodata
    if scale is not None or offset is not None:
        attrs["gdal_metadata"] = {
            "SCALE": str(scale if scale is not None else 1.0),
            "OFFSET": str(offset if offset is not None else 0.0),
        }
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(h, 0, -1) - 0.5, "x": np.arange(w) + 0.5},
        attrs=attrs,
    )
    to_geotiff(da, path)
    return path


def _reopen(path, chunks):
    return (open_geotiff(path, unpack=True) if chunks is None
            else open_geotiff(path, unpack=True, chunks=chunks))


# ---------------------------------------------------------------------------
# Round trip: no scale/offset (the int8 + nodata case from the issue)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_restores_int8_no_scale(tmp_path, chunks):
    """int8 + nodata, read with mask_and_scale, packs back to int8."""
    data = np.array([[1, 2, -128], [4, 5, 6]], dtype=np.int8)
    src = _write_int_tiff(tmp_path / "src_int8_3064.tif", data, nodata=-128)

    decoded = _reopen(src, chunks)
    assert decoded.dtype == np.float64
    assert decoded.attrs.get("mask_and_scale_dtype") == "int8"

    out = str(tmp_path / "out_int8_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "int8"
    np.testing.assert_array_equal(back.data, data)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_restores_uint16_no_scale(tmp_path, chunks):
    data = np.array([[10, 20, 65535], [40, 50, 60]], dtype=np.uint16)
    src = _write_int_tiff(tmp_path / "src_u16_3064.tif", data, nodata=65535)

    decoded = _reopen(src, chunks)
    out = str(tmp_path / "out_u16_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "uint16"
    np.testing.assert_array_equal(back.data, data)


# ---------------------------------------------------------------------------
# Float source: the masked sentinel must be restored, not left as NaN (#3078)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_restores_float_nodata_sentinel(tmp_path, chunks):
    """A float raster masks its sentinel to NaN on an ``unpack=True`` read.
    ``pack=True`` must put the declared sentinel back so the pixels on disk
    match the GDAL_NODATA tag -- otherwise the file declares nodata=-9999 but
    stores NaN, which a non-masking reader silently treats as a valid value.
    """
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_int_tiff(tmp_path / "src_f32_3078.tif", data, nodata=-9999.0)

    decoded = _reopen(src, chunks)
    assert np.isnan(np.asarray(decoded.data)).any()  # sentinel was masked

    out = str(tmp_path / "out_f32_3078.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Plain (non-masking) read: the sentinel is back and no NaN survives.
    back = np.asarray(open_geotiff(out).data)
    assert not np.isnan(back).any()
    assert back[0, 2] == -9999.0
    assert open_geotiff(out).attrs.get("nodata") == -9999.0

    # An unpack read still round-trips the masked value to NaN.
    repacked = open_geotiff(out, unpack=True)
    np.testing.assert_array_equal(
        np.asarray(repacked.data), np.asarray(decoded.data))


# ---------------------------------------------------------------------------
# Round trip: with SCALE/OFFSET -- the file must unpack, not double-scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_with_scale_offset_round_trip(tmp_path, chunks):
    data = np.array([[1, 2, 3], [4, 5, 255]], dtype=np.uint8)
    src = _write_int_tiff(
        tmp_path / "src_so_3064.tif", data, nodata=255, scale=2.0, offset=10.0)

    decoded = _reopen(src, chunks)
    # unpacked values: data * 2 + 10, sentinel -> NaN
    assert decoded.attrs.get("mask_and_scale_dtype") == "uint8"

    out = str(tmp_path / "out_so_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Raw read: the stored integers round-trip exactly.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint8"
    np.testing.assert_array_equal(raw.data, data)

    # The SCALE/OFFSET tags are kept, so a mask_and_scale read of the packed
    # file reproduces the original decoded values rather than scaling twice.
    eager_decoded = open_geotiff(src, unpack=True)
    repacked_decoded = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        repacked_decoded.data, eager_decoded.data, equal_nan=True)


# ---------------------------------------------------------------------------
# GPU-backed input: ``unpack=True`` works with ``gpu=True`` since #3075, so
# the ``pack=True`` inverse must round-trip there too. Both legs crash today
# (#3112): ``_pack``'s ``fillna`` breaks on cupy-backed arrays. Strict xfail
# so the marker flips loudly once the crash is fixed.
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.xfail(
    strict=True,
    # eager gpu: xarray's where() calls cupy.astype (AttributeError);
    # dask+gpu: numpy fill value inside cupy.where (TypeError). Pinning
    # raises= keeps an unrelated assertion failure from hiding under the
    # known crash.
    raises=(AttributeError, TypeError),
    reason="to_geotiff(pack=True) crashes on cupy-backed input (#3112)")
@pytest.mark.parametrize("chunks", [None, 2], ids=["gpu", "dask-gpu"])
def test_pack_round_trip_gpu(tmp_path, chunks):
    """A ``gpu=True`` unpack read packs back to the integer source dtype."""
    data = np.array([[1, 2, 3], [4, 5, 255]], dtype=np.uint8)
    src = _write_int_tiff(
        tmp_path / "src_gpu_3114.tif", data,
        nodata=255, scale=2.0, offset=10.0)

    decoded = (open_geotiff(src, unpack=True, gpu=True) if chunks is None
               else open_geotiff(src, unpack=True, gpu=True, chunks=chunks))
    assert decoded.attrs.get("mask_and_scale_dtype") == "uint8"

    out = str(tmp_path / f"out_gpu_3114_{chunks}.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Raw read: the stored integers round-trip exactly.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint8"
    np.testing.assert_array_equal(np.asarray(raw.data), data)

    # Parity with the CPU eager unpack of the same source.
    cpu_decoded = open_geotiff(src, unpack=True)
    repacked_decoded = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(repacked_decoded.data), np.asarray(cpu_decoded.data),
        equal_nan=True)


# ---------------------------------------------------------------------------
# Read-side attr: recorded for mask_and_scale, not for a plain masked read
# ---------------------------------------------------------------------------


def test_mask_and_scale_dtype_recorded_eager_dask_match(tmp_path):
    data = np.array([[1, 2, 255], [4, 5, 6]], dtype=np.uint8)
    src = _write_int_tiff(tmp_path / "src_attr_3064.tif", data, nodata=255)

    eager = open_geotiff(src, unpack=True)
    lazy = open_geotiff(src, unpack=True, chunks=2)
    assert eager.attrs.get("mask_and_scale_dtype") == "uint8"
    assert (eager.attrs.get("mask_and_scale_dtype")
            == lazy.attrs.get("mask_and_scale_dtype"))


def test_mask_and_scale_dtype_absent_on_plain_masked_read(tmp_path):
    """A plain ``masked=True`` read (not mask_and_scale) does not stamp the
    attr; ``pack`` is the inverse of ``mask_and_scale`` specifically."""
    data = np.array([[1, 2, 255], [4, 5, 6]], dtype=np.uint8)
    src = _write_int_tiff(tmp_path / "src_masked_3064.tif", data, nodata=255)

    masked = open_geotiff(src, masked=True)
    assert masked.dtype == np.float64  # masking still promotes
    assert "mask_and_scale_dtype" not in masked.attrs


def test_contract_version_is_5(tmp_path):
    src = _write_int_tiff(
        tmp_path / "src_ver_3064.tif",
        np.array([[1, 2], [3, 4]], dtype=np.uint8))
    da = open_geotiff(src)
    assert da.attrs["_xrspatial_geotiff_contract"] == 5


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_pack_rejects_bare_array(tmp_path):
    arr = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="pack=True requires a DataArray"):
        to_geotiff(arr, str(tmp_path / "x_3064.tif"), pack=True)


def test_pack_rejects_array_without_mask_and_scale_state(tmp_path):
    da = xr.DataArray(
        np.ones((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs={"crs": 4326},
    )
    with pytest.raises(ValueError, match="no mask_and_scale state"):
        da.xrs.to_geotiff(str(tmp_path / "y_3064.tif"), pack=True)
