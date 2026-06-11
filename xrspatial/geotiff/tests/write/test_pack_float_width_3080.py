"""``to_geotiff(pack=True)`` must keep a float32 source float32 (#3080).

``_pack`` casts to the dtype recorded on ``attrs['mask_and_scale_dtype']``.
Float sources never used to record that attr (they are not promoted on
read), so ``_pack`` fell back to the dtype of the nodata scalar -- a plain
Python float, i.e. float64 -- and widened the packed file. The fix records
the source dtype on ``unpack=True`` reads of float sources too, and stops
the fallback from keying off float-typed sentinels.

GPU / dask+GPU pack round-trips are pinned on #3112 / covered by #3114;
this file exercises the numpy and dask+numpy legs.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _write_f32_tiff(path, data, *, nodata=None, scale=None, offset=None):
    """Write float32 ``data`` as a GeoTIFF, optionally carrying a nodata
    sentinel and GDAL SCALE/OFFSET packing metadata."""
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


def _reopen(path, chunks, **kwargs):
    return (open_geotiff(path, **kwargs) if chunks is None
            else open_geotiff(path, chunks=chunks, **kwargs))


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_preserves_float32_width(tmp_path, chunks):
    """float32 + nodata, read with unpack=True, packs back to float32."""
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_f32_tiff(tmp_path / "src_f32_3080.tif", data, nodata=-9999.0)

    decoded = _reopen(src, chunks, unpack=True)
    assert decoded.dtype == np.float32  # read never promotes float sources
    assert decoded.attrs.get("mask_and_scale_dtype") == "float32"

    out = str(tmp_path / "out_f32_3080.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "float32"
    np.testing.assert_array_equal(np.asarray(back.data), data)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_preserves_float32_width_with_scale_offset(tmp_path, chunks):
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_f32_tiff(
        tmp_path / "src_f32_so_3080.tif", data,
        nodata=-9999.0, scale=2.0, offset=10.0)

    decoded = _reopen(src, chunks, unpack=True)
    assert decoded.dtype == np.float32
    assert decoded.attrs.get("mask_and_scale_dtype") == "float32"

    out = str(tmp_path / "out_f32_so_3080.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Raw read: float32 width and the stored values round-trip exactly
    # (1.5 / 2.0 / 10.0 are all binary-exact in float32).
    raw = open_geotiff(out)
    assert str(raw.dtype) == "float32"
    np.testing.assert_array_equal(np.asarray(raw.data), data)

    # The kept SCALE/OFFSET tags unpack the packed file to the same
    # values as the source, not scaled twice.
    src_decoded = open_geotiff(src, unpack=True)
    repacked_decoded = open_geotiff(out, unpack=True)
    np.testing.assert_array_equal(
        np.asarray(repacked_decoded.data), np.asarray(src_decoded.data))


def test_float32_attr_recorded_eager_dask_match(tmp_path):
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_f32_tiff(
        tmp_path / "src_f32_attr_3080.tif", data, nodata=-9999.0)

    eager = open_geotiff(src, unpack=True)
    lazy = open_geotiff(src, unpack=True, chunks=2)
    assert eager.attrs.get("mask_and_scale_dtype") == "float32"
    assert (eager.attrs.get("mask_and_scale_dtype")
            == lazy.attrs.get("mask_and_scale_dtype"))


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_masked_read_keeps_float32_width(tmp_path, chunks):
    """A plain ``masked=True`` read records no ``mask_and_scale_dtype``
    (pack is the inverse of unpack specifically), so ``_pack`` used to
    fall back to the float64 width of the nodata scalar. The buffer's own
    width is the right target: float sources are never promoted on read.
    """
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_f32_tiff(
        tmp_path / "src_f32_masked_3080.tif", data, nodata=-9999.0)

    decoded = _reopen(src, chunks, masked=True)
    assert "mask_and_scale_dtype" not in decoded.attrs

    out = str(tmp_path / "out_f32_masked_3080.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "float32"
    np.testing.assert_array_equal(np.asarray(back.data), data)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_still_rejects_no_op_float_unpack_read(tmp_path, chunks):
    """A float source with no sentinel and no scale / offset carries no
    unpack state to reverse. Recording the float dtype is gated on real
    unpack state so this read still gets the loud error rather than a
    silent pass-through write."""
    data = np.array([[1.5, 2.5], [4.5, 5.5]], dtype=np.float32)
    src = _write_f32_tiff(tmp_path / "src_f32_noop_3080.tif", data)

    decoded = _reopen(src, chunks, unpack=True)
    assert "mask_and_scale_dtype" not in decoded.attrs

    with pytest.raises(ValueError, match="no unpack state"):
        decoded.xrs.to_geotiff(
            str(tmp_path / "out_f32_noop_3080.tif"), pack=True)
