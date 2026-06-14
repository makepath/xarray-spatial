"""``to_geotiff(pack=True, nodata=X)`` -- the kwarg must drive the NaN fill
(#3168).

``_pack`` used to fill NaN holes with the attrs sentinel while the
GDAL_NODATA tag was written from the ``nodata=`` kwarg (the documented
"kwarg overrides attrs" precedence). The two halves of one write
disagreed: holes carried sentinel A, the tag declared sentinel B, and a
``masked=True`` re-read returned the holes as valid pixels. The kwarg is
now threaded into the fill step so the filled pixels and the tag always
agree.
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
    # A gdal_metadata SCALE/OFFSET dict on a fresh array is an
    # experimental rich-tag write (#3320).
    to_geotiff(da, path, allow_experimental_codecs="gdal_metadata" in attrs)
    return path


def _reopen(path, chunks, gpu=False):
    kwargs = {"unpack": True}
    if gpu:
        kwargs["gpu"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return open_geotiff(path, **kwargs)


# ---------------------------------------------------------------------------
# The issue repro: sentinel A on read, nodata=B on write, masked re-read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_nodata_kwarg_fills_holes_with_kwarg(tmp_path, chunks):
    """unpack read with sentinel -32768, pack write with nodata=-1: the
    holes must be filled with -1 (the kwarg), matching the GDAL_NODATA
    tag, so a masked re-read masks them again."""
    data = np.array([[10, 20, -32768], [40, -32768, 60]], dtype=np.int16)
    src = _write_int_tiff(tmp_path / "src_i16_3168.tif", data, nodata=-32768)

    decoded = _reopen(src, chunks)
    hole_mask = np.isnan(np.asarray(decoded.data))
    assert hole_mask.sum() == 2  # both sentinels were masked on read

    out = str(tmp_path / "out_i16_3168.tif")
    decoded.xrs.to_geotiff(out, pack=True, nodata=-1)

    # Raw read: the holes carry the kwarg sentinel, and the tag agrees.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "int16"
    raw_vals = np.asarray(raw.data)
    np.testing.assert_array_equal(raw_vals[hole_mask], -1)
    assert raw.attrs.get("nodata") == -1
    assert not (raw_vals == -32768).any()

    # Masked re-read: the holes come back as NaN, not as valid pixels.
    masked = open_geotiff(out, masked=True)
    masked_vals = np.asarray(masked.data)
    np.testing.assert_array_equal(np.isnan(masked_vals), hole_mask)
    np.testing.assert_array_equal(masked_vals[~hole_mask], data[~hole_mask])


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_nodata_kwarg_unpack_round_trip_with_scale_offset(
        tmp_path, chunks):
    """With SCALE/OFFSET in play, the re-packed file must still unpack to
    the original decoded values when the sentinel was swapped via the
    kwarg: same NaN holes, same valid pixels, no double-scaling."""
    data = np.array([[1, 2, 255], [4, 255, 6]], dtype=np.uint8)
    src = _write_int_tiff(
        tmp_path / "src_so_3168.tif", data, nodata=255, scale=2.0,
        offset=10.0)

    decoded = _reopen(src, chunks)
    out = str(tmp_path / "out_so_3168.tif")
    decoded.xrs.to_geotiff(out, pack=True, nodata=250)

    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint8"
    assert raw.attrs.get("nodata") == 250
    assert np.asarray(raw.data)[0, 2] == 250

    # An unpack read of the re-packed file reproduces the decoded values:
    # the new sentinel masks back to NaN and the rest unpacks once.
    repacked = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(repacked.data), np.asarray(decoded.data), equal_nan=True)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_nodata_kwarg_float_target(tmp_path, chunks):
    """Float pack targets skip the integer range guard and the cast, so
    pin the kwarg fill on that branch too: the hole carries the kwarg
    sentinel, the tag agrees, and a masked re-read masks it."""
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_int_tiff(tmp_path / "src_f32_3168.tif", data, nodata=-9999.0)

    decoded = _reopen(src, chunks)
    hole_mask = np.isnan(np.asarray(decoded.data))
    assert hole_mask.sum() == 1

    out = str(tmp_path / "out_f32_3168.tif")
    decoded.xrs.to_geotiff(out, pack=True, nodata=-5.0)

    raw = open_geotiff(out)
    raw_vals = np.asarray(raw.data)
    assert raw_vals[0, 2] == -5.0
    assert raw.attrs.get("nodata") == -5.0
    assert not np.isnan(raw_vals).any()

    masked = open_geotiff(out, masked=True)
    np.testing.assert_array_equal(
        np.isnan(np.asarray(masked.data)), hole_mask)


def test_pack_without_nodata_kwarg_still_uses_attrs_sentinel(tmp_path):
    """No kwarg: the attrs sentinel keeps driving both the fill and the
    tag, as before."""
    data = np.array([[10, 20, -32768], [40, 50, 60]], dtype=np.int16)
    src = _write_int_tiff(tmp_path / "src_dflt_3168.tif", data, nodata=-32768)

    decoded = open_geotiff(src, unpack=True)
    out = str(tmp_path / "out_dflt_3168.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    raw = open_geotiff(out)
    np.testing.assert_array_equal(np.asarray(raw.data), data)
    assert raw.attrs.get("nodata") == -32768


def test_pack_rejects_nodata_kwarg_outside_packed_dtype_range(tmp_path):
    """An override that cannot be represented in the packed integer dtype
    would wrap in the cast and recreate the fill-vs-tag mismatch; refuse
    it instead."""
    data = np.array([[1, 2, 255], [4, 5, 6]], dtype=np.uint8)
    src = _write_int_tiff(tmp_path / "src_oor_3168.tif", data, nodata=255)

    decoded = open_geotiff(src, unpack=True)
    out = str(tmp_path / "out_oor_3168.tif")
    with pytest.raises(ValueError, match="cannot be represented"):
        decoded.xrs.to_geotiff(out, pack=True, nodata=-1)
    with pytest.raises(ValueError, match="cannot be represented"):
        decoded.xrs.to_geotiff(out, pack=True, nodata=250.5)


# ---------------------------------------------------------------------------
# GPU legs: ``pack=True`` accepts cupy / dask+cupy input since #3240, so the
# kwarg-driven fill must hold there too (#3266).
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("chunks", [None, 2], ids=["gpu", "dask-gpu"])
def test_pack_nodata_kwarg_fills_holes_with_kwarg_gpu(tmp_path, chunks):
    """A ``gpu=True`` unpack read packed with ``nodata=-1``: the holes in
    the cupy buffer carry the kwarg sentinel and the tag agrees, so a
    masked re-read masks them again."""
    data = np.array([[10, 20, -32768], [40, -32768, 60]], dtype=np.int16)
    src = _write_int_tiff(
        tmp_path / "src_i16_gpu_3266.tif", data, nodata=-32768)

    decoded = _reopen(src, chunks, gpu=True)
    hole_mask = data == -32768

    out = str(tmp_path / f"out_i16_gpu_3266_{chunks}.tif")
    decoded.xrs.to_geotiff(out, pack=True, nodata=-1)

    # Raw read: the holes carry the kwarg sentinel, and the tag agrees.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "int16"
    raw_vals = np.asarray(raw.data)
    np.testing.assert_array_equal(raw_vals[hole_mask], -1)
    assert raw.attrs.get("nodata") == -1
    assert not (raw_vals == -32768).any()

    # Masked re-read: the holes come back as NaN, not as valid pixels.
    masked = open_geotiff(out, masked=True)
    masked_vals = np.asarray(masked.data)
    np.testing.assert_array_equal(np.isnan(masked_vals), hole_mask)
    np.testing.assert_array_equal(masked_vals[~hole_mask], data[~hole_mask])
