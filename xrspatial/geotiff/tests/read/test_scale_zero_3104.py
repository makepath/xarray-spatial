"""Zero / non-finite SCALE and OFFSET rejection under unpack (#3104).

#2992 made ``unpack=True`` fail closed on SCALE/OFFSET values that do
not parse as floats. Values that parse but cannot be honoured slipped
through: ``SCALE=0`` collapses every pixel to the offset, a non-finite
SCALE or OFFSET destroys the array, and ``to_geotiff(pack=True)``
divides by SCALE, so none of them have an inverse. All are now rejected
with the same :class:`MalformedScaleOffsetError` on every read path:
eager numpy, dask, GPU, and dask+GPU all resolve the metadata through
``_extract_scale_offset``.

``_pack`` carries its own guard for hand-edited attrs: a
``scale_factor`` of zero or a non-finite ``scale_factor``/``add_offset``
raises ``ValueError`` instead of writing a corrupt file.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._errors import MalformedScaleOffsetError

from .._helpers.markers import requires_gpu


def _scale_tiff(path, scale, offset="0", nodata=None):
    """uint8 raster carrying the given SCALE/OFFSET strings."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    attrs = {
        "crs": 4326,
        "gdal_metadata": {"SCALE": scale, "OFFSET": offset},
    }
    if nodata is not None:
        attrs["nodata"] = nodata
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
        attrs=attrs,
    )
    # gdal_metadata dict on a fresh array is an experimental rich-tag
    # write (#3320).
    to_geotiff(da, path, allow_experimental_codecs=True)
    return path


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
@pytest.mark.parametrize("scale", ["0", "0.0", "-0.0", "nan", "inf", "-inf"])
def test_unpack_rejects_unhonourable_scale(tmp_path, chunks, scale):
    path = _scale_tiff(
        str(tmp_path / f"t3104_scale_{scale}_{chunks}.tif"), scale=scale)
    kwargs = {} if chunks is None else {"chunks": chunks}
    with pytest.raises(MalformedScaleOffsetError, match="SCALE"):
        open_geotiff(path, unpack=True, **kwargs)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
@pytest.mark.parametrize("offset", ["nan", "inf", "-inf"])
def test_unpack_rejects_nonfinite_offset(tmp_path, chunks, offset):
    path = _scale_tiff(
        str(tmp_path / f"t3104_offset_{offset}_{chunks}.tif"),
        scale="2.0", offset=offset)
    kwargs = {} if chunks is None else {"chunks": chunks}
    with pytest.raises(MalformedScaleOffsetError, match="OFFSET"):
        open_geotiff(path, unpack=True, **kwargs)


def test_zero_scale_ignored_without_unpack(tmp_path):
    """Without unpack the metadata is never applied, so no rejection."""
    path = _scale_tiff(str(tmp_path / "t3104_no_unpack.tif"), scale="0")
    out = open_geotiff(path)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out.data, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_negative_scale_still_honoured(tmp_path, chunks):
    """A negative finite SCALE is legitimate and keeps working."""
    path = _scale_tiff(
        str(tmp_path / f"t3104_neg_{chunks}.tif"), scale="-2.0", offset="1.0")
    kwargs = {} if chunks is None else {"chunks": chunks}
    out = open_geotiff(path, unpack=True, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(out.data),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64) * -2.0 + 1.0)


@requires_gpu
@pytest.mark.parametrize("chunks", [None, 2], ids=["gpu", "dask+gpu"])
def test_unpack_rejects_zero_scale_gpu(tmp_path, chunks):
    """GPU and dask+GPU unpack (#3075) hit the same rejection."""
    path = _scale_tiff(
        str(tmp_path / f"t3104_gpu_{chunks}.tif"), scale="0")
    kwargs = {} if chunks is None else {"chunks": chunks}
    with pytest.raises(MalformedScaleOffsetError, match="SCALE"):
        open_geotiff(path, unpack=True, gpu=True, **kwargs)


@requires_gpu
def test_valid_scale_still_unpacks_gpu(tmp_path):
    path = _scale_tiff(
        str(tmp_path / "t3104_gpu_good.tif"), scale="2.0", offset="1.0")
    out = open_geotiff(path, unpack=True, gpu=True)
    arr = out.data
    if hasattr(arr, "get"):
        arr = arr.get()
    np.testing.assert_array_equal(
        np.asarray(arr),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64) * 2.0 + 1.0)


def test_pack_rejects_zero_scale_factor_attr(tmp_path):
    """Hand-edited attrs cannot drive _pack into a divide-by-zero."""
    path = _scale_tiff(
        str(tmp_path / "t3104_pack_src.tif"), scale="2.0", offset="1.0",
        nodata=0)
    out = open_geotiff(path, unpack=True)
    out.attrs["scale_factor"] = 0.0
    with pytest.raises(ValueError, match="scale_factor"):
        to_geotiff(out, str(tmp_path / "t3104_pack_out.tif"), pack=True)


@pytest.mark.parametrize("attr,value", [
    ("scale_factor", float("nan")),
    ("scale_factor", float("inf")),
    ("add_offset", float("nan")),
])
def test_pack_rejects_nonfinite_scale_offset_attrs(tmp_path, attr, value):
    path = _scale_tiff(
        str(tmp_path / f"t3104_pack_{attr}_src.tif"), scale="2.0",
        offset="1.0", nodata=0)
    out = open_geotiff(path, unpack=True)
    out.attrs[attr] = value
    with pytest.raises(ValueError, match="pack=True cannot reverse"):
        to_geotiff(
            out, str(tmp_path / f"t3104_pack_{attr}_out.tif"), pack=True)
