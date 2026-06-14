"""Eager/dask dtype parity for ``mask_and_scale`` reads (#3066).

The dask reader used to promote any integer source to float64 for
``unpack=True`` regardless of whether a scale/offset or a fittable
nodata sentinel was present, while the eager reader kept the integer dtype
when there was nothing to transform. These tests pin the two paths to the
same dtype (and values) across the relevant cases.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _write(path, data, *, nodata=None, scale=None, offset=None):
    h, w = data.shape
    attrs = {"crs": 4326}
    if nodata is not None:
        attrs["nodata"] = nodata
    if scale is not None or offset is not None:
        attrs["gdal_metadata"] = {
            "SCALE": str(scale if scale is not None else 1.0),
            "OFFSET": str(offset if offset is not None else 0.0),
        }
    # gdal_metadata SCALE/OFFSET dict on a fresh array is an experimental
    # rich-tag write (#3320).
    to_geotiff(
        xr.DataArray(
            data, dims=("y", "x"),
            coords={"y": np.arange(h, 0, -1) - 0.5, "x": np.arange(w) + 0.5},
            attrs=attrs),
        path,
        allow_experimental_codecs="gdal_metadata" in attrs)
    return path


_DATA = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)


@pytest.mark.parametrize(
    "label, nodata, scale, offset, expected_dtype",
    [
        # Nothing to mask or scale: stays integer on both paths (the #3066 fix).
        ("bare", None, None, None, "uint8"),
        # Fittable sentinel: masking promotes on both paths.
        ("masked", 6, None, None, "float64"),
        # Non-identity scale/offset: scaling promotes on both paths.
        ("scaled", None, 2.0, 10.0, "float64"),
        # Both: promotes on both paths.
        ("masked_scaled", 6, 2.0, 10.0, "float64"),
    ],
)
def test_eager_dask_dtype_parity(tmp_path, label, nodata, scale, offset,
                                 expected_dtype):
    path = _write(
        tmp_path / f"src_{label}_3066.tif", _DATA,
        nodata=nodata, scale=scale, offset=offset)

    eager = open_geotiff(path, unpack=True)
    lazy = open_geotiff(path, unpack=True, chunks=2)

    assert str(eager.dtype) == expected_dtype
    assert str(lazy.dtype) == expected_dtype
    np.testing.assert_allclose(
        np.asarray(eager.data), np.asarray(lazy.data.compute()),
        equal_nan=True)
