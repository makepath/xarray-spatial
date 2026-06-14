"""``unpack`` docstring matches behaviour for scale/offset-free sources (#3263).

The ``unpack`` parameter doc used to say "A source without scale / offset
metadata is a no-op", but ``unpack=True`` folds into the masking gate
(``_finalize_eager_read``: ``mask_and_scale`` implies masking, matching
rioxarray), so a sentinel-bearing integer source still gets its sentinel
replaced with NaN and is promoted to float64. Only a source with neither
scale/offset metadata nor a nodata sentinel reads unchanged.

These tests pin the corrected docstring wording and the behaviour it now
describes. Dtype parity across the eager/dask paths is covered separately
in ``test_mask_and_scale_dtype_parity_3066.py``.
"""
import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _write(path, *, nodata=None):
    data = np.array([[1, 2], [3, 60000]], dtype=np.uint16)
    attrs = {"crs": 4326}
    if nodata is not None:
        attrs["nodata"] = nodata
    to_geotiff(
        xr.DataArray(
            data, dims=("y", "x"),
            coords={"y": [2.0, 1.0], "x": [0.5, 1.5]},
            attrs=attrs),
        path)
    return path


def test_unpack_doc_does_not_claim_noop_3263():
    """The docstring no longer calls the scale/offset-free case a no-op."""
    doc = open_geotiff.__doc__
    # Scope to the ``unpack`` paragraph so an unrelated parameter
    # documenting a legitimate no-op case cannot fail this pin.
    start = doc.index("unpack : bool")
    end = doc.index("mask_and_scale : bool", start)
    unpack_doc = doc[start:end]
    assert "is a no-op" not in unpack_doc
    # The corrected wording names what actually happens.
    assert "skips the scaling step" in unpack_doc
    assert "sentinel-to-NaN mask still runs" in unpack_doc


def test_unpack_sentinel_only_source_masks_and_promotes_3263(tmp_path):
    """No SCALE/OFFSET, declared sentinel: unpack masks + promotes anyway."""
    path = _write(tmp_path / "t3263_sentinel.tif", nodata=60000)

    plain = open_geotiff(path)
    unpacked = open_geotiff(path, unpack=True)

    # Without unpack the read is untouched.
    assert str(plain.dtype) == "uint16"
    assert plain.values[1, 1] == 60000

    # With unpack the sentinel masks to NaN and the dtype promotes, even
    # though there is no scale/offset to apply.
    assert str(unpacked.dtype) == "float64"
    assert np.isnan(unpacked.values[1, 1])
    np.testing.assert_array_equal(
        unpacked.values[[0, 0, 1], [0, 1, 0]], [1.0, 2.0, 3.0])
    assert "scale_factor" not in unpacked.attrs
    assert unpacked.attrs["mask_and_scale_dtype"] == "uint16"


def test_unpack_bare_source_reads_unchanged_3263(tmp_path):
    """Neither scale/offset nor sentinel: unpack really is a no-op."""
    path = _write(tmp_path / "t3263_bare.tif")

    unpacked = open_geotiff(path, unpack=True)

    assert str(unpacked.dtype) == "uint16"
    assert unpacked.values[1, 1] == 60000
    assert "scale_factor" not in unpacked.attrs
    assert "mask_and_scale_dtype" not in unpacked.attrs
