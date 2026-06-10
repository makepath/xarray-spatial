"""``to_geotiff(pack=True)`` after a band-subset ``unpack=True`` read (#3161).

``_pack`` keeps the source's GDAL_METADATA so the packed file declares how
to unpack. After a band-subset read, per-band ``(SCALE, i)`` / ``(OFFSET, i)``
entries still describe the source's band indices, which the written file no
longer has: re-reading the packed file raised ``MixedBandMetadataError``, and
``band=0`` silently applied the wrong band's scale. ``_pack`` now rewrites
per-band entries as dataset-level values carrying the (scale, offset) pair
that was actually applied.

``mask_and_scale`` is a CPU eager + dask read feature, so the round-trip is
exercised on numpy and dask.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _write_two_band_tiff(path, *, gdal_metadata, nodata=65535):
    """2-band uint16 file; band 0 holds 1..6, band 1 holds 10..60."""
    data = np.dstack([
        np.arange(1, 7).reshape(2, 3),
        np.arange(10, 70, 10).reshape(2, 3),
    ]).astype(np.uint16)
    da = xr.DataArray(
        data,
        dims=("y", "x", "band"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
        attrs={"crs": 4326, "nodata": nodata, "gdal_metadata": gdal_metadata},
    )
    to_geotiff(da, path)
    return path, data


def _open(path, chunks, **kwargs):
    if chunks is not None:
        kwargs["chunks"] = chunks
    return open_geotiff(path, **kwargs)


# ---------------------------------------------------------------------------
# The issue repro: distinct per-band SCALE, band-subset read, pack, re-read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_band_subset_round_trips_per_band_scale(tmp_path, chunks):
    src, data = _write_two_band_tiff(
        tmp_path / "src_two_band_3161.tif",
        gdal_metadata={("SCALE", 0): "0.1", ("SCALE", 1): "0.2"},
    )

    decoded = _open(src, chunks, band=1, unpack=True)
    np.testing.assert_allclose(
        np.asarray(decoded.data), data[:, :, 1].astype(np.float64) * 0.2)

    out = str(tmp_path / "packed_subset_3161.tif")
    to_geotiff(decoded, out, pack=True)

    # The raw integers round-trip exactly.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint16"
    np.testing.assert_array_equal(np.asarray(raw.data), data[:, :, 1])

    # Re-reading with unpack=True used to raise MixedBandMetadataError
    # (the packed file declared both source bands' SCALE values).
    back = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(back.data), np.asarray(decoded.data), equal_nan=True)

    # band=0 (the only band of the output) used to apply the stale
    # band-0 SCALE of the source (0.1) instead of the applied 0.2.
    back0 = open_geotiff(out, unpack=True, band=0)
    np.testing.assert_allclose(
        np.asarray(back0.data), np.asarray(decoded.data), equal_nan=True)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_band_subset_per_band_offset(tmp_path, chunks):
    src, data = _write_two_band_tiff(
        tmp_path / "src_offset_3161.tif",
        gdal_metadata={
            ("SCALE", 0): "0.5", ("SCALE", 1): "2.0",
            ("OFFSET", 0): "-1.0", ("OFFSET", 1): "3.0",
        },
    )

    decoded = _open(src, chunks, band=1, unpack=True)
    np.testing.assert_allclose(
        np.asarray(decoded.data), data[:, :, 1].astype(np.float64) * 2.0 + 3.0)

    out = str(tmp_path / "packed_offset_3161.tif")
    to_geotiff(decoded, out, pack=True)

    back = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(back.data), np.asarray(decoded.data), equal_nan=True)
    md = back.attrs.get("gdal_metadata") or {}
    assert not any(isinstance(k, tuple) for k in md)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_band_subset_selected_band_without_scale(tmp_path, chunks):
    """Selected band carries no SCALE entry; the other band's stale entry
    must not survive into the packed file and get applied on re-read."""
    src, data = _write_two_band_tiff(
        tmp_path / "src_noscale_3161.tif",
        gdal_metadata={("SCALE", 1): "0.2"},
    )

    decoded = _open(src, chunks, band=0, unpack=True)  # identity scale
    np.testing.assert_allclose(
        np.asarray(decoded.data), data[:, :, 0].astype(np.float64))

    out = str(tmp_path / "packed_noscale_3161.tif")
    to_geotiff(decoded, out, pack=True)

    back = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(back.data), np.asarray(decoded.data), equal_nan=True)


# ---------------------------------------------------------------------------
# Full (band=None) reads keep round-tripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_full_read_uniform_per_band_scale(tmp_path, chunks):
    """A full read of uniform per-band SCALE entries still unpacks to the
    same values after pack: the entries collapse to one dataset-level pair."""
    src, data = _write_two_band_tiff(
        tmp_path / "src_uniform_3161.tif",
        gdal_metadata={("SCALE", 0): "0.5", ("SCALE", 1): "0.5"},
    )

    decoded = _open(src, chunks, unpack=True)
    out = str(tmp_path / "packed_uniform_3161.tif")
    to_geotiff(decoded, out, pack=True)

    back = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(back.data), np.asarray(decoded.data), equal_nan=True)
    np.testing.assert_array_equal(
        np.asarray(open_geotiff(out).data), data)


def test_pack_dataset_level_metadata_kept_verbatim(tmp_path):
    """Dataset-level SCALE/OFFSET is index-free, so ``_pack`` leaves the
    metadata (and the raw GDAL_METADATA XML) untouched."""
    from xrspatial.geotiff._attrs import _pack

    src, _ = _write_two_band_tiff(
        tmp_path / "src_dslevel_3161.tif",
        gdal_metadata={"SCALE": "0.25", "OFFSET": "2.0"},
    )

    decoded = open_geotiff(src, band=1, unpack=True)
    xml_before = decoded.attrs.get("gdal_metadata_xml")
    assert xml_before is not None

    packed = _pack(decoded)
    assert packed.attrs.get("gdal_metadata") == decoded.attrs["gdal_metadata"]
    assert packed.attrs.get("gdal_metadata_xml") == xml_before
