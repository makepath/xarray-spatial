"""rioxarray ``open_rasterio`` compatibility for ``open_geotiff`` (#2961).

Covers the renamed parameters and the masking-off default flip:

* ``masked`` (canonical) <- ``mask_nodata`` (deprecated alias), default
  flipped from True to False to match rioxarray.
* ``default_name`` (canonical) <- ``name`` (deprecated alias).
* ``mask_and_scale`` (new): apply GDAL SCALE/OFFSET + mask.
* ``parse_coordinates`` (new): skip x/y coords.
* ``lock`` / ``cache`` (new, accept-and-warn shims).
* GPU / VRT gating for ``mask_and_scale`` / ``parse_coordinates=False``.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    MalformedScaleOffsetError, MixedBandMetadataError, _build_vrt,
    _read_geotiff_dask, _read_geotiff_gpu, _read_vrt, open_geotiff, to_geotiff)
from xrspatial.geotiff._runtime import GeoTIFFFallbackWarning
from xrspatial.geotiff.tests._helpers.markers import requires_gpu


def _int_sentinel_tiff(path, sentinel=255):
    """uint8 raster with one pixel equal to ``sentinel`` declared nodata."""
    data = np.array([[1, 2, sentinel], [4, 5, 6]], dtype=np.uint8)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5, 2.5]},
        attrs={"nodata": sentinel, "crs": 4326},
    )
    to_geotiff(da, path)
    return path


def _scale_offset_tiff(path, scale=2.0, offset=10.0, sentinel=255):
    """uint8 raster carrying GDAL SCALE/OFFSET metadata + a nodata pixel."""
    data = np.array([[1, 2, 3], [4, 5, sentinel]], dtype=np.uint8)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5, 2.5]},
        attrs={
            "nodata": sentinel,
            "crs": 4326,
            "gdal_metadata": {"SCALE": str(scale), "OFFSET": str(offset)},
        },
    )
    to_geotiff(da, path)
    return path


# ---------------------------------------------------------------------------
# masked default flip + mask_nodata deprecation alias
# ---------------------------------------------------------------------------

def test_default_does_not_mask(tmp_path):
    """A bare read leaves the sentinel in place (rioxarray masked=False)."""
    path = _int_sentinel_tiff(str(tmp_path / "t2961_default.tif"))
    out = open_geotiff(path)
    assert out.dtype == np.uint8
    assert (out.data == 255).any()
    assert not np.isnan(out.data.astype(float)).any()
    assert out.attrs.get("masked_nodata") is False
    # The raw sentinel is still on attrs either way.
    assert out.attrs.get("nodata") == 255


# ---------------------------------------------------------------------------
# direct backend defaults match open_geotiff's unmasked default (#2976)
#
# The three direct backend entry points (_read_geotiff_dask,
# _read_geotiff_gpu, _read_vrt) used to default to mask_nodata=True while
# open_geotiff defaults to masked=False. A bare backend call therefore
# returned a different dtype + NaN-substituted values than the public path.
# These tests pin the backends to the public unmasked default.
# ---------------------------------------------------------------------------

def test_read_geotiff_dask_default_matches_open_geotiff_2976(tmp_path):
    """Bare ``_read_geotiff_dask`` keeps the source dtype and sentinel,
    matching ``open_geotiff(path, chunks=...)``."""
    path = _int_sentinel_tiff(str(tmp_path / "t2976_dask.tif"))
    public = open_geotiff(path, chunks=2).compute()
    direct = _read_geotiff_dask(path, chunks=2).compute()

    assert direct.dtype == public.dtype == np.uint8
    assert (direct.data == 255).any()
    assert not np.isnan(direct.data.astype(float)).any()
    assert direct.attrs.get("masked_nodata") is False
    assert direct.attrs.get("nodata") == public.attrs.get("nodata") == 255
    np.testing.assert_array_equal(direct.data, public.data)


def test_read_vrt_default_matches_open_geotiff_2976(tmp_path):
    """Bare ``_read_vrt`` keeps the source dtype and sentinel, matching
    ``open_geotiff(<vrt>)``."""
    src = _int_sentinel_tiff(str(tmp_path / "t2976_vrt_src.tif"))
    vrt = _build_vrt(str(tmp_path / "t2976.vrt"), source_files=[src])
    public = open_geotiff(vrt)
    direct = _read_vrt(vrt)

    assert direct.dtype == public.dtype == np.uint8
    assert (np.asarray(direct.values) == 255).any()
    assert not np.isnan(np.asarray(direct.values, dtype=float)).any()
    assert direct.attrs.get("masked_nodata") is False
    assert direct.attrs.get("nodata") == public.attrs.get("nodata") == 255
    np.testing.assert_array_equal(
        np.asarray(direct.values), np.asarray(public.values))


@requires_gpu
def test_read_geotiff_gpu_default_matches_open_geotiff_2976(tmp_path):
    """Bare ``_read_geotiff_gpu`` keeps the source dtype and sentinel,
    matching ``open_geotiff(path, gpu=True)``."""
    path = _int_sentinel_tiff(str(tmp_path / "t2976_gpu.tif"))
    public = open_geotiff(path, gpu=True)
    direct = _read_geotiff_gpu(path)

    assert direct.dtype == public.dtype == np.uint8
    direct_np = direct.data.get()
    public_np = public.data.get()
    assert (direct_np == 255).any()
    assert not np.isnan(direct_np.astype(float)).any()
    assert direct.attrs.get("masked_nodata") is False
    assert direct.attrs.get("nodata") == public.attrs.get("nodata") == 255
    np.testing.assert_array_equal(direct_np, public_np)


def test_masked_true_promotes_and_masks(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_masked.tif"))
    out = open_geotiff(path, masked=True)
    assert out.dtype == np.float64
    assert np.isnan(out.data).sum() == 1
    assert out.attrs.get("masked_nodata") is True


def test_mask_nodata_alias_warns_and_matches(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_alias.tif"))
    with pytest.warns(DeprecationWarning, match="mask_nodata.*deprecated"):
        legacy = open_geotiff(path, mask_nodata=True)
    canonical = open_geotiff(path, masked=True)
    np.testing.assert_array_equal(
        np.isnan(legacy.data), np.isnan(canonical.data))
    assert legacy.dtype == canonical.dtype == np.float64


def test_masked_and_mask_nodata_both_raises(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_both.tif"))
    with pytest.raises(TypeError, match="either 'masked' or"):
        open_geotiff(path, masked=True, mask_nodata=True)


def test_canonical_masked_false_emits_no_warning(tmp_path, recwarn):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_nowarn.tif"))
    open_geotiff(path, masked=False)
    assert not [w for w in recwarn.list
                if issubclass(w.category, DeprecationWarning)]


# ---------------------------------------------------------------------------
# default_name / name deprecation alias
# ---------------------------------------------------------------------------

def test_default_name_sets_array_name(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_name.tif"))
    out = open_geotiff(path, default_name="elevation")
    assert out.name == "elevation"


def test_name_alias_warns_and_matches(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_name_alias.tif"))
    with pytest.warns(DeprecationWarning, match="name.*deprecated"):
        out = open_geotiff(path, name="elevation")
    assert out.name == "elevation"


def test_default_name_and_name_both_raises(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_name_both.tif"))
    with pytest.raises(TypeError, match="either 'default_name' or"):
        open_geotiff(path, default_name="a", name="b")


# ---------------------------------------------------------------------------
# mask_and_scale
# ---------------------------------------------------------------------------

def test_mask_and_scale_eager(tmp_path):
    path = _scale_offset_tiff(str(tmp_path / "t2961_ms_eager.tif"))
    out = open_geotiff(path, mask_and_scale=True)
    assert out.dtype.kind == "f"
    # data * 2 + 10, sentinel pixel -> NaN
    expected = np.array([[12.0, 14.0, 16.0], [18.0, 20.0, np.nan]])
    np.testing.assert_array_equal(out.data, expected)
    assert out.attrs.get("scale_factor") == 2.0
    assert out.attrs.get("add_offset") == 10.0


def test_mask_and_scale_dask_matches_eager(tmp_path):
    path = _scale_offset_tiff(str(tmp_path / "t2961_ms_dask.tif"))
    eager = open_geotiff(path, mask_and_scale=True)
    lazy = open_geotiff(path, mask_and_scale=True, chunks=2)
    np.testing.assert_array_equal(eager.data, lazy.compute().data)
    assert lazy.attrs.get("scale_factor") == 2.0


def test_mask_and_scale_no_metadata_is_noop(tmp_path):
    """A source with no SCALE/OFFSET keeps raw values (scale 1, offset 0)."""
    path = _int_sentinel_tiff(str(tmp_path / "t2961_ms_noop.tif"))
    out = open_geotiff(path, mask_and_scale=True)
    # sentinel still masked, but values otherwise unscaled
    assert out.data[0, 0] == 1.0
    assert np.isnan(out.data[0, 2])
    assert "scale_factor" not in out.attrs


def test_mask_and_scale_int_dtype_raises(tmp_path):
    path = _scale_offset_tiff(str(tmp_path / "t2961_ms_int.tif"))
    with pytest.raises(ValueError):
        open_geotiff(path, mask_and_scale=True, dtype="uint8")


# ---------------------------------------------------------------------------
# malformed SCALE/OFFSET rejection (#2987)
# ---------------------------------------------------------------------------

def _malformed_scale_tiff(path, scale="abc", offset="0"):
    """uint8 raster carrying an unparseable SCALE/OFFSET in GDAL_METADATA."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5, 2.5]},
        attrs={
            "crs": 4326,
            "gdal_metadata": {"SCALE": scale, "OFFSET": offset},
        },
    )
    to_geotiff(da, path)
    return path


def test_mask_and_scale_malformed_scale_raises(tmp_path):
    """A present-but-unparseable SCALE fails closed under mask_and_scale."""
    path = _malformed_scale_tiff(str(tmp_path / "t2987_bad_scale.tif"))
    with pytest.raises(MalformedScaleOffsetError, match="SCALE"):
        open_geotiff(path, mask_and_scale=True)


def test_mask_and_scale_malformed_offset_raises(tmp_path):
    path = _malformed_scale_tiff(
        str(tmp_path / "t2987_bad_offset.tif"), scale="1", offset="xyz")
    with pytest.raises(MalformedScaleOffsetError, match="OFFSET"):
        open_geotiff(path, mask_and_scale=True)


def test_mask_and_scale_malformed_scale_dask_raises(tmp_path):
    path = _malformed_scale_tiff(str(tmp_path / "t2987_bad_scale_dask.tif"))
    with pytest.raises(MalformedScaleOffsetError, match="SCALE"):
        open_geotiff(path, mask_and_scale=True, chunks=2)


def test_malformed_scale_ignored_without_mask_and_scale(tmp_path):
    """Without mask_and_scale the metadata is never read, so no rejection."""
    path = _malformed_scale_tiff(str(tmp_path / "t2987_no_ms.tif"))
    out = open_geotiff(path)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out.data, [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# malformed GDAL_METADATA XML rejection (#2998)
# ---------------------------------------------------------------------------

def _malformed_xml_tiff(path):
    """uint8 raster whose GDAL_METADATA tag holds non-well-formed XML.

    The writer escapes every SCALE/OFFSET value it serialises, so a
    malformed payload has to be injected through the raw
    ``gdal_metadata_xml`` writer kwarg (gated behind the rich-tag opt-in).
    The unclosed ``<Item>`` makes ``ET.fromstring`` raise ``ParseError``.
    """
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5, 2.5]},
        attrs={
            "crs": 4326,
            "gdal_metadata_xml": (
                '<GDALMetadata><Item name="SCALE">2.0</GDALMetadata>'
            ),
        },
    )
    to_geotiff(da, path, allow_experimental_codecs=True)
    return path


def test_mask_and_scale_malformed_xml_raises(tmp_path):
    """Malformed GDAL_METADATA XML fails closed under mask_and_scale."""
    path = _malformed_xml_tiff(str(tmp_path / "t2998_bad_xml.tif"))
    with pytest.raises(MalformedScaleOffsetError, match="XML"):
        open_geotiff(path, mask_and_scale=True)


def test_mask_and_scale_malformed_xml_dask_raises(tmp_path):
    path = _malformed_xml_tiff(str(tmp_path / "t2998_bad_xml_dask.tif"))
    with pytest.raises(MalformedScaleOffsetError, match="XML"):
        open_geotiff(path, mask_and_scale=True, chunks=2)


def test_malformed_xml_ignored_without_mask_and_scale(tmp_path):
    """Without mask_and_scale the XML is never parsed for scale/offset."""
    path = _malformed_xml_tiff(str(tmp_path / "t2998_no_ms.tif"))
    out = open_geotiff(path)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out.data, [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# parse_coordinates
# ---------------------------------------------------------------------------

def test_parse_coordinates_false_drops_xy_keeps_attrs(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_pc_eager.tif"))
    out = open_geotiff(path, parse_coordinates=False)
    assert "x" not in out.coords
    assert "y" not in out.coords
    assert "transform" in out.attrs
    assert "crs" in out.attrs


def test_parse_coordinates_true_default_has_xy(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_pc_default.tif"))
    out = open_geotiff(path)
    assert "x" in out.coords
    assert "y" in out.coords


def test_parse_coordinates_false_dask(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_pc_dask.tif"))
    out = open_geotiff(path, parse_coordinates=False, chunks=2)
    assert "x" not in out.coords
    assert "transform" in out.attrs


# ---------------------------------------------------------------------------
# lock / cache accept-and-warn shims
# ---------------------------------------------------------------------------

def test_lock_emits_fallback_warning(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_lock.tif"))
    with pytest.warns(GeoTIFFFallbackWarning, match="lock.*cache"):
        out = open_geotiff(path, lock=object())
    assert out.dtype == np.uint8


def test_cache_false_emits_fallback_warning(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_cache.tif"))
    with pytest.warns(GeoTIFFFallbackWarning, match="lock.*cache"):
        open_geotiff(path, cache=False)


def test_default_lock_cache_no_warning(tmp_path, recwarn):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_lc_default.tif"))
    open_geotiff(path)
    assert not [w for w in recwarn.list
                if issubclass(w.category, GeoTIFFFallbackWarning)]


# ---------------------------------------------------------------------------
# GPU / VRT gating for the new behavioral options
# ---------------------------------------------------------------------------

def test_mask_and_scale_gpu_rejected(tmp_path):
    path = _scale_offset_tiff(str(tmp_path / "t2961_gate_gpu.tif"))
    with pytest.raises(ValueError, match="mask_and_scale.*gpu=True"):
        open_geotiff(path, mask_and_scale=True, gpu=True)


def test_parse_coordinates_false_gpu_rejected(tmp_path):
    path = _int_sentinel_tiff(str(tmp_path / "t2961_gate_gpu_pc.tif"))
    with pytest.raises(ValueError, match="parse_coordinates=False.*gpu=True"):
        open_geotiff(path, parse_coordinates=False, gpu=True)


def test_mask_and_scale_vrt_rejected(tmp_path):
    src = _int_sentinel_tiff(str(tmp_path / "t2961_gate_vrt_src.tif"))
    vrt = _build_vrt(str(tmp_path / "t2961_gate.vrt"), source_files=[src])
    with pytest.raises(ValueError, match="mask_and_scale.*.vrt"):
        open_geotiff(vrt, mask_and_scale=True)


def test_parse_coordinates_false_vrt_rejected(tmp_path):
    src = _int_sentinel_tiff(str(tmp_path / "t2961_gate_vrt_pc_src.tif"))
    vrt = _build_vrt(str(tmp_path / "t2961_gate_pc.vrt"), source_files=[src])
    with pytest.raises(ValueError, match="parse_coordinates=False.*.vrt"):
        open_geotiff(vrt, parse_coordinates=False)


# ---------------------------------------------------------------------------
# mask_and_scale per-band SCALE/OFFSET (#2988)
# ---------------------------------------------------------------------------

def _per_band_scale_tiff(path, scales, offsets):
    """3x4 multi-band float raster with per-band SCALE/OFFSET metadata.

    ``scales`` / ``offsets`` are per-band lists; the raw band ``b`` is filled
    with the constant ``b + 1`` so the expected scaled value is easy to assert.
    """
    n = len(scales)
    arr = np.empty((3, 4, n), dtype=np.float32)
    for b in range(n):
        arr[:, :, b] = b + 1
    meta = {}
    for b, (s, o) in enumerate(zip(scales, offsets)):
        meta[("SCALE", b)] = str(s)
        meta[("OFFSET", b)] = str(o)
    da = xr.DataArray(
        arr,
        dims=("y", "x", "band"),
        coords={
            "y": [0.5, 1.5, 2.5],
            "x": [0.5, 1.5, 2.5, 3.5],
            "band": list(range(n)),
        },
        attrs={"crs": 4326, "gdal_metadata": meta},
    )
    to_geotiff(da, path)
    return path


def test_mask_and_scale_mixed_per_band_eager_raises(tmp_path):
    """Differing per-band scale + no band selection -> MixedBandMetadataError."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_mixed_eager.tif"),
        scales=[2.0, 4.0, 8.0], offsets=[0.0, 0.0, 0.0])
    with pytest.raises(MixedBandMetadataError, match="per-band SCALE"):
        open_geotiff(path, mask_and_scale=True)


def test_mask_and_scale_mixed_per_band_offset_raises(tmp_path):
    """Differing per-band OFFSET is rejected even when SCALE agrees."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_mixed_offset.tif"),
        scales=[2.0, 2.0, 2.0], offsets=[1.0, 5.0, 9.0])
    with pytest.raises(MixedBandMetadataError, match="per-band OFFSET"):
        open_geotiff(path, mask_and_scale=True)


def test_mask_and_scale_mixed_per_band_dask_raises(tmp_path):
    """The dask path rejects the same source at graph-build time."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_mixed_dask.tif"),
        scales=[2.0, 4.0, 8.0], offsets=[0.0, 0.0, 0.0])
    with pytest.raises(MixedBandMetadataError, match="per-band SCALE"):
        open_geotiff(path, mask_and_scale=True, chunks=2)


def test_mask_and_scale_band_selects_own_scale(tmp_path):
    """Selecting a band applies that band's scale/offset, no error."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_band_sel.tif"),
        scales=[2.0, 4.0, 8.0], offsets=[1.0, 5.0, 9.0])
    # band 1: raw value 2, scale 4, offset 5 -> 2 * 4 + 5 = 13.
    out = open_geotiff(path, mask_and_scale=True, band=1)
    assert out.attrs.get("scale_factor") == 4.0
    assert out.attrs.get("add_offset") == 5.0
    np.testing.assert_array_equal(out.data, np.full((3, 4), 13.0))


def test_mask_and_scale_band_select_dask_matches_eager(tmp_path):
    """band= scaling agrees between eager and dask paths."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_band_dask.tif"),
        scales=[2.0, 4.0, 8.0], offsets=[1.0, 5.0, 9.0])
    eager = open_geotiff(path, mask_and_scale=True, band=2)
    lazy = open_geotiff(path, mask_and_scale=True, band=2, chunks=2)
    np.testing.assert_array_equal(eager.data, lazy.compute().data)
    assert lazy.attrs.get("scale_factor") == 8.0


def test_mask_and_scale_uniform_per_band_applies(tmp_path):
    """Per-band values that agree across bands apply to the whole array."""
    path = _per_band_scale_tiff(
        str(tmp_path / "t2988_uniform.tif"),
        scales=[3.0, 3.0, 3.0], offsets=[2.0, 2.0, 2.0])
    out = open_geotiff(path, mask_and_scale=True)
    assert out.attrs.get("scale_factor") == 3.0
    assert out.attrs.get("add_offset") == 2.0
    # band b raw value (b + 1) * 3 + 2.
    expected = np.stack(
        [np.full((3, 4), (b + 1) * 3.0 + 2.0) for b in range(3)], axis=-1)
    np.testing.assert_array_equal(out.data, expected)
