"""Cross-backend write-path attrs parity tests for issue #1563.

Before the fix, ``write_geotiff_gpu`` silently dropped most metadata
attrs that the CPU ``to_geotiff`` preserves:

* ``crs_wkt`` (the WKT-only CRS fallback)
* ``gdal_metadata`` / ``gdal_metadata_xml``
* ``extra_tags``
* ``image_description`` / ``extra_samples`` / ``colormap``
* ``x_resolution`` / ``y_resolution`` / ``resolution_unit``

It also ignored ``attrs['transform']`` and re-derived the GeoTransform
from coords, which drifts on fractional pixel sizes.

These tests pin the contract that the GPU writer mirrors the CPU
writer's attr-resolution logic.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, write_geotiff_gpu
from xrspatial.geotiff._geotags import GeoTransform, _epsg_to_wkt
from xrspatial.geotiff._writer import write


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


@_gpu_only
def test_crs_wkt_only_attr_round_trips(tmp_path):
    """When the source CRS only resolves to WKT (no EPSG attr), the GPU
    writer must still emit GeoKeys so the file has a CRS at all."""
    import cupy
    wkt = _epsg_to_wkt(4326)
    if wkt is None:
        pytest.skip("pyproj not available")
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs_wkt': wkt},
    )
    out = str(tmp_path / 'crs_wkt_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    # The WKT should resolve back to EPSG 4326 on read.
    assert rd.attrs.get('crs') == 4326, (
        f"crs_wkt was dropped on the GPU write path; "
        f"got attrs={sorted(rd.attrs.keys())}"
    )


@_gpu_only
def test_image_description_round_trips_via_gpu_writer(tmp_path):
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'image_description': 'gpu-attr-test-1563'},
    )
    out = str(tmp_path / 'desc_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('image_description') == 'gpu-attr-test-1563'


@_gpu_only
def test_extra_samples_single_band_writer_compat(tmp_path):
    """Single-band write with ``extra_samples`` set must not crash, even
    though TIFF 6.0 only surfaces ExtraSamples on multi-sample images
    (the reader drops it for 1-sample files). The multi-band case is
    covered by ``test_extra_samples_round_trips_multiband_via_gpu_writer``
    below.
    """
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'extra_samples': [1]},
    )
    out = str(tmp_path / 'es_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('crs') == 4326


@_gpu_only
def test_extra_samples_round_trips_multiband_via_gpu_writer(tmp_path):
    """Multi-band write: ExtraSamples surfaces on the reader because
    SamplesPerPixel > 1. The assembler auto-synthesizes the tag for
    multi-band minisblack (same behaviour as the CPU writer), so we
    just assert the attr appears -- pinning that the GPU path reaches
    the multi-band writer code without dropping ExtraSamples entirely.
    """
    import cupy
    arr = np.zeros((4, 5, 2), dtype=np.uint8)
    arr[..., 1] = 255
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x', 'band'],
        coords={'y': np.arange(4.0), 'x': np.arange(5.0)},
        attrs={'crs': 4326},
    )
    out = str(tmp_path / 'es_multi_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    es = rd.attrs.get('extra_samples')
    assert es is not None, (
        f"extra_samples dropped on multi-band GPU write; "
        f"attrs={sorted(rd.attrs.keys())}"
    )


@_gpu_only
def test_colormap_round_trips_via_gpu_writer(tmp_path):
    """A uint8 raster with a 768-entry colormap (3*256 uint16 values for
    R/G/B) must surface ``attrs['colormap']`` after the GPU round-trip --
    the synthesized tag 320 entry rides in ``extra_tags`` and the read
    path projects it back onto the friendly attr."""
    import cupy
    palette = []
    for ch_offset in (0, 1, 2):
        for i in range(256):
            palette.append((i * 257 + ch_offset) & 0xFFFF)
    assert len(palette) == 768
    pixels = np.array([[0, 1, 2, 254, 255],
                       [10, 20, 30, 40, 50]], dtype=np.uint8)
    da_gpu = xr.DataArray(
        cupy.asarray(pixels),
        dims=['y', 'x'],
        coords={'y': np.arange(2.0), 'x': np.arange(5.0)},
        attrs={'crs': 4326, 'colormap': palette},
    )
    out = str(tmp_path / 'cmap_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    rd_cmap = rd.attrs.get('colormap')
    assert rd_cmap is not None, (
        f"colormap dropped on GPU write; attrs={sorted(rd.attrs.keys())}"
    )
    assert tuple(rd_cmap) == tuple(palette)


@_gpu_only
def test_extra_tags_custom_tag_round_trips_via_gpu_writer(tmp_path):
    """A user-supplied ``extra_tags`` entry (here: Software, tag 305,
    ASCII) must be forwarded to ``_assemble_tiff`` and reappear in
    ``attrs['extra_tags']`` on read."""
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    software = "xrspatial-1563-test"
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'extra_tags': [(305, 2, len(software) + 1, software)],
        },
    )
    out = str(tmp_path / 'extra_tags_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    et = rd.attrs.get('extra_tags') or []
    by_id = {t[0]: t for t in et}
    assert 305 in by_id, (
        f"extra_tags Software (305) dropped on GPU write; got ids "
        f"{sorted(by_id.keys())}"
    )
    value = by_id[305][3]
    # ASCII values may be returned as bytes or str; strip NUL terminator.
    if isinstance(value, bytes):
        value = value.decode('ascii')
    assert value.rstrip('\x00') == software


@_gpu_only
def test_resolution_tags_round_trip_via_gpu_writer(tmp_path):
    """``x_resolution`` / ``y_resolution`` / ``resolution_unit`` must
    survive a GPU write -> CPU read cycle. The writer maps the unit
    string back to its TIFF id (1=none, 2=inch, 3=centimeter)."""
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={
            'crs': 4326,
            'x_resolution': 300.0,
            'y_resolution': 300.0,
            'resolution_unit': 'inch',
        },
    )
    out = str(tmp_path / 'res_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('x_resolution') == pytest.approx(300.0, rel=0.01), (
        f"x_resolution drift: got {rd.attrs.get('x_resolution')}"
    )
    assert rd.attrs.get('y_resolution') == pytest.approx(300.0, rel=0.01), (
        f"y_resolution drift: got {rd.attrs.get('y_resolution')}"
    )
    assert rd.attrs.get('resolution_unit') == 'inch', (
        f"resolution_unit drift: got {rd.attrs.get('resolution_unit')}"
    )


@_gpu_only
def test_gdal_metadata_round_trips_via_gpu_writer(tmp_path):
    import cupy
    arr = np.zeros((8, 8), dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(8.0), 'x': np.arange(8.0)},
        attrs={'crs': 4326,
               'gdal_metadata': {'AREA_OR_POINT': 'Area',
                                 'CUSTOM_KEY': 'val_1563'}},
    )
    out = str(tmp_path / 'gdal_meta_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    meta = rd.attrs.get('gdal_metadata') or {}
    assert meta.get('CUSTOM_KEY') == 'val_1563', (
        f"gdal_metadata was dropped on the GPU write path; "
        f"got {meta}"
    )


@_gpu_only
def test_transform_attr_round_trip_bit_stable(tmp_path):
    """Reading a file with a fractional pixel size, then writing it back
    through ``write_geotiff_gpu`` must preserve ``attrs['transform']``
    bit-for-bit (the CPU writer guarantees this; the GPU writer used to
    drop the attr and recompute from coords, which drifts).
    """
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    gt = GeoTransform(
        origin_x=-122.123456789,
        origin_y=37.987654321,
        pixel_width=1.0 / 3600.0 + 1e-12,
        pixel_height=-(1.0 / 3600.0 + 1e-12),
    )
    src = str(tmp_path / 'frac_in_1563.tif')
    write(arr, src, geo_transform=gt, crs_epsg=4326,
          compression='none', tiled=False)
    eager_in = open_geotiff(src)

    da_gpu = open_geotiff(src, gpu=True)
    out = str(tmp_path / 'frac_out_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    after = open_geotiff(out)
    assert after.attrs['transform'] == eager_in.attrs['transform'], (
        f"transform drifted on GPU round-trip:\n"
        f"  in : {eager_in.attrs['transform']}\n"
        f"  out: {after.attrs['transform']}"
    )


@_gpu_only
def test_no_data_attr_still_round_trips_after_fix(tmp_path):
    """Regression guard: the existing nodata + raster_type pass-through
    still works after wiring in the new attrs."""
    import cupy
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    da_gpu = xr.DataArray(
        cupy.asarray(arr),
        dims=['y', 'x'],
        coords={'y': np.arange(2.0), 'x': np.arange(2.0)},
        attrs={'crs': 4326, 'nodata': -9999.0, 'raster_type': 'point'},
    )
    out = str(tmp_path / 'nodata_1563.tif')
    write_geotiff_gpu(da_gpu, out, compression='none')

    rd = open_geotiff(out)
    assert rd.attrs.get('nodata') == -9999.0
    assert rd.attrs.get('raster_type') == 'point'
