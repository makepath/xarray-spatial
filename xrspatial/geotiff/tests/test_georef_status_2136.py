"""Canonical ``attrs['georef_status']`` covers the five reader states (#2136).

Pre-#2136 the read paths emitted three independent signals --
``attrs['crs']`` / ``attrs['crs_wkt']``, ``attrs['transform']``, and the
``_xrspatial_no_georef`` marker -- and downstream code had to reconcile
them by hand. Two distinct on-disk situations (rotated-with-``allow_rotated``
vs truly-no-transform) looked identical via the public attrs.

Contract v3 adds one canonical attr that encodes the five distinct
states the reader can land in:

* ``full``              -- CRS resolved + axis-aligned transform.
* ``transform_only``    -- transform present, no CRS.
* ``crs_only``          -- CRS present, no transform tags.
* ``none``              -- neither CRS nor transform.
* ``rotated_dropped``   -- rotated transform tags dropped under
                            ``allow_rotated=True``.

The attr is additive: ``crs`` / ``crs_wkt`` / ``transform`` /
``_xrspatial_no_georef`` keep their existing semantics. The tests here
pin the value per reader state and on round-trip.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_vrt, to_geotiff
from xrspatial.geotiff._attrs import (_ATTRS_CONTRACT_VERSION, GEOREF_STATUS_CRS_ONLY,
                                      GEOREF_STATUS_FULL, GEOREF_STATUS_NONE,
                                      GEOREF_STATUS_ROTATED_DROPPED, GEOREF_STATUS_TRANSFORM_ONLY,
                                      _compute_georef_status, _compute_georef_status_from_parts)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY
from xrspatial.geotiff._errors import RotatedTransformError
from xrspatial.geotiff._geotags import GeoInfo, GeoTransform

tifffile = pytest.importorskip("tifffile")

# Reuse the rotated-TIFF writer from the #2115 test rather than copying
# the byte layout. The function is private to that test module but
# the test runner sees the package directory so the import succeeds.
from xrspatial.geotiff.tests.test_allow_rotated_geotiff_2115 import \
    _write_rotated_tiff  # noqa: E402

_STATUS_KEY = 'georef_status'


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


# ---------------------------------------------------------------------------
# Unit tests on the helpers
# ---------------------------------------------------------------------------


def test_contract_version_is_at_least_three():
    """The ``georef_status`` attr lands in v3; pin a lower bound so future
    contract bumps that keep the attr (e.g. ``rotated_affine`` in v4 /
    issue #2129) do not regress this test."""
    assert _ATTRS_CONTRACT_VERSION >= 3


def test_public_constants_reexported():
    """The five status constants and ``GEOREF_STATUS_VALUES`` are part
    of the public surface (issue #2136 / review follow-up). Downstream
    consumers should be able to import them from ``xrspatial.geotiff``
    rather than reaching into the private ``_attrs`` module."""
    import xrspatial.geotiff as pkg
    assert pkg.GEOREF_STATUS_FULL == GEOREF_STATUS_FULL
    assert pkg.GEOREF_STATUS_TRANSFORM_ONLY == GEOREF_STATUS_TRANSFORM_ONLY
    assert pkg.GEOREF_STATUS_CRS_ONLY == GEOREF_STATUS_CRS_ONLY
    assert pkg.GEOREF_STATUS_NONE == GEOREF_STATUS_NONE
    assert pkg.GEOREF_STATUS_ROTATED_DROPPED == GEOREF_STATUS_ROTATED_DROPPED
    assert pkg.GEOREF_STATUS_VALUES == frozenset({
        'full', 'transform_only', 'crs_only', 'none', 'rotated_dropped',
    })


def test_compute_status_full():
    info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=1.0, pixel_height=-1.0),
        has_georef=True, crs_epsg=4326,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_FULL


def test_compute_status_transform_only():
    info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=1.0, pixel_height=-1.0),
        has_georef=True,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_TRANSFORM_ONLY


def test_compute_status_crs_only():
    info = GeoInfo(has_georef=False, crs_epsg=4326)
    assert _compute_georef_status(info) == GEOREF_STATUS_CRS_ONLY


def test_compute_status_crs_only_wkt_no_epsg():
    """``crs_wkt`` without an EPSG code is still ``crs_only``: the
    decision rule treats either CRS signal as present."""
    info = GeoInfo(has_georef=False, crs_wkt='GEOGCRS["custom"]')
    assert _compute_georef_status(info) == GEOREF_STATUS_CRS_ONLY


def test_compute_status_none():
    info = GeoInfo(has_georef=False)
    assert _compute_georef_status(info) == GEOREF_STATUS_NONE


def test_compute_status_rotated_dropped():
    """``rotated_affine`` set on the transform is the dropped-rotation
    signal; the value wins over CRS presence and ``has_georef``."""
    info = GeoInfo(
        transform=GeoTransform(rotated_affine=(1.0, 0.5, 0.0, 0.5, -1.0, 0.0)),
        has_georef=False, crs_epsg=4326,
    )
    assert _compute_georef_status(info) == GEOREF_STATUS_ROTATED_DROPPED


@pytest.mark.parametrize(
    "has_transform,has_crs,rotated,expected",
    [
        (True, True, False, GEOREF_STATUS_FULL),
        (True, False, False, GEOREF_STATUS_TRANSFORM_ONLY),
        (False, True, False, GEOREF_STATUS_CRS_ONLY),
        (False, False, False, GEOREF_STATUS_NONE),
        (False, False, True, GEOREF_STATUS_ROTATED_DROPPED),
        (True, True, True, GEOREF_STATUS_ROTATED_DROPPED),
    ],
)
def test_compute_status_from_parts(has_transform, has_crs, rotated, expected):
    """The VRT-side helper mirrors the GeoInfo decision exactly."""
    assert _compute_georef_status_from_parts(
        has_transform=has_transform, has_crs=has_crs,
        rotated_dropped=rotated,
    ) == expected


# ---------------------------------------------------------------------------
# Read-path coverage: TIFF
# ---------------------------------------------------------------------------


def _make_full_tiff(path):
    """Float coords + CRS -> ``full``. Goes through to_geotiff so the
    fixture exercises the same writer path real callers hit."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, path)


def _make_transform_only_tiff(path):
    """Float coords, no CRS -> ``transform_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, path)


def _make_crs_only_tiff(path):
    """No-georef marker + CRS -> ``crs_only``."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, path)


def _make_none_tiff(path):
    """No-georef marker, no CRS -> ``none``. Equivalent to a plain image."""
    arr = np.zeros((4, 4), dtype=np.float32)
    # Write a bare TIFF with no GeoTIFF tags at all -- this is the
    # tightest 'none' fixture; it has no GeoKeyDirectory, no transform
    # tags, and no marker on disk. The reader stamps the marker on read.
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )


def test_full_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_full.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL
    assert 'transform' in rd.attrs
    assert rd.attrs.get('crs') == 4326


def test_transform_only_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_xform_only.tif")
    _make_transform_only_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_TRANSFORM_ONLY
    assert 'transform' in rd.attrs
    assert 'crs' not in rd.attrs
    # ``crs_wkt`` may be absent or present depending on whether the
    # writer emitted GeoKey citation text; either way the status is the
    # source of truth for "no CRS".


def test_crs_only_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_crs_only.tif")
    _make_crs_only_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_CRS_ONLY
    assert rd.attrs.get('crs') == 4326
    assert 'transform' not in rd.attrs
    # Reader still stamps the legacy marker for back-compat. The new
    # attr is additive; pinning both keys here so a future change that
    # drops one is caught at the same site.
    assert rd.attrs.get(_NO_GEOREF_KEY) is True


def test_none_status(tmp_path):
    path = str(tmp_path / "georef_status_2136_none.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE
    assert 'transform' not in rd.attrs
    assert 'crs' not in rd.attrs
    assert rd.attrs.get(_NO_GEOREF_KEY) is True


def test_rotated_dropped_status(tmp_path):
    """Rotated ``ModelTransformationTag`` + ``allow_rotated=True`` is
    the only state where the attr disambiguates a case that all four
    other public signals were silent about."""
    path = str(tmp_path / "georef_status_2136_rotated.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(path, arr)
    rd = open_geotiff(path, allow_rotated=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_ROTATED_DROPPED
    # ``transform`` is intentionally absent on the dropped path -- a
    # rotated affine cannot be expressed in the axis-aligned 6-tuple.
    assert 'transform' not in rd.attrs


def test_rotated_default_still_raises(tmp_path):
    """Default ``allow_rotated=False`` keeps the existing refusal so the
    rotated_dropped state is only reachable via the explicit opt-in.
    Pinned here so the status attr does not accidentally relax the
    refusal contract from #2115."""
    path = str(tmp_path / "georef_status_2136_rotated_default.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(path, arr)
    with pytest.raises(RotatedTransformError, match="rotation"):
        open_geotiff(path)


# ---------------------------------------------------------------------------
# Dask-backed read path stamps the same attr
# ---------------------------------------------------------------------------


def test_full_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_full_dask.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


def test_none_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_none_dask.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


def test_rotated_status_dask(tmp_path):
    path = str(tmp_path / "georef_status_2136_rotated_dask.tif")
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    _write_rotated_tiff(path, arr)
    rd = open_geotiff(path, allow_rotated=True, chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_ROTATED_DROPPED


# ---------------------------------------------------------------------------
# GPU read path stamps the same attr
# ---------------------------------------------------------------------------


@_gpu_only
def test_full_status_gpu(tmp_path):
    path = str(tmp_path / "georef_status_2136_full_gpu.tif")
    _make_full_tiff(path)
    rd = open_geotiff(path, gpu=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


@_gpu_only
def test_none_status_gpu(tmp_path):
    path = str(tmp_path / "georef_status_2136_none_gpu.tif")
    _make_none_tiff(path)
    rd = open_geotiff(path, gpu=True)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


# ---------------------------------------------------------------------------
# VRT read paths stamp the same attr (eager + chunked).
# ---------------------------------------------------------------------------


def _write_vrt(vrt_path, source_name, *, height, width,
               crs_wkt=None, geo_transform=None):
    """Write a minimal VRT with optional CRS + geo_transform elements."""
    crs_elem = (
        f'  <SRS>{crs_wkt}</SRS>\n' if crs_wkt else ''
    )
    gt_elem = (
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        if geo_transform else ''
    )
    vrt_path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'{crs_elem}{gt_elem}'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_vrt_full_status(tmp_path):
    src = tmp_path / "georef_status_2136_vrt_full_src.tif"
    _make_full_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_full.vrt"
    _write_vrt(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
        geo_transform='100.0, 1.0, 0.0, 200.5, 0.0, -1.0',
    )
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


def test_vrt_crs_only_status(tmp_path):
    """VRT with SRS but no GeoTransform element -> ``crs_only``."""
    src = tmp_path / "georef_status_2136_vrt_crsonly_src.tif"
    _make_none_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_crsonly.vrt"
    _write_vrt(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
    )
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_CRS_ONLY


def test_vrt_none_status(tmp_path):
    """VRT with neither SRS nor GeoTransform -> ``none``."""
    src = tmp_path / "georef_status_2136_vrt_none_src.tif"
    _make_none_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_none.vrt"
    _write_vrt(vrt, os.path.basename(src), height=4, width=4)
    rd = read_vrt(str(vrt))
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_NONE


def test_vrt_full_status_chunked(tmp_path):
    src = tmp_path / "georef_status_2136_vrt_full_chunked_src.tif"
    _make_full_tiff(str(src))
    vrt = tmp_path / "georef_status_2136_vrt_full_chunked.vrt"
    _write_vrt(
        vrt, os.path.basename(src), height=4, width=4,
        crs_wkt='EPSG:4326',
        geo_transform='100.0, 1.0, 0.0, 200.5, 0.0, -1.0',
    )
    rd = read_vrt(str(vrt), chunks=2)
    assert rd.attrs[_STATUS_KEY] == GEOREF_STATUS_FULL


# ---------------------------------------------------------------------------
# Round-trip stability: the attr survives write -> read for every state
# that has a writable encoding (full, transform_only, crs_only, none).
# rotated_dropped is read-only by nature: ``to_geotiff`` does not emit
# rotated ``ModelTransformationTag`` entries, so a write would
# axis-align the matrix. That asymmetry is documented in the issue
# under "Out of scope" / "rotated_dropped is for consumers".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make,expected",
    [
        (_make_full_tiff, GEOREF_STATUS_FULL),
        (_make_transform_only_tiff, GEOREF_STATUS_TRANSFORM_ONLY),
        (_make_crs_only_tiff, GEOREF_STATUS_CRS_ONLY),
        (_make_none_tiff, GEOREF_STATUS_NONE),
    ],
)
def test_roundtrip_preserves_status(tmp_path, make, expected):
    """Write a fixture of each writable state, read it, then write +
    read again. The status attr is stable across the cycle because the
    underlying CRS / transform decisions the writer makes are
    deterministic from the input attrs.

    The fixture set deliberately overlaps with the per-state tests
    above; the value of this parametrised test is the *cycle*, not the
    one-shot read.
    """
    p1 = str(tmp_path / f"georef_status_2136_rt_{expected}_1.tif")
    make(p1)
    rd1 = open_geotiff(p1)
    assert rd1.attrs[_STATUS_KEY] == expected

    p2 = str(tmp_path / f"georef_status_2136_rt_{expected}_2.tif")
    to_geotiff(rd1, p2)
    rd2 = open_geotiff(p2)
    assert rd2.attrs[_STATUS_KEY] == expected
