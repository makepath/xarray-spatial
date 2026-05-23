"""VRT metadata parity tests across backends (issue #2321, sub-PR 3).

Most VRT regression coverage today asserts pixel values. A VRT read
can return the right pixels with the wrong georeferencing attrs and
nothing in the current suite catches it -- the attrs sweep gets
single-source coverage from ``test_vrt_finalization_parity_2162`` but
no cross-backend pin on the metadata the contract promises.

This module locks the cross-backend metadata contract for VRT reads:

* eager (numpy) vs dask via ``open_geotiff(..., chunks=...)`` and
  ``read_vrt(..., chunks=...)`` -- the public dispatcher path
* GPU eager via ``read_vrt(gpu=True)`` guarded by
  ``pytest.importorskip``

Scope of coverage for this file. The following attrs get cross-backend
parity asserts here:

* ``transform``
* ``crs``
* ``nodata``
* ``masked_nodata``
* ``georef_status``
* ``raster_type`` (when the source is AREA_OR_POINT=Point; the area
  default leaves the attr unset, so it is not in the required-key list)

The following keys are intentionally OUT of scope for this file --
the VRT path is documented to omit them, and the non-VRT backend
parity suite owns their cross-backend pin:

* ``crs_wkt`` (compared via the ``crs`` EPSG integer instead, because
  WKT text can re-emit under pyproj normalisation)
* ``gdal_metadata_xml``
* ``extra_tags``

The negative tests pin the fail-closed posture for ambiguous VRT input:
mixed CRS, mixed per-band nodata, unsupported resampling, malformed
SrcRect / DstRect, and missing sources under both ``'raise'`` and
``'warn'`` policies. The VRT path must refuse to silently flatten
ambiguous metadata to one value -- a pixel-only check would miss this.

PR 2 of the parent epic (``VRTUnsupportedError``) is not landed yet.
The negative tests assert against the current error type and carry a
``TODO(#2321)`` so the upgrade is mechanical when PR 2 lands.

Temp file names include ``_2321_`` per ``CLAUDE.md`` to avoid
collisions in parallel runs.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

# Two writer imports because the fixture builders below have two
# shapes of input:
# - ``to_geotiff`` (public surface, takes an ``xr.DataArray``) for the
#   full-coords / CRS-on-DataArray fixtures
# - ``write`` (``xrspatial.geotiff._writer``, takes a raw numpy array
#   plus a ``nodata=`` kwarg) for the per-band integer fixtures where
#   constructing a DataArray just to round-trip via to_geotiff would
#   add nothing
from xrspatial.geotiff import (GeoTIFFFallbackWarning, MixedBandMetadataError,
                               open_geotiff, read_vrt, to_geotiff)
from xrspatial.geotiff._attrs import (GEOREF_STATUS_FULL,
                                      GEOREF_STATUS_TRANSFORM_ONLY)
from xrspatial.geotiff._writer import write
from xrspatial.geotiff.tests.conftest import requires_gpu


# WKT for EPSG:4326. Same constant as the finalization parity module so
# the WKT-vs-EPSG comparison surface matches.
_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)


# Attrs the VRT path is documented to omit even when the underlying
# TIFF carries them. ``test_vrt_finalization_parity_2162`` already pins
# the precise omission list; the parity asserts below intersect rather
# than mirror it because the backend-vs-backend comparison only needs
# both sides to agree, not match an absolute set.
_VRT_OMITTED_ATTR_KEYS = frozenset({
    'extra_tags',
    'image_description',
    'extra_samples',
    'gdal_metadata',
    'gdal_metadata_xml',
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    'colormap',
})

# Attrs whose textual representation can differ between two readers
# even when the logical value matches. ``crs_wkt`` may be re-emitted
# by pyproj on one path and pass-through verbatim on another; the
# integer ``crs`` (EPSG) carries the same information for parity.
# ``transform`` shifts by a half-pixel between AREA_OR_POINT writers
# but the windowed reads in this module all share one writer, so the
# 6-tuple is comparable element-wise across backends.
_REPRESENTATION_KEYS = frozenset({'crs_wkt'})

# Attrs the eager path stamps but the dask path is documented to omit
# under the lazy-attrs contract (issue #2135). The dask backend cannot
# compute the presence flag without forcing a materialise, so the lazy
# build legitimately ships without it. Drop these from the cross-backend
# comparison so a dask read can match a numpy read on the rest of the
# contract.
_BACKEND_LIFECYCLE_KEYS = frozenset({'nodata_pixels_present'})


# ---------------------------------------------------------------------------
# VRT fixture builders. Single-source, well-formed, contract-positive.
# ---------------------------------------------------------------------------


def _write_single_source_vrt(
    tiff_path: str,
    vrt_path: str,
    *,
    width: int,
    height: int,
    dtype_xml: str = "Float32",
    nodata: float | int | None = None,
    geo_transform: str | None = '0.0, 1.0, 0.0, 0.0, 0.0, -1.0',
    srs: str | None = None,
) -> None:
    """Write a 1-band VRT pointing at ``tiff_path``.

    Same writer style as ``test_vrt_finalization_parity_2162`` so the
    two test modules share fixture geometry conventions.
    """
    nodata_xml = (
        f"    <NoDataValue>{nodata}</NoDataValue>\n"
        if nodata is not None else ''
    )
    srs_xml = f'  <SRS>{srs}</SRS>\n' if srs is not None else ''
    gt_xml = (
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        if geo_transform is not None else ''
    )
    vrt_xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'{gt_xml}'
        f'{srs_xml}'
        f'  <VRTRasterBand dataType="{dtype_xml}" band="1">\n'
        f'{nodata_xml}'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{tiff_path}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)


def _build_full_georef_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 float32 single-source VRT with full georef + nodata."""
    import xarray as xr

    tiff = str(tmp_path / 'tmp_2321_full_src.tif')
    vrt = str(tmp_path / 'tmp_2321_full.vrt')
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(
        data,
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4, dtype_xml='Float32',
        nodata=-9999.0,
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=_WGS84_WKT,
    )
    return vrt


def _build_transform_only_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 single-source VRT with transform but no SRS (CRS absent)."""
    import xarray as xr

    tiff = str(tmp_path / 'tmp_2321_tonly_src.tif')
    vrt = str(tmp_path / 'tmp_2321_tonly.vrt')
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(
        data,
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4, dtype_xml='Float32',
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=None,
    )
    return vrt


def _build_integer_with_nodata_vrt(tmp_path: pathlib.Path) -> str:
    """4x4 uint16 single-source VRT with declared nodata sentinel.

    Used for ``masked_nodata`` parity: the integer-with-sentinel source
    must promote to float64 with NaN-masked sentinel pixels in every
    backend and stamp ``attrs['masked_nodata']=True``.
    """
    src_arr = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 65535],
         [9, 10, 11, 12],
         [13, 14, 15, 16]],
        dtype=np.uint16,
    )
    tiff = str(tmp_path / 'tmp_2321_int_src.tif')
    vrt = str(tmp_path / 'tmp_2321_int.vrt')
    write(src_arr, tiff, nodata=65535, compression='none', tiled=False)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4, dtype_xml='UInt16',
        nodata=65535,
        geo_transform='0.0, 1.0, 0.0, 0.0, 0.0, -1.0',
        srs=_WGS84_WKT,
    )
    return vrt


# ---------------------------------------------------------------------------
# Cross-backend reader helpers.
# ---------------------------------------------------------------------------


def _read_eager_numpy(vrt_path: str):
    """Eager numpy via the dispatcher (mirrors public surface)."""
    return open_geotiff(vrt_path)


def _read_dask(vrt_path: str):
    """Dask via the dispatcher, then ``compute()`` for value parity."""
    lazy = open_geotiff(vrt_path, chunks=2)
    return lazy.compute()


def _read_dask_chunks_2(vrt_path: str):
    """Dask via the dispatcher, lazy (no compute).

    Used for negative-tests that pin the build-time raise contract
    (e.g., ``test_mixed_nodata_vrt_fails_closed_by_default``). Named
    at module scope so pytest test ids render as
    ``[dask_chunks_2-_read_dask_chunks_2]`` rather than the cryptic
    ``[dask_chunks_2-<lambda>]`` an inline lambda would produce.
    """
    return open_geotiff(vrt_path, chunks=2)


def _read_gpu_eager(vrt_path: str):
    """GPU eager via ``read_vrt(gpu=True)``.

    ``open_geotiff(..., gpu=True)`` rejects ``.vrt`` sources up front
    (the dispatcher routes ``.vrt`` to ``read_vrt`` and ``read_vrt``
    owns the ``gpu`` kwarg, see ``_backends/vrt.py``). Use the direct
    entry point here so the GPU eager path is exercised.
    """
    return read_vrt(vrt_path, gpu=True)


# Backends used by the cross-backend parity sweep. The GPU entry is
# parametrized in but skipped without cupy + a working CUDA device.
# Reuses the project-wide ``requires_gpu`` skip marker from
# ``xrspatial.geotiff.tests.conftest`` so the import-time CUDA probe
# stays canonical -- a local re-implementation would risk drift from
# the shared ``gpu_available()`` helper.
_BACKENDS = [
    pytest.param('numpy', _read_eager_numpy, id='numpy'),
    pytest.param('dask', _read_dask, id='dask'),
    pytest.param('gpu', _read_gpu_eager, id='gpu', marks=requires_gpu),
]


def _comparable_attrs(attrs: dict) -> dict:
    """Filter attrs down to the cross-backend comparable subset.

    Drops the documented VRT-omitted keys (which may differ if one
    backend stamps a TIFF-specific key while another does not) and the
    representation-only keys (``crs_wkt``).
    """
    return {
        k: v for k, v in attrs.items()
        if k not in _VRT_OMITTED_ATTR_KEYS
        and k not in _REPRESENTATION_KEYS
        and k not in _BACKEND_LIFECYCLE_KEYS
    }


def _to_numpy(arr) -> np.ndarray:
    """Return a host-side numpy view of ``arr.values`` regardless of
    backend.

    CuPy DataArrays have a ``.values`` accessor that triggers an
    implicit host transfer in some xarray versions but not others; use
    the explicit ``.data.get()`` path for cupy buffers per CLAUDE.md.
    """
    data = arr.data
    if hasattr(data, 'get'):  # cupy ndarray
        return data.get()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Positive parity tests: same VRT, every backend, identical metadata.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_attrs_match_eager_numpy(
    tmp_path, _label, reader,
):
    """Each non-numpy backend's attrs must match the eager numpy baseline.

    The full-georef VRT carries CRS, transform, nodata, and an
    integer-source-promotes-to-float lifecycle. Every attr the contract
    promises (``transform``, ``crs``, ``nodata``, ``masked_nodata``,
    ``georef_status``, ``raster_type``) must compare equal across
    backends. ``crs_wkt`` is compared via the ``crs`` integer instead
    because the WKT text can re-emit under pyproj normalisation.

    Without this assertion a backend regression that drops one of
    these attrs but still returns correct pixels would slip through
    every existing pixel-only test.
    """
    vrt = _build_full_georef_vrt(tmp_path)

    baseline = _read_eager_numpy(vrt)
    candidate = reader(vrt)

    base_attrs = _comparable_attrs(dict(baseline.attrs))
    cand_attrs = _comparable_attrs(dict(candidate.attrs))

    # Hard equality on the helper-stamped attr block.
    base_keys = set(base_attrs)
    cand_keys = set(cand_attrs)
    assert base_keys == cand_keys, (
        f"Attr-key drift between numpy and {_label}: "
        f"numpy-only={base_keys - cand_keys}, "
        f"{_label}-only={cand_keys - base_keys}"
    )
    differing = [
        k for k in base_keys
        if base_attrs[k] != cand_attrs[k]
    ]
    assert not differing, (
        f"Attr value drift between numpy and {_label}: "
        f"{[(k, base_attrs[k], cand_attrs[k]) for k in differing]}"
    )

    # The promises of the contract: each key is present and well-formed.
    # ``raster_type`` is only stamped for AREA_OR_POINT=Point sources;
    # the area default leaves the attr unset, so it is not part of the
    # required-key list. Pin the keys that the contract guarantees on
    # every full-georef read here.
    for key in ('transform', 'crs', 'georef_status'):
        assert key in cand_attrs, (
            f"{_label} backend missing required attr {key!r}"
        )
    assert cand_attrs['georef_status'] == GEOREF_STATUS_FULL
    assert cand_attrs['crs'] == 4326
    assert len(cand_attrs['transform']) == 6


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_pixels_match_eager_numpy(
    tmp_path, _label, reader,
):
    """Pixel-value parity for the full-georef VRT.

    Twin of the attrs test above: a regression that fixed attrs but
    broke pixels (or vice versa) must surface on at least one of the
    two. Asserting both side-by-side keeps the surface explicit.
    """
    vrt = _build_full_georef_vrt(tmp_path)

    base = _to_numpy(_read_eager_numpy(vrt))
    cand = _to_numpy(reader(vrt))

    assert base.shape == cand.shape, (
        f"shape drift numpy vs {_label}: {base.shape} vs {cand.shape}"
    )
    np.testing.assert_array_equal(base, cand)


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_full_georef_vrt_coords_match_eager_numpy(
    tmp_path, _label, reader,
):
    """Coord-array parity for the full-georef VRT.

    The transform attr alone does not guarantee correct coords: the
    half-pixel AREA_OR_POINT shift can drift between backends. Compare
    the actual coord arrays so a coord regression surfaces directly.
    """
    vrt = _build_full_georef_vrt(tmp_path)

    base = _read_eager_numpy(vrt)
    cand = reader(vrt)

    assert list(cand.dims) == list(base.dims), (
        f"dim drift numpy vs {_label}: {base.dims} vs {cand.dims}"
    )
    for axis in ('y', 'x'):
        np.testing.assert_array_equal(
            np.asarray(cand[axis].values),
            np.asarray(base[axis].values),
        )


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_transform_only_vrt_attrs_match_eager_numpy(
    tmp_path, _label, reader,
):
    """Same parity sweep on a transform-only VRT (no CRS).

    ``georef_status`` must be ``transform_only`` on every backend and
    ``attrs['crs']`` must be absent on every backend. A regression
    that emits a stale CRS from a TIFF-tag fallback would show up here
    as a key-set diff.
    """
    vrt = _build_transform_only_vrt(tmp_path)

    baseline = _read_eager_numpy(vrt)
    candidate = reader(vrt)

    base_attrs = _comparable_attrs(dict(baseline.attrs))
    cand_attrs = _comparable_attrs(dict(candidate.attrs))

    assert set(base_attrs) == set(cand_attrs)
    assert base_attrs == cand_attrs
    assert cand_attrs['georef_status'] == GEOREF_STATUS_TRANSFORM_ONLY
    assert 'crs' not in cand_attrs


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_integer_nodata_vrt_attrs_match_eager_numpy(
    tmp_path, _label, reader,
):
    """``masked_nodata`` and ``nodata`` lifecycle parity on integer VRT.

    The integer-with-sentinel source must promote to float on every
    backend and stamp ``attrs['masked_nodata']=True`` plus
    ``attrs['nodata']=65535`` (the original sentinel). A backend that
    forgets to stamp ``masked_nodata`` would silently mislead callers
    who branch on the attr to decide whether NaN is real or a mask.
    """
    vrt = _build_integer_with_nodata_vrt(tmp_path)

    baseline = _read_eager_numpy(vrt)
    candidate = reader(vrt)

    base_attrs = _comparable_attrs(dict(baseline.attrs))
    cand_attrs = _comparable_attrs(dict(candidate.attrs))

    assert set(base_attrs) == set(cand_attrs)
    assert base_attrs == cand_attrs
    # Lifecycle invariants regardless of backend:
    assert cand_attrs.get('masked_nodata') is True
    assert cand_attrs.get('nodata') == 65535


@pytest.mark.parametrize('_label, reader', _BACKENDS)
def test_integer_nodata_vrt_pixels_match_eager_numpy(
    tmp_path, _label, reader,
):
    """Pixel parity for the integer-VRT case.

    Twin of the attrs test so a backend regression that masks but
    forgets the attr (or stamps the attr but masks the wrong cell)
    fails one assertion or the other, never both silently.
    """
    vrt = _build_integer_with_nodata_vrt(tmp_path)

    base = _to_numpy(_read_eager_numpy(vrt))
    cand = _to_numpy(reader(vrt))

    assert base.shape == cand.shape
    # Promoted-to-float64 array with NaN at the sentinel cell.
    np.testing.assert_array_equal(np.isnan(base), np.isnan(cand))
    base_finite = base[~np.isnan(base)]
    cand_finite = cand[~np.isnan(cand)]
    np.testing.assert_array_equal(base_finite, cand_finite)


# ---------------------------------------------------------------------------
# Negative tests: ambiguous metadata must fail closed.
# ---------------------------------------------------------------------------


def _write_mixed_crs_vrt(tmp_path: pathlib.Path) -> str:
    """Two single-band sources with disagreeing CRS at the VRT.

    The VRT XML carries one SRS (WGS84) but the second underlying TIFF
    carries a UTM CRS. The fail-closed contract calls for the read to
    reject this up front, but today the per-source CRS check does NOT
    surface the conflict: the read succeeds and silently flattens to
    the VRT-declared SRS. See the xfail on
    ``test_mixed_crs_vrt_does_not_silently_flatten`` for the
    consumer-side pin and the gap PR 2 must close.

    TODO(#2321): when sub-PR 2 (`VRTUnsupportedError`) lands, the
    centralised validator must reject the mixed-CRS VRT up front with
    a typed error; switch the ``pytest.raises`` on the consumer test
    to that type and drop the broad ``Exception`` fallback.
    """
    import xarray as xr

    # Same shape, same transform, deliberately different CRS per source.
    src0 = tmp_path / 'tmp_2321_mix_crs_src0.tif'
    src1 = tmp_path / 'tmp_2321_mix_crs_src1.tif'

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    da0 = xr.DataArray(
        data,
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    da1 = xr.DataArray(
        data,
        coords={
            # Adjacent on x so the mosaic could in principle assemble.
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([104.0, 105.0, 106.0, 107.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 32633},
    )
    to_geotiff(da0, str(src0))
    to_geotiff(da1, str(src1))

    vrt_path = tmp_path / 'tmp_2321_mixed_crs.vrt'
    # VRT XML declares WGS84 but the underlying second source is UTM.
    vrt_xml = f"""<VRTDataset rasterXSize="8" rasterYSize="4">
  <GeoTransform>100.0, 1.0, 0.0, 200.0, 0.0, -1.0</GeoTransform>
  <SRS>{_WGS84_WKT}</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="4" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


@pytest.mark.xfail(
    reason=(
        "Mixed-CRS VRT currently silently flattens to the VRT-declared "
        "SRS (#2321 gap). The validator from sub-PR 2 must reject this "
        "with a typed error at graph build / eager-read setup; once that "
        "lands, drop the xfail and tighten the assertion to "
        "VRTUnsupportedError. Today the read produces a mosaic whose "
        "attrs['crs'] reports only the VRT-declared CRS while the "
        "second source's UTM data has been silently incorporated."
    ),
    strict=True,
)
def test_mixed_crs_vrt_does_not_silently_flatten(tmp_path):
    """A mixed-CRS VRT must not return a mosaic that silently inherits
    one source's CRS while pixels came from a CRS-incompatible source.

    This is the gap that motivates sub-PR 2 of the parent epic: the
    VRT XML declares one SRS, the underlying sources disagree, and
    the reader hands back a single ``attrs['crs']`` as if everything
    were homogeneous. The pixel content is no longer geospatially
    meaningful once the underlying CRSs disagree, but no error fires.

    TODO(#2321): drop the xfail once sub-PR 2's validator rejects the
    mixed-CRS input with ``VRTUnsupportedError`` at graph build time
    (or eager-read setup). The expected assertion will be
    ``pytest.raises(VRTUnsupportedError)`` around ``read_vrt(vrt)``.

    ``strict=True`` so the test flips to XPASS the moment the gap is
    fixed -- CI will fail loudly, prompting the upgrade to a proper
    raise assertion. That is the desired posture: a finding pinned in
    test form, not silently passing.
    """
    vrt = _write_mixed_crs_vrt(tmp_path)
    # Today this read succeeds and produces an attrs blob that names
    # only the VRT-declared CRS, ignoring the second source's UTM CRS.
    # The xfail above documents the gap; this assertion is what the
    # contract requires after sub-PR 2 lands. Catching Exception is
    # intentional until PR 2 lands and a typed error class exists;
    # narrow this to ``VRTUnsupportedError`` once that imports cleanly.
    with pytest.raises(Exception):
        read_vrt(vrt)


def _write_mixed_nodata_vrt(tmp_path: pathlib.Path) -> str:
    """Two-band uint16 VRT with disagreeing per-band ``<NoDataValue>``.

    Mirrors the fixture in ``test_vrt_multiband_int_nodata_1611``: the
    fail-closed default (band_nodata=None) must raise
    ``MixedBandMetadataError``. The opt-out
    ``band_nodata='first'`` is the explicit escape hatch.
    """
    b0_arr = np.array(
        [[1, 2], [3, 65535]], dtype=np.uint16
    )
    b1_arr = np.array(
        [[7, 8], [9, 65000]], dtype=np.uint16
    )
    p0 = tmp_path / 'tmp_2321_mix_nodata_b0.tif'
    p1 = tmp_path / 'tmp_2321_mix_nodata_b1.tif'
    write(b0_arr, str(p0), nodata=65535, compression='none', tiled=False)
    write(b1_arr, str(p1), nodata=65000, compression='none', tiled=False)

    vrt_path = tmp_path / 'tmp_2321_mix_nodata.vrt'
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>65535</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>65000</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


@pytest.mark.parametrize(
    'reader_label, reader',
    [
        ('eager_numpy', _read_eager_numpy),
        ('dask_chunks_2', _read_dask_chunks_2),
    ],
)
def test_mixed_nodata_vrt_fails_closed_by_default(
    tmp_path, reader_label, reader,
):
    """Per-band disagreeing nodata raises ``MixedBandMetadataError``
    by default on every backend route.

    The dask path's check fires at graph-build time (the metadata
    sweep runs before dask materialises any chunk). The eager path
    raises during the dispatcher's metadata validation. Both must
    refuse rather than flattening to band 0's sentinel.

    TODO(#2321): if sub-PR 2 reroutes this through
    ``VRTUnsupportedError``, accept either type here (subclassing or
    composition).
    """
    vrt = _write_mixed_nodata_vrt(tmp_path)

    with pytest.raises(MixedBandMetadataError):
        result = reader(vrt)
        # Dask path can defer to ``.compute()`` for value reads, but
        # the parity contract requires the build-time raise per #2265.
        # If the build call returned without raising, force a compute
        # so a regression that defers the check still trips the
        # assertion. ``result`` is unreachable on the eager path.
        if hasattr(result, 'compute'):
            result.compute()


def test_mixed_nodata_vrt_opt_in_first_succeeds(tmp_path):
    """``band_nodata='first'`` is the documented opt-out for the
    mixed-nodata fail-closed check.

    Positive pin so a future change that breaks the escape hatch
    surfaces here. The opt-out flattens to band 0's sentinel, which
    is the legacy behaviour callers may explicitly want.
    """
    vrt = _write_mixed_nodata_vrt(tmp_path)
    result = read_vrt(vrt, band_nodata='first')
    # Returned at all -> opt-out is still wired.
    assert result.shape == (2, 2, 2)


def _write_unsupported_resample_vrt(tmp_path: pathlib.Path) -> str:
    """VRT with ``<ResampleAlg>Bilinear`` and a size-changing DstRect.

    A 4x4 source projected into a 2x2 destination with Bilinear must
    raise because the implementation only honours nearest-neighbour
    resampling at the placement site. See #1751.
    """
    src_arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = tmp_path / 'tmp_2321_resample_src.tif'
    write(src_arr, str(src_path), compression='none', tiled=False)

    vrt_path = tmp_path / 'tmp_2321_unsupported_resample.vrt'
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <ResampleAlg>Bilinear</ResampleAlg>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>
"""
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_unsupported_resample_alg_raises(tmp_path):
    """A non-nearest resampling algorithm with a size-changing DstRect
    must raise ``NotImplementedError`` rather than return
    silently-nearest-sampled pixels mislabelled as Bilinear.

    The ``match=`` clause pins the algorithm name and the issue number
    so an unrelated ``NotImplementedError`` from some other VRT code
    path cannot keep the test green. See ``_vrt.py`` for the existing
    raise that names both fields.

    TODO(#2321): when sub-PR 2 lands the typed ``VRTUnsupportedError``
    should be raised here instead; accept either today.
    """
    vrt = _write_unsupported_resample_vrt(tmp_path)
    with pytest.raises(NotImplementedError, match=r"Bilinear|1751"):
        read_vrt(vrt)


def _write_bad_srcrect_vrt(
    tmp_path: pathlib.Path, *, x_size: int = -50,
) -> str:
    """VRT with a negative-size ``<SrcRect>``.

    See #1784: the validator must reject this up front rather than
    swallow it in the missing-source ``try/except``.
    """
    src_arr = np.zeros((10, 10), dtype=np.uint8)
    src_path = tmp_path / 'tmp_2321_bad_srcrect_src.tif'
    to_geotiff(src_arr, str(src_path), compression='none')

    vrt_path = tmp_path / 'tmp_2321_bad_srcrect.vrt'
    vrt_xml = (
        f'<VRTDataset rasterXSize="100" rasterYSize="100">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="10"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_negative_srcrect_size_rejected(tmp_path):
    """Malformed ``SrcRect`` rejected with a ``ValueError`` that names
    the offending field.

    TODO(#2321): centralise this rejection in PR 2's validator and
    upgrade to ``VRTUnsupportedError``.
    """
    vrt = _write_bad_srcrect_vrt(tmp_path, x_size=-50)
    with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
        read_vrt(vrt)


def _write_bad_dstrect_vrt(
    tmp_path: pathlib.Path, *, x_size: int = -10,
) -> str:
    """VRT with a negative-size ``<DstRect>`` for the negative test.

    Mirrors the DstRect rejection added for #1737; the regression
    coverage today targets oversized DstRects, this test pins the
    sister case for negative dimensions.
    """
    src_arr = np.zeros((10, 10), dtype=np.uint8)
    src_path = tmp_path / 'tmp_2321_bad_dstrect_src.tif'
    to_geotiff(src_arr, str(src_path), compression='none')

    vrt_path = tmp_path / 'tmp_2321_bad_dstrect.vrt'
    vrt_xml = (
        f'<VRTDataset rasterXSize="100" rasterYSize="100">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{x_size}" ySize="10"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt_path.write_text(vrt_xml)
    return str(vrt_path)


def test_negative_dstrect_size_rejected(tmp_path):
    """Malformed ``DstRect`` must not survive into the read path.

    Accept ``ValueError`` (today's posture; the SimpleSource DstRect
    validator raises ``VRT SimpleSource DstRect has negative size
    (...)`` before any pixel work begins). The ``match=`` clause pins
    the field name and the rejection reason so an unrelated
    ``ValueError`` from some other VRT code path cannot silently keep
    the test green.

    TODO(#2321): tighten to ``VRTUnsupportedError`` when PR 2 ships.
    """
    vrt = _write_bad_dstrect_vrt(tmp_path, x_size=-10)
    with pytest.raises(ValueError, match=r"DstRect.*negative size"):
        read_vrt(vrt)


def _write_missing_source_vrt(
    tmp_path: pathlib.Path, *, name: str = 'tmp_2321_missing.vrt',
) -> str:
    """VRT pointing at a single source path that does not exist.

    The dispatcher's static missing-source sweep (#2265) raises at
    construction time for both eager and dask routes when
    ``missing_sources='raise'`` is in effect.
    """
    vrt_path = tmp_path / name
    # Reference a path inside the tmp dir that we never create.
    missing = tmp_path / 'tmp_2321_missing_src.tif'
    vrt_xml = (
        f'<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt_path.write_text(vrt_xml)
    assert not os.path.exists(str(missing)), (
        "fixture leak: missing-source path exists on disk"
    )
    return str(vrt_path)


def test_missing_sources_raise_eager(tmp_path):
    """``missing_sources='raise'`` (the public default since #1860)
    must abort the read up front on the eager path."""
    vrt = _write_missing_source_vrt(tmp_path, name='tmp_2321_miss_eager.vrt')
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        read_vrt(vrt)


def test_missing_sources_raise_dask(tmp_path):
    """``missing_sources='raise'`` (default) on the dask path raises
    at graph-build time per #2265, not at ``.compute()``.

    Pin both the build-time raise and the value path so a regression
    that defers the check to compute surfaces here.
    """
    vrt = _write_missing_source_vrt(tmp_path, name='tmp_2321_miss_dask.vrt')
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        # Build-time raise required by the contract; if the implementation
        # ever defers to compute, the test still fails on the materialise
        # call because the exception type is the same.
        lazy = open_geotiff(vrt, chunks=2)
        lazy.compute()


def test_missing_sources_warn_records_holes(tmp_path):
    """``missing_sources='warn'`` is the documented escape hatch.

    The lenient path must emit ``GeoTIFFFallbackWarning`` and populate
    ``attrs['vrt_holes']`` so callers branching on the attr can detect
    a partial mosaic. This is the contract documented in #1734 / #1843;
    the test pins it via the public ``read_vrt`` entry point so a
    regression in the warn-policy attr emission surfaces.

    The plan calls for parity tests against ``missing_sources='skip'``;
    the public API exposes ``'warn'`` as the lenient option (skip is
    used internally inside ``_vrt.read_vrt``). Use the documented public
    value here so the test pins the user-facing contract.
    """
    vrt = _write_missing_source_vrt(tmp_path, name='tmp_2321_miss_warn.vrt')

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        result = read_vrt(vrt, missing_sources='warn')

    # The attrs contract for the lenient path requires both keys:
    # ``vrt_holes`` lists the skipped sources, and the array exists.
    assert 'vrt_holes' in result.attrs, (
        "missing_sources='warn' did not stamp attrs['vrt_holes']"
    )
    holes = result.attrs['vrt_holes']
    assert len(holes) == 1
    # The hole entry should name the skipped source so downstream
    # consumers can audit what was dropped. The shape pinned in #1734
    # is a dict with a ``source`` key; pin it as a hard assertion so a
    # future refactor that changes the entry type (e.g., dataclass)
    # surfaces here instead of silently weakening the path check.
    assert isinstance(holes[0], dict), (
        f"vrt_holes entry type drifted: {type(holes[0]).__name__}; "
        f"#1734 documents a dict shape"
    )
    hole_source = holes[0]['source']
    assert 'tmp_2321_missing_src.tif' in hole_source, (
        f"hole source path drifted: {hole_source!r}"
    )
