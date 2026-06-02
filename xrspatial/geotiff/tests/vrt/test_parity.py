"""Cross-backend parity and backend-coverage for the VRT read path.

Covers:

* Backend parity for VRT reads with sidecar / overview interactions:
  eager-vs-dask pixel + metadata (coords, transform, CRS,
  ``georef_status``) parity, sidecar-vs-inline-overview attrs, and the
  windowed coord / transform shift.
* Cross-backend parity for the VRT finalization pipeline: VRT eager vs
  ``open_geotiff`` and VRT chunked vs ``read_geotiff_dask`` for the five
  canonical georef states, ``band_nodata='first'`` per-band attrs,
  ``dtype=`` no-sentinel branch, ``missing_sources='warn'`` vrt_holes,
  and eager/chunked internal parity.
* Backend / parameter coverage for ``read_vrt``: the GPU and dask+GPU
  decode paths, ``dtype=`` / ``name=`` kwargs, and the file-like +
  backend-kwarg rejection on ``open_geotiff``.

The parity helpers (``_materialise`` / ``_assert_pixels_equal`` /
``_assert_metadata_parity``) mirror ``parity/test_backend_matrix.py`` so
cross-test parity reads the same way; this file keeps them VRT-local
rather than re-homing the shared harness.
"""
from __future__ import annotations

import importlib.util
import io
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, read_vrt, to_geotiff
from xrspatial.geotiff._attrs import (GEOREF_STATUS_CRS_ONLY, GEOREF_STATUS_FULL,
                                      GEOREF_STATUS_NONE, GEOREF_STATUS_ROTATED_DROPPED,
                                      GEOREF_STATUS_TRANSFORM_ONLY)
from xrspatial.geotiff._coords import _NO_GEOREF_KEY
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal
from xrspatial.geotiff._writer import write

tifffile = pytest.importorskip("tifffile")


# ===========================================================================
# GPU gating (matches the rest of the geotiff test suite's predicate).
# ===========================================================================


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


# ===========================================================================
# Backend parity with sidecar / overview interactions
# ===========================================================================
#
# Asserts eager / dask parity on the surface most likely to drift:
# metadata (transform, crs, crs_wkt, georef_status), windowed coords,
# and sidecar (.tif.ovr) interactions. Acceptance: the VRT path cannot
# pass by returning correct pixel values with wrong georeferencing attrs.

_GOLDEN = (
    pathlib.Path(__file__).resolve().parent.parent
    / "golden_corpus"
    / "fixtures"
)
_SIDECAR_TIF = _GOLDEN / "overview_external_ovr_uint16.tif"
_SIDECAR_OVR = _GOLDEN / "overview_external_ovr_uint16.tif.ovr"
_INLINE_OVR_TIF = _GOLDEN / "overview_internal_uint16.tif"


def _sidecar_fixture_or_skip() -> Path:
    """Return the bundled sidecar TIFF or skip if absent."""
    if not _SIDECAR_TIF.exists() or not _SIDECAR_OVR.exists():
        pytest.skip("sidecar overview fixture not present in golden_corpus")
    return _SIDECAR_TIF


def _inline_overview_fixture_or_skip() -> Path:
    if not _INLINE_OVR_TIF.exists():
        pytest.skip("inline overview fixture not present in golden_corpus")
    return _INLINE_OVR_TIF


def _materialise(da: xr.DataArray) -> np.ndarray:
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _coord_view(da: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(da.coords[name].values)


def _assert_pixels_equal(ref: np.ndarray, actual: np.ndarray,
                         *, label: str) -> None:
    """Pixel equality, dtype-aware (mirrors test_backend_matrix.py)."""
    assert ref.dtype == actual.dtype, (
        f"{label}: dtype differs ref={ref.dtype} actual={actual.dtype}"
    )
    assert ref.shape == actual.shape, (
        f"{label}: shape differs ref={ref.shape} actual={actual.shape}"
    )
    if ref.dtype.kind == "f":
        assert np.array_equal(ref, actual, equal_nan=True), (
            f"{label}: float pixels differ (NaN-aware)"
        )
    else:
        assert ref.tobytes() == actual.tobytes(), (
            f"{label}: integer pixel bytes differ"
        )


def _assert_metadata_parity(
    ref: xr.DataArray,
    actual: xr.DataArray,
    *,
    label: str,
    expected_dims: tuple[str, ...],
) -> None:
    """Fail if any parity-critical attr / coord drifts between two reads."""
    assert actual.dims == expected_dims, (
        f"{label}: dims {actual.dims!r} != expected {expected_dims!r}"
    )
    assert ref.dims == expected_dims, (
        f"{label}: ref dims {ref.dims!r} != expected {expected_dims!r}"
    )

    for axis in expected_dims:
        if axis not in ref.coords:
            continue
        ref_c = _coord_view(ref, axis)
        actual_c = _coord_view(actual, axis)
        assert ref_c.dtype == actual_c.dtype, (
            f"{label}: coord {axis!r} dtype "
            f"ref={ref_c.dtype} actual={actual_c.dtype}"
        )
        assert ref_c.shape == actual_c.shape, (
            f"{label}: coord {axis!r} shape "
            f"ref={ref_c.shape} actual={actual_c.shape}"
        )
        assert ref_c.tobytes() == actual_c.tobytes(), (
            f"{label}: coord {axis!r} bytes differ "
            f"(ref[:3]={ref_c[:3].tolist()!r}, "
            f"actual[:3]={actual_c[:3].tolist()!r})"
        )

    ref_t = ref.attrs.get("transform")
    actual_t = actual.attrs.get("transform")
    assert ref_t == actual_t, (
        f"{label}: transform tuple differs "
        f"ref={ref_t!r} actual={actual_t!r}"
    )

    assert ref.attrs.get("crs") == actual.attrs.get("crs"), (
        f"{label}: attrs['crs'] differs "
        f"ref={ref.attrs.get('crs')!r} actual={actual.attrs.get('crs')!r}"
    )
    assert ref.attrs.get("crs_wkt") == actual.attrs.get("crs_wkt"), (
        f"{label}: crs_wkt differs"
    )
    assert ref.attrs.get("georef_status") == actual.attrs.get(
        "georef_status"
    ), (
        f"{label}: georef_status differs "
        f"ref={ref.attrs.get('georef_status')!r} "
        f"actual={actual.attrs.get('georef_status')!r}"
    )


def _build_two_tile_float32_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """Two 16x16 float32 tiles laid out side-by-side as a 16x32 mosaic."""
    tile_h, tile_w = 16, 16
    paths: list[str] = []
    for c in range(2):
        arr = np.full(
            (tile_h, tile_w), float(c + 1) * 1000.0, dtype=np.float32
        )
        arr[0, 0] = -7.0 + c
        arr[tile_h - 1, tile_w - 1] = 9000.0 + c
        origin_x = float(c * tile_w)
        da = xr.DataArray(
            arr, dims=["y", "x"],
            coords={
                "y": np.arange(tile_h - 1, -1, -1, dtype=np.float64),
                "x": np.arange(
                    origin_x, origin_x + tile_w, dtype=np.float64),
            },
            attrs={"crs": 4326},
        )
        tile_path = tmp_path / f"tile_2321_{c}.tif"
        to_geotiff(da, str(tile_path), compression="none", tiled=False)
        paths.append(str(tile_path))
    vrt_path = tmp_path / "two_tile_2321_.vrt"
    _write_vrt_internal(str(vrt_path), paths, relative=False)
    return vrt_path, np.dtype("float32")


def _build_sidecar_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """VRT over a copy of the bundled sidecar TIFF + its ``.ovr`` partner."""
    src = _sidecar_fixture_or_skip()
    base = tmp_path / "sidecar_2321_.tif"
    shutil.copy(src, base)
    shutil.copy(str(src) + ".ovr", str(base) + ".ovr")
    vrt_path = tmp_path / "sidecar_2321_.vrt"
    _write_vrt_internal(str(vrt_path), [str(base)], relative=False)
    return vrt_path, np.dtype("uint16")


def _build_inline_overview_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """VRT over a copy of the inline-overview fixture (no sidecar)."""
    src = _inline_overview_fixture_or_skip()
    base = tmp_path / "inline_2321_.tif"
    shutil.copy(src, base)
    vrt_path = tmp_path / "inline_2321_.vrt"
    _write_vrt_internal(str(vrt_path), [str(base)], relative=False)
    return vrt_path, np.dtype("uint16")


@dataclass(frozen=True)
class _BackendSpec:
    backend_id: str
    kwargs: dict[str, Any]


_BACKENDS: tuple[_BackendSpec, ...] = (
    _BackendSpec(backend_id="eager", kwargs={}),
    _BackendSpec(backend_id="dask", kwargs={"chunks": (16, 16)}),
)


def _backend_params() -> list:
    return [pytest.param(b, id=b.backend_id) for b in _BACKENDS]


@dataclass(frozen=True)
class _FixtureSpec:
    fix_id: str
    builder: Callable[[Path], tuple[Path, np.dtype]]
    expected_dims: tuple[str, ...]
    window: tuple[int, int, int, int] | None


_FIXTURES: tuple[_FixtureSpec, ...] = (
    _FixtureSpec(
        fix_id="two-tile-float32-full",
        builder=_build_two_tile_float32_vrt,
        expected_dims=("y", "x"),
        window=None,
    ),
    _FixtureSpec(
        fix_id="two-tile-float32-window-spans-seam",
        builder=_build_two_tile_float32_vrt,
        expected_dims=("y", "x"),
        window=(4, 8, 12, 24),
    ),
    _FixtureSpec(
        fix_id="sidecar-uint16-full",
        builder=_build_sidecar_vrt,
        expected_dims=("y", "x"),
        window=None,
    ),
    _FixtureSpec(
        fix_id="sidecar-uint16-window",
        builder=_build_sidecar_vrt,
        expected_dims=("y", "x"),
        window=(8, 8, 56, 56),
    ),
)


def _fixture_params() -> list:
    return [pytest.param(f, id=f.fix_id) for f in _FIXTURES]


@pytest.fixture(scope="session")
def _vrt_parity_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("vrt_parity_2321_")


@pytest.fixture(scope="session")
def _vrt_parity_cache() -> dict[str, tuple[Path, np.dtype]]:
    """Session-scoped (path, dtype) cache shared across every cell.

    A function-scoped cache would rebuild the same VRT per cell; on
    Windows that surfaces as PermissionError when ``to_geotiff`` renames
    over a file another cell still holds mapped.
    """
    return {}


@pytest.fixture
def vrt_fixture(_vrt_parity_dir, _vrt_parity_cache):
    """Resolve a :class:`_FixtureSpec` to a (vrt_path, dtype) pair on disk."""
    base = _vrt_parity_dir
    cache = _vrt_parity_cache

    def _resolve(spec: _FixtureSpec) -> tuple[Path, np.dtype]:
        key = spec.builder.__name__
        if key in cache:
            return cache[key]
        sub = base / key
        sub.mkdir(exist_ok=True)
        result = spec.builder(sub)
        cache[key] = result
        return result
    return _resolve


@pytest.mark.parametrize("spec", _fixture_params())
@pytest.mark.parametrize("backend", _backend_params())
def test_vrt_backend_parity(spec, backend, vrt_fixture):
    """One cell per (fixture, backend). Asserts pixels + metadata parity."""
    vrt_path, expected_dtype = vrt_fixture(spec)

    open_kwargs: dict[str, Any] = {}
    if spec.window is not None:
        open_kwargs["window"] = spec.window

    ref = open_geotiff(str(vrt_path), **open_kwargs)
    actual = open_geotiff(str(vrt_path), **open_kwargs, **backend.kwargs)

    label = (
        f"fixture={spec.fix_id} backend={backend.backend_id} "
        f"window={spec.window!r}"
    )

    ref_arr = _materialise(ref)
    actual_arr = _materialise(actual)

    assert ref_arr.dtype == expected_dtype, (
        f"{label}: reference dtype {ref_arr.dtype} != "
        f"expected {expected_dtype}"
    )
    assert actual_arr.dtype == expected_dtype, (
        f"{label}: actual dtype {actual_arr.dtype} != "
        f"expected {expected_dtype}"
    )

    _assert_pixels_equal(ref_arr, actual_arr, label=label)
    _assert_metadata_parity(
        ref, actual, label=label, expected_dims=spec.expected_dims,
    )


@pytest.mark.parametrize("backend", _backend_params())
def test_sidecar_vrt_attrs_match_inline(backend, tmp_path):
    """Sidecar-backed and inline-overview-backed VRTs report identical
    georef attrs and pixels at the base level."""
    side_sub = tmp_path / "sidecar"
    inline_sub = tmp_path / "inline"
    side_sub.mkdir()
    inline_sub.mkdir()
    side_vrt, side_dtype = _build_sidecar_vrt(side_sub)
    inline_vrt, inline_dtype = _build_inline_overview_vrt(inline_sub)

    assert side_dtype == inline_dtype, (
        f"sidecar dtype {side_dtype} != inline dtype {inline_dtype}; "
        f"the golden_corpus fixtures should share a base IFD"
    )

    side = open_geotiff(str(side_vrt), **backend.kwargs)
    inline = open_geotiff(str(inline_vrt), **backend.kwargs)

    label = f"sidecar-vs-inline backend={backend.backend_id}"

    assert side.shape == inline.shape, (
        f"{label}: shape differs side={side.shape} inline={inline.shape}"
    )

    _assert_pixels_equal(
        _materialise(inline), _materialise(side), label=label,
    )
    _assert_metadata_parity(
        inline, side, label=label, expected_dims=("y", "x"),
    )


def test_windowed_vrt_shifts_coords_and_transform_consistently(tmp_path):
    """Eager and lazy windowed VRT reads agree on shape, coords, attrs,
    and values."""
    vrt_path, _ = _build_two_tile_float32_vrt(tmp_path)
    window = (3, 5, 13, 27)

    eager = open_geotiff(str(vrt_path), window=window)
    lazy = open_geotiff(str(vrt_path), window=window, chunks=(5, 11))

    assert eager.shape == (10, 22)
    assert lazy.shape == (10, 22)

    np.testing.assert_array_equal(eager["y"].values, lazy["y"].values)
    np.testing.assert_array_equal(eager["x"].values, lazy["x"].values)
    assert eager["y"].dtype == lazy["y"].dtype
    assert eager["x"].dtype == lazy["x"].dtype

    assert eager["y"].values[0] == 12.0
    assert eager["x"].values[0] == 5.0

    eager_t = eager.attrs.get("transform")
    lazy_t = lazy.attrs.get("transform")
    assert eager_t == lazy_t, (
        f"transform differs eager={eager_t!r} lazy={lazy_t!r}"
    )
    assert eager_t[0] == 1.0 and eager_t[4] == -1.0, (
        f"pixel size mismatch in windowed transform {eager_t!r}"
    )

    np.testing.assert_array_equal(eager.values, lazy.compute().values)

    assert eager.attrs.get("crs") == lazy.attrs.get("crs")
    assert eager.attrs.get("crs_wkt") == lazy.attrs.get("crs_wkt")
    assert eager.attrs.get("georef_status") == lazy.attrs.get(
        "georef_status"
    )


def test_sidecar_window_shifts_to_known_coords(tmp_path):
    """The sidecar VRT read with ``window=(8, 8, 56, 56)`` lands on the
    coords / transform an absolute calculation predicts."""
    vrt_path, _ = _build_sidecar_vrt(tmp_path)
    window = (8, 8, 56, 56)

    eager = open_geotiff(str(vrt_path), window=window)

    assert eager.shape == (48, 48)
    t = eager.attrs.get("transform")
    assert t is not None, "windowed sidecar VRT dropped attrs['transform']"
    assert t[0] == pytest.approx(0.001)
    assert t[4] == pytest.approx(-0.001)
    assert t[2] == pytest.approx(-120.0 + 8 * 0.001)
    assert t[5] == pytest.approx(45.0 + 8 * -0.001)


def test_assert_metadata_parity_flags_transform_drift(tmp_path):
    """A transform-only drift between two otherwise-identical DataArrays
    fails the parity helper (locks the harness behaviour)."""
    vrt_path, _ = _build_two_tile_float32_vrt(tmp_path)
    da_ref = open_geotiff(str(vrt_path))
    da_bad = da_ref.copy()
    da_bad.attrs = dict(da_ref.attrs)
    old_t = da_bad.attrs["transform"]
    da_bad.attrs["transform"] = (
        old_t[0], old_t[1], old_t[2] + 1.0,
        old_t[3], old_t[4], old_t[5],
    )
    with pytest.raises(AssertionError, match="transform"):
        _assert_metadata_parity(
            da_ref, da_bad, label="harness-sanity",
            expected_dims=("y", "x"),
        )


# ===========================================================================
# VRT finalization-pipeline parity
# ===========================================================================
#
# The VRT eager and chunked paths route through the shared
# ``_finalize_lazy_read_attrs`` helper. These tests pin parity for the
# attrs the helper stamps against the non-VRT eager / dask readers.

_NON_VRT_ONLY_KEYS = frozenset({
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

_REPRESENTATION_KEYS = frozenset({'crs_wkt', 'transform'})


def _shared_canonical_attrs(attrs: dict) -> dict:
    """Return the helper-emitted attrs that should match across writers."""
    return {
        k: v for k, v in attrs.items()
        if k not in _NON_VRT_ONLY_KEYS and k not in _REPRESENTATION_KEYS
    }


def _write_single_source_vrt(tiff_path, vrt_path, *, width, height,
                             dtype='Float32', nodata=None,
                             geo_transform='0.0, 1.0, 0.0, 0.0, 0.0, -1.0',
                             srs=None):
    """Write a one-band VRT pointing at ``tiff_path``."""
    nodata_xml = (
        f"    <NoDataValue>{nodata}</NoDataValue>\n" if nodata is not None
        else ''
    )
    srs_xml = (
        f'  <SRS>{srs}</SRS>\n' if srs is not None
        else ''
    )
    gt_xml = (
        f'  <GeoTransform>{geo_transform}</GeoTransform>\n'
        if geo_transform is not None
        else ''
    )
    vrt_xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'{gt_xml}'
        f'{srs_xml}'
        f'  <VRTRasterBand dataType="{dtype}" band="1">\n'
        f'{nodata_xml}'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)


_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)


def _make_full_pair(tmp_path, name):
    """Full georef: float coords + CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
        attrs={'crs': 4326},
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=_WGS84_WKT,
    )
    return tiff, vrt


def _make_transform_only_pair(tmp_path, name):
    """Float coords, no CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.array([200.0, 199.0, 198.0, 197.0]),
            'x': np.array([100.0, 101.0, 102.0, 103.0]),
        },
        dims=('y', 'x'),
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform='100.0, 1.0, 0.0, 200.0, 0.0, -1.0',
        srs=None,
    )
    return tiff, vrt


def _make_crs_only_pair(tmp_path, name):
    """No-georef marker + CRS."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        coords={
            'y': np.arange(4, dtype=np.int64),
            'x': np.arange(4, dtype=np.int64),
        },
        dims=('y', 'x'),
        attrs={_NO_GEOREF_KEY: True, 'crs': 4326},
    )
    to_geotiff(da, tiff)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform=None,
        srs=_WGS84_WKT,
    )
    return tiff, vrt


def _make_none_pair(tmp_path, name):
    """No CRS, no transform."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    arr = np.zeros((4, 4), dtype=np.float32)
    tifffile.imwrite(
        tiff, arr, photometric='minisblack', planarconfig='contig',
        metadata=None,
    )
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform=None,
        srs=None,
    )
    return tiff, vrt


def _make_rotated_pair(tmp_path, name):
    """Rotated VRT with ``allow_rotated=True``: lands at ``rotated_dropped``."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    write(arr, tiff, compression='none', tiled=False)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4, dtype='UInt16',
        geo_transform='0.0, 1.0, 0.5, 0.0, 0.5, -1.0',
        srs=None,
    )
    return tiff, vrt


def test_vrt_eager_full_matches_open_geotiff(tmp_path):
    """A single-source VRT wrapping a ``full`` TIFF emits the same
    canonical helper-stamped attrs as the underlying TIFF read."""
    tiff, vrt = _make_full_pair(tmp_path, 'full_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs, (
        f"TIFF/VRT attrs diverged:\n"
        f"  tiff only: {set(tiff_attrs) - set(vrt_attrs)}\n"
        f"  vrt only:  {set(vrt_attrs) - set(tiff_attrs)}\n"
        f"  shared keys with different values: "
        f"{[k for k in set(tiff_attrs) & set(vrt_attrs) if tiff_attrs[k] != vrt_attrs[k]]}"
    )
    full_tiff_attrs = dict(open_geotiff(tiff).attrs)
    full_vrt_attrs = dict(read_vrt(vrt).attrs)
    assert full_tiff_attrs['crs'] == full_vrt_attrs['crs'] == 4326
    assert len(full_tiff_attrs['transform']) == 6
    assert len(full_vrt_attrs['transform']) == 6


def test_vrt_eager_transform_only_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_transform_only_pair(tmp_path, 'tonly_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_TRANSFORM_ONLY


def test_vrt_eager_crs_only_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_crs_only_pair(tmp_path, 'crsonly_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_CRS_ONLY


def test_vrt_eager_none_matches_open_geotiff(tmp_path):
    tiff, vrt = _make_none_pair(tmp_path, 'none_2180')
    tiff_attrs = _shared_canonical_attrs(dict(open_geotiff(tiff).attrs))
    vrt_attrs = _shared_canonical_attrs(dict(read_vrt(vrt).attrs))
    assert tiff_attrs == vrt_attrs
    assert tiff_attrs['georef_status'] == GEOREF_STATUS_NONE


def test_vrt_eager_rotated_dropped_matches_open_geotiff(tmp_path):
    """The rotated branch is the VRT-specific path: a non-zero skew lands
    in ``rotated_dropped`` and the helper drops crs / transform / crs_wkt
    while emitting ``rotated_affine`` plus the no-georef marker."""
    _, vrt = _make_rotated_pair(tmp_path, 'rot_2180')
    attrs = dict(read_vrt(vrt, allow_rotated=True).attrs)
    assert attrs['georef_status'] == GEOREF_STATUS_ROTATED_DROPPED
    assert attrs.get(_NO_GEOREF_KEY) is True
    assert 'rotated_affine' in attrs
    assert attrs.get('crs') is None
    assert attrs.get('crs_wkt') is None
    assert 'transform' not in attrs


def test_vrt_chunked_full_matches_dask(tmp_path):
    tiff, vrt = _make_full_pair(tmp_path, 'full_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_transform_only_matches_dask(tmp_path):
    tiff, vrt = _make_transform_only_pair(tmp_path, 'tonly_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_crs_only_matches_dask(tmp_path):
    tiff, vrt = _make_crs_only_pair(tmp_path, 'crsonly_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_chunked_none_matches_dask(tmp_path):
    tiff, vrt = _make_none_pair(tmp_path, 'none_chunked_2180')
    tiff_attrs = _shared_canonical_attrs(
        dict(read_geotiff_dask(tiff, chunks=2).attrs)
    )
    vrt_attrs = _shared_canonical_attrs(
        dict(read_vrt(vrt, chunks=2).attrs)
    )
    assert tiff_attrs == vrt_attrs


def test_vrt_eager_none_synthesizes_pixel_coords(tmp_path):
    """Issue #2818: a no-``<GeoTransform>`` VRT read eagerly must
    synthesise integer x/y pixel coords, matching the non-VRT
    no-georef read instead of dropping coords entirely."""
    tiff, vrt = _make_none_pair(tmp_path, 'none_coords_2818')
    ref = open_geotiff(tiff)
    vrt_da = read_vrt(vrt)
    assert ref.attrs['georef_status'] == GEOREF_STATUS_NONE
    assert vrt_da.attrs['georef_status'] == GEOREF_STATUS_NONE
    assert 'x' in vrt_da.coords and 'y' in vrt_da.coords
    np.testing.assert_array_equal(
        _coord_view(vrt_da, 'x'), _coord_view(ref, 'x'))
    np.testing.assert_array_equal(
        _coord_view(vrt_da, 'y'), _coord_view(ref, 'y'))
    assert vrt_da.coords['x'].dtype == ref.coords['x'].dtype
    assert vrt_da.coords['y'].dtype == ref.coords['y'].dtype


def test_vrt_chunked_none_synthesizes_pixel_coords(tmp_path):
    """Issue #2818: a no-``<GeoTransform>`` VRT read chunked must
    synthesise the same integer x/y pixel coords as the non-VRT dask
    no-georef read rather than dropping coords entirely."""
    tiff, vrt = _make_none_pair(tmp_path, 'none_coords_chunked_2818')
    ref = read_geotiff_dask(tiff, chunks=2)
    vrt_da = read_vrt(vrt, chunks=2)
    assert ref.attrs['georef_status'] == GEOREF_STATUS_NONE
    assert vrt_da.attrs['georef_status'] == GEOREF_STATUS_NONE
    assert 'x' in vrt_da.coords and 'y' in vrt_da.coords
    np.testing.assert_array_equal(
        _coord_view(vrt_da, 'x'), _coord_view(ref, 'x'))
    np.testing.assert_array_equal(
        _coord_view(vrt_da, 'y'), _coord_view(ref, 'y'))
    assert vrt_da.coords['x'].dtype == ref.coords['x'].dtype
    assert vrt_da.coords['y'].dtype == ref.coords['y'].dtype


def test_vrt_none_windowed_synthesizes_offset_pixel_coords(tmp_path):
    """Issue #2818: a windowed no-georef VRT read shifts the synthesised
    integer coords to the window offset, matching the non-VRT windowed
    read, on both the eager and chunked paths."""
    tiff, vrt = _make_none_pair(tmp_path, 'none_coords_win_2818')
    window = (1, 2, 4, 4)
    ref = open_geotiff(tiff, window=window)
    eager = read_vrt(vrt, window=window)
    chunked = read_vrt(vrt, window=window, chunks=2)
    for label, actual in (('eager', eager), ('chunked', chunked)):
        assert 'x' in actual.coords and 'y' in actual.coords, label
        np.testing.assert_array_equal(
            _coord_view(actual, 'x'), _coord_view(ref, 'x'),
            err_msg=label)
        np.testing.assert_array_equal(
            _coord_view(actual, 'y'), _coord_view(ref, 'y'),
            err_msg=label)


def test_vrt_chunked_rotated_dropped(tmp_path):
    _, vrt = _make_rotated_pair(tmp_path, 'rot_chunked_2180')
    attrs = dict(read_vrt(vrt, allow_rotated=True, chunks=2).attrs)
    assert attrs['georef_status'] == GEOREF_STATUS_ROTATED_DROPPED
    assert attrs.get(_NO_GEOREF_KEY) is True
    assert 'rotated_affine' in attrs


def _write_two_band_per_band_nodata_vrt(tmp_path):
    band0 = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 65000]], dtype=np.uint16)
    p0 = str(tmp_path / 'vrt_band0_2180.tif')
    p1 = str(tmp_path / 'vrt_band1_2180.tif')
    write(band0, p0, nodata=65535, compression='none', tiled=False)
    write(band1, p1, nodata=65000, compression='none', tiled=False)

    vrt_path = str(tmp_path / 'two_band_per_band_nodata_2180.vrt')
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
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_band_nodata_first_band_attrs(tmp_path):
    """``band=1`` with ``band_nodata='first'`` surfaces band 1's sentinel
    on attrs and masks against it."""
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first')
    assert r.attrs['nodata'] == 65000.0
    assert r.attrs['masked_nodata'] is True
    assert np.isnan(r.values[1, 1])
    assert r.attrs.get('nodata_pixels_present') is True


def test_band_nodata_chunked_first_band_attrs(tmp_path):
    """The chunked path threads the same per-band sentinel onto attrs."""
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first', chunks=2)
    assert r.attrs['nodata'] == 65000.0
    assert r.attrs['masked_nodata'] is True
    assert 'nodata_pixels_present' not in r.attrs


def _make_no_sentinel_vrt(tmp_path, name):
    """A single-band float VRT with no ``<NoDataValue>``."""
    tiff = str(tmp_path / f'{name}_tiff.tif')
    vrt = str(tmp_path / f'{name}.vrt')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    write(arr, tiff, compression='none', tiled=False)
    _write_single_source_vrt(
        tiff, vrt, width=4, height=4,
        geo_transform='0.0, 1.0, 0.0, 0.0, 0.0, -1.0',
        nodata=None,
    )
    return vrt


def test_dtype_cast_no_sentinel_omits_attr_eager(tmp_path):
    """Eager VRT with ``dtype=`` and no declared sentinel:
    ``nodata_dtype_cast`` stays absent."""
    vrt = _make_no_sentinel_vrt(tmp_path, 'no_sentinel_eager_2180')
    r = read_vrt(vrt, dtype=np.float64)
    assert r.dtype == np.float64
    assert 'nodata' not in r.attrs
    assert 'masked_nodata' not in r.attrs
    assert 'nodata_dtype_cast' not in r.attrs


def test_dtype_cast_no_sentinel_omits_attr_chunked(tmp_path):
    """Chunked VRT with ``dtype=`` and no declared sentinel: same
    ``nodata_dtype_cast`` pop as the eager branch."""
    vrt = _make_no_sentinel_vrt(tmp_path, 'no_sentinel_chunked_2180')
    r = read_vrt(vrt, dtype=np.float64, chunks=2)
    assert r.dtype == np.float64
    assert 'nodata' not in r.attrs
    assert 'masked_nodata' not in r.attrs
    assert 'nodata_dtype_cast' not in r.attrs


def test_missing_sources_eager_surfaces_vrt_holes(tmp_path):
    """The eager VRT path keeps populating ``attrs['vrt_holes']`` after
    the finalization migration."""
    tiff_path = str(tmp_path / 'present_2180.tif')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    write(arr, tiff_path, compression='none', tiled=False)

    missing_path = str(tmp_path / 'missing_2180.tif')  # never created
    vrt_path = str(tmp_path / 'mosaic_2180.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="8">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{missing_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="4" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = read_vrt(vrt_path, missing_sources='warn')
    assert 'vrt_holes' in r.attrs
    holes = r.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1
    for hole in holes:
        assert 'source' in hole
        assert 'band' in hole
        assert 'dst_rect' in hole
        assert 'error' in hole


def test_missing_sources_chunked_surfaces_vrt_holes(tmp_path):
    """Chunked path's parse-time existence sweep still populates
    ``attrs['vrt_holes']`` after the migration."""
    tiff_path = str(tmp_path / 'present_chunked_2180.tif')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    write(arr, tiff_path, compression='none', tiled=False)

    missing_path = str(tmp_path / 'missing_chunked_2180.tif')
    vrt_path = str(tmp_path / 'mosaic_chunked_2180.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="8">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{tiff_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{missing_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="4" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    r = read_vrt(vrt_path, missing_sources='warn', chunks=2)
    assert 'vrt_holes' in r.attrs
    holes = r.attrs['vrt_holes']
    assert isinstance(holes, list) and len(holes) >= 1


_STATUS_PAIRS = [
    pytest.param(_make_full_pair, GEOREF_STATUS_FULL, False, id="full"),
    pytest.param(
        _make_transform_only_pair, GEOREF_STATUS_TRANSFORM_ONLY,
        False, id="transform_only",
    ),
    pytest.param(
        _make_crs_only_pair, GEOREF_STATUS_CRS_ONLY,
        False, id="crs_only",
    ),
    pytest.param(_make_none_pair, GEOREF_STATUS_NONE, False, id="none"),
    pytest.param(
        _make_rotated_pair, GEOREF_STATUS_ROTATED_DROPPED, True,
        id="rotated_dropped",
    ),
]


@pytest.mark.parametrize("pair_factory,expected_status,allow_rotated",
                         _STATUS_PAIRS)
def test_georef_status_eager_parity(tmp_path, pair_factory, expected_status,
                                    allow_rotated):
    """VRT eager and (where applicable) non-VRT eager agree on
    ``georef_status``."""
    tiff, vrt = pair_factory(tmp_path, f'georef_eager_{expected_status}')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    vrt_status = read_vrt(vrt, **kwargs).attrs.get('georef_status')
    assert vrt_status == expected_status
    if not allow_rotated:
        tiff_status = open_geotiff(tiff, **kwargs).attrs.get('georef_status')
        assert tiff_status == expected_status
        assert vrt_status == tiff_status


@pytest.mark.parametrize("pair_factory,expected_status,allow_rotated",
                         _STATUS_PAIRS)
def test_georef_status_chunked_parity(tmp_path, pair_factory, expected_status,
                                      allow_rotated):
    """VRT chunked and non-VRT chunked agree on ``georef_status``."""
    tiff, vrt = pair_factory(tmp_path, f'georef_chunked_{expected_status}')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    vrt_status = read_vrt(vrt, chunks=2, **kwargs).attrs.get('georef_status')
    assert vrt_status == expected_status
    if not allow_rotated:
        tiff_status = read_geotiff_dask(
            tiff, chunks=2, **kwargs
        ).attrs.get('georef_status')
        assert tiff_status == expected_status
        assert vrt_status == tiff_status


_VRT_FACTORIES = [
    pytest.param(_make_full_pair, False, id="full"),
    pytest.param(_make_transform_only_pair, False, id="transform_only"),
    pytest.param(_make_crs_only_pair, False, id="crs_only"),
    pytest.param(_make_none_pair, False, id="none"),
    pytest.param(_make_rotated_pair, True, id="rotated_dropped"),
]


@pytest.mark.parametrize("pair_factory,allow_rotated", _VRT_FACTORIES)
def test_vrt_eager_chunked_internal_parity(tmp_path, pair_factory,
                                           allow_rotated):
    """Eager and chunked VRT reads of the same fixture agree on the shared
    canonical attrs (modulo the lazy ``nodata_pixels_present`` carve-out)."""
    _, vrt = pair_factory(tmp_path, 'internal_parity_2180')
    kwargs = {'allow_rotated': True} if allow_rotated else {}
    eager_attrs = dict(read_vrt(vrt, **kwargs).attrs)
    chunked_attrs = dict(read_vrt(vrt, chunks=2, **kwargs).attrs)
    eager_attrs.pop('nodata_pixels_present', None)
    chunked_attrs.pop('nodata_pixels_present', None)
    assert eager_attrs == chunked_attrs


# ===========================================================================
# read_vrt backend / parameter coverage
# ===========================================================================
#
# Covers the GPU and dask+GPU decode paths the read_vrt body handles, the
# ``dtype=`` / ``name=`` kwargs, and the open_geotiff file-like +
# backend-kwarg rejection.


@pytest.fixture
def single_tile_vrt(tmp_path):
    """A trivial single-tile float32 VRT plus its source array."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    tile_path = str(tmp_path / 'tile.tif')
    to_geotiff(arr, tile_path)
    vrt_path = str(tmp_path / 'mosaic.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    return vrt_path, arr


@_gpu_only
class TestReadVrtGpuBackend:
    """``read_vrt(gpu=True)`` returns a CuPy-backed DataArray."""

    def test_read_vrt_gpu_returns_cupy(self, single_tile_vrt):
        import cupy

        vrt_path, arr = single_tile_vrt
        da = read_vrt(vrt_path, gpu=True)
        assert isinstance(da.data, cupy.ndarray), (
            f"expected cupy.ndarray, got {type(da.data).__name__}"
        )
        np.testing.assert_array_equal(da.data.get(), arr)

    def test_read_vrt_gpu_chunks_returns_dask_cupy(self, single_tile_vrt):
        """``read_vrt(gpu=True, chunks=N)`` is the dask+cupy VRT entry
        point; the trailing ``result.chunk(...)`` block wraps the cupy
        backing without falling back to numpy."""
        import cupy
        import dask.array as da_mod

        vrt_path, arr = single_tile_vrt
        result = read_vrt(vrt_path, gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array), (
            f"expected dask Array, got {type(result.data).__name__}"
        )
        assert isinstance(result.data._meta, cupy.ndarray), (
            f"expected cupy._meta, got "
            f"{type(result.data._meta).__module__}."
            f"{type(result.data._meta).__name__}"
        )
        assert result.data.chunks == ((2, 2), (2, 2))

        computed = result.compute()
        assert isinstance(computed.data, cupy.ndarray)
        np.testing.assert_array_equal(computed.data.get(), arr)

    def test_open_geotiff_vrt_gpu_routes_through(self, single_tile_vrt):
        """``open_geotiff('.vrt', gpu=True)`` dispatches to ``read_vrt``
        and surfaces the cupy data unchanged."""
        import cupy

        vrt_path, arr = single_tile_vrt
        da = open_geotiff(vrt_path, gpu=True)
        assert isinstance(da.data, cupy.ndarray)
        np.testing.assert_array_equal(da.data.get(), arr)

    def test_open_geotiff_vrt_gpu_chunks(self, single_tile_vrt):
        """``open_geotiff('.vrt', gpu=True, chunks=N)`` is the combined
        dask+cupy entry point."""
        import cupy
        import dask.array as da_mod

        vrt_path, arr = single_tile_vrt
        result = open_geotiff(vrt_path, gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert isinstance(result.data._meta, cupy.ndarray)
        assert result.data.chunks == ((2, 2), (2, 2))

        computed = result.compute()
        np.testing.assert_array_equal(computed.data.get(), arr)


class TestReadVrtDtypeKwarg:
    """``read_vrt(dtype=...)`` casts after decode and validates the cast."""

    def test_safe_widening_cast(self, single_tile_vrt):
        """float32 -> float64 is permitted; values survive bit-for-bit."""
        vrt_path, arr = single_tile_vrt
        da = read_vrt(vrt_path, dtype='float64')
        assert da.dtype == np.float64
        np.testing.assert_array_equal(da.values, arr.astype(np.float64))

    def test_float_to_int_rejected(self, single_tile_vrt):
        """Float-to-int is lossy and refused with a descriptive error."""
        vrt_path, _ = single_tile_vrt
        with pytest.raises(ValueError, match="Cannot cast float"):
            read_vrt(vrt_path, dtype='int32')


class TestReadVrtNameKwarg:
    """``read_vrt(name='custom')`` overrides the file-stem derivation."""

    def test_explicit_name_used(self, single_tile_vrt):
        vrt_path, _ = single_tile_vrt
        da = read_vrt(vrt_path, name='custom_name')
        assert da.name == 'custom_name'

    def test_default_name_from_stem(self, single_tile_vrt):
        vrt_path, _ = single_tile_vrt
        da = read_vrt(vrt_path)
        assert da.name == os.path.splitext(os.path.basename(vrt_path))[0]


class TestOpenGeotiffFileLikeKwargRejection:
    """File-like sources reject ``gpu=True`` and ``chunks=N`` up front."""

    @staticmethod
    def _buf_with_tiff(tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'src.tif')
        to_geotiff(arr, path)
        with open(path, 'rb') as fh:
            return io.BytesIO(fh.read())

    def test_gpu_with_file_like_raises(self, tmp_path):
        buf = self._buf_with_tiff(tmp_path)
        with pytest.raises(ValueError, match="gpu=True is not supported"):
            open_geotiff(buf, gpu=True)

    def test_chunks_with_file_like_raises(self, tmp_path):
        buf = self._buf_with_tiff(tmp_path)
        with pytest.raises(ValueError, match="chunks=.*file-like"):
            open_geotiff(buf, chunks=64)

    def test_chunks_with_pathlib_path_still_works(self, tmp_path):
        """pathlib.Path is not file-like and must keep working through the
        dask path."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = tmp_path / 'sample.tif'
        to_geotiff(arr, str(path))

        import dask.array as da_mod
        result = open_geotiff(path, chunks=2)
        assert isinstance(result.data, da_mod.Array)
        np.testing.assert_array_equal(np.asarray(result.data), arr)
