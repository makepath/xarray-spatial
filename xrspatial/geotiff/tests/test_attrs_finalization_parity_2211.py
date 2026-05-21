"""Table-driven attrs parity across all read backends (PR-D of #2211).

After PRs #2200, #2205, #2207, #2209 from epic #2162, every read
backend routes its attrs assembly through one of two finalization
entry points in ``xrspatial.geotiff._attrs``:

* :func:`_finalize_eager_read` for the eager numpy + GPU paths.
* :func:`_finalize_lazy_read_attrs` for the dask + dask-GPU + VRT
  paths.

PR-D of #2211 closes the loop on issue #2227 by removing the last
post-helper ``attrs[k] = v`` writes from the backends (the VRT eager
path's ``nodata_pixels_present`` stamp now rides through the helper's
``pixels_present`` kwarg). The test below pins the resulting contract:
for the same on-disk fixture, every backend that handles a given
read emits the same canonical attrs.

The test is parametrized over a small fixture matrix (no nodata,
float sentinel, integer sentinel, uint8 no-nodata) and over the
backends available on the runner (eager numpy is always present;
dask+numpy ditto; GPU + dask+GPU only when CuPy + CUDA are usable;
VRT exercised via a tiny wrapper that points at the underlying TIFF).
A small set of backend-specific keys is excluded from the comparison:

* The VRT path intentionally omits TIFF tag pass-through attrs:
  ``extra_tags``, ``image_description``, ``extra_samples``,
  ``colormap``, ``gdal_metadata``, ``gdal_metadata_xml``,
  ``x_resolution``, ``y_resolution``, and ``resolution_unit``.
  The VRT carries no TIFF tags of its own; the non-VRT path
  documented those keys as TIFF-only.
* The VRT path adds ``vrt_holes`` on missing-source reads.
* ``nodata_pixels_present`` rides on the eager + VRT paths via a
  one-pass scan but stays absent on dask paths (issue #2135).

These exclusions are encoded in :data:`_BACKEND_SPECIFIC_KEYS` so a
future migration that promotes one of those keys to all-backends has
a single place to remove the carve-out.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

import xarray as xr

from xrspatial.geotiff import open_geotiff, read_vrt, to_geotiff


tifffile = pytest.importorskip("tifffile")


def _coord_array(arr: np.ndarray) -> xr.DataArray:
    """Wrap a 2-D ``arr`` in a DataArray with axis-aligned x/y coords + CRS.

    ``to_geotiff`` stamps the TIFF GeoKey set when the source DataArray
    has y/x coords and ``attrs['crs']``. Using the writer for the
    fixtures keeps the GeoKey emission identical to a real read/write
    round-trip so the test exercises the same code path users hit.

    Only 2-D arrays are supported because every fixture in this file
    is single-band. A multi-band fixture would also need a ``band``
    coord and the VRT helper would need per-band nodata bookkeeping;
    that is out of scope for the canonical-attrs parity assertion.
    """
    assert arr.ndim == 2, "test fixtures only use 2-D arrays"
    h, w = arr.shape
    assert (h, w) == (_FIX_HEIGHT, _FIX_WIDTH), (
        "fixture geometry constants are out of sync with array shape; "
        "update _FIX_HEIGHT / _FIX_WIDTH together with the fixture writers"
    )
    y = np.linspace(_FIX_ORIGIN_Y, _FIX_ORIGIN_Y - _FIX_PIXEL * (h - 1), h)
    x = np.linspace(_FIX_ORIGIN_X, _FIX_ORIGIN_X + _FIX_PIXEL * (w - 1), w)
    da = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
    )
    da.attrs['crs'] = _FIX_CRS_EPSG
    return da


def _gpu_available() -> bool:
    """Return True iff cupy is importable and the runtime sees a CUDA device."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


# Canonical fixture geometry. Both the TIFF writer (via ``_coord_array``)
# and the VRT wrapper read from these constants so the VRT helper does
# not have to call ``open_geotiff`` to discover the on-disk transform.
# A silent drift in the eager read path therefore cannot propagate
# into the VRT helper.
_FIX_ORIGIN_X = -100.0
_FIX_ORIGIN_Y = 40.0
_FIX_PIXEL = 0.001
_FIX_CRS_EPSG = 4326
_FIX_HEIGHT = 32
_FIX_WIDTH = 32


# Keys to drop before comparing attrs across backends. Each key is
# documented as backend-specific in the attrs contract (issue #1984)
# and the surrounding modules:
#
# * ``vrt_holes`` -- VRT-only, populated from skipped sources at decode
#   time. Plain TIFF reads never see it.
# * ``nodata_pixels_present`` -- emitted by the eager + VRT paths after
#   a one-pass scan but absent on dask paths (#2135). The dask backends
#   would have to force ``.compute()`` to produce it, breaking lazy.
# * TIFF tag pass-through attrs -- the VRT path documents these as
#   omitted because the VRT carries no TIFF tags of its own. They are
#   pinned in ``test_attrs_parity_1548`` for the non-VRT backends.
_BACKEND_SPECIFIC_KEYS = frozenset({
    'vrt_holes',
    'nodata_pixels_present',
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


def _attrs_for_parity(attrs) -> dict:
    """Drop backend-specific keys before comparing attrs across paths."""
    return {k: v for k, v in dict(attrs).items()
            if k not in _BACKEND_SPECIFIC_KEYS}


# Tolerance for the lone numeric key that needs one (``transform``).
# The VRT writer emits the geo-transform as ``%.6f`` ASCII (GDAL's own
# convention) while the TIFF writer keeps the original float64 values,
# so the same logical transform comes back as ``0.001`` vs
# ``0.0010000000000047748``. The diff sits well below GDAL's own
# rounding step and below any sane pixel-size tolerance.
_TRANSFORM_RTOL = 1e-9
_TRANSFORM_ATOL = 1e-9


def _attrs_close(a: dict, b: dict) -> bool:
    """Compare attrs dicts, allowing tiny numeric drift in ``transform``.

    All keys other than ``transform`` must match exactly. Only the
    ``transform`` 6-tuple is compared with a tolerance (see
    :data:`_TRANSFORM_RTOL` / :data:`_TRANSFORM_ATOL`).
    """
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        vb = b[k]
        if k == 'transform' and isinstance(va, tuple) and isinstance(vb, tuple):
            if len(va) != len(vb):
                return False
            for x, y in zip(va, vb):
                if not np.isclose(
                    float(x), float(y),
                    rtol=_TRANSFORM_RTOL, atol=_TRANSFORM_ATOL,
                ):
                    return False
        else:
            if va != vb:
                return False
    return True


# ---------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Fixture:
    """One row in the parity matrix.

    ``writer`` materializes the TIFF on disk and returns a
    :class:`_FixtureMeta` describing the on-disk layout (dtype,
    declared nodata sentinel). The VRT helper consumes the meta
    directly rather than re-deriving it from a TIFF read, which
    keeps the VRT wrapper independent of the eager read path.
    """
    name: str
    writer: Callable[[str], '_FixtureMeta']
    vrt_compatible: bool = True


@dataclass(frozen=True)
class _FixtureMeta:
    """Layout facts the VRT helper needs to wrap the on-disk TIFF."""
    vrt_dtype: str   # GDAL VRT DataType label ('Float32', 'UInt16', ...)
    nodata: Any = None


def _write_plain_float(path) -> _FixtureMeta:
    """Plain float32 TIFF: no nodata, axis-aligned transform, EPSG:4326."""
    arr = np.random.default_rng(seed=2227).random(
        (32, 32)).astype(np.float32)
    to_geotiff(_coord_array(arr), path)
    return _FixtureMeta(vrt_dtype='Float32')


def _write_float_with_nodata(path) -> _FixtureMeta:
    """Float32 TIFF with a declared sentinel (-9999.0). Some pixels match."""
    rng = np.random.default_rng(seed=2227)
    arr = rng.random((32, 32)).astype(np.float32)
    arr[0:4, 0:4] = -9999.0
    da = _coord_array(arr)
    da.attrs['nodata'] = -9999.0
    to_geotiff(da, path)
    return _FixtureMeta(vrt_dtype='Float32', nodata=-9999.0)


def _write_int_with_nodata(path) -> _FixtureMeta:
    """uint16 TIFF with a representable sentinel pixel."""
    rng = np.random.default_rng(seed=2227)
    arr = rng.integers(0, 1000, size=(32, 32), dtype=np.uint16)
    arr[0:4, 0:4] = 65535
    da = _coord_array(arr)
    da.attrs['nodata'] = 65535
    to_geotiff(da, path)
    return _FixtureMeta(vrt_dtype='UInt16', nodata=65535)


def _write_uint8_no_nodata(path) -> _FixtureMeta:
    """uint8 photometric MinIsBlack baseline (no MinIsWhite via to_geotiff).

    ``to_geotiff`` does not currently expose a MinIsWhite kwarg, so the
    fixture is a uint8 with photometric=MinIsBlack. The MinIsWhite
    branch of ``_finalize_eager_read``'s sentinel resolution is
    covered by ``test_eager_finalization_parity_2162``; here we just
    exercise the uint8 dtype against the canonical schema.
    """
    rng = np.random.default_rng(seed=2227)
    arr = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    to_geotiff(_coord_array(arr), path)
    return _FixtureMeta(vrt_dtype='Byte')


_FIXTURES = (
    _Fixture('plain_float', _write_plain_float),
    _Fixture('float_with_nodata', _write_float_with_nodata),
    _Fixture('int_with_nodata', _write_int_with_nodata),
    _Fixture('uint8_no_nodata', _write_uint8_no_nodata),
)


# ---------------------------------------------------------------------
# Backend table
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Backend:
    name: str
    open_fn: Callable
    available: bool = True


def _open_eager(path):
    return open_geotiff(path)


def _open_dask(path):
    return open_geotiff(path, chunks=16)


def _open_gpu(path):
    return open_geotiff(path, gpu=True)


def _open_dask_gpu(path):
    return open_geotiff(path, gpu=True, chunks=16)


def _open_vrt(path, meta):
    """Wrap the TIFF in a single-source VRT and read it via ``read_vrt``.

    Building a one-liner VRT on the fly lets the test exercise the
    VRT eager backend through the same finalization helper without
    needing a hand-written VRT fixture per case. ``meta`` carries the
    on-disk dtype and nodata sentinel that the fixture writer just
    used; the geometry, CRS, and pixel size come from the module-level
    fixture constants. Both inputs are derived from the same source of
    truth as ``_coord_array`` (via ``to_geotiff``), so a silent drift
    in :func:`open_geotiff` cannot propagate into this helper.
    """
    import os
    from pyproj import CRS

    height = _FIX_HEIGHT
    width = _FIX_WIDTH

    # GDAL GeoTransform XML wants (origin_x, pixel_width, row_skew,
    # origin_y, col_skew, pixel_height). y axis decreases.
    #
    # The TIFF writer follows the RasterPixelIsArea convention and
    # stamps the upper-left CORNER as the origin, while the fixture
    # constants name the upper-left CENTER (matching ``_coord_array``,
    # which uses center-based y/x coords). Shift by half a pixel here
    # so the VRT-side transform matches the TIFF-side transform
    # within ``_TRANSFORM_RTOL``.
    corner_x = _FIX_ORIGIN_X - _FIX_PIXEL / 2.0
    corner_y = _FIX_ORIGIN_Y + _FIX_PIXEL / 2.0
    geo_transform = (
        f"{corner_x:.6f}, {_FIX_PIXEL:.6f}, 0.0, "
        f"{corner_y:.6f}, 0.0, {-_FIX_PIXEL:.6f}"
    )
    crs_wkt = CRS.from_epsg(_FIX_CRS_EPSG).to_wkt()

    nodata_xml = (f"<NoDataValue>{meta.nodata}</NoDataValue>"
                  if meta.nodata is not None else '')

    vrt_path = path + '.vrt'
    abs_src = os.path.abspath(path)
    xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f'  <SRS>{crs_wkt}</SRS>'
        f'  <GeoTransform>{geo_transform}</GeoTransform>'
        f'  <VRTRasterBand dataType="{meta.vrt_dtype}" band="1">'
        f'    {nodata_xml}'
        f'    <SimpleSource>'
        f'      <SourceFilename relativeToVRT="0">{abs_src}</SourceFilename>'
        f'      <SourceBand>1</SourceBand>'
        f'      <SrcRect xOff="0" yOff="0" '
        f'               xSize="{width}" ySize="{height}"/>'
        f'      <DstRect xOff="0" yOff="0" '
        f'               xSize="{width}" ySize="{height}"/>'
        f'    </SimpleSource>'
        f'  </VRTRasterBand>'
        f'</VRTDataset>'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)
    return read_vrt(vrt_path)


_BACKENDS = (
    _Backend('eager_numpy', lambda path, meta: _open_eager(path)),
    _Backend('dask_numpy', lambda path, meta: _open_dask(path)),
    _Backend('gpu', lambda path, meta: _open_gpu(path), available=_HAS_GPU),
    _Backend('dask_gpu', lambda path, meta: _open_dask_gpu(path),
             available=_HAS_GPU),
    _Backend('vrt', _open_vrt),
)


_AVAILABLE_BACKENDS = tuple(b for b in _BACKENDS if b.available)


# ---------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------

@pytest.mark.parametrize('fixture', _FIXTURES, ids=lambda f: f.name)
def test_canonical_attrs_match_across_backends(tmp_path, fixture):
    """Every backend stamps the same canonical attrs for the same fixture.

    The eager numpy path is the canonical reference. For each fixture
    we open the file via every available backend (skipping the VRT
    backend on fixtures whose wrapper cannot model them) and assert
    the comparable attrs (canonical contract minus the documented
    backend-specific carve-outs) match the eager-numpy baseline.

    Any divergence here means a backend has slipped out of lockstep
    with the finalization helpers in ``_attrs.py``. The expected fix
    is to route that backend through the helper rather than papering
    over the diff with a new entry in :data:`_BACKEND_SPECIFIC_KEYS`.
    """
    path = str(tmp_path / f'parity_2227_{fixture.name}.tif')
    meta = fixture.writer(path)

    baseline = _attrs_for_parity(open_geotiff(path).attrs)

    divergences = {}
    for backend in _AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            # Already the baseline; comparing it to itself adds noise.
            continue
        try:
            da = backend.open_fn(path, meta)
        except Exception as exc:  # pragma: no cover - surfaced via the assert
            divergences[backend.name] = f"open failed: {exc!r}"
            continue
        candidate = _attrs_for_parity(da.attrs)
        if not _attrs_close(candidate, baseline):
            only_in_baseline = {
                k: baseline.get(k) for k in baseline
                if baseline.get(k) != candidate.get(k)
            }
            only_in_candidate = {
                k: candidate.get(k) for k in candidate
                if candidate.get(k) != baseline.get(k)
            }
            divergences[backend.name] = {
                'baseline_diff': only_in_baseline,
                'candidate_diff': only_in_candidate,
            }

    assert not divergences, (
        f"attrs diverged from eager-numpy baseline for fixture "
        f"{fixture.name!r}:\n  baseline: {baseline}\n  diffs: {divergences}"
    )


@pytest.mark.parametrize('fixture', _FIXTURES, ids=lambda f: f.name)
def test_canonical_attrs_keys_match_across_backends(tmp_path, fixture):
    """Stronger contract: the set of canonical attr keys is identical.

    Even when values match the per-key comparison above, a backend
    that silently *drops* a key from the canonical set would slip
    through if the value happens to be ``None``. This check pins the
    keyset so a future regression that omits ``georef_status`` or
    ``crs_wkt`` from one backend surfaces immediately.
    """
    path = str(tmp_path / f'parity_2227_keys_{fixture.name}.tif')
    meta = fixture.writer(path)

    baseline_keys = set(_attrs_for_parity(open_geotiff(path).attrs).keys())

    diffs = {}
    for backend in _AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            continue
        try:
            da = backend.open_fn(path, meta)
        except Exception as exc:  # pragma: no cover
            diffs[backend.name] = f"open failed: {exc!r}"
            continue
        keys = set(_attrs_for_parity(da.attrs).keys())
        if keys != baseline_keys:
            diffs[backend.name] = {
                'missing': sorted(baseline_keys - keys),
                'extra': sorted(keys - baseline_keys),
            }

    assert not diffs, (
        f"canonical attrs keyset diverged from eager-numpy baseline for "
        f"fixture {fixture.name!r}:\n  baseline keys: {sorted(baseline_keys)}\n"
        f"  diffs: {diffs}"
    )


def test_backend_specific_keys_carveout_is_documented():
    """Sanity: every key in the carve-out is documented in this module.

    The module docstring lists which keys are backend-specific and
    therefore excluded from the parity assertions. The carve-out and
    the docstring drift apart easily; the check here is a string scan
    so a future maintainer who adds a key to the frozenset has to
    update the docstring too.
    """
    module_doc = __doc__ or ''
    missing = [k for k in _BACKEND_SPECIFIC_KEYS if k not in module_doc]
    assert not missing, (
        f"keys in _BACKEND_SPECIFIC_KEYS are not mentioned in the module "
        f"docstring: {missing}"
    )
