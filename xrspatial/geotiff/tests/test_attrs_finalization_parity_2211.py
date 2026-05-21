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

The test is parametrized over a small fixture matrix (no nodata, float
sentinel, integer sentinel, MinIsWhite photometric) and over the
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
from typing import Callable

import numpy as np
import pytest

import xarray as xr

from xrspatial.geotiff import open_geotiff, read_vrt, to_geotiff


tifffile = pytest.importorskip("tifffile")


def _coord_array(arr: np.ndarray) -> xr.DataArray:
    """Wrap ``arr`` in a DataArray with axis-aligned x/y coords + CRS.

    ``to_geotiff`` stamps the TIFF GeoKey set when the source DataArray
    has y/x coords and ``attrs['crs']``. Using the writer for the
    fixtures keeps the GeoKey emission identical to a real read/write
    round-trip so the test exercises the same code path users hit.
    """
    h, w = arr.shape[:2]
    y = np.linspace(40.0, 40.0 - 0.001 * (h - 1), h)
    x = np.linspace(-100.0, -100.0 + 0.001 * (w - 1), w)
    da = xr.DataArray(
        arr, dims=['y', 'x'] if arr.ndim == 2 else ['y', 'x', 'band'],
        coords={'y': y, 'x': x},
    )
    da.attrs['crs'] = 4326
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


def _attrs_close(a: dict, b: dict, *, rtol: float = 1e-9) -> bool:
    """Compare attrs dicts, allowing tiny numeric drift in ``transform``.

    The VRT writer emits the geo-transform as ``%.6f`` ASCII (GDAL's
    own convention) while the TIFF writer keeps the original float64
    values, so the same logical transform comes back as
    ``0.001`` vs ``0.0010000000000047748``. The diff is well below
    GDAL's own rounding step and below any sane pixel-size tolerance,
    so the parity check accepts it. All other keys must match exactly.
    """
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        vb = b[k]
        if k == 'transform' and isinstance(va, tuple) and isinstance(vb, tuple):
            if len(va) != len(vb):
                return False
            for x, y in zip(va, vb):
                if not np.isclose(float(x), float(y), rtol=rtol, atol=1e-9):
                    return False
        else:
            if va != vb:
                return False
    return True


# ---------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------

def _write_plain_float(path):
    """Plain float32 TIFF: no nodata, axis-aligned transform, EPSG:4326."""
    arr = np.random.default_rng(seed=2227).random(
        (32, 32)).astype(np.float32)
    to_geotiff(_coord_array(arr), path)
    return arr


def _write_float_with_nodata(path):
    """Float32 TIFF with a declared sentinel (-9999.0). Some pixels match."""
    rng = np.random.default_rng(seed=2227)
    arr = rng.random((32, 32)).astype(np.float32)
    arr[0:4, 0:4] = -9999.0
    da = _coord_array(arr)
    da.attrs['nodata'] = -9999.0
    to_geotiff(da, path)
    return arr


def _write_int_with_nodata(path):
    """uint16 TIFF with a representable sentinel pixel."""
    rng = np.random.default_rng(seed=2227)
    arr = rng.integers(0, 1000, size=(32, 32), dtype=np.uint16)
    arr[0:4, 0:4] = 65535
    da = _coord_array(arr)
    da.attrs['nodata'] = 65535
    to_geotiff(da, path)
    return arr


def _write_minis_white(path):
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
    return arr


@dataclass(frozen=True)
class _Fixture:
    """One row in the parity matrix.

    ``writer`` materializes the TIFF on disk; ``vrt_compatible`` tells
    the test whether to also exercise the VRT backend (some sentinel
    layouts need extra VRT bookkeeping that this minimal wrapper does
    not synthesise).
    """
    name: str
    writer: Callable[[str], np.ndarray]
    vrt_compatible: bool = True


_FIXTURES = (
    _Fixture('plain_float', _write_plain_float),
    _Fixture('float_with_nodata', _write_float_with_nodata),
    _Fixture('int_with_nodata', _write_int_with_nodata),
    _Fixture('uint8_no_nodata', _write_minis_white),
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


def _open_vrt(path):
    """Wrap the TIFF in a single-source VRT and read it via ``read_vrt``.

    Building a one-liner VRT on the fly lets the test exercise the
    VRT eager backend through the same finalization helper without
    needing a hand-written VRT fixture per case. The source TIFF's
    geometry, CRS, and nodata are read back via the public
    :func:`open_geotiff` so the wrapper does not need to re-parse the
    TIFF header internals (which keeps it stable across header
    refactors).
    """
    import os

    src = open_geotiff(path)
    height, width = src.shape[:2]

    transform = src.attrs.get('transform')
    crs_wkt = src.attrs.get('crs_wkt') or ''
    nodata = src.attrs.get('nodata')

    # Map numpy dtype to the GDAL VRT DataType label.
    dtype_map = {
        np.dtype('uint8'): 'Byte',
        np.dtype('int8'): 'Int8',
        np.dtype('uint16'): 'UInt16',
        np.dtype('int16'): 'Int16',
        np.dtype('uint32'): 'UInt32',
        np.dtype('int32'): 'Int32',
        np.dtype('float32'): 'Float32',
        np.dtype('float64'): 'Float64',
    }
    vrt_dtype = dtype_map.get(src.dtype, 'Float32')

    if transform is not None:
        # rasterio-style tuple is (pixel_width, 0.0, origin_x, 0.0,
        # pixel_height, origin_y); GDAL GeoTransform XML wants the
        # GDAL order (origin_x, pixel_width, row_skew, origin_y,
        # col_skew, pixel_height).
        pixel_width, _, origin_x, _, pixel_height, origin_y = transform
        geo_transform = (
            f"{origin_x:.6f}, {pixel_width:.6f}, 0.0, "
            f"{origin_y:.6f}, 0.0, {pixel_height:.6f}"
        )
    else:
        # No georef: use an identity-like transform; ``read_vrt`` still
        # parses the file but emits ``georef_status='transform_only'``
        # because the synthesised GeoTransform is axis-aligned.
        geo_transform = '0.0, 1.0, 0.0, 0.0, 0.0, -1.0'

    nodata_xml = (f"<NoDataValue>{nodata}</NoDataValue>"
                  if nodata is not None else '')

    vrt_path = path + '.vrt'
    abs_src = os.path.abspath(path)
    xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f'  <SRS>{crs_wkt}</SRS>'
        f'  <GeoTransform>{geo_transform}</GeoTransform>'
        f'  <VRTRasterBand dataType="{vrt_dtype}" band="1">'
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
    _Backend('eager_numpy', _open_eager),
    _Backend('dask_numpy', _open_dask),
    _Backend('gpu', _open_gpu, available=_HAS_GPU),
    _Backend('dask_gpu', _open_dask_gpu, available=_HAS_GPU),
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
    fixture.writer(path)

    baseline = _attrs_for_parity(open_geotiff(path).attrs)

    divergences = {}
    for backend in _AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            # Already the baseline; comparing it to itself adds noise.
            continue
        try:
            da = backend.open_fn(path)
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
    fixture.writer(path)

    baseline_keys = set(_attrs_for_parity(open_geotiff(path).attrs).keys())

    diffs = {}
    for backend in _AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            continue
        try:
            da = backend.open_fn(path)
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
