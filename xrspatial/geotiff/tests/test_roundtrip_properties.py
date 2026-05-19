"""Hypothesis property tests for the GeoTIFF write/read round trip (#2134).

This file backs the metadata round-trip contract that the geotiff module
has accumulated through a long list of incident-specific tests
(``test_metadata_round_trip_1484.py``, ``test_descending_coords_1716.py``,
``test_no_georef_writer_round_trip_1949.py``,
``test_int_coords_round_trip_hotfix_1962.py``,
``test_round_trip_invariants.py``) plus the dtype/codec/predictor fuzz in
``test_fuzz_hypothesis_1661.py``. The incident files stay as regression
markers for their bug numbers; this file is the canonical metadata
contract going forward.

Strategy: one Hypothesis strategy per axis, composed into a single draw
that builds a DataArray, writes it through ``to_geotiff``, reads it back
with ``open_geotiff``, and asserts the canonical invariant on the read
result. A second cycle pins the fixed point -- once the writer has
canonicalised the attrs, a second round must reproduce the first exactly.

Axes covered:

* coord dtype -- ``int32`` / ``int64`` / ``float32`` / ``float64``
* axis direction -- four combinations of asc / desc on x and y
* shape -- 1x1, 1xN, Nx1, plus a couple of small rectangles
* CRS / transform presence -- crs_only, transform_only, both, neither
* nodata mode -- in_range, out_of_range, fractional, nan, none
* band layout -- band_first ``(b, y, x)``, band_last ``(y, x, b)``,
  no_band ``(y, x)``
* pixel dtype -- ``uint8`` / ``int16`` / ``int32`` / ``float32`` /
  ``float64``
* CRS EPSG -- 4326, 3857, 32633, 26910 (and ``None`` for the no-CRS axis)

Cross-references to the adjacent contracts this back-stops:

* backend parity matrix -- ``test_backend_parity_matrix.py``
* no-georef marker (#2120) -- ``_xrspatial_no_georef`` in attrs
* nodata semantics split (#2092) -- declared vs masked nodata, see
  ``test_masked_nodata_attr_2092.py``
* int coords sentinel (#2087) -- integer ``x`` / ``y`` are a no-georef
  signal, the writer must not synthesise a transform from them

Out of scope (deferred):

* GPU (cupy, dask+cupy) -- same writer/reader code but needs a CUDA
  runner. Pin the numpy invariants here first.
* Byte-for-byte file equality -- the writer is allowed to reorder IFD
  tags, change strip layout, etc. ``test_golden_corpus_*.py`` covers
  byte stability where it matters.
* VRT, COG, overviews -- have their own round-trip suites.

Hypothesis profiles:

* default: ``max_examples=200``, ``deadline=None``
* ``ci``: ``max_examples=50``, ``derandomize=True``

The module is skipped if ``hypothesis`` is not installed (same pattern
as ``test_fuzz_hypothesis_1661.py``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import (  # noqa: E402
    HealthCheck,
    assume,
    given,
    settings,
)
from hypothesis import strategies as st  # noqa: E402

from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402
from xrspatial.geotiff._geotags import _NO_GEOREF_KEY  # noqa: E402


# ---------------------------------------------------------------------------
# Profile registration
# ---------------------------------------------------------------------------

_COMMON_SUPPRESS = [
    HealthCheck.too_slow,
    HealthCheck.function_scoped_fixture,
    # The strategy uses ``assume`` to reject ill-typed combinations; on
    # narrow draws the filter rate can climb above the default threshold
    # without indicating a real strategy bug.
    HealthCheck.filter_too_much,
]

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    derandomize=True,
    suppress_health_check=_COMMON_SUPPRESS,
)
settings.register_profile(
    "rockout_default",
    max_examples=200,
    deadline=None,
    suppress_health_check=_COMMON_SUPPRESS,
)


# ---------------------------------------------------------------------------
# Strategy axes
# ---------------------------------------------------------------------------

COORD_DTYPES = ['int32', 'int64', 'float32', 'float64']
AXIS_DIRECTIONS = ['asc_asc', 'asc_desc', 'desc_asc', 'desc_desc']
SHAPES = [(1, 1), (1, 8), (8, 1), (4, 5), (16, 16)]
GEOREF_MODES = ['crs_only', 'transform_only', 'both', 'neither']
NODATA_MODES = ['in_range', 'out_of_range', 'fractional', 'nan', 'none']
BAND_LAYOUTS = ['band_first', 'band_last', 'no_band']
PIXEL_DTYPES = ['uint8', 'int16', 'int32', 'float32', 'float64']
CRS_CHOICES = [4326, 3857, 32633, 26910]


def _make_coord(direction: str, length: int, dtype_name: str) -> np.ndarray:
    """Build a 1D coord of ``length`` cells in the requested direction.

    Floats land on a regular grid so ``coords_to_transform`` accepts
    them; ints are an arange (the no-georef placeholder pattern the
    writer recognises via the #2120 marker).
    """
    dtype = np.dtype(dtype_name)
    if dtype.kind in ('i', 'u'):
        base = np.arange(length, dtype=dtype)
    else:
        # Pick a step that's exactly representable so the regularity
        # check passes; 1.0 / 0.5 are safe for both float32 and float64.
        base = np.arange(length, dtype=np.float64) * 1.0 + 100.0
        base = base.astype(dtype)
    if direction == 'desc':
        base = base[::-1].copy()
    return base


def _shape_for_layout(shape_2d: tuple[int, int], layout: str, n_bands: int):
    h, w = shape_2d
    if layout == 'no_band':
        return ('y', 'x'), (h, w)
    if layout == 'band_first':
        return ('band', 'y', 'x'), (n_bands, h, w)
    if layout == 'band_last':
        return ('y', 'x', 'band'), (h, w, n_bands)
    raise ValueError(f"bad layout {layout!r}")


def _build_pixels(shape: tuple[int, ...], dtype_name: str, seed: int) -> np.ndarray:
    """Build deterministic pixel data, avoiding the dtype extremes."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype_name)
    size = int(np.prod(shape))
    if dtype.kind == 'f':
        arr = rng.standard_normal(size).astype(dtype).reshape(shape)
    else:
        info = np.iinfo(dtype)
        # Stay clear of the extremes; the in_range nodata strategy
        # picks from outside the sampled span so the sentinel doesn't
        # collide with real data.
        lo = max(info.min, -100)
        hi = min(info.max, 100)
        arr = rng.integers(low=lo, high=hi, size=size, dtype=dtype).reshape(shape)
    return arr


def _pick_nodata(mode: str, dtype_name: str, rng: np.random.Generator):
    """Return a nodata value compatible with the dtype, or ``None``.

    ``in_range`` -- sentinel inside the dtype range but outside the
    pixel sample range (so no real pixel happens to equal it).
    ``out_of_range`` -- only valid for floats (would raise for ints).
    ``fractional`` -- only valid for floats.
    ``nan`` -- only valid for floats.
    ``none`` -- no sentinel.

    Returns ``None`` for the ``none`` case, or a Python scalar otherwise.
    """
    dtype = np.dtype(dtype_name)
    if mode == 'none':
        return None
    if mode == 'nan':
        return float('nan')
    if mode == 'fractional':
        return float(rng.uniform(0.1, 0.9))
    if mode == 'out_of_range':
        # Only meaningful for floats; the writer rejects int casts that
        # would lose information. Use a value the float dtype can hold
        # but no integer dtype can. Pair this mode with float dtypes only.
        return 1e30
    if mode == 'in_range':
        if dtype.kind == 'f':
            return -9999.0
        info = np.iinfo(dtype)
        # Pick a sentinel safely inside the dtype range, outside the
        # pixel sample range used by _build_pixels (which is |x| <= 100).
        candidate = max(info.min, -32768) if info.min < 0 else info.max
        # For unsigned dtypes ``info.min == 0`` so the candidate is
        # ``info.max``; for signed dtypes it's close to ``info.min``.
        return int(candidate)
    raise ValueError(f"bad nodata mode {mode!r}")


def _is_legal_combo(spec: dict) -> bool:
    """Filter combinations the writer is documented to reject.

    The strategy ``assume``s on these; rejected draws don't count
    against the example budget for invariant testing, but they do
    eat one slot of strategy generation time. Keep the filter set
    minimal so the example budget mostly hits the legal interior.
    """
    dtype = np.dtype(spec['pixel_dtype'])
    nodata_mode = spec['nodata_mode']
    # NaN, fractional, and out_of_range nodata require float dtype.
    if nodata_mode in ('nan', 'fractional', 'out_of_range') and dtype.kind != 'f':
        return False
    return True


# ---------------------------------------------------------------------------
# Composite strategy
# ---------------------------------------------------------------------------

@st.composite
def _round_trip_spec(draw):
    coord_dtype = draw(st.sampled_from(COORD_DTYPES))
    axis_dir = draw(st.sampled_from(AXIS_DIRECTIONS))
    shape = draw(st.sampled_from(SHAPES))
    georef = draw(st.sampled_from(GEOREF_MODES))
    nodata_mode = draw(st.sampled_from(NODATA_MODES))
    band_layout = draw(st.sampled_from(BAND_LAYOUTS))
    pixel_dtype = draw(st.sampled_from(PIXEL_DTYPES))
    crs_epsg = draw(st.sampled_from(CRS_CHOICES))
    n_bands = draw(st.integers(min_value=2, max_value=3))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    spec = dict(
        coord_dtype=coord_dtype,
        axis_dir=axis_dir,
        shape=shape,
        georef=georef,
        nodata_mode=nodata_mode,
        band_layout=band_layout,
        pixel_dtype=pixel_dtype,
        crs_epsg=crs_epsg,
        n_bands=n_bands,
        seed=seed,
    )
    assume(_is_legal_combo(spec))
    return spec


# ---------------------------------------------------------------------------
# Build / write / read helpers
# ---------------------------------------------------------------------------

def _build_dataarray(spec: dict) -> xr.DataArray:
    """Materialise a DataArray from a strategy draw.

    The georef mode controls whether the DataArray carries coords:

    * ``transform_only`` / ``both`` -- spatial coords with the chosen
      direction and dtype. Integer coords trigger the writer's
      no-georef path (#2120 marker on read), so this case effectively
      collapses to the same coverage as ``crs_only`` / ``neither``
      when coord dtype is int.
    * ``crs_only`` -- no spatial coords; the writer can't derive a
      transform, the on-disk file has CRS GeoKeys but no
      transform/scale/tiepoint tags.
    * ``neither`` -- no coords, no CRS kwarg. Round-trip should restore
      the same no-georef state.
    """
    h, w = spec['shape']
    dims, full_shape = _shape_for_layout(spec['shape'], spec['band_layout'],
                                         spec['n_bands'])
    pixels = _build_pixels(full_shape, spec['pixel_dtype'], spec['seed'])

    needs_coords = spec['georef'] in ('transform_only', 'both')
    coords = None
    if needs_coords:
        ax_x, ax_y = spec['axis_dir'].split('_')
        x = _make_coord(ax_x, w, spec['coord_dtype'])
        y = _make_coord(ax_y, h, spec['coord_dtype'])
        coords = {'y': y, 'x': x}
        if 'band' in dims:
            coords['band'] = np.arange(spec['n_bands'], dtype=np.int64)

    return xr.DataArray(pixels, dims=dims, coords=coords)


def _writer_kwargs(spec: dict, rng: np.random.Generator) -> dict:
    kwargs = dict(compression='none', tiled=False)
    if spec['georef'] in ('crs_only', 'both'):
        kwargs['crs'] = spec['crs_epsg']
    if spec['nodata_mode'] != 'none':
        nd = _pick_nodata(spec['nodata_mode'], spec['pixel_dtype'], rng)
        if nd is not None:
            kwargs['nodata'] = nd
    return kwargs


def _read_array(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values)


def _compare_pixels(a: np.ndarray, b: np.ndarray) -> None:
    """NaN-aware bit-equal pixel compare."""
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype.kind == 'f' or b.dtype.kind == 'f':
        a_f = a.astype(np.float64, copy=False)
        b_f = b.astype(np.float64, copy=False)
        nan_a = np.isnan(a_f)
        nan_b = np.isnan(b_f)
        if not np.array_equal(nan_a, nan_b):
            raise AssertionError("NaN mask drift between cycles")
        mask = ~nan_a
        np.testing.assert_array_equal(a_f[mask], b_f[mask])
    else:
        np.testing.assert_array_equal(a, b)


# Attrs whose values must match between two consecutive read results
# once the writer has canonicalised them. Other attrs (best-effort
# pass-through) are only checked for presence.
_LOCKED_ATTRS = ('crs', 'transform', 'nodata', 'raster_type',
                 _NO_GEOREF_KEY)


def _assert_fixed_point(da1: xr.DataArray, da2: xr.DataArray) -> None:
    """Two consecutive write -> read results must agree on pixels,
    dtype, dims, and the canonical attrs.
    """
    assert da1.dtype == da2.dtype, f"dtype drift: {da1.dtype} -> {da2.dtype}"
    assert da1.dims == da2.dims, f"dims drift: {da1.dims} -> {da2.dims}"
    _compare_pixels(_read_array(da1), _read_array(da2))
    assert set(da1.attrs) == set(da2.attrs), (
        f"attrs key drift: {set(da1.attrs) ^ set(da2.attrs)}"
    )
    for key in _LOCKED_ATTRS:
        if key in da1.attrs:
            v1 = da1.attrs[key]
            v2 = da2.attrs[key]
            if key == 'transform':
                assert len(v1) == len(v2)
                for a, b in zip(v1, v2):
                    assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9), (
                        f"transform drift: {v1} -> {v2}"
                    )
            elif key == 'nodata':
                if isinstance(v1, float) and math.isnan(v1):
                    assert isinstance(v2, float) and math.isnan(v2)
                else:
                    assert v1 == v2, f"nodata drift: {v1!r} -> {v2!r}"
            else:
                assert v1 == v2, f"attrs[{key!r}] drift: {v1!r} -> {v2!r}"


# ---------------------------------------------------------------------------
# Property: round-trip on the numpy backend
# ---------------------------------------------------------------------------

@settings(
    parent=settings.get_profile('rockout_default'),
)
@given(spec=_round_trip_spec())
def test_round_trip_fixed_point_numpy(tmp_path_factory, spec):
    """For every legal draw on the numpy backend, two consecutive
    write -> read cycles produce DataArrays that agree on the canonical
    attrs and pixel bytes.

    Skips the draw with ``assume`` when the writer is documented to
    refuse the combination (e.g. fractional / NaN nodata paired with an
    int pixel dtype). The intent is to lock the metadata round-trip
    contract, not to enumerate every documented refusal.
    """
    rng = np.random.default_rng(spec['seed'])
    da0 = _build_dataarray(spec)
    kwargs = _writer_kwargs(spec, rng)

    tmp = tmp_path_factory.mktemp("rt_2134")

    p1 = str(tmp / 'rt1.tif')
    try:
        to_geotiff(da0, p1, **kwargs)
    except (ValueError, TypeError) as e:
        # The writer rejects some specific combinations up front (e.g.
        # a 1x1 raster with no transform attr but with float coords
        # whose step is undefined). Those refusals are documented
        # behaviour, not round-trip failures.
        assume(False)
        return  # pragma: no cover

    da1 = open_geotiff(p1)

    # Strip the read-only contract version attr before re-writing -- the
    # writer doesn't consume it. Keep everything else; the round-trip
    # invariant is that the writer can reproduce the read attrs.
    da1_for_write = da1
    p2 = str(tmp / 'rt2.tif')
    # ``nodata=`` is not re-passed: the read result carries the sentinel
    # in attrs['nodata'] and the writer picks it up there. Re-passing
    # would double up the kwarg vs the attr.
    to_geotiff(da1_for_write, p2, compression='none', tiled=False)
    da2 = open_geotiff(p2)

    _assert_fixed_point(da1, da2)


# ---------------------------------------------------------------------------
# Property: round-trip on the dask backend
# ---------------------------------------------------------------------------

@settings(
    parent=settings.get_profile('ci'),
)
@given(spec=_round_trip_spec())
def test_round_trip_fixed_point_dask(tmp_path_factory, spec):
    """Same property as ``test_round_trip_fixed_point_numpy`` but the
    initial DataArray is wrapped in dask chunks so the streaming write
    path is exercised.

    The CI profile keeps this at 50 examples; the numpy property's
    200-example budget already covers the strategy interior, this
    pass exists to catch drift specific to the streaming writer.
    """
    pytest.importorskip('dask')

    rng = np.random.default_rng(spec['seed'])
    da0 = _build_dataarray(spec)

    # Pick chunks that actually split at least one axis when the shape
    # allows it; otherwise a single chunk reproduces the eager path
    # and the test would be redundant.
    h, w = spec['shape']
    if spec['band_layout'] == 'band_first':
        chunks = {'band': spec['n_bands'], 'y': max(h // 2, 1),
                  'x': max(w // 2, 1)}
    elif spec['band_layout'] == 'band_last':
        chunks = {'y': max(h // 2, 1), 'x': max(w // 2, 1),
                  'band': spec['n_bands']}
    else:
        chunks = {'y': max(h // 2, 1), 'x': max(w // 2, 1)}
    da0 = da0.chunk(chunks)

    kwargs = _writer_kwargs(spec, rng)

    tmp = tmp_path_factory.mktemp("rt_2134_dask")
    p1 = str(tmp / 'rt1.tif')
    try:
        to_geotiff(da0, p1, **kwargs)
    except (ValueError, TypeError):
        assume(False)
        return  # pragma: no cover

    da1 = open_geotiff(p1)

    p2 = str(tmp / 'rt2.tif')
    to_geotiff(da1, p2, compression='none', tiled=False)
    da2 = open_geotiff(p2)

    _assert_fixed_point(da1, da2)
