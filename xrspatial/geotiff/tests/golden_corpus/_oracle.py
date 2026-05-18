"""Oracle harness for the geotiff golden corpus (issue #1930, Phase 1.2).

``compare_to_oracle(fixture_path, candidate_da)`` reads ``fixture_path`` with
rasterio (the reference implementation) and asserts that an xrspatial-produced
``xarray.DataArray`` agrees on every observable property: pixel values,
dtype, transform, CRS, nodata sentinel, and a small set of canonical attrs.

The Phase 3 backend cells (numpy, dask+numpy, cupy, dask+cupy, HTTP, VRT)
each call this single function with the DataArray they produced; the oracle
encapsulates "what parity means" so every backend agrees on it.

Scope notes:

* The full canonical-attrs contract is tracked in issue #1984 and is not
  settled yet. Until it lands, the oracle asserts only the obvious subset
  (``crs``, ``transform``, ``nodata``, ``dtype``) and leaves a hook
  (``_assert_canonical_attrs``) that later PRs can fill in. See the TODO
  in that function.
* ``lossy=True`` skips bit-exact pixel comparison (for JPEG cells in
  Phase 2) and instead verifies only shape, dtype, transform, and CRS.
* The oracle does not import the corpus manifest from Phase 1 PR 1; it
  takes a raw filesystem path so the two PRs stay decoupled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Rasterio lazy import
# ---------------------------------------------------------------------------

def _require_rasterio():
    """Import rasterio lazily so the module is importable in environments
    without it; tests using the oracle call ``pytest.importorskip('rasterio')``
    at module load time.
    """
    try:
        import rasterio  # noqa: F401
        import rasterio.crs  # noqa: F401
        import rasterio.transform  # noqa: F401
        return rasterio
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            'rasterio is required for the golden-corpus oracle. '
            'Install it with `pip install rasterio`.'
        ) from exc


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

# xrspatial stores its transform under ``attrs['transform']`` as
# ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)`` -- the same
# 6-tuple order as ``rasterio.Affine(a, b, c, d, e, f)``. See
# ``xrspatial.geotiff._coords.transform_tuple_from_pixel_geometry``.
_TRANSFORM_ATOL = 1e-9


def _candidate_transform(candidate_da: xr.DataArray):
    """Derive a rasterio ``Affine`` from the candidate.

    Preference order:

    1. ``attrs['transform']`` -- the canonical xrspatial 6-tuple. Byte-exact
       round-trips depend on this attr (see ``_coords.transform_tuple``).
    2. ``rio.transform()`` -- if rioxarray decorated the DataArray.
    3. Derived from the y/x coords, treating them as pixel centres.

    Returns ``None`` when no transform can be recovered (a no-georef raster).
    """
    rasterio = _require_rasterio()
    Affine = rasterio.transform.Affine

    t = candidate_da.attrs.get('transform')
    if t is not None:
        if len(t) != 6:
            raise AssertionError(
                f"candidate attrs['transform'] must be a 6-tuple, got {t!r}")
        return Affine(*[float(v) for v in t])

    rio = getattr(candidate_da, 'rio', None)
    if rio is not None:
        try:
            return rio.transform()
        except Exception:
            pass

    if 'y' in candidate_da.coords and 'x' in candidate_da.coords:
        y = np.asarray(candidate_da.coords['y'].values, dtype=float)
        x = np.asarray(candidate_da.coords['x'].values, dtype=float)
        if y.size >= 2 and x.size >= 2:
            pw = float(x[1] - x[0])
            ph = float(y[1] - y[0])
            ox = float(x[0]) - 0.5 * pw
            oy = float(y[0]) - 0.5 * ph
            return Affine(pw, 0.0, ox, 0.0, ph, oy)
    return None


def _affine_close(a, b, *, atol: float = _TRANSFORM_ATOL) -> bool:
    return all(
        abs(float(x) - float(y)) <= atol
        for x, y in zip(tuple(a)[:6], tuple(b)[:6])
    )


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def _candidate_crs(candidate_da: xr.DataArray):
    """Best-effort CRS extraction from a candidate DataArray.

    Tries, in order:

    1. ``attrs['crs']`` -- xrspatial stores an EPSG int here when known.
    2. ``attrs['crs_wkt']`` -- WKT fallback.
    3. ``rio.crs`` -- rioxarray decoration (mostly for sibling tools).

    The return value is whatever ``rasterio.crs.CRS.from_user_input`` accepts,
    or ``None`` when no CRS is recorded.
    """
    rasterio = _require_rasterio()
    attrs = candidate_da.attrs

    crs_val = attrs.get('crs')
    if crs_val is not None:
        return rasterio.crs.CRS.from_user_input(crs_val)

    wkt = attrs.get('crs_wkt')
    if wkt:
        return rasterio.crs.CRS.from_user_input(wkt)

    rio = getattr(candidate_da, 'rio', None)
    if rio is not None:
        crs = getattr(rio, 'crs', None)
        if crs is not None:
            return rasterio.crs.CRS.from_user_input(crs)
    return None


def _crs_equal(ref, cand) -> bool:
    """EPSG-aware CRS equality with a PROJ-dict fallback.

    rasterio's ``CRS.__eq__`` compares WKT structurally, which makes
    EPSG-equivalent WKTs (one from PROJ, one from libgeotiff) compare
    unequal even when they describe the same coordinate system. Fall back
    to EPSG-code comparison when both sides resolve to an EPSG code.

    Citation-only CRSes (a user-supplied name with no AUTHORITY tag, e.g.
    the Phase 2 PR 8 ``crs_citation_only`` fixture) cannot be compared by
    EPSG code because neither side has one. PROJ's ``to_dict()`` projects
    them onto a small set of canonical fields (proj kind, ellipsoid
    radius, units), which is stable across the libgeotiff round-trip
    that mutates WKT axis order and adds AUTHORITY["EPSG","9122"] to the
    UNIT block. Use that as a last resort, but only when both sides
    produce a non-empty dict (``CRS.to_dict()`` returns ``{}`` for
    LOCAL_CS-style WKTs, which would otherwise let any two unrecognised
    CRSes compare equal).

    Known limit: ``CRS.to_dict()`` drops the GEOGCS / PROJCS name, so two
    citation-only CRSes with the same shape but different names compare
    equal here. The current corpus only has one citation fixture so this
    is theoretical; if it becomes load-bearing, switch to a name-aware
    comparison via ``to_dict(projjson=True)`` (which preserves the name
    but mutates axis order on round-trip and would need its own
    normaliser).
    """
    if ref is None and cand is None:
        return True
    if ref is None or cand is None:
        return False
    if ref == cand:
        return True
    try:
        ref_epsg = ref.to_epsg()
        cand_epsg = cand.to_epsg()
    except Exception:
        ref_epsg = None
        cand_epsg = None
    if ref_epsg is not None and cand_epsg is not None:
        return ref_epsg == cand_epsg
    if ref_epsg is None and cand_epsg is None:
        try:
            ref_dict = ref.to_dict()
            cand_dict = cand.to_dict()
        except Exception:
            return False
        # Empty dict means "PROJ has no canonical form for this CRS"
        # (e.g. LOCAL_CS). Refuse to declare equality in that case
        # rather than match any other empty-dict CRS.
        if not ref_dict or not cand_dict:
            return False
        return ref_dict == cand_dict
    return False


# ---------------------------------------------------------------------------
# Pixel / nodata helpers
# ---------------------------------------------------------------------------

def _candidate_pixels(candidate_da: xr.DataArray) -> np.ndarray:
    """Return a 2-D or 3-D numpy view regardless of backend (dask/cupy)."""
    raw = candidate_da.data
    if hasattr(raw, 'compute'):
        raw = raw.compute()
    if hasattr(raw, 'get'):
        raw = raw.get()
    return np.asarray(raw)


def _nodata_equal(ref, cand) -> bool:
    """NaN-aware nodata comparison.

    Treats two NaN sentinels as equal (``float('nan') != float('nan')`` in
    Python's default semantics, but they are the same nodata for our
    purposes). ``None`` on both sides also counts as equal.
    """
    if ref is None and cand is None:
        return True
    if ref is None or cand is None:
        return False
    try:
        ref_f = float(ref)
        cand_f = float(cand)
    except (TypeError, ValueError):
        return ref == cand
    if np.isnan(ref_f) and np.isnan(cand_f):
        return True
    return ref_f == cand_f


def _pixels_equal(ref: np.ndarray, cand: np.ndarray) -> bool:
    """Bit-exact pixel comparison, NaN-aware for floats."""
    if ref.shape != cand.shape:
        return False
    if ref.dtype.kind == 'f' or cand.dtype.kind == 'f':
        return np.array_equal(ref, cand, equal_nan=True)
    return np.array_equal(ref, cand)


# ---------------------------------------------------------------------------
# Masked-nodata normalisation (issue #1988)
# ---------------------------------------------------------------------------

def _has_masked_nodata(candidate_da: xr.DataArray) -> bool:
    """True when the candidate reports xrspatial's masked-nodata contract.

    The contract (issue #1988): xrspatial reads an integer GeoTIFF whose
    nodata tag carries an integer sentinel, masks the sentinel-equal
    pixels to NaN, and upcasts the array to float so NaN can live in
    it. The reader stamps ``attrs['masked_nodata'] = True`` to record
    that the masking happened; ``attrs['nodata']`` still carries the
    original integer sentinel so a write round-trip can put the tag
    back.
    """
    return bool(candidate_da.attrs.get('masked_nodata', False))


def _normalise_for_masked_nodata(
    ref_pixels: np.ndarray,
    ref_dtype: np.dtype,
    ref_nodata,
    candidate_da: xr.DataArray,
) -> tuple[np.ndarray, np.dtype]:
    """Apply xrspatial's masked-nodata contract to the rasterio reference.

    When the candidate reports ``attrs['masked_nodata']=True``, the
    reference is rewritten so it matches what the candidate produced:
    cast to the candidate's float dtype, then any pixel equal to the
    integer sentinel ``ref_nodata`` becomes ``NaN``. The ``ref_dtype``
    returned alongside is the candidate's dtype, so the downstream
    ``_assert_dtype`` check passes by design.

    If the candidate does not report masked nodata, or if ``ref_nodata``
    is not a finite integer sentinel, the inputs pass through
    unchanged. That keeps every existing fixture's behaviour intact.
    """
    if not _has_masked_nodata(candidate_da):
        return ref_pixels, ref_dtype
    cand_dtype = np.dtype(candidate_da.dtype)
    if cand_dtype.kind != 'f':
        # The masked_nodata contract requires a float dtype on the
        # candidate so NaN can live in the array. If it is anything
        # else, fall through to the normal strict comparison and let
        # it fail with a clear message.
        return ref_pixels, ref_dtype
    # The sentinel must be a finite real number that fits in the
    # source integer dtype. A NaN / Inf sentinel cannot match any
    # integer pixel and would be the wrong contract.
    try:
        nd_float = float(ref_nodata)
    except (TypeError, ValueError):
        return ref_pixels, ref_dtype
    if not np.isfinite(nd_float):
        return ref_pixels, ref_dtype
    new_ref = ref_pixels.astype(cand_dtype, copy=True)
    new_ref[ref_pixels == ref_pixels.dtype.type(nd_float)] = np.nan
    return new_ref, cand_dtype


# ---------------------------------------------------------------------------
# Canonical-attrs hook (issue #1984)
# ---------------------------------------------------------------------------

# Attrs the oracle currently asserts. Kept intentionally small until
# issue #1984 (canonical attrs contract) settles which keys are canonical,
# which are aliases, and which are pass-through metadata that may legally
# diverge between rasterio and xrspatial reads.
_CANONICAL_ATTR_KEYS_PROVISIONAL: tuple[str, ...] = (
    'crs',
    'transform',
    'nodata',
    'dtype',
)


def _assert_canonical_attrs(
    _ref_attrs: dict[str, Any],
    _candidate_da: xr.DataArray,
) -> None:
    """Assert the canonical-attrs subset.

    No-op today on purpose. The four sibling helpers
    (``_assert_dtype`` / ``_assert_transform`` / ``_assert_crs`` /
    ``_assert_nodata``) cover the provisional contract; this stub exists
    so test code references stay stable when issue #1984 lands.

    TODO(#1984): Expand to the full canonical-attrs contract once it
    settles. The provisional key set is in
    ``_CANONICAL_ATTR_KEYS_PROVISIONAL``. Likely additions when #1984
    lands: ``raster_type`` (PixelIsArea vs PixelIsPoint), resolution
    keys, and a canonicalised view of GDAL metadata. Pass-through tags
    (``gdal_metadata``, ``extra_tags``) stay out of scope.

    Parameters are prefixed with ``_`` because they are deliberately
    unused today; rename them when this function gains a body.
    """


# ---------------------------------------------------------------------------
# Per-property assertions
# ---------------------------------------------------------------------------

def _assert_dtype(ref_dtype: np.dtype, candidate_da: xr.DataArray) -> None:
    cand_dtype = np.dtype(candidate_da.dtype)
    ref_dtype = np.dtype(ref_dtype)
    assert cand_dtype == ref_dtype, (
        f'dtype mismatch: rasterio={ref_dtype}, xrspatial={cand_dtype}')


def _assert_transform(
    ref_transform,
    candidate_da: xr.DataArray,
    *,
    ref_has_georef: bool,
) -> None:
    """Compare transforms.

    ``ref_has_georef`` is True when the rasterio source carried real
    GeoTIFF tags (i.e. a CRS *or* a non-identity transform). When it is
    False the file is bare -- rasterio returns ``Affine.identity()`` for
    such files regardless of pixel size -- and xrspatial may legitimately
    drop the transform attr (#1710). Identity-equal transforms alone are
    NOT enough to declare "no georef": a real raster written at origin
    (0, 0) with 1.0 pixel size also matches identity.
    """
    cand_t = _candidate_transform(candidate_da)
    if not ref_has_georef:
        # Bare file. The candidate may legitimately:
        # * carry no transform attr at all (xrspatial's no-georef path,
        #   #1710),
        # * carry an identity transform attr,
        # * carry a transform derived from integer-pixel-center coords
        #   (origin -0.5, pixel size 1.0 -- what xrspatial's
        #   ``coords_from_pixel_geometry`` emits with ``has_georef=False``).
        # We require only that the candidate also disclaims a real georef
        # via one of those routes; we do not compare the transform tuple
        # itself, because rasterio synthesises ``Affine.identity()`` here
        # while xrspatial may produce any of the three forms above.
        if cand_t is None:
            return
        if candidate_da.attrs.get('transform') is None:
            # Came from coord derivation only -- treat as no transform.
            return
        if _is_default_transform(cand_t):
            return
        raise AssertionError(
            'fixture has no georef but candidate carries a non-identity '
            f'transform attr: {tuple(cand_t)}')
    assert cand_t is not None, (
        'rasterio transform is set but candidate has no transform '
        f'attr/coords. ref={tuple(ref_transform)}')
    assert _affine_close(ref_transform, cand_t), (
        f'transform mismatch:\n  rasterio:  {tuple(ref_transform)[:6]}\n'
        f'  xrspatial: {tuple(cand_t)[:6]}')


def _is_default_transform(t) -> bool:
    rasterio = _require_rasterio()
    return tuple(t)[:6] == tuple(rasterio.transform.Affine.identity())[:6]


def _ref_has_georef(src) -> bool:
    """Has the rasterio source got any georef at all?

    True if it has a CRS *or* a non-identity transform. rasterio returns
    ``Affine.identity()`` for files with no georef, but a fully-real
    raster can also have an identity transform (origin 0, 1.0 pixels),
    so identity alone is not a "no georef" signal -- pair it with the
    absence of a CRS.
    """
    has_crs = src.crs is not None
    has_non_id_transform = not _is_default_transform(src.transform)
    return has_crs or has_non_id_transform


def _assert_crs(ref_crs, candidate_da: xr.DataArray) -> None:
    cand_crs = _candidate_crs(candidate_da)
    assert _crs_equal(ref_crs, cand_crs), (
        f'CRS mismatch:\n  rasterio:  {ref_crs!r}\n  xrspatial: {cand_crs!r}')


def _assert_nodata(ref_nodata, candidate_da: xr.DataArray) -> None:
    cand_nodata = candidate_da.attrs.get('nodata')
    assert _nodata_equal(ref_nodata, cand_nodata), (
        f'nodata mismatch: rasterio={ref_nodata!r}, '
        f'xrspatial={cand_nodata!r}')


def _assert_pixels(ref_pixels: np.ndarray, candidate_da: xr.DataArray) -> None:
    cand_pixels = _candidate_pixels(candidate_da)
    # Single-band rasterio reads come back as (1, H, W); xrspatial may drop
    # the band axis. Squeeze a leading length-1 axis on either side so the
    # comparison is band-agnostic for the single-band case.
    if ref_pixels.ndim == 3 and ref_pixels.shape[0] == 1 and cand_pixels.ndim == 2:
        ref_pixels = ref_pixels[0]
    elif cand_pixels.ndim == 3 and cand_pixels.shape[0] == 1 and ref_pixels.ndim == 2:
        cand_pixels = cand_pixels[0]
    assert _pixels_equal(ref_pixels, cand_pixels), (
        'pixel arrays differ (bit-exact / NaN-aware comparison failed). '
        f'ref shape={ref_pixels.shape} dtype={ref_pixels.dtype}, '
        f'cand shape={cand_pixels.shape} dtype={cand_pixels.dtype}')


def _assert_shape_only(ref_pixels: np.ndarray, candidate_da: xr.DataArray) -> None:
    cand_pixels = _candidate_pixels(candidate_da)
    if ref_pixels.ndim == 3 and ref_pixels.shape[0] == 1 and cand_pixels.ndim == 2:
        ref_shape = ref_pixels.shape[1:]
    elif cand_pixels.ndim == 3 and cand_pixels.shape[0] == 1 and ref_pixels.ndim == 2:
        ref_shape = ref_pixels.shape
        cand_pixels = cand_pixels[0]
    else:
        ref_shape = ref_pixels.shape
    assert ref_shape == cand_pixels.shape, (
        f'shape mismatch: rasterio={ref_shape}, xrspatial={cand_pixels.shape}')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_to_oracle(
    fixture_path: str | Path,
    candidate_da: xr.DataArray,
    *,
    lossy: bool = False,
) -> None:
    """Assert that ``candidate_da`` matches the rasterio read of ``fixture_path``.

    Parameters
    ----------
    fixture_path
        Path to a TIFF on disk. The oracle does not consult the corpus
        manifest; callers (Phase 3 test cells) pass a raw path.
    candidate_da
        The xarray DataArray produced by an xrspatial read backend.
    lossy
        When ``True``, skip bit-exact pixel comparison and assert only
        shape, dtype, transform, and CRS. Use this for JPEG cells where
        the codec is intrinsically lossy (Phase 2 PR 5).

    Raises
    ------
    AssertionError
        On the first property that disagrees. The message identifies the
        property and prints both sides.

    Notes
    -----
    The canonical-attrs contract is tracked in issue #1984. Until it
    settles, only the four obvious keys (crs/transform/nodata/dtype) are
    checked; ``_assert_canonical_attrs`` is the hook later PRs will fill
    in. Pass-through GeoTIFF metadata (gdal_metadata, extra_tags, etc.)
    is intentionally not asserted here.
    """
    rasterio = _require_rasterio()
    fixture_path = Path(fixture_path)
    if not fixture_path.exists():
        raise FileNotFoundError(f'oracle fixture not found: {fixture_path}')

    with rasterio.open(fixture_path) as src:
        ref_pixels = src.read()  # shape (bands, H, W)
        ref_dtype = np.dtype(src.dtypes[0]) if src.dtypes else ref_pixels.dtype
        ref_transform = src.transform
        ref_crs = src.crs
        ref_nodata = src.nodata
        ref_has_georef = _ref_has_georef(src)
        ref_attrs = {
            'crs': ref_crs,
            'transform': ref_transform,
            'nodata': ref_nodata,
            'dtype': ref_dtype,
        }

    # When the candidate reports the masked-nodata contract (#1988),
    # rewrite the rasterio reference to match: cast to the candidate's
    # float dtype and replace sentinel-equal pixels with NaN. Then the
    # dtype + pixel assertions run on directly comparable arrays.
    ref_pixels, ref_dtype = _normalise_for_masked_nodata(
        ref_pixels, ref_dtype, ref_nodata, candidate_da
    )

    _assert_dtype(ref_dtype, candidate_da)
    _assert_transform(ref_transform, candidate_da, ref_has_georef=ref_has_georef)
    _assert_crs(ref_crs, candidate_da)
    _assert_nodata(ref_nodata, candidate_da)
    if lossy:
        _assert_shape_only(ref_pixels, candidate_da)
    else:
        _assert_pixels(ref_pixels, candidate_da)
    _assert_canonical_attrs(ref_attrs, candidate_da)


__all__ = ['compare_to_oracle']
