"""Input validators shared by the geotiff entry points.

Pure leaves over numpy dtypes and Python primitives. Called from
``to_geotiff``, ``read_geotiff_dask``, ``read_geotiff_gpu``,
``read_vrt``, and ``write_geotiff_gpu`` so the rejection rules
(non-positive chunks, lossy float-to-int casts, ambiguous 3D dim
layouts, tile-size multiples of 16, etc.) stay in lockstep across
every backend.

Extracted in step 4 of issue #1813.

Ambiguous-metadata hooks (issue #1987)
--------------------------------------
``validate_read_metadata`` and ``validate_write_metadata`` are
plug-points for the per-case checks listed in #1987 (unparseable CRS,
rotated transforms, non-uniform coords, mixed band metadata, conflicting
crs/crs_wkt, conflicting nodata aliases). PR 0 lands the hook
signatures and a registry; each follow-up PR registers its check.
The hooks are no-ops until at least one check is registered, so
behaviour does not change until a per-case PR opts in.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ._coords import _BAND_DIM_NAMES
from ._errors import (
    ConflictingCRSError,
    ConflictingNodataError,
    MixedBandMetadataError,
    NonUniformCoordsError,
    RotatedTransformError,
    UnparseableCRSError,
)
from ._runtime import _TIME_DIM_NAMES, _X_DIM_NAMES, _Y_DIM_NAMES


def _is_temporal_dim_name(name) -> bool:
    """Return True if ``name`` is a known temporal dim alias.

    Compared case-insensitively against ``_TIME_DIM_NAMES`` so that
    CF-style ``'TIME'`` / ``'Time'`` reach the friendly temporal error
    in the 3D writer validator instead of slipping through the
    ``(y, x, *)`` band-position fallback (#1972).
    """
    return isinstance(name, str) and name.lower() in _TIME_DIM_NAMES


def _validate_3d_writer_dims(dims) -> None:
    """Reject ambiguous 3D writer inputs (issue #1812).

    The writer interprets a 3D DataArray as either ``(band, y, x)`` or
    ``(y, x, band)``. ``data.dims[0] in _BAND_DIM_NAMES`` decides which
    branch fires the ``moveaxis``. Anything else (e.g. ``('time', 'y', 'x')``)
    used to fall through silently: the writer kept the leading axis as
    the spatial ``y`` axis and the result was a TIFF with the leading
    axis values laid out along ``y`` (silent data corruption -- on
    read-back the array round-tripped with a swapped shape).

    Refuse the ambiguous case at the entry point. The message tells the
    caller exactly how to fix the input (rename to one of
    ``_BAND_DIM_NAMES`` or transpose to ``(y, x, band)``).
    """
    if len(dims) != 3:
        return
    d0, d1, d2 = dims
    band_layout = (d0 in _BAND_DIM_NAMES
                   and d1 in _Y_DIM_NAMES
                   and d2 in _X_DIM_NAMES)
    yxb_layout = (d0 in _Y_DIM_NAMES
                  and d1 in _X_DIM_NAMES
                  and d2 in _BAND_DIM_NAMES)
    if band_layout or yxb_layout:
        return
    # Bare (y, x, *) where the third dim is unnamed but spatial -- the
    # writer's old behaviour treats the non-spatial axis as bands.
    # Accept that only when the unknown dim is in the band position
    # (last), which matches how raw numpy callers typically build a
    # band-last array. Refuse known *temporal* dim names so a
    # ``(y, x, time)`` stack is rejected with a clear error instead of
    # silently being written as a 3-band TIFF (issue #1972). The
    # mirror case ``(time, y, x)`` was already caught -- this closes
    # the asymmetry.
    if d0 in _Y_DIM_NAMES and d1 in _X_DIM_NAMES:
        if _is_temporal_dim_name(d2):
            raise ValueError(
                f"3D writer input has temporal trailing dim {d2!r} in dims "
                f"{dims!r}. The writer would otherwise treat the time axis "
                f"as bands and silently write a multiband TIFF. Select a "
                f"single time slice (e.g. ``data.isel({d2}=0)``), reduce "
                f"with a stat (``data.mean({d2!r})``), or rename to one of "
                f"{_BAND_DIM_NAMES} if you really intend the temporal "
                f"axis to round-trip as TIFF bands (issue #1972)."
            )
        return
    # Symmetrise the friendly temporal message for the leading-dim case
    # ``(time, y, x)``. The generic ``ambiguous dims`` error below
    # already rejects this layout, but the temporal-specific message
    # tells the caller exactly how to fix it (#1972).
    if _is_temporal_dim_name(d0) and d1 in _Y_DIM_NAMES and d2 in _X_DIM_NAMES:
        raise ValueError(
            f"3D writer input has temporal leading dim {d0!r} in dims "
            f"{dims!r}. The writer would otherwise treat the time axis "
            f"as bands and silently write a multiband TIFF. Select a "
            f"single time slice (e.g. ``data.isel({d0}=0)``), reduce "
            f"with a stat (``data.mean({d0!r})``), or rename to one of "
            f"{_BAND_DIM_NAMES} if you really intend the temporal "
            f"axis to round-trip as TIFF bands (issue #1972)."
        )
    raise ValueError(
        f"3D writer input has ambiguous dims {dims!r}. Expected "
        f"(band, y, x) or (y, x, band); accepted band-dim aliases are "
        f"{_BAND_DIM_NAMES} and spatial aliases are y={_Y_DIM_NAMES} / "
        f"x={_X_DIM_NAMES}. Rename the non-spatial dim to 'band' or "
        f"transpose the array so spatial dims come first (e.g. "
        f"``da.transpose('y', 'x', {dims[0]!r})``). The writer cannot "
        f"infer which axis is the band axis from arbitrary dim names "
        f"and would otherwise silently treat the leading axis as the "
        f"spatial y axis (issue #1812)."
    )


def _validate_dtype_cast(source_dtype, target_dtype):
    """Validate that casting source_dtype to target_dtype is allowed.

    Raises ValueError for float-to-int casts (lossy in a way users
    often don't intend).  All other casts are permitted -- the user
    asked for them explicitly.
    """
    src = np.dtype(source_dtype)
    tgt = np.dtype(target_dtype)
    if src.kind == 'f' and tgt.kind in ('u', 'i'):
        raise ValueError(
            f"Cannot cast float ({src}) to int ({tgt}). "
            f"This loses fractional data and is usually unintentional. "
            f"Cast explicitly after reading if you really want this.")


def _validate_tile_size(tile_size) -> None:
    """Validate ``tile_size`` for the tiled GeoTIFF writers.

    Shared by ``to_geotiff`` (when ``tiled=True``) and
    ``write_geotiff_gpu`` (always tiled) so the accepted types, the
    non-positive rejection, and the multiple-of-16 hint stay in lockstep.
    The tiled writer computes the tile grid as
    ``math.ceil(width / tile_size)``; ``tile_size=0`` hits
    ``ZeroDivisionError`` deep inside the writer, and negative values
    produce a nonsensical tile grid. The TIFF 6 spec also requires
    ``TileWidth`` and ``TileLength`` to be positive multiples of 16
    for broad interoperability with libtiff / GDAL strict readers; a
    value like 17 would otherwise round-trip through the in-repo
    reader but be rejected elsewhere.
    """
    if not isinstance(tile_size, (int, np.integer)) or isinstance(
            tile_size, bool):
        raise ValueError(
            f"tile_size must be a positive int, got "
            f"{tile_size!r} (type {type(tile_size).__name__}).")
    if tile_size <= 0:
        raise ValueError(
            f"tile_size must be a positive int, got tile_size={tile_size}.")
    if tile_size % 16 != 0:
        lower = (int(tile_size) // 16) * 16
        upper = lower + 16
        # ``lower`` is 0 for tile_size < 16; suppress it from the hint
        # because 0 is not a valid tile size on its own.
        if lower <= 0:
            hint = f"try tile_size={upper}"
        else:
            hint = f"try tile_size={lower} or tile_size={upper}"
        raise ValueError(
            f"tile_size must be a positive multiple of 16 (TIFF 6 "
            f"spec requirement for TileWidth/TileLength), got "
            f"tile_size={tile_size}; {hint}.")


def _validate_chunks_arg(chunks, *, allow_none=False):
    """Validate the ``chunks`` kwarg shared across the dask read entry points.

    Centralises the rejection rule that ``read_geotiff_dask`` already
    runs so ``read_geotiff_gpu`` and ``read_vrt`` can share the same
    error format. With ``allow_none=True`` a ``None`` value passes
    through unchanged (used by entry points whose default is
    ``chunks=None``, e.g. ``read_geotiff_gpu`` and ``read_vrt``).
    With ``allow_none=False`` (default, matches ``read_geotiff_dask``)
    a ``None`` is rejected with the same ``ValueError`` format as any
    other non-int / non-tuple value, so callers see a clear
    parameter-named error instead of a downstream ``TypeError`` from
    the chunk-unpacking math.
    Otherwise ``chunks`` must be a positive int or a 2-tuple of
    positive ints. Booleans are rejected because ``True``/``False``
    are int subclasses that would otherwise sneak through the integer
    check. Returns the coerced int when given an ``np.integer`` scalar
    so downstream ``isinstance(chunks, int)`` checks stay accurate.

    Mirrors the chunks-validation #1752 added to ``read_geotiff_dask``;
    extends it to the GPU read and VRT read entry points per #1776.
    """
    if chunks is None:
        if allow_none:
            return chunks
        raise ValueError(
            f"chunks must be a positive int or (row, col) tuple of "
            f"positive ints, got chunks=None.")
    if (isinstance(chunks, (int, np.integer))
            and not isinstance(chunks, bool)):
        if chunks <= 0:
            raise ValueError(
                f"chunks must be a positive int or (row, col) tuple of "
                f"positive ints, got chunks={chunks}.")
        return int(chunks)
    if isinstance(chunks, tuple):
        if len(chunks) != 2:
            raise ValueError(
                f"chunks tuple must have length 2 (row, col), got "
                f"chunks={chunks!r} with length {len(chunks)}.")
        for _v in chunks:
            if (not isinstance(_v, (int, np.integer))
                    or isinstance(_v, bool)
                    or _v <= 0):
                raise ValueError(
                    f"chunks must be a positive int or (row, col) tuple "
                    f"of positive ints, got chunks={chunks!r}.")
        return chunks
    raise ValueError(
        f"chunks must be a positive int or (row, col) tuple of "
        f"positive ints, got chunks={chunks!r} "
        f"(type {type(chunks).__name__}).")


def _validate_tile_size_arg(tile_size):
    """Validate the ``tile_size`` kwarg for the tiled writer entry points.

    Wrapper kept for backwards internal compatibility; delegates to
    ``_validate_tile_size`` so to_geotiff/write_geotiff_gpu share one
    validation path (positive int + multiple-of-16 for tiled output).
    """
    _validate_tile_size(tile_size)


def _validate_predictor_sample_format(predictor, sample_format) -> None:
    """Reject ``Predictor=3`` paired with a non-float ``SampleFormat`` (issue #1933).

    TIFF Technical Note 3 defines the floating-point predictor for IEEE
    float samples only. A reader-side input file (malformed, hand-crafted,
    or adversarial) that claims ``Predictor=3`` with an integer
    ``SampleFormat`` (1=uint, 2=int) used to be accepted silently: the
    byte-swizzle unshuffle ran on integer bytes and produced garbage
    pixel values that look like valid integers, with no warning.

    The writer side already enforces this contract in
    ``_writer._resolve_predictor`` (raises ``ValueError`` on non-float
    dtypes), so this validator gives the reader symmetric behaviour.

    Parameters
    ----------
    predictor : int or tuple
        The IFD ``Predictor`` tag value (1=none, 2=horizontal, 3=float).
        Accepts a single-element tuple (the resolved value of a malformed
        ``count > 1`` tag) and normalizes to its first element; the TIFF
        spec defines ``Predictor`` as a single SHORT.
    sample_format : int
        The IFD ``SampleFormat`` tag value (1=uint, 2=int, 3=float,
        4=undefined).

    Raises
    ------
    ValueError
        If ``predictor == 3`` and ``sample_format != 3``.
    """
    # IFD.predictor delegates to IFD.get_value, which can return a tuple
    # for a malformed Predictor tag with count > 1. tuple == 3 is always
    # False, so a tuple-valued predictor would bypass the guard. Normalize
    # to int first so the (3, non-3) case still fires.
    if isinstance(predictor, tuple):
        predictor = predictor[0] if predictor else 1
    # Only the float-predictor case is asymmetric; predictor=1 (none) and
    # predictor=2 (horizontal) are sample-format-agnostic by design.
    if predictor == 3 and sample_format != 3:
        raise ValueError(
            f"Predictor=3 (floating-point) requires SampleFormat=3 "
            f"(IEEE float), got SampleFormat={sample_format}. The TIFF "
            f"file is malformed: the floating-point horizontal predictor "
            f"(TIFF Technical Note 3) is only defined for float samples. "
            f"Decoding integer data through it would produce garbage. "
            f"Re-encode the file with a matching predictor/sample-format "
            f"pair, e.g. `gdal_translate -co PREDICTOR=2` for integers or "
            f"`-co PREDICTOR=1` to drop the predictor."
        )


def _validate_nodata_arg(nodata) -> None:
    """Reject non-numeric ``nodata=`` at the writer entry point (#1973).

    ``None`` (no sentinel) passes through. ``bool`` is rejected with
    ``TypeError`` so all three writer entry points (eager, GPU, VRT)
    refuse ``nodata=True`` / ``nodata=False`` the same way the eager
    path already does for issue #1911 -- ``float(True) == 1.0`` would
    otherwise slip a bool past the numeric branch on the GPU/VRT paths
    that do not have their own bool guard. Anything else is run
    through ``float()``: success means the writer's downstream
    ``np.isnan(nodata)`` and integer-cast paths will not blow up.
    Failure raises ``ValueError`` with the offending repr, so users
    see ``nodata='missing'`` flagged at the boundary instead of an
    opaque ``ufunc 'isnan' not supported`` TypeError from inside the
    writer.
    """
    if nodata is None:
        return
    if isinstance(nodata, (bool, np.bool_)):
        raise TypeError(
            f"nodata must be numeric (int or float), got {nodata!r}")
    try:
        float(nodata)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"nodata must be numeric or None, got {nodata!r} "
            f"(type {type(nodata).__name__}). The writer compares it "
            f"against pixel values via ``np.isnan`` and casts it to "
            f"the array dtype; a non-numeric value would otherwise "
            f"crash inside NumPy with a ufunc TypeError."
        ) from e


# ---------------------------------------------------------------------------
# Ambiguous-metadata hooks (issue #1987 PR 0)
#
# Each per-case PR (#1987 PRs 2-7) registers a check via
# ``register_read_metadata_check`` / ``register_write_metadata_check``.
# The hooks below iterate the registered checks in registration order.
# A check raises one of the ``_errors.GeoTIFFAmbiguousMetadataError``
# subclasses to refuse the input; returning normally lets the call
# continue.
#
# The registry is process-global and additive. Tests that need to
# unregister a check should use ``unregister_*`` rather than mutating
# the list in place so the surrounding helpers stay typed.
# ---------------------------------------------------------------------------

_ReadCheck = Callable[[Mapping[str, Any]], None]
_WriteCheck = Callable[[Mapping[str, Any]], None]

_READ_METADATA_CHECKS: list[_ReadCheck] = []
_WRITE_METADATA_CHECKS: list[_WriteCheck] = []


def register_read_metadata_check(check: _ReadCheck) -> _ReadCheck:
    """Register a read-side ambiguous-metadata check (issue #1987).

    Returns ``check`` so the call can be used as a decorator. Idempotent:
    re-registering the same callable is a no-op.
    """
    if check not in _READ_METADATA_CHECKS:
        _READ_METADATA_CHECKS.append(check)
    return check


def register_write_metadata_check(check: _WriteCheck) -> _WriteCheck:
    """Register a write-side ambiguous-metadata check (issue #1987)."""
    if check not in _WRITE_METADATA_CHECKS:
        _WRITE_METADATA_CHECKS.append(check)
    return check


def unregister_read_metadata_check(check: _ReadCheck) -> None:
    """Remove a previously-registered read-side check.

    Tolerant of ``check`` not being registered so tests can call this
    in teardown without guarding on ``in``.
    """
    try:
        _READ_METADATA_CHECKS.remove(check)
    except ValueError:
        pass


def unregister_write_metadata_check(check: _WriteCheck) -> None:
    """Remove a previously-registered write-side check."""
    try:
        _WRITE_METADATA_CHECKS.remove(check)
    except ValueError:
        pass


def _registered_read_metadata_checks() -> Iterable[_ReadCheck]:
    """Snapshot of registered read-side checks for testing/introspection."""
    return tuple(_READ_METADATA_CHECKS)


def _registered_write_metadata_checks() -> Iterable[_WriteCheck]:
    """Snapshot of registered write-side checks for testing/introspection."""
    return tuple(_WRITE_METADATA_CHECKS)


def validate_read_metadata(context: Mapping[str, Any] | None = None) -> None:
    """Run all registered read-side ambiguous-metadata checks (issue #1987).

    Parameters
    ----------
    context : mapping, optional
        Keys consumed by the registered checks. The PR-0 hook does not
        prescribe a schema; each per-case PR documents the keys it
        reads (e.g. ``'crs_wkt'``, ``'transform'``, ``'band_nodata'``).
        A missing key is treated as "nothing to check" by the
        downstream check, not as an error here.

    Raises
    ------
    GeoTIFFAmbiguousMetadataError
        Or one of its subclasses, from a registered check.

    Notes
    -----
    No-op when no checks are registered, so PR 0 does not change
    behaviour at any entry point.
    """
    if not _READ_METADATA_CHECKS:
        return
    ctx: Mapping[str, Any] = {} if context is None else context
    # Iterate over a snapshot. A check that registers or unregisters another
    # check during dispatch (whether on purpose or via an import side effect)
    # would otherwise reshape the list mid-loop and skip or repeat entries.
    # The cost is one tuple per dispatch, paid only when at least one check
    # is registered.
    for check in tuple(_READ_METADATA_CHECKS):
        check(ctx)


def validate_write_metadata(context: Mapping[str, Any] | None = None) -> None:
    """Run all registered write-side ambiguous-metadata checks (issue #1987).

    Mirror of ``validate_read_metadata`` for ``to_geotiff`` /
    ``write_geotiff_gpu`` / ``write_vrt``. See that docstring for the
    context-schema convention and the no-op-when-empty guarantee.
    """
    if not _WRITE_METADATA_CHECKS:
        return
    ctx: Mapping[str, Any] = {} if context is None else context
    # Snapshot for the same reason as the read hook above.
    for check in tuple(_WRITE_METADATA_CHECKS):
        check(ctx)


# ---------------------------------------------------------------------------
# Conflicting crs / crs_wkt write check (issue #1987 PR 1)
# ---------------------------------------------------------------------------


def _check_write_conflicting_crs(context: Mapping[str, Any]) -> None:
    """Refuse writes whose ``attrs['crs']`` and ``attrs['crs_wkt']`` disagree.

    The legacy writer consults both attrs (``crs`` takes priority; the
    WKT is a fallback). When the two encode different CRSes, the writer
    silently emits the EPSG and drops the WKT, producing a TIFF whose
    on-disk CRS does not match what the caller's DataArray advertised.

    This check runs whenever both attrs are populated. It canonicalises
    each via :mod:`pyproj` and raises :class:`ConflictingCRSError` if
    they do not match. A read-back DataArray that legitimately carries
    both attrs (the reader emits both when the file has a CRS) passes
    because the two derive from the same on-disk CRS.

    Soft preconditions kept lenient:

    * ``pyproj`` not installed -> no-op (downstream paths handle the
      missing dependency).
    * Either attr unparseable -> no-op (a sibling check in this issue's
      series, :class:`UnparseableCRSError`, will refuse those on its
      own PR).

    Context keys consumed:

    * ``crs_kwarg`` -- the value of the explicit ``crs=`` writer kwarg.
      When this is not ``None`` the kwarg overrides both attrs, so the
      attrs disagreement does not affect this write and the check
      short-circuits.
    * ``attrs_crs`` -- the value of ``data.attrs.get('crs')``.
    * ``attrs_crs_wkt`` -- the value of ``data.attrs.get('crs_wkt')``.

    All keys are optional; absence is treated as "nothing to compare".
    """
    if context.get('crs_kwarg') is not None:
        return
    crs_attr = context.get('attrs_crs')
    crs_wkt_attr = context.get('attrs_crs_wkt')
    if crs_attr is None or not crs_wkt_attr:
        return
    try:
        from pyproj import CRS as _PyProjCRS
    except ImportError:
        return
    try:
        crs_from_attr = _PyProjCRS.from_user_input(crs_attr)
        crs_from_wkt = _PyProjCRS.from_wkt(crs_wkt_attr)
    except Exception:
        return
    if crs_from_attr.equals(crs_from_wkt):
        return
    # Truncate the WKT in the message; full WKTs are 1-3 kB and would
    # make the error unreadable. The user's attrs still carry the full
    # string so they can introspect it directly.
    wkt_excerpt = crs_wkt_attr if len(crs_wkt_attr) <= 60 else (
        crs_wkt_attr[:57] + '...'
    )
    raise ConflictingCRSError(
        f"attrs['crs']={crs_attr!r} and attrs['crs_wkt']={wkt_excerpt!r} "
        f"resolve to different CRSes after pyproj canonicalisation. The "
        f"writer would silently pick attrs['crs'] and drop the WKT, so "
        f"the on-disk CRS would not match what the DataArray advertises. "
        f"Reconcile the two attrs (drop the stale one, or pass the "
        f"intended CRS explicitly via the ``crs=`` kwarg, which overrides "
        f"both attrs) and retry. See issue #1987."
    )


# Active by default: the conflicting-CRS write check is registered at
# import time so every entry point that calls ``validate_write_metadata``
# enforces the rule. Tests that need to scope around the check should
# use ``unregister_write_metadata_check(_check_write_conflicting_crs)``
# in a fixture rather than mutating the registry directly.
register_write_metadata_check(_check_write_conflicting_crs)


# ---------------------------------------------------------------------------
# Conflicting nodata aliases write check (issue #1987 PR 7)
# ---------------------------------------------------------------------------


def _check_write_conflicting_nodata(context: Mapping[str, Any]) -> None:
    """Refuse writes whose ``attrs['nodata']`` and ``attrs['nodatavals']``
    disagree.

    The legacy resolver ``_resolve_nodata_attr`` prefers
    ``attrs['nodata']`` and falls back to ``attrs['nodatavals']`` only
    when the canonical scalar is missing. When both are set to
    different sentinels, the writer silently picks the canonical one
    and drops the rioxarray-style tuple. This check refuses the
    ambiguity at the boundary.

    Tolerant of bands that legitimately disagree with the canonical
    sentinel as long as at least one band in ``nodatavals`` matches.
    The rioxarray convention is per-band; xrspatial's canonical
    ``nodata`` flattens to a single value. A tuple where every band
    disagrees with ``attrs['nodata']`` is the regression case.

    ``_FillValue`` stays deprioritised per the existing resolver
    convention; this check does not look at it.

    Context keys consumed:

    * ``nodata_kwarg`` -- the explicit ``nodata=`` writer kwarg.
      When set, it overrides attrs and the check short-circuits.
    * ``attrs_nodata`` -- ``data.attrs.get('nodata')``.
    * ``attrs_nodatavals`` -- ``data.attrs.get('nodatavals')``.
    """
    if context.get('nodata_kwarg') is not None:
        return
    nodata_attr = context.get('attrs_nodata')
    nodatavals_attr = context.get('attrs_nodatavals')
    if nodata_attr is None or nodatavals_attr is None:
        return
    # Coerce to float once for comparison; non-numeric entries (None,
    # strings) and NaN are already handled by _resolve_nodata_attr and
    # are out of scope for this disagreement check.
    try:
        canonical = float(nodata_attr)
    except (TypeError, ValueError):
        return
    try:
        vals = list(nodatavals_attr)
    except TypeError:
        return
    concrete: list[float] = []
    for v in vals:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        concrete.append(fv)
    if not concrete:
        return
    # NaN canonical handling: if attrs['nodata'] is NaN, any concrete
    # entry in the tuple is by definition "different". But NaN means
    # "the float NaN is the sentinel", which contradicts a concrete
    # numeric in the tuple. Refuse rather than guess.
    if np.isnan(canonical):
        raise ConflictingNodataError(
            f"attrs['nodata']=nan but attrs['nodatavals']={nodatavals_attr!r} "
            f"contains concrete numeric sentinels {concrete!r}. The "
            f"canonical scalar (NaN) and the per-band tuple disagree on "
            f"what the missing-data sentinel is. Reconcile the two attrs "
            f"(drop the stale one, or pass the intended sentinel via the "
            f"``nodata=`` kwarg) and retry. See issue #1987."
        )
    if any(v == canonical for v in concrete):
        return
    raise ConflictingNodataError(
        f"attrs['nodata']={nodata_attr!r} disagrees with every concrete "
        f"entry in attrs['nodatavals']={nodatavals_attr!r}. The writer "
        f"would silently use attrs['nodata'] and discard the rioxarray "
        f"tuple, so the on-disk sentinel would not match what the "
        f"per-band tuple advertised. Reconcile the two attrs (drop the "
        f"stale one, or pass the intended sentinel via the ``nodata=`` "
        f"kwarg, which overrides both attrs) and retry. See issue #1987."
    )


register_write_metadata_check(_check_write_conflicting_nodata)


# ---------------------------------------------------------------------------
# Non-uniform coords write check (issue #1987 PR 4)
# ---------------------------------------------------------------------------


def _check_write_non_uniform_coords(context: Mapping[str, Any]) -> None:
    """Refuse writes whose ``coords`` imply a non-uniform pixel grid.

    The writer treats the first two y / x coord values as the pixel
    size and emits a uniform-grid ``GeoTransform``. When the rest of
    the coords disagree with that step (variable cell size, gaps),
    the on-disk file silently misrepresents the geometry.

    Exemption: int-dtype coords keep the existing #1969 sentinel
    contract. Those coords are the 0..N-1 fallback the no-georef
    reader emits, not geographic positions, so non-uniformity is
    meaningless for them.

    Context keys consumed:

    * ``coord_y`` -- ``data.coords['y'].values`` (or equivalent).
    * ``coord_x`` -- ``data.coords['x'].values``.

    Missing or non-numeric coords are out of scope (handled by other
    writer validation upstream).
    """
    for axis_name in ('y', 'x'):
        coord = context.get(f'coord_{axis_name}')
        if coord is None:
            continue
        arr = np.asarray(coord)
        if arr.size < 3:
            # Need at least 3 samples to detect non-uniformity.
            continue
        if np.issubdtype(arr.dtype, np.integer):
            # #1969 sentinel exemption.
            continue
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        diffs = np.diff(arr.astype(np.float64))
        # Use a relative tolerance pegged to the step magnitude so that
        # float32 rounding noise (~1e-7 relative) does not trip the
        # check. Same tolerance the existing #1720 coord-regularity
        # check uses.
        step = float(diffs[0])
        if step == 0.0:
            raise NonUniformCoordsError(
                f"data.coords[{axis_name!r}] is constant; the writer cannot "
                f"derive a pixel size from a zero-step axis. See issue #1987."
            )
        rel = np.abs(diffs - step) / np.abs(step)
        if not np.all(rel < 1e-5):
            worst = int(np.argmax(rel))
            raise NonUniformCoordsError(
                f"data.coords[{axis_name!r}] is non-uniform: step "
                f"{step!r} expected, but found {float(diffs[worst])!r} at "
                f"index {worst} (relative drift {float(rel[worst]):.2e}). "
                f"The writer treats the first two coord values as the "
                f"pixel size and would emit a uniform-grid GeoTransform "
                f"that misrepresents the rest of the axis. Resample to a "
                f"uniform grid before writing (e.g. ``data.interp(...)``) "
                f"or write each contiguous block as a separate file. "
                f"See issue #1987."
            )


register_write_metadata_check(_check_write_non_uniform_coords)


# ---------------------------------------------------------------------------
# Unparseable CRS read check (issue #1987 PR 2)
#
# The write-side equivalent (refusing to emit an unparseable string to
# GTCitationGeoKey) is already in place in ``_crs._validate_crs_fallback``
# and now raises ``UnparseableCRSError`` (subclass of ``ValueError`` so
# existing ``except ValueError`` callers keep working).
# ---------------------------------------------------------------------------


def _check_read_unparseable_crs(context: Mapping[str, Any]) -> None:
    """Refuse reads whose ``crs_wkt`` cannot be parsed by pyproj.

    A WKT that the reader extracts from ``GeoAsciiParams`` or a VRT
    ``SRS`` tag may be partial or corrupted. The legacy reader emitted
    it verbatim in ``attrs['crs_wkt']``; downstream code reading the
    attr through pyproj would then crash with a less obvious message.

    Context keys consumed:

    * ``allow_unparseable_crs`` -- caller opt-out kwarg. When True,
      this check is silent (matches the existing write-side behaviour).
    * ``crs_wkt`` -- the WKT string extracted from the file.

    Soft preconditions:

    * No pyproj installed -> no-op (cannot prove the WKT is bad).
    * WKT empty or None -> no-op.
    * WKT that pyproj parses cleanly -> no-op.
    """
    if context.get('allow_unparseable_crs'):
        return
    crs_wkt = context.get('crs_wkt')
    if not crs_wkt:
        return
    try:
        from pyproj import CRS as _PyProjCRS
        from pyproj.exceptions import CRSError as _PyProjCRSError
    except ImportError:
        return
    try:
        # ``from_user_input`` accepts WKT, EPSG-style ``"EPSG:NNNN"`` and
        # PROJ strings -- anything pyproj can resolve. The GDAL VRT
        # ``<SRS>`` tag (and a few other source paths) routinely stash
        # non-WKT but pyproj-parseable strings into ``crs_wkt``, so the
        # check rejects only the strict "pyproj cannot parse" case.
        _PyProjCRS.from_user_input(crs_wkt)
    except _PyProjCRSError as e:
        excerpt = crs_wkt if len(crs_wkt) <= 80 else (crs_wkt[:77] + '...')
        raise UnparseableCRSError(
            f"GeoTIFF source advertises a CRS string that pyproj cannot "
            f"parse: {excerpt!r} ({type(e).__name__}: {e}). The legacy "
            f"reader would emit the unparseable string verbatim in "
            f"attrs['crs_wkt'] and let downstream code crash on first use. "
            f"Pass ``allow_unparseable_crs=True`` to keep that behaviour, "
            f"or re-encode the source with a valid CRS. See issue #1987."
        ) from e


register_read_metadata_check(_check_read_unparseable_crs)


# ---------------------------------------------------------------------------
# Rotated transform read check (issue #1987 PR 3)
# ---------------------------------------------------------------------------


def _check_read_rotated_transform(context: Mapping[str, Any]) -> None:
    """Refuse reads whose ``transform`` has non-zero rotation/shear terms.

    Downstream xrspatial functions assume axis-aligned rasters
    (the GeoTransform's b and d terms are zero). A rotated grid would
    silently produce wrong results in slope / aspect / hillshade /
    proximity / zonal-statistics.

    Context keys consumed:

    * ``allow_rotated`` -- caller opt-out kwarg.
    * ``transform`` -- the 6-tuple ``(pixel_width, b, origin_x, d,
      pixel_height, origin_y)`` (rasterio Affine ordering).

    A transform with ``b == 0`` and ``d == 0`` is axis-aligned and
    passes. A missing transform is out of scope (the no-georef path
    handles it).
    """
    if context.get('allow_rotated'):
        return
    transform = context.get('transform')
    if not transform:
        return
    try:
        # rasterio Affine ordering: (pixel_width, b, origin_x, d, pixel_height, origin_y)
        b = float(transform[1])
        d = float(transform[3])
    except (IndexError, TypeError, ValueError):
        return
    if b == 0.0 and d == 0.0:
        return
    raise RotatedTransformError(
        f"GeoTIFF source carries a rotated affine transform (b={b!r}, "
        f"d={d!r}; both must be 0.0 for an axis-aligned grid). "
        f"Downstream xrspatial ops (slope, aspect, hillshade, proximity, "
        f"zonal) assume axis-aligned rasters and would silently produce "
        f"wrong results on a rotated grid. Pass ``allow_rotated=True`` "
        f"to read the pixel grid without the geospatial assumption "
        f"(useful when you only want the array, not the geo-aware "
        f"downstream ops). See issue #1987."
    )


register_read_metadata_check(_check_read_rotated_transform)


# ---------------------------------------------------------------------------
# Mixed band metadata read check (issue #1987 PR 5)
# ---------------------------------------------------------------------------


def _check_read_mixed_band_metadata(context: Mapping[str, Any]) -> None:
    """Refuse VRT reads where bands declare disagreeing per-band nodata.

    A VRT can mosaic sources with different per-band nodata sentinels.
    The legacy reader flattens to one value silently, which means one
    band's "valid" pixels can collide with another band's sentinel
    after the flatten. The fail-closed default refuses the ambiguity;
    pass ``band_nodata='first'`` to keep the legacy behaviour
    explicitly.

    Context keys consumed:

    * ``band_nodata`` -- caller opt-out kwarg. ``None`` means
      strict (raise on disagreement); ``'first'`` keeps the legacy
      flatten-to-first behaviour.
    * ``band_nodata_values`` -- iterable of per-band nodata sentinels
      extracted from the VRT (one entry per band; ``None`` for bands
      without a declared sentinel).
    """
    band_nodata = context.get('band_nodata')
    if band_nodata == 'first':
        return
    vals = context.get('band_nodata_values')
    if not vals:
        return
    seen: list = []
    for v in vals:
        if v is None:
            continue
        # Two NaNs are considered equal here (both mean "NaN sentinel"),
        # matching how _resolve_nodata_attr treats them.
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if any(_same_nodata(fv, s) for s in seen):
            continue
        seen.append(fv)
    if len(seen) <= 1:
        return
    raise MixedBandMetadataError(
        f"VRT source declares disagreeing per-band nodata sentinels: "
        f"{vals!r}. Flattening to a single value would silently collide "
        f"with another band's valid pixels. Pass "
        f"``band_nodata='first'`` to keep the legacy flatten-to-first "
        f"behaviour, or rebuild the VRT with a single shared sentinel. "
        f"See issue #1987."
    )


def _same_nodata(a: float, b: float) -> bool:
    """Equality with NaN-equals-NaN semantics for nodata comparison."""
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b


# NOT registered by default. The mixed-band check has a much larger
# migration cost than its sibling read-side checks (around 35 existing
# test sites would need to opt in via ``band_nodata='first'`` to keep
# their legacy assertions), so it lands as a follow-up PR that bundles
# the check activation with the test migration. The check itself and
# the ``band_nodata=`` VRT kwarg ship now so the follow-up is a
# one-line registration call plus the test sweep.
# register_read_metadata_check(_check_read_mixed_band_metadata)
