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
    for check in _READ_METADATA_CHECKS:
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
    for check in _WRITE_METADATA_CHECKS:
        check(ctx)
