"""Input validators shared by the geotiff entry points.

Pure leaves over numpy dtypes and Python primitives. Called from
``to_geotiff``, ``_read_geotiff_dask``, ``_read_geotiff_gpu``,
``_read_vrt``, and ``_write_geotiff_gpu`` so the rejection rules
(non-positive chunks, lossy float-to-int casts, ambiguous 3D dim
layouts, tile-size multiples of 16, etc.) stay in lockstep across
every backend.

Ambiguous-metadata hooks
------------------------
``validate_read_metadata`` and ``validate_write_metadata`` are
plug-points for per-case checks (unparseable CRS, rotated transforms,
non-uniform coords, mixed band metadata, conflicting crs/crs_wkt,
conflicting nodata aliases). The hooks are no-ops until at least one
check is registered.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ._coords import _BAND_DIM_NAMES
from ._errors import (ConflictingCRSError, ConflictingNodataError, InconsistentGeoKeysError,
                      InvalidIntegerNodataError, MixedBandMetadataError, NonUniformCoordsError,
                      RemoteStableSourcesOnlyError, RotatedTransformError, UnparseableCRSError,
                      VRTStableSourcesOnlyError, _distinct_per_band_nodatavals_msg)
from ._runtime import (_MISSING_SOURCES_SENTINEL, _ON_GPU_FAILURE_SENTINEL, _TIME_DIM_NAMES,
                       _X_DIM_NAMES, _Y_DIM_NAMES)


def _is_temporal_dim_name(name) -> bool:
    """Return True if ``name`` is a known temporal dim alias.

    Compared case-insensitively against ``_TIME_DIM_NAMES`` so that
    CF-style ``'TIME'`` / ``'Time'`` reach the friendly temporal error
    in the 3D writer validator instead of slipping through the
    ``(y, x, *)`` band-position fallback.
    """
    return isinstance(name, str) and name.lower() in _TIME_DIM_NAMES


def _validate_3d_writer_dims(dims) -> None:
    """Reject ambiguous 3D writer inputs.

    The writer interprets a 3D DataArray as either ``(band, y, x)`` or
    ``(y, x, band)``. ``data.dims[0] in _BAND_DIM_NAMES`` decides which
    branch fires the ``moveaxis``. Anything else (e.g. ``('time', 'y', 'x')``)
    used to fall through silently: the writer kept the leading axis as
    the spatial ``y`` axis and the result was a TIFF with the leading
    axis values laid out along ``y`` (silent data corruption -- on
    read-back the array round-tripped with a swapped shape).

    DataArray dim contract: the trailing axis of a
    ``(y, x, *)`` layout must be a recognized band alias
    (``_BAND_DIM_NAMES``). Previously the validator silently accepted
    any unknown trailing name (e.g. ``('y', 'x', 'z')``,
    ``('lat', 'lon', 'scenario')``) and wrote those values as TIFF
    bands. That escape hatch was a hold-over for raw-ndarray callers
    that build band-last arrays without dim metadata; raw ndarrays
    never flow through this DataArray-dim validator (callers gate the
    call on ``isinstance(data, xr.DataArray)``), so removing it here
    only tightens the DataArray contract and leaves the bare-ndarray
    path unaffected.

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
    # ``(y, x, *)`` with a non-band trailing dim. Temporal names get a
    # dedicated friendly message; every other non-band trailing name
    # (``z``, ``level``, ``scenario``, ``foo``, ...) is rejected with
    # the dedicated ``"non-band trailing dim"`` message immediately
    # below, closing the escape hatch that previously let unknown
    # trailing names through the band-position fallback. The generic
    # ``"ambiguous dims"`` message at the bottom
    # of the function only fires for layouts that are neither
    # ``(y, x, *)`` nor ``(_TIME_, y, x)`` (e.g. ``('foo', 'y', 'x')``).
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
        raise ValueError(
            f"3D writer input has non-band trailing dim {d2!r} in dims "
            f"{dims!r}. The writer cannot infer that {d2!r} represents "
            f"TIFF bands and used to silently write a multiband file "
            f"with the {d2!r} axis stuffed into the band slot. Rename "
            f"the trailing dim to one of {_BAND_DIM_NAMES}, reduce or "
            f"select along it (e.g. ``data.isel({d2}=0)`` or "
            f"``data.mean({d2!r})``), or pass a raw ndarray with "
            f"shape ``(y, x, bands)`` if you really want the band-last "
            f"layout written without a dim-name check (issue #2240)."
        )
    # Symmetrise the friendly temporal message for the leading-dim case
    # ``(time, y, x)``. The generic ``ambiguous dims`` error below
    # already rejects this layout, but the temporal-specific message
    # tells the caller exactly how to fix it.
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


def _validate_writer_spatial_shape(shape, dims=None,
                                   entry_point: str = "to_geotiff") -> None:
    """Reject empty spatial shapes at the writer entry point.

    Clip and window pipelines can produce empty rasters. The eager and
    streaming writers used to accept those inputs, write a TIFF whose
    IFD claimed shape ``(0, N)`` / ``(N, 0)``, and then the reader
    rejected the file with a generic "Invalid image dimensions" message
    that never named the writer as the source.

    Validate up front. ``shape`` is the array shape (2D ``(h, w)``,
    band-first 3D ``(bands, h, w)``, or band-last 3D ``(h, w, bands)``).
    ``dims`` is the DataArray ``dims`` tuple when available; it lets the
    helper pick the right spatial pair when a 3D band-first input
    arrives. Without ``dims`` the helper assumes band-last for 3D
    (consistent with the writer's pre-moveaxis layout invariant), so
    pass ``dims`` for DataArray inputs to avoid mis-naming the axis.
    ``entry_point`` is the function name used in the error message so
    direct callers of ``write`` / ``write_streaming`` / ``_write_geotiff_gpu``
    see the function they actually invoked.

    Also rejects 3D inputs whose band/sample axis is zero.
    Without the band check, a DataArray of shape ``(0, y, x)`` band-first
    or ``(y, x, 0)`` band-last passed every spatial guard and reached
    the IFD assembly with ``samples_per_pixel == 0``. The resulting TIFF
    was readable as a 2D single-band raster, masking the upstream
    collapse of the band axis.

    Note that this validator runs before ``_validate_3d_writer_dims``
    in ``to_geotiff``. For an ambiguous-dim
    input like ``(5, 5, 0)`` with dims ``('y', 'x', 'time')``, the
    band-last branch sees ``bands == 0`` and the "no bands" error wins
    over the friendlier ambiguous-dim message. Both errors name the
    right call to fix, so the ordering is acceptable; reorder only if
    the ambiguous-dim diagnostic becomes more important than the
    empty-axis one.
    """
    if shape is None:
        return
    ndim = len(shape)
    bands = None
    if ndim == 2:
        h, w = int(shape[0]), int(shape[1])
    elif ndim == 3:
        # Decide band-first vs band-last from ``dims`` when available.
        # Both layouts are valid writer inputs; the spatial axes are
        # whichever two are not the band axis.
        band_first = False
        if dims is not None and len(dims) == 3:
            band_first = dims[0] in _BAND_DIM_NAMES
        if band_first:
            bands = int(shape[0])
            h, w = int(shape[1]), int(shape[2])
        else:
            h, w = int(shape[0]), int(shape[1])
            bands = int(shape[2])
    else:
        # Other rank errors are handled by the existing ndim check; do
        # not shadow that message.
        return
    if h <= 0 or w <= 0:
        raise ValueError(
            f"{entry_point} cannot write an empty raster: got shape "
            f"{tuple(int(s) for s in shape)} with height={h}, width={w}. "
            f"Both spatial dims must be positive. A common cause is a "
            f"clip or window that produced an empty selection; check "
            f"the upstream operation before writing."
        )
    if bands is not None and bands <= 0:
        raise ValueError(
            f"{entry_point} cannot write a raster with no bands: got "
            f"shape {tuple(int(s) for s in shape)} with {bands} bands. "
            f"The band/sample dimension must be positive. A common "
            f"cause is a selection or reduction that collapsed the "
            f"band axis upstream; reduce to 2D before writing, or "
            f"select at least one band (issue #2095)."
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


def _validate_tile_size(tile_size, *, require_multiple_of_16: bool = True
                        ) -> None:
    """Validate ``tile_size`` for the tiled GeoTIFF writers.

    Shared by ``to_geotiff`` (when ``tiled=True``) and
    ``_write_geotiff_gpu`` (always tiled) so the accepted types, the
    non-positive rejection, and the multiple-of-16 hint stay in lockstep.
    The tiled writer computes the tile grid as
    ``math.ceil(width / tile_size)``; ``tile_size=0`` hits
    ``ZeroDivisionError`` deep inside the writer, and negative values
    produce a nonsensical tile grid. The TIFF 6 spec also requires
    ``TileWidth`` and ``TileLength`` to be positive multiples of 16
    for broad interoperability with libtiff / GDAL strict readers; a
    value like 17 would otherwise round-trip through the in-repo
    reader but be rejected elsewhere.

    ``require_multiple_of_16`` defaults to ``True`` for the public
    boundary (``to_geotiff`` / ``_write_geotiff_gpu``), which enforces
    spec interop. The array-level private writers (``_write`` /
    ``_write_streaming``) pass ``False``: they still reject the
    crash-inducing cases (non-int, non-positive) but allow the small
    spec-noncompliant tile sizes that the in-repo reader round-trips
    and that the internal tests rely on. The multiple-of-16 policy
    stays at the public boundary (issue #2997).
    """
    if not isinstance(tile_size, (int, np.integer)) or isinstance(
            tile_size, bool):
        raise ValueError(
            f"tile_size must be a positive int, got "
            f"{tile_size!r} (type {type(tile_size).__name__}).")
    if tile_size <= 0:
        raise ValueError(
            f"tile_size must be a positive int, got tile_size={tile_size}.")
    if require_multiple_of_16 and tile_size % 16 != 0:
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

    Centralises the rejection rule that ``_read_geotiff_dask`` already
    runs so ``_read_geotiff_gpu`` and ``_read_vrt`` can share the same
    error format. With ``allow_none=True`` a ``None`` value passes
    through unchanged (used by entry points whose default is
    ``chunks=None``, e.g. ``_read_geotiff_gpu`` and ``_read_vrt``).
    With ``allow_none=False`` (default, matches ``_read_geotiff_dask``)
    a ``None`` is rejected with the same ``ValueError`` format as any
    other non-int / non-tuple value, so callers see a clear
    parameter-named error instead of a downstream ``TypeError`` from
    the chunk-unpacking math.
    Otherwise ``chunks`` must be a positive int or a 2-tuple of
    positive ints. Booleans are rejected because ``True``/``False``
    are int subclasses that would otherwise sneak through the integer
    check. Returns the coerced int when given an ``np.integer`` scalar
    so downstream ``isinstance(chunks, int)`` checks stay accurate.

    Mirrors the chunks-validation in ``_read_geotiff_dask``; extends it
    to the GPU read and VRT read entry points.
    """
    if chunks is None:
        if allow_none:
            return chunks
        raise ValueError(
            "chunks must be a positive int or (row, col) tuple of "
            "positive ints, got chunks=None.")
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


def _validate_tile_size_arg(tile_size, *,
                            require_multiple_of_16: bool = True):
    """Validate the ``tile_size`` kwarg for the tiled writer entry points.

    Wrapper kept for backwards internal compatibility; delegates to
    ``_validate_tile_size`` so to_geotiff/_write_geotiff_gpu share one
    validation path (positive int + multiple-of-16 for tiled output).
    Pass ``require_multiple_of_16=False`` from the array-level private
    writers, which enforce only the crash-inducing positivity/type rules
    (issue #2997).
    """
    _validate_tile_size(
        tile_size, require_multiple_of_16=require_multiple_of_16)


def _validate_overview_level_arg(overview_level) -> None:
    """Validate the ``overview_level`` kwarg type.

    Accepts ``None`` (default, treated as full resolution) and any
    ``int`` / ``numpy.integer`` instance. ``bool`` is rejected
    explicitly because Python treats it as a subclass of ``int``;
    without the check, ``overview_level=True`` is coerced to ``1`` and
    silently returns the first overview level. Non-int types like
    ``str`` and ``float`` would otherwise leak raw ``TypeError``
    messages from the internal numeric comparison or list indexing.

    Called by ``open_geotiff`` (public entry point) and by
    ``select_overview_ifd`` (defense in depth for the dask, GPU, and
    accessor readers that reach the selector without going through
    ``open_geotiff``).
    """
    if overview_level is None:
        return
    if (not isinstance(overview_level, (int, np.integer))
            or isinstance(overview_level, bool)):
        raise TypeError(
            f"overview_level must be an int or None, got "
            f"{type(overview_level).__name__}: {overview_level!r}"
        )


def _validate_gpu_arg(gpu, *, allow_none: bool = False) -> None:
    """Validate the ``gpu`` dispatch flag type.

    ``gpu`` selects the GPU backend by truthiness on both the read
    (``open_geotiff`` / ``_validate_dispatch_kwargs``) and write
    (``to_geotiff``) paths. Without a type check, a non-bool such as
    ``gpu="False"`` is truthy and silently routes to the GPU path
    instead of raising. ``bool`` is also a subclass of ``int``, so
    ``gpu=1`` / ``gpu=0`` would otherwise sneak through any
    ``isinstance(gpu, int)`` style guard; reject them as non-bool.

    ``np.bool_`` is accepted alongside Python ``bool`` (it is not a
    ``bool`` subclass but is the obvious numpy equivalent, matching how
    ``nodata`` validation treats ``(bool, np.bool_)``).

    ``allow_none=True`` keeps the writer's ``gpu=None`` "auto-detect
    from the data" sentinel valid; the read path leaves it False so
    only a ``bool`` is accepted.
    """
    if allow_none and gpu is None:
        return
    if not isinstance(gpu, (bool, np.bool_)):
        suffix = " or None" if allow_none else ""
        raise TypeError(
            f"gpu must be a bool{suffix}, got "
            f"{type(gpu).__name__}: {gpu!r}")


def _validate_dispatch_kwargs(
    *,
    source,
    gpu: bool,
    chunks,
    overview_level,
    on_gpu_failure,
    missing_sources,
    band_nodata,
    max_cloud_bytes,
) -> None:
    """Validate dispatcher-level kwargs across the GeoTIFF read entry points.

    Holds the kwarg-rejection rules that ``open_geotiff`` used to run
    inline so the three direct backends (``_read_geotiff_dask``,
    ``_read_geotiff_gpu``, ``_read_vrt``) get the same validation when
    called directly. Before this helper, a caller who passed
    ``max_cloud_bytes`` straight to ``_read_geotiff_dask`` (or
    ``band_nodata`` to ``_read_geotiff_gpu``) got no error at all because
    the kwarg either silently dropped or raised an unrelated
    ``TypeError`` from the signature.

    Rules enforced:

    - ``overview_level`` must be ``None`` or a non-bool int. Delegates to
      :func:`_validate_overview_level_arg`.
    - ``on_gpu_failure`` requires ``gpu=True`` when explicit. The
      CPU/dask branches have no GPU failure policy and would otherwise
      drop the kwarg silently.
    - ``missing_sources`` requires a ``.vrt`` source when explicit. The
      non-VRT branches do not run the VRT mosaic loop.
    - ``band_nodata`` requires a ``.vrt`` source when not ``None``. The
      fail-closed mixed-band-metadata check is VRT-only.
    - ``max_cloud_bytes`` is incompatible with ``.vrt``, ``gpu=True``,
      and ``chunks=...`` when explicit. The eager non-VRT non-GPU
      non-dask branch is the only consumer.
    - File-like sources reject ``gpu=True`` and ``chunks=...``. The GPU
      and dask paths re-open the source by path from worker tasks.

    Sentinel defaults (``_ON_GPU_FAILURE_SENTINEL``,
    ``_MISSING_SOURCES_SENTINEL``, ``_MAX_CLOUD_BYTES_SENTINEL``) keep
    "caller did not pass this kwarg" silent so defaults round-trip
    through every entry point without tripping a rejection.

    Parameters
    ----------
    source : str or binary file-like
        Already coerced by ``_coerce_path`` so the helper can detect
        a ``.vrt`` extension via ``isinstance(source, str)``.
    gpu : bool
        True when the call routes through the GPU pipeline (either
        ``open_geotiff(gpu=True)`` or a direct ``_read_geotiff_gpu``).
        This is the dispatch bool, type-checked via
        ``_validate_gpu_arg``. Not to be confused with
        ``_read_geotiff_gpu``'s own deprecated ``gpu='strict'/'auto'/
        'loose'`` string parameter (the legacy ``on_gpu_failure``
        alias), which never reaches this helper -- ``_read_geotiff_gpu``
        passes a literal ``True`` here.
    chunks : int, tuple, or None
        Caller's ``chunks=`` value. ``None`` means eager.
    overview_level
        Forwarded to ``_validate_overview_level_arg`` verbatim.
    on_gpu_failure
        ``_ON_GPU_FAILURE_SENTINEL`` when omitted; any other value is
        treated as caller-set.
    missing_sources
        ``_MISSING_SOURCES_SENTINEL`` when omitted; any other value is
        treated as caller-set.
    band_nodata
        ``None`` when omitted; any other value is treated as caller-set.
    max_cloud_bytes
        ``_MAX_CLOUD_BYTES_SENTINEL`` when omitted; any other value
        (including an explicit ``None``) is treated as caller-set.

    Raises
    ------
    TypeError
        If ``overview_level`` has a non-int / bool type.
    ValueError
        On any of the rule violations listed above.
    """
    # Local import avoids a circular dependency with ``_reader`` when
    # ``_validation`` is imported at module load time of the geotiff
    # subpackage. The sentinel binding is cheap to look up.
    # TODO: move ``_MAX_CLOUD_BYTES_SENTINEL`` to ``_runtime`` alongside
    # the other dispatch sentinels so this local import can hoist to
    # module scope.
    from ._reader import _MAX_CLOUD_BYTES_SENTINEL

    _validate_overview_level_arg(overview_level)
    _validate_gpu_arg(gpu)

    if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL and not gpu:
        raise ValueError(
            "on_gpu_failure only applies when gpu=True. "
            "Pass gpu=True to enable the GPU pipeline, or drop "
            "on_gpu_failure to keep the default CPU path.")

    # File-like buffers do not support the GPU or dask code paths
    # because those re-open the source by path from worker tasks or
    # device-side readers. Reject early with a clear message that names
    # the actual blocker (gpu=True or chunks=...), ahead of the
    # VRT-dependent kwarg checks below. A file-like source cannot be a
    # VRT either, so the later VRT-only rejections would fire on
    # ``missing_sources`` / ``band_nodata`` with a less specific
    # diagnostic. Surfacing the file-like restriction first gives the
    # caller the direct fix (pass a path string).
    if not isinstance(source, str):
        if gpu:
            raise ValueError(
                "gpu=True is not supported for file-like sources. "
                "Pass a path string instead.")
        if chunks is not None:
            raise ValueError(
                "chunks=... (dask) is not supported for file-like sources. "
                "Pass a path string instead.")

    missing_sources_passed = (
        missing_sources is not _MISSING_SOURCES_SENTINEL)
    is_vrt_source = (
        isinstance(source, str) and source.lower().endswith('.vrt'))
    if missing_sources_passed and not is_vrt_source:
        raise ValueError(
            "missing_sources only applies to VRT sources. "
            "Pass a .vrt path to enable the VRT pipeline, or drop "
            "missing_sources to keep the default GeoTIFF path.")

    if band_nodata is not None and not is_vrt_source:
        raise ValueError(
            "band_nodata only applies to VRT sources. "
            "Pass a .vrt path to enable the VRT pipeline, or drop "
            "band_nodata to keep the default GeoTIFF path. "
            "See issue #1987.")

    if max_cloud_bytes is not _MAX_CLOUD_BYTES_SENTINEL:
        if is_vrt_source:
            raise ValueError(
                "max_cloud_bytes is not supported for VRT sources. "
                "The VRT reader does not apply the cloud-byte budget; "
                "drop the kwarg, or call open_geotiff on the underlying "
                ".tif source.")
        if gpu:
            raise ValueError(
                "max_cloud_bytes is not supported when gpu=True. "
                "The GPU reader does not accept the kwarg directly; "
                "for URL/fsspec sources the budget is honoured via "
                "the XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES env var "
                "(see issue #2161). Drop the kwarg, set the env var, "
                "or pass gpu=False to use the eager CPU path.")
        if chunks is not None:
            raise ValueError(
                "max_cloud_bytes is not supported when chunks=... (dask). "
                "The dask reader does not apply the cloud-byte budget; "
                "drop the kwarg, or drop chunks to use the eager path.")


def _validate_predictor_sample_format(predictor, sample_format) -> None:
    """Reject ``Predictor=3`` paired with a non-float ``SampleFormat``.

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
    """Reject non-numeric ``nodata=`` at the writer entry point.

    ``None`` (no sentinel) passes through. ``bool`` is rejected with
    ``TypeError`` so all three writer entry points (eager, GPU, VRT)
    refuse ``nodata=True`` / ``nodata=False`` the same way the eager
    path already does -- ``float(True) == 1.0`` would
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


def _validate_int_nodata_for_dtype(
    nodata,
    dtype,
    *,
    allow_invalid_nodata: bool = False,
) -> None:
    """Reject a non-finite or fractional ``GDAL_NODATA`` on an integer source.

    Mirrors the masking-time gate in ``_nodata._sentinel_fits_dtype``: a
    NaN / Inf / fractional ``GDAL_NODATA`` value cannot match any pixel
    in an integer buffer, so the reader cannot honour the sentinel. The
    legacy behaviour parsed the value into ``attrs['nodata']``
    and silently skipped the masking step. The release contract upgrades
    that silent no-op to a typed rejection so downstream code does not
    quietly see the raw sentinel value in the buffer instead of NaN.

    Parameters
    ----------
    nodata : scalar or None
        The parsed ``GDAL_NODATA`` value (``int`` for plain integer
        literals, ``float`` otherwise; ``None`` when the tag is absent).
        ``None`` is a no-op.
    dtype : numpy.dtype or None
        The file's source pixel dtype. Float dtypes and ``None`` are
        no-ops; only signed/unsigned integer kinds enter the validation
        branch.
    allow_invalid_nodata : bool, default False
        Opt-in for the pre-rejection behaviour. ``True`` restores the
        legacy silent no-op without raising; callers that need to read
        files known to carry such sentinels (e.g. external tooling that
        emits ``GDAL_NODATA="nan"`` on integer outputs) pass this flag.

    Raises
    ------
    InvalidIntegerNodataError
        When ``dtype`` is integer and ``nodata`` is non-finite (NaN /
        Inf) or fractional. The message names ``nodata``, the integer
        dtype, the sentinel value, the opt-in flag, and cites the
        ``release_gate_geotiff`` contract document.
    """
    if allow_invalid_nodata or nodata is None or dtype is None:
        return
    np_dtype = np.dtype(dtype)
    if np_dtype.kind not in ('u', 'i'):
        return
    try:
        as_float = float(nodata)
    except (TypeError, ValueError):
        # Non-numeric sentinel slipped past _parse_nodata_str (defensive;
        # the parser returns int / float / None). Fall back to the
        # validator's "no opinion" branch; downstream masking gates will
        # treat it as a no-op the same way the legacy path did.
        return
    if np.isfinite(as_float) and as_float.is_integer():
        return
    if not np.isfinite(as_float):
        kind = "non-finite"
    else:
        kind = "fractional"
    raise InvalidIntegerNodataError(
        f"GeoTIFF declares GDAL_NODATA={nodata!r} ({kind}) on a "
        f"{np_dtype.name} source. The masking step cannot match a "
        f"{kind} sentinel against an integer pixel buffer, so the "
        f"reader would silently skip masking and leave the raw "
        f"sentinel value in the data. Pass allow_invalid_nodata=True "
        f"to keep the legacy no-op behaviour, or re-encode the file "
        f"with a finite in-range integer sentinel. See "
        f"docs/source/reference/release_gate_geotiff.rst for the "
        f"release contract on nodata handling (#1774 / #2341)."
    )


def _validate_stable_only_vrt(
    source: str,
    *,
    stable_only: bool,
    allow_experimental_codecs: bool = False,
) -> None:
    """Reject a VRT source when the caller asks for stable-only sources.

    Implements the read-side gate the release-contract test
    ``test_release_gate_negative_mixed_tier_vrt_children`` pins. The
    ``reader.vrt`` entry point sits at the ``advanced``
    tier in :data:`xrspatial.geotiff.SUPPORTED_FEATURES`, and VRT child
    sources can declare any codec / capability the underlying GeoTIFF
    reader supports (including the ``experimental`` and ``internal_only``
    tiers). A caller who asks for stable-only sources via
    ``stable_only=True`` therefore cannot be served from a VRT mosaic
    without naming the broader-tier opt-in (``allow_experimental_codecs``).
    Reject up front at graph-build / eager-read setup time so the
    failure surfaces before any pixel decode work.

    Parameters
    ----------
    source : str
        Path to the ``.vrt`` file. Embedded in the rejection message so
        the caller can locate the offending file without re-parsing.
        Non-string sources (eager file-like buffers) and string paths
        that do not end in ``.vrt`` are silently passed through; the
        gate only fires for sources the caller could reasonably have
        intended as a VRT mosaic, so a future call site that routes a
        non-VRT path through this helper does not mislabel the failure.
    stable_only : bool
        Caller's opt-in for stable-only sources. ``False`` is a no-op;
        ``True`` raises :class:`VRTStableSourcesOnlyError` (provided
        ``source`` looks like a VRT path).
    allow_experimental_codecs : bool, default False
        Companion opt-in. When the caller passes both ``stable_only=True``
        and ``allow_experimental_codecs=True`` the request is internally
        contradictory, but ``allow_experimental_codecs=True`` is the
        documented unlock so we honour it: the gate becomes a no-op and
        the per-source codec gate downstream handles the rest. Pure
        ``stable_only=True`` (without the unlock) raises.

    Raises
    ------
    VRTStableSourcesOnlyError
        When ``stable_only=True`` and the source path ends in ``.vrt``
        (case-insensitive) and the caller did not pass
        ``allow_experimental_codecs=True``. The message names the
        offending VRT path, both flags, and cites the release-contract
        document.
    """
    if not stable_only:
        return
    if allow_experimental_codecs:
        return
    # Defensive extension check: every public-API call site routes only
    # ``.vrt`` paths into this helper, but a future call site could
    # forward a non-VRT path. Pass through silently in that case so the
    # rejection message never mislabels a non-VRT source as a VRT.
    if not (isinstance(source, str) and source.lower().endswith('.vrt')):
        return
    raise VRTStableSourcesOnlyError(
        f"VRT source '{source}' cannot be opened under stable_only=True. "
        f"The VRT reader (``reader.vrt``) sits at the advanced tier in "
        f"SUPPORTED_FEATURES, and VRT child sources can declare any "
        f"codec or capability the GeoTIFF reader supports, including "
        f"the experimental and internal-only tiers. The stable-only "
        f"request cannot be served from a VRT mosaic without an "
        f"explicit broader-tier opt-in. Pass "
        f"allow_experimental_codecs=True to opt in to the advanced / "
        f"experimental tiers, or drop stable_only=True to keep the "
        f"default behaviour. See "
        f"docs/source/reference/release_gate_geotiff.rst for the "
        f"release contract and epic #2342 for the tracking issue."
    )


def _validate_stable_only_remote(
    source,
    *,
    stable_only: bool,
    allow_experimental_codecs: bool = False,
) -> None:
    """Reject an HTTP / fsspec source when stable-only sources are asked.

    Companion to :func:`_validate_stable_only_vrt` for the non-VRT read
    paths. ``reader.http`` and ``reader.fsspec`` sit at the ``advanced``
    tier in :data:`xrspatial.geotiff.SUPPORTED_FEATURES`, so a caller who
    asks for stable-only sources via ``stable_only=True`` must not be
    served from a remote source. The VRT dispatcher already gates this at
    graph-build / eager-read setup; the eager and dask non-VRT paths route
    around that gate (the VRT validator only fires for ``.vrt`` paths), so
    HTTP / fsspec sources used to slip through ``stable_only=True``.
    Reject up front, before any range GET or decode work.

    Parameters
    ----------
    source : str or path-like or file-like
        The read source. Only ``http(s)://`` URLs and fsspec URIs raise;
        local-file paths and eager file-like buffers pass through, since
        ``reader.local_file`` is a stable-tier reader.
    stable_only : bool
        Caller's opt-in for stable-only sources. ``False`` is a no-op;
        ``True`` raises :class:`RemoteStableSourcesOnlyError` for a remote
        source unless ``allow_experimental_codecs=True``.
    allow_experimental_codecs : bool, default False
        Documented unlock. When set, the gate becomes a no-op so the
        caller can opt into the advanced / experimental tiers explicitly,
        mirroring :func:`_validate_stable_only_vrt`.

    Raises
    ------
    RemoteStableSourcesOnlyError
        When ``stable_only=True``, the caller did not pass
        ``allow_experimental_codecs=True``, and ``source`` is an HTTP(S)
        URL or an fsspec URI. The message names the offending source,
        both flags, and cites the release-contract document.
    """
    if not stable_only:
        return
    if allow_experimental_codecs:
        return
    # Local imports: the validation layer avoids eager-importing the
    # sources / reader scheme helpers at module load, matching the dask
    # backend so the package imports without the cloud-read deps in
    # environments that only consume local-file reads.
    from ._reader import _is_fsspec_uri
    from ._sources import _is_http_source
    if not (_is_http_source(source) or _is_fsspec_uri(source)):
        return
    raise RemoteStableSourcesOnlyError(
        f"Remote source '{source}' cannot be opened under "
        f"stable_only=True. The HTTP and fsspec readers (``reader.http`` "
        f"/ ``reader.fsspec``) sit at the advanced tier in "
        f"SUPPORTED_FEATURES, so the stable-only request cannot be served "
        f"from a remote source. Pass allow_experimental_codecs=True to "
        f"opt in to the advanced / experimental tiers, or drop "
        f"stable_only=True to keep the default behaviour. See "
        f"docs/source/reference/release_gate_geotiff.rst for the "
        f"release contract and epic #2342 for the tracking issue."
    )


def _validate_no_rotated_affine(attrs, *, drop_rotation: bool,
                                entry_point: str = "to_geotiff") -> None:
    """Refuse writes that would silently drop ``attrs['rotated_affine']``.

    The reader exposes the rotated 6-tuple on ``attrs['rotated_affine']``
    when called with ``allow_rotated=True``. The writer does not emit a
    ``ModelTransformationTag``, so a read-then-write round-trip used to
    silently drop the rotation and write an identity-affine output.
    Refuse the write by default so the loss is impossible without an
    explicit opt-in.

    Parameters
    ----------
    attrs : Mapping or None
        The DataArray's ``attrs``. Bare ``numpy.ndarray`` / ``cupy``
        inputs have no attrs and skip the check.
    drop_rotation : bool
        When True, the caller has explicitly accepted that the rotated
        mapping will be lost on write; the check returns silently.
    entry_point : str
        Name of the calling writer for the error message (``to_geotiff``,
        ``_write_geotiff_gpu``). Lets the two writers surface the same
        wording while still naming the opt-in correctly for either entry
        point.
    """
    if not attrs:
        return
    # ``is None`` skips both the missing-key case and the explicit-None
    # case (the reader's ``_populate_attrs_from_geo_info`` only sets the
    # attr to a tuple OR omits it; an explicit ``None`` arises from
    # caller code that pre-allocates the key). Any other value -- tuple,
    # empty tuple, list, ndarray -- falls through to the rejection
    # branch so a malformed marker still fails closed.
    if attrs.get('rotated_affine') is None:
        return
    if drop_rotation:
        return
    raise ValueError(
        f"{entry_point}: refusing to write a DataArray carrying "
        f"attrs['rotated_affine']. The writer does not emit a "
        f"ModelTransformationTag (issue #2115), so writing this input "
        f"would silently drop the rotated mapping and produce an "
        f"identity-affine GeoTIFF. Either reproject onto an axis-aligned "
        f"grid first, or pass drop_rotation=True to accept the loss "
        f"explicitly (issue #2216)."
    )


# ---------------------------------------------------------------------------
# Ambiguous-metadata hooks
#
# Per-case checks register via
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
    """Register a read-side ambiguous-metadata check.

    Returns ``check`` so the call can be used as a decorator. Idempotent:
    re-registering the same callable is a no-op.
    """
    if check not in _READ_METADATA_CHECKS:
        _READ_METADATA_CHECKS.append(check)
    return check


def register_write_metadata_check(check: _WriteCheck) -> _WriteCheck:
    """Register a write-side ambiguous-metadata check."""
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
    """Run all registered read-side ambiguous-metadata checks.

    Parameters
    ----------
    context : mapping, optional
        Keys consumed by the registered checks. The hook does not
        prescribe a schema; each check documents the keys it reads
        (e.g. ``'crs_wkt'``, ``'transform'``, ``'band_nodata'``).
        A missing key is treated as "nothing to check" by the
        downstream check, not as an error here.

    Raises
    ------
    GeoTIFFAmbiguousMetadataError
        Or one of its subclasses, from a registered check.

    Notes
    -----
    No-op when no checks are registered.
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
    """Run all registered write-side ambiguous-metadata checks.

    Mirror of ``validate_read_metadata`` for ``to_geotiff`` /
    ``_write_geotiff_gpu`` / ``_build_vrt``. See that docstring for the
    context-schema convention and the no-op-when-empty guarantee.
    """
    if not _WRITE_METADATA_CHECKS:
        return
    ctx: Mapping[str, Any] = {} if context is None else context
    # Snapshot for the same reason as the read hook above.
    for check in tuple(_WRITE_METADATA_CHECKS):
        check(ctx)


# ---------------------------------------------------------------------------
# Conflicting crs / crs_wkt write check
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
    * Either attr unparseable -> no-op (the
      :class:`UnparseableCRSError` check refuses those separately).

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
# Conflicting nodata aliases write check
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
# Distinct per-band nodatavals write check
# ---------------------------------------------------------------------------


def _check_write_distinct_per_band_nodatavals(
    context: Mapping[str, Any],
) -> None:
    """Refuse writes whose ``attrs['nodatavals']`` declares two or more
    distinct concrete per-band sentinels.

    The TIFF / GDAL data model stores a single ``GDAL_NODATA`` tag per
    file, so a multi-band raster shares one sentinel across every band
    (see ``tests/parity/test_reference.py::TestMultibandUint16SharedNodata``
    for the documented limitation). The legacy resolver in
    ``_attrs._resolve_nodata_attr`` would silently pick the first usable
    value and drop the rest, so bands that declared a different sentinel
    would keep their sentinel cells as real data on the resulting file.

    The earlier conflict check ``_check_write_conflicting_nodata`` only
    catches the case where every entry in the tuple disagrees with the
    canonical ``attrs['nodata']`` scalar. It misses the more dangerous
    case where the tuple itself carries multiple distinct concrete
    values (with or without an accompanying ``attrs['nodata']``), because
    one of the per-band values happens to equal the flattened scalar.

    Allowed:

    * A tuple with a single distinct concrete value (possibly repeated).
    * A tuple of all NaN / None entries.
    * A tuple mixing ``None`` with a single concrete value.

    Refused:

    * Two or more distinct concrete (non-NaN, non-None) values, e.g.
      ``(-9999.0, -8888.0)`` or ``(-9999.0, -9999.0, 0.0)``.

    The explicit ``nodata=`` writer kwarg overrides attrs entirely and
    short-circuits this check; the caller has stated which single
    sentinel to write, so the per-band disagreement is irrelevant.

    Context keys consumed:

    * ``nodata_kwarg`` -- the explicit ``nodata=`` writer kwarg.
    * ``attrs_nodatavals`` -- ``data.attrs.get('nodatavals')``.
    """
    if context.get('nodata_kwarg') is not None:
        return
    nodatavals_attr = context.get('attrs_nodatavals')
    if nodatavals_attr is None:
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
    # Deduplicate by exact float equality. The TIFF GDAL_NODATA tag is a
    # single scalar, so any two concrete values that compare unequal are
    # a collision the writer cannot represent.
    distinct = []
    for fv in concrete:
        if not any(fv == d for d in distinct):
            distinct.append(fv)
    if len(distinct) < 2:
        return
    raise ConflictingNodataError(
        _distinct_per_band_nodatavals_msg(nodatavals_attr, distinct)
    )


register_write_metadata_check(_check_write_distinct_per_band_nodatavals)


# ---------------------------------------------------------------------------
# Non-uniform coords write check
# ---------------------------------------------------------------------------


def _check_write_non_uniform_coords(context: Mapping[str, Any]) -> None:
    """Refuse writes whose ``coords`` imply a non-uniform pixel grid.

    The writer treats the first two y / x coord values as the pixel
    size and emits a uniform-grid ``GeoTransform``. When the rest of
    the coords disagree with that step (variable cell size, gaps),
    the on-disk file silently misrepresents the geometry.

    Exemption: arrays carrying the no-georef marker
    (``attrs[_NO_GEOREF_KEY] is True``, stamped by the reader on
    files without GeoTIFF transform tags) skip the check. Their
    coords are the placeholder pixel-index fallback rather than
    projected positions, so uniformity is meaningless for them.
    The exemption keys off the marker, not coord dtype: a user-authored
    int-coord grid no longer earns the exemption just by having an
    integer dtype, only by carrying the marker.

    Context keys consumed:

    * ``coord_y`` -- ``data.coords['y'].values`` (or equivalent).
    * ``coord_x`` -- ``data.coords['x'].values``.
    * ``no_georef_marker`` -- ``data.attrs[_NO_GEOREF_KEY] is True``.

    Missing or non-numeric coords are out of scope (handled by other
    writer validation upstream).
    """
    if context.get('no_georef_marker') is True:
        # Arrays the reader stamped as no-georef carry the 0..N-1
        # placeholder coords. Uniformity is not meaningful there;
        # the writer skips transform synthesis entirely.
        return
    for axis_name in ('y', 'x'):
        coord = context.get(f'coord_{axis_name}')
        if coord is None:
            continue
        arr = np.asarray(coord)
        if arr.size < 3:
            # Need at least 3 samples to detect non-uniformity.
            continue
        if not (
            np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)
        ):
            continue
        diffs = np.diff(arr.astype(np.float64))
        # Use a relative tolerance pegged to the step magnitude so that
        # float32 rounding noise (~1e-7 relative) does not trip the
        # check. Same tolerance the existing coord-regularity check uses.
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
# Unparseable CRS read check
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
# Rotated transform read check
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
        f"(``reader.allow_rotated`` is the experimental tier opt-in; "
        f"see ``docs/source/reference/release_gate_geotiff.rst`` for "
        f"the release-contract row; issue #1987)."
    )


register_read_metadata_check(_check_read_rotated_transform)


def _gdal_geotransform_to_affine_tuple(gt) -> tuple | None:
    """Reorder a GDAL ``GeoTransform`` 6-tuple into rasterio ``Affine``
    layout for ``_check_read_rotated_transform``.

    GDAL:    ``(origin_x, pixel_width, b, origin_y, d, pixel_height)``
    rasterio:``(pixel_width, b, origin_x, d, pixel_height, origin_y)``

    Returns ``None`` when ``gt`` is ``None`` so callers can pass the
    result straight into a validation context dict.
    """
    if gt is None:
        return None
    return (gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


# ---------------------------------------------------------------------------
# Mixed band metadata read check
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


# Callers that still want the legacy flatten-to-first-band behaviour
# pass ``band_nodata='first'`` to ``_read_vrt`` / ``open_geotiff`` /
# ``_read_geotiff_dask``; the explicit opt-in surfaces the per-band
# ambiguity at the call site instead of papering over it silently.
register_read_metadata_check(_check_read_mixed_band_metadata)


# ---------------------------------------------------------------------------
# Inconsistent GeoKey read check
# ---------------------------------------------------------------------------


# Mirrors the GeoTIFF spec ModelType enum values. Re-declared here so the
# validator does not pull a runtime import from ``_geotags`` (which would
# otherwise create a circular dep at module load time -- ``_geotags``
# already lives below ``_validation`` in the import order).
_MODEL_TYPE_UNDEFINED = 0
_MODEL_TYPE_PROJECTED = 1
_MODEL_TYPE_GEOGRAPHIC = 2
_MODEL_TYPE_GEOCENTRIC = 3
_GEOKEY_USER_DEFINED = 32767


def _check_read_inconsistent_geokeys(context: Mapping[str, Any]) -> None:
    """Refuse reads whose GeoKey shape is internally contradictory.

    The legacy reader took ``ProjectedCSTypeGeoKey`` first, fell back to
    ``GeographicTypeGeoKey``, and never cross-checked either against
    ``ModelTypeGeoKey``. A TIFF declaring::

        ModelTypeGeoKey       = 2 (geographic)
        ProjectedCSTypeGeoKey = 4326
        GeographicTypeGeoKey  = absent

    is internally contradictory -- 4326 is a geographic CRS but appears
    under the projected key while ``ModelTypeGeoKey`` declares the
    model is geographic. The legacy path published the EPSG verbatim
    in ``attrs['crs']`` and synthesised a matching ``attrs['crs_wkt']``,
    handing downstream callers trustworthy-looking spatial metadata
    fabricated from inconsistent inputs.

    The check rejects two structurally bad combinations:

    1. ``ModelTypeGeoKey = projected`` with ``GeographicTypeGeoKey``
       populated but ``ProjectedCSTypeGeoKey`` absent.
    2. ``ModelTypeGeoKey = geographic`` with ``ProjectedCSTypeGeoKey``
       populated (the geographic model may not stash a projected code).

    A file that populates *both* ``ProjectedCSTypeGeoKey`` and
    ``GeographicTypeGeoKey`` with different EPSG codes is **not**
    rejected: that is the normal shape of a projected GeoTIFF, where
    ``GeographicTypeGeoKey`` names the base geographic CRS the
    projection is built on and the pixel coordinates live in the
    projected CRS. The reader already takes the projected code first
    (see ``extract_geo_info``), so this case resolves correctly.

    ``32767`` (user-defined) is treated as "no resolved code" because
    the actual CRS is defined by sibling keys (``GeogGeodeticDatumGeoKey``
    et al.) rather than the type-code slot itself.

    ModelType enum values per the GeoTIFF spec:

    * ``_MODEL_TYPE_UNDEFINED`` (0): no ModelTypeGeoKey present.
    * ``_MODEL_TYPE_PROJECTED`` (1): projected CRS.
    * ``_MODEL_TYPE_GEOGRAPHIC`` (2): geographic CRS.
    * ``_MODEL_TYPE_GEOCENTRIC`` (3): geocentric CRS (not used by
      xrspatial readers today; passes through with no contradiction
      against either type-code slot).

    Context keys consumed:

    * ``model_type`` -- value of ``ModelTypeGeoKey`` (0 if absent).
    * ``projected_cs_type`` -- value of ``ProjectedCSTypeGeoKey`` (None
      if absent).
    * ``geographic_type`` -- value of ``GeographicTypeGeoKey`` (None
      if absent).

    All context keys are optional. With no GeoKey context the check is
    a no-op, so callers that did not (or could not) parse a GeoKey
    directory at all still pass through unchanged.
    """
    proj_cs = context.get('projected_cs_type')
    geog = context.get('geographic_type')
    model_type = context.get('model_type')
    if model_type is None and proj_cs is None and geog is None:
        return

    # Coerce ints. The reader stashes raw GeoKey ints, but a caller could
    # plausibly pass floats; tolerate both and ignore the rest. NaN and
    # inf would make ``int()`` raise (ValueError / OverflowError) on a
    # caller hand-building a garbage context dict; catch them and treat
    # as "not declared" rather than crash the validator.
    def _as_int(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except (ValueError, OverflowError):
                return None
        return None

    model_type_i = _as_int(model_type)
    proj_cs_i = _as_int(proj_cs)
    geog_i = _as_int(geog)

    # ``32767`` (user-defined) is treated as "no resolved code" for
    # consistency checks; the EPSG slot is intentionally a placeholder
    # in that case.
    proj_resolved = (
        proj_cs_i is not None and proj_cs_i != _GEOKEY_USER_DEFINED)
    geog_resolved = (
        geog_i is not None and geog_i != _GEOKEY_USER_DEFINED)

    # Case 1: ModelType = projected but only GeographicTypeGeoKey set.
    if (model_type_i == _MODEL_TYPE_PROJECTED
            and not proj_resolved and geog_resolved):
        raise InconsistentGeoKeysError(
            f"GeoTIFF source declares ModelTypeGeoKey=projected (1) "
            f"but carries only GeographicTypeGeoKey={geog_i!r} with no "
            f"resolved ProjectedCSTypeGeoKey. A projected model must "
            f"declare the projected CRS code in ProjectedCSTypeGeoKey; "
            f"the geographic key on its own cannot identify the "
            f"projection. Fix the source's GeoKey directory so the "
            f"projected CRS code is present. See issue #2417."
        )

    # Case 2: ModelType = geographic but ProjectedCSTypeGeoKey set.
    if model_type_i == _MODEL_TYPE_GEOGRAPHIC and proj_resolved:
        raise InconsistentGeoKeysError(
            f"GeoTIFF source declares ModelTypeGeoKey=geographic (2) "
            f"but populates ProjectedCSTypeGeoKey={proj_cs_i!r}. The "
            f"geographic model may not carry a projected CRS code; the "
            f"legacy reader would publish EPSG {proj_cs_i!r} verbatim "
            f"in attrs['crs'] from contradictory inputs. Fix the "
            f"source's GeoKey directory so ModelTypeGeoKey matches the "
            f"CRS it declares. See issue #2417."
        )


register_read_metadata_check(_check_read_inconsistent_geokeys)
