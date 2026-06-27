"""xarray accessors for xarray-spatial.

Registers ``.xrs`` accessors on :class:`xarray.DataArray` and
:class:`xarray.Dataset` so that spatial operations can be called
directly via tab-completion::

    import xrspatial  # registers accessors

    raster.xrs.slope()
    raster.xrs.hillshade(azimuth=315)
    nir.xrs.ndvi(red)
"""

import functools
import html
import inspect
import re
import textwrap
import warnings

import xarray as xr


def _require_matplotlib():
    """Import matplotlib or raise a helpful error.

    matplotlib is an optional dependency (the ``plot`` extra). The
    plotting helpers call this before importing ``pyplot`` so a missing
    install produces a clear message instead of a bare
    ``ModuleNotFoundError``. Importing ``pyplot`` (not just the top-level
    package) also catches the case where matplotlib is installed but has
    no usable backend.
    """
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting but is not installed. "
            "Install it with: pip install xarray-spatial[plot]"
        ) from e


def _listed_colormap_from_attrs(attrs):
    """Build a :class:`matplotlib.colors.ListedColormap` from
    ``attrs['colormap']`` (raw uint16 RGB triples from TIFF tag 320).

    Returns ``None`` when ``attrs`` carries no colormap, when the
    layout does not match the TIFF ColorMap spec, or when matplotlib
    is not importable. The two ``.xrs.plot`` paths fall back to the
    default plot when this returns ``None``.

    Contract v2 (issue #2016) removed the ``attrs['cmap']`` shortcut
    that older xrspatial releases set on palette reads. Callers and
    the accessor build the colormap on the fly from the canonical
    ``attrs['colormap']``.
    """
    raw = attrs.get('colormap')
    if raw is None:
        return None
    try:
        from matplotlib.colors import ListedColormap
    except ImportError:
        return None
    total = len(raw)
    if total == 0 or total % 3 != 0:
        return None
    n_colors = total // 3
    colors = []
    for i in range(n_colors):
        r = raw[i] / 65535.0
        g = raw[n_colors + i] / 65535.0
        b = raw[2 * n_colors + i] / 65535.0
        colors.append((r, g, b, 1.0))
    return ListedColormap(colors, name='tiff_palette')


def _pick_representative_dataarray(ds, *, var=None):
    """Return the DataArray used to drive backend / CRS lookup for
    Dataset-level GeoTIFF accessor methods.

    ``var`` -> ``ds[var]``. Otherwise the first 2D ``y``/``x`` variable.
    """
    if var is not None:
        return ds[var]
    for v in ds.data_vars:
        d = ds[v]
        if d.ndim >= 2 and 'y' in d.dims and 'x' in d.dims:
            return d
    raise ValueError(
        "Dataset has no 2D variable with 'y' and 'x' dimensions to use "
        "for backend inference. Pass var='<name>' to select one, or "
        "call xrspatial.geotiff.open_geotiff(source, ...) directly."
    )


def _infer_caller_chunk(obj, axis):
    """First chunk size along *axis* of a dask-backed DataArray, or None."""
    try:
        axis_num = obj.get_axis_num(axis)
    except (ValueError, KeyError):
        return None
    chunks_tuple = getattr(obj, 'chunks', None)
    if not chunks_tuple:
        return None
    try:
        axis_chunks = chunks_tuple[axis_num]
    except (IndexError, TypeError):
        return None
    if not axis_chunks:
        return None
    return int(axis_chunks[0])


def _to_pyproj_crs(crs):
    """Normalize a CRS value (int EPSG, ``"EPSG:xxxx"``, WKT, PROJ
    string, or ``pyproj.CRS``) into a ``pyproj.CRS`` instance.

    Returns ``None`` only when ``crs`` is itself ``None`` (the
    backward-compatible "no CRS to compare" path). A malformed but
    present value raises ``ValueError`` so the caller sees the typo
    instead of having the mismatch safety net silently disabled.
    """
    if crs is None:
        return None
    from pyproj import CRS as _PyprojCRS
    from pyproj.exceptions import CRSError
    try:
        return _PyprojCRS(crs)
    except CRSError as e:
        raise ValueError(
            f"attrs['crs']={crs!r} is not a valid CRS: {e}"
        ) from e


def _bbox_edge_samples(x_min, y_min, x_max, y_max, n_per_side=20):
    """Return ``(xs, ys)`` arrays sampling each edge of the bbox.

    Sampling the perimeter (instead of only the 4 corners) before
    transforming into the source CRS keeps the envelope of the
    transformed points close to the true projected bbox even when the
    transform has curvature across the box (high latitudes, large
    extents).
    """
    import numpy as np
    a = np.linspace(x_min, x_max, n_per_side)
    b = np.full(n_per_side, y_min)
    c = np.full(n_per_side, y_max)
    d = np.linspace(y_min, y_max, n_per_side)
    e = np.full(n_per_side, x_min)
    f = np.full(n_per_side, x_max)
    xs = np.concatenate([a, a, e, f])
    ys = np.concatenate([b, c, d, d])
    return xs, ys


_RESAMPLING_CHOICES = ('auto', 'nearest', 'bilinear', 'cubic')

# coregister=True paints the file onto the caller's full grid, NaN-filling
# everything outside the file footprint. Below this coverage fraction the
# result is mostly NaN, so warn and point at the crop-the-caller-first
# pattern instead.
_COREGISTER_COVERAGE_WARN_FRACTION = 0.10


def _resolve_resampling(resampling, result):
    """Resolve an ``'auto'`` resampling choice for the reproject step.

    ``'auto'`` picks ``'nearest'`` for categorical rasters (a paletted
    colormap is present, or the data has an integer dtype) and
    ``'bilinear'`` otherwise. Any explicit mode is returned unchanged
    for :func:`xrspatial.reproject.reproject` to validate.
    """
    if resampling != 'auto':
        return resampling
    import numpy as np
    has_colormap = result.attrs.get('colormap') is not None
    if has_colormap:
        return 'nearest'
    if np.issubdtype(result.dtype, np.integer):
        # An integer dtype alone does not prove categorical data: an
        # int16 DEM is continuous, and 'nearest' silently costs it the
        # interpolation quality. Say what was picked and how to choose.
        warnings.warn(
            "resampling='auto' chose 'nearest' because the data has an "
            f"integer dtype ({result.dtype}) and no colormap. If this "
            "raster is continuous (e.g. a DEM), pass "
            "resampling='bilinear'; pass resampling='nearest' to keep "
            "this choice and silence the warning.",
            UserWarning, stacklevel=4,
        )
        return 'nearest'
    return 'bilinear'


def _axis_res(coord, n, obj, axis):
    """Resolution (cell size) for one axis of the caller's grid.

    With 2+ cells it is the centre spacing. A single-cell axis cannot
    give a spacing, so fall back to the caller's ``attrs['res']`` or
    ``attrs['transform']`` pixel size, and raise if neither is present.
    """
    if n >= 2:
        return abs(float(coord[-1]) - float(coord[0])) / (n - 1)
    res = obj.attrs.get('res')
    if res is not None:
        return abs(float(res[0] if axis == 'x' else res[1]))
    transform = obj.attrs.get('transform')
    if transform is not None:
        return abs(float(transform[0] if axis == 'x' else transform[4]))
    raise ValueError(
        f"coregister cannot infer {axis} resolution for a single-cell "
        f"template; use a template with 2+ cells along {axis}, or set "
        f"attrs['res'] / attrs['transform']"
    )


def _validate_template_grid(obj):
    """Reject caller templates ``_caller_grid`` cannot reproduce.

    The coregister output grid is derived from the template's first /
    last coords and a uniform spacing, in reproject's output convention
    (``x`` ascending, ``y`` descending). An irregular or differently
    oriented template would come back silently misaligned, so refuse it
    and name the offending axis.
    """
    import numpy as np
    for axis, direction in (('x', 'ascending'), ('y', 'descending')):
        coord = np.asarray(obj.coords[axis].values, dtype='float64')
        if coord.size < 2:
            continue
        diffs = np.diff(coord)
        sign_ok = (
            (diffs > 0).all() if direction == 'ascending'
            else (diffs < 0).all()
        )
        if not sign_ok:
            if direction == 'descending':
                hint = f".sortby('{axis}', ascending=False)"
            else:
                hint = f".sortby('{axis}')"
            raise ValueError(
                f"coregister=True needs the caller's {axis} coords in "
                f"{direction} order to line up cell-for-cell with "
                f"reproject's output; sort the caller along {axis} "
                f"first (e.g. {hint})."
            )
        if coord.size >= 3:
            steps = np.abs(diffs)
            mean = steps.mean()
            if (np.abs(steps - mean) > 1e-6 * mean).any():
                raise ValueError(
                    f"coregister=True needs a regular caller grid, but "
                    f"the {axis} coords are not evenly spaced (max "
                    f"spacing {steps.max():g}, min {steps.min():g}). "
                    "Resample the caller onto a regular grid first."
                )


def _select_overview_level(source, base_transform, res_x, res_y):
    """Pick the coarsest overview still finer than the caller's grid.

    ``res_x`` / ``res_y`` are the caller's cell sizes in file-CRS units.
    Probes overview levels with header-only reads and returns the
    highest level whose pixel size stays <= the caller's on both axes,
    or ``None`` when the full-resolution read is the right choice (no
    overviews, or the caller grid is too fine to benefit). Levels are
    probed in order, so a missing level ends the search.

    Each probe re-parses the header: sub-millisecond mmap work for
    local paths, one bounded range fetch per level for fsspec / HTTP
    sources. Real pyramids top out around 8 levels, and the probes
    replace full-resolution tile fetches that can run to gigabytes on
    remote COGs, so the trade holds on both source types.
    """
    from .geotiff import _read_geo_info

    # A level-1 overview is conventionally ~2x the base pixel size; skip
    # the probing reads entirely when even that could not stay finer
    # than the caller's grid.
    if (abs(base_transform.pixel_width) * 2 > res_x
            or abs(base_transform.pixel_height) * 2 > res_y):
        return None
    selected = None
    level = 1
    while True:
        try:
            geo_info, _h, _w, _dtype, _nbands = _read_geo_info(
                source, overview_level=level)
        except ValueError:
            # out of range: no more pyramid levels
            break
        t = geo_info.transform
        if (abs(t.pixel_width) > res_x or abs(t.pixel_height) > res_y):
            break
        selected = level
        level += 1
    return selected


def _caller_grid(obj):
    """Edge bounds + shape that make :func:`reproject` reproduce the
    caller's exact pixel centres.

    Returns ``((left, bottom, right, top), width, height)``. ``reproject``
    emits pixel-centre coords ``linspace(left+res/2, right-res/2, W)``, so
    bounds offset half a pixel outside the caller's centre extent map the
    output cells back onto the caller's coordinates.

    Assumes a regular grid with ``x`` ascending and ``y`` descending (the
    convention ``_open_geotiff_windowed`` already works in); resolution is
    the centre spacing. An irregular or descending-``x`` template would
    not line up cell-for-cell.
    """
    y = obj.coords['y'].values
    x = obj.coords['x'].values
    width, height = len(x), len(y)
    res_x = _axis_res(x, width, obj, 'x')
    res_y = _axis_res(y, height, obj, 'y')
    x_lo, x_hi = float(min(x[0], x[-1])), float(max(x[0], x[-1]))
    y_lo, y_hi = float(min(y[0], y[-1])), float(max(y[0], y[-1]))
    bounds = (x_lo - res_x / 2, y_lo - res_y / 2,
              x_hi + res_x / 2, y_hi + res_y / 2)
    return bounds, width, height


def _open_geotiff_windowed(obj, source, *, auto_reproject=False,
                           coregister=False, resampling='auto', **kwargs):
    """Shared implementation for ``.xrs.open_geotiff`` on DataArray and
    Dataset.

    ``obj`` is a DataArray that carries the bbox (``y``/``x`` coords),
    the CRS (``attrs['crs']``), and the backend used to infer
    ``gpu=`` / ``chunks=`` kwargs for the underlying read.

    The windowing extent is expanded by half a pixel on each side so
    edge pixels of the caller's footprint are captured. CRS values on
    both sides (``attrs['crs']`` and the file's georeference) are
    normalized through ``pyproj.CRS`` for equality, so int EPSG codes,
    ``"EPSG:xxxx"`` strings, WKT strings, and ``pyproj.CRS`` instances
    all compare correctly.

    When ``auto_reproject=True`` and the CRSs differ, the caller's
    bbox is sampled along its perimeter (not just the four corners)
    before transformation into the file CRS, so the windowed read
    covers the full footprint even when the transform has curvature
    across the bbox.
    """
    from .geotiff import (open_geotiff, _read_geo_info, _extent_to_window,
                          _validate_chunks_arg)
    from .utils import _classify_backend

    if resampling not in _RESAMPLING_CHOICES:
        raise ValueError(
            f"resampling must be one of {list(_RESAMPLING_CHOICES)}, "
            f"got {resampling!r}"
        )

    if coregister:
        gpu_requested = (
            kwargs.get('gpu', False)
            or _classify_backend(obj) in ("cupy", "dask+cupy")
        )
        is_vrt = str(source).lower().endswith('.vrt')
        if gpu_requested or is_vrt:
            raise ValueError(
                "coregister=True runs an unpack-and-reproject read that "
                "runs on CPU only and is not supported with gpu=True or "
                ".vrt sources. Read on CPU, or pass coregister=False."
            )
        if kwargs.get('allow_rotated'):
            raise ValueError(
                "coregister=True is not supported with allow_rotated="
                "True: a rotated file reads as an ungeoreferenced pixel "
                "grid, so reproject would treat its row/col indices as "
                "map coordinates and mis-place every pixel. Read the "
                "file without coregister and resample it explicitly."
            )
        # coregister implies an unpacked (scaled + masked) read; this
        # overrides an explicit unpack=False.
        kwargs['unpack'] = True

    if 'y' not in obj.coords or 'x' not in obj.coords:
        raise ValueError(
            "Caller must have 'y' and 'x' coordinates to compute a "
            "spatial window"
        )
    if coregister:
        _validate_template_grid(obj)
    y = obj.coords['y'].values
    x = obj.coords['x'].values
    y_min, y_max = float(y.min()), float(y.max())
    x_min, x_max = float(x.min()), float(x.max())

    geo_info, file_h, file_w, _dtype, _nbands = _read_geo_info(source)
    t = geo_info.transform
    caller_crs_raw = obj.attrs.get('crs')
    file_crs_raw = geo_info.crs_epsg
    caller_crs = _to_pyproj_crs(caller_crs_raw)
    file_crs = _to_pyproj_crs(file_crs_raw)

    crs_mismatch = (
        caller_crs is not None
        and file_crs is not None
        and not caller_crs.equals(file_crs)
    )

    if crs_mismatch and not (auto_reproject or coregister):
        raise ValueError(
            f"CRS mismatch: caller has {caller_crs_raw!r} but file has "
            f"EPSG:{file_crs_raw}. Pass auto_reproject=True to project "
            f"the caller bbox into the file CRS for the windowed read "
            f"and reproject the result back to the caller CRS."
        )

    crosses_antimeridian = False
    if crs_mismatch:
        import numpy as np
        from pyproj import Transformer
        transformer = Transformer.from_crs(
            caller_crs, file_crs, always_xy=True
        )
        # 20 samples per side is enough to keep the projected envelope
        # within sub-pixel of the true bbox at any realistic latitude
        # without making the transform call noticeably slower.
        xs, ys = _bbox_edge_samples(x_min, y_min, x_max, y_max)
        px, py = transformer.transform(xs, ys)
        x_min = float(px.min())
        x_max = float(px.max())
        y_min = float(py.min())
        y_max = float(py.max())
        # A caller grid straddling the antimeridian lands as two
        # longitude clusters near +180 and -180: a gap wider than 180
        # degrees between consecutive sorted samples is the signature
        # (a genuinely global caller fills the range without such a
        # gap). min/max windowing then covers the full file width --
        # still correct, but a much larger read than the footprint.
        if file_crs.is_geographic:
            gaps = np.diff(np.sort(np.asarray(px, dtype='float64')))
            if gaps.size and float(gaps.max()) > 180.0:
                crosses_antimeridian = True
                warnings.warn(
                    "the caller grid appears to cross the antimeridian "
                    f"in {source!r}'s geographic CRS; the windowed read "
                    "falls back to the full file width, which is "
                    "correct but may be slow. Consider splitting the "
                    "analysis at the dateline or reprojecting the file "
                    "into the caller's CRS.",
                    UserWarning, stacklevel=3,
                )

    # open_geotiff interprets window= in the pixel space of the selected
    # overview level, so an explicit overview_level= means the window
    # must be computed against that level's transform and dimensions,
    # not the full-resolution ones read above.
    overview_arg = kwargs.get('overview_level')
    if overview_arg not in (None, 0):
        geo_info, file_h, file_w, _dtype, _nbands = _read_geo_info(
            source, overview_level=overview_arg)
        t = geo_info.transform
    elif (coregister and overview_arg is None and isinstance(source, str)
          and not crosses_antimeridian):
        # The caller's grid fixes the output resolution, so decoding
        # file pixels finer than that grid is wasted work: read the
        # coarsest overview that still meets the caller's resolution.
        # Pass overview_level=0 to force the full-resolution read.
        # (Skipped across the antimeridian: the wrapped bbox span would
        # grossly overstate the caller's cell size.)
        res_x_caller = (x_max - x_min) / len(x)
        res_y_caller = (y_max - y_min) / len(y)
        level = _select_overview_level(source, t, res_x_caller,
                                       res_y_caller)
        if level is not None:
            kwargs['overview_level'] = level
            geo_info, file_h, file_w, _dtype, _nbands = _read_geo_info(
                source, overview_level=level)
            t = geo_info.transform

    # Expand extent by half a pixel so we capture edge pixels
    y_min -= abs(t.pixel_height) * 0.5
    y_max += abs(t.pixel_height) * 0.5
    x_min -= abs(t.pixel_width) * 0.5
    x_max += abs(t.pixel_width) * 0.5

    window = _extent_to_window(t, file_h, file_w,
                               y_min, y_max, x_min, x_max)
    kwargs.pop('window', None)

    if coregister:
        # Both extents are in file pixel units here: the caller bbox was
        # transformed into the file CRS above when the two differ, and
        # ``window`` is that bbox clamped to the file bounds, so the area
        # ratio is the fraction of the caller grid the file can fill.
        row_start, col_start, row_stop, col_stop = window
        window_area = (max(0, row_stop - row_start)
                       * max(0, col_stop - col_start))
        req_rows = (y_max - y_min) / abs(t.pixel_height)
        req_cols = (x_max - x_min) / abs(t.pixel_width)
        coverage = window_area / (req_rows * req_cols)
        if coverage == 0:
            raise ValueError(
                f"{source!r} does not overlap the caller grid; the "
                "coregistered result would be entirely NaN. Check the "
                "caller's coords / attrs['crs'] against the file's "
                "georeferencing."
            )
        if 0 < coverage < _COREGISTER_COVERAGE_WARN_FRACTION:
            warnings.warn(
                f"{source!r} covers only {coverage:.1%} of the caller "
                "grid; the coregistered result will be mostly NaN. "
                "Consider opening the small raster first and cropping "
                "the caller to its bounds before coregistering.",
                UserWarning, stacklevel=3,
            )

    # Infer backend kwargs. Caller-supplied values always win.
    backend = _classify_backend(obj)
    # Coerce here because reproject's chunk layout only unpacks builtin
    # ints, while the read path accepts np.integer scalars.
    explicit_chunks = _validate_chunks_arg(
        kwargs.get('chunks'), allow_none=True
    )
    if backend in ("cupy", "dask+cupy"):
        kwargs.setdefault('gpu', True)
    if backend in ("dask+numpy", "dask+cupy"):
        inferred_chunk = _infer_caller_chunk(obj, 'y')
        if inferred_chunk is not None:
            kwargs.setdefault('chunks', inferred_chunk)

    result = open_geotiff(source, window=window, **kwargs)

    # reproject defaults its dask output chunks to 512x512; forward the
    # explicit ``chunks=`` kwarg, or failing that the caller's chunk
    # layout, so a coregistered / auto-reprojected result keeps the
    # chunking of the array it is meant to align with (#3234).
    caller_y_chunk = _infer_caller_chunk(obj, 'y')
    caller_x_chunk = _infer_caller_chunk(obj, 'x')
    if explicit_chunks is not None:
        reproject_chunk_size = explicit_chunks
    elif caller_y_chunk is not None and caller_x_chunk is not None:
        reproject_chunk_size = (caller_y_chunk, caller_x_chunk)
    else:
        reproject_chunk_size = None

    # A masked read replaces the nodata sentinel with NaN but leaves
    # attrs['nodata'] as the raw sentinel; tell reproject the holes are
    # NaN so it does not resample them as ordinary values.
    nodata_arg = float('nan') if result.attrs.get('masked_nodata') else None

    if coregister:
        from .reproject import reproject
        target_crs = caller_crs if caller_crs is not None else file_crs
        if target_crs is None:
            raise ValueError(
                "coregister=True needs a CRS: set attrs['crs'] on the "
                "caller or read a file that declares one"
            )
        # A file that declares no CRS is treated as already in the
        # caller's CRS, so coregister stays a pure resample instead of
        # failing reproject's source-CRS detection.
        source_crs = file_crs if file_crs is not None else target_crs
        bounds, width_out, height_out = _caller_grid(obj)
        result = reproject(
            result,
            target_crs=target_crs,
            source_crs=source_crs,
            bounds=bounds,
            width=width_out,
            height=height_out,
            resampling=_resolve_resampling(resampling, result),
            nodata=nodata_arg,
            chunk_size=reproject_chunk_size,
        )
    elif crs_mismatch:
        from .reproject import reproject
        result = reproject(
            result,
            target_crs=caller_crs,
            source_crs=file_crs,
            resampling=_resolve_resampling(resampling, result),
            nodata=nodata_arg,
            chunk_size=reproject_chunk_size,
        )

    return result


# ---------------------------------------------------------------------------
# Accessor repr: list the available ``.xrs`` operations grouped by category so
# users can discover the toolset from a plain REPL or notebook, without relying
# on tab-completion or an LSP (issue #3476).
# ---------------------------------------------------------------------------

# A category banner in the accessor source, e.g. ``    # ---- Hydrology ----``.
_REPR_BANNER_RE = re.compile(r'^\s*#\s*----\s*(.+?)\s*----\s*$')
# A method definition inside a class body (indented ``def name(``).
_REPR_METHOD_RE = re.compile(r'^\s+def\s+([A-Za-z][A-Za-z0-9_]*)\s*\(')


@functools.lru_cache(maxsize=None)
def _accessor_categories(cls):
    """Parse ``# ---- Category ----`` banners out of *cls* source.

    Returns an ordered tuple of ``(category, (method_name, ...))`` pairs,
    one per banner that has at least one public method beneath it. Returns
    an empty tuple when the source is unavailable (e.g. a frozen build), so
    callers fall back to a flat listing.
    """
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return ()
    categories = []
    current = None
    for line in source.splitlines():
        banner = _REPR_BANNER_RE.match(line)
        if banner:
            current = (banner.group(1), [])
            categories.append(current)
            continue
        method = _REPR_METHOD_RE.match(line)
        if method and current is not None and not method.group(1).startswith('_'):
            current[1].append(method.group(1))
    return tuple(
        (name, tuple(methods)) for name, methods in categories if methods
    )


def _public_accessor_methods(cls):
    """Sorted public method names defined directly on *cls*."""
    return tuple(sorted(
        name for name, member in vars(cls).items()
        if callable(member) and not name.startswith('_')
    ))


def _accessor_kind(cls):
    return 'Dataset' if cls.__name__.endswith('DatasetAccessor') else 'DataArray'


# Single category used by both reprs when the source banners can't be parsed.
_FALLBACK_CATEGORY = 'Available tools'


def _categories_or_fallback(cls):
    """Parsed categories, or one flat ``Available tools`` group as a fallback."""
    categories = _accessor_categories(cls)
    if categories:
        return categories
    return ((_FALLBACK_CATEGORY, _public_accessor_methods(cls)),)


def _accessor_repr_text(cls):
    """Plain-text repr listing the accessor's operations by category."""
    lines = [
        f"<{cls.__name__}> xarray-spatial tools for this {_accessor_kind(cls)}",
        "call as: .xrs.<name>(...)",
        "",
    ]
    for name, methods in _categories_or_fallback(cls):
        lines.append(f"{name}:")
        for chunk in textwrap.wrap(
            ", ".join(methods), width=74,
            break_long_words=False, break_on_hyphens=False,
        ):
            lines.append(f"    {chunk}")
    return "\n".join(lines)


def _accessor_repr_html(cls):
    """Jupyter HTML repr: the same listing as a compact table."""
    rows = []
    for name, methods in _categories_or_fallback(cls):
        methods_html = ", ".join(
            f"<code>{html.escape(m)}</code>" for m in methods
        )
        rows.append(
            "<tr>"
            "<td style='text-align:right;vertical-align:top;"
            "padding-right:0.75em;font-weight:bold;white-space:nowrap'>"
            f"{html.escape(name)}</td>"
            f"<td style='text-align:left'>{methods_html}</td>"
            "</tr>"
        )
    return (
        f"<div><strong>xarray-spatial</strong> tools for this "
        f"{_accessor_kind(cls)} &mdash; call as "
        "<code>.xrs.&lt;name&gt;(...)</code>"
        f"<table>{''.join(rows)}</table></div>"
    )


# ---------------------------------------------------------------------------
# Per-tool repr: evaluating a single tool (``da.xrs.slope``) without calling it
# should show that tool's own signature and docstring, not the whole accessor
# catalog. A bound method's repr is ``<bound method Cls.name of {repr(self)}>``,
# and ``repr(self)`` routes through the accessor's catalog ``__repr__``, so
# without intervention the full listing gets embedded (issue #3478). The
# accessor wraps public methods in this proxy on attribute access instead.
# ---------------------------------------------------------------------------


def _accessor_tool_repr_text(method):
    """Plain-text repr for a single accessor tool: signature + docstring."""
    name = method.__name__
    try:
        sig = str(inspect.signature(method))
    except (TypeError, ValueError):
        sig = "(...)"
    lines = [f".xrs.{name}{sig}"]
    doc = inspect.getdoc(method)
    if doc:
        lines += ["", doc]
    return "\n".join(lines)


def _accessor_tool_repr_html(method):
    """Jupyter HTML repr for a single tool: the text repr in a <pre> block."""
    return (
        "<pre style='white-space:pre-wrap'>"
        f"{html.escape(_accessor_tool_repr_text(method))}"
        "</pre>"
    )


class _AccessorTool:
    """Callable proxy returned for a public accessor method on attribute access.

    Forwards calls to the wrapped bound method unchanged, but renders its own
    signature and docstring when evaluated in a REPL or notebook instead of the
    bound-method repr (which would embed the whole accessor catalog via
    ``repr(self)``; issue #3478). ``__doc__`` / ``__wrapped__`` are mirrored so
    ``help()`` and ``inspect`` keep seeing the delegated documentation.
    """

    def __init__(self, method):
        self._method = method
        self.__wrapped__ = method
        self.__name__ = method.__name__
        self.__qualname__ = method.__qualname__
        self.__doc__ = method.__doc__

    def __call__(self, *args, **kwargs):
        return self._method(*args, **kwargs)

    def __repr__(self):
        return _accessor_tool_repr_text(self._method)

    def _repr_html_(self):
        return _accessor_tool_repr_html(self._method)


def _wrap_accessor_attr(accessor, name):
    """``__getattribute__`` body shared by both accessors.

    Returns public bound methods wrapped in :class:`_AccessorTool`; everything
    else (private attributes, the catalog ``__repr__``) is returned untouched.
    """
    attr = object.__getattribute__(accessor, name)
    if name.startswith("_") or not inspect.ismethod(attr):
        return attr
    return _AccessorTool(attr)


def _check_point_crs_match(src, target):
    """Raise if point CRS and caller CRS disagree.

    Both arguments are already-normalized ``pyproj.CRS`` instances (or
    ``None``).  A ``None`` on either side is a no-op, mirroring how
    ``rasterize`` treats a missing CRS.
    """
    if src is None or target is None:
        return
    if not src.equals(target):
        def _label(crs):
            epsg = crs.to_epsg()
            return f"EPSG:{epsg}" if epsg is not None else crs.name
        raise ValueError(
            f"CRS mismatch: points are in {_label(src)} but the caller is "
            f"in {_label(target)}. Pass coregister=True to reproject the "
            f"points into the caller CRS."
        )


def _interp_on_accessor(obj, func, points, rest, *, coregister, **kwargs):
    """Run interpolation *func* against caller raster *obj* as the template.

    *obj* supplies the output grid, chunks, CRS, and attrs (passed as
    ``template``).  *points* is the first argument to *func*: a GeoDataFrame,
    or raw ``x`` for the legacy array form, in which case *rest* carries the
    remaining positional coordinates (``y``, ``z``).  When *points* is a
    GeoDataFrame and ``coregister=True``, its geometries are reprojected into
    the caller's CRS before interpolation; otherwise a genuine CRS mismatch
    raises (matching ``rasterize``'s default).  ``coregister`` needs a CRS on
    both sides and only applies to GeoDataFrame input.
    """
    from xrspatial.interpolate._vector import is_geodataframe

    if is_geodataframe(points):
        target = _to_pyproj_crs(obj.attrs.get('crs'))
        src = _to_pyproj_crs(points.crs)
        if coregister:
            if src is None or target is None:
                raise ValueError(
                    "coregister=True needs a CRS on both the points "
                    "(GeoDataFrame.crs) and the caller (attrs['crs'])"
                )
            points = points.to_crs(target)
        else:
            _check_point_crs_match(src, target)
    elif coregister:
        raise ValueError(
            "coregister=True requires a GeoDataFrame input carrying a CRS"
        )
    return func(points, *rest, template=obj, **kwargs)


def _rasterize_on_accessor(obj, geometries, *, coregister, **kwargs):
    """Run ``rasterize`` against caller raster *obj* as the ``like`` template.

    *obj* supplies the output grid, chunks, CRS, and attrs (passed as
    ``like``).  When *geometries* is a GeoDataFrame and ``coregister=True``,
    its geometries are reprojected into the caller's CRS before rasterizing,
    the vector analog of ``open_geotiff(coregister=True)``.  With
    ``coregister=False`` (default) a genuine CRS mismatch raises through
    ``rasterize``'s own ``check_crs``.  ``coregister`` needs a CRS on both
    sides and only applies to GeoDataFrame input.
    """
    from .rasterize import rasterize, _like_crs
    from xrspatial.interpolate._vector import is_geodataframe

    if coregister:
        # is_geodataframe matches the interpolation accessor (#3481): plain
        # geopandas only, not dask_geopandas, even though standalone
        # rasterize._parse_input accepts dask_geopandas.  Reprojecting a
        # dask_geopandas frame under coregister is left for a follow-up.
        if not is_geodataframe(geometries):
            raise ValueError(
                "coregister=True requires a GeoDataFrame input carrying a CRS"
            )
        src = _to_pyproj_crs(geometries.crs)
        # _like_crs detects the caller CRS the same way rasterize's check_crs
        # does, so the reprojected frame compares equal to the template.
        target = _to_pyproj_crs(_like_crs(obj))
        if src is None or target is None:
            raise ValueError(
                "coregister=True needs a CRS on both the geometries "
                "(GeoDataFrame.crs) and the caller (its detected raster CRS)"
            )
        geometries = geometries.to_crs(target)
    return rasterize(geometries, like=obj, **kwargs)


def _coregister_on_accessor(obj, data, *, resampling, **kwargs):
    """Put *data* on caller raster *obj*'s grid (CRS + bounds + shape).

    Dispatches on the input type:

    - A GeoDataFrame is rasterized onto the grid, reprojected into the caller
      CRS first -- the same path as ``rasterize(coregister=True)``. The
      ``resampling`` argument does not apply to a burn and is ignored.
    - A raster ``DataArray`` is reprojected and resampled onto the grid.

    Either way the result shares *obj*'s ``y``/``x`` coordinates exactly, so it
    lines up cell-for-cell with the template and any other layer coregistered
    the same way.
    """
    from xrspatial.interpolate._vector import is_geodataframe

    if is_geodataframe(data):
        return _rasterize_on_accessor(obj, data, coregister=True, **kwargs)
    if isinstance(data, xr.DataArray):
        return _reproject_onto_accessor(obj, data, resampling=resampling,
                                        **kwargs)
    raise TypeError(
        "coregister expects a raster xarray.DataArray or a GeoDataFrame, "
        f"got {type(data).__name__}"
    )


def _reproject_onto_accessor(target, source, *, resampling, **kwargs):
    """Reproject raster *source* onto *target*'s exact grid."""
    import numpy as np

    from .reproject import reproject
    from .rasterize import _like_crs

    target_crs = _like_crs(target)
    if target_crs is None:
        raise ValueError(
            "coregister of a raster needs a CRS on the caller grid "
            "(its detected raster CRS, e.g. attrs['crs'])"
        )
    y = np.asarray(target['y'].values, dtype='float64')
    x = np.asarray(target['x'].values, dtype='float64')
    if y.size < 2 or x.size < 2:
        raise ValueError(
            "coregister needs a caller grid of at least 2x2 cells"
        )
    res = target.attrs.get('res')
    if res is not None:
        res_x, res_y = abs(float(res[0])), abs(float(res[1]))
    else:
        res_x = abs(float(x[1] - x[0]))
        res_y = abs(float(y[1] - y[0]))
    bounds = (float(x.min()) - res_x / 2.0, float(y.min()) - res_y / 2.0,
              float(x.max()) + res_x / 2.0, float(y.max()) + res_y / 2.0)
    out = reproject(source, target_crs, bounds=bounds,
                    width=x.size, height=y.size, resampling=resampling,
                    **kwargs)
    # reproject emits north-up (descending y, ascending x) regardless of the
    # source order. Match the caller's axis directions before snapping so a
    # grid stored in either order lines up by geography, not by row index.
    if x[0] > x[-1]:
        out = out.isel(x=slice(None, None, -1))
    if y[0] < y[-1]:
        out = out.isel(y=slice(None, None, -1))
    # Snap to the caller's exact coordinates and carry its CRS attrs so the
    # result is a cell-for-cell drop-in for the template grid.
    out = out.assign_coords(y=target['y'], x=target['x'])
    for k in ('crs', 'crs_wkt', 'grid_mapping_name'):
        if k in target.attrs:
            out.attrs[k] = target.attrs[k]
    return out


@xr.register_dataarray_accessor("xrs")
class XrsSpatialDataArrayAccessor:
    """DataArray accessor exposing xarray-spatial operations."""

    def __init__(self, obj):
        self._obj = obj

    def __getattribute__(self, name):
        return _wrap_accessor_attr(self, name)

    def __repr__(self):
        return _accessor_repr_text(type(self))

    def _repr_html_(self):
        return _accessor_repr_html(type(self))

    # ---- Plot ----

    def plot(self, **kwargs):
        """Plot the DataArray with helpful defaults.

        Computes dask arrays, applies embedded GeoTIFF colormaps,
        and sets equal aspect ratio.

        Parameters
        ----------
        **kwargs
            Passed to ``da.plot()``.  Common extras: ``cmap``,
            ``figsize``, ``ax``, ``add_colorbar``.

        Returns
        -------
        matplotlib artist (from ``da.plot()``)
        """
        _require_matplotlib()
        import matplotlib.pyplot as plt
        import numpy as np

        da = self._obj

        # Materialise dask arrays so matplotlib can render them.
        try:
            da = da.compute()
        except (AttributeError, TypeError):
            pass

        # Use embedded GeoTIFF colormap when present. Build the
        # ``ListedColormap`` on the fly from the canonical
        # ``attrs['colormap']``; the older ``attrs['cmap']`` shortcut
        # was removed by contract v2 (issue #2016).
        cmap = _listed_colormap_from_attrs(da.attrs)
        if cmap is not None and 'cmap' not in kwargs:
            from matplotlib.colors import BoundaryNorm
            n_colors = len(cmap.colors)
            boundaries = np.arange(n_colors + 1) - 0.5
            kwargs.setdefault('cmap', cmap)
            kwargs.setdefault('norm', BoundaryNorm(boundaries, n_colors))
            kwargs.setdefault('add_colorbar', True)

        # Create a figure with sensible size if none provided. Use
        # ``constrained`` layout so palette colorbars and titles get sized
        # correctly without calling ``tight_layout``; the tight-layout
        # solver triggers a deepcopy of internal MarkerStyle objects on
        # matplotlib 3.10 that recurses past Python's stack limit when a
        # BoundaryNorm colorbar is attached (issue #1874).
        if 'ax' not in kwargs:
            fig, ax = plt.subplots(
                figsize=kwargs.get('figsize', (8, 6)),
                layout='constrained',
            )
            kwargs.pop('figsize', None)
            kwargs['ax'] = ax

        result = da.plot(**kwargs)

        kwargs['ax'].set_aspect('equal')
        return result

    # ---- Surface ----

    def slope(self, **kwargs):
        from .slope import slope
        return slope(self._obj, **kwargs)

    def aspect(self, **kwargs):
        from .aspect import aspect
        return aspect(self._obj, **kwargs)

    def hillshade(self, **kwargs):
        from .hillshade import hillshade
        return hillshade(self._obj, **kwargs)

    def curvature(self, **kwargs):
        from .curvature import curvature
        return curvature(self._obj, **kwargs)

    # ---- Terrain Metrics ----

    def tri(self, **kwargs):
        from .terrain_metrics import tri
        return tri(self._obj, **kwargs)

    def tpi(self, **kwargs):
        from .terrain_metrics import tpi
        return tpi(self._obj, **kwargs)

    def roughness(self, **kwargs):
        from .terrain_metrics import roughness
        return roughness(self._obj, **kwargs)

    # ---- Hydrology ----

    def flow_direction(self, **kwargs):
        from .hydro import flow_direction
        return flow_direction(self._obj, **kwargs)

    def flow_direction_dinf(self, **kwargs):
        from .hydro import flow_direction_dinf
        return flow_direction_dinf(self._obj, **kwargs)

    def flow_direction_mfd(self, **kwargs):
        from .hydro import flow_direction_mfd
        return flow_direction_mfd(self._obj, **kwargs)

    def flow_accumulation(self, **kwargs):
        from .hydro import flow_accumulation
        return flow_accumulation(self._obj, **kwargs)

    def flow_accumulation_mfd(self, **kwargs):
        from .hydro import flow_accumulation_mfd
        return flow_accumulation_mfd(self._obj, **kwargs)

    def watershed(self, pour_points, **kwargs):
        from .hydro import watershed
        return watershed(self._obj, pour_points, **kwargs)

    def basin(self, **kwargs):
        from .hydro import basin
        return basin(self._obj, **kwargs)

    def basins(self, **kwargs):
        from .hydro import basins
        return basins(self._obj, **kwargs)

    def sink(self, **kwargs):
        from .hydro import sink
        return sink(self._obj, **kwargs)

    def fill(self, **kwargs):
        from .hydro import fill
        return fill(self._obj, **kwargs)

    def stream_order(self, flow_accum, **kwargs):
        from .hydro import stream_order
        return stream_order(self._obj, flow_accum, **kwargs)

    def stream_link(self, flow_accum, **kwargs):
        from .hydro import stream_link
        return stream_link(self._obj, flow_accum, **kwargs)

    def snap_pour_point(self, pour_points, **kwargs):
        from .hydro import snap_pour_point
        return snap_pour_point(self._obj, pour_points, **kwargs)

    def flow_path(self, start_points, **kwargs):
        from .hydro import flow_path
        return flow_path(self._obj, start_points, **kwargs)

    def flow_length(self, **kwargs):
        from .hydro import flow_length
        return flow_length(self._obj, **kwargs)

    def twi(self, slope_agg, **kwargs):
        from .hydro import twi
        return twi(self._obj, slope_agg, **kwargs)

    def hand(self, flow_accum, elevation, **kwargs):
        from .hydro import hand
        return hand(self._obj, flow_accum, elevation, **kwargs)

    # ---- Flood ----

    def flood_depth(self, water_level, **kwargs):
        from .flood import flood_depth
        return flood_depth(self._obj, water_level, **kwargs)

    def inundation(self, water_level, **kwargs):
        from .flood import inundation
        return inundation(self._obj, water_level, **kwargs)

    def curve_number_runoff(self, curve_number, **kwargs):
        from .flood import curve_number_runoff
        return curve_number_runoff(self._obj, curve_number, **kwargs)

    def travel_time(self, slope_agg, mannings_n, **kwargs):
        from .flood import travel_time
        return travel_time(self._obj, slope_agg, mannings_n, **kwargs)

    def vegetation_roughness(self, **kwargs):
        from .flood import vegetation_roughness
        return vegetation_roughness(self._obj, **kwargs)

    def vegetation_curve_number(self, soil_group_agg, **kwargs):
        from .flood import vegetation_curve_number
        return vegetation_curve_number(self._obj, soil_group_agg, **kwargs)

    def flood_depth_vegetation(self, slope_agg, mannings_n,
                               unit_discharge, **kwargs):
        from .flood import flood_depth_vegetation
        return flood_depth_vegetation(self._obj, slope_agg, mannings_n,
                                      unit_discharge, **kwargs)

    def viewshed(self, x, y, **kwargs):
        from .viewshed import viewshed
        return viewshed(self._obj, x, y, **kwargs)

    def cumulative_viewshed(self, observers, **kwargs):
        from .visibility import cumulative_viewshed
        return cumulative_viewshed(self._obj, observers, **kwargs)

    def visibility_frequency(self, observers, **kwargs):
        from .visibility import visibility_frequency
        return visibility_frequency(self._obj, observers, **kwargs)

    def line_of_sight(self, x0, y0, x1, y1, **kwargs):
        from .visibility import line_of_sight
        return line_of_sight(self._obj, x0, y0, x1, y1, **kwargs)

    def min_observable_height(self, x, y, **kwargs):
        from .experimental.min_observable_height import min_observable_height
        return min_observable_height(self._obj, x, y, **kwargs)

    # ---- Classification ----

    def natural_breaks(self, **kwargs):
        from .classify import natural_breaks
        return natural_breaks(self._obj, **kwargs)

    def equal_interval(self, **kwargs):
        from .classify import equal_interval
        return equal_interval(self._obj, **kwargs)

    def quantile(self, **kwargs):
        from .classify import quantile
        return quantile(self._obj, **kwargs)

    def reclassify(self, bins, new_values, **kwargs):
        from .classify import reclassify
        return reclassify(self._obj, bins, new_values, **kwargs)

    def binary(self, values, **kwargs):
        from .classify import binary
        return binary(self._obj, values, **kwargs)

    def percentiles(self, **kwargs):
        from .classify import percentiles
        return percentiles(self._obj, **kwargs)

    # ---- Focal ----

    def focal_mean(self, **kwargs):
        from .focal import mean
        return mean(self._obj, **kwargs)

    def bilateral(self, **kwargs):
        from .bilateral import bilateral
        return bilateral(self._obj, **kwargs)

    def glcm_texture(self, **kwargs):
        from .glcm import glcm_texture
        return glcm_texture(self._obj, **kwargs)

    # ---- Edge Detection ----

    def sobel_x(self, **kwargs):
        from .edge_detection import sobel_x
        return sobel_x(self._obj, **kwargs)

    def sobel_y(self, **kwargs):
        from .edge_detection import sobel_y
        return sobel_y(self._obj, **kwargs)

    def laplacian(self, **kwargs):
        from .edge_detection import laplacian
        return laplacian(self._obj, **kwargs)

    def prewitt_x(self, **kwargs):
        from .edge_detection import prewitt_x
        return prewitt_x(self._obj, **kwargs)

    def prewitt_y(self, **kwargs):
        from .edge_detection import prewitt_y
        return prewitt_y(self._obj, **kwargs)

    # ---- Morphological ----

    def morph_erode(self, **kwargs):
        from .morphology import morph_erode
        return morph_erode(self._obj, **kwargs)

    def morph_dilate(self, **kwargs):
        from .morphology import morph_dilate
        return morph_dilate(self._obj, **kwargs)

    def morph_opening(self, **kwargs):
        from .morphology import morph_opening
        return morph_opening(self._obj, **kwargs)

    def morph_closing(self, **kwargs):
        from .morphology import morph_closing
        return morph_closing(self._obj, **kwargs)

    def morph_gradient(self, **kwargs):
        from .morphology import morph_gradient
        return morph_gradient(self._obj, **kwargs)

    def morph_white_tophat(self, **kwargs):
        from .morphology import morph_white_tophat
        return morph_white_tophat(self._obj, **kwargs)

    def morph_black_tophat(self, **kwargs):
        from .morphology import morph_black_tophat
        return morph_black_tophat(self._obj, **kwargs)

    # ---- Proximity / Distance ----

    def proximity(self, **kwargs):
        from .proximity import proximity
        return proximity(self._obj, **kwargs)

    def allocation(self, **kwargs):
        from .proximity import allocation
        return allocation(self._obj, **kwargs)

    def direction(self, **kwargs):
        from .proximity import direction
        return direction(self._obj, **kwargs)

    def cost_distance(self, friction, **kwargs):
        from .cost_distance import cost_distance
        return cost_distance(self._obj, friction, **kwargs)

    def surface_distance(self, elevation, **kwargs):
        from .surface_distance import surface_distance
        return surface_distance(self._obj, elevation, **kwargs)

    def surface_allocation(self, elevation, **kwargs):
        from .surface_distance import surface_allocation
        return surface_allocation(self._obj, elevation, **kwargs)

    def surface_direction(self, elevation, **kwargs):
        from .surface_distance import surface_direction
        return surface_direction(self._obj, elevation, **kwargs)

    # ---- Pathfinding ----

    def a_star_search(self, start, goal, **kwargs):
        from .pathfinding import a_star_search
        return a_star_search(self._obj, start, goal, **kwargs)

    def multi_stop_search(self, waypoints, **kwargs):
        from .pathfinding import multi_stop_search
        return multi_stop_search(self._obj, waypoints, **kwargs)

    # ---- Zonal ----

    def zonal_stats(self, zones, **kwargs):
        from .zonal import stats
        return stats(zones, self._obj, **kwargs)

    def zonal_apply(self, zones, func, **kwargs):
        from .zonal import apply
        return apply(zones, self._obj, func, **kwargs)

    def zonal_crosstab(self, zones, **kwargs):
        from .zonal import crosstab
        return crosstab(zones, self._obj, **kwargs)

    def zonal_hypsometric_integral(self, zones, **kwargs):
        from .zonal import hypsometric_integral
        return hypsometric_integral(zones, self._obj, **kwargs)

    def crop(self, zones, zones_ids, **kwargs):
        from .zonal import crop
        return crop(zones, self._obj, zones_ids, **kwargs)

    def clip_polygon(self, geometry, **kwargs):
        from .polygon_clip import clip_polygon
        return clip_polygon(self._obj, geometry, **kwargs)

    def trim(self, **kwargs):
        from .zonal import trim
        return trim(self._obj, **kwargs)

    def regions(self, **kwargs):
        from .zonal import regions
        return regions(self._obj, **kwargs)

    # ---- Diffusion ----

    def diffuse(self, **kwargs):
        from .diffusion import diffuse
        return diffuse(self._obj, **kwargs)

    # ---- Dasymetric ----

    def disaggregate(self, values, weight, **kwargs):
        from .dasymetric import disaggregate
        return disaggregate(self._obj, values, weight, **kwargs)

    def pycnophylactic(self, values, **kwargs):
        from .dasymetric import pycnophylactic
        return pycnophylactic(self._obj, values, **kwargs)

    # ---- Terrain generation ----

    def generate_terrain(self, **kwargs):
        from .terrain import generate_terrain
        return generate_terrain(self._obj, **kwargs)

    def perlin(self, **kwargs):
        from .perlin import perlin
        return perlin(self._obj, **kwargs)

    # ---- Mahalanobis ----

    def mahalanobis(self, other_bands, **kwargs):
        from .mahalanobis import mahalanobis
        return mahalanobis([self._obj] + list(other_bands), **kwargs)

    # ---- Interpolation ----

    def kde(self, x, y=None, *, coregister=False, **kwargs):
        """Kernel density of points onto this DataArray's grid.

        Pass a GeoDataFrame of Point geometries as the first argument
        (``column`` selects per-point weights), or raw ``x``/``y`` arrays.
        With ``coregister=True`` a GeoDataFrame's points are reprojected
        from their CRS into this array's CRS first.  See
        :func:`xrspatial.kde`.
        """
        from .kde import kde
        return _interp_on_accessor(self._obj, kde, x, (y,),
                                   coregister=coregister, **kwargs)

    def idw(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Inverse-distance-weighted interpolation onto this grid.

        Pass a GeoDataFrame (``column`` selects the value to interpolate)
        or raw ``x``/``y``/``z`` arrays.  ``coregister=True`` reprojects a
        GeoDataFrame's points into this array's CRS first.  See
        :func:`xrspatial.idw`.
        """
        from .interpolate import idw
        return _interp_on_accessor(self._obj, idw, x, (y, z),
                                   coregister=coregister, **kwargs)

    def kriging(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Ordinary-kriging interpolation onto this grid.

        Pass a GeoDataFrame (``column`` selects the value to interpolate)
        or raw ``x``/``y``/``z`` arrays.  ``coregister=True`` reprojects a
        GeoDataFrame's points into this array's CRS first.  See
        :func:`xrspatial.kriging`.
        """
        from .interpolate import kriging
        return _interp_on_accessor(self._obj, kriging, x, (y, z),
                                   coregister=coregister, **kwargs)

    def spline(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Thin-plate-spline interpolation onto this grid.

        Pass a GeoDataFrame (``column`` selects the value to interpolate)
        or raw ``x``/``y``/``z`` arrays.  ``coregister=True`` reprojects a
        GeoDataFrame's points into this array's CRS first.  See
        :func:`xrspatial.spline`.
        """
        from .interpolate import spline
        return _interp_on_accessor(self._obj, spline, x, (y, z),
                                   coregister=coregister, **kwargs)

    # ---- Preview ----

    def preview(self, **kwargs):
        from .preview import preview
        return preview(self._obj, **kwargs)

    # ---- Normalization ----

    def rescale(self, **kwargs):
        from .normalize import rescale
        return rescale(self._obj, **kwargs)

    def standardize(self, **kwargs):
        from .normalize import standardize
        return standardize(self._obj, **kwargs)

    # ---- Reproject ----

    def reproject(self, target_crs, **kwargs):
        from .reproject import reproject
        return reproject(self._obj, target_crs, **kwargs)

    # ---- Raster to vector ----

    def polygonize(self, **kwargs):
        from .polygonize import polygonize
        return polygonize(self._obj, **kwargs)

    def contours(self, **kwargs):
        from .contour import contours
        return contours(self._obj, **kwargs)

    # ---- Fire ----

    def dnbr(self, post_nbr_agg, **kwargs):
        from .fire import dnbr
        return dnbr(self._obj, post_nbr_agg, **kwargs)

    def rdnbr(self, pre_nbr_agg, **kwargs):
        from .fire import rdnbr
        return rdnbr(self._obj, pre_nbr_agg, **kwargs)

    def burn_severity_class(self, **kwargs):
        from .fire import burn_severity_class
        return burn_severity_class(self._obj, **kwargs)

    def fireline_intensity(self, spread_rate_agg, **kwargs):
        from .fire import fireline_intensity
        return fireline_intensity(self._obj, spread_rate_agg, **kwargs)

    def flame_length(self, **kwargs):
        from .fire import flame_length
        return flame_length(self._obj, **kwargs)

    def rate_of_spread(self, wind_speed_agg, fuel_moisture_agg, **kwargs):
        from .fire import rate_of_spread
        return rate_of_spread(self._obj, wind_speed_agg, fuel_moisture_agg, **kwargs)

    def kbdi(self, max_temp_agg, precip_agg, annual_precip, **kwargs):
        from .fire import kbdi
        return kbdi(self._obj, max_temp_agg, precip_agg, annual_precip, **kwargs)

    # ---- Multispectral (self = green band for water indices) ----

    def ndwi(self, nir_agg, **kwargs):
        from .multispectral import ndwi
        return ndwi(self._obj, nir_agg, **kwargs)

    def mndwi(self, swir_agg, **kwargs):
        from .multispectral import mndwi
        return mndwi(self._obj, swir_agg, **kwargs)

    # ---- Multispectral (self = NIR band) ----

    def ndvi(self, red_agg, **kwargs):
        from .multispectral import ndvi
        return ndvi(self._obj, red_agg, **kwargs)

    def evi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import evi
        return evi(self._obj, red_agg, blue_agg, **kwargs)

    def arvi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import arvi
        return arvi(self._obj, red_agg, blue_agg, **kwargs)

    def savi(self, red_agg, **kwargs):
        from .multispectral import savi
        return savi(self._obj, red_agg, **kwargs)

    def nbr(self, swir2_agg, **kwargs):
        from .multispectral import nbr
        return nbr(self._obj, swir2_agg, **kwargs)

    def sipi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import sipi
        return sipi(self._obj, red_agg, blue_agg, **kwargs)

    # ---- Rasterize ----

    def rasterize(self, geometries, *, coregister=False, **kwargs):
        """Rasterize *geometries* onto this DataArray's grid.

        Equivalent to ``rasterize(geometries, like=self._obj, **kwargs)``.
        With ``coregister=True`` a GeoDataFrame's geometries are reprojected
        from their CRS into this raster's CRS before rasterizing; otherwise a
        genuine CRS mismatch raises (see ``check_crs``).  ``coregister`` needs
        a CRS on both sides and only applies to GeoDataFrame input.

        See :func:`xrspatial.rasterize.rasterize` for full parameter docs.
        """
        return _rasterize_on_accessor(self._obj, geometries,
                                      coregister=coregister, **kwargs)

    def coregister(self, data, *, resampling='bilinear', **kwargs):
        """Put *data* on this raster's grid, ready to analyze.

        One call to land another layer on this grid, whatever form it takes:

        - A raster ``xarray.DataArray`` is reprojected and resampled from its
          own CRS onto this grid's CRS, bounds, and shape.
        - A ``GeoDataFrame`` is burned onto the grid with
          :meth:`rasterize`, reprojected into this CRS first (the
          ``coregister=True`` path).

        The result shares this raster's ``y``/``x`` coordinates exactly, so it
        stacks cell-for-cell with the template and with anything else
        coregistered onto it. Pair it with :func:`xrspatial.from_template` to
        go from a region name to an analysis-ready grid::

            grid = from_template('conus')
            elevation = grid.xrs.coregister(my_dem)      # raster -> grid
            roads = grid.xrs.coregister(my_roads_gdf)    # vectors -> grid
            slope = elevation.xrs.slope()

        Parameters
        ----------
        data : xarray.DataArray or geopandas.GeoDataFrame
            The layer to place on this grid. A raster DataArray needs a
            detectable CRS (``attrs['crs']`` or ``crs_wkt``); a GeoDataFrame
            needs ``.crs``.
        resampling : str, default 'bilinear'
            Resampling method for the raster path, forwarded to
            :func:`xrspatial.reproject.reproject` (e.g. ``'nearest'`` for
            categorical data). Ignored when *data* is a GeoDataFrame.
        **kwargs
            Forwarded to :func:`~xrspatial.reproject.reproject` (raster input)
            or :func:`~xrspatial.rasterize.rasterize` (GeoDataFrame input).

        Returns
        -------
        xarray.DataArray
            *data* on this raster's grid.
        """
        return _coregister_on_accessor(self._obj, data, resampling=resampling,
                                       **kwargs)

    # ---- GeoTIFF I/O ----

    def to_geotiff(self, path, **kwargs):
        """Write this DataArray as a GeoTIFF.

        Equivalent to ``to_geotiff(da, path, **kwargs)``.

        See :func:`xrspatial.geotiff.to_geotiff` for full parameter docs.
        """
        from .geotiff import to_geotiff
        return to_geotiff(self._obj, path, **kwargs)

    def open_geotiff(self, source, *, auto_reproject=False,
                     coregister=False, resampling='auto', **kwargs):
        """Read a GeoTIFF windowed to this DataArray's spatial extent.

        Uses ``self``'s ``y``/``x`` coordinates to compute a pixel window
        and reads only that region from the file. The returned DataArray's
        backend is matched to ``self``'s backend by inferring
        ``gpu=`` / ``chunks=`` from ``self`` (caller-supplied ``gpu=`` /
        ``chunks=`` override the inference).

        Parameters
        ----------
        source : str
            File path to the GeoTIFF.
        auto_reproject : bool
            If False (default) and ``self.attrs['crs']`` differs from
            the file's CRS, raises ``ValueError``. If True, projects
            ``self``'s bbox into the file CRS for the windowed read,
            then reprojects the result back to ``self``'s CRS via
            :func:`xrspatial.reproject.reproject` so the returned
            DataArray lines up with ``self``.
        coregister : bool
            If True, read the file unpacked (``unpack=True``: scaled and
            masked), reproject into ``self``'s CRS, and
            resample onto ``self``'s exact grid, so the result shares
            ``self``'s ``y``/``x`` coordinates and shape. The grid snap
            happens even when the CRS already matches. The
            unpack-and-reproject read runs on CPU only, so
            ``coregister=True`` raises ``ValueError`` with ``gpu=True`` or
            ``.vrt`` sources, and it overrides an explicit ``unpack=False``.
            Cells outside the file footprint come back NaN; if the file
            covers less than 10% of ``self``'s grid a ``UserWarning``
            suggests cropping ``self`` to the file bounds first, and a
            file that does not overlap ``self`` at all raises
            ``ValueError``. ``self``'s grid must be regularly spaced
            with ``x`` ascending and ``y`` descending (reproject's
            output convention); other templates raise ``ValueError``,
            as does combining ``coregister=True`` with
            ``allow_rotated=True``.
            When the file carries overviews (a COG) and ``self``'s grid
            is coarser than the full-resolution pixels, the read
            automatically uses the coarsest overview that still meets
            ``self``'s resolution; pass ``overview_level=0`` to force
            the full-resolution read or ``overview_level=N`` to pick a
            level yourself.
            The resample mode follows ``resampling``. This is the heavier
            counterpart of ``auto_reproject``, which keeps the file's native
            resolution.
        resampling : {'auto', 'nearest', 'bilinear', 'cubic'}
            Resampling mode for the ``auto_reproject`` / ``coregister``
            reproject step; ignored when no reprojection happens.
            ``'auto'`` (default) picks ``'nearest'`` for categorical
            rasters (a paletted colormap is present, or the data has an
            integer dtype) and ``'bilinear'`` otherwise. Note an
            integer-typed continuous DEM (e.g. int16 elevation) is
            treated as categorical under ``'auto'`` (a ``UserWarning``
            points this out); pass ``resampling='bilinear'`` for those.
            Conversely, an integer raster read with a nodata sentinel is
            promoted to float on read, so a categorical raster that
            carries nodata and no colormap resolves to ``'bilinear'``;
            pass ``resampling='nearest'`` for those.
        **kwargs
            Forwarded to :func:`xrspatial.geotiff.open_geotiff` (except
            ``window=``, which is computed automatically; an explicit
            ``overview_level=`` is honoured and the window is computed
            in that overview's pixel space).

        Returns
        -------
        xr.DataArray
            The windowed portion of the GeoTIFF. In ``self``'s CRS when
            ``auto_reproject`` reprojection occurred, on ``self``'s exact
            grid when ``coregister`` is set, and otherwise in the file's
            native CRS. The ``auto_reproject`` / ``coregister``
            reproject step chunks its dask output like the explicit
            ``chunks=`` kwarg when given, and otherwise like ``self``,
            rather than reverting to
            :func:`~xrspatial.reproject.reproject`'s own default.
        """
        return _open_geotiff_windowed(
            self._obj, source, auto_reproject=auto_reproject,
            coregister=coregister, resampling=resampling, **kwargs
        )

    # ---- Chunking ----

    def rechunk_no_shuffle(self, **kwargs):
        from .utils import rechunk_no_shuffle
        return rechunk_no_shuffle(self._obj, **kwargs)

    def fused_overlap(self, *stages, **kwargs):
        from .utils import fused_overlap
        return fused_overlap(self._obj, *stages, **kwargs)

    def multi_overlap(self, func, n_outputs, **kwargs):
        from .utils import multi_overlap
        return multi_overlap(self._obj, func, n_outputs, **kwargs)

    # ---- Diagnostics ----

    def validate(self, *, raise_on_error=False):
        """Check this DataArray against the xarray-spatial input contract.

        Returns a :class:`~xrspatial.validate.ValidationReport` listing
        every contract violation as an error (a spatial op will fail) or
        a warning (behavior degrades), each with a suggested fix. The
        report is truthy when there are no error-level issues, so
        ``if not da.xrs.validate(): ...`` reads naturally.

        Parameters
        ----------
        raise_on_error : bool, default False
            If True, raise :class:`~xrspatial.validate.XrsContractError`
            when any error-level issue is found instead of only
            recording it in the report.
        """
        from .validate import validate
        report = validate(self._obj)
        if raise_on_error:
            report.raise_if_errors()
        return report


@xr.register_dataset_accessor("xrs")
class XrsSpatialDatasetAccessor:
    """Dataset accessor exposing xarray-spatial operations.

    Single-input functions apply the operation to each data variable
    (via the existing ``@supports_dataset`` decorator).  Multi-input
    (multispectral) functions accept string kwargs that map band
    aliases to Dataset variable names.
    """

    def __init__(self, obj):
        self._obj = obj

    def __getattribute__(self, name):
        return _wrap_accessor_attr(self, name)

    def __repr__(self):
        return _accessor_repr_text(type(self))

    def _repr_html_(self):
        return _accessor_repr_html(type(self))

    # ---- Plot ----

    def plot(self, vars=None, cols=3, **kwargs):
        """Plot 2D data variables as a grid of subplots.

        Parameters
        ----------
        vars : list of str, optional
            Variable names to plot.  If None, plots all 2D variables.
        cols : int, default 3
            Maximum number of columns in the subplot grid.
        **kwargs
            Passed to each subplot's ``da.plot()``.  Common extras:
            ``cmap``, ``figsize``, ``add_colorbar``.

        Returns
        -------
        numpy.ndarray of matplotlib.axes.Axes
        """
        _require_matplotlib()
        import math
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import BoundaryNorm

        ds = self._obj

        # Collect 2D variables to plot.
        if vars is not None:
            names = [v for v in vars if v in ds.data_vars]
        else:
            names = [
                v for v in ds.data_vars
                if ds[v].ndim == 2
            ]

        if not names:
            raise ValueError("No 2D variables found to plot")

        n = len(names)
        ncols = min(n, cols)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=kwargs.pop('figsize', (5 * ncols, 4 * nrows)),
            squeeze=False,
        )

        for idx, name in enumerate(names):
            ax = axes[idx // ncols][idx % ncols]
            da = ds[name]

            # Materialise dask arrays so matplotlib can render them.
            try:
                da = da.compute()
            except (AttributeError, TypeError):
                pass

            # Use embedded GeoTIFF colormap when present. See the
            # DataArray ``.xrs.plot`` path above for the contract v2
            # migration note.
            cmap = _listed_colormap_from_attrs(da.attrs)
            kw = dict(kwargs)
            if cmap is not None and 'cmap' not in kw:
                n_colors = len(cmap.colors)
                boundaries = np.arange(n_colors + 1) - 0.5
                kw.setdefault('cmap', cmap)
                kw.setdefault('norm', BoundaryNorm(boundaries, n_colors))
                kw.setdefault('add_colorbar', True)

            kw.setdefault('ax', ax)
            da.plot(**kw)
            ax.set_title(name)
            ax.set_aspect('equal')

        # Hide unused axes.
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        return axes

    # ---- Surface ----

    def slope(self, **kwargs):
        from .slope import slope
        return slope(self._obj, **kwargs)

    def aspect(self, **kwargs):
        from .aspect import aspect
        return aspect(self._obj, **kwargs)

    def hillshade(self, **kwargs):
        from .hillshade import hillshade
        return hillshade(self._obj, **kwargs)

    def curvature(self, **kwargs):
        from .curvature import curvature
        return curvature(self._obj, **kwargs)

    # ---- Terrain Metrics ----

    def tri(self, **kwargs):
        from .terrain_metrics import tri
        return tri(self._obj, **kwargs)

    def tpi(self, **kwargs):
        from .terrain_metrics import tpi
        return tpi(self._obj, **kwargs)

    def roughness(self, **kwargs):
        from .terrain_metrics import roughness
        return roughness(self._obj, **kwargs)

    # ---- Hydrology ----

    def flow_direction(self, **kwargs):
        from .hydro import flow_direction
        return flow_direction(self._obj, **kwargs)

    def flow_direction_dinf(self, **kwargs):
        from .hydro import flow_direction_dinf
        return flow_direction_dinf(self._obj, **kwargs)

    def flow_direction_mfd(self, **kwargs):
        from .hydro import flow_direction_mfd
        return flow_direction_mfd(self._obj, **kwargs)

    def flow_accumulation(self, **kwargs):
        from .hydro import flow_accumulation
        return flow_accumulation(self._obj, **kwargs)

    def flow_accumulation_mfd(self, **kwargs):
        from .hydro import flow_accumulation_mfd
        return flow_accumulation_mfd(self._obj, **kwargs)

    def watershed(self, pour_points, **kwargs):
        from .hydro import watershed
        return watershed(self._obj, pour_points, **kwargs)

    def basin(self, **kwargs):
        from .hydro import basin
        return basin(self._obj, **kwargs)

    def basins(self, **kwargs):
        from .hydro import basins
        return basins(self._obj, **kwargs)

    def sink(self, **kwargs):
        from .hydro import sink
        return sink(self._obj, **kwargs)

    def fill(self, **kwargs):
        from .hydro import fill
        return fill(self._obj, **kwargs)

    def stream_order(self, flow_accum, **kwargs):
        from .hydro import stream_order
        return stream_order(self._obj, flow_accum, **kwargs)

    def stream_link(self, flow_accum, **kwargs):
        from .hydro import stream_link
        return stream_link(self._obj, flow_accum, **kwargs)

    def snap_pour_point(self, pour_points, **kwargs):
        from .hydro import snap_pour_point
        return snap_pour_point(self._obj, pour_points, **kwargs)

    def flow_path(self, start_points, **kwargs):
        from .hydro import flow_path
        return flow_path(self._obj, start_points, **kwargs)

    def flow_length(self, **kwargs):
        from .hydro import flow_length
        return flow_length(self._obj, **kwargs)

    def twi(self, slope_agg, **kwargs):
        from .hydro import twi
        return twi(self._obj, slope_agg, **kwargs)

    def hand(self, flow_accum, elevation, **kwargs):
        from .hydro import hand
        return hand(self._obj, flow_accum, elevation, **kwargs)

    # ---- Flood ----

    def flood_depth(self, water_level, **kwargs):
        from .flood import flood_depth
        return flood_depth(self._obj, water_level, **kwargs)

    def inundation(self, water_level, **kwargs):
        from .flood import inundation
        return inundation(self._obj, water_level, **kwargs)

    def curve_number_runoff(self, curve_number, **kwargs):
        from .flood import curve_number_runoff
        return curve_number_runoff(self._obj, curve_number, **kwargs)

    def travel_time(self, slope_agg, mannings_n, **kwargs):
        from .flood import travel_time
        return travel_time(self._obj, slope_agg, mannings_n, **kwargs)

    def vegetation_roughness(self, **kwargs):
        from .flood import vegetation_roughness
        return vegetation_roughness(self._obj, **kwargs)

    def vegetation_curve_number(self, soil_group_agg, **kwargs):
        from .flood import vegetation_curve_number
        return vegetation_curve_number(self._obj, soil_group_agg, **kwargs)

    def flood_depth_vegetation(self, slope_agg, mannings_n,
                               unit_discharge, **kwargs):
        from .flood import flood_depth_vegetation
        return flood_depth_vegetation(self._obj, slope_agg, mannings_n,
                                      unit_discharge, **kwargs)

    # ---- Classification ----

    def natural_breaks(self, **kwargs):
        from .classify import natural_breaks
        return natural_breaks(self._obj, **kwargs)

    def equal_interval(self, **kwargs):
        from .classify import equal_interval
        return equal_interval(self._obj, **kwargs)

    def quantile(self, **kwargs):
        from .classify import quantile
        return quantile(self._obj, **kwargs)

    def reclassify(self, bins, new_values, **kwargs):
        from .classify import reclassify
        return reclassify(self._obj, bins, new_values, **kwargs)

    def binary(self, values, **kwargs):
        from .classify import binary
        return binary(self._obj, values, **kwargs)

    def percentiles(self, **kwargs):
        from .classify import percentiles
        return percentiles(self._obj, **kwargs)

    # ---- Focal ----

    def focal_mean(self, **kwargs):
        from .focal import mean
        return mean(self._obj, **kwargs)

    def bilateral(self, **kwargs):
        from .bilateral import bilateral
        return bilateral(self._obj, **kwargs)

    def glcm_texture(self, **kwargs):
        from .glcm import glcm_texture
        return glcm_texture(self._obj, **kwargs)

    # ---- Edge Detection ----

    def sobel_x(self, **kwargs):
        from .edge_detection import sobel_x
        return sobel_x(self._obj, **kwargs)

    def sobel_y(self, **kwargs):
        from .edge_detection import sobel_y
        return sobel_y(self._obj, **kwargs)

    def laplacian(self, **kwargs):
        from .edge_detection import laplacian
        return laplacian(self._obj, **kwargs)

    def prewitt_x(self, **kwargs):
        from .edge_detection import prewitt_x
        return prewitt_x(self._obj, **kwargs)

    def prewitt_y(self, **kwargs):
        from .edge_detection import prewitt_y
        return prewitt_y(self._obj, **kwargs)

    # ---- Morphological ----

    def morph_erode(self, **kwargs):
        from .morphology import morph_erode
        return morph_erode(self._obj, **kwargs)

    def morph_dilate(self, **kwargs):
        from .morphology import morph_dilate
        return morph_dilate(self._obj, **kwargs)

    def morph_opening(self, **kwargs):
        from .morphology import morph_opening
        return morph_opening(self._obj, **kwargs)

    def morph_closing(self, **kwargs):
        from .morphology import morph_closing
        return morph_closing(self._obj, **kwargs)

    def morph_gradient(self, **kwargs):
        from .morphology import morph_gradient
        return morph_gradient(self._obj, **kwargs)

    def morph_white_tophat(self, **kwargs):
        from .morphology import morph_white_tophat
        return morph_white_tophat(self._obj, **kwargs)

    def morph_black_tophat(self, **kwargs):
        from .morphology import morph_black_tophat
        return morph_black_tophat(self._obj, **kwargs)

    # ---- Diffusion ----

    def diffuse(self, **kwargs):
        from .diffusion import diffuse
        return diffuse(self._obj, **kwargs)

    # ---- Proximity ----

    def proximity(self, **kwargs):
        from .proximity import proximity
        return proximity(self._obj, **kwargs)

    def allocation(self, **kwargs):
        from .proximity import allocation
        return allocation(self._obj, **kwargs)

    def direction(self, **kwargs):
        from .proximity import direction
        return direction(self._obj, **kwargs)

    def cost_distance(self, friction, **kwargs):
        from .cost_distance import cost_distance
        return cost_distance(self._obj, friction, **kwargs)

    def surface_distance(self, elevation, **kwargs):
        from .surface_distance import surface_distance
        return surface_distance(self._obj, elevation, **kwargs)

    def surface_allocation(self, elevation, **kwargs):
        from .surface_distance import surface_allocation
        return surface_allocation(self._obj, elevation, **kwargs)

    def surface_direction(self, elevation, **kwargs):
        from .surface_distance import surface_direction
        return surface_direction(self._obj, elevation, **kwargs)

    # ---- Pathfinding ----

    def multi_stop_search(self, waypoints, **kwargs):
        from .pathfinding import multi_stop_search
        return multi_stop_search(self._obj, waypoints, **kwargs)

    # ---- Preview ----

    def preview(self, **kwargs):
        from .preview import preview
        return preview(self._obj, **kwargs)

    # ---- Normalization ----

    def rescale(self, **kwargs):
        from .normalize import rescale
        return rescale(self._obj, **kwargs)

    def standardize(self, **kwargs):
        from .normalize import standardize
        return standardize(self._obj, **kwargs)

    # ---- Fire ----

    def burn_severity_class(self, **kwargs):
        from .fire import burn_severity_class
        return burn_severity_class(self._obj, **kwargs)

    def flame_length(self, **kwargs):
        from .fire import flame_length
        return flame_length(self._obj, **kwargs)

    # ---- Multispectral (band mapping via kwargs) ----

    def ndwi(self, green, nir, **kwargs):
        from .multispectral import ndwi
        return ndwi(self._obj, green=green, nir=nir, **kwargs)

    def mndwi(self, green, swir, **kwargs):
        from .multispectral import mndwi
        return mndwi(self._obj, green=green, swir=swir, **kwargs)

    def ndvi(self, nir, red, **kwargs):
        from .multispectral import ndvi
        return ndvi(self._obj, nir=nir, red=red, **kwargs)

    def evi(self, nir, red, blue, **kwargs):
        from .multispectral import evi
        return evi(self._obj, nir=nir, red=red, blue=blue, **kwargs)

    def arvi(self, nir, red, blue, **kwargs):
        from .multispectral import arvi
        return arvi(self._obj, nir=nir, red=red, blue=blue, **kwargs)

    def savi(self, nir, red, **kwargs):
        from .multispectral import savi
        return savi(self._obj, nir=nir, red=red, **kwargs)

    def nbr(self, nir, swir2, **kwargs):
        from .multispectral import nbr
        return nbr(self._obj, nir=nir, swir2=swir2, **kwargs)

    def sipi(self, nir, red, blue, **kwargs):
        from .multispectral import sipi
        return sipi(self._obj, nir=nir, red=red, blue=blue, **kwargs)

    # ---- Rasterize ----

    def rasterize(self, geometries, *, coregister=False, **kwargs):
        """Rasterize *geometries* onto a 2-D variable's grid.

        See :meth:`XrsSpatialDataArrayAccessor.rasterize`.
        """
        ds = self._obj
        # Find a 2D variable with y/x dims to use as template
        for var in ds.data_vars:
            da = ds[var]
            if da.ndim == 2 and 'y' in da.dims and 'x' in da.dims:
                return _rasterize_on_accessor(da, geometries,
                                              coregister=coregister, **kwargs)
        raise ValueError(
            "Dataset has no 2D variable with 'y' and 'x' dimensions "
            "to use as rasterize template"
        )

    # ---- Interpolation ----

    def _interp_template(self):
        """First 2-D y/x variable to use as an interpolation template."""
        ds = self._obj
        for var in ds.data_vars:
            da = ds[var]
            if da.ndim == 2 and 'y' in da.dims and 'x' in da.dims:
                return da
        raise ValueError(
            "Dataset has no 2D variable with 'y' and 'x' dimensions "
            "to use as an interpolation template"
        )

    def kde(self, x, y=None, *, coregister=False, **kwargs):
        """Kernel density of points onto a 2-D variable's grid.

        See :meth:`XrsSpatialDataArrayAccessor.kde`.
        """
        from .kde import kde
        return _interp_on_accessor(self._interp_template(), kde, x, (y,),
                                   coregister=coregister, **kwargs)

    def idw(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Inverse-distance-weighted interpolation onto a 2-D variable's grid.

        See :meth:`XrsSpatialDataArrayAccessor.idw`.
        """
        from .interpolate import idw
        return _interp_on_accessor(self._interp_template(), idw, x, (y, z),
                                   coregister=coregister, **kwargs)

    def spline(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Thin-plate-spline interpolation onto a 2-D variable's grid.

        See :meth:`XrsSpatialDataArrayAccessor.spline`.
        """
        from .interpolate import spline
        return _interp_on_accessor(self._interp_template(), spline, x, (y, z),
                                   coregister=coregister, **kwargs)

    def kriging(self, x, y=None, z=None, *, coregister=False, **kwargs):
        """Ordinary-kriging interpolation onto a 2-D variable's grid.

        See :meth:`XrsSpatialDataArrayAccessor.kriging`.
        """
        from .interpolate import kriging
        return _interp_on_accessor(self._interp_template(), kriging, x, (y, z),
                                   coregister=coregister, **kwargs)

    # ---- GeoTIFF I/O ----

    def to_geotiff(self, path, var=None, **kwargs):
        """Write a Dataset variable as a GeoTIFF.

        Parameters
        ----------
        path : str
            Output file path.
        var : str or None
            Variable name to write.  If None, uses the first 2D variable
            with y/x dimensions.
        **kwargs
            Passed to :func:`xrspatial.geotiff.to_geotiff`.
        """
        from .geotiff import to_geotiff
        ds = self._obj
        if var is not None:
            return to_geotiff(ds[var], path, **kwargs)
        for v in ds.data_vars:
            da = ds[v]
            if da.ndim >= 2 and 'y' in da.dims and 'x' in da.dims:
                return to_geotiff(da, path, **kwargs)
        raise ValueError(
            "Dataset has no variable with 'y' and 'x' dimensions to write"
        )

    def open_geotiff(self, source, *, auto_reproject=False, var=None,
                     coregister=False, resampling='auto', **kwargs):
        """Read a GeoTIFF windowed to this Dataset's spatial extent.

        Uses the Dataset's ``y``/``x`` coordinates to compute a pixel
        window and reads only that region from the file. The returned
        DataArray's backend is matched to the Dataset's first 2D
        ``y``/``x`` data variable (or ``ds[var]`` if ``var`` is given)
        by inferring ``gpu=`` / ``chunks=`` from that variable.
        Caller-supplied ``gpu=`` / ``chunks=`` override the inference.

        Parameters
        ----------
        source : str
            File path to the GeoTIFF.
        auto_reproject : bool
            If False (default) and the Dataset's CRS differs from the
            file's CRS, raises ``ValueError``. If True, projects the
            Dataset's bbox into the file CRS for the windowed read and
            reprojects the result back to the Dataset's CRS via
            :func:`xrspatial.reproject.reproject`.
        var : str or None
            Data variable used for backend inference and CRS lookup. If
            None, picks the first 2D variable with ``y``/``x`` dims.
        coregister : bool
            If True, read the file unpacked (``unpack=True``: scaled and
            masked), reproject into the Dataset's CRS,
            and resample onto the Dataset's exact grid, so the result
            shares the Dataset's ``y``/``x`` coordinates and shape. The
            grid snap happens even when the CRS already matches. The
            unpack-and-reproject read runs on CPU only, so
            ``coregister=True`` raises ``ValueError`` with ``gpu=True`` or
            ``.vrt`` sources, and it overrides an explicit ``unpack=False``.
            Cells outside the file footprint come back NaN; if the file
            covers less than 10% of the Dataset's grid a ``UserWarning``
            suggests cropping the Dataset to the file bounds first, and
            a file that does not overlap the Dataset at all raises
            ``ValueError``. The Dataset's grid must be regularly spaced
            with ``x`` ascending and ``y`` descending; other grids raise
            ``ValueError``, as does combining ``coregister=True`` with
            ``allow_rotated=True``.
            When the file carries overviews (a COG) and the Dataset's
            grid is coarser than the full-resolution pixels, the read
            automatically uses the coarsest overview that still meets
            the Dataset's resolution; pass ``overview_level=0`` to force
            the full-resolution read.
            The resample mode follows ``resampling``.
        resampling : {'auto', 'nearest', 'bilinear', 'cubic'}
            Resampling mode for the ``auto_reproject`` / ``coregister``
            reproject step; ignored when no reprojection happens.
            ``'auto'`` (default) picks ``'nearest'`` for categorical
            rasters (a paletted colormap is present, or the data has an
            integer dtype) and ``'bilinear'`` otherwise. Note an
            integer-typed continuous DEM (e.g. int16 elevation) is
            treated as categorical under ``'auto'``; pass
            ``resampling='bilinear'`` for those. Conversely, an integer
            raster read with a nodata sentinel is promoted to float on
            read, so a categorical raster that carries nodata and no
            colormap resolves to ``'bilinear'``; pass
            ``resampling='nearest'`` for those.
        **kwargs
            Forwarded to :func:`xrspatial.geotiff.open_geotiff` (except
            ``window=``, which is computed automatically).

        Returns
        -------
        xr.DataArray
            The windowed portion of the GeoTIFF.
        """
        ds = self._obj
        if 'y' not in ds.coords or 'x' not in ds.coords:
            raise ValueError(
                "Dataset must have 'y' and 'x' coordinates to compute "
                "a spatial window"
            )
        rep = _pick_representative_dataarray(ds, var=var)
        # Fall back to Dataset-level CRS when the variable lacks it
        if 'crs' not in rep.attrs and 'crs' in ds.attrs:
            rep = rep.copy()
            rep.attrs = {**rep.attrs, 'crs': ds.attrs['crs']}
        return _open_geotiff_windowed(
            rep, source, auto_reproject=auto_reproject,
            coregister=coregister, resampling=resampling, **kwargs
        )

    # ---- Chunking ----

    def rechunk_no_shuffle(self, **kwargs):
        from .utils import rechunk_no_shuffle
        return rechunk_no_shuffle(self._obj, **kwargs)

    # ---- Diagnostics ----

    def validate(self, *, raise_on_error=False):
        """Check each data variable against the xarray-spatial contract.

        Returns a
        :class:`~xrspatial.validate.DatasetValidationReport` holding a
        per-variable :class:`~xrspatial.validate.ValidationReport`. The
        report is truthy when every variable is compliant.

        Parameters
        ----------
        raise_on_error : bool, default False
            If True, raise :class:`~xrspatial.validate.XrsContractError`
            when any variable has an error-level issue.
        """
        from .validate import validate_dataset
        report = validate_dataset(self._obj)
        if raise_on_error:
            report.raise_if_errors()
        return report


# ---------------------------------------------------------------------------
# Surface standalone-function docstrings on accessor methods so that, e.g.,
# ``help(da.xrs.slope)`` shows the same documentation as ``help(slope)``.
# ---------------------------------------------------------------------------

# Accessor method name -> name of the standalone function whose docstring
# should be surfaced (resolved from the top-level ``xrspatial`` namespace, or
# from ``xrspatial.hydro`` for the internal ``*_d8`` variants). Only needed
# when the method name differs from the function name, or when the direct
# delegate's docstring is a generic dispatcher: the hydrology unified wrappers
# route by ``routing=`` and carry only a stub docstring, so their help text is
# taken from the documented default-algorithm (``*_d8``) variants instead.
_DOC_SOURCE_OVERRIDES = {
    'focal_mean': 'mean',
    'zonal_hypsometric_integral': 'hypsometric_integral',
    'fill': 'fill_d8',
    'flow_direction': 'flow_direction_d8',
    'flow_accumulation': 'flow_accumulation_d8',
    'basin': 'basin_d8',
    'basins': 'basins_d8',
    'watershed': 'watershed_d8',
    'snap_pour_point': 'snap_pour_point_d8',
    'flow_path': 'flow_path_d8',
    'flow_length': 'flow_length_d8',
    'sink': 'sink_d8',
    'stream_link': 'stream_link_d8',
    'stream_order': 'stream_order_d8',
    'twi': 'twi_d8',
    'hand': 'hand_d8',
}


def _delegated_doc(method_name):
    """Return the docstring to surface for an accessor method, or None."""
    if method_name == 'min_observable_height':
        from .experimental.min_observable_height import min_observable_height
        return min_observable_height.__doc__
    import xrspatial
    source_name = _DOC_SOURCE_OVERRIDES.get(method_name, method_name)
    func = getattr(xrspatial, source_name, None)
    if func is None:
        # The hydrology ``*_d8`` doc sources are internal to xrspatial.hydro;
        # only the routing wrappers are exported at the top level.
        from . import hydro
        func = getattr(hydro, source_name, None)
    return func.__doc__ if func is not None else None


def _attach_delegated_docs(cls):
    for name, member in vars(cls).items():
        if name.startswith('_') or not callable(member) or member.__doc__:
            continue
        try:
            doc = _delegated_doc(name)
        except Exception:
            continue
        if doc:
            member.__doc__ = doc


_attach_delegated_docs(XrsSpatialDataArrayAccessor)
_attach_delegated_docs(XrsSpatialDatasetAccessor)
