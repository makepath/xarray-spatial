"""Empty study-area DataArrays for common regions.

:func:`from_template` turns a region name into an empty (NaN-filled) raster you
can start an analysis from, instead of looking up a bounding box, choosing a
CRS, and assembling coordinates by hand. The result follows the xarray-spatial
array contract (2-D ``['y', 'x']`` grid with pixel-center coordinates and
``res``/``crs`` attributes), so it feeds straight into ``slope``, ``hillshade``,
``rasterize``, and the rest of the library.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from xrspatial._template_data import (_CITIES, _CITY_DEFAULT_RESOLUTION, _COUNTRY_BBOXES,
                                      _COUNTRY_DEFAULT_RESOLUTION, _EQUAL_AREA_FALLBACK_EPSG,
                                      _REGION_ALIASES, _REGIONS, _UPS_NORTH_EPSG,
                                      _UPS_SOUTH_EPSG)
from xrspatial.reproject._crs_utils import _require_pyproj, _resolve_crs
from xrspatial.reproject._grid import _edge_samples, _make_output_coords, _transform_boundary

_PRESERVE_OPTIONS = ("area", "shape")

# Guard against a stray fine resolution allocating an enormous array. Applied to
# the eager backends (numpy, cupy) only: those materialize the whole grid up
# front, so a huge shape is an immediate out-of-memory risk. A dask grid is
# lazy (da.full builds a chunk graph, never the array), so the cap is skipped
# there -- a large chunked study area is a legitimate workflow.
_MAX_CELLS = 500_000_000

# A lazy dask grid never materializes, but its task graph does: with a fixed
# chunk size the block count grows with the cell count, and a typo-level
# resolution can build a graph large enough to bog down the client during
# construction. Guard on the estimated block count (not cells) so a legitimate
# large grid with sensible chunks still passes -- the new_england@10m repro is
# ~26k blocks, so 1e6 leaves wide headroom while catching the runaway case.
_MAX_CHUNKS = 1_000_000

# Target block edge (cells) for the default dask tiling. dask's byte-based
# 'auto' picks one chunk shape from the array's total bytes, which leaves most
# templates as a single giant block (no parallelism) or with thin ragged edge
# slivers. A fixed square block tiles evenly and keeps each chunk friendly to
# the neighborhood ops (slope, focal, ...) that run on the result via
# map_overlap. 2048 is ~16 MB at float32: small enough to parallelize a
# moderate grid, large enough that overlap halos stay cheap.
_DASK_BLOCK = 2048

# Ceiling on the block count for the default tiling. A 2048-cell block would
# explode the graph at a typo-level fine resolution, so for very large grids the
# block edge grows to keep the count near this many blocks. That keeps the
# default 'auto' path always under _MAX_CHUNKS (it never errors on its own), and
# leaves any grid up to ~8e11 cells on the plain 2048 block.
_DASK_MAX_BLOCKS = 200_000


def _balanced_axis(size, block):
    """Split ``size`` into near-equal blocks of about ``block`` cells.

    Returns a tuple of chunk lengths that sum to ``size`` and differ by at most
    one, so there is no tiny trailing sliver. A dimension at or below ~1.5x the
    block stays a single chunk.
    """
    n = max(1, round(size / block))
    base, rem = divmod(size, n)
    return tuple(base + 1 if i < rem else base for i in range(n))


def _block_edge(cells):
    """Block edge (cells) for the default tiling of a ``cells``-cell grid.

    A fixed ``_DASK_BLOCK`` edge, grown for very large grids so the block count
    stays near ``_DASK_MAX_BLOCKS`` instead of exploding the task graph.
    """
    import math
    return max(_DASK_BLOCK, math.ceil(math.sqrt(cells / _DASK_MAX_BLOCKS)))


def _neighborhood_chunks(shape, block=None):
    """Even, square-ish dask chunks for the default tiling.

    Uses a ~``_DASK_BLOCK`` block, growing it for very large grids so the block
    count stays near ``_DASK_MAX_BLOCKS`` instead of exploding the task graph.
    Pass ``block`` to reuse an edge already computed (so a padded shape tiles
    into the same blocks it was padded to).
    """
    if block is None:
        block = _block_edge(shape[0] * shape[1])
    return tuple(_balanced_axis(size, block) for size in shape)


def _pad_to_tiles(shape, block):
    """Grow each axis up to a whole multiple of ``block``.

    An axis that already fits in a single block (``_balanced_axis`` would keep
    it one chunk) is left alone, so a small template is not bloated out to a
    full block. A multi-block axis is padded up to the next ``block`` multiple
    so its tiling has no ragged remainder chunk.
    """
    import math
    out = []
    for size in shape:
        if round(size / block) >= 2:
            out.append(int(math.ceil(size / block) * block))
        else:
            out.append(size)
    return tuple(out)


def _resolve(name):
    """Resolve a name to a spec dict.

    Keys: ``bounds`` (projected, native CRS), ``crs`` (native EPSG int),
    ``default_resolution``, ``key``, ``lonlat`` (lon/lat bbox), and the optional
    ``area_epsg`` / ``shape_epsg`` overrides used by the ``preserve`` path.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string region name or country code, "
            f"got {type(name).__name__}"
        )

    name_l = name.lower()
    region = _REGIONS.get(_REGION_ALIASES.get(name_l, name_l))
    if region is not None:
        return dict(
            bounds=region["bounds"], crs=region["crs"],
            default_resolution=region["default_resolution"], key=name_l,
            lonlat=region["lonlat"], area_epsg=region.get("area_epsg"),
            shape_epsg=region.get("shape_epsg"),
        )

    city = _CITIES.get(name.lower())
    if city is not None:
        return dict(
            bounds=city["bounds"], crs=city["crs"],
            default_resolution=_CITY_DEFAULT_RESOLUTION, key=name.lower(),
            lonlat=city["lonlat"], area_epsg=None, shape_epsg=None,
        )

    code = name.upper()
    bbox = _COUNTRY_BBOXES.get(code)
    if bbox is not None:
        return dict(
            bounds=bbox, crs=4326,
            default_resolution=_COUNTRY_DEFAULT_RESOLUTION, key=code,
            lonlat=bbox, area_epsg=None, shape_epsg=None,
        )

    regions = ", ".join(sorted(set(_REGIONS) | set(_REGION_ALIASES)))
    raise ValueError(
        f"Unknown template {name!r}. Available named regions: {regions}. "
        f"{len(_CITIES)} world cities are also supported (lowercase name, e.g. "
        f"'london', 'tokyo'). Countries must be an ISO-3166 / GADM alpha-3 code "
        f"(e.g. 'USA', 'FRA', 'JPN'); {len(_COUNTRY_BBOXES)} are supported. "
        f"Call xrspatial.templates.list_templates() to list every accepted name."
    )


def _preserve_epsg(lonlat, preserve, area_epsg, shape_epsg):
    """Pick an EPSG code that preserves ``preserve`` for the lon/lat bbox."""
    lon_min, lat_min, lon_max, lat_max = lonlat
    lat_c = (lat_min + lat_max) / 2.0
    lon_c = (lon_min + lon_max) / 2.0
    # Normalize to [-180, 180] so antimeridian-wrapped country boxes (lon > 180)
    # still query a sensible UTM zone.
    lon_c = ((lon_c + 180.0) % 360.0) - 180.0

    if preserve == "area":
        return int(area_epsg) if area_epsg is not None \
            else _EQUAL_AREA_FALLBACK_EPSG

    # preserve == "shape": curated override, else UTM zone (UPS at the poles).
    if shape_epsg is not None:
        return int(shape_epsg)
    if abs(lat_c) > 84.0:
        return _UPS_NORTH_EPSG if lat_c > 0 else _UPS_SOUTH_EPSG

    _require_pyproj()
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info
    aoi = AreaOfInterest(lon_c - 0.01, lat_c - 0.01, lon_c + 0.01, lat_c + 0.01)
    infos = query_utm_crs_info("WGS 84", aoi)
    if not infos:
        return _UPS_NORTH_EPSG if lat_c >= 0 else _UPS_SOUTH_EPSG
    return int(infos[0].code)


def _project_bbox_bounds(lonlat, epsg):
    """Project a lon/lat bbox into ``epsg`` and return (left, bottom, right, top)."""
    lon_min, lat_min, lon_max, lat_max = lonlat
    xs, ys = _edge_samples(lon_min, lat_min, lon_max, lat_max, 101)
    tx, ty = _transform_boundary(_resolve_crs(4326), _resolve_crs(epsg), xs, ys)
    tx, ty = np.asarray(tx), np.asarray(ty)
    valid = np.isfinite(tx) & np.isfinite(ty)
    if not valid.any():
        raise ValueError(
            f"projecting the template into EPSG:{epsg} produced no finite "
            f"coordinates"
        )
    tx, ty = tx[valid], ty[valid]
    return (float(tx.min()), float(ty.min()), float(tx.max()), float(ty.max()))


def _default_preserve_resolution(bounds):
    """Default metre resolution: ~1000 cells on the long axis, snapped 1/2/5."""
    import math
    left, bottom, right, top = bounds
    raw = max(right - left, top - bottom) / 1000.0
    if raw <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5):
        if raw <= m * base:
            return float(m * base)
    return float(10 * base)


def _normalize_resolution(resolution, default):
    """Return a positive (res_x, res_y) tuple from a scalar/tuple/None."""
    if resolution is None:
        resolution = default
    if isinstance(resolution, (tuple, list)):
        if len(resolution) != 2:
            raise ValueError(
                "resolution tuple must be (res_x, res_y), "
                f"got {len(resolution)} values"
            )
        res_x, res_y = resolution
    else:
        res_x = res_y = resolution
    res_x, res_y = float(res_x), float(res_y)
    if res_x <= 0 or res_y <= 0:
        raise ValueError(f"resolution must be positive, got {(res_x, res_y)}")
    return res_x, res_y


def _make_data(shape, fill, backend, chunks):
    """Build a NaN-filled array of ``shape`` for the requested backend."""
    backend = backend.lower()
    if backend in ("dask", "dask+numpy"):
        import dask.array as da
        return da.full(shape, fill, dtype="float32", chunks=chunks)
    if backend == "numpy":
        return np.full(shape, fill, dtype="float32")
    if backend == "cupy":
        import cupy
        return cupy.full(shape, fill, dtype="float32")
    if backend == "dask+cupy":
        import cupy
        import dask.array as da
        meta = cupy.empty((0, 0), dtype="float32")
        return da.full(shape, fill, dtype="float32",
                       chunks=chunks).map_blocks(cupy.asarray, meta=meta)
    raise ValueError(
        f"Unknown backend {backend!r}; choose from 'numpy', 'dask+numpy', "
        f"'cupy', 'dask+cupy'."
    )


def _estimate_n_chunks(shape, chunks):
    """Number of blocks ``da.full`` would build for ``shape`` and ``chunks``.

    Uses dask's own ``normalize_chunks`` so every ``chunks`` form -- ``'auto'``
    (where dask picks the block shape from the chunk-size config), an int, a
    tuple, or a dict -- resolves to the same block grid the real array would
    have. dtype is ``float32`` to match the grid ``_make_data`` builds. The call
    only sizes the block tuples, so it stays cheap even when the count is huge.
    """
    from dask.array.core import normalize_chunks
    normalized = normalize_chunks(chunks, shape=shape, dtype="float32")
    n = 1
    for dim in normalized:
        n *= len(dim)
    return n


def _cf_crs_attrs(crs):
    """CF Conventions grid-mapping attributes for an EPSG code.

    Emits the two canonical CF-1.7 identifiers via :meth:`pyproj.CRS.to_cf`:
    ``grid_mapping_name`` (the controlled-vocabulary projection token, e.g.
    ``'albers_conical_equal_area'``) and ``crs_wkt`` (the full WKT, which
    carries the human-readable CRS name). This mirrors the rest of the
    library, which identifies a CRS by ``crs`` / ``crs_wkt`` and derives
    anything else with pyproj on demand.

    Best-effort: without pyproj installed the mapping is empty, so the
    default (non-reproject) path stays dependency-free, same as the old
    ``crs_name`` behaviour. ``grid_mapping_name`` is also left out for
    projections CF does not define (e.g. Equal Earth), where ``to_cf``
    returns ``crs_wkt`` alone.
    """
    try:
        import pyproj
    except ImportError:
        return {}
    cf = pyproj.CRS.from_epsg(int(crs)).to_cf()
    return {k: cf[k] for k in ("grid_mapping_name", "crs_wkt")
            if cf.get(k) is not None}


def list_templates(kind: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
    """List the template names ``from_template`` accepts.

    Every name in the result is a valid ``from_template`` argument: region and
    city names are lowercase, country codes are uppercase ISO-3166 / GADM
    alpha-3.

    Parameters
    ----------
    kind : {'regions', 'cities', 'countries'}, optional
        Return just one group as a sorted list. When omitted (the default),
        return a dict mapping each group to its sorted list of names.

    Returns
    -------
    dict of str to list of str, or list of str
        With ``kind=None``, ``{'regions': [...], 'cities': [...],
        'countries': [...]}``. With ``kind`` set, the sorted list for that
        group.

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial.templates import list_templates
        >>> names = list_templates()
        >>> sorted(names)
        ['cities', 'countries', 'regions']
        >>> 'conus' in names['regions']
        True
        >>> 'london' in list_templates('cities')
        True
    """
    groups = {
        "regions": sorted(set(_REGIONS) | set(_REGION_ALIASES)),
        "cities": sorted(_CITIES),
        "countries": sorted(_COUNTRY_BBOXES),
    }
    if kind is None:
        return groups
    if kind not in groups:
        raise ValueError(
            f"kind must be one of {tuple(groups)} or None, got {kind!r}."
        )
    return groups[kind]


def from_template(name: str,
                  resolution: Optional[Union[float, Tuple[float, float]]] = None,
                  *, height: Optional[int] = None, width: Optional[int] = None,
                  preserve: Optional[str] = None, backend: str = "numpy",
                  fill: float = np.nan,
                  chunks: Optional[Union[int, str, Tuple]] = None) -> xr.DataArray:
    """Create an empty DataArray for a common study area.

    The returned raster is NaN-filled and obeys the xarray-spatial array
    contract: a 2-D ``['y', 'x']`` grid with pixel-center 1-D coordinates
    (north-up, descending ``y``) and ``res``/``crs`` attributes. It covers the
    study area's rectangular bounding box and is meant as a starting canvas.

    The requested ``resolution`` is honored exactly. The study-area box is
    rarely a whole number of cells wide, so the far edges (right, top) are
    nudged out by up to half a cell to land on an exact multiple of the cell
    size, anchoring the lower-left corner. ``attrs['res']`` therefore matches
    the ``resolution`` you pass.

    Parameters
    ----------
    name : str
        A curated region name (case-insensitive): a national/metro area such as
        ``'conus'`` or ``'nyc'``, a continental or subcontinental region such as
        ``'europe'``, ``'southeast_asia'``, ``'east_africa'``, or
        ``'south_america'`` (call :func:`list_templates` for the full set), or
        ``'world'``; a global-projection name, e.g. ``'web_mercator'``
        (EPSG:3857), ``'wgs84'`` / ``'latlon'`` (EPSG:4326, the same grid as
        ``'world'``), ``'equal_earth'`` (EPSG:8857), or ``'pacific'`` / ``'pdc'``
        (EPSG:3832, a Pacific-centered PDC Mercator); a world-city name
        (case-insensitive), e.g. ``'london'``,
        ``'tokyo'``, ``'sao_paulo'``; or an ISO-3166 / GADM alpha-3 country
        code, e.g. ``'USA'``, ``'FRA'``, ``'JPN'``. Curated regions and cities
        come back in a projected CRS (cities in their UTM zone); country codes
        come back in EPSG:4326. Where two cities share a name the larger keeps
        the bare name and the others take a ``_<iso2>`` suffix (e.g.
        ``'hyderabad'`` vs ``'hyderabad_pk'``).
    resolution : float or tuple of float, optional
        Cell size in the template's CRS units (metres for projected regions,
        degrees for country codes). A scalar gives square cells; a
        ``(res_x, res_y)`` tuple sets each axis. Defaults to a per-template
        value so a bare ``from_template('conus')`` works. Ignored on the
        ``height`` / ``width`` path unless given (see below).
    height, width : int, optional
        Grid shape in cells. Supply both (or neither). When given, the result
        is exactly ``height`` x ``width`` cells anchored at the region's
        lower-left corner, and the extent floats off that anchor rather than
        snapping to the study-area box: with ``resolution`` the extent is
        ``shape x resolution``; without it the resolution is derived so the
        exact shape spans the region bbox. Use this to ask for a
        tiling-friendly shape directly.
    preserve : {'area', 'shape'}, optional
        Reproject the template into an EPSG-coded projection chosen for the
        property it preserves, instead of the template's default CRS. ``'area'``
        gives an equal-area projection (a curated national/continental code such
        as EPSG:5070, or EPSG:8857 Equal Earth where none is curated).
        ``'shape'`` gives a conformal projection: the centroid's UTM zone (UPS at
        the poles). When set, ``resolution`` is in metres and ``attrs['crs']`` is
        the chosen EPSG int. Requires pyproj.
    backend : str, default='numpy'
        Array backend: ``'numpy'``, ``'dask+numpy'`` (alias ``'dask'``),
        ``'cupy'``, or ``'dask+cupy'``. The eager backends (``'numpy'``,
        ``'cupy'``) materialize the whole grid, so they are subject to a
        500-million-cell cap; the dask backends build a lazy chunk graph and
        are not.
    fill : float, default=numpy.nan
        Value the grid is filled with. The dtype is always ``float32``.
    chunks : int, str, or tuple, optional
        Dask chunk specification. Supplying it returns a lazy, chunked grid:
        an eager backend is promoted to its dask variant (``'numpy'`` to
        ``'dask+numpy'``, ``'cupy'`` to ``'dask+cupy'``), and the cell cap no
        longer applies. When omitted (or ``'auto'``), the dask backends tile
        the grid into even, square-ish blocks (~2048 cells per side) tuned for
        the neighborhood ops -- ``slope``, ``hillshade``, ``focal`` -- that run
        on the result through ``map_overlap``. To make that tiling exact, a
        multi-block grid has its extent padded out from the lower-left anchor
        so each axis is a whole number of blocks: every chunk is full-size with
        no ragged remainder, the requested ``resolution`` is unchanged, and the
        padded grid still covers the study area (it only grows). The padding is
        dask-only -- eager grids and explicit ``height`` / ``width`` keep the
        exact bbox-derived shape -- and a grid small enough to not be worth
        splitting stays a single chunk, unpadded.
        Pass an explicit value to override the tiling. The data stays lazy, but
        a very fine resolution still builds one task per chunk, so an extreme
        shape with small explicit chunks can make a task graph large enough to
        bog down the client; a grid that would split into more than 1,000,000
        chunks raises ``ValueError``. The default tiling grows its block for
        such grids so it never trips that cap on its own.

    Returns
    -------
    template : xarray.DataArray
        Empty 2-D raster with ``dims=('y', 'x')`` and pixel-center
        coordinates. ``attrs`` carries ``res`` and ``crs`` plus the CF
        Conventions grid-mapping keys ``grid_mapping_name`` (the projection
        token, e.g. ``'albers_conical_equal_area'``) and ``crs_wkt`` (full
        WKT, which carries the human-readable CRS name). The ``x`` / ``y``
        coordinates follow CF axis conventions: ``units`` ``'m'`` with
        ``standard_name`` ``'projection_x_coordinate'`` /
        ``'projection_y_coordinate'`` for projected templates, or
        ``'degrees_east'`` / ``'degrees_north'`` with ``standard_name``
        ``'longitude'`` / ``'latitude'`` for EPSG:4326. The grid-mapping keys
        require pyproj; without it they are omitted (the default,
        dependency-free path), and ``grid_mapping_name`` is also omitted for
        projections CF does not define (e.g. Equal Earth), leaving
        ``crs_wkt`` alone. These keys sit directly on ``attrs`` (the library's
        flat CRS convention), not on a separate CF grid-mapping variable, so a
        strict CF reader will not auto-detect them as a grid mapping.

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial import from_template
        >>> agg = from_template("conus")            # Albers, default 5 km cells
        >>> agg.attrs["crs"]
        5070
        >>> agg = from_template("conus", resolution=1000)   # 1 km cells
        >>> agg = from_template("FRA")              # France bbox in EPSG:4326
        >>> agg.attrs["crs"]
        4326
        >>> from_template("web_mercator").attrs["crs"]   # global EPSG:3857
        3857
        >>> from_template("latlon").attrs["crs"]         # alias for wgs84
        4326
        >>> from_template("equal_earth").attrs["crs"]    # global equal-area
        8857
        >>> from_template("london").attrs["crs"]    # greater London, UTM 30N
        32630
        >>> from_template("FRA", preserve="shape").attrs["crs"]   # UTM 30N
        32630
        >>> from_template("FRA", preserve="area").attrs["crs"]    # Equal Earth
        8857
        >>> # passing chunks returns a lazy dask grid, exempt from the cell cap
        >>> agg = from_template("new_england", resolution=10, chunks=512)
        >>> type(agg.data).__name__
        'Array'
        >>> # ask for an exact tiling-friendly shape; the extent floats
        >>> from_template("conus", resolution=1000, height=4096, width=6144).shape
        (4096, 6144)

    Recipes
    -------
    Pick a grid, drop your own data onto it with
    :meth:`DataArray.xrs.coregister <xrspatial.accessor.XrsSpatialDataArrayAccessor.coregister>`,
    then run any tool. ``coregister`` reprojects a raster or rasterizes a
    GeoDataFrame onto the template's exact grid, so layers line up
    cell-for-cell.

    .. sourcecode:: python

        >>> grid = from_template("conus", resolution=1000)
        >>> elevation = grid.xrs.coregister(my_dem)        # raster -> grid
        >>> roads = grid.xrs.coregister(my_roads_gdf)      # vectors -> grid
        >>> slope = elevation.xrs.slope()

    For an out-of-core workflow, ask for a dask backend; the default tiling
    keeps the downstream graph parallel and overlap-friendly:

    .. sourcecode:: python

        >>> grid = from_template("conus", resolution=250, backend="dask")
        >>> slope = grid.xrs.coregister(my_dem).xrs.slope()  # stays lazy
    """
    spec = _resolve(name)
    key = spec["key"]

    if preserve is not None:
        if not isinstance(preserve, str) or preserve.lower() \
                not in _PRESERVE_OPTIONS:
            raise ValueError(
                f"preserve must be one of {_PRESERVE_OPTIONS} or None, "
                f"got {preserve!r}. (No EPSG-coded projection exists for "
                f"distance/direction, so those are not offered.)"
            )
        preserve = preserve.lower()
        crs = _preserve_epsg(spec["lonlat"], preserve,
                             spec["area_epsg"], spec["shape_epsg"])
        bounds = _project_bbox_bounds(spec["lonlat"], crs)
        default_res = _default_preserve_resolution(bounds)
    else:
        bounds = spec["bounds"]
        crs = spec["crs"]
        default_res = spec["default_resolution"]

    left, bottom, right, top = bounds

    # height/width are an all-or-nothing pair that fixes the grid shape exactly.
    explicit_shape = height is not None or width is not None
    if explicit_shape and (height is None or width is None):
        raise ValueError(
            "supply both height and width together, or neither"
        )

    if explicit_shape:
        # Exact shape: the grid is height x width cells anchored at the region's
        # lower-left corner, and the extent floats off that anchor. With a
        # resolution the extent is shape x resolution; without one the
        # resolution is derived so the exact shape spans the region bbox.
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"height and width must be positive, got {(height, width)}"
            )
        if resolution is None:
            res_x = (right - left) / width
            res_y = (top - bottom) / height
        else:
            res_x, res_y = _normalize_resolution(resolution, default_res)
    else:
        res_x, res_y = _normalize_resolution(resolution, default_res)
        width = max(1, int(round((right - left) / res_x)))
        height = max(1, int(round((top - bottom) / res_y)))

    # Supplying `chunks` signals intent for a lazy, chunked grid: promote an
    # eager backend to its dask variant so the result never materializes.
    backend = backend.lower()
    if chunks is not None and backend in ("numpy", "cupy"):
        backend = "dask+numpy" if backend == "numpy" else "dask+cupy"
    is_dask = backend in ("dask", "dask+numpy", "dask+cupy")

    # Default dask tiling pads the shape out to whole blocks so every chunk is
    # full-size with no ragged remainder. Only on a dask backend under the
    # default tiling, and never when the caller fixed the shape with
    # height/width or drove the tiling with an explicit chunks=. Reuse the same
    # block edge for the padding and the chunking so they agree exactly.
    default_tiling = chunks is None or chunks == "auto"
    block = None
    if is_dask and default_tiling and not explicit_shape:
        block = _block_edge(height * width)
        height, width = _pad_to_tiles((height, width), block)

    # Name the knob the caller actually set when a cap is hit: height/width on
    # the explicit-shape path (its resolution is derived), resolution otherwise.
    if explicit_shape:
        shape_desc = f"height={height}, width={width}"
        coarsen = "Use a smaller height/width"
    else:
        shape_desc = f"resolution {(res_x, res_y)}"
        coarsen = "Use a coarser resolution"

    n_cells = width * height
    if not is_dask and n_cells > _MAX_CELLS:
        raise ValueError(
            f"{shape_desc} produces a {height} x {width} grid "
            f"({n_cells:,} cells), exceeding the {_MAX_CELLS:,}-cell limit. "
            f"{coarsen}, or pass chunks=... for a lazy dask grid."
        )

    # 'auto' (the default) tiles into even square blocks tuned for the
    # neighborhood ops that run on the result; an explicit chunks= is honored
    # verbatim. Either way the block count is checked against _MAX_CHUNKS below.
    if default_tiling:
        effective_chunks = _neighborhood_chunks((height, width), block) \
            if is_dask else "auto"
    else:
        effective_chunks = chunks
    if is_dask:
        n_chunks = _estimate_n_chunks((height, width), effective_chunks)
        if n_chunks > _MAX_CHUNKS:
            # Report the request, not the expanded per-block tuple the default
            # tiling produces, so the message stays readable.
            chunks_display = chunks if chunks not in (None, "auto") \
                else f"the default ~{_DASK_BLOCK}-cell tiling"
            raise ValueError(
                f"{shape_desc} produces a {height} x {width} "
                f"grid that splits into {n_chunks:,} chunks with "
                f"{chunks_display!r}, exceeding the "
                f"{_MAX_CHUNKS:,}-chunk limit. A task graph this large can bog "
                f"down the client even though no data is computed. "
                f"{coarsen} or use larger chunks."
            )

    # Honor the requested resolution exactly: anchor the lower-left corner and
    # nudge the far edges to an exact multiple of the cell size, so res comes
    # back as (res_x, res_y) instead of drifting when the bbox extent isn't a
    # whole number of cells. Mirrors the reproject grid path
    # (_compute_output_grid in reproject/_grid.py).
    right = left + width * res_x
    top = bottom + height * res_y
    ys, xs = _make_output_coords((left, bottom, right, top), (height, width))

    data = _make_data((height, width), fill, backend, effective_chunks)

    attrs = {"res": (res_x, res_y), "crs": crs}
    attrs.update(_cf_crs_attrs(crs))

    template = xr.DataArray(
        data,
        name=key,
        coords={"y": ys, "x": xs},
        dims=["y", "x"],
        attrs=attrs,
    )
    # CF coordinate metadata (CF Conventions sec. 4). Every template emits
    # either EPSG:4326 (country codes) or a metre-based projected CRS
    # (curated regions and all preserve paths).
    if crs == 4326:
        template["x"].attrs.update(units="degrees_east", standard_name="longitude")
        template["y"].attrs.update(units="degrees_north", standard_name="latitude")
    else:
        template["x"].attrs.update(units="m", standard_name="projection_x_coordinate")
        template["y"].attrs.update(units="m", standard_name="projection_y_coordinate")
    return template
