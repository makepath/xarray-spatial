"""xarray accessors for xarray-spatial.

Registers ``.xrs`` accessors on :class:`xarray.DataArray` and
:class:`xarray.Dataset` so that spatial operations can be called
directly via tab-completion::

    import xrspatial  # registers accessors

    raster.xrs.slope()
    raster.xrs.hillshade(azimuth=315)
    nir.xrs.ndvi(red)
"""

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


def _infer_caller_y_chunk(obj):
    """First y-axis chunk size of a dask-backed DataArray, or None."""
    try:
        y_axis = obj.get_axis_num('y')
    except (ValueError, KeyError):
        return None
    chunks_tuple = getattr(obj, 'chunks', None)
    if not chunks_tuple:
        return None
    try:
        y_chunks = chunks_tuple[y_axis]
    except (IndexError, TypeError):
        return None
    if not y_chunks:
        return None
    return int(y_chunks[0])


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


def _open_geotiff_windowed(obj, source, *, auto_reproject=False, **kwargs):
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
    from .geotiff import open_geotiff, _read_geo_info, _extent_to_window
    from .utils import _classify_backend

    if 'y' not in obj.coords or 'x' not in obj.coords:
        raise ValueError(
            "Caller must have 'y' and 'x' coordinates to compute a "
            "spatial window"
        )
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

    if crs_mismatch and not auto_reproject:
        raise ValueError(
            f"CRS mismatch: caller has {caller_crs_raw!r} but file has "
            f"EPSG:{file_crs_raw}. Pass auto_reproject=True to project "
            f"the caller bbox into the file CRS for the windowed read "
            f"and reproject the result back to the caller CRS."
        )

    if crs_mismatch:
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

    # Expand extent by half a pixel so we capture edge pixels
    y_min -= abs(t.pixel_height) * 0.5
    y_max += abs(t.pixel_height) * 0.5
    x_min -= abs(t.pixel_width) * 0.5
    x_max += abs(t.pixel_width) * 0.5

    window = _extent_to_window(t, file_h, file_w,
                               y_min, y_max, x_min, x_max)
    kwargs.pop('window', None)

    # Infer backend kwargs. Caller-supplied values always win.
    backend = _classify_backend(obj)
    if backend in ("cupy", "dask+cupy"):
        kwargs.setdefault('gpu', True)
    if backend in ("dask+numpy", "dask+cupy"):
        inferred_chunk = _infer_caller_y_chunk(obj)
        if inferred_chunk is not None:
            kwargs.setdefault('chunks', inferred_chunk)

    result = open_geotiff(source, window=window, **kwargs)

    if crs_mismatch:
        from .reproject import reproject
        result = reproject(
            result,
            target_crs=caller_crs,
            source_crs=file_crs,
        )

    return result


@xr.register_dataarray_accessor("xrs")
class XrsSpatialDataArrayAccessor:
    """DataArray accessor exposing xarray-spatial operations."""

    def __init__(self, obj):
        self._obj = obj

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

    def idw(self, x, y, z, **kwargs):
        from .interpolate import idw
        return idw(x, y, z, self._obj, **kwargs)

    def kriging(self, x, y, z, **kwargs):
        from .interpolate import kriging
        return kriging(x, y, z, self._obj, **kwargs)

    def spline(self, x, y, z, **kwargs):
        from .interpolate import spline
        return spline(x, y, z, self._obj, **kwargs)

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

    def rasterize(self, geometries, **kwargs):
        from .rasterize import rasterize
        return rasterize(geometries, like=self._obj, **kwargs)

    # ---- GeoTIFF I/O ----

    def to_geotiff(self, path, **kwargs):
        """Write this DataArray as a GeoTIFF.

        Equivalent to ``to_geotiff(da, path, **kwargs)``.

        See :func:`xrspatial.geotiff.to_geotiff` for full parameter docs.
        """
        from .geotiff import to_geotiff
        return to_geotiff(self._obj, path, **kwargs)

    def open_geotiff(self, source, *, auto_reproject=False, **kwargs):
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
        **kwargs
            Forwarded to :func:`xrspatial.geotiff.open_geotiff` (except
            ``window=``, which is computed automatically).

        Returns
        -------
        xr.DataArray
            The windowed portion of the GeoTIFF, in ``self``'s CRS when
            ``auto_reproject`` reprojection occurred and otherwise in the
            file's native CRS.
        """
        return _open_geotiff_windowed(
            self._obj, source, auto_reproject=auto_reproject, **kwargs
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

    def rasterize(self, geometries, **kwargs):
        from .rasterize import rasterize
        ds = self._obj
        # Find a 2D variable with y/x dims to use as template
        for var in ds.data_vars:
            da = ds[var]
            if da.ndim == 2 and 'y' in da.dims and 'x' in da.dims:
                return rasterize(geometries, like=da, **kwargs)
        raise ValueError(
            "Dataset has no 2D variable with 'y' and 'x' dimensions "
            "to use as rasterize template"
        )

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

    def open_geotiff(self, source, *, auto_reproject=False, var=None, **kwargs):
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
            rep, source, auto_reproject=auto_reproject, **kwargs
        )

    # ---- Chunking ----

    def rechunk_no_shuffle(self, **kwargs):
        from .utils import rechunk_no_shuffle
        return rechunk_no_shuffle(self._obj, **kwargs)


# ---------------------------------------------------------------------------
# Surface standalone-function docstrings on accessor methods so that, e.g.,
# ``help(da.xrs.slope)`` shows the same documentation as ``help(slope)``.
# ---------------------------------------------------------------------------

# Accessor method name -> name of the standalone function (in the top-level
# ``xrspatial`` namespace) whose docstring should be surfaced. Only needed
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
