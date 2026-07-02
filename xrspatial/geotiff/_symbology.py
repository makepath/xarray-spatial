"""Symbology sidecars for continuous (non-categorical) rasters.

A continuous single-band GeoTIFF opens in QGIS as a flat grayscale stretch
unless QGIS finds styling next to the file. Two sidecars give it a sensible
default color ramp on open:

* a QGIS ``.qml`` style file (``<base>.qml``) with a ``singlebandpseudocolor``
  renderer -- this is what makes QGIS draw real colors, and
* band statistics (min/max/mean/stddev) in the PAM ``<file>.aux.xml`` sidecar
  (written via :mod:`._pam`), which GDAL tools and QGIS use for a min/max
  stretch even when the ``.qml`` is ignored.

This is the continuous-raster counterpart to the categorical RAT sidecar in
:mod:`._pam`. Color ramps are hardcoded control points (no matplotlib runtime
dependency); ``viridis`` is the default.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import utils

# Named color ramps, each sampled at 9 evenly spaced control points
# (matplotlib colormaps, committed as static constants so we carry no
# matplotlib runtime dependency). Each entry is (position 0..1, (r, g, b)).
_COLOR_RAMPS = {
    'viridis': [(0.0, (68, 1, 84)), (0.125, (71, 45, 123)), (0.25, (59, 82, 139)),
                (0.375, (44, 114, 142)), (0.5, (33, 145, 140)),
                (0.625, (40, 174, 128)), (0.75, (94, 201, 98)),
                (0.875, (173, 220, 48)), (1.0, (253, 231, 37))],
    'plasma': [(0.0, (13, 8, 135)), (0.125, (76, 2, 161)), (0.25, (126, 3, 168)),
               (0.375, (170, 35, 149)), (0.5, (204, 71, 120)),
               (0.625, (230, 108, 92)), (0.75, (248, 149, 64)),
               (0.875, (253, 197, 39)), (1.0, (240, 249, 33))],
    'magma': [(0.0, (0, 0, 4)), (0.125, (29, 17, 71)), (0.25, (81, 18, 124)),
              (0.375, (131, 38, 129)), (0.5, (183, 55, 121)),
              (0.625, (231, 82, 99)), (0.75, (252, 137, 97)),
              (0.875, (254, 196, 136)), (1.0, (252, 253, 191))],
    'inferno': [(0.0, (0, 0, 4)), (0.125, (33, 12, 74)), (0.25, (87, 16, 110)),
                (0.375, (138, 34, 106)), (0.5, (188, 55, 84)),
                (0.625, (228, 90, 49)), (0.75, (249, 142, 9)),
                (0.875, (249, 203, 53)), (1.0, (252, 255, 164))],
    'cividis': [(0.0, (0, 34, 78)), (0.125, (26, 56, 111)), (0.25, (67, 78, 108)),
                (0.375, (97, 101, 111)), (0.5, (125, 124, 120)),
                (0.625, (155, 148, 118)), (0.75, (188, 174, 108)),
                (0.875, (222, 201, 88)), (1.0, (254, 232, 56))],
    'greys': [(0.0, (255, 255, 255)), (0.125, (240, 240, 240)),
              (0.25, (217, 217, 217)), (0.375, (189, 189, 189)),
              (0.5, (149, 149, 149)), (0.625, (114, 114, 114)),
              (0.75, (81, 81, 81)), (0.875, (36, 36, 36)), (1.0, (0, 0, 0))],
    'spectral': [(0.0, (158, 1, 66)), (0.125, (221, 74, 76)), (0.25, (249, 142, 82)),
                 (0.375, (254, 212, 129)), (0.5, (255, 255, 190)),
                 (0.625, (214, 238, 155)), (0.75, (134, 207, 165)),
                 (0.875, (61, 149, 184)), (1.0, (94, 79, 162))],
    'terrain': [(0.0, (51, 51, 153)), (0.125, (8, 136, 238)), (0.25, (1, 204, 102)),
                (0.375, (129, 230, 128)), (0.5, (254, 254, 152)),
                (0.625, (190, 172, 118)), (0.75, (129, 94, 86)),
                (0.875, (193, 176, 172)), (1.0, (255, 255, 255))],
}

_DEFAULT_RAMP = 'viridis'


def resolve_ramp(name):
    """Return the control-point list for color ramp *name*.

    ``True`` selects the default ramp (``viridis``). Unknown names raise
    ``ValueError`` listing the available ramps so a typo fails fast, before
    any bytes are written.
    """
    if name is True:
        name = _DEFAULT_RAMP
    key = str(name).lower()
    try:
        return _COLOR_RAMPS[key]
    except KeyError:
        raise ValueError(
            f"unknown color_ramp {name!r}; choose from {sorted(_COLOR_RAMPS)}")


def qml_path(path: str) -> str:
    """Return the QGIS style sidecar path QGIS auto-loads for *path*.

    QGIS builds its default-style path from ``completeBaseName + '.qml'``: it
    *replaces* the extension (``dem.tif`` -> ``dem.qml``), unlike the PAM
    ``.aux.xml`` sidecar which is *appended* (``dem.tif.aux.xml``). Writing the
    appended form here would leave the style silently unloaded.
    """
    return os.path.splitext(path)[0] + '.qml'


def _num(value) -> str:
    """Format a number for QML/PAM with enough precision to round-trip."""
    return '{:.10g}'.format(float(value))


def build_qml(stops, vmin, vmax) -> str:
    """Build a QGIS ``.qml`` singlebandpseudocolor style for one band.

    *stops* is a ``[(pos, (r, g, b)), ...]`` ramp; each stop becomes a
    ``<colorrampshader>`` item at ``vmin + pos * (vmax - vmin)`` (the item list
    is what QGIS actually renders). The sibling ``<colorramp>`` block lets the
    QGIS style dialog show the ramp as an editable gradient.
    """
    vmin = float(vmin)
    vmax = float(vmax)
    span = vmax - vmin

    items = []
    for pos, (r, g, b) in stops:
        value = vmin + pos * span
        color = '#%02x%02x%02x' % (r, g, b)
        items.append(
            f'          <item value="{_num(value)}" color="{color}" '
            f'alpha="255" label="{_num(value)}"/>')
    item_xml = '\n'.join(items)

    color1 = '%d,%d,%d,255' % stops[0][1]
    color2 = '%d,%d,%d,255' % stops[-1][1]
    interior = ':'.join(
        '%s;%d,%d,%d,255' % (_num(pos), r, g, b)
        for pos, (r, g, b) in stops[1:-1])

    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" \
classificationMin="{_num(vmin)}" classificationMax="{_num(vmax)}" \
alphaBand="-1" opacity="1" nodataColor="">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" \
clip="0" minimumValue="{_num(vmin)}" maximumValue="{_num(vmax)}">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" name="color1" value="{color1}"/>
              <Option type="QString" name="color2" value="{color2}"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
              <Option type="QString" name="stops" value="{interior}"/>
            </Option>
          </colorramp>
{item_xml}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def _is_single_band(data) -> bool:
    """True when *data* is a single-band raster (2D, or 3D with one band)."""
    if data.ndim == 2:
        return True
    if data.ndim == 3:
        from ._coords import _BAND_DIM_NAMES
        for dim, size in zip(data.dims, data.shape):
            if dim in _BAND_DIM_NAMES:
                return size == 1
    return False


def _finite_stats(data, nodata=None):
    """Return ``(min, max, mean, stddev)`` over finite, non-nodata values.

    Population stddev (``ddof=0``), matching GDAL's ``STATISTICS_STDDEV``.
    Returns ``None`` when no finite values remain. Backend-aware (numpy, cupy,
    dask+numpy, dask+cupy); the dask path fuses the reductions into a single
    ``dask.compute`` so the source graph is read once, not four times.
    """
    arr = getattr(data, 'data', data)
    has_nodata = nodata is not None and not (
        isinstance(nodata, float) and math.isnan(nodata))

    if utils.has_dask_array():
        import dask.array as dask_array
        if isinstance(arr, dask_array.Array):
            return _dask_finite_stats(arr, nodata if has_nodata else None)
    return _eager_finite_stats(arr, nodata if has_nodata else None)


def _eager_finite_stats(arr, nodata):
    """``_finite_stats`` for an in-memory numpy or cupy array."""
    xp = np
    if utils.is_cupy_array(arr):
        import cupy
        xp = cupy

    mask = xp.isfinite(arr)
    if nodata is not None:
        mask = mask & (arr != nodata)
    vals = arr[mask]
    if vals.size == 0:
        return None
    return (utils._to_float_scalar(vals.min()),
            utils._to_float_scalar(vals.max()),
            utils._to_float_scalar(vals.mean()),
            utils._to_float_scalar(vals.std()))


def _dask_finite_stats(arr, nodata):
    """``_finite_stats`` for a dask array (numpy- or cupy-backed)."""
    import dask
    import dask.array as dask_array

    mask = dask_array.isfinite(arr)
    if nodata is not None:
        mask = mask & (arr != nodata)
    # ``where`` keeps the array shape and NaNs out excluded cells so the nan*
    # reductions ignore them; integer inputs upcast to float, which is fine
    # for statistics.
    vals = dask_array.where(mask, arr, np.nan)
    count, mn, mx, mean, std = dask.compute(
        mask.sum(),
        dask_array.nanmin(vals),
        dask_array.nanmax(vals),
        dask_array.nanmean(vals),
        dask_array.nanstd(vals),
    )
    if utils._to_float_scalar(count) == 0:
        return None
    return (utils._to_float_scalar(mn), utils._to_float_scalar(mx),
            utils._to_float_scalar(mean), utils._to_float_scalar(std))


class StreamingStats:
    """One-pass ``(min, max, mean, stddev)`` accumulator over streamed buffers.

    The streaming dask writer already materialises every pixel exactly once
    (issue #3597); feeding those buffers through :meth:`update` yields the
    same statistics as :func:`_finite_stats` without a second execution of
    the source graph. Matches ``_finite_stats`` semantics: non-finite values
    and the *nodata* sentinel (unless it is NaN) are excluded, and the
    stddev is the population stddev (``ddof=0``).

    Mean and variance combine across buffers with Chan's parallel formula,
    accumulated in float64, so the result is stable regardless of how the
    writer bands / segments the raster.
    """

    def __init__(self, nodata=None):
        if isinstance(nodata, float) and math.isnan(nodata):
            nodata = None
        self._nodata = nodata
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf

    def update(self, buf):
        """Fold one materialised numpy buffer into the running statistics."""
        if buf.dtype.kind == 'f':
            mask = np.isfinite(buf)
            if self._nodata is not None:
                mask &= (buf != self._nodata)
            # ``buf[mask]`` copies; skip it for the common all-valid buffer.
            vals = buf.ravel() if bool(mask.all()) else buf[mask]
        elif self._nodata is not None:
            mask = buf != self._nodata
            vals = buf.ravel() if bool(mask.all()) else buf[mask]
        else:
            vals = buf.ravel()
        # Plain ``int`` so the running count (and the moment arithmetic
        # built on it) stays in unbounded Python ints, not numpy scalars.
        n_b = int(vals.size)
        if n_b == 0:
            return
        # Reductions with float64 accumulators on the original buffer:
        # an ``astype(float64)`` copy here would double the memory
        # traffic of every update for no precision gain.
        mean_b = float(vals.mean(dtype=np.float64))
        m2_b = float(vals.var(dtype=np.float64)) * n_b
        self._min = min(self._min, float(vals.min()))
        self._max = max(self._max, float(vals.max()))
        if self._count == 0:
            self._count, self._mean, self._m2 = n_b, mean_b, m2_b
        else:
            total = self._count + n_b
            delta = mean_b - self._mean
            self._m2 += m2_b + delta * delta * self._count * n_b / total
            self._mean += delta * n_b / total
            self._count = total

    def result(self):
        """Return ``(min, max, mean, stddev)``, or ``None`` if nothing valid
        was accumulated -- the same contract as :func:`_finite_stats`."""
        if self._count == 0:
            return None
        std = math.sqrt(max(self._m2 / self._count, 0.0))
        return (self._min, self._max, self._mean, std)


def write_symbology_sidecars(path, data, *, stops, nodata=None,
                             ramp_range=None, stats=None):
    """Write continuous-raster symbology sidecars next to *path*.

    Writes band statistics into the PAM ``.aux.xml`` and, when the data range
    is non-degenerate, a QGIS ``.qml`` color-ramp style. No-op for multiband
    arrays or arrays with no finite data.

    When *ramp_range* ``(vmin, vmax)`` is given it sets the ramp/stretch bounds
    directly and skips the statistics reduction -- the escape hatch for large
    dask graphs -- so only ``STATISTICS_MINIMUM`` / ``STATISTICS_MAXIMUM`` are
    written (mean/stddev would need the pass it avoids).

    *stats* is an optional :class:`StreamingStats` that already accumulated
    the statistics during the write (the streaming dask path, issue #3597);
    when given, its result replaces the :func:`_finite_stats` reduction so
    the source graph is not executed a second time.
    """
    from . import _pam

    if not _is_single_band(data):
        return

    if ramp_range is not None:
        vmin, vmax = float(ramp_range[0]), float(ramp_range[1])
        stats_dict = {'STATISTICS_MINIMUM': vmin, 'STATISTICS_MAXIMUM': vmax}
    else:
        result = (stats.result() if stats is not None
                  else _finite_stats(data, nodata))
        if result is None:
            return
        vmin, vmax, vmean, vstd = result
        stats_dict = {
            'STATISTICS_MINIMUM': vmin,
            'STATISTICS_MAXIMUM': vmax,
            'STATISTICS_MEAN': vmean,
            'STATISTICS_STDDEV': vstd,
        }

    _pam.write_stats_pam_sidecar(path, stats_dict)

    # A constant raster (vmin == vmax) has no range to ramp across, so the
    # stats stretch is the useful part and the QML would be a degenerate
    # single-stop ramp -- skip it.
    if vmax > vmin:
        with open(qml_path(path), 'w', encoding='utf-8') as fh:
            fh.write(build_qml(stops, vmin, vmax))
