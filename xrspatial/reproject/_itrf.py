"""Time-dependent ITRF frame transformations.

Implements 14-parameter Helmert transforms (7 static + 7 rates)
for converting between International Terrestrial Reference Frames.

The parameters are published by IGN France and shipped with PROJ.
Shifts are mm-level for position and mm/year for rates -- relevant
for precision geodesy, negligible for most raster reprojection.

Usage
-----
>>> from xrspatial.reproject import itrf_transform
>>> lon2, lat2, h2 = itrf_transform(
...     -74.0, 40.7, 0.0,
...     src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
... )
"""
from __future__ import annotations

import math
import os
import re
import threading

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Parse PROJ ITRF parameter files
# ---------------------------------------------------------------------------

def _find_proj_data_dir():
    """Locate the PROJ data directory."""
    try:
        import pyproj
        return pyproj.datadir.get_data_dir()
    except Exception:
        return None


def _parse_itrf_file(path):
    """Parse a PROJ ITRF parameter file.

    Returns dict mapping target_frame -> parameter dict.
    """
    transforms = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('<metadata>'):
                continue
            # Format: <TARGET_FRAME> +proj=helmert +x=... +dx=... +t_epoch=...
            m = re.match(r'<(\w+)>\s+(.+)', line)
            if not m:
                continue
            target = m.group(1)
            params_str = m.group(2)
            params = {}
            for token in params_str.split():
                if '=' in token:
                    key, val = token.lstrip('+').split('=', 1)
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val
                elif token.startswith('+'):
                    params[token.lstrip('+')] = True
            transforms[target] = params
    return transforms


def _load_all_itrf_params():
    """Load all ITRF transformation parameters from PROJ data files.

    Returns a nested dict: {source_frame: {target_frame: params}}.
    """
    proj_dir = _find_proj_data_dir()
    if proj_dir is None:
        return {}

    all_params = {}
    for filename in os.listdir(proj_dir):
        if not filename.startswith('ITRF'):
            continue
        source_frame = filename
        path = os.path.join(proj_dir, filename)
        if not os.path.isfile(path):
            continue
        transforms = _parse_itrf_file(path)
        all_params[source_frame] = transforms

    return all_params


# Lazy-loaded parameter cache
_itrf_params = None
_itrf_params_lock = threading.Lock()


def _get_itrf_params():
    global _itrf_params
    with _itrf_params_lock:
        if _itrf_params is None:
            _itrf_params = _load_all_itrf_params()
        return _itrf_params


def _find_transform(src, tgt):
    """Find the 14-parameter Helmert from src to tgt frame.

    Returns parameter dict or None. Tries direct lookup first,
    then reverse (with negated parameters).
    """
    params = _get_itrf_params()

    # Direct: src file contains entry for tgt
    if src in params and tgt in params[src]:
        return params[src][tgt], False

    # Reverse: tgt file contains entry for src
    if tgt in params and src in params[tgt]:
        return params[tgt][src], True  # need to negate

    return None, False


# ---------------------------------------------------------------------------
# 14-parameter time-dependent Helmert (Numba JIT)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _helmert14_point(X, Y, Z,
                     tx, ty, tz, s, rx, ry, rz,
                     dtx, dty, dtz, ds, drx, dry, drz,
                     t_epoch, t_obs, position_vector):
    """Apply 14-parameter Helmert transform to a single ECEF point.

    Parameters are in metres (translations), ppb (scale), and
    arcseconds (rotations).  Rates are per year.
    """
    dt = t_obs - t_epoch

    # Effective parameters at observation epoch
    tx_e = tx + dtx * dt
    ty_e = ty + dty * dt
    tz_e = tz + dtz * dt
    s_e = 1.0 + (s + ds * dt) * 1e-9  # ppb -> scale factor
    # Rotations: arcsec -> radians
    AS2RAD = math.pi / (180.0 * 3600.0)
    rx_e = (rx + drx * dt) * AS2RAD
    ry_e = (ry + dry * dt) * AS2RAD
    rz_e = (rz + drz * dt) * AS2RAD

    if position_vector:
        # Position vector convention (IERS/IGN)
        X2 = tx_e + s_e * (X - rz_e * Y + ry_e * Z)
        Y2 = ty_e + s_e * (rz_e * X + Y - rx_e * Z)
        Z2 = tz_e + s_e * (-ry_e * X + rx_e * Y + Z)
    else:
        # Coordinate frame convention (transpose rotation)
        X2 = tx_e + s_e * (X + rz_e * Y - ry_e * Z)
        Y2 = ty_e + s_e * (-rz_e * X + Y + rx_e * Z)
        Z2 = tz_e + s_e * (ry_e * X - rx_e * Y + Z)

    return X2, Y2, Z2


@njit(nogil=True, cache=True)
def _geodetic_to_ecef(lon_deg, lat_deg, h, a, f):
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    e2 = 2.0 * f - f * f
    slat = math.sin(lat)
    clat = math.cos(lat)
    N = a / math.sqrt(1.0 - e2 * slat * slat)
    X = (N + h) * clat * math.cos(lon)
    Y = (N + h) * clat * math.sin(lon)
    Z = (N * (1.0 - e2) + h) * slat
    return X, Y, Z


@njit(nogil=True, cache=True)
def _ecef_to_geodetic(X, Y, Z, a, f):
    e2 = 2.0 * f - f * f
    lon = math.atan2(Y, X)
    p = math.sqrt(X * X + Y * Y)
    lat = math.atan2(Z, p * (1.0 - e2))
    for _ in range(10):
        slat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * slat * slat)
        lat = math.atan2(Z + e2 * N * slat, p)
    N = a / math.sqrt(1.0 - e2 * math.sin(lat) * math.sin(lat))
    h = p / math.cos(lat) - N if abs(lat) < math.pi / 4 else Z / math.sin(lat) - N * (1 - e2)
    return math.degrees(lon), math.degrees(lat), h


@njit(nogil=True, cache=True, parallel=True)
def _itrf_batch(lon_arr, lat_arr, h_arr,
                out_lon, out_lat, out_h,
                tx, ty, tz, s, rx, ry, rz,
                dtx, dty, dtz, ds, drx, dry, drz,
                t_epoch, t_obs, position_vector,
                a, f):
    for i in prange(lon_arr.shape[0]):
        X, Y, Z = _geodetic_to_ecef(lon_arr[i], lat_arr[i], h_arr[i], a, f)
        X2, Y2, Z2 = _helmert14_point(
            X, Y, Z,
            tx, ty, tz, s, rx, ry, rz,
            dtx, dty, dtz, ds, drx, dry, drz,
            t_epoch, t_obs, position_vector,
        )
        out_lon[i], out_lat[i], out_h[i] = _ecef_to_geodetic(X2, Y2, Z2, a, f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# WGS84 ellipsoid
_A = 6378137.0
_F = 1.0 / 298.257223563


def list_frames():
    """List available ITRF frames.

    Returns
    -------
    list of str
        Available frame names (e.g. ['ITRF2000', 'ITRF2008', 'ITRF2014', 'ITRF2020']).
    """
    return sorted(_get_itrf_params().keys())


def itrf_transform(lon, lat, h=0.0, *, src, tgt, epoch):
    """Transform coordinates between ITRF frames at a given epoch.

    Parameters
    ----------
    lon, lat : float or array-like
        Geographic coordinates in degrees.
    h : float or array-like
        Ellipsoidal height in metres (default 0).
    src : str
        Source ITRF frame (e.g. 'ITRF2014').
    tgt : str
        Target ITRF frame (e.g. 'ITRF2020').
    epoch : float
        Observation epoch as decimal year (e.g. 2024.0).

    Returns
    -------
    (lon, lat, h) : tuple of float or ndarray
        Transformed coordinates.

    Examples
    --------
    >>> itrf_transform(-74.0, 40.7, 10.0, src='ITRF2014', tgt='ITRF2020', epoch=2024.0)
    """
    if not isinstance(src, str) or not src:
        raise ValueError(
            f"itrf_transform(): src must be a non-empty string, got {src!r}"
        )
    if not isinstance(tgt, str) or not tgt:
        raise ValueError(
            f"itrf_transform(): tgt must be a non-empty string, got {tgt!r}"
        )

    try:
        epoch_float = float(epoch)
    except (TypeError, ValueError):
        raise ValueError(
            f"itrf_transform(): epoch must be a finite number, got {epoch!r}"
        )
    if not np.isfinite(epoch_float):
        raise ValueError(
            f"itrf_transform(): epoch must be a finite number, got {epoch!r}"
        )

    raw_params, is_reverse = _find_transform(src, tgt)
    if raw_params is None:
        raise ValueError(
            f"No transform found between {src} and {tgt}. "
            f"Available frames: {list_frames()}"
        )

    # Extract parameters (default 0 for missing)
    def g(key):
        return raw_params.get(key, 0.0)

    tx, ty, tz = g('x'), g('y'), g('z')
    s = g('s')
    rx, ry, rz = g('rx'), g('ry'), g('rz')
    dtx, dty, dtz = g('dx'), g('dy'), g('dz')
    ds = g('ds')
    drx, dry, drz = g('drx'), g('dry'), g('drz')
    t_epoch = g('t_epoch')
    convention = raw_params.get('convention', 'position_vector')
    position_vector = convention == 'position_vector'

    if is_reverse:
        # Negate all parameters for the reverse direction
        tx, ty, tz = -tx, -ty, -tz
        s = -s
        rx, ry, rz = -rx, -ry, -rz
        dtx, dty, dtz = -dtx, -dty, -dtz
        ds = -ds
        drx, dry, drz = -drx, -dry, -drz

    scalar = np.ndim(lon) == 0 and np.ndim(lat) == 0
    lon_in = np.asarray(lon, dtype=np.float64)
    lat_in = np.asarray(lat, dtype=np.float64)
    # Reject mismatched lon/lat shapes before raveling. The numba kernel
    # below runs under @njit(parallel=True) and indexes lat / h by
    # lon.shape[0], so a shorter lat array would read past its end and
    # silently return wrong values. See GH issue #2026.
    if lon_in.shape != lat_in.shape:
        raise ValueError(
            f"itrf_transform(): lon and lat must have the same shape, "
            f"got lon.shape={lon_in.shape} and lat.shape={lat_in.shape}."
        )
    lon_arr = np.atleast_1d(lon_in).ravel()
    lat_arr = np.atleast_1d(lat_in).ravel()
    # h must broadcast against the raveled, 1-D lon_arr (not the original
    # lon shape). Compare against lon_arr.shape directly so that shapes
    # like h=(1,3) vs lon=(3,), which broadcast against the original 2-D
    # lon but not against its 1-D raveled form, are rejected here with
    # the same wording as the lon/lat check rather than leaking numpy's
    # broadcast_to error from below.
    h_in = np.asarray(h, dtype=np.float64)
    if h_in.ndim != 0 and h_in.shape != lon_arr.shape:
        # Allow numpy's broadcasting to attempt anyway (it handles e.g.
        # 1-element arrays), but pre-empt obviously bad shapes so the
        # error message is specific to the public API. The broadcast
        # result must equal lon_arr.shape exactly, not merely succeed:
        # e.g. h=(1,3) and lon_arr=(3,) broadcast to (1,3) which would
        # then fail at np.broadcast_to(..., lon_arr.shape) below.
        try:
            broadcast_result = np.broadcast_shapes(h_in.shape, lon_arr.shape)
        except ValueError as e:
            raise ValueError(
                f"itrf_transform(): h shape {h_in.shape} cannot be "
                f"broadcast to lon shape {lon_arr.shape}."
            ) from e
        if broadcast_result != lon_arr.shape:
            raise ValueError(
                f"itrf_transform(): h shape {h_in.shape} cannot be "
                f"broadcast to lon shape {lon_arr.shape}."
            )
    h_arr = np.broadcast_to(np.atleast_1d(h_in), lon_arr.shape).copy()

    n = lon_arr.shape[0]
    out_lon = np.empty(n, dtype=np.float64)
    out_lat = np.empty(n, dtype=np.float64)
    out_h = np.empty(n, dtype=np.float64)

    _itrf_batch(
        lon_arr, lat_arr, h_arr,
        out_lon, out_lat, out_h,
        tx, ty, tz, s, rx, ry, rz,
        dtx, dty, dtz, ds, drx, dry, drz,
        t_epoch, float(epoch), position_vector,
        _A, _F,
    )

    if scalar:
        return float(out_lon[0]), float(out_lat[0]), float(out_h[0])
    return out_lon, out_lat, out_h
