"""
Geodesic math primitives for slope / aspect on ellipsoidal Earth.

Algorithm:
1. Convert each 3x3 neighbourhood from (lat, lon, elevation) to ECEF.
2. Project the 9 ECEF points into the local tangent-plane frame
   (East, North, Up) centered on the middle cell.
3. Apply curvature correction: u += (e^2 + n^2) / (2*R)
   to remove the systematic "drop" from the ellipsoid curving away.
4. Fit  u = A*e + B*n  via least-squares in the local frame.
5. slope  = atan(sqrt(A^2 + B^2))
6. aspect = atan2(-A, -B)  (downslope direction, compass bearing)

All functions use WGS-84 constants and float64 precision.
"""
from __future__ import annotations

from math import atan, atan2, cos, sin, sqrt

import numpy as np
from numba import cuda

from xrspatial.utils import ngjit

# ---- WGS-84 ellipsoid constants ----
WGS84_A = 6378137.0            # semi-major axis (m)
WGS84_B = 6356752.314245       # semi-minor axis (m)
WGS84_A2 = WGS84_A * WGS84_A
WGS84_B2 = WGS84_B * WGS84_B
# Mean radius for curvature correction
WGS84_R_MEAN = (2.0 * WGS84_A + WGS84_B) / 3.0
# 1 / (2 * R_mean)
INV_2R = 1.0 / (2.0 * WGS84_R_MEAN)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    # Try /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    # Try psutil
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    # Fallback: 2 GB
    return 2 * 1024 ** 3


def _check_geodesic_memory(rows, cols, func_name):
    """Raise MemoryError if the geodesic working buffers would exceed 50%
    of available RAM.

    The geodesic backends allocate, per call:
      - one ``(3, H, W)`` float64 stacked array (data, lat, lon) = 24 bytes/cell
      - one ``(H, W)`` float32 output                            =  4 bytes/cell
      - a padded copy of each channel when boundary != 'nan'     = up to 24 bytes/cell

    Budget ~56 bytes per cell to cover all allocations plus small temporaries.
    """
    if rows <= 0 or cols <= 0:
        return
    required = rows * cols * 56
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"{func_name}(method='geodesic'): a ({rows}, {cols}) raster "
            f"needs ~{required / 1e9:.1f} GB of working memory "
            f"(stacked float64 + padded copies + float32 output), but only "
            f"{available / 1e9:.1f} GB is available. "
            f"Use a smaller raster or process it in chunks."
        )


# =====================================================================
# CPU (Numba ngjit) primitives
# =====================================================================

@ngjit
def _geodetic_to_ecef(lat_rad, lon_rad, h, a2, b2):
    """Convert geodetic (lat, lon, height) to ECEF (X, Y, Z)."""
    cos_lat = cos(lat_rad)
    sin_lat = sin(lat_rad)
    cos_lon = cos(lon_rad)
    sin_lon = sin(lon_rad)
    N = a2 / sqrt(a2 * cos_lat * cos_lat + b2 * sin_lat * sin_lat)
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (b2 / a2 * N + h) * sin_lat
    return X, Y, Z


@ngjit
def _local_frame_project_and_fit(lat_deg, lon_deg, elev, neighbor_lats,
                                 neighbor_lons, neighbor_elevs, a2, b2,
                                 z_factor, inv_2r):
    """
    Project 9 ECEF points into local (E, N, U) frame, apply curvature
    correction, and fit u = A*e + B*n via least-squares.

    Returns (A, B, valid) where valid=False if data has NaN or system is degenerate.
    """
    deg2rad = 3.141592653589793 / 180.0

    for k in range(9):
        if neighbor_elevs[k] != neighbor_elevs[k]:
            return 0.0, 0.0, False

    lat_c = lat_deg * deg2rad
    lon_c = lon_deg * deg2rad
    Xc, Yc, Zc = _geodetic_to_ecef(lat_c, lon_c, elev * z_factor, a2, b2)

    cos_lat = cos(lat_c)
    sin_lat = sin(lat_c)
    cos_lon = cos(lon_c)
    sin_lon = sin(lon_c)

    # Local tangent frame unit vectors
    ex = -sin_lon;    ey = cos_lon;    ez = 0.0
    nx = -sin_lat * cos_lon; ny = -sin_lat * sin_lon; nz = cos_lat
    ux = cos_lat * cos_lon;  uy = cos_lat * sin_lon;  uz = sin_lat

    e9 = np.empty(9, dtype=np.float64)
    n9 = np.empty(9, dtype=np.float64)
    u9 = np.empty(9, dtype=np.float64)

    for k in range(9):
        lat_r = neighbor_lats[k] * deg2rad
        lon_r = neighbor_lons[k] * deg2rad
        h = neighbor_elevs[k] * z_factor
        Xk, Yk, Zk = _geodetic_to_ecef(lat_r, lon_r, h, a2, b2)
        dx = Xk - Xc
        dy = Yk - Yc
        dz = Zk - Zc
        ek = dx * ex + dy * ey + dz * ez
        nk = dx * nx + dy * ny + dz * nz
        uk = dx * ux + dy * uy + dz * uz
        # Curvature correction: compensate for Earth curving away
        uk += (ek * ek + nk * nk) * inv_2r
        e9[k] = ek
        n9[k] = nk
        u9[k] = uk

    # Centered normal equations for u = A*e + B*n + C
    me = 0.0; mn = 0.0; mu = 0.0
    for k in range(9):
        me += e9[k]; mn += n9[k]; mu += u9[k]
    inv9 = 1.0 / 9.0
    me *= inv9; mn *= inv9; mu *= inv9

    See = 0.0; Snn = 0.0; Sen = 0.0; Seu = 0.0; Snu = 0.0
    for k in range(9):
        de = e9[k] - me
        dn = n9[k] - mn
        du = u9[k] - mu
        See += de * de
        Snn += dn * dn
        Sen += de * dn
        Seu += de * du
        Snu += dn * du

    det = See * Snn - Sen * Sen
    if abs(det) < 1e-30:
        return 0.0, 0.0, True  # degenerate → flat

    A = (Seu * Snn - Snu * Sen) / det
    B = (Snu * See - Seu * Sen) / det
    return A, B, True


@ngjit
def _geodesic_slope_at_point(lat_deg, lon_deg, elev, neighbor_lats, neighbor_lons,
                             neighbor_elevs, a2, b2, z_factor, inv_2r):
    A, B, valid = _local_frame_project_and_fit(
        lat_deg, lon_deg, elev, neighbor_lats, neighbor_lons,
        neighbor_elevs, a2, b2, z_factor, inv_2r
    )
    if not valid:
        return np.nan
    slope_rad = atan(sqrt(A * A + B * B))
    return slope_rad * (180.0 / 3.141592653589793)


@ngjit
def _geodesic_aspect_at_point(lat_deg, lon_deg, elev, neighbor_lats, neighbor_lons,
                              neighbor_elevs, a2, b2, z_factor, inv_2r):
    A, B, valid = _local_frame_project_and_fit(
        lat_deg, lon_deg, elev, neighbor_lats, neighbor_lons,
        neighbor_elevs, a2, b2, z_factor, inv_2r
    )
    if not valid:
        return np.nan

    slope_mag = sqrt(A * A + B * B)
    if slope_mag < 1e-7:
        return -1.0

    # Downslope direction in (east, north): (-A, -B)
    aspect_rad = atan2(-A, -B)
    aspect_deg = aspect_rad * (180.0 / 3.141592653589793)
    if aspect_deg < 0:
        aspect_deg += 360.0
    if aspect_deg >= 360.0:
        aspect_deg -= 360.0
    return aspect_deg


# =====================================================================
# CPU kernels operating on stacked (3, H, W) arrays
#   channel 0 = elevation, channel 1 = lat_deg, channel 2 = lon_deg
# =====================================================================

@ngjit
def _cpu_geodesic_slope(stacked, a2, b2, z_factor):
    H = stacked.shape[1]
    W = stacked.shape[2]
    out = np.full((H, W), np.nan, dtype=np.float32)
    inv_2r = 1.0 / (2.0 * 6370994.884953014)  # WGS84_R_MEAN

    neighbor_lats = np.empty(9, dtype=np.float64)
    neighbor_lons = np.empty(9, dtype=np.float64)
    neighbor_elevs = np.empty(9, dtype=np.float64)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            idx = 0
            for dy in range(-1, 2):
                for dx_ in range(-1, 2):
                    neighbor_elevs[idx] = stacked[0, y + dy, x + dx_]
                    neighbor_lats[idx] = stacked[1, y + dy, x + dx_]
                    neighbor_lons[idx] = stacked[2, y + dy, x + dx_]
                    idx += 1

            out[y, x] = _geodesic_slope_at_point(
                stacked[1, y, x], stacked[2, y, x], stacked[0, y, x],
                neighbor_lats, neighbor_lons, neighbor_elevs,
                a2, b2, z_factor, inv_2r
            )
    return out


@ngjit
def _cpu_geodesic_aspect(stacked, a2, b2, z_factor):
    H = stacked.shape[1]
    W = stacked.shape[2]
    out = np.full((H, W), np.nan, dtype=np.float32)
    inv_2r = 1.0 / (2.0 * 6370994.884953014)

    neighbor_lats = np.empty(9, dtype=np.float64)
    neighbor_lons = np.empty(9, dtype=np.float64)
    neighbor_elevs = np.empty(9, dtype=np.float64)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            idx = 0
            for dy in range(-1, 2):
                for dx_ in range(-1, 2):
                    neighbor_elevs[idx] = stacked[0, y + dy, x + dx_]
                    neighbor_lats[idx] = stacked[1, y + dy, x + dx_]
                    neighbor_lons[idx] = stacked[2, y + dy, x + dx_]
                    idx += 1

            out[y, x] = _geodesic_aspect_at_point(
                stacked[1, y, x], stacked[2, y, x], stacked[0, y, x],
                neighbor_lats, neighbor_lons, neighbor_elevs,
                a2, b2, z_factor, inv_2r
            )
    return out


# =====================================================================
# GPU (CUDA) device functions
# =====================================================================

@cuda.jit(device=True)
def _gpu_geodetic_to_ecef(lat_rad, lon_rad, h, a2, b2):
    cos_lat = cos(lat_rad)
    sin_lat = sin(lat_rad)
    cos_lon = cos(lon_rad)
    sin_lon = sin(lon_rad)
    N = a2 / sqrt(a2 * cos_lat * cos_lat + b2 * sin_lat * sin_lat)
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (b2 / a2 * N + h) * sin_lat
    return X, Y, Z


@cuda.jit(device=True)
def _gpu_local_frame_fit(elev_arr, lat_arr, lon_arr, a2, b2, z_factor, inv_2r):
    """
    Project 3x3 neighbourhood into local frame, apply curvature correction,
    fit u = A*e + B*n. Returns (A, B, valid_flag).
    """
    deg2rad = 3.141592653589793 / 180.0

    # NaN check
    for dy in range(3):
        for dx in range(3):
            v = elev_arr[dy, dx]
            if v != v:
                return 0.0, 0.0, 0.0  # invalid

    lat_c = lat_arr[1, 1] * deg2rad
    lon_c = lon_arr[1, 1] * deg2rad
    Xc, Yc, Zc = _gpu_geodetic_to_ecef(lat_c, lon_c, elev_arr[1, 1] * z_factor, a2, b2)

    cos_lat = cos(lat_c)
    sin_lat = sin(lat_c)
    cos_lon = cos(lon_c)
    sin_lon = sin(lon_c)

    ex = -sin_lon;    ey = cos_lon;    ez = 0.0
    nxv = -sin_lat * cos_lon; nyv = -sin_lat * sin_lon; nzv = cos_lat
    ux = cos_lat * cos_lon;  uy = cos_lat * sin_lon;  uz = sin_lat

    # Accumulate means and sums for LSQ
    me = 0.0; mn_sum = 0.0; mu = 0.0
    See = 0.0; Snn = 0.0; Sen = 0.0; Seu = 0.0; Snu = 0.0

    # First pass: accumulate means (store projected coords in local vars)
    # We need all 9 values for the centered fit, so do two passes
    e_vals_0 = 0.0; e_vals_1 = 0.0; e_vals_2 = 0.0
    e_vals_3 = 0.0; e_vals_4 = 0.0; e_vals_5 = 0.0
    e_vals_6 = 0.0; e_vals_7 = 0.0; e_vals_8 = 0.0
    n_vals_0 = 0.0; n_vals_1 = 0.0; n_vals_2 = 0.0
    n_vals_3 = 0.0; n_vals_4 = 0.0; n_vals_5 = 0.0
    n_vals_6 = 0.0; n_vals_7 = 0.0; n_vals_8 = 0.0
    u_vals_0 = 0.0; u_vals_1 = 0.0; u_vals_2 = 0.0
    u_vals_3 = 0.0; u_vals_4 = 0.0; u_vals_5 = 0.0
    u_vals_6 = 0.0; u_vals_7 = 0.0; u_vals_8 = 0.0

    for idx in range(9):
        dy = idx // 3
        dx = idx % 3
        lat_r = lat_arr[dy, dx] * deg2rad
        lon_r = lon_arr[dy, dx] * deg2rad
        Xk, Yk, Zk = _gpu_geodetic_to_ecef(lat_r, lon_r, elev_arr[dy, dx] * z_factor, a2, b2)
        ddx = Xk - Xc; ddy = Yk - Yc; ddz = Zk - Zc
        ek = ddx * ex + ddy * ey + ddz * ez
        nk = ddx * nxv + ddy * nyv + ddz * nzv
        uk = ddx * ux + ddy * uy + ddz * uz
        uk += (ek * ek + nk * nk) * inv_2r  # curvature correction

        if idx == 0: e_vals_0 = ek; n_vals_0 = nk; u_vals_0 = uk
        elif idx == 1: e_vals_1 = ek; n_vals_1 = nk; u_vals_1 = uk
        elif idx == 2: e_vals_2 = ek; n_vals_2 = nk; u_vals_2 = uk
        elif idx == 3: e_vals_3 = ek; n_vals_3 = nk; u_vals_3 = uk
        elif idx == 4: e_vals_4 = ek; n_vals_4 = nk; u_vals_4 = uk
        elif idx == 5: e_vals_5 = ek; n_vals_5 = nk; u_vals_5 = uk
        elif idx == 6: e_vals_6 = ek; n_vals_6 = nk; u_vals_6 = uk
        elif idx == 7: e_vals_7 = ek; n_vals_7 = nk; u_vals_7 = uk
        elif idx == 8: e_vals_8 = ek; n_vals_8 = nk; u_vals_8 = uk

        me += ek; mn_sum += nk; mu += uk

    inv9 = 1.0 / 9.0
    me *= inv9; mn_sum *= inv9; mu *= inv9

    es = (e_vals_0, e_vals_1, e_vals_2, e_vals_3, e_vals_4,
          e_vals_5, e_vals_6, e_vals_7, e_vals_8)
    ns = (n_vals_0, n_vals_1, n_vals_2, n_vals_3, n_vals_4,
          n_vals_5, n_vals_6, n_vals_7, n_vals_8)
    us = (u_vals_0, u_vals_1, u_vals_2, u_vals_3, u_vals_4,
          u_vals_5, u_vals_6, u_vals_7, u_vals_8)

    for k in range(9):
        de = es[k] - me
        dn = ns[k] - mn_sum
        du = us[k] - mu
        See += de * de
        Snn += dn * dn
        Sen += de * dn
        Seu += de * du
        Snu += dn * du

    det = See * Snn - Sen * Sen
    if abs(det) < 1e-30:
        return 0.0, 0.0, 1.0  # degenerate → flat, but valid

    A = (Seu * Snn - Snu * Sen) / det
    B = (Snu * See - Seu * Sen) / det
    return A, B, 1.0


@cuda.jit(device=True)
def _gpu_geodesic_slope_cell(elev_arr, lat_arr, lon_arr, a2, b2, z_factor, inv_2r):
    A, B, flag = _gpu_local_frame_fit(elev_arr, lat_arr, lon_arr, a2, b2, z_factor, inv_2r)
    if flag == 0.0:
        return np.nan
    slope_rad = atan(sqrt(A * A + B * B))
    return slope_rad * (180.0 / 3.141592653589793)


@cuda.jit(device=True)
def _gpu_geodesic_aspect_cell(elev_arr, lat_arr, lon_arr, a2, b2, z_factor, inv_2r):
    A, B, flag = _gpu_local_frame_fit(elev_arr, lat_arr, lon_arr, a2, b2, z_factor, inv_2r)
    if flag == 0.0:
        return np.nan
    slope_mag = sqrt(A * A + B * B)
    if slope_mag < 1e-7:
        return -1.0
    aspect_rad = atan2(-A, -B)
    aspect_deg = aspect_rad * (180.0 / 3.141592653589793)
    if aspect_deg < 0:
        aspect_deg += 360.0
    if aspect_deg >= 360.0:
        aspect_deg -= 360.0
    return aspect_deg


# =====================================================================
# GPU global kernels — operate on (3, H, W) stacked arrays
# =====================================================================

@cuda.jit
def _run_gpu_geodesic_slope(stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out):
    i, j = cuda.grid(2)
    H = out.shape[0]
    W = out.shape[1]
    if i >= 1 and i < H - 1 and j >= 1 and j < W - 1:
        out[i, j] = _gpu_geodesic_slope_cell(
            stacked[0, i - 1:i + 2, j - 1:j + 2],
            stacked[1, i - 1:i + 2, j - 1:j + 2],
            stacked[2, i - 1:i + 2, j - 1:j + 2],
            a2_arr[0], b2_arr[0], zf_arr[0], inv_2r_arr[0],
        )


@cuda.jit
def _run_gpu_geodesic_aspect(stacked, a2_arr, b2_arr, zf_arr, inv_2r_arr, out):
    i, j = cuda.grid(2)
    H = out.shape[0]
    W = out.shape[1]
    if i >= 1 and i < H - 1 and j >= 1 and j < W - 1:
        out[i, j] = _gpu_geodesic_aspect_cell(
            stacked[0, i - 1:i + 2, j - 1:j + 2],
            stacked[1, i - 1:i + 2, j - 1:j + 2],
            stacked[2, i - 1:i + 2, j - 1:j + 2],
            a2_arr[0], b2_arr[0], zf_arr[0], inv_2r_arr[0],
        )
