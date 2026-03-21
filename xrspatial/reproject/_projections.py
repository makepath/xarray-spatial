"""Numba JIT coordinate transforms for common projections.

Replaces pyproj for the most-used CRS pairs, giving ~30x speedup
via parallelised Numba kernels.

Supported fast paths
--------------------
- WGS84 (EPSG:4326) <-> Web Mercator (EPSG:3857)
- WGS84 / NAD83 <-> UTM zones (EPSG:326xx / 327xx / 269xx)
- WGS84 / NAD83 <-> Ellipsoidal Mercator (EPSG:3395)
- WGS84 / NAD83 <-> Lambert Conformal Conic (e.g. EPSG:2154)
- WGS84 / NAD83 <-> Albers Equal Area (e.g. EPSG:5070)
- WGS84 / NAD83 <-> Cylindrical Equal Area (e.g. EPSG:6933)

All other CRS pairs fall back to pyproj.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# WGS84 ellipsoid constants
# ---------------------------------------------------------------------------
_WGS84_A = 6378137.0                        # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563              # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)     # semi-minor axis
_WGS84_N = (_WGS84_A - _WGS84_B) / (_WGS84_A + _WGS84_B)  # third flattening
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2  # eccentricity squared
_WGS84_E = math.sqrt(_WGS84_E2)             # eccentricity

# ---------------------------------------------------------------------------
# Web Mercator  (EPSG:3857)  --  spherical, trivial
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _merc_fwd_point(lon_deg, lat_deg):
    """(lon, lat) in degrees -> (x, y) in EPSG:3857 metres."""
    x = _WGS84_A * math.radians(lon_deg)
    phi = math.radians(lat_deg)
    y = _WGS84_A * math.log(math.tan(math.pi / 4.0 + phi / 2.0))
    return x, y


@njit(nogil=True, cache=True)
def _merc_inv_point(x, y):
    """(x, y) in EPSG:3857 metres -> (lon, lat) in degrees."""
    lon = math.degrees(x / _WGS84_A)
    lat = math.degrees(math.atan(math.sinh(y / _WGS84_A)))
    return lon, lat


@njit(nogil=True, cache=True, parallel=True)
def merc_forward(lons, lats, out_x, out_y):
    """Batch WGS84 -> Web Mercator.  Writes into pre-allocated arrays."""
    for i in prange(lons.shape[0]):
        out_x[i], out_y[i] = _merc_fwd_point(lons[i], lats[i])


@njit(nogil=True, cache=True, parallel=True)
def merc_inverse(xs, ys, out_lon, out_lat):
    """Batch Web Mercator -> WGS84.  Writes into pre-allocated arrays."""
    for i in prange(xs.shape[0]):
        out_lon[i], out_lat[i] = _merc_inv_point(xs[i], ys[i])


# ---------------------------------------------------------------------------
# Shared helpers  (PROJ pj_tsfn, pj_sinhpsi2tanphi, authalic latitude)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _pj_tsfn(phi, sinphi, e):
    """Isometric co-latitude: ts = exp(-psi).

    Equivalent to tan(pi/4 - phi/2) / ((1-e*sinphi)/(1+e*sinphi))^(e/2).
    """
    es = e * sinphi
    return math.tan(math.pi / 4.0 - phi / 2.0) * math.pow(
        (1.0 + es) / (1.0 - es), e / 2.0
    )


@njit(nogil=True, cache=True)
def _pj_sinhpsi2tanphi(taup, e):
    """Newton iteration: recover tan(phi) from sinh(isometric lat).

    Matches PROJ's pj_sinhpsi2tanphi -- 5 iterations, always converges.
    """
    e2 = e * e
    tau = taup
    tau1 = math.sqrt(1.0 + tau * tau)

    for _ in range(5):
        tau1 = math.sqrt(1.0 + tau * tau)
        sig = math.sinh(e * math.atanh(e * tau / tau1))
        sig1 = math.sqrt(1.0 + sig * sig)
        taupa = sig1 * tau - sig * tau1
        dtau = ((taup - taupa) * (1.0 + (1.0 - e2) * tau * tau)
                / ((1.0 - e2) * tau1 * math.sqrt(1.0 + taupa * taupa)))
        tau += dtau
        if abs(dtau) < 1e-12:
            break
    return tau


@njit(nogil=True, cache=True)
def _authalic_q(sinphi, e):
    """Authalic latitude q-parameter: q(phi) for given sinphi and e."""
    e2 = e * e
    es = e * sinphi
    return (1.0 - e2) * (sinphi / (1.0 - es * es) + math.atanh(es) / e)


def _authalic_apa(e):
    """Precompute 3 coefficients for the authalic latitude inverse series.

    Returns array [APA0, APA1, APA2] used by _authalic_inv.
    Matches PROJ's pj_authlat.
    """
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    apa = np.empty(3, dtype=np.float64)
    apa[0] = e2 / 3.0 + 31.0 * e4 / 180.0 + 59.0 * e6 / 560.0
    apa[1] = 17.0 * e4 / 360.0 + 61.0 * e6 / 1260.0
    apa[2] = 383.0 * e6 / 45360.0
    return apa


@njit(nogil=True, cache=True)
def _authalic_inv(beta, apa):
    """Inverse authalic latitude: beta (authalic, rad) -> phi (geodetic, rad).

    Uses the 3-term Fourier series from PROJ's pj_authlat.
    """
    t = beta + beta
    return beta + apa[0] * math.sin(t) + apa[1] * math.sin(2.0 * t) + apa[2] * math.sin(3.0 * t)


# Precompute authalic coefficients for WGS84
_APA = _authalic_apa(_WGS84_E)
_QP = _authalic_q(1.0, _WGS84_E)  # q at the pole


# ---------------------------------------------------------------------------
# Ellipsoidal Mercator  (EPSG:3395)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _emerc_fwd_point(lon_deg, lat_deg, k0, e):
    """(lon, lat) deg -> (x, y) metres, ellipsoidal Mercator."""
    lam = math.radians(lon_deg)
    phi = math.radians(lat_deg)
    sinphi = math.sin(phi)
    x = k0 * _WGS84_A * lam
    y = k0 * _WGS84_A * (math.asinh(math.tan(phi)) - e * math.atanh(e * sinphi))
    return x, y


@njit(nogil=True, cache=True)
def _emerc_inv_point(x, y, k0, e):
    """(x, y) metres -> (lon, lat) deg, ellipsoidal Mercator."""
    lam = x / (k0 * _WGS84_A)
    taup = math.sinh(y / (k0 * _WGS84_A))
    tau = _pj_sinhpsi2tanphi(taup, e)
    return math.degrees(lam), math.degrees(math.atan(tau))


@njit(nogil=True, cache=True, parallel=True)
def emerc_forward(lons, lats, out_x, out_y, k0, e):
    for i in prange(lons.shape[0]):
        out_x[i], out_y[i] = _emerc_fwd_point(lons[i], lats[i], k0, e)


@njit(nogil=True, cache=True, parallel=True)
def emerc_inverse(xs, ys, out_lon, out_lat, k0, e):
    for i in prange(xs.shape[0]):
        out_lon[i], out_lat[i] = _emerc_inv_point(xs[i], ys[i], k0, e)


# ---------------------------------------------------------------------------
# Lambert Conformal Conic  (LCC)
# ---------------------------------------------------------------------------

def _lcc_params(crs):
    """Extract LCC projection parameters from a pyproj CRS.

    Returns (lon0, lat0, n, c, rho0, k0) or None.
    """
    try:
        d = crs.to_dict()
    except Exception:
        return None
    if d.get('proj') != 'lcc':
        return None

    lat_1 = math.radians(d.get('lat_1', d.get('lat_0', 0.0)))
    lat_2 = math.radians(d.get('lat_2', lat_1))
    lat_0 = math.radians(d.get('lat_0', 0.0))
    lon_0 = math.radians(d.get('lon_0', 0.0))
    k0_param = d.get('k_0', d.get('k', 1.0))

    e = _WGS84_E
    a = _WGS84_A

    sinphi1 = math.sin(lat_1)
    cosphi1 = math.cos(lat_1)
    sinphi2 = math.sin(lat_2)

    m1 = cosphi1 / math.sqrt(1.0 - _WGS84_E2 * sinphi1 * sinphi1)
    ts1 = math.tan(math.pi / 4.0 - lat_1 / 2.0) * math.pow(
        (1.0 + e * sinphi1) / (1.0 - e * sinphi1), e / 2.0)

    if abs(lat_1 - lat_2) > 1e-10:
        m2 = cosphi2 = math.cos(lat_2)
        cosphi2 /= math.sqrt(1.0 - _WGS84_E2 * sinphi2 * sinphi2)
        ts2 = math.tan(math.pi / 4.0 - lat_2 / 2.0) * math.pow(
            (1.0 + e * sinphi2) / (1.0 - e * sinphi2), e / 2.0)
        n = math.log(m1 / cosphi2) / math.log(ts1 / ts2)
    else:
        n = sinphi1

    c = m1 * math.pow(ts1, -n) / n
    sinphi0 = math.sin(lat_0)
    ts0 = math.tan(math.pi / 4.0 - lat_0 / 2.0) * math.pow(
        (1.0 + e * sinphi0) / (1.0 - e * sinphi0), e / 2.0)
    rho0 = a * k0_param * c * math.pow(ts0, n)

    fe = d.get('x_0', 0.0)
    fn = d.get('y_0', 0.0)

    return lon_0, n, c, rho0, k0_param, fe, fn


@njit(nogil=True, cache=True)
def _lcc_fwd_point(lon_deg, lat_deg, lon0, n, c, rho0, k0, e, a):
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg) - lon0
    sinphi = math.sin(phi)
    ts = math.tan(math.pi / 4.0 - phi / 2.0) * math.pow(
        (1.0 + e * sinphi) / (1.0 - e * sinphi), e / 2.0)
    rho = a * k0 * c * math.pow(ts, n)
    lam_n = n * lam
    x = rho * math.sin(lam_n)
    y = rho0 - rho * math.cos(lam_n)
    return x, y


@njit(nogil=True, cache=True)
def _lcc_inv_point(x, y, lon0, n, c, rho0, k0, e, a):
    rho0_y = rho0 - y
    if n < 0.0:
        rho = -math.hypot(x, rho0_y)
        lam_n = math.atan2(-x, -rho0_y)
    else:
        rho = math.hypot(x, rho0_y)
        lam_n = math.atan2(x, rho0_y)
    if abs(rho) < 1e-30:
        return math.degrees(lon0 + lam_n / n), 90.0 if n > 0 else -90.0
    ts = math.pow(rho / (a * k0 * c), 1.0 / n)
    # Recover phi from ts via Newton (pj_sinhpsi2tanphi)
    phi_approx = math.pi / 2.0 - 2.0 * math.atan(ts)
    taup = math.sinh(math.log(1.0 / ts))  # sinh(psi)
    tau = _pj_sinhpsi2tanphi(taup, e)
    phi = math.atan(tau)
    lam = lam_n / n
    return math.degrees(lam + lon0), math.degrees(phi)


@njit(nogil=True, cache=True, parallel=True)
def lcc_forward(lons, lats, out_x, out_y,
                lon0, n, c, rho0, k0, fe, fn, e, a):
    for i in prange(lons.shape[0]):
        x, y = _lcc_fwd_point(lons[i], lats[i], lon0, n, c, rho0, k0, e, a)
        out_x[i] = x + fe
        out_y[i] = y + fn


@njit(nogil=True, cache=True, parallel=True)
def lcc_inverse(xs, ys, out_lon, out_lat,
                lon0, n, c, rho0, k0, fe, fn, e, a):
    for i in prange(xs.shape[0]):
        out_lon[i], out_lat[i] = _lcc_inv_point(
            xs[i] - fe, ys[i] - fn, lon0, n, c, rho0, k0, e, a)


# ---------------------------------------------------------------------------
# Albers Equal Area Conic  (AEA)
# ---------------------------------------------------------------------------

def _aea_params(crs):
    """Extract AEA projection parameters from a pyproj CRS.

    Returns (lon0, n, c, dd, rho0, fe, fn) or None.
    """
    try:
        d = crs.to_dict()
    except Exception:
        return None
    if d.get('proj') != 'aea':
        return None

    lat_1 = math.radians(d.get('lat_1', 0.0))
    lat_2 = math.radians(d.get('lat_2', lat_1))
    lat_0 = math.radians(d.get('lat_0', 0.0))
    lon_0 = math.radians(d.get('lon_0', 0.0))

    e = _WGS84_E
    e2 = _WGS84_E2
    a = _WGS84_A

    sinphi1 = math.sin(lat_1)
    cosphi1 = math.cos(lat_1)
    sinphi2 = math.sin(lat_2)
    cosphi2 = math.cos(lat_2)

    m1 = cosphi1 / math.sqrt(1.0 - e2 * sinphi1 * sinphi1)
    m2 = cosphi2 / math.sqrt(1.0 - e2 * sinphi2 * sinphi2)
    q1 = _authalic_q(sinphi1, e)
    q2 = _authalic_q(sinphi2, e)
    q0 = _authalic_q(math.sin(lat_0), e)

    if abs(lat_1 - lat_2) > 1e-10:
        n = (m1 * m1 - m2 * m2) / (q2 - q1)
    else:
        n = sinphi1

    C = m1 * m1 + n * q1
    rho0 = a * math.sqrt(C - n * q0) / n

    fe = d.get('x_0', 0.0)
    fn = d.get('y_0', 0.0)

    return lon_0, n, C, rho0, fe, fn


@njit(nogil=True, cache=True)
def _aea_fwd_point(lon_deg, lat_deg, lon0, n, C, rho0, e, a):
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg) - lon0
    q = _authalic_q(math.sin(phi), e)
    val = C - n * q
    if val < 0.0:
        val = 0.0
    rho = a * math.sqrt(val) / n
    theta = n * lam
    x = rho * math.sin(theta)
    y = rho0 - rho * math.cos(theta)
    return x, y


@njit(nogil=True, cache=True)
def _aea_inv_point(x, y, lon0, n, C, rho0, e, a, qp, apa):
    rho0_y = rho0 - y
    if n < 0.0:
        rho = -math.hypot(x, rho0_y)
        theta = math.atan2(-x, -rho0_y)
    else:
        rho = math.hypot(x, rho0_y)
        theta = math.atan2(x, rho0_y)
    q = (C - (rho * rho * n * n) / (a * a)) / n
    # beta = asin(q / qp), clamped
    ratio = q / qp
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    beta = math.asin(ratio)
    phi = _authalic_inv(beta, apa)
    lam = theta / n
    return math.degrees(lam + lon0), math.degrees(phi)


@njit(nogil=True, cache=True, parallel=True)
def aea_forward(lons, lats, out_x, out_y,
                lon0, n, C, rho0, fe, fn, e, a):
    for i in prange(lons.shape[0]):
        x, y = _aea_fwd_point(lons[i], lats[i], lon0, n, C, rho0, e, a)
        out_x[i] = x + fe
        out_y[i] = y + fn


@njit(nogil=True, cache=True, parallel=True)
def aea_inverse(xs, ys, out_lon, out_lat,
                lon0, n, C, rho0, fe, fn, e, a, qp, apa):
    for i in prange(xs.shape[0]):
        out_lon[i], out_lat[i] = _aea_inv_point(
            xs[i] - fe, ys[i] - fn, lon0, n, C, rho0, e, a, qp, apa)


# ---------------------------------------------------------------------------
# Cylindrical Equal Area  (CEA)
# ---------------------------------------------------------------------------

def _cea_params(crs):
    """Extract CEA projection parameters from a pyproj CRS.

    Returns (lon0, k0, fe, fn) or None.
    """
    try:
        d = crs.to_dict()
    except Exception:
        return None
    if d.get('proj') != 'cea':
        return None

    lon_0 = math.radians(d.get('lon_0', 0.0))
    lat_ts = math.radians(d.get('lat_ts', 0.0))
    sinlts = math.sin(lat_ts)
    coslts = math.cos(lat_ts)
    # k0 = cos(lat_ts) / sqrt(1 - e² sin²(lat_ts))
    k0 = coslts / math.sqrt(1.0 - _WGS84_E2 * sinlts * sinlts)
    fe = d.get('x_0', 0.0)
    fn = d.get('y_0', 0.0)
    return lon_0, k0, fe, fn


@njit(nogil=True, cache=True)
def _cea_fwd_point(lon_deg, lat_deg, lon0, k0, e, a, qp):
    lam = math.radians(lon_deg) - lon0
    phi = math.radians(lat_deg)
    q = _authalic_q(math.sin(phi), e)
    x = a * k0 * lam
    y = a * q / (2.0 * k0)
    return x, y


@njit(nogil=True, cache=True)
def _cea_inv_point(x, y, lon0, k0, e, a, qp, apa):
    lam = x / (a * k0)
    ratio = 2.0 * y * k0 / (a * qp)
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < -1.0:
        ratio = -1.0
    beta = math.asin(ratio)
    phi = _authalic_inv(beta, apa)
    return math.degrees(lam + lon0), math.degrees(phi)


@njit(nogil=True, cache=True, parallel=True)
def cea_forward(lons, lats, out_x, out_y,
                lon0, k0, fe, fn, e, a, qp):
    for i in prange(lons.shape[0]):
        x, y = _cea_fwd_point(lons[i], lats[i], lon0, k0, e, a, qp)
        out_x[i] = x + fe
        out_y[i] = y + fn


@njit(nogil=True, cache=True, parallel=True)
def cea_inverse(xs, ys, out_lon, out_lat,
                lon0, k0, fe, fn, e, a, qp, apa):
    for i in prange(xs.shape[0]):
        out_lon[i], out_lat[i] = _cea_inv_point(
            xs[i] - fe, ys[i] - fn, lon0, k0, e, a, qp, apa)


# ---------------------------------------------------------------------------
# Transverse Mercator / UTM  --  6th-order Krueger series (Karney 2011)
# ---------------------------------------------------------------------------

def _tmerc_coefficients(n):
    """Precompute all series coefficients from third flattening *n*.

    Returns (alpha, beta, cbg, cgb, Qn) where:
    - alpha[0..5]: forward Krueger (conformal sphere -> rectifying)
    - beta[0..5]:  inverse Krueger (rectifying -> conformal sphere)
    - cbg[0..5]:   geographic -> conformal latitude
    - cgb[0..5]:   conformal -> geographic latitude
    - Qn:          rectifying radius * k0
    """
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    n5 = n4 * n
    n6 = n5 * n

    # Rectifying radius (scaled by k0 later)
    A = _WGS84_A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0)

    # Forward Krueger: alpha[1..6]
    alpha = np.array([
        n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0
        + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0 + 7891.0 * n6 / 37800.0,

        13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0
        + 281.0 * n5 / 630.0 - 1983433.0 * n6 / 1935360.0,

        61.0 * n3 / 240.0 - 103.0 * n4 / 140.0 + 15061.0 * n5 / 26880.0
        + 167603.0 * n6 / 181440.0,

        49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0
        + 6601661.0 * n6 / 7257600.0,

        34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,

        212378941.0 * n6 / 319334400.0,
    ], dtype=np.float64)

    # Inverse Krueger: beta[1..6]
    beta = np.array([
        n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0
        - n4 / 360.0 - 81.0 * n5 / 512.0 + 96199.0 * n6 / 604800.0,

        n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1440.0
        + 46.0 * n5 / 105.0 - 1118711.0 * n6 / 3870720.0,

        17.0 * n3 / 480.0 - 37.0 * n4 / 840.0
        - 209.0 * n5 / 4480.0 + 5569.0 * n6 / 90720.0,

        4397.0 * n4 / 161280.0 - 11.0 * n5 / 504.0
        - 830251.0 * n6 / 7257600.0,

        4583.0 * n5 / 161280.0 - 108847.0 * n6 / 3991680.0,

        20648693.0 * n6 / 638668800.0,
    ], dtype=np.float64)

    # Geographic -> Conformal latitude: cbg[1..6]
    cbg = np.array([
        n * (-2.0 + n * (2.0 / 3.0 + n * (4.0 / 3.0 + n * (-82.0 / 45.0
        + n * (32.0 / 45.0 + n * 4642.0 / 4725.0))))),

        n2 * (5.0 / 3.0 + n * (-16.0 / 15.0 + n * (-13.0 / 9.0
        + n * (904.0 / 315.0 - n * 1522.0 / 945.0)))),

        n3 * (-26.0 / 15.0 + n * (34.0 / 21.0 + n * (8.0 / 5.0
        - n * 12686.0 / 2835.0))),

        n4 * (1237.0 / 630.0 + n * (-12.0 / 5.0
        - n * 24832.0 / 14175.0)),

        n5 * (-734.0 / 315.0 + n * 109598.0 / 31185.0),

        n6 * 444337.0 / 155925.0,
    ], dtype=np.float64)

    # Conformal -> Geographic latitude: cgb[1..6]
    cgb = np.array([
        n * (2.0 + n * (-2.0 / 3.0 + n * (-2.0 + n * (116.0 / 45.0
        + n * (26.0 / 45.0 - n * 2854.0 / 675.0))))),

        n2 * (7.0 / 3.0 + n * (-8.0 / 5.0 + n * (-227.0 / 45.0
        + n * (2704.0 / 315.0 + n * 2323.0 / 945.0)))),

        n3 * (56.0 / 15.0 + n * (-136.0 / 35.0 + n * (-1262.0 / 105.0
        + n * 73814.0 / 2835.0))),

        n4 * (4279.0 / 630.0 + n * (-332.0 / 35.0
        - n * 399572.0 / 14175.0)),

        n5 * (4174.0 / 315.0 - n * 144838.0 / 6237.0),

        n6 * 601676.0 / 22275.0,
    ], dtype=np.float64)

    return alpha, beta, cbg, cgb, A


# Precompute WGS84 coefficients once at import time
_ALPHA, _BETA, _CBG, _CGB, _A_RECT = _tmerc_coefficients(_WGS84_N)


@njit(nogil=True, cache=True)
def _clenshaw_sin(coeffs, angle):
    """Evaluate SUM_{k=1}^{N} coeffs[k-1] * sin(2*k*angle) via Clenshaw."""
    N = coeffs.shape[0]
    X = 2.0 * math.cos(2.0 * angle)
    u0 = 0.0
    u1 = 0.0
    for k in range(N - 1, -1, -1):
        t = X * u0 - u1 + coeffs[k]
        u1 = u0
        u0 = t
    return math.sin(2.0 * angle) * u0


@njit(nogil=True, cache=True)
def _clenshaw_complex(coeffs, sin2Cn, cos2Cn, sinh2Ce, cosh2Ce):
    """Complex Clenshaw summation for Krueger series.

    Evaluates SUM a[k] * sin(2k*(Cn + i*Ce)) returning (dCn, dCe).
    """
    N = coeffs.shape[0]
    r = 2.0 * cos2Cn * cosh2Ce
    im = -2.0 * sin2Cn * sinh2Ce

    hr = 0.0
    hi = 0.0
    hr1 = 0.0
    hi1 = 0.0
    for k in range(N - 1, -1, -1):
        hr2 = hr1
        hi2 = hi1
        hr1 = hr
        hi1 = hi
        hr = -hr2 + r * hr1 - im * hi1 + coeffs[k]
        hi = -hi2 + im * hr1 + r * hi1

    dCn = sin2Cn * cosh2Ce * hr - cos2Cn * sinh2Ce * hi
    dCe = sin2Cn * cosh2Ce * hi + cos2Cn * sinh2Ce * hr
    return dCn, dCe


@njit(nogil=True, cache=True)
def _tmerc_fwd_point(lon_deg, lat_deg, lon0_rad, k0, Qn,
                     alpha, cbg):
    """(lon, lat) degrees -> (E, N) metres for a Transverse Mercator projection."""
    lam = math.radians(lon_deg) - lon0_rad
    phi = math.radians(lat_deg)

    # Step 1: geographic -> conformal latitude via Clenshaw
    chi = phi + _clenshaw_sin(cbg, phi)

    sin_chi = math.sin(chi)
    cos_chi = math.cos(chi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    # Step 2: conformal sphere -> isometric
    denom = math.hypot(sin_chi, cos_chi * cos_lam)
    if denom < 1e-30:
        denom = 1e-30
    Cn = math.atan2(sin_chi, cos_chi * cos_lam)
    tan_Ce = sin_lam * cos_chi / denom
    # Clamp to avoid NaN in asinh at extreme values
    if tan_Ce > 1e15:
        tan_Ce = 1e15
    elif tan_Ce < -1e15:
        tan_Ce = -1e15
    Ce = math.asinh(tan_Ce)

    # Step 3: Krueger series correction (complex Clenshaw)
    inv_d = 1.0 / denom
    inv_d2 = inv_d * inv_d
    cos_chi_cos_lam = cos_chi * cos_lam
    sin2 = 2.0 * sin_chi * cos_chi_cos_lam * inv_d2
    cos2 = 2.0 * cos_chi_cos_lam * cos_chi_cos_lam * inv_d2 - 1.0
    sinh2 = 2.0 * tan_Ce * inv_d
    cosh2 = 2.0 * inv_d2 - 1.0

    dCn, dCe = _clenshaw_complex(alpha, sin2, cos2, sinh2, cosh2)
    Cn += dCn
    Ce += dCe

    # Step 4: scale
    x = Qn * Ce   # easting before false easting
    y = Qn * Cn   # northing (Zb = 0 for UTM since phi0 = 0)
    return x, y


@njit(nogil=True, cache=True)
def _tmerc_inv_point(x, y, lon0_rad, k0, Qn, beta, cgb):
    """(E, N) metres -> (lon, lat) degrees for a Transverse Mercator projection."""
    Cn = y / Qn
    Ce = x / Qn

    # Step 2: inverse Krueger series
    sin2Cn = math.sin(2.0 * Cn)
    cos2Cn = math.cos(2.0 * Cn)
    exp2Ce = math.exp(2.0 * Ce)
    inv_exp2Ce = 1.0 / exp2Ce
    sinh2Ce = 0.5 * (exp2Ce - inv_exp2Ce)
    cosh2Ce = 0.5 * (exp2Ce + inv_exp2Ce)

    dCn, dCe = _clenshaw_complex(beta, sin2Cn, cos2Cn, sinh2Ce, cosh2Ce)
    Cn -= dCn
    Ce -= dCe

    # Step 3: isometric -> conformal sphere
    sin_Cn = math.sin(Cn)
    cos_Cn = math.cos(Cn)
    sinh_Ce = math.sinh(Ce)

    lam = math.atan2(sinh_Ce, cos_Cn)

    # Step 4: conformal -> geographic latitude
    modulus = math.hypot(sinh_Ce, cos_Cn)
    chi = math.atan2(sin_Cn, modulus)

    phi = chi + _clenshaw_sin(cgb, chi)

    lon = math.degrees(lam + lon0_rad)
    lat = math.degrees(phi)
    return lon, lat


@njit(nogil=True, cache=True, parallel=True)
def tmerc_forward(lons, lats, out_x, out_y,
                  lon0_rad, k0, false_e, false_n,
                  Qn, alpha, cbg):
    """Batch geographic -> Transverse Mercator."""
    for i in prange(lons.shape[0]):
        x, y = _tmerc_fwd_point(lons[i], lats[i], lon0_rad, k0, Qn,
                                alpha, cbg)
        out_x[i] = x + false_e
        out_y[i] = y + false_n


@njit(nogil=True, cache=True, parallel=True)
def tmerc_inverse(xs, ys, out_lon, out_lat,
                  lon0_rad, k0, false_e, false_n,
                  Qn, beta, cgb):
    """Batch Transverse Mercator -> geographic."""
    for i in prange(xs.shape[0]):
        lon, lat = _tmerc_inv_point(
            xs[i] - false_e, ys[i] - false_n,
            lon0_rad, k0, Qn, beta, cgb)
        out_lon[i] = lon
        out_lat[i] = lat


# ---------------------------------------------------------------------------
# UTM zone helpers
# ---------------------------------------------------------------------------

def _utm_params(epsg_code):
    """Extract UTM zone parameters from EPSG code.

    Returns (lon0_rad, k0, false_easting, false_northing) or None.
    """
    # EPSG:326xx = UTM North, EPSG:327xx = UTM South (WGS84)
    # EPSG:269xx = UTM North (NAD83, effectively same ellipsoid)
    if 32601 <= epsg_code <= 32660:
        zone = epsg_code - 32600
        south = False
    elif 32701 <= epsg_code <= 32760:
        zone = epsg_code - 32700
        south = True
    elif 26901 <= epsg_code <= 26923:
        # NAD83 UTM zones 1-23
        zone = epsg_code - 26900
        south = False
    else:
        return None

    lon0 = math.radians((zone - 1) * 6.0 - 180.0 + 3.0)  # central meridian
    k0 = 0.9996
    false_e = 500000.0
    false_n = 10000000.0 if south else 0.0
    return lon0, k0, false_e, false_n


# ---------------------------------------------------------------------------
# Dispatch: detect fast-path CRS pairs
# ---------------------------------------------------------------------------

def _get_epsg(crs):
    """Extract integer EPSG code from a pyproj.CRS, or None."""
    try:
        auth = crs.to_authority()
        if auth and auth[0].upper() == 'EPSG':
            return int(auth[1])
    except Exception:
        pass
    return None


def _is_geographic_wgs84_or_nad83(epsg):
    """True for EPSG:4326 (WGS84) or EPSG:4269 (NAD83)."""
    return epsg in (4326, 4269)


def try_numba_transform(src_crs, tgt_crs, chunk_bounds, chunk_shape):
    """Attempt a Numba JIT coordinate transform for the given CRS pair.

    Returns (src_y, src_x) arrays if a fast path exists, or None to
    fall back to pyproj.
    """
    src_epsg = _get_epsg(src_crs)
    tgt_epsg = _get_epsg(tgt_crs)
    if src_epsg is None or tgt_epsg is None:
        return None

    height, width = chunk_shape
    left, bottom, right, top = chunk_bounds
    res_x = (right - left) / width
    res_y = (top - bottom) / height

    # Build output coordinate arrays (target CRS)
    col_1d = np.arange(width, dtype=np.float64)
    row_1d = np.arange(height, dtype=np.float64)
    out_x_1d = left + (col_1d + 0.5) * res_x
    out_y_1d = top - (row_1d + 0.5) * res_y

    # Flatten for batch transform
    out_x_flat = np.tile(out_x_1d, height)
    out_y_flat = np.repeat(out_y_1d, width)
    n = out_x_flat.shape[0]
    src_x_flat = np.empty(n, dtype=np.float64)
    src_y_flat = np.empty(n, dtype=np.float64)

    # --- Geographic -> Web Mercator (inverse: Merc -> Geo) ---
    if _is_geographic_wgs84_or_nad83(src_epsg) and tgt_epsg == 3857:
        # Target is Mercator, need inverse: merc -> geo
        merc_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat)
        return (src_y_flat.reshape(height, width),
                src_x_flat.reshape(height, width))

    if src_epsg == 3857 and _is_geographic_wgs84_or_nad83(tgt_epsg):
        # Target is geographic, need forward: geo -> merc... wait, no.
        # We need the INVERSE transformer: target -> source.
        # target=geo, source=merc. So: geo -> merc (forward).
        merc_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat)
        return (src_y_flat.reshape(height, width),
                src_x_flat.reshape(height, width))

    # --- Geographic -> UTM (inverse: UTM -> Geo) ---
    if _is_geographic_wgs84_or_nad83(src_epsg):
        utm = _utm_params(tgt_epsg)
        if utm is not None:
            lon0, k0, fe, fn = utm
            Qn = k0 * _A_RECT
            # Target is UTM, need inverse: UTM -> Geo
            tmerc_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                          lon0, k0, fe, fn, Qn, _BETA, _CGB)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    # --- UTM -> Geographic (forward: Geo -> UTM) ---
    utm_src = _utm_params(src_epsg)
    if utm_src is not None and _is_geographic_wgs84_or_nad83(tgt_epsg):
        lon0, k0, fe, fn = utm_src
        Qn = k0 * _A_RECT
        # Target is geographic, need forward: Geo -> UTM
        tmerc_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                      lon0, k0, fe, fn, Qn, _ALPHA, _CBG)
        return (src_y_flat.reshape(height, width),
                src_x_flat.reshape(height, width))

    # --- Ellipsoidal Mercator (EPSG:3395) ---
    if _is_geographic_wgs84_or_nad83(src_epsg) and tgt_epsg == 3395:
        emerc_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                      1.0, _WGS84_E)
        return (src_y_flat.reshape(height, width),
                src_x_flat.reshape(height, width))
    if src_epsg == 3395 and _is_geographic_wgs84_or_nad83(tgt_epsg):
        emerc_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                      1.0, _WGS84_E)
        return (src_y_flat.reshape(height, width),
                src_x_flat.reshape(height, width))

    # --- Parameterised projections (LCC, AEA, CEA) ---
    # For these we need to parse the CRS parameters, so we operate on
    # the pyproj CRS objects directly rather than just EPSG codes.

    # LCC
    if _is_geographic_wgs84_or_nad83(src_epsg):
        params = _lcc_params(tgt_crs)
        if params is not None:
            lon0, nn, c, rho0, k0, fe, fn = params
            lcc_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, nn, c, rho0, k0, fe, fn, _WGS84_E, _WGS84_A)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    if _is_geographic_wgs84_or_nad83(tgt_epsg):
        params = _lcc_params(src_crs)
        if params is not None:
            lon0, nn, c, rho0, k0, fe, fn = params
            lcc_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, nn, c, rho0, k0, fe, fn, _WGS84_E, _WGS84_A)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    # AEA
    if _is_geographic_wgs84_or_nad83(src_epsg):
        params = _aea_params(tgt_crs)
        if params is not None:
            lon0, nn, C, rho0, fe, fn = params
            aea_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, nn, C, rho0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP, _APA)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    if _is_geographic_wgs84_or_nad83(tgt_epsg):
        params = _aea_params(src_crs)
        if params is not None:
            lon0, nn, C, rho0, fe, fn = params
            aea_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, nn, C, rho0, fe, fn,
                        _WGS84_E, _WGS84_A)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    # CEA
    if _is_geographic_wgs84_or_nad83(src_epsg):
        params = _cea_params(tgt_crs)
        if params is not None:
            lon0, k0, fe, fn = params
            cea_inverse(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, k0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP, _APA)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    if _is_geographic_wgs84_or_nad83(tgt_epsg):
        params = _cea_params(src_crs)
        if params is not None:
            lon0, k0, fe, fn = params
            cea_forward(out_x_flat, out_y_flat, src_x_flat, src_y_flat,
                        lon0, k0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP)
            return (src_y_flat.reshape(height, width),
                    src_x_flat.reshape(height, width))

    return None
