"""CUDA JIT coordinate transforms for common projections.

GPU equivalents of the Numba CPU kernels in ``_projections.py``.
Each kernel computes source CRS coordinates directly on-device,
avoiding the CPU->GPU transfer of coordinate arrays.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Ellipsoid constants (duplicated here so CUDA device functions see them
# as compile-time constants rather than module-level loads).
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = 2.0 * _F - _F * _F
_E = math.sqrt(_E2)

if not HAS_CUDA:
    # Provide a no-op so the module can be imported without CUDA.
    def try_cuda_transform(*args, **kwargs):
        return None
else:

    # -----------------------------------------------------------------
    # Shared device helpers
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_pj_sinhpsi2tanphi(taup, e):
        e2 = e * e
        tau = taup
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

    @cuda.jit(device=True)
    def _d_authalic_q(sinphi, e):
        e2 = e * e
        es = e * sinphi
        return (1.0 - e2) * (sinphi / (1.0 - es * es) + math.atanh(es) / e)

    @cuda.jit(device=True)
    def _d_authalic_inv(beta, apa0, apa1, apa2, apa3, apa4):
        t = 2.0 * beta
        return (beta
                + apa0 * math.sin(t)
                + apa1 * math.sin(2.0 * t)
                + apa2 * math.sin(3.0 * t)
                + apa3 * math.sin(4.0 * t)
                + apa4 * math.sin(5.0 * t))

    @cuda.jit(device=True)
    def _d_clenshaw_sin(c0, c1, c2, c3, c4, c5, angle):
        X = 2.0 * math.cos(2.0 * angle)
        u0 = 0.0
        u1 = 0.0
        for c in (c5, c4, c3, c2, c1, c0):
            t = X * u0 - u1 + c
            u1 = u0
            u0 = t
        return math.sin(2.0 * angle) * u0

    @cuda.jit(device=True)
    def _d_clenshaw_complex(a0, a1, a2, a3, a4, a5,
                            sin2Cn, cos2Cn, sinh2Ce, cosh2Ce):
        r = 2.0 * cos2Cn * cosh2Ce
        im = -2.0 * sin2Cn * sinh2Ce
        hr = 0.0; hi = 0.0; hr1 = 0.0; hi1 = 0.0
        for a in (a5, a4, a3, a2, a1, a0):
            hr2 = hr1; hi2 = hi1; hr1 = hr; hi1 = hi
            hr = -hr2 + r * hr1 - im * hi1 + a
            hi = -hi2 + im * hr1 + r * hi1
        dCn = sin2Cn * cosh2Ce * hr - cos2Cn * sinh2Ce * hi
        dCe = sin2Cn * cosh2Ce * hi + cos2Cn * sinh2Ce * hr
        return dCn, dCe

    @cuda.jit(device=True)
    def _d_norm_lon_rad(lon):
        """Normalize longitude to [-pi, pi]."""
        while lon > math.pi:
            lon -= 2.0 * math.pi
        while lon < -math.pi:
            lon += 2.0 * math.pi
        return lon

    # -----------------------------------------------------------------
    # Web Mercator (EPSG:3857)  --  spherical
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_merc_inv(x, y):
        lon = math.degrees(x / _A)
        lat = math.degrees(math.atan(math.sinh(y / _A)))
        return lon, lat

    @cuda.jit(device=True)
    def _d_merc_fwd(lon_deg, lat_deg):
        x = _A * math.radians(lon_deg)
        phi = math.radians(lat_deg)
        y = _A * math.log(math.tan(math.pi / 4.0 + phi / 2.0))
        return x, y

    @cuda.jit
    def _k_merc_inverse(out_src_x, out_src_y,
                        left, top, res_x, res_y):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x
            ty = top - (i + 0.5) * res_y
            lon, lat = _d_merc_inv(tx, ty)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_merc_forward(out_src_x, out_src_y,
                        left, top, res_x, res_y):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_merc_fwd(lon, lat)
            out_src_x[i, j] = x
            out_src_y[i, j] = y

    # -----------------------------------------------------------------
    # Ellipsoidal Mercator (EPSG:3395)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_emerc_inv(x, y, k0, e):
        lam = x / (k0 * _A)
        taup = math.sinh(y / (k0 * _A))
        tau = _d_pj_sinhpsi2tanphi(taup, e)
        return math.degrees(lam), math.degrees(math.atan(tau))

    @cuda.jit(device=True)
    def _d_emerc_fwd(lon_deg, lat_deg, k0, e):
        lam = math.radians(lon_deg)
        phi = math.radians(lat_deg)
        sinphi = math.sin(phi)
        x = k0 * _A * lam
        y = k0 * _A * (math.asinh(math.tan(phi)) - e * math.atanh(e * sinphi))
        return x, y

    @cuda.jit
    def _k_emerc_inverse(out_src_x, out_src_y,
                         left, top, res_x, res_y, k0, e):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x
            ty = top - (i + 0.5) * res_y
            lon, lat = _d_emerc_inv(tx, ty, k0, e)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_emerc_forward(out_src_x, out_src_y,
                         left, top, res_x, res_y, k0, e):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_emerc_fwd(lon, lat, k0, e)
            out_src_x[i, j] = x
            out_src_y[i, j] = y

    # -----------------------------------------------------------------
    # Transverse Mercator / UTM  --  Krueger series
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_tmerc_fwd(lon_deg, lat_deg, lon0, Qn,
                     a0, a1, a2, a3, a4, a5,
                     c0, c1, c2, c3, c4, c5):
        lam = math.radians(lon_deg) - lon0
        phi = math.radians(lat_deg)
        chi = phi + _d_clenshaw_sin(c0, c1, c2, c3, c4, c5, phi)
        sin_chi = math.sin(chi)
        cos_chi = math.cos(chi)
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        denom = math.hypot(sin_chi, cos_chi * cos_lam)
        if denom < 1e-30:
            denom = 1e-30
        Cn = math.atan2(sin_chi, cos_chi * cos_lam)
        tan_Ce = sin_lam * cos_chi / denom
        if tan_Ce > 1e15:
            tan_Ce = 1e15
        elif tan_Ce < -1e15:
            tan_Ce = -1e15
        Ce = math.asinh(tan_Ce)
        inv_d = 1.0 / denom
        inv_d2 = inv_d * inv_d
        ccl = cos_chi * cos_lam
        sin2 = 2.0 * sin_chi * ccl * inv_d2
        cos2 = 2.0 * ccl * ccl * inv_d2 - 1.0
        sinh2 = 2.0 * tan_Ce * inv_d
        cosh2 = 2.0 * inv_d2 - 1.0
        dCn, dCe = _d_clenshaw_complex(a0, a1, a2, a3, a4, a5,
                                        sin2, cos2, sinh2, cosh2)
        return Qn * (Ce + dCe), Qn * (Cn + dCn)

    @cuda.jit(device=True)
    def _d_tmerc_inv(x, y, lon0, Qn,
                     b0, b1, b2, b3, b4, b5,
                     g0, g1, g2, g3, g4, g5):
        Cn = y / Qn
        Ce = x / Qn
        sin2Cn = math.sin(2.0 * Cn)
        cos2Cn = math.cos(2.0 * Cn)
        exp2Ce = math.exp(2.0 * Ce)
        inv_exp = 1.0 / exp2Ce
        sinh2Ce = 0.5 * (exp2Ce - inv_exp)
        cosh2Ce = 0.5 * (exp2Ce + inv_exp)
        dCn, dCe = _d_clenshaw_complex(b0, b1, b2, b3, b4, b5,
                                        sin2Cn, cos2Cn, sinh2Ce, cosh2Ce)
        Cn -= dCn
        Ce -= dCe
        sin_Cn = math.sin(Cn)
        cos_Cn = math.cos(Cn)
        sinh_Ce = math.sinh(Ce)
        lam = math.atan2(sinh_Ce, cos_Cn)
        modulus = math.hypot(sinh_Ce, cos_Cn)
        chi = math.atan2(sin_Cn, modulus)
        phi = chi + _d_clenshaw_sin(g0, g1, g2, g3, g4, g5, chi)
        return math.degrees(lam + lon0), math.degrees(phi)

    @cuda.jit
    def _k_tmerc_inverse(out_src_x, out_src_y,
                         left, top, res_x, res_y,
                         lon0, fe, fn, Qn,
                         b0, b1, b2, b3, b4, b5,
                         g0, g1, g2, g3, g4, g5):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_tmerc_inv(tx, ty, lon0, Qn,
                                     b0, b1, b2, b3, b4, b5,
                                     g0, g1, g2, g3, g4, g5)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_tmerc_forward(out_src_x, out_src_y,
                         left, top, res_x, res_y,
                         lon0, fe, fn, Qn,
                         a0, a1, a2, a3, a4, a5,
                         c0, c1, c2, c3, c4, c5):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_tmerc_fwd(lon, lat, lon0, Qn,
                                 a0, a1, a2, a3, a4, a5,
                                 c0, c1, c2, c3, c4, c5)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Lambert Conformal Conic (LCC)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_lcc_fwd(lon_deg, lat_deg, lon0, n, c, rho0, k0, e, a):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - lon0
        sinphi = math.sin(phi)
        es = e * sinphi
        ts = math.tan(math.pi / 4.0 - phi / 2.0) * math.pow(
            (1.0 + es) / (1.0 - es), e / 2.0)
        rho = a * k0 * c * math.pow(ts, n)
        lam_n = n * lam
        return rho * math.sin(lam_n), rho0 - rho * math.cos(lam_n)

    @cuda.jit(device=True)
    def _d_lcc_inv(x, y, lon0, n, c, rho0, k0, e, a):
        rho0_y = rho0 - y
        if n < 0.0:
            rho = -math.hypot(x, rho0_y)
            lam_n = math.atan2(-x, -rho0_y)
        else:
            rho = math.hypot(x, rho0_y)
            lam_n = math.atan2(x, rho0_y)
        if abs(rho) < 1e-30:
            lat = 90.0 if n > 0 else -90.0
            return math.degrees(_d_norm_lon_rad(lon0 + lam_n / n)), lat
        ts = math.pow(rho / (a * k0 * c), 1.0 / n)
        taup = math.sinh(math.log(1.0 / ts))
        tau = _d_pj_sinhpsi2tanphi(taup, e)
        return math.degrees(_d_norm_lon_rad(lam_n / n + lon0)), math.degrees(math.atan(tau))

    @cuda.jit
    def _k_lcc_inverse(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, n, c, rho0, k0, fe, fn, e, a):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_lcc_inv(tx, ty, lon0, n, c, rho0, k0, e, a)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_lcc_forward(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, n, c, rho0, k0, fe, fn, e, a):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_lcc_fwd(lon, lat, lon0, n, c, rho0, k0, e, a)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Albers Equal Area (AEA)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_aea_fwd(lon_deg, lat_deg, lon0, n, C, rho0, e, a):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - lon0
        q = _d_authalic_q(math.sin(phi), e)
        val = C - n * q
        if val < 0.0:
            val = 0.0
        rho = a * math.sqrt(val) / n
        theta = n * lam
        return rho * math.sin(theta), rho0 - rho * math.cos(theta)

    @cuda.jit(device=True)
    def _d_aea_inv(x, y, lon0, n, C, rho0, e, a, qp,
                   apa0, apa1, apa2, apa3, apa4):
        rho0_y = rho0 - y
        if n < 0.0:
            rho = -math.hypot(x, rho0_y)
            theta = math.atan2(-x, -rho0_y)
        else:
            rho = math.hypot(x, rho0_y)
            theta = math.atan2(x, rho0_y)
        q = (C - (rho * rho * n * n) / (a * a)) / n
        ratio = q / qp
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        beta = math.asin(ratio)
        phi = _d_authalic_inv(beta, apa0, apa1, apa2, apa3, apa4)
        return math.degrees(_d_norm_lon_rad(theta / n + lon0)), math.degrees(phi)

    @cuda.jit
    def _k_aea_inverse(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, n, C, rho0, fe, fn, e, a, qp,
                       apa0, apa1, apa2, apa3, apa4):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_aea_inv(tx, ty, lon0, n, C, rho0, e, a, qp,
                                   apa0, apa1, apa2, apa3, apa4)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_aea_forward(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, n, C, rho0, fe, fn, e, a):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_aea_fwd(lon, lat, lon0, n, C, rho0, e, a)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Cylindrical Equal Area (CEA)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_cea_fwd(lon_deg, lat_deg, lon0, k0, e, a, qp):
        lam = math.radians(lon_deg) - lon0
        phi = math.radians(lat_deg)
        q = _d_authalic_q(math.sin(phi), e)
        return a * k0 * lam, a * q / (2.0 * k0)

    @cuda.jit(device=True)
    def _d_cea_inv(x, y, lon0, k0, e, a, qp, apa0, apa1, apa2, apa3, apa4):
        lam = x / (a * k0)
        ratio = 2.0 * y * k0 / (a * qp)
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        beta = math.asin(ratio)
        phi = _d_authalic_inv(beta, apa0, apa1, apa2, apa3, apa4)
        return math.degrees(_d_norm_lon_rad(lam + lon0)), math.degrees(phi)

    @cuda.jit
    def _k_cea_inverse(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, k0, fe, fn, e, a, qp,
                       apa0, apa1, apa2, apa3, apa4):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_cea_inv(tx, ty, lon0, k0, e, a, qp,
                                   apa0, apa1, apa2, apa3, apa4)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_cea_forward(out_src_x, out_src_y,
                       left, top, res_x, res_y,
                       lon0, k0, fe, fn, e, a, qp):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_cea_fwd(lon, lat, lon0, k0, e, a, qp)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Sinusoidal (ellipsoidal)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_mlfn(phi, sinphi, cosphi, en0, en1, en2, en3, en4):
        cphi = cosphi * sinphi
        sphi = sinphi * sinphi
        return en0 * phi - cphi * (en1 + sphi * (en2 + sphi * (en3 + sphi * en4)))

    @cuda.jit(device=True)
    def _d_inv_mlfn(arg, e2, en0, en1, en2, en3, en4):
        k = 1.0 / (1.0 - e2)
        phi = arg
        for _ in range(20):
            s = math.sin(phi)
            c = math.cos(phi)
            t = 1.0 - e2 * s * s
            dphi = (arg - _d_mlfn(phi, s, c, en0, en1, en2, en3, en4)) * t * math.sqrt(t) * k
            phi += dphi
            if abs(dphi) < 1e-14:
                break
        return phi

    @cuda.jit(device=True)
    def _d_sinu_fwd(lon_deg, lat_deg, lon0, e2, a, en0, en1, en2, en3, en4):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - lon0
        s = math.sin(phi)
        c = math.cos(phi)
        ms = _d_mlfn(phi, s, c, en0, en1, en2, en3, en4)
        x = a * lam * c / math.sqrt(1.0 - e2 * s * s)
        y = a * ms
        return x, y

    @cuda.jit(device=True)
    def _d_sinu_inv(x, y, lon0, e2, a, en0, en1, en2, en3, en4):
        phi = _d_inv_mlfn(y / a, e2, en0, en1, en2, en3, en4)
        s = math.sin(phi)
        c = math.cos(phi)
        if abs(c) < 1e-14:
            lam = 0.0
        else:
            lam = x * math.sqrt(1.0 - e2 * s * s) / (a * c)
        return math.degrees(_d_norm_lon_rad(lam + lon0)), math.degrees(phi)

    @cuda.jit
    def _k_sinu_inverse(out_src_x, out_src_y,
                        left, top, res_x, res_y,
                        lon0, fe, fn, e2, a,
                        en0, en1, en2, en3, en4):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_sinu_inv(tx, ty, lon0, e2, a, en0, en1, en2, en3, en4)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_sinu_forward(out_src_x, out_src_y,
                        left, top, res_x, res_y,
                        lon0, fe, fn, e2, a,
                        en0, en1, en2, en3, en4):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_sinu_fwd(lon, lat, lon0, e2, a, en0, en1, en2, en3, en4)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Lambert Azimuthal Equal Area (LAEA)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_laea_fwd(lon_deg, lat_deg, lon0, sinb1, cosb1,
                    xmf, ymf, rq, qp, e, a, e2, mode):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - lon0
        sinphi = math.sin(phi)
        q = _d_authalic_q(sinphi, e)
        sinb = q / qp
        if sinb > 1.0:
            sinb = 1.0
        elif sinb < -1.0:
            sinb = -1.0
        cosb = math.sqrt(1.0 - sinb * sinb)
        coslam = math.cos(lam)
        sinlam = math.sin(lam)
        if mode == 0:  # OBLIQ
            denom = 1.0 + sinb1 * sinb + cosb1 * cosb * coslam
            if denom < 1e-30:
                denom = 1e-30
            b = math.sqrt(2.0 / denom)
            x = a * xmf * b * cosb * sinlam
            y = a * ymf * b * (cosb1 * sinb - sinb1 * cosb * coslam)
        elif mode == 1:  # EQUIT
            denom = 1.0 + cosb * coslam
            if denom < 1e-30:
                denom = 1e-30
            b = math.sqrt(2.0 / denom)
            x = a * xmf * b * cosb * sinlam
            y = a * ymf * b * sinb
        elif mode == 2:  # N_POLE
            q_diff = qp - q
            if q_diff < 0.0:
                q_diff = 0.0
            rho = a * math.sqrt(q_diff)
            x = rho * sinlam
            y = -rho * coslam
        else:  # S_POLE
            q_diff = qp + q
            if q_diff < 0.0:
                q_diff = 0.0
            rho = a * math.sqrt(q_diff)
            x = rho * sinlam
            y = rho * coslam
        return x, y

    @cuda.jit(device=True)
    def _d_laea_inv(x, y, lon0, sinb1, cosb1,
                    xmf, ymf, rq, qp, e, a, e2, mode,
                    apa0, apa1, apa2, apa3, apa4):
        if mode == 2 or mode == 3:
            x_a = x / a
            y_a = y / a
            rho = math.hypot(x_a, y_a)
            if rho < 1e-30:
                lat = 90.0 if mode == 2 else -90.0
                return math.degrees(lon0), lat
            q = qp - rho * rho
            if mode == 3:
                q = -(qp - rho * rho)
                lam = math.atan2(x_a, y_a)
            else:
                lam = math.atan2(x_a, -y_a)
        else:
            xn = x / (a * xmf)
            yn = y / (a * ymf)
            rho = math.hypot(xn, yn)
            if rho < 1e-30:
                return math.degrees(lon0), math.degrees(math.asin(sinb1))
            sce = 2.0 * math.asin(0.5 * rho / rq)
            sinz = math.sin(sce)
            cosz = math.cos(sce)
            if mode == 0:
                ab = cosz * sinb1 + yn * sinz * cosb1 / rho
                lam = math.atan2(xn * sinz,
                                  rho * cosb1 * cosz - yn * sinb1 * sinz)
            else:
                ab = yn * sinz / rho
                lam = math.atan2(xn * sinz, rho * cosz)
            q = qp * ab
        ratio = q / qp
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        beta = math.asin(ratio)
        phi = _d_authalic_inv(beta, apa0, apa1, apa2, apa3, apa4)
        return math.degrees(_d_norm_lon_rad(lam + lon0)), math.degrees(phi)

    @cuda.jit
    def _k_laea_inverse(out_src_x, out_src_y,
                        left, top, res_x, res_y,
                        lon0, sinb1, cosb1, xmf, ymf, rq, qp,
                        fe, fn, e, a, e2, mode,
                        apa0, apa1, apa2, apa3, apa4):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_laea_inv(tx, ty, lon0, sinb1, cosb1,
                                    xmf, ymf, rq, qp, e, a, e2, mode,
                                    apa0, apa1, apa2, apa3, apa4)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_laea_forward(out_src_x, out_src_y,
                        left, top, res_x, res_y,
                        lon0, sinb1, cosb1, xmf, ymf, rq, qp,
                        fe, fn, e, a, e2, mode):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_laea_fwd(lon, lat, lon0, sinb1, cosb1,
                                xmf, ymf, rq, qp, e, a, e2, mode)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Polar Stereographic (N/S pole)
    # -----------------------------------------------------------------

    @cuda.jit(device=True)
    def _d_stere_fwd(lon_deg, lat_deg, lon0, akm1, e, is_south):
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg) - lon0
        abs_phi = -phi if is_south else phi
        sinphi = math.sin(abs_phi)
        es = e * sinphi
        ts = math.tan(math.pi / 4.0 - abs_phi / 2.0) * math.pow(
            (1.0 + es) / (1.0 - es), e / 2.0)
        rho = akm1 * ts
        if is_south:
            return rho * math.sin(lam), rho * math.cos(lam)
        else:
            return rho * math.sin(lam), -rho * math.cos(lam)

    @cuda.jit(device=True)
    def _d_stere_inv(x, y, lon0, akm1, e, is_south):
        if is_south:
            rho = math.hypot(x, y)
            lam = math.atan2(x, y)
        else:
            rho = math.hypot(x, y)
            lam = math.atan2(x, -y)
        if rho < 1e-30:
            lat = -90.0 if is_south else 90.0
            return math.degrees(lon0), lat
        tp = rho / akm1
        half_e = e / 2.0
        phi = math.pi / 2.0 - 2.0 * math.atan(tp)
        for _ in range(15):
            sinphi = math.sin(phi)
            es = e * sinphi
            phi_new = math.pi / 2.0 - 2.0 * math.atan(
                tp * math.pow((1.0 - es) / (1.0 + es), half_e))
            if abs(phi_new - phi) < 1e-14:
                phi = phi_new
                break
            phi = phi_new
        if is_south:
            phi = -phi
        return math.degrees(_d_norm_lon_rad(lam + lon0)), math.degrees(phi)

    @cuda.jit
    def _k_stere_inverse(out_src_x, out_src_y,
                         left, top, res_x, res_y,
                         lon0, akm1, fe, fn, e, is_south):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            tx = left + (j + 0.5) * res_x - fe
            ty = top - (i + 0.5) * res_y - fn
            lon, lat = _d_stere_inv(tx, ty, lon0, akm1, e, is_south)
            out_src_x[i, j] = lon
            out_src_y[i, j] = lat

    @cuda.jit
    def _k_stere_forward(out_src_x, out_src_y,
                         left, top, res_x, res_y,
                         lon0, akm1, fe, fn, e, is_south):
        i, j = cuda.grid(2)
        if i < out_src_x.shape[0] and j < out_src_x.shape[1]:
            lon = left + (j + 0.5) * res_x
            lat = top - (i + 0.5) * res_y
            x, y = _d_stere_fwd(lon, lat, lon0, akm1, e, is_south)
            out_src_x[i, j] = x + fe
            out_src_y[i, j] = y + fn

    # -----------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------

    def _cuda_dims(shape):
        """Compute (blocks_per_grid, threads_per_block) for a 2D kernel."""
        tpb = (16, 16)  # conservative to avoid register pressure
        bpg = (
            (shape[0] + tpb[0] - 1) // tpb[0],
            (shape[1] + tpb[1] - 1) // tpb[1],
        )
        return bpg, tpb

    def try_cuda_transform(src_crs, tgt_crs, chunk_bounds, chunk_shape):
        """Attempt a CUDA JIT coordinate transform.

        Returns (src_y, src_x) as CuPy arrays if a fast path exists,
        or None to fall back to CPU.
        """
        import cupy as cp
        from ._projections import (
            _get_epsg, _is_geographic_wgs84_or_nad83, _utm_params,
            _tmerc_params, _lcc_params, _aea_params, _cea_params,
            _sinu_params, _laea_params, _stere_params,
            _ALPHA, _BETA, _CBG, _CGB, _A_RECT, _QP, _APA,
            _WGS84_E2, _MLFN_EN,
        )

        src_epsg = _get_epsg(src_crs)
        tgt_epsg = _get_epsg(tgt_crs)
        if src_epsg is None and tgt_epsg is None:
            return None

        height, width = chunk_shape
        left, bottom, right, top = chunk_bounds
        res_x = (right - left) / width
        res_y = (top - bottom) / height

        out_src_x = cp.empty((height, width), dtype=cp.float64)
        out_src_y = cp.empty((height, width), dtype=cp.float64)
        bpg, tpb = _cuda_dims((height, width))

        # --- Web Mercator ---
        if _is_geographic_wgs84_or_nad83(src_epsg) and tgt_epsg == 3857:
            _k_merc_inverse[bpg, tpb](out_src_x, out_src_y,
                                       left, top, res_x, res_y)
            return out_src_y, out_src_x

        if src_epsg == 3857 and _is_geographic_wgs84_or_nad83(tgt_epsg):
            _k_merc_forward[bpg, tpb](out_src_x, out_src_y,
                                       left, top, res_x, res_y)
            return out_src_y, out_src_x

        # --- UTM ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            utm = _utm_params(tgt_epsg)
            if utm is not None:
                lon0, k0, fe, fn = utm
                Qn = k0 * _A_RECT
                _k_tmerc_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, fe, fn, Qn,
                    _BETA[0], _BETA[1], _BETA[2], _BETA[3], _BETA[4], _BETA[5],
                    _CGB[0], _CGB[1], _CGB[2], _CGB[3], _CGB[4], _CGB[5],
                )
                return out_src_y, out_src_x

        utm_src = _utm_params(src_epsg) if src_epsg else None
        if utm_src is not None and _is_geographic_wgs84_or_nad83(tgt_epsg):
            lon0, k0, fe, fn = utm_src
            Qn = k0 * _A_RECT
            _k_tmerc_forward[bpg, tpb](
                out_src_x, out_src_y, left, top, res_x, res_y,
                lon0, fe, fn, Qn,
                _ALPHA[0], _ALPHA[1], _ALPHA[2], _ALPHA[3], _ALPHA[4], _ALPHA[5],
                _CBG[0], _CBG[1], _CBG[2], _CBG[3], _CBG[4], _CBG[5],
            )
            return out_src_y, out_src_x

        # --- Ellipsoidal Mercator ---
        if _is_geographic_wgs84_or_nad83(src_epsg) and tgt_epsg == 3395:
            _k_emerc_inverse[bpg, tpb](out_src_x, out_src_y,
                                        left, top, res_x, res_y, 1.0, _E)
            return out_src_y, out_src_x

        if src_epsg == 3395 and _is_geographic_wgs84_or_nad83(tgt_epsg):
            _k_emerc_forward[bpg, tpb](out_src_x, out_src_y,
                                        left, top, res_x, res_y, 1.0, _E)
            return out_src_y, out_src_x

        # --- Generic Transverse Mercator (State Plane, etc.) ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            tmerc_p = _tmerc_params(tgt_crs)
            if tmerc_p is not None:
                lon0, k0, fe, fn, Zb, to_m = tmerc_p
                Qn = k0 * _A_RECT
                if to_m != 1.0:
                    _k_tmerc_inverse[bpg, tpb](
                        out_src_x, out_src_y,
                        left * to_m, top * to_m, res_x * to_m, res_y * to_m,
                        lon0, fe, fn + Zb, Qn,
                        _BETA[0], _BETA[1], _BETA[2], _BETA[3], _BETA[4], _BETA[5],
                        _CGB[0], _CGB[1], _CGB[2], _CGB[3], _CGB[4], _CGB[5],
                    )
                else:
                    _k_tmerc_inverse[bpg, tpb](
                        out_src_x, out_src_y, left, top, res_x, res_y,
                        lon0, fe, fn + Zb, Qn,
                        _BETA[0], _BETA[1], _BETA[2], _BETA[3], _BETA[4], _BETA[5],
                        _CGB[0], _CGB[1], _CGB[2], _CGB[3], _CGB[4], _CGB[5],
                    )
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            tmerc_p = _tmerc_params(src_crs)
            if tmerc_p is not None:
                lon0, k0, fe, fn, Zb, to_m = tmerc_p
                Qn = k0 * _A_RECT
                _k_tmerc_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, fe, fn + Zb, Qn,
                    _ALPHA[0], _ALPHA[1], _ALPHA[2], _ALPHA[3], _ALPHA[4], _ALPHA[5],
                    _CBG[0], _CBG[1], _CBG[2], _CBG[3], _CBG[4], _CBG[5],
                )
                if to_m != 1.0:
                    out_src_x /= to_m
                    out_src_y /= to_m
                return out_src_y, out_src_x

        # --- LCC ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _lcc_params(tgt_crs)
            if params is not None:
                lon0, nn, c, rho0, k0, fe, fn, to_m = params
                if to_m != 1.0:
                    _k_lcc_inverse[bpg, tpb](
                        out_src_x, out_src_y,
                        left * to_m, top * to_m, res_x * to_m, res_y * to_m,
                        lon0, nn, c, rho0, k0, fe, fn, _E, _A)
                else:
                    _k_lcc_inverse[bpg, tpb](
                        out_src_x, out_src_y, left, top, res_x, res_y,
                        lon0, nn, c, rho0, k0, fe, fn, _E, _A)
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _lcc_params(src_crs)
            if params is not None:
                lon0, nn, c, rho0, k0, fe, fn, to_m = params
                _k_lcc_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, nn, c, rho0, k0, fe, fn, _E, _A)
                if to_m != 1.0:
                    out_src_x /= to_m
                    out_src_y /= to_m
                return out_src_y, out_src_x

        # --- AEA ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _aea_params(tgt_crs)
            if params is not None:
                lon0, nn, C, rho0, fe, fn = params
                _k_aea_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, nn, C, rho0, fe, fn, _E, _A, _QP,
                    _APA[0], _APA[1], _APA[2], _APA[3], _APA[4])
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _aea_params(src_crs)
            if params is not None:
                lon0, nn, C, rho0, fe, fn = params
                _k_aea_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, nn, C, rho0, fe, fn, _E, _A)
                return out_src_y, out_src_x

        # --- CEA ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _cea_params(tgt_crs)
            if params is not None:
                lon0, k0, fe, fn = params
                _k_cea_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, k0, fe, fn, _E, _A, _QP,
                    _APA[0], _APA[1], _APA[2], _APA[3], _APA[4])
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _cea_params(src_crs)
            if params is not None:
                lon0, k0, fe, fn = params
                _k_cea_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, k0, fe, fn, _E, _A, _QP)
                return out_src_y, out_src_x

        # --- Sinusoidal ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _sinu_params(tgt_crs)
            if params is not None:
                lon0, fe, fn = params
                en = _MLFN_EN
                _k_sinu_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, fe, fn, _WGS84_E2, _A,
                    en[0], en[1], en[2], en[3], en[4])
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _sinu_params(src_crs)
            if params is not None:
                lon0, fe, fn = params
                en = _MLFN_EN
                _k_sinu_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, fe, fn, _WGS84_E2, _A,
                    en[0], en[1], en[2], en[3], en[4])
                return out_src_y, out_src_x

        # --- LAEA ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _laea_params(tgt_crs)
            if params is not None:
                lon0, lat0, sinb1, cosb1, dd, xmf, ymf, rq, qp, fe, fn, mode = params
                _k_laea_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, sinb1, cosb1, xmf, ymf, rq, qp,
                    fe, fn, _E, _A, _WGS84_E2, mode,
                    _APA[0], _APA[1], _APA[2], _APA[3], _APA[4])
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _laea_params(src_crs)
            if params is not None:
                lon0, lat0, sinb1, cosb1, dd, xmf, ymf, rq, qp, fe, fn, mode = params
                _k_laea_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, sinb1, cosb1, xmf, ymf, rq, qp,
                    fe, fn, _E, _A, _WGS84_E2, mode)
                return out_src_y, out_src_x

        # --- Polar Stereographic ---
        if _is_geographic_wgs84_or_nad83(src_epsg):
            params = _stere_params(tgt_crs)
            if params is not None:
                lon0, k0, akm1, fe, fn, is_south = params
                _k_stere_inverse[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, akm1, fe, fn, _E, is_south)
                return out_src_y, out_src_x

        if _is_geographic_wgs84_or_nad83(tgt_epsg):
            params = _stere_params(src_crs)
            if params is not None:
                lon0, k0, akm1, fe, fn, is_south = params
                _k_stere_forward[bpg, tpb](
                    out_src_x, out_src_y, left, top, res_x, res_y,
                    lon0, akm1, fe, fn, _E, is_south)
                return out_src_y, out_src_x

        return None
