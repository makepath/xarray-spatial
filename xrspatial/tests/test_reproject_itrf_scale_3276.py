"""Regression tests for GH #3276.

``_helmert14_point`` converted the Helmert scale with ``* 1e-9``
("ppb"), but the parameters come from PROJ's ITRF data files and PROJ's
``+proj=helmert`` defines ``+s`` / ``+ds`` in ppm. The published
ppb-magnitude ITRF scale terms appear in the files as ppm values
(ITRF2014->ITRF2000 carries ``+s=0.00212`` ppm = 2.12 ppb), so the old
factor made the scale term 1000x too small: ~23 mm of height error for
that pair at epoch 2024, on a function whose stated purpose is mm-level
frame transforms.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for ITRF tests"
)


def _direct_params(src, tgt):
    """Raw helmert params for a direct (non-reversed) src->tgt entry."""
    from xrspatial.reproject._itrf import _find_transform

    params, is_reverse = _find_transform(src, tgt)
    if params is None or is_reverse:
        pytest.skip(f"no direct {src}->{tgt} entry in the PROJ data files")
    return params


def _pyproj_pipeline(params):
    """Build a 4D cart->helmert->cart pipeline from a raw param dict."""
    keys = ('x', 'y', 'z', 's', 'rx', 'ry', 'rz',
            'dx', 'dy', 'dz', 'ds', 'drx', 'dry', 'drz', 't_epoch')
    parts = [f"+{k}={params[k]}" for k in keys if k in params]
    convention = params.get('convention', 'position_vector')
    pipeline = (
        "+proj=pipeline "
        "+step +proj=cart +ellps=WGS84 "
        f"+step +proj=helmert {' '.join(parts)} +convention={convention} "
        "+step +inv +proj=cart +ellps=WGS84"
    )
    return pyproj.Transformer.from_pipeline(pipeline)


class TestItrfMatchesPyprojHelmert:
    @pytest.mark.parametrize("src,tgt", [
        ("ITRF2014", "ITRF2000"),
        ("ITRF2014", "ITRF2008"),
        ("ITRF2020", "ITRF2014"),
    ])
    def test_direct_pair_matches_pyproj_4d(self, src, tgt):
        """itrf_transform vs a pyproj pipeline from the same file entry.

        pyproj only applies the rate terms when given 4D coordinates,
        so the reference transform passes the epoch as the 4th value.
        Sub-mm agreement requires the ppm scale interpretation; the old
        ppb factor left a ~23 mm height gap on ITRF2014->ITRF2000.
        """
        from xrspatial.reproject import itrf_transform

        params = _direct_params(src, tgt)
        tr = _pyproj_pipeline(params)

        lon, lat, h = -74.0, 40.7, 100.0
        epoch = 2024.0
        rx, ry, rz, _ = tr.transform(lon, lat, h, epoch)
        ox, oy, oz = itrf_transform(lon, lat, h, src=src, tgt=tgt, epoch=epoch)

        de = (ox - rx) * 111320.0 * math.cos(math.radians(lat))
        dn = (oy - ry) * 111320.0
        dh = oz - rz
        err_mm = math.sqrt(de * de + dn * dn + dh * dh) * 1000.0
        assert err_mm < 0.5, f"{src}->{tgt} differs from pyproj by {err_mm:.3f} mm"

    def test_scale_term_not_dropped(self):
        """The scale term must contribute at its ppm magnitude.

        ITRF2014->ITRF2000 has s + ds*(2024-2010) = 0.00366 ppm, which
        is ~23 mm radially at the Earth's surface. With the broken ppb
        factor the contribution collapsed to ~23 um, so zeroing the
        scale barely changed the output. Compare the real transform
        against one with s and ds removed and require the difference to
        show up.
        """
        from xrspatial.reproject import itrf_transform

        params = _direct_params("ITRF2014", "ITRF2000")
        s = params.get('s', 0.0)
        ds = params.get('ds', 0.0)
        t_epoch = params.get('t_epoch', 2010.0)
        epoch = 2024.0
        s_eff_ppm = s + ds * (epoch - t_epoch)
        if abs(s_eff_ppm) < 1e-4:
            pytest.skip("scale term too small in this PROJ data version")

        lon, lat, h = -74.0, 40.7, 100.0
        ox, oy, oz = itrf_transform(lon, lat, h, src='ITRF2014',
                                    tgt='ITRF2000', epoch=epoch)
        # Reference without scale: pyproj pipeline with s/ds zeroed.
        params_noscale = dict(params)
        params_noscale['s'] = 0.0
        params_noscale['ds'] = 0.0
        tr = _pyproj_pipeline(params_noscale)
        nx, ny, nz, _ = tr.transform(lon, lat, h, epoch)

        de = (ox - nx) * 111320.0 * math.cos(math.radians(lat))
        dn = (oy - ny) * 111320.0
        dh = oz - nz
        shift_mm = math.sqrt(de * de + dn * dn + dh * dh) * 1000.0
        # Expected radial effect: s_eff * R. R = 6.37e6 m approximates
        # the geocentric radius at the test point (~6368 km); the 5%
        # tolerance absorbs that approximation. With the broken ppb
        # factor the measured shift is 1000x smaller and fails by three
        # orders of magnitude.
        expected_mm = abs(s_eff_ppm) * 1e-6 * 6.37e6 * 1000.0
        assert shift_mm == pytest.approx(expected_mm, rel=0.05)

    def test_reversed_lookup_matches_pyproj_inverse(self):
        """Reversed lookups negate the file parameters, scale included.

        ITRF2000 -> ITRF2014 has no direct file entry; _find_transform
        negates the ITRF2014 -> ITRF2000 parameters. The reference is
        the same pipeline with an inverse helmert step.
        """
        from xrspatial.reproject import itrf_transform
        from xrspatial.reproject._itrf import _find_transform

        params, is_reverse = _find_transform("ITRF2000", "ITRF2014")
        if params is None or not is_reverse:
            pytest.skip("ITRF2000->ITRF2014 is not a reversed lookup here")
        keys = ('x', 'y', 'z', 's', 'rx', 'ry', 'rz',
                'dx', 'dy', 'dz', 'ds', 'drx', 'dry', 'drz', 't_epoch')
        parts = [f"+{k}={params[k]}" for k in keys if k in params]
        convention = params.get('convention', 'position_vector')
        tr = pyproj.Transformer.from_pipeline(
            "+proj=pipeline "
            "+step +proj=cart +ellps=WGS84 "
            f"+step +inv +proj=helmert {' '.join(parts)} "
            f"+convention={convention} "
            "+step +inv +proj=cart +ellps=WGS84"
        )
        lon, lat, h = -74.0, 40.7, 100.0
        epoch = 2024.0
        rx, ry, rz, _ = tr.transform(lon, lat, h, epoch)
        ox, oy, oz = itrf_transform(lon, lat, h, src='ITRF2000',
                                    tgt='ITRF2014', epoch=epoch)
        de = (ox - rx) * 111320.0 * math.cos(math.radians(lat))
        dn = (oy - ry) * 111320.0
        dh = oz - rz
        err_mm = math.sqrt(de * de + dn * dn + dh * dh) * 1000.0
        assert err_mm < 0.5

    def test_array_and_scalar_agree(self):
        from xrspatial.reproject import itrf_transform

        _direct_params("ITRF2014", "ITRF2000")
        lons = np.array([-74.0, 10.0])
        lats = np.array([40.7, 50.0])
        hs = np.array([100.0, 0.0])
        alon, alat, ah = itrf_transform(lons, lats, hs, src='ITRF2014',
                                        tgt='ITRF2000', epoch=2024.0)
        slon, slat, sh = itrf_transform(-74.0, 40.7, 100.0, src='ITRF2014',
                                        tgt='ITRF2000', epoch=2024.0)
        assert alon[0] == pytest.approx(slon, abs=1e-12)
        assert alat[0] == pytest.approx(slat, abs=1e-12)
        assert ah[0] == pytest.approx(sh, abs=1e-9)
