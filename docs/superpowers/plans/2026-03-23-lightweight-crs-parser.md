# Lightweight CRS Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make common reprojection (UTM, Web Mercator, LCC, Albers, etc.) work without pyproj by embedding CRS metadata for EPSG codes that have Numba fast paths.

**Architecture:** New `CRS` class in `_lite_crs.py` with an embedded EPSG table. `_crs_utils.py` tries our `CRS` first, falls back to pyproj. `_grid.py` gets a `_transform_points()` helper that reuses existing Numba kernels for scatter-point boundary transforms. Chunk functions and `_source_footprint_in_target` use two-tier CRS resolution.

**Tech Stack:** Python 3, NumPy, Numba (existing), pytest

**Spec:** `docs/superpowers/specs/2026-03-23-lightweight-crs-parser-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `xrspatial/reproject/_lite_crs.py` | **New.** `CRS` class, EPSG lookup table, WKT generation |
| `xrspatial/reproject/_crs_utils.py` | **Modify.** Two-tier `_resolve_crs()`, new `_crs_from_wkt()` |
| `xrspatial/reproject/_projections.py` | **Modify.** New `transform_points()` scatter-point helper |
| `xrspatial/reproject/_grid.py` | **Modify.** Use `transform_points()` with pyproj fallback |
| `xrspatial/reproject/__init__.py` | **Modify.** Use `_crs_from_wkt()` in chunk functions; restructure CuPy chunk; update `_source_footprint_in_target` |
| `xrspatial/tests/test_lite_crs.py` | **New.** Unit + integration tests |

---

### Task 1: CRS class -- construction and basic interface

**Files:**
- Create: `xrspatial/tests/test_lite_crs.py`
- Create: `xrspatial/reproject/_lite_crs.py`

- [ ] **Step 1: Write failing tests for CRS construction and interface**

```python
"""Tests for the lightweight CRS class."""
from __future__ import annotations

import pytest


class TestCRSConstruction:
    def test_from_epsg_int(self):
        from xrspatial.reproject._lite_crs import CRS
        crs = CRS(4326)
        assert crs.to_epsg() == 4326

    def test_from_epsg_classmethod(self):
        from xrspatial.reproject._lite_crs import CRS
        crs = CRS.from_epsg(4326)
        assert crs.to_epsg() == 4326

    def test_from_authority_string(self):
        from xrspatial.reproject._lite_crs import CRS
        crs = CRS("EPSG:32632")
        assert crs.to_epsg() == 32632

    def test_unknown_epsg_raises(self):
        from xrspatial.reproject._lite_crs import CRS
        with pytest.raises(ValueError, match="not in the built-in table"):
            CRS(9999)

    def test_to_authority(self):
        from xrspatial.reproject._lite_crs import CRS
        crs = CRS(4326)
        assert crs.to_authority() == ('EPSG', '4326')

    def test_is_geographic_true(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(4326).is_geographic is True
        assert CRS(4269).is_geographic is True
        assert CRS(4267).is_geographic is True

    def test_is_geographic_false(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(3857).is_geographic is False
        assert CRS(32632).is_geographic is False
        assert CRS(5070).is_geographic is False


class TestCRSEquality:
    def test_equal_same_epsg(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(4326) == CRS(4326)

    def test_not_equal_different_epsg(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(4326) != CRS(3857)

    def test_hash_equal(self):
        from xrspatial.reproject._lite_crs import CRS
        assert hash(CRS(4326)) == hash(CRS(4326))

    def test_hash_in_set(self):
        from xrspatial.reproject._lite_crs import CRS
        s = {CRS(4326), CRS(4326), CRS(3857)}
        assert len(s) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_lite_crs.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement CRS class with EPSG table**

Create `xrspatial/reproject/_lite_crs.py` with:

```python
"""Lightweight CRS class for common EPSG codes.

Drop-in replacement for pyproj.CRS for the subset of EPSG codes that
have Numba JIT fast paths in _projections.py.  Allows basic reprojection
without installing pyproj.
"""
from __future__ import annotations

import math
import re

# ---- WGS84 ellipsoid constants (must match _projections.py) ----
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563

# ---- Ellipsoid definitions: (a, f) ----
_ELLIPSOID_WGS84 = (_WGS84_A, _WGS84_F)
_ELLIPSOID_CLARKE1866 = (6378206.4, 1.0 / 294.9786982)


# ---- Named EPSG entries ----
# Each value: dict with PROJ4-style keys + '_is_geographic' flag
# Only codes with Numba fast paths in _projections.py belong here.
_NAMED_EPSG = {
    # Geographic
    4326: {
        'proj': 'longlat', 'datum': 'WGS84', 'ellps': 'WGS84',
        'no_defs': True, '_is_geographic': True,
        '_name': 'WGS 84',
    },
    4269: {
        'proj': 'longlat', 'datum': 'NAD83', 'ellps': 'GRS80',
        'no_defs': True, '_is_geographic': True,
        '_name': 'NAD83',
    },
    4267: {
        'proj': 'longlat', 'datum': 'NAD27', 'ellps': 'clrk66',
        'no_defs': True, '_is_geographic': True,
        '_name': 'NAD27',
    },
    # Web Mercator
    3857: {
        'proj': 'merc', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': 0.0, 'lat_ts': 0.0, 'x_0': 0.0, 'y_0': 0.0,
        'k_0': 1.0, 'units': 'm', 'no_defs': True,
        '_is_geographic': False,
        '_name': 'WGS 84 / Pseudo-Mercator',
    },
    # Ellipsoidal Mercator
    3395: {
        'proj': 'merc', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': 0.0, 'lat_ts': 0.0, 'x_0': 0.0, 'y_0': 0.0,
        'k_0': 1.0, 'units': 'm', 'no_defs': True,
        '_is_geographic': False,
        '_name': 'WGS 84 / World Mercator',
    },
    # Lambert Conformal Conic -- France
    2154: {
        'proj': 'lcc', 'datum': 'WGS84', 'ellps': 'GRS80',
        'lon_0': 3.0, 'lat_0': 46.5, 'lat_1': 49.0, 'lat_2': 44.0,
        'x_0': 700000.0, 'y_0': 6600000.0, 'k_0': 1.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'RGF93 v1 / Lambert-93',
    },
    # Albers Equal Area -- CONUS
    5070: {
        'proj': 'aea', 'datum': 'NAD83', 'ellps': 'GRS80',
        'lon_0': -96.0, 'lat_0': 23.0, 'lat_1': 29.5, 'lat_2': 45.5,
        'x_0': 0.0, 'y_0': 0.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'NAD83 / Conus Albers',
    },
    # Lambert Azimuthal Equal Area -- Europe
    3035: {
        'proj': 'laea', 'datum': 'WGS84', 'ellps': 'GRS80',
        'lon_0': 10.0, 'lat_0': 52.0,
        'x_0': 4321000.0, 'y_0': 3210000.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'ETRS89-extended / LAEA Europe',
    },
    # Polar Stereographic South
    3031: {
        'proj': 'stere', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': 0.0, 'lat_0': -90.0, 'lat_ts': -71.0,
        'x_0': 0.0, 'y_0': 0.0, 'k_0': 1.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'WGS 84 / Antarctic Polar Stereographic',
    },
    # Polar Stereographic North (NSIDC)
    3413: {
        'proj': 'stere', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': -45.0, 'lat_0': 90.0, 'lat_ts': 70.0,
        'x_0': 0.0, 'y_0': 0.0, 'k_0': 1.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'WGS 84 / NSIDC Sea Ice Polar Stereographic North',
    },
    # Arctic Polar Stereographic
    3996: {
        'proj': 'stere', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': 0.0, 'lat_0': 90.0, 'lat_ts': 75.0,
        'x_0': 0.0, 'y_0': 0.0, 'k_0': 1.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'WGS 84 / Arctic Polar Stereographic',
    },
    # Oblique Stereographic -- Netherlands RD New
    28992: {
        'proj': 'sterea', 'datum': 'WGS84', 'ellps': 'bessel',
        'lon_0': 5.38763888889, 'lat_0': 52.15616055556,
        'x_0': 155000.0, 'y_0': 463000.0, 'k_0': 0.9999079,
        'units': 'm', 'no_defs': True, '_is_geographic': False,
        '_name': 'Amersfoort / RD New',
    },
    # Cylindrical Equal Area
    6933: {
        'proj': 'cea', 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': 0.0, 'lat_ts': 30.0,
        'x_0': 0.0, 'y_0': 0.0, 'k_0': 1.0, 'units': 'm',
        'no_defs': True, '_is_geographic': False,
        '_name': 'WGS 84 / NSIDC EASE-Grid 2.0 Global',
    },
}


def _utm_entry(epsg):
    """Generate PROJ4 dict for a UTM EPSG code, or return None."""
    if 32601 <= epsg <= 32660:
        zone = epsg - 32600
        south = False
    elif 32701 <= epsg <= 32760:
        zone = epsg - 32700
        south = True
    elif 26901 <= epsg <= 26923:
        zone = epsg - 26900
        south = False
        # NAD83 UTM
        lon0 = (zone - 1) * 6 - 177
        return {
            'proj': 'utm', 'zone': zone, 'datum': 'NAD83', 'ellps': 'GRS80',
            'lon_0': float(lon0), 'lat_0': 0.0,
            'x_0': 500000.0, 'y_0': 0.0,
            'k_0': 0.9996, 'units': 'm', 'no_defs': True,
            '_is_geographic': False,
            '_name': f'NAD83 / UTM zone {zone}N',
        }
    else:
        return None
    lon0 = (zone - 1) * 6 - 177
    fn = 10000000.0 if south else 0.0
    hemisphere = 'S' if south else 'N'
    return {
        'proj': 'utm', 'zone': zone, 'datum': 'WGS84', 'ellps': 'WGS84',
        'lon_0': float(lon0), 'lat_0': 0.0,
        'x_0': 500000.0, 'y_0': fn,
        'k_0': 0.9996, 'units': 'm', 'no_defs': True,
        '_is_geographic': False,
        '_name': f'WGS 84 / UTM zone {zone}{hemisphere}',
    }


def _lookup(epsg):
    """Look up an EPSG code in the built-in table.

    Returns the PROJ4-style dict or raises ValueError.
    """
    entry = _NAMED_EPSG.get(epsg)
    if entry is not None:
        return entry
    entry = _utm_entry(epsg)
    if entry is not None:
        return entry
    raise ValueError(
        f"EPSG:{epsg} is not in the built-in table. "
        f"Install pyproj for full CRS support:  "
        f"pip install pyproj  or:  pip install xarray-spatial[reproject]"
    )


# Regex for AUTHORITY["EPSG","XXXX"] in WKT strings
_WKT_EPSG_RE = re.compile(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]')


class CRS:
    """Lightweight CRS for common EPSG codes.

    Drop-in replacement for pyproj.CRS for the subset of codes that
    have Numba JIT fast paths.  Raises ValueError for codes not in
    the embedded table.
    """

    def __init__(self, input_value):
        if isinstance(input_value, int):
            self._epsg = input_value
            self._params = _lookup(input_value)
        elif isinstance(input_value, str):
            m = re.match(r'^EPSG:(\d+)$', input_value, re.IGNORECASE)
            if m:
                self._epsg = int(m.group(1))
                self._params = _lookup(self._epsg)
            else:
                # Try WKT
                epsg = self._extract_epsg_from_wkt(input_value)
                if epsg is not None:
                    self._epsg = epsg
                    self._params = _lookup(epsg)
                else:
                    raise ValueError(
                        f"Cannot parse CRS from string (no EPSG code found). "
                        f"Install pyproj for full CRS support."
                    )
        else:
            raise TypeError(
                f"CRS() expects int or str, got {type(input_value).__name__}"
            )

    @classmethod
    def from_epsg(cls, code):
        """Construct from an integer EPSG code."""
        return cls(int(code))

    @classmethod
    def from_wkt(cls, wkt):
        """Construct from a WKT string by extracting the EPSG code."""
        epsg = cls._extract_epsg_from_wkt(wkt)
        if epsg is None:
            raise ValueError(
                "No AUTHORITY[\"EPSG\",\"...\"] found in WKT string. "
                "Install pyproj for full CRS support."
            )
        return cls(epsg)

    @staticmethod
    def _extract_epsg_from_wkt(wkt):
        """Extract the last EPSG code from a WKT string, or None."""
        matches = _WKT_EPSG_RE.findall(wkt)
        if matches:
            return int(matches[-1])
        return None

    def to_epsg(self):
        """Return the integer EPSG code."""
        return self._epsg

    def to_authority(self):
        """Return (authority_name, code) tuple."""
        return ('EPSG', str(self._epsg))

    def to_dict(self):
        """Return PROJ4-style parameter dict.

        Internal keys prefixed with '_' are stripped.
        """
        return {k: v for k, v in self._params.items()
                if not k.startswith('_')}

    @property
    def is_geographic(self):
        """True if this is a geographic (lat/lon) CRS."""
        return self._params['_is_geographic']

    def to_wkt(self):
        """Return an OGC WKT1 string for this CRS."""
        return _build_wkt(self._epsg, self._params)

    def __eq__(self, other):
        if isinstance(other, CRS):
            return self._epsg == other._epsg
        # Duck-type: try to_epsg() on the other object (e.g. pyproj.CRS)
        try:
            return self._epsg == other.to_epsg()
        except (AttributeError, TypeError):
            return NotImplemented

    def __hash__(self):
        return hash(('CRS', self._epsg))

    def __repr__(self):
        name = self._params.get('_name', '')
        if name:
            return f"CRS(EPSG:{self._epsg} -- {name})"
        return f"CRS(EPSG:{self._epsg})"


def _build_wkt(epsg, params):
    """Build a minimal OGC WKT1 string from EPSG + params.

    This does not need to be a perfect WKT -- it just needs to contain
    AUTHORITY["EPSG","XXXX"] so that from_wkt() can round-trip, and
    be accepted by pyproj.CRS.from_wkt() if the user mixes types.
    """
    name = params.get('_name', f'EPSG:{epsg}')
    if params['_is_geographic']:
        return (
            f'GEOGCS["{name}",'
            f'DATUM["unknown",SPHEROID["WGS 84",{_WGS84_A},{1/_WGS84_F}]],'
            f'PRIMEM["Greenwich",0],'
            f'UNIT["degree",0.0174532925199433],'
            f'AUTHORITY["EPSG","{epsg}"]]'
        )
    proj = params.get('proj', '')
    fe = params.get('x_0', 0.0)
    fn = params.get('y_0', 0.0)
    lon0 = params.get('lon_0', 0.0)
    lat0 = params.get('lat_0', 0.0)
    k0 = params.get('k_0', 1.0)
    lat1 = params.get('lat_1', 0.0)
    lat2 = params.get('lat_2', 0.0)
    return (
        f'PROJCS["{name}",'
        f'GEOGCS["GCS_WGS_1984",'
        f'DATUM["D_WGS_1984",SPHEROID["WGS_1984",{_WGS84_A},{1/_WGS84_F}]],'
        f'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
        f'PROJECTION["{proj}"],'
        f'PARAMETER["false_easting",{fe}],'
        f'PARAMETER["false_northing",{fn}],'
        f'PARAMETER["central_meridian",{lon0}],'
        f'PARAMETER["latitude_of_origin",{lat0}],'
        f'PARAMETER["scale_factor",{k0}],'
        f'PARAMETER["standard_parallel_1",{lat1}],'
        f'PARAMETER["standard_parallel_2",{lat2}],'
        f'UNIT["Meter",1],'
        f'AUTHORITY["EPSG","{epsg}"]]'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_lite_crs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/reproject/_lite_crs.py xrspatial/tests/test_lite_crs.py
git commit -m "Add lightweight CRS class with embedded EPSG table (#1057)"
```

---

### Task 2: CRS to_dict() and to_wkt() correctness

**Files:**
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write failing tests for to_dict() and to_wkt() round-trip**

Add to `test_lite_crs.py`:

```python
class TestCRSToDict:
    """Verify to_dict() returns correct PROJ4 keys for each projection family."""

    def test_geographic_4326(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(4326).to_dict()
        assert d['proj'] == 'longlat'
        assert d['datum'] == 'WGS84'
        # No internal keys leaked
        assert not any(k.startswith('_') for k in d)

    def test_utm_north_zone_32(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(32632).to_dict()
        assert d['proj'] == 'utm'
        assert d['zone'] == 32
        assert d['k_0'] == 0.9996
        assert d['x_0'] == 500000.0
        assert d['y_0'] == 0.0  # north hemisphere
        assert d['lon_0'] == 9.0  # (32-1)*6 - 177

    def test_utm_south_zone_55(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(32755).to_dict()
        assert d['zone'] == 55
        assert d['y_0'] == 10000000.0  # south hemisphere

    def test_utm_nad83_zone_10(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(26910).to_dict()
        assert d['datum'] == 'NAD83'
        assert d['zone'] == 10
        assert d['lon_0'] == -123.0

    def test_lcc_2154(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(2154).to_dict()
        assert d['proj'] == 'lcc'
        assert d['lat_1'] == 49.0
        assert d['lat_2'] == 44.0

    def test_aea_5070(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(5070).to_dict()
        assert d['proj'] == 'aea'
        assert d['lon_0'] == -96.0

    def test_web_mercator_3857(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(3857).to_dict()
        assert d['proj'] == 'merc'

    def test_laea_3035(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(3035).to_dict()
        assert d['proj'] == 'laea'
        assert d['lon_0'] == 10.0
        assert d['lat_0'] == 52.0

    def test_stere_3031(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(3031).to_dict()
        assert d['proj'] == 'stere'
        assert d['lat_0'] == -90.0

    def test_sterea_28992(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(28992).to_dict()
        assert d['proj'] == 'sterea'

    def test_cea_6933(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(6933).to_dict()
        assert d['proj'] == 'cea'


class TestCRSWktRoundTrip:
    def test_roundtrip_geographic(self):
        from xrspatial.reproject._lite_crs import CRS
        crs1 = CRS(4326)
        wkt = crs1.to_wkt()
        crs2 = CRS.from_wkt(wkt)
        assert crs2.to_epsg() == 4326

    def test_roundtrip_projected(self):
        from xrspatial.reproject._lite_crs import CRS
        crs1 = CRS(32632)
        wkt = crs1.to_wkt()
        crs2 = CRS.from_wkt(wkt)
        assert crs2.to_epsg() == 32632

    def test_roundtrip_all_named(self):
        from xrspatial.reproject._lite_crs import CRS, _NAMED_EPSG
        for epsg in _NAMED_EPSG:
            crs = CRS(epsg)
            wkt = crs.to_wkt()
            assert CRS.from_wkt(wkt).to_epsg() == epsg, f"Failed for EPSG:{epsg}"

    def test_wkt_contains_authority(self):
        from xrspatial.reproject._lite_crs import CRS
        wkt = CRS(5070).to_wkt()
        assert 'AUTHORITY["EPSG","5070"]' in wkt
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_lite_crs.py -v`
Expected: All PASS (implementation was written in Task 1)

- [ ] **Step 3: Verify against pyproj (if installed)**

Add a parametrized test that compares our `.to_dict()` against pyproj's output for every embedded code. This test skips when pyproj is not installed:

```python
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestCRSMatchesPyproj:
    """Compare our CRS output against pyproj for correctness."""

    @pytest.mark.parametrize("epsg", [
        4326, 4269, 4267, 3857, 3395, 2154, 5070, 3035,
        3031, 3413, 3996, 28992, 6933,
        32601, 32632, 32660, 32701, 32755, 32760,
        26901, 26910, 26923,
    ])
    def test_to_dict_proj_key_matches(self, epsg):
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        lite_d = LiteCRS(epsg).to_dict()
        pyproj_d = pyproj.CRS.from_epsg(epsg).to_dict()
        # The 'proj' key must match for dispatch to work
        assert lite_d['proj'] == pyproj_d.get('proj', pyproj_d.get('type', '')), \
            f"EPSG:{epsg}: proj mismatch: {lite_d['proj']} vs {pyproj_d}"

    @pytest.mark.parametrize("epsg", [
        4326, 4269, 4267, 3857, 3395, 2154, 5070, 3035,
        3031, 3413, 3996, 28992, 6933,
        32632, 32755, 26910,
    ])
    def test_is_geographic_matches(self, epsg):
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        assert LiteCRS(epsg).is_geographic == pyproj.CRS.from_epsg(epsg).is_geographic

    @pytest.mark.parametrize("epsg", [
        4326, 3857, 32632, 5070, 3035, 3031, 28992, 6933,
    ])
    def test_equality_with_pyproj(self, epsg):
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        lite = LiteCRS(epsg)
        pp = pyproj.CRS.from_epsg(epsg)
        assert lite == pp  # our __eq__ calls pp.to_epsg()
```

- [ ] **Step 4: Run full test suite**

Run: `pytest xrspatial/tests/test_lite_crs.py -v`
Expected: All PASS (pyproj tests pass or skip)

- [ ] **Step 5: Commit**

```bash
git add xrspatial/tests/test_lite_crs.py
git commit -m "Add to_dict/to_wkt correctness tests for lite CRS (#1057)"
```

---

### Task 3: Two-tier `_resolve_crs` and `_crs_from_wkt`

**Files:**
- Modify: `xrspatial/reproject/_crs_utils.py`
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write failing tests for two-tier resolution**

Add to `test_lite_crs.py`:

```python
class TestTwoTierResolution:
    def test_resolve_crs_int_uses_lite(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        crs = _resolve_crs(4326)
        assert isinstance(crs, LiteCRS)
        assert crs.to_epsg() == 4326

    def test_resolve_crs_string_uses_lite(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        crs = _resolve_crs("EPSG:32632")
        assert isinstance(crs, LiteCRS)
        assert crs.to_epsg() == 32632

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
    def test_resolve_crs_unknown_falls_back(self):
        """Unknown EPSG codes should fall back to pyproj."""
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        crs = _resolve_crs(2193)  # NZGD2000 -- not in our table
        assert not isinstance(crs, LiteCRS)
        assert crs.to_epsg() == 2193

    def test_resolve_crs_none_returns_none(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        assert _resolve_crs(None) is None

    def test_resolve_crs_passes_through_lite_crs(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        original = LiteCRS(4326)
        result = _resolve_crs(original)
        assert result is original

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
    def test_resolve_crs_passes_through_pyproj(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        original = pyproj.CRS.from_epsg(4326)
        result = _resolve_crs(original)
        assert result is original

    def test_crs_from_wkt_lite(self):
        from xrspatial.reproject._crs_utils import _crs_from_wkt
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        wkt = LiteCRS(4326).to_wkt()
        crs = _crs_from_wkt(wkt)
        assert isinstance(crs, LiteCRS)
        assert crs.to_epsg() == 4326
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestTwoTierResolution -v`
Expected: FAIL -- `_resolve_crs` still requires pyproj, `_crs_from_wkt` doesn't exist

- [ ] **Step 3: Implement two-tier `_resolve_crs` and `_crs_from_wkt`**

Replace `xrspatial/reproject/_crs_utils.py` contents:

```python
"""CRS detection utilities with lightweight fallback."""
from __future__ import annotations


def _require_pyproj():
    """Import and return the pyproj module, raising a clear error if missing."""
    try:
        import pyproj
        return pyproj
    except ImportError:
        raise ImportError(
            "pyproj is required for CRS reprojection. "
            "Install it with:  pip install pyproj  "
            "or:  pip install xarray-spatial[reproject]"
        )


def _resolve_crs(crs_input):
    """Convert *crs_input* to a CRS object.

    Tries the lightweight built-in CRS first for known EPSG codes,
    then falls back to pyproj.CRS for anything else.

    Returns None if *crs_input* is None.
    """
    if crs_input is None:
        return None

    # Pass through existing CRS objects (ours or pyproj)
    from ._lite_crs import CRS as LiteCRS
    if isinstance(crs_input, LiteCRS):
        return crs_input
    try:
        pyproj = _try_import_pyproj()
        if pyproj is not None and isinstance(crs_input, pyproj.CRS):
            return crs_input
    except Exception:
        pass

    # Try lightweight CRS first
    try:
        return LiteCRS(crs_input)
    except (ValueError, TypeError):
        pass

    # Fall back to pyproj
    pyproj = _require_pyproj()
    return pyproj.CRS(crs_input)


def _crs_from_wkt(wkt):
    """Reconstruct a CRS from a WKT string.

    Tries the lightweight CRS (extracts EPSG from AUTHORITY tag),
    falls back to pyproj.CRS.from_wkt().
    """
    from ._lite_crs import CRS as LiteCRS
    try:
        return LiteCRS.from_wkt(wkt)
    except (ValueError, TypeError):
        pass
    pyproj = _require_pyproj()
    return pyproj.CRS.from_wkt(wkt)


def _try_import_pyproj():
    """Try to import pyproj, return None if not installed."""
    try:
        import pyproj
        return pyproj
    except ImportError:
        return None


def _detect_source_crs(raster):
    """Auto-detect the CRS of a DataArray.

    Fallback chain:
    1. ``raster.attrs['crs']`` (EPSG int from xrspatial.geotiff)
    2. ``raster.attrs['crs_wkt']`` (WKT string from xrspatial.geotiff)
    3. ``raster.rio.crs`` (rioxarray, if installed)
    4. None
    """
    # attrs (xrspatial.geotiff convention)
    crs_attr = raster.attrs.get('crs')
    if crs_attr is not None:
        return _resolve_crs(crs_attr)

    crs_wkt = raster.attrs.get('crs_wkt')
    if crs_wkt is not None:
        return _resolve_crs(crs_wkt)

    # rioxarray fallback
    try:
        rio_crs = raster.rio.crs
        if rio_crs is not None:
            return _resolve_crs(rio_crs)
    except Exception:
        pass

    return None


def _detect_nodata(raster, nodata=None):
    """Determine nodata value from explicit arg, rioxarray, or attrs."""
    if nodata is not None:
        return float(nodata)

    # rioxarray
    try:
        rio_nd = raster.rio.nodata
        if rio_nd is not None:
            return float(rio_nd)
    except Exception:
        pass

    # attrs
    for key in ('_FillValue', 'nodata', 'missing_value'):
        val = raster.attrs.get(key)
        if val is not None:
            return float(val)

    return float('nan')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_lite_crs.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing reproject tests to verify no regressions**

Run: `pytest xrspatial/tests/test_reproject.py -v -x`
Expected: All PASS (existing tests still work because pyproj is installed)

- [ ] **Step 6: Commit**

```bash
git add xrspatial/reproject/_crs_utils.py xrspatial/tests/test_lite_crs.py
git commit -m "Two-tier CRS resolution: lite CRS first, pyproj fallback (#1057)"
```

---

### Task 4: Scatter-point transform helper in `_projections.py`

**Files:**
- Modify: `xrspatial/reproject/_projections.py` (add `transform_points()` near end of file)
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write failing tests for `transform_points`**

Add to `test_lite_crs.py`:

```python
@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestTransformPoints:
    """Test the Numba scatter-point transform helper."""

    def test_wgs84_to_web_mercator(self):
        from xrspatial.reproject._projections import transform_points
        from xrspatial.reproject._lite_crs import CRS
        import numpy as np
        src = CRS(4326)
        tgt = CRS(3857)
        xs = np.array([0.0, 10.0, -10.0])
        ys = np.array([0.0, 45.0, -30.0])
        tx, ty = transform_points(src, tgt, xs, ys)
        assert tx is not None
        # lon=0 -> x=0 in Web Mercator
        assert abs(tx[0]) < 1.0

    def test_wgs84_to_utm_zone32(self):
        from xrspatial.reproject._projections import transform_points
        from xrspatial.reproject._lite_crs import CRS
        import numpy as np
        src = CRS(4326)
        tgt = CRS(32632)
        xs = np.array([9.0])  # central meridian of zone 32
        ys = np.array([48.0])
        tx, ty = transform_points(src, tgt, xs, ys)
        assert tx is not None
        # Central meridian -> false easting = 500000
        assert abs(tx[0] - 500000.0) < 1.0

    def test_unsupported_pair_returns_none(self):
        from xrspatial.reproject._projections import transform_points
        import pyproj
        import numpy as np
        # Two projected CRS -- no fast path
        src = pyproj.CRS.from_epsg(32632)
        tgt = pyproj.CRS.from_epsg(32633)
        xs = np.array([500000.0])
        ys = np.array([5000000.0])
        result = transform_points(src, tgt, xs, ys)
        assert result is None

    def test_matches_pyproj_transformer(self):
        """Numba scatter-point transform matches pyproj within tolerance."""
        from xrspatial.reproject._projections import transform_points
        from xrspatial.reproject._lite_crs import CRS
        import numpy as np
        src = CRS(4326)
        tgt = CRS(5070)  # Albers
        xs = np.linspace(-100, -80, 20)
        ys = np.linspace(30, 45, 20)
        tx, ty = transform_points(src, tgt, xs, ys)
        assert tx is not None
        # Compare with pyproj
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), pyproj.CRS.from_epsg(5070),
            always_xy=True
        )
        px, py = transformer.transform(xs, ys)
        np.testing.assert_allclose(tx, px, atol=0.01)
        np.testing.assert_allclose(ty, py, atol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestTransformPoints -v`
Expected: FAIL -- `transform_points` does not exist

- [ ] **Step 3: Implement `transform_points()` in `_projections.py`**

Add near the end of `xrspatial/reproject/_projections.py`, before the `_try_numba_transform_inner` line:

```python
def transform_points(src_crs, tgt_crs, xs, ys):
    """Transform scatter points between CRS using Numba kernels.

    Unlike try_numba_transform() which operates on a regular grid, this
    accepts arbitrary (xs, ys) arrays.  Used by _grid.py for boundary
    estimation.

    Returns (tx, ty) numpy arrays, or None if no fast path exists.
    """
    src_epsg = _get_epsg(src_crs)
    tgt_epsg = _get_epsg(tgt_crs)
    if src_epsg is None and tgt_epsg is None:
        return None

    src_is_geo = _is_supported_geographic(src_epsg)
    tgt_is_geo = _is_supported_geographic(tgt_epsg)
    if not src_is_geo and not tgt_is_geo:
        return None

    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    n = xs.shape[0]
    tx = np.empty(n, dtype=np.float64)
    ty = np.empty(n, dtype=np.float64)

    # Geographic -> Web Mercator
    if src_is_geo and tgt_epsg == 3857:
        merc_forward(xs, ys, tx, ty)
        return tx, ty
    if src_epsg == 3857 and tgt_is_geo:
        merc_inverse(xs, ys, tx, ty)
        return tx, ty

    # Geographic -> UTM
    if src_is_geo:
        utm = _utm_params(tgt_epsg)
        if utm is not None:
            lon0, k0, fe, fn = utm
            Qn = k0 * _A_RECT
            tmerc_forward(xs, ys, tx, ty, lon0, k0, fe, fn, Qn, _ALPHA, _CBG)
            return tx, ty
    utm_src = _utm_params(src_epsg)
    if utm_src is not None and tgt_is_geo:
        lon0, k0, fe, fn = utm_src
        Qn = k0 * _A_RECT
        tmerc_inverse(xs, ys, tx, ty, lon0, k0, fe, fn, Qn, _BETA, _CGB)
        return tx, ty

    # Ellipsoidal Mercator
    if src_is_geo and tgt_epsg == 3395:
        emerc_forward(xs, ys, tx, ty, 1.0, _WGS84_E)
        return tx, ty
    if src_epsg == 3395 and tgt_is_geo:
        emerc_inverse(xs, ys, tx, ty, 1.0, _WGS84_E)
        return tx, ty

    # LCC
    if src_is_geo:
        params = _lcc_params(tgt_crs)
        if params is not None:
            lon0, nn, c, rho0, k0, fe, fn, to_m = params
            lcc_forward(xs, ys, tx, ty, lon0, nn, c, rho0, k0, fe, fn,
                        _WGS84_E, _WGS84_A)
            return tx, ty
    if tgt_is_geo:
        params = _lcc_params(src_crs)
        if params is not None:
            lon0, nn, c, rho0, k0, fe, fn, to_m = params
            # lcc_inverse expects metres; convert from native units first
            in_xs = xs * to_m if to_m != 1.0 else xs
            in_ys = ys * to_m if to_m != 1.0 else ys
            lcc_inverse(in_xs, in_ys, tx, ty, lon0, nn, c, rho0, k0, fe, fn,
                        _WGS84_E, _WGS84_A)
            return tx, ty

    # AEA
    if src_is_geo:
        params = _aea_params(tgt_crs)
        if params is not None:
            lon0, nn, C, rho0, fe, fn = params
            aea_forward(xs, ys, tx, ty, lon0, nn, C, rho0, fe, fn,
                        _WGS84_E, _WGS84_A)
            return tx, ty
    if tgt_is_geo:
        params = _aea_params(src_crs)
        if params is not None:
            lon0, nn, C, rho0, fe, fn = params
            aea_inverse(xs, ys, tx, ty, lon0, nn, C, rho0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP, _APA)
            return tx, ty

    # CEA
    if src_is_geo:
        params = _cea_params(tgt_crs)
        if params is not None:
            lon0, k0, fe, fn = params
            cea_forward(xs, ys, tx, ty, lon0, k0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP)
            return tx, ty
    if tgt_is_geo:
        params = _cea_params(src_crs)
        if params is not None:
            lon0, k0, fe, fn = params
            cea_inverse(xs, ys, tx, ty, lon0, k0, fe, fn,
                        _WGS84_E, _WGS84_A, _QP, _APA)
            return tx, ty

    # LAEA
    if src_is_geo:
        params = _laea_params(tgt_crs)
        if params is not None:
            lon0, lat0, sinb1, cosb1, dd, xmf, ymf, rq, qp, fe, fn, mode = params
            laea_forward(xs, ys, tx, ty, lon0, sinb1, cosb1, xmf, ymf,
                         rq, qp, fe, fn, _WGS84_E, _WGS84_A, _WGS84_E2, mode)
            return tx, ty
    if tgt_is_geo:
        params = _laea_params(src_crs)
        if params is not None:
            lon0, lat0, sinb1, cosb1, dd, xmf, ymf, rq, qp, fe, fn, mode = params
            laea_inverse(xs, ys, tx, ty, lon0, sinb1, cosb1, xmf, ymf,
                         rq, qp, fe, fn, _WGS84_E, _WGS84_A, _WGS84_E2, mode, _APA)
            return tx, ty

    # Polar Stereographic
    if src_is_geo:
        params = _stere_params(tgt_crs)
        if params is not None:
            lon0, k0, akm1, fe, fn, is_south = params
            stere_forward(xs, ys, tx, ty, lon0, akm1, fe, fn, _WGS84_E, is_south)
            return tx, ty
    if tgt_is_geo:
        params = _stere_params(src_crs)
        if params is not None:
            lon0, k0, akm1, fe, fn, is_south = params
            stere_inverse(xs, ys, tx, ty, lon0, akm1, fe, fn, _WGS84_E, is_south)
            return tx, ty

    # Oblique Stereographic
    if src_is_geo:
        params = _sterea_params(tgt_crs)
        if params is not None:
            lon0, sinc0, cosc0, R2, C, K, ratexp, fe, fn, e = params
            sterea_forward(xs, ys, tx, ty, lon0, sinc0, cosc0, R2, C, K,
                           ratexp, fe, fn, e)
            return tx, ty
    if tgt_is_geo:
        params = _sterea_params(src_crs)
        if params is not None:
            lon0, sinc0, cosc0, R2, C, K, ratexp, fe, fn, e = params
            sterea_inverse(xs, ys, tx, ty, lon0, sinc0, cosc0, R2, C, K,
                           ratexp, fe, fn, e)
            return tx, ty

    return None
```

Note: This function mirrors the dispatch logic of `try_numba_transform` but operates on flat scatter-point arrays instead of building a regular grid. The existing batch Numba kernels (e.g. `merc_forward`, `tmerc_forward`, etc.) already accept 1-D arrays, so no new Numba code is needed.

**Intentional omissions from `try_numba_transform`:**
- **Datum shift wrapping** -- `try_numba_transform` wraps projection kernels with Helmert datum shifts for NAD27 etc. For boundary estimation (the primary consumer of `transform_points`), the metre-level error from omitting the datum shift is sub-pixel and does not affect bounding box correctness. A comment in the code should document this.
- **Sinusoidal and Generic Transverse Mercator** -- these dispatch via `.to_dict()['proj']` not EPSG code, so they only fire when pyproj provides the CRS. Omitting them from `transform_points` means those CRS pairs fall through to the pyproj Transformer fallback in `_transform_boundary`, which is correct behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestTransformPoints -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/reproject/_projections.py xrspatial/tests/test_lite_crs.py
git commit -m "Add scatter-point transform_points() for boundary estimation (#1057)"
```

---

### Task 5: Wire `_grid.py` to use `transform_points` with pyproj fallback

**Files:**
- Modify: `xrspatial/reproject/_grid.py`
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write failing test for pyproj-free grid computation**

Add to `test_lite_crs.py`:

```python
class TestGridWithoutPyproj:
    def test_compute_output_grid_with_lite_crs(self):
        """_compute_output_grid works with lite CRS objects (no pyproj)."""
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._lite_crs import CRS
        src_crs = CRS(4326)
        tgt_crs = CRS(32632)
        source_bounds = (6.0, 47.0, 12.0, 55.0)  # Germany
        source_shape = (64, 64)
        grid = _compute_output_grid(
            source_bounds, source_shape, src_crs, tgt_crs
        )
        assert 'bounds' in grid
        assert 'shape' in grid
        h, w = grid['shape']
        assert h > 0 and w > 0
        left, bottom, right, top = grid['bounds']
        assert right > left
        assert top > bottom
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestGridWithoutPyproj -v`
Expected: FAIL -- `_grid.py` still calls `_require_pyproj()`

- [ ] **Step 3: Modify `_grid.py` to use `transform_points` with pyproj fallback**

Replace the transformer creation and transform calls in `_compute_output_grid()`:

```python
"""Output grid computation and chunk layout for reprojection."""
from __future__ import annotations

import numpy as np


def _transform_boundary(source_crs, target_crs, xs, ys):
    """Transform scatter points, trying Numba fast path first."""
    try:
        from ._projections import transform_points
        result = transform_points(source_crs, target_crs, xs, ys)
        if result is not None:
            return result
    except (ImportError, ModuleNotFoundError):
        pass
    # Fall back to pyproj
    from ._crs_utils import _require_pyproj
    pyproj = _require_pyproj()
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    tx, ty = transformer.transform(xs, ys)
    return np.asarray(tx), np.asarray(ty)


def _compute_output_grid(source_bounds, source_shape, source_crs, target_crs,
                         resolution=None, bounds=None, width=None, height=None):
    # ... (see step 3 body below)
```

The body of `_compute_output_grid` replaces all `transformer.transform(...)` calls with `_transform_boundary(source_crs, target_crs, ...)`. Remove the `_require_pyproj()` and `Transformer.from_crs()` calls at the top. The `source_crs.is_geographic` check stays as-is (works with both CRS types). Three call sites to replace:

1. Line 79: `tx, ty = transformer.transform(xs, ys)` -> `tx, ty = _transform_boundary(source_crs, target_crs, xs, ys)`
2. Line 113: `itx, ity = transformer.transform(ixx.ravel(), iyy.ravel())` -> `itx, ity = _transform_boundary(source_crs, target_crs, ixx.ravel(), iyy.ravel())`
3. Lines 153-158: resolution estimation (3 transform calls) -> batch into one `_transform_boundary` call:
   ```python
   pts_x = np.array([center_x, center_x + src_res_x, center_x])
   pts_y = np.array([center_y, center_y, center_y + src_res_y])
   tp_x, tp_y = _transform_boundary(source_crs, target_crs, pts_x, pts_y)
   tc_x, tc_y = float(tp_x[0]), float(tp_y[0])
   tx_x, tx_y = float(tp_x[1]), float(tp_y[1])
   ty_x, ty_y = float(tp_x[2]), float(tp_y[2])
   ```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestGridWithoutPyproj -v`
Expected: PASS

- [ ] **Step 5: Run existing reproject tests to verify no regressions**

Run: `pytest xrspatial/tests/test_reproject.py -v -x`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add xrspatial/reproject/_grid.py xrspatial/tests/test_lite_crs.py
git commit -m "Use Numba scatter-point transform for grid boundary estimation (#1057)"
```

---

### Task 6: Update chunk functions and `_source_footprint_in_target`

**Files:**
- Modify: `xrspatial/reproject/__init__.py`
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write failing integration test**

Add to `test_lite_crs.py`:

```python
class TestReprojWithLiteCRS:
    def test_reproject_wgs84_to_utm_with_lite_crs(self):
        """Full reproject() works when _resolve_crs returns a lite CRS."""
        import xarray as xr
        from xrspatial.reproject import reproject
        from xrspatial.reproject._lite_crs import CRS as LiteCRS
        import numpy as np
        # Small test raster in WGS84
        h, w = 32, 32
        y = np.linspace(49, 47, h)
        x = np.linspace(8, 10, w)
        data = np.random.default_rng(42).random((h, w))
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        result = reproject(raster, target_crs=32632)
        assert result.attrs['crs'] is not None
        assert result.shape[0] > 0 and result.shape[1] > 0
```

- [ ] **Step 2: Modify `_reproject_chunk_numpy` to use `_crs_from_wkt`**

In `xrspatial/reproject/__init__.py`, change lines 195-199:

Before:
```python
    from ._crs_utils import _require_pyproj
    pyproj = _require_pyproj()
    src_crs = pyproj.CRS.from_wkt(src_wkt)
    tgt_crs = pyproj.CRS.from_wkt(tgt_wkt)
```

After:
```python
    from ._crs_utils import _crs_from_wkt
    src_crs = _crs_from_wkt(src_wkt)
    tgt_crs = _crs_from_wkt(tgt_wkt)
```

Also update the fallback Transformer creation (lines 214-217) to conditionally import pyproj only when needed:

```python
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(
            tgt_crs, src_crs, always_xy=True
        )
```

- [ ] **Step 3: Restructure `_reproject_chunk_cupy` to defer Transformer**

In `xrspatial/reproject/__init__.py`, change lines 324-332:

Before:
```python
    from ._crs_utils import _require_pyproj
    pyproj = _require_pyproj()
    src_crs = pyproj.CRS.from_wkt(src_wkt)
    tgt_crs = pyproj.CRS.from_wkt(tgt_wkt)
    transformer = pyproj.Transformer.from_crs(
        tgt_crs, src_crs, always_xy=True
    )
```

After:
```python
    from ._crs_utils import _crs_from_wkt
    src_crs = _crs_from_wkt(src_wkt)
    tgt_crs = _crs_from_wkt(tgt_wkt)
```

Move the Transformer creation into the `else` block (line 372, CPU fallback):

```python
    else:
        # CPU fallback (Numba JIT or pyproj)
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(
            tgt_crs, src_crs, always_xy=True
        )
        src_y, src_x = _transform_coords(
            transformer, chunk_bounds_tuple, chunk_shape, transform_precision,
            src_crs=src_crs, tgt_crs=tgt_crs,
        )
```

- [ ] **Step 4: Update `_source_footprint_in_target`**

In `xrspatial/reproject/__init__.py`, change the function at line 1122:

```python
def _source_footprint_in_target(src_bounds, src_wkt, tgt_wkt):
    """Compute approximate bounding box of source raster in target CRS."""
    try:
        from ._crs_utils import _crs_from_wkt
        src_crs = _crs_from_wkt(src_wkt)
        tgt_crs = _crs_from_wkt(tgt_wkt)
    except Exception:
        return None

    sl, sb, sr, st = src_bounds
    mx = (sl + sr) / 2
    my = (sb + st) / 2
    xs = np.array([sl, mx, sr, sl, mx, sr, sl, mx, sr, sl, sr, mx])
    ys = np.array([sb, sb, sb, my, my, my, st, st, st, mx, mx, sb])

    try:
        # Try Numba scatter-point transform first
        from ._projections import transform_points
        result = transform_points(src_crs, tgt_crs, xs, ys)
        if result is not None:
            tx, ty = result
            tx = [v for v in tx if np.isfinite(v)]
            ty = [v for v in ty if np.isfinite(v)]
            if not tx or not ty:
                return None
            return (min(tx), min(ty), max(tx), max(ty))
    except (ImportError, ModuleNotFoundError):
        pass

    # Fall back to pyproj Transformer
    try:
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(
            src_crs, tgt_crs, always_xy=True
        )
        tx, ty = transformer.transform(xs.tolist(), ys.tolist())
        tx = [v for v in tx if np.isfinite(v)]
        ty = [v for v in ty if np.isfinite(v)]
        if not tx or not ty:
            return None
        return (min(tx), min(ty), max(tx), max(ty))
    except Exception:
        return None
```

- [ ] **Step 5: Remove unconditional `_require_pyproj()` guards in `reproject()` and `merge()`**

In `xrspatial/reproject/__init__.py`:

1. **`reproject()` at line 525**: Remove the `_require_pyproj()` call. The `_resolve_crs()` call on line 528 handles importing pyproj only if needed.

2. **`merge()` at line 1308**: Same pattern -- remove the unconditional `_require_pyproj()` call. The chunk functions and `_source_footprint_in_target` already handle their own fallback to pyproj when needed.

Also remove the now-unused `from ._crs_utils import _require_pyproj` import at lines 516 and 1301 (these functions no longer call `_require_pyproj` directly).

- [ ] **Step 6: Run tests**

Run: `pytest xrspatial/tests/test_lite_crs.py xrspatial/tests/test_reproject.py -v -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add xrspatial/reproject/__init__.py xrspatial/tests/test_lite_crs.py
git commit -m "Wire chunk functions and merge helper to use lite CRS (#1057)"
```

---

### Task 7: Integration test -- reprojection without pyproj

**Files:**
- Modify: `xrspatial/tests/test_lite_crs.py`

- [ ] **Step 1: Write integration test that mocks pyproj as unavailable**

```python
class TestNoPyproj:
    """Verify reprojection works for supported CRS pairs when pyproj is absent."""

    def test_reproject_without_pyproj(self, monkeypatch):
        """Reprojection should work without pyproj for known EPSG codes."""
        import sys
        import xarray as xr
        from xrspatial.reproject._lite_crs import CRS
        import numpy as np

        # Block pyproj import
        monkeypatch.setitem(sys.modules, 'pyproj', None)

        # Force fresh imports to pick up blocked pyproj
        from xrspatial.reproject._crs_utils import _resolve_crs, _crs_from_wkt

        src_crs = CRS(4326)
        tgt_crs = CRS(3857)
        assert src_crs.to_epsg() == 4326
        assert tgt_crs.is_geographic is False

        # _crs_from_wkt round-trips without pyproj
        wkt = src_crs.to_wkt()
        restored = _crs_from_wkt(wkt)
        assert restored.to_epsg() == 4326

    def test_unknown_epsg_without_pyproj_raises(self, monkeypatch):
        """Unknown EPSG codes raise clear error when pyproj is absent."""
        import sys
        monkeypatch.setitem(sys.modules, 'pyproj', None)
        from xrspatial.reproject._crs_utils import _resolve_crs
        with pytest.raises((ImportError, ValueError)):
            _resolve_crs(2193)
```

- [ ] **Step 2: Run tests**

Run: `pytest xrspatial/tests/test_lite_crs.py::TestNoPyproj -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add xrspatial/tests/test_lite_crs.py
git commit -m "Add integration tests for pyproj-free CRS resolution (#1057)"
```

---

### Task 8: Final validation and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest xrspatial/tests/test_lite_crs.py xrspatial/tests/test_reproject.py -v`
Expected: All PASS

- [ ] **Step 2: Verify no other pyproj imports leaked**

Check that the chunk functions and grid code only import pyproj via `_require_pyproj()` or `_crs_from_wkt()`, not directly:

Run: `grep -n "import pyproj" xrspatial/reproject/__init__.py xrspatial/reproject/_grid.py`
Expected: No direct `import pyproj` statements in these files

- [ ] **Step 3: Commit any final cleanups**

```bash
git add -u
git commit -m "Clean up pyproj imports in reproject module (#1057)"
```
