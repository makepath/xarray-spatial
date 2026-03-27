"""Lightweight CRS class with an embedded EPSG lookup table.

Provides a drop-in subset of ``pyproj.CRS`` for the most common EPSG
codes so that reprojection can work without pyproj installed.
"""
from __future__ import annotations

import re
from typing import Tuple

# -------------------------------------------------------------------
# Ellipsoid definitions: (a, f)
# -------------------------------------------------------------------
_ELLIPSOIDS = {
    "WGS84": (6378137.0, 1.0 / 298.257223563),
    "GRS80": (6378137.0, 1.0 / 298.257222101),
    "clrk66": (6378206.4, 1.0 / 294.9786982),
    "bessel": (6377397.155, 1.0 / 299.1528128),
}

# -------------------------------------------------------------------
# Named EPSG table  (internal keys prefixed with _ are stripped from
# to_dict output)
# -------------------------------------------------------------------
_EPSG_TABLE: dict[int, dict] = {
    # Geographic ---------------------------------------------------
    4326: {
        "proj": "longlat",
        "datum": "WGS84",
        "ellps": "WGS84",
        "_is_geographic": True,
        "_name": "WGS 84",
    },
    4269: {
        "proj": "longlat",
        "datum": "NAD83",
        "ellps": "GRS80",
        "_is_geographic": True,
        "_name": "NAD83",
    },
    4267: {
        "proj": "longlat",
        "datum": "NAD27",
        "ellps": "clrk66",
        "_is_geographic": True,
        "_name": "NAD27",
    },
    # Web Mercator -------------------------------------------------
    3857: {
        "proj": "merc",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lat_ts": 0,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / Pseudo-Mercator",
    },
    # Ellipsoidal Mercator -----------------------------------------
    3395: {
        "proj": "merc",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lat_ts": 0,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / World Mercator",
    },
    # Lambert Conformal Conic --------------------------------------
    2154: {
        "proj": "lcc",
        "ellps": "GRS80",
        "lon_0": 3,
        "lat_0": 46.5,
        "lat_1": 49,
        "lat_2": 44,
        "x_0": 700000,
        "y_0": 6600000,
        "_is_geographic": False,
        "_name": "RGF93 / Lambert-93",
    },
    # Albers Equal Area --------------------------------------------
    5070: {
        "proj": "aea",
        "datum": "NAD83",
        "ellps": "GRS80",
        "lon_0": -96,
        "lat_0": 23,
        "lat_1": 29.5,
        "lat_2": 45.5,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "NAD83 / Conus Albers",
    },
    # Lambert Azimuthal Equal Area ---------------------------------
    3035: {
        "proj": "laea",
        "ellps": "GRS80",
        "lon_0": 10,
        "lat_0": 52,
        "x_0": 4321000,
        "y_0": 3210000,
        "_is_geographic": False,
        "_name": "ETRS89-extended / LAEA Europe",
    },
    # Polar Stereographic ------------------------------------------
    3031: {
        "proj": "stere",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lon_0": 0,
        "lat_0": -90,
        "lat_ts": -71,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / Antarctic Polar Stereographic",
    },
    3413: {
        "proj": "stere",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lon_0": -45,
        "lat_0": 90,
        "lat_ts": 70,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / NSIDC Sea Ice Polar Stereographic North",
    },
    3996: {
        "proj": "stere",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lon_0": 0,
        "lat_0": 90,
        "lat_ts": 75,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / IBCAO Polar Stereographic",
    },
    # Oblique Stereographic ----------------------------------------
    28992: {
        "proj": "sterea",
        "ellps": "bessel",
        "lon_0": 5.38763888889,
        "lat_0": 52.15616055556,
        "k_0": 0.9999079,
        "x_0": 155000,
        "y_0": 463000,
        "_is_geographic": False,
        "_name": "Amersfoort / RD New",
    },
    # Cylindrical Equal Area ---------------------------------------
    6933: {
        "proj": "cea",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lon_0": 0,
        "lat_ts": 30,
        "x_0": 0,
        "y_0": 0,
        "_is_geographic": False,
        "_name": "WGS 84 / NSIDC EASE-Grid 2.0 Global",
    },
}


def _make_utm_entry(zone: int, south: bool, datum: str, ellps: str) -> dict:
    """Build an EPSG table entry for a UTM zone."""
    d: dict = {
        "proj": "utm",
        "zone": zone,
        "datum": datum,
        "ellps": ellps,
        "x_0": 500000,
        "y_0": 10000000 if south else 0,
        "_is_geographic": False,
    }
    if south:
        d["south"] = True
    hemi = "S" if south else "N"
    d["_name"] = f"{datum} / UTM zone {zone}{hemi}"
    return d


def _populate_utm():
    """Add WGS84 and NAD83 UTM zones to the EPSG table."""
    # WGS84 UTM North: 32601-32660
    for zone in range(1, 61):
        _EPSG_TABLE[32600 + zone] = _make_utm_entry(
            zone, south=False, datum="WGS84", ellps="WGS84"
        )
    # WGS84 UTM South: 32701-32760
    for zone in range(1, 61):
        _EPSG_TABLE[32700 + zone] = _make_utm_entry(
            zone, south=True, datum="WGS84", ellps="WGS84"
        )
    # NAD83 UTM: 26901-26923
    for zone in range(1, 24):
        _EPSG_TABLE[26900 + zone] = _make_utm_entry(
            zone, south=False, datum="NAD83", ellps="GRS80"
        )


_populate_utm()

# WKT projection name mapping
_PROJ_TO_WKT_METHOD = {
    "longlat": None,  # geographic, no projection method
    "merc": "Mercator_1SP",
    "utm": "Transverse_Mercator",
    "lcc": "Lambert_Conformal_Conic_2SP",
    "aea": "Albers_Conic_Equal_Area",
    "laea": "Lambert_Azimuthal_Equal_Area",
    "stere": "Polar_Stereographic",
    "sterea": "Oblique_Stereographic",
    "cea": "Cylindrical_Equal_Area",
}

# Regex to extract AUTHORITY["EPSG","XXXX"] from WKT
_AUTHORITY_RE = re.compile(r'AUTHORITY\["EPSG"\s*,\s*"(\d+)"\]')


class CRS:
    """Lightweight coordinate reference system, compatible with a subset
    of the ``pyproj.CRS`` API.

    Parameters
    ----------
    value : int or str
        EPSG code (``4326``) or authority string (``"EPSG:4326"``).
    """

    __slots__ = ("_epsg", "_entry")

    def __init__(self, value: int | str):
        epsg = self._parse_input(value)
        if epsg not in _EPSG_TABLE:
            raise ValueError(
                f"EPSG:{epsg} is not in the built-in table. "
                f"Install pyproj for full CRS support:  "
                f"pip install pyproj  or:  pip install xarray-spatial[reproject]"
            )
        self._epsg = epsg
        self._entry = _EPSG_TABLE[epsg]

    # -- construction helpers ------------------------------------------

    @staticmethod
    def _parse_input(value: int | str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            m = re.match(r"^EPSG:(\d+)$", value, re.IGNORECASE)
            if m:
                return int(m.group(1))
            raise ValueError(f"Cannot parse CRS string: {value!r}")
        raise TypeError(f"Expected int or str, got {type(value).__name__}")

    @classmethod
    def from_epsg(cls, code: int) -> "CRS":
        """Construct from an integer EPSG code."""
        return cls(code)

    @classmethod
    def from_wkt(cls, wkt: str) -> "CRS":
        """Construct from an OGC WKT string by extracting the AUTHORITY tag.

        Uses the last AUTHORITY["EPSG","..."] match, which in OGC WKT1
        is the outermost (root) authority.
        """
        matches = _AUTHORITY_RE.findall(wkt)
        if not matches:
            raise ValueError("No AUTHORITY[\"EPSG\",...] found in WKT string")
        epsg = int(matches[-1])
        return cls(epsg)

    # -- properties ----------------------------------------------------

    @property
    def is_geographic(self) -> bool:
        """True if this CRS is geographic (lat/lon), False if projected."""
        return self._entry["_is_geographic"]

    # -- output methods ------------------------------------------------

    def to_epsg(self) -> int:
        """Return the EPSG code as an integer."""
        return self._epsg

    def to_authority(self) -> Tuple[str, str]:
        """Return ``("EPSG", "<code>")`` tuple."""
        return ("EPSG", str(self._epsg))

    def to_dict(self) -> dict:
        """Return a proj4-style parameter dictionary.

        Internal keys (prefixed with ``_``) are stripped.
        """
        return {k: v for k, v in self._entry.items() if not k.startswith("_")}

    def to_wkt(self) -> str:
        """Generate a minimal OGC WKT1 string with AUTHORITY tag."""
        entry = self._entry
        name = entry.get("_name", f"EPSG:{self._epsg}")
        ellps_name = entry.get("ellps", "WGS84")
        a, f = _ELLIPSOIDS.get(ellps_name, _ELLIPSOIDS["WGS84"])
        rf = 1.0 / f if f != 0 else 0.0

        # Datum name
        datum_name = entry.get("datum", name)

        # SPHEROID
        spheroid = (
            f'SPHEROID["{ellps_name}",{a},{rf}]'
        )

        if self.is_geographic:
            # GEOGCS
            wkt = (
                f'GEOGCS["{name}",'
                f'DATUM["{datum_name}",{spheroid}],'
                f'PRIMEM["Greenwich",0],'
                f'UNIT["degree",0.0174532925199433],'
                f'AUTHORITY["EPSG","{self._epsg}"]]'
            )
        else:
            # PROJCS wrapping a GEOGCS
            proj = entry["proj"]
            method = _PROJ_TO_WKT_METHOD.get(proj, proj)

            # Build GEOGCS for the datum
            geogcs = (
                f'GEOGCS["{datum_name}",'
                f'DATUM["{datum_name}",{spheroid}],'
                f'PRIMEM["Greenwich",0],'
                f'UNIT["degree",0.0174532925199433]]'
            )

            # Projection parameters
            params = self._wkt_parameters(entry)
            param_str = "," + ",".join(params) if params else ""

            wkt = (
                f'PROJCS["{name}",{geogcs},'
                f'PROJECTION["{method}"]{param_str},'
                f'UNIT["metre",1],'
                f'AUTHORITY["EPSG","{self._epsg}"]]'
            )

        return wkt

    @staticmethod
    def _wkt_parameters(entry: dict) -> list[str]:
        """Build WKT PARAMETER[] entries from a proj dict."""
        # For UTM entries, expand zone into explicit TM parameters so that
        # parsers (including pyproj) get the correct central meridian and
        # scale factor rather than defaulting to 0 / 1.
        if entry.get("proj") == "utm" and "zone" in entry:
            zone = entry["zone"]
            lon_0 = zone * 6 - 183
            k_0 = 0.9996
            lat_0 = 0
            x_0 = entry.get("x_0", 500000)
            y_0 = entry.get("y_0", 0)
            return [
                f'PARAMETER["latitude_of_origin",{lat_0}]',
                f'PARAMETER["central_meridian",{lon_0}]',
                f'PARAMETER["scale_factor",{k_0}]',
                f'PARAMETER["false_easting",{x_0}]',
                f'PARAMETER["false_northing",{y_0}]',
            ]

        # Map from proj keys to WKT parameter names
        key_map = {
            "lat_0": "latitude_of_origin",
            "lon_0": "central_meridian",
            "lat_1": "standard_parallel_1",
            "lat_2": "standard_parallel_2",
            "lat_ts": "latitude_of_true_scale",
            "k_0": "scale_factor",
            "x_0": "false_easting",
            "y_0": "false_northing",
        }
        params = []
        for proj_key, wkt_name in key_map.items():
            if proj_key in entry and not proj_key.startswith("_"):
                params.append(f'PARAMETER["{wkt_name}",{entry[proj_key]}]')
        return params

    # -- equality / hashing -------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CRS):
            return self._epsg == other._epsg
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._epsg)

    def __repr__(self) -> str:
        return f"CRS(epsg={self._epsg})"
