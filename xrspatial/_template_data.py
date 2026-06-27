"""Static study-area definitions for :func:`xrspatial.templates.from_template`.

Two registries back ``from_template``:

``_REGIONS``
    Curated named study areas. Each value carries projected ``bounds``
    ``(left, bottom, right, top)`` in its native CRS, an EPSG ``crs`` code, a
    ``default_resolution`` (cell size in CRS units), and a human ``label``.
    Projected bounds were computed by densely sampling the region's lon/lat
    box and projecting it with pyproj.

``_COUNTRY_BBOXES``
    ISO-3166-1 / GADM alpha-3 country code -> ``(lon_min, lat_min, lon_max,
    lat_max)`` in EPSG:4326 (degrees). Derived from Natural Earth 50m admin_0
    boundaries (public domain). Extents cover a country's full territory,
    including overseas regions, so wide nations (e.g. France with overseas
    departments) get correspondingly wide grids; reach for a curated region or
    pass an explicit ``resolution`` when you want a compact canvas.
    Longitudes may exceed 180 for countries whose territory crosses the
    antimeridian (e.g. Russia, Fiji), keeping those bounding boxes contiguous.
"""

# Curated named regions. ``bounds`` are in the region's native projected CRS.
# ``lonlat`` is the region's (lon_min, lat_min, lon_max, lat_max) in EPSG:4326,
# used by the ``preserve`` path to pick and project into an EPSG-coded CRS.
# ``area_epsg`` / ``shape_epsg`` override the equal-area / conformal EPSG the
# ``preserve`` path would otherwise derive (Equal Earth fallback / UTM zone).
_REGIONS = {
    'conus': dict(bounds=(-2917619, 152524, 2947060, 3257588), crs=5070,
                  default_resolution=5000, label='CONUS (Albers Equal-Area)',
                  lonlat=(-124.8, 24.4, -66.9, 49.4), area_epsg=5070),
    'alaska': dict(bounds=(-1784455, 110372, 1725461, 2570024), crs=3338,
                   default_resolution=5000, label='Alaska (Albers)',
                   lonlat=(-179.9, 51.0, -129.0, 71.5), area_epsg=3338),
    'hawaii': dict(bounds=(363084, 2089763, 942628, 2472061), crs=32604,
                   default_resolution=1000, label='Hawaii (UTM 4N)',
                   lonlat=(-160.3, 18.9, -154.8, 22.3)),
    'california': dict(bounds=(-423462, -612396, 555056, 457663), crs=3310,
                       default_resolution=1000, label='California (Albers)',
                       lonlat=(-124.5, 32.5, -114.1, 42.0), area_epsg=3310),
    'europe': dict(bounds=(1172449, 1218199, 7469551, 5735170), crs=3035,
                   default_resolution=10000, label='Europe (LAEA)',
                   lonlat=(-25.0, 34.0, 45.0, 72.0), area_epsg=3035,
                   shape_epsg=3034),
    'nyc': dict(bounds=(558916, 4481270, 614426, 4534084), crs=32618,
                default_resolution=30, label='New York City (UTM 18N)',
                lonlat=(-74.30, 40.48, -73.65, 40.95)),
    # Continental regions in their EPSG-coded GLANCE equal-area projection
    # (Lambert azimuthal equal-area), the same family as Europe's LAEA. bounds
    # are the lon/lat box projected into the GLANCE CRS. No shape_epsg is set:
    # there is no EPSG conformal projection for these continental extents, so
    # preserve='shape' falls back to the centroid's UTM zone (covers a slice
    # only). The lon/lat boxes follow the real region extent and may run a
    # degree past the GLANCE area of use near the equator; LAEA still projects
    # those points finitely.
    'southeast_asia': dict(
        bounds=(-987821, -5961342, 4923248, -932582), crs=10594,
        default_resolution=10000, label='Southeast Asia (GLANCE Asia LAEA)',
        lonlat=(92.0, -11.0, 141.0, 28.0), area_epsg=10594),
    'central_america': dict(
        bounds=(821182, -4620810, 2695977, -3111382), crs=10598,
        default_resolution=2000, label='Central America (GLANCE N. America LAEA)',
        lonlat=(-92.5, 7.0, -77.0, 18.5), area_epsg=10598),
    'caribbean': dict(
        bounds=(1500961, -4302986, 4618873, -1455289), crs=10598,
        default_resolution=5000, label='Caribbean (GLANCE N. America LAEA)',
        lonlat=(-85.0, 9.0, -59.0, 27.5), area_epsg=10598),
    'west_africa': dict(
        bounds=(-4141633, -109304, -404383, 2659299), crs=10592,
        default_resolution=5000, label='West Africa (GLANCE Africa LAEA)',
        lonlat=(-18.0, 4.0, 16.0, 27.0), area_epsg=10592),
    'north_africa': dict(
        bounds=(-3867207, 1434990, 1804670, 3865653), crs=10592,
        default_resolution=10000, label='North Africa (GLANCE Africa LAEA)',
        lonlat=(-17.0, 18.0, 37.0, 38.0), area_epsg=10592),
    'east_africa': dict(
        bounds=(851984, -1872363, 3520386, 1573813), crs=10592,
        default_resolution=5000, label='East Africa (GLANCE Africa LAEA)',
        lonlat=(28.0, -12.0, 52.0, 18.0), area_epsg=10592),
    'southern_africa': dict(
        bounds=(-997536, -4373971, 2316937, -1421722), crs=10592,
        default_resolution=5000, label='Southern Africa (GLANCE Africa LAEA)',
        lonlat=(11.0, -35.0, 41.0, -8.0), area_epsg=10592),
    'south_asia': dict(
        bounds=(-4560937, -4342133, -175948, 56016), crs=10594,
        default_resolution=5000, label='South Asia (GLANCE Asia LAEA)',
        lonlat=(60.0, 5.0, 98.0, 38.0), area_epsg=10594),
    'east_asia': dict(
        bounds=(-2888124, -2967169, 4752508, 1872058), crs=10594,
        default_resolution=10000, label='East Asia (GLANCE Asia LAEA)',
        lonlat=(73.0, 18.0, 146.0, 54.0), area_epsg=10594),
    'central_asia': dict(
        bounds=(-4529046, -1031518, -748494, 2365024), crs=10594,
        default_resolution=5000, label='Central Asia (GLANCE Asia LAEA)',
        lonlat=(46.0, 35.0, 88.0, 56.0), area_epsg=10594),
    'middle_east': dict(
        bounds=(-6742347, -2794074, -2936764, 1798849), crs=10594,
        default_resolution=5000, label='Middle East (GLANCE Asia LAEA)',
        lonlat=(34.0, 12.0, 63.0, 42.0), area_epsg=10594),
    'south_america': dict(
        bounds=(-2461362, -4624186, 2901620, 3066400), crs=10603,
        default_resolution=10000, label='South America (GLANCE S. America LAEA)',
        lonlat=(-82.0, -56.0, -34.0, 13.0), area_epsg=10603),
    # Oceania bounded west of the antimeridian (Australia, New Guinea, New
    # Zealand) so the lon/lat box does not wrap 180.
    'oceania': dict(
        bounds=(-2736534, -4140479, 4725674, 773877), crs=10601,
        default_resolution=10000, label='Oceania (GLANCE Oceania LAEA)',
        lonlat=(110.0, -48.0, 179.0, -8.0), area_epsg=10601),
    'australia': dict(
        bounds=(-2397336, -3314170, 2074139, 552968), crs=10601,
        default_resolution=5000, label='Australia (GLANCE Oceania LAEA)',
        lonlat=(113.0, -44.0, 154.0, -10.0), area_epsg=10601),
    'new_zealand': dict(
        bounds=(2357704, -4140479, 3965299, -2362651), crs=10601,
        default_resolution=2000, label='New Zealand (GLANCE Oceania LAEA)',
        lonlat=(166.0, -48.0, 179.0, -34.0), area_epsg=10601),
    'central_africa': dict(
        bounds=(-1335054, -2091682, 1224158, 789768), crs=10592,
        default_resolution=5000, label='Central Africa (GLANCE Africa LAEA)',
        lonlat=(8.0, -14.0, 31.0, 12.0), area_epsg=10592),
    'north_asia': dict(
        bounds=(-2829330, 333534, 4672506, 4568106), crs=10594,
        default_resolution=10000, label='North Asia (GLANCE Asia LAEA)',
        lonlat=(60.0, 48.0, 179.0, 78.0), area_epsg=10594),
    'greenland': dict(
        bounds=(307428, 1266923, 3614205, 4339302), crs=10598,
        default_resolution=5000, label='Greenland (GLANCE N. America LAEA)',
        lonlat=(-74.0, 59.0, -11.0, 84.0), area_epsg=10598),
    'canada': dict(
        bounds=(-3271722, -999331, 3748086, 3935367), crs=10598,
        default_resolution=10000, label='Canada (GLANCE N. America LAEA)',
        lonlat=(-141.0, 41.0, -52.0, 84.0), area_epsg=10598),
    'mexico': dict(
        bounds=(-2026872, -3928666, 1581449, -1690630), crs=10598,
        default_resolution=5000, label='Mexico (GLANCE N. America LAEA)',
        lonlat=(-118.0, 14.0, -86.0, 33.0), area_epsg=10598),
    'great_lakes': dict(
        bounds=(506191, -972693, 2067092, 243847), crs=10598,
        default_resolution=2000, label='Great Lakes (GLANCE N. America LAEA)',
        lonlat=(-93.0, 41.0, -75.0, 49.5), area_epsg=10598),
    'pacific_northwest': dict(
        bounds=(-2033861, -823605, -752346, 508938), crs=10598,
        default_resolution=2000, label='Pacific Northwest (GLANCE N. America LAEA)',
        lonlat=(-125.0, 42.0, -111.0, 52.0), area_epsg=10598),
    'gulf_coast': dict(
        bounds=(193569, -2859113, 1963599, -1884446), crs=10598,
        default_resolution=2000, label='Gulf Coast (GLANCE N. America LAEA)',
        lonlat=(-98.0, 24.0, -81.0, 31.0), area_epsg=10598),
    'new_england': dict(
        bounds=(1914015, -633460, 2686814, 258722), crs=10598,
        default_resolution=1000, label='New England (GLANCE N. America LAEA)',
        lonlat=(-74.0, 41.0, -67.0, 47.5), area_epsg=10598),
    'great_plains': dict(
        bounds=(-483626, -2100615, 387002, -99072), crs=10598,
        default_resolution=5000, label='Great Plains (GLANCE N. America LAEA)',
        lonlat=(-105.0, 31.0, -96.0, 49.0), area_epsg=10598),
    'american_southwest': dict(
        bounds=(-1913308, -2095218, -249048, -674284), crs=10598,
        default_resolution=2000, label='American Southwest (GLANCE N. America LAEA)',
        lonlat=(-120.0, 31.0, -103.0, 42.0), area_epsg=10598),
    'amazon_basin': dict(
        bounds=(-2129139, -422334, 1795411, 2309720), crs=10603,
        default_resolution=5000, label='Amazon Basin (GLANCE S. America LAEA)',
        lonlat=(-79.0, -18.0, -44.0, 6.0), area_epsg=10603),
    'andes': dict(
        bounds=(-2350812, -4564243, -133276, 2958360), crs=10603,
        default_resolution=10000, label='Andes (GLANCE S. America LAEA)',
        lonlat=(-81.0, -56.0, -62.0, 12.0), area_epsg=10603),
    'southern_cone': dict(
        bounds=(-1697271, -4517109, 744960, -221307), crs=10603,
        default_resolution=5000, label='Southern Cone (GLANCE S. America LAEA)',
        lonlat=(-76.0, -56.0, -53.0, -17.0), area_epsg=10603),
    'western_europe': dict(
        bounds=(-2382583, -1327155, -191916, 406028), crs=10596,
        default_resolution=5000, label='Western Europe (GLANCE Europe LAEA)',
        lonlat=(-10.0, 43.0, 17.0, 55.0), area_epsg=10596),
    'eastern_europe': dict(
        bounds=(-482758, -1221647, 2340946, 916013), crs=10596,
        default_resolution=5000, label='Eastern Europe (GLANCE Europe LAEA)',
        lonlat=(14.0, 44.0, 50.0, 60.0), area_epsg=10596),
    'northern_europe': dict(
        bounds=(-1039068, -111313, 782636, 1847310), crs=10596,
        default_resolution=2000, label='Northern Europe (GLANCE Europe LAEA)',
        lonlat=(4.0, 54.0, 32.0, 71.0), area_epsg=10596),
    'southern_europe': dict(
        bounds=(-2698894, -2211793, 739758, -416605), crs=10596,
        default_resolution=5000, label='Southern Europe (GLANCE Europe LAEA)',
        lonlat=(-10.0, 35.0, 28.0, 47.0), area_epsg=10596),
    # Antarctica uses the de-facto standard Antarctic Polar Stereographic
    # (EPSG:3031, conformal), so preserve='area' falls back to the EPSG-coded
    # south-polar equal-area grid (EPSG:6932) rather than claiming 3031 is
    # equal-area. shape_epsg is 3031 itself (already conformal).
    'antarctica': dict(
        bounds=(-3333134, -3333134, 3333134, 3333134), crs=3031,
        default_resolution=10000, label='Antarctica (Polar Stereographic)',
        lonlat=(-180.0, -90.0, 180.0, -60.0), area_epsg=6932, shape_epsg=3031),
    # The default (non-preserve) world grid spans the full +/-90 in EPSG:4326.
    # The preserve path uses a +/-85 latitude band (the conventional Web
    # Mercator limit) so 'shape' (World Mercator) does not diverge at the poles.
    'world': dict(bounds=(-180.0, -90.0, 180.0, 90.0), crs=4326,
                  default_resolution=0.5, label='World (WGS84)',
                  lonlat=(-180.0, -85.0, 180.0, 85.0), area_epsg=8857,
                  shape_epsg=3395),
    # Global Web Mercator (EPSG:3857). Native bounds are the canonical square
    # extent +/-20037508 m, which is +/-180 lon and +/-85.0511287798 lat (the
    # latitude where the Mercator y matches the x half-extent). lonlat stops at
    # that latitude because Mercator y diverges to infinity at the poles.
    'web_mercator': dict(
        bounds=(-20037508, -20037508, 20037508, 20037508), crs=3857,
        default_resolution=50000, label='World (Web Mercator)',
        lonlat=(-180.0, -85.0511287798, 180.0, 85.0511287798),
        area_epsg=8857, shape_epsg=3395),
    # Global equal-area grid (EPSG:8857, Equal Earth). Native bounds are the
    # +/-90 world projected into Equal Earth.
    'equal_earth': dict(
        bounds=(-17243959, -8392928, 17243959, 8392928), crs=8857,
        default_resolution=50000, label='World (Equal Earth)',
        lonlat=(-180.0, -90.0, 180.0, 90.0),
        area_epsg=8857, shape_epsg=3395),
    # Pacific-centered world (EPSG:3832, WGS 84 / PDC Mercator). lon_0=150 so the
    # Pacific Ocean is continuous, with the map seam in the Atlantic (~30 W). x
    # spans the full a*pi longitude extent; y is the ellipsoidal Mercator value
    # at the conventional +/-85.0511287798 latitude limit. Conformal, so
    # area_epsg falls back to Equal Earth for preserve='area'.
    'pacific': dict(
        bounds=(-20037508, -19994875, 20037508, 19994875), crs=3832,
        default_resolution=50000, label='Pacific-centered World (PDC Mercator)',
        lonlat=(-180.0, -85.0511287798, 180.0, 85.0511287798),
        area_epsg=8857, shape_epsg=3832),
}

# Alternate spellings that resolve to a curated region (single source of truth).
# 'wgs84' / 'latlon' are friendly names for the EPSG:4326 'world' grid; 'pdc' is
# the Pacific Disaster Center's name for its Pacific-centered Mercator.
_REGION_ALIASES = {
    'wgs84': 'world',
    'latlon': 'world',
    'pdc': 'pacific',
}

# Equal-area fallback when a template has no curated ``area_epsg``
# (all country codes, plus regions like hawaii/nyc): EPSG:8857 Equal Earth.
_EQUAL_AREA_FALLBACK_EPSG = 8857

# Polar conformal fallback for the ``shape`` path when |lat| exceeds the UTM
# usable range: EPSG:5041 (UPS North) / 5042 (UPS South).
_UPS_NORTH_EPSG = 5041
_UPS_SOUTH_EPSG = 5042

# Default cell size (degrees) for country-code templates returned in EPSG:4326.
_COUNTRY_DEFAULT_RESOLUTION = 0.1

# Default cell size (metres) for city templates, which come back in a UTM zone.
# Matches the 30 m of the curated 'nyc' region.
_CITY_DEFAULT_RESOLUTION = 30

# ISO-3166-1 / GADM alpha-3 -> (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
_COUNTRY_BBOXES = {
    'ABW': (-70.066, 12.423, -69.896, 12.614),  # Aruba
    'AFG': (60.486, 29.392, 74.891, 38.456),  # Afghanistan
    'AGO': (11.743, -18.02, 24.047, -4.429),  # Angola
    'AIA': (-63.16, 18.171, -62.98, 18.27),  # Anguilla
    'ALA': (19.519, 60.012, 20.611, 60.406),  # Åland
    'ALB': (19.281, 39.654, 21.031, 42.648),  # Albania
    'AND': (1.415, 42.434, 1.74, 42.643),  # Andorra
    'ARE': (51.568, 22.621, 56.388, 26.068),  # United Arab Emirates
    'ARG': (-73.576, -55.032, -53.669, -21.803),  # Argentina
    'ARM': (43.439, 38.869, 46.585, 41.291),  # Armenia
    'ASM': (-170.821, -14.36, -170.568, -14.257),  # American Samoa
    'ATA': (0, -89.999, 359.815, -60.521),  # Antarctica
    'ATF': (51.659, -49.71, 70.555, -46.327),  # French Southern and Antarctic Lands
    'ATG': (-61.887, 16.997, -61.686, 17.714),  # Antigua and Barbuda
    'AUS': (112.908, -54.749, 158.959, -10.052),  # Australia
    'AUT': (9.524, 46.4, 17.147, 49.001),  # Austria
    'AZE': (44.768, 38.399, 50.366, 41.891),  # Azerbaijan
    'BDI': (29.014, -4.456, 30.811, -2.313),  # Burundi
    'BEL': (2.525, 49.511, 6.364, 51.491),  # Belgium
    'BEN': (0.763, 6.217, 3.834, 12.384),  # Benin
    'BFA': (-5.524, 9.425, 2.389, 15.078),  # Burkina Faso
    'BGD': (88.023, 20.79, 92.632, 26.572),  # Bangladesh
    'BGR': (22.344, 41.244, 28.585, 44.238),  # Bulgaria
    'BHR': (50.452, 25.807, 50.617, 26.246),  # Bahrain
    'BHS': (-78.986, 20.937, -72.747, 26.94),  # The Bahamas
    'BIH': (15.737, 42.56, 19.584, 45.277),  # Bosnia and Herzegovina
    'BLM': (-62.875, 17.875, -62.8, 17.922),  # Saint Barthélemy
    'BLR': (23.175, 51.265, 32.71, 56.146),  # Belarus
    'BLZ': (-89.237, 15.889, -87.789, 18.482),  # Belize
    'BMU': (-64.863, 32.26, -64.668, 32.387),  # Bermuda
    'BOL': (-69.646, -22.892, -57.496, -9.71),  # Bolivia
    'BRA': (-74.002, -33.742, -34.805, 5.258),  # Brazil
    'BRB': (-59.647, 13.062, -59.428, 13.318),  # Barbados
    'BRN': (114.064, 4.024, 115.327, 5.022),  # Brunei
    'BTN': (88.739, 26.702, 92.083, 28.311),  # Bhutan
    'BWA': (19.977, -26.854, 29.365, -17.788),  # Botswana
    'CAF': (14.431, 2.27, 27.403, 10.996),  # Central African Republic
    'CAN': (-141.002, 41.675, -52.654, 83.116),  # Canada
    'CHE': (5.97, 45.83, 10.455, 47.776),  # Switzerland
    'CHL': (-109.434, -55.892, -66.436, -17.506),  # Chile
    'CHN': (73.607, 18.218, 134.752, 53.556),  # People's Republic of China
    'CIV': (-8.604, 4.351, -2.506, 10.724),  # Ivory Coast
    'CMR': (8.533, 1.676, 16.183, 13.079),  # Cameroon
    'COD': (12.214, -13.454, 31.274, 5.312),  # Democratic Republic of the Congo
    'COG': (11.13, -5.004, 18.622, 3.687),  # Republic of the Congo
    'COK': (-159.842, -21.25, -159.737, -21.186),  # Cook Islands
    'COL': (-79.025, -4.236, -66.876, 12.434),  # Colombia
    'COM': (43.227, -12.368, 44.527, -11.368),  # Comoros
    'CPV': (-25.342, 14.818, -22.682, 17.194),  # Cape Verde
    'CRI': (-85.908, 8.071, -82.564, 11.189),  # Costa Rica
    'CUB': (-84.887, 19.855, -74.137, 23.19),  # Cuba
    'CUW': (-69.159, 12.045, -68.751, 12.38),  # Curaçao
    'CYM': (-81.419, 19.272, -79.742, 19.766),  # Cayman Islands
    'CYN': (32.713, 35.0, 34.556, 35.662),  # Turkish Republic of Northern Cyprus
    'CYP': (32.301, 34.57, 34.05, 35.183),  # Cyprus
    'CZE': (12.09, 48.576, 18.832, 51.038),  # Czech Republic
    'DEU': (5.858, 47.279, 15.017, 55.059),  # Germany
    'DJI': (41.765, 10.941, 43.41, 12.709),  # Djibouti
    'DMA': (-61.481, 15.227, -61.251, 15.633),  # Dominica
    'DNK': (8.121, 54.629, 15.137, 57.737),  # Denmark
    'DOM': (-72.0, 17.636, -68.339, 19.914),  # Dominican Republic
    'DZA': (-8.683, 18.987, 11.968, 37.092),  # Algeria
    'ECU': (-91.654, -4.991, -75.25, 1.455),  # Ecuador
    'EGY': (24.703, 21.995, 36.871, 31.655),  # Egypt
    'ERI': (36.427, 12.377, 43.117, 18.005),  # Eritrea
    'ESH': (-17.099, 20.806, -8.682, 27.656),  # Western Sahara
    'ESP': (-18.161, 27.646, 4.322, 43.765),  # Spain
    'EST': (21.854, 57.525, 28.151, 59.639),  # Estonia
    'ETH': (32.999, 3.456, 47.978, 14.852),  # Ethiopia
    'FIN': (20.622, 59.816, 31.537, 70.065),  # Finland
    'FJI': (174.587, -21.706, 181.749, -12.477),  # Fiji
    'FLK': (-61.145, -52.308, -57.792, -51.27),  # Falkland Islands
    'FRA': (-61.794, -21.369, 55.839, 51.097),  # France
    'FRO': (-7.423, 61.414, -6.406, 62.356),  # Faroe Islands
    'FSM': (138.062, 5.277, 162.993, 9.593),  # Federated States of Micronesia
    'GAB': (8.703, -3.916, 14.481, 2.302),  # Gabon
    'GBR': (-8.145, 50.021, 1.747, 60.832),  # United Kingdom
    'GEO': (39.978, 41.07, 46.673, 43.57),  # Georgia
    'GGY': (-2.646, 49.429, -2.512, 49.507),  # Guernsey
    'GHA': (-3.244, 4.762, 1.187, 11.167),  # Ghana
    'GIN': (-15.051, 7.216, -7.681, 12.674),  # Guinea
    'GMB': (-16.825, 13.064, -13.827, 13.812),  # The Gambia
    'GNB': (-16.712, 10.94, -13.674, 12.68),  # Guinea-Bissau
    'GNQ': (8.434, 0.96, 11.335, 3.758),  # Equatorial Guinea
    'GRC': (19.646, 34.934, 28.232, 41.744),  # Greece
    'GRD': (-61.782, 12.008, -61.607, 12.237),  # Grenada
    'GRL': (-72.818, 59.815, -11.426, 83.6),  # Greenland
    'GTM': (-92.235, 13.737, -88.228, 17.816),  # Guatemala
    'GUM': (144.649, 13.258, 144.941, 13.622),  # Guam
    'GUY': (-61.391, 1.201, -56.483, 8.549),  # Guyana
    'HKG': (113.839, 22.195, 114.335, 22.565),  # Hong Kong
    'HMD': (73.251, -53.185, 73.838, -52.966),  # Heard Island and McDonald Islands
    'HND': (-89.363, 12.979, -83.158, 16.514),  # Honduras
    'HRV': (13.517, 42.433, 19.401, 46.535),  # Croatia
    'HTI': (-74.478, 18.039, -71.645, 20.094),  # Haiti
    'HUN': (16.093, 45.753, 22.877, 48.553),  # Hungary
    'IDN': (95.207, -10.91, 140.976, 5.907),  # Indonesia
    'IMN': (-4.785, 54.059, -4.338, 54.407),  # Isle of Man
    'IND': (68.165, 6.749, 97.344, 35.496),  # India
    'IOT': (72.35, -7.435, 72.499, -7.22),  # British Indian Ocean Territory
    'IRL': (-10.39, 51.474, -6.027, 55.366),  # Ireland
    'IRN': (44.023, 25.102, 63.305, 39.769),  # Iran
    'IRQ': (38.774, 29.064, 48.546, 37.372),  # Iraq
    'ISL': (-24.476, 63.407, -13.556, 66.526),  # Iceland
    'ISR': (34.245, 29.477, 35.888, 33.416),  # Israel
    'ITA': (6.628, 36.688, 18.486, 47.082),  # Italy
    'JAM': (-78.34, 17.715, -76.211, 18.522),  # Jamaica
    'JEY': (-2.236, 49.17, -2.01, 49.266),  # Jersey
    'JOR': (34.951, 29.19, 39.293, 33.372),  # Jordan
    'JPN': (123.68, 24.266, 145.833, 45.51),  # Japan
    'KAS': (76.767, 35.11, 77.799, 35.662),  # Siachen Glacier
    'KAZ': (46.609, 40.609, 87.323, 55.39),  # Kazakhstan
    'KEN': (33.9, -4.692, 41.884, 5.492),  # Kenya
    'KGZ': (69.229, 39.208, 80.246, 43.24),  # Kyrgyzstan
    'KHM': (102.32, 10.411, 107.605, 14.705),  # Cambodia
    'KIR': (169.523, -11.457, 208.217, 3.924),  # Kiribati
    'KNA': (-62.84, 17.101, -62.532, 17.403),  # Saint Kitts and Nevis
    'KOR': (126.008, 33.202, 130.934, 38.623),  # South Korea
    'KOS': (20.029, 41.854, 21.753, 43.261),  # Kosovo
    'KWT': (46.531, 28.533, 48.442, 30.097),  # Kuwait
    'LAO': (100.115, 13.921, 107.653, 22.495),  # Laos
    'LBN': (35.109, 33.076, 36.585, 34.679),  # Lebanon
    'LBR': (-11.508, 4.351, -7.4, 8.538),  # Liberia
    'LBY': (9.31, 19.497, 25.15, 33.182),  # Libya
    'LCA': (-61.073, 13.718, -60.887, 14.093),  # Saint Lucia
    'LIE': (9.479, 47.057, 9.611, 47.271),  # Liechtenstein
    'LKA': (79.708, 5.949, 81.877, 9.813),  # Sri Lanka
    'LSO': (27.052, -30.642, 29.391, -28.582),  # Lesotho
    'LTU': (20.9, 53.893, 26.776, 56.411),  # Lithuania
    'LUX': (5.725, 49.445, 6.494, 50.167),  # Luxembourg
    'LVA': (21.015, 55.668, 28.202, 58.063),  # Latvia
    'MAC': (113.479, 22.196, 113.548, 22.246),  # Macau
    'MAF': (-63.123, 18.069, -63.009, 18.115),  # Saint Martin
    'MAR': (-17.003, 21.421, -1.066, 35.93),  # Morocco
    'MCO': (7.378, 43.732, 7.439, 43.771),  # Monaco
    'MDA': (26.619, 45.45, 30.131, 48.478),  # Moldova
    'MDG': (43.257, -25.571, 50.483, -12.08),  # Madagascar
    'MDV': (73.382, 3.229, 73.528, 4.248),  # Maldives
    'MEX': (-118.401, 14.545, -86.696, 32.715),  # Mexico
    'MHL': (166.845, 5.8, 171.757, 11.169),  # Marshall Islands
    'MKD': (20.449, 40.85, 23.006, 42.358),  # North Macedonia
    'MLI': (-12.281, 10.143, 4.235, 24.996),  # Mali
    'MLT': (14.18, 35.82, 14.566, 36.076),  # Malta
    'MMR': (92.18, 9.875, 101.147, 28.517),  # Myanmar
    'MNE': (18.436, 41.869, 20.348, 43.542),  # Montenegro
    'MNG': (87.743, 41.596, 119.898, 52.117),  # Mongolia
    'MNP': (145.152, 14.111, 145.835, 18.807),  # Northern Mariana Islands
    'MOZ': (30.222, -26.862, 40.845, -10.464),  # Mozambique
    'MRT': (-17.064, 14.745, -4.823, 27.286),  # Mauritania
    'MSR': (-62.223, 16.681, -62.148, 16.81),  # Montserrat
    'MUS': (57.318, -20.513, 57.792, -19.99),  # Mauritius
    'MWI': (32.67, -17.131, 35.893, -9.395),  # Malawi
    'MYS': (99.646, 0.862, 119.266, 7.352),  # Malaysia
    'NAM': (11.722, -28.939, 25.259, -16.968),  # Namibia
    'NCL': (159.928, -22.661, 168.139, -19.115),  # New Caledonia
    'NER': (0.164, 11.696, 15.963, 23.518),  # Niger
    'NFK': (167.906, -29.096, 167.99, -29.014),  # Norfolk Island
    'NGA': (2.686, 4.277, 14.627, 13.873),  # Nigeria
    'NIC': (-87.67, 10.735, -83.158, 15.008),  # Nicaragua
    'NIU': (-169.948, -19.138, -169.793, -18.966),  # Niue
    'NLD': (-68.371, 12.032, 7.197, 53.625),  # Netherlands
    'NOR': (-9.099, 58.021, 33.629, 80.478),  # Norway
    'NPL': (80.052, 26.36, 88.162, 30.387),  # Nepal
    'NRU': (166.907, -0.551, 166.958, -0.489),  # Nauru
    'NZL': (165.889, -52.57, 188.814, -8.546),  # New Zealand
    'OMN': (51.978, 16.648, 59.837, 26.356),  # Oman
    'PAK': (60.843, 23.753, 77.049, 37.037),  # Pakistan
    'PAN': (-83.027, 7.22, -77.196, 9.598),  # Panama
    'PCN': (-128.35, -24.413, -128.29, -24.323),  # Pitcairn Islands
    'PER': (-81.337, -18.346, -68.685, -0.042),  # Peru
    'PHL': (116.97, 5.06, 126.593, 20.841),  # Philippines
    'PLW': (131.135, 3.022, 134.66, 7.712),  # Palau
    'PNG': (140.862, -11.631, 155.958, -1.353),  # Papua New Guinea
    'POL': (14.129, 49.021, 24.106, 54.838),  # Poland
    'PRI': (-67.937, 17.947, -65.295, 18.522),  # Puerto Rico
    'PRK': (124.349, 37.719, 130.687, 42.998),  # North Korea
    'PRT': (-31.283, 32.648, -6.213, 42.137),  # Portugal
    'PRY': (-62.651, -27.554, -54.242, -19.286),  # Paraguay
    'PSE': (34.198, 31.208, 35.572, 32.534),  # Palestine
    'PYF': (-151.512, -20.876, -136.294, -8.782),  # French Polynesia
    'QAT': (50.755, 24.565, 51.609, 26.153),  # Qatar
    'ROU': (20.242, 43.671, 29.706, 48.263),  # Romania
    'RUS': (19.604, 41.199, 190.271, 81.854),  # Russia
    'RWA': (28.858, -2.809, 30.877, -1.063),  # Rwanda
    'SAU': (34.616, 16.372, 55.641, 32.125),  # Saudi Arabia
    'SDN': (21.825, 8.666, 38.609, 22.202),  # Sudan
    'SEN': (-17.536, 12.328, -11.382, 16.679),  # Senegal
    'SGP': (103.65, 1.265, 103.996, 1.447),  # Singapore
    'SGS': (-38.017, -58.492, -26.26, -53.984),  # South Georgia and the South Sandwich Islands
    'SHN': (-14.415, -16.004, -5.66, -7.883),  # Saint Helena
    'SLB': (155.678, -11.832, 166.929, -6.609),  # Solomon Islands
    'SLE': (-13.293, 6.907, -10.283, 9.997),  # Sierra Leone
    'SLV': (-90.106, 13.164, -87.715, 14.431),  # El Salvador
    'SMR': (12.397, 43.894, 12.515, 43.99),  # San Marino
    'SOL': (42.656, 7.997, 48.939, 11.5),  # Somaliland
    'SOM': (40.964, -1.695, 51.39, 11.984),  # Somalia
    'SPM': (-56.387, 46.753, -56.137, 47.099),  # Saint Pierre and Miquelon
    'SRB': (18.839, 42.242, 22.977, 46.169),  # Serbia
    'SSD': (24.147, 3.491, 35.268, 12.223),  # South Sudan
    'STP': (6.468, 0.047, 7.452, 1.699),  # São Tomé and Príncipe
    'SUR': (-58.054, 1.842, -53.99, 5.993),  # Suriname
    'SVK': (16.863, 47.763, 22.539, 49.598),  # Slovakia
    'SVN': (13.378, 45.428, 16.516, 46.863),  # Slovenia
    'SWE': (11.147, 55.346, 24.155, 69.037),  # Sweden
    'SWZ': (30.788, -27.31, 32.113, -25.743),  # Eswatini
    'SXM': (-63.125, 18.019, -63.011, 18.069),  # Sint Maarten
    'SYC': (55.383, -4.786, 55.543, -4.559),  # Seychelles
    'SYR': (35.764, 32.317, 42.359, 37.297),  # Syria
    'TCA': (-72.342, 21.752, -71.637, 21.952),  # Turks and Caicos Islands
    'TCD': (13.448, 7.475, 23.983, 23.445),  # Chad
    'TGO': (-0.09, 6.089, 1.778, 11.116),  # Togo
    'THA': (97.374, 5.637, 105.641, 20.424),  # Thailand
    'TJK': (67.35, 36.684, 75.119, 41.035),  # Tajikistan
    'TKM': (52.494, 35.171, 66.629, 42.778),  # Turkmenistan
    'TLS': (124.036, -9.512, 127.296, -8.14),  # East Timor
    'TON': (-175.362, -21.451, -173.922, -18.565),  # Tonga
    'TTO': (-61.906, 10.065, -60.525, 11.325),  # Trinidad and Tobago
    'TUN': (7.496, 30.229, 11.536, 37.34),  # Tunisia
    'TUR': (25.669, 35.831, 44.817, 42.093),  # Turkey
    'TUV': (179.196, -8.535, 179.217, -8.466),  # Tuvalu
    'TWN': (118.287, 21.925, 121.929, 25.277),  # Taiwan
    'TZA': (29.323, -11.716, 40.464, -0.995),  # Tanzania
    'UGA': (29.562, -1.47, 34.978, 4.22),  # Uganda
    'UKR': (22.132, 45.234, 40.128, 52.354),  # Ukraine
    'URY': (-58.438, -34.933, -53.126, -30.101),  # Uruguay
    'USA': (172.495, 18.964, 293.013, 71.408),  # United States of America
    'UZB': (55.976, 37.172, 73.137, 45.555),  # Uzbekistan
    'VAT': (12.428, 41.898, 12.439, 41.906),  # Vatican City
    'VCT': (-61.354, 12.695, -61.124, 13.359),  # Saint Vincent and the Grenadines
    'VEN': (-73.366, 0.688, -59.829, 12.178),  # Venezuela
    'VGB': (-64.695, 18.399, -64.274, 18.753),  # British Virgin Islands
    'VIR': (-65.024, 17.702, -64.58, 18.385),  # United States Virgin Islands
    'VNM': (102.127, 8.583, 109.445, 23.345),  # Vietnam
    'VUT': (166.526, -20.242, 169.896, -13.709),  # Vanuatu
    'WLF': (-178.194, -14.325, -176.128, -13.222),  # Wallis and Futuna
    'WSM': (-172.779, -14.047, -171.45, -13.465),  # Samoa
    'YEM': (42.549, 12.319, 54.511, 18.996),  # Yemen
    'ZAF': (16.448, -46.963, 37.888, -22.146),  # South Africa
    'ZMB': (21.979, -18.042, 33.662, -8.194),  # Zambia
    'ZWE': (25.224, -22.402, 33.007, -15.643),  # Zimbabwe
}


# Major world cities and important regional metros, generated
# from Natural Earth 10m populated places (public domain):
# national capitals, places with POP_MAX >= 1.2 million, and a
# curated set of recognizable US secondary cities (Austin, New
# Orleans, ... -- NE's POP_MAX/SCALERANK underrate US metros).
# Each city's CRS is its UTM zone (EPSG:326xx north / 327xx south),
# picked from the centroid -- a standard EPSG projection, never a
# synthesized one. The bounding box is a metro-scale lon/lat box
# (half-width scales with population: ~30 km megacity, ~20 km large,
# ~12 km regional) projected into that UTM zone for `bounds`.
# Slug = ascii-folded lowercase name. When two cities share a slug
# the larger-population one keeps the bare name and the others get a
# `<name>_<iso2>` suffix (e.g. 'hyderabad' vs 'hyderabad_pk').
_CITIES = {
    'abidjan': dict(
        bounds=(364445, 568403, 404627, 608305), crs=32630,
        lonlat=(-4.223, 5.1418, -3.861, 5.5021), label='Abidjan (UTM 30N)'),
    'abu_dhabi': dict(
        bounds=(220781, 2696239, 245332, 2720653), crs=32640,
        lonlat=(54.2478, 24.3586, 54.4854, 24.5748), label='Abu Dhabi (UTM 40N)'),
    'abuja': dict(
        bounds=(318450, 984606, 358733, 1024626), crs=32632,
        lonlat=(7.3489, 8.9051, 7.7139, 9.2655), label='Abuja (UTM 32N)'),
    'accra': dict(
        bounds=(788009, 594375, 828348, 634449), crs=32630,
        lonlat=(-0.3997, 5.3718, -0.0376, 5.7322), label='Accra (UTM 30N)'),
    'ad_damman': dict(
        bounds=(389795, 2903560, 430201, 2943747), crs=32639,
        lonlat=(49.8965, 26.25, 50.2989, 26.6103), label='Ad Damman (UTM 39N)'),
    'adana': dict(
        bounds=(685761, 4076593, 726891, 4117549), crs=32636,
        lonlat=(35.0925, 36.8168, 35.5437, 37.1771), label='Adana (UTM 36N)'),
    'addis_ababa': dict(
        bounds=(446736, 978835, 486873, 1018702), crs=32637,
        lonlat=(38.5156, 8.8551, 38.8805, 9.2154), label='Addis Ababa (UTM 37N)'),
    'agra': dict(
        bounds=(183435, 2988620, 224549, 3029517), crs=32644,
        lonlat=(77.8105, 26.9922, 78.2156, 27.3525), label='Agra (UTM 44N)'),
    'ahmedabad': dict(
        bounds=(221141, 2518711, 282343, 2579590), crs=32643,
        lonlat=(72.2844, 22.7617, 72.8717, 23.3023), label='Ahmedabad (UTM 43N)'),
    'albany': dict(
        bounds=(584501, 4712683, 608936, 4737028), crs=32618,
        lonlat=(-73.967, 42.5619, -73.6729, 42.7781), label='Albany (UTM 18N)'),
    'albuquerque': dict(
        bounds=(338150, 3873740, 362636, 3898116), crs=32613,
        lonlat=(-106.7735, 34.9969, -106.5092, 35.2131), label='Albuquerque (UTM 13N)'),
    'aleppo': dict(
        bounds=(314869, 3990880, 355773, 4031620), crs=32637,
        lonlat=(36.9447, 36.0517, 37.3914, 36.4121), label='Aleppo (UTM 37N)'),
    'alexandria': dict(
        bounds=(760327, 3435237, 801552, 3476264), crs=32635,
        lonlat=(29.7374, 31.0218, 30.1587, 31.3821), label='Alexandria (UTM 35N)'),
    'algiers': dict(
        bounds=(484229, 4048814, 524467, 4088828), crs=32631,
        lonlat=(2.8237, 36.5848, 3.2735, 36.9452), label='Algiers (UTM 31N)'),
    'allahabad': dict(
        bounds=(564095, 2795751, 604472, 2835904), crs=32644,
        lonlat=(81.6385, 25.2768, 82.0376, 25.6371), label='Allahabad (UTM 44N)'),
    'almaty': dict(
        bounds=(634623, 4778463, 675706, 4819398), crs=32643,
        lonlat=(76.6654, 43.1468, 77.1608, 43.5071), label='Almaty (UTM 43N)'),
    'amaravati': dict(
        bounds=(419048, 1798081, 479350, 1858016), crs=32644,
        lonlat=(80.2424, 16.2634, 80.8062, 16.8039), label='Amaravati (UTM 44N)'),
    'amman': dict(
        bounds=(756497, 3518361, 797739, 3559419), crs=32636,
        lonlat=(35.719, 31.7718, 36.1437, 32.1322), label='Amman (UTM 36N)'),
    'amritsar': dict(
        bounds=(467357, 3480783, 507572, 3520769), crs=32643,
        lonlat=(74.6564, 31.4618, 75.0797, 31.8221), label='Amritsar (UTM 43N)'),
    'amsterdam': dict(
        bounds=(609866, 5781372, 651107, 5822517), crs=32631,
        lonlat=(4.6197, 52.1717, 5.2097, 52.5321), label='Amsterdam (UTM 31N)'),
    'anchorage': dict(
        bounds=(331632, 6778200, 356806, 6803336), crs=32606,
        lonlat=(-150.1248, 61.1119, -149.6757, 61.3281), label='Anchorage (UTM 6N)'),
    'andorra': dict(
        bounds=(365829, 4694156, 390341, 4718584), crs=32631,
        lonlat=(1.3699, 42.3919, 1.6631, 42.6081), label='Andorra (UTM 31N)'),
    'ankara': dict(
        bounds=(468087, 4399899, 508344, 4439966), crs=32636,
        lonlat=(32.6275, 39.749, 33.0974, 40.1094), label='Ankara (UTM 36N)'),
    'anshan': dict(
        bounds=(474653, 4531741, 514925, 4571783), crs=32651,
        lonlat=(122.6989, 40.9368, 123.1773, 41.2971), label='Anshan (UTM 51N)'),
    'antananarivo': dict(
        bounds=(744536, 7886473, 785245, 7926952), crs=32738,
        lonlat=(47.3242, -19.0949, 47.7051, -18.7345), label='Antananarivo (UTM 38S)'),
    'apia': dict(
        bounds=(408100, 8457672, 432241, 8481670), crs=32702,
        lonlat=(-171.85, -13.9497, -171.6273, -13.7334), label='Apia (UTM 2S)'),
    'asansol': dict(
        bounds=(478008, 2599436, 518183, 2639352), crs=32645,
        lonlat=(86.7846, 23.5051, 87.1781, 23.8655), label='Asansol (UTM 45N)'),
    'ashgabat': dict(
        bounds=(609329, 4189005, 633777, 4213352), crs=32640,
        lonlat=(58.2462, 37.8419, 58.5204, 38.0581), label='Ashgabat (UTM 40N)'),
    'asmara': dict(
        bounds=(480798, 1683234, 504876, 1707156), crs=32637,
        lonlat=(38.8212, 15.2252, 39.0454, 15.4414), label='Asmara (UTM 37N)'),
    'astana': dict(
        bounds=(657259, 5660362, 682155, 5685197), crs=32642,
        lonlat=(71.2553, 51.073, 71.6002, 51.2892), label='Astana (UTM 42N)'),
    'asuncion': dict(
        bounds=(415040, 7182231, 455349, 7222322), crs=32721,
        lonlat=(-57.8427, -25.4746, -57.4442, -25.1143), label='Asuncion (UTM 21S)'),
    'athens': dict(
        bounds=(719261, 4187142, 760595, 4228319), crs=32634,
        lonlat=(23.5028, 37.8051, 23.96, 38.1655), label='Athens (UTM 34N)'),
    'atlanta': dict(
        bounds=(719891, 3726093, 761056, 3767072), crs=32616,
        lonlat=(-84.6188, 33.6518, -84.185, 34.0121), label='Atlanta (UTM 16N)'),
    'auckland': dict(
        bounds=(279963, 5899165, 321054, 5940082), crs=32760,
        lonlat=(174.5379, -37.0282, 174.9882, -36.6679), label='Auckland (UTM 60S)'),
    'austin': dict(
        bounds=(600497, 3329076, 641072, 3369461), crs=32614,
        lonlat=(-97.9533, 30.0887, -97.5361, 30.4491), label='Austin (UTM 14N)'),
    'baghdad': dict(
        bounds=(413041, 3659116, 473599, 3719404), crs=32638,
        lonlat=(44.0684, 33.0703, 44.7154, 33.6109), label='Baghdad (UTM 38N)'),
    'baku': dict(
        bounds=(382890, 4452234, 423564, 4492756), crs=32639,
        lonlat=(49.6237, 40.217, 50.0969, 40.5774), label='Baku (UTM 39N)'),
    'baltimore': dict(
        bounds=(339660, 4331212, 380534, 4371920), crs=32618,
        lonlat=(-76.8548, 39.1218, -76.3891, 39.4821), label='Baltimore (UTM 18N)'),
    'bamako': dict(
        bounds=(588268, 1378861, 628528, 1418863), crs=32629,
        lonlat=(-8.1866, 12.4718, -7.8173, 12.8321), label='Bamako (UTM 29N)'),
    'bandar_seri_begawan': dict(
        bounds=(258732, 528124, 282880, 552112), crs=32650,
        lonlat=(114.8248, 4.7752, 115.0418, 4.9914), label='Bandar Seri Begawan (UTM 50N)'),
    'bandung': dict(
        bounds=(763604, 9211163, 803958, 9251261), crs=32748,
        lonlat=(107.3866, -7.1283, 107.7496, -6.7679), label='Bandung (UTM 48S)'),
    'bangkok': dict(
        bounds=(633527, 1490743, 694071, 1550916), crs=32647,
        lonlat=(100.2365, 13.4817, 100.7929, 14.0222), label='Bangkok (UTM 47N)'),
    'bangui': dict(
        bounds=(216923, 471093, 241077, 495102), crs=32634,
        lonlat=(18.4499, 4.2585, 18.6667, 4.4748), label='Bangui (UTM 34N)'),
    'banjul': dict(
        bounds=(315570, 1475855, 339806, 1499929), crs=32628,
        lonlat=(-16.7029, 13.3458, -16.4805, 13.562), label='Banjul (UTM 28N)'),
    'baotou': dict(
        bounds=(379850, 4480795, 420544, 4521330), crs=32649,
        lonlat=(109.5826, 40.474, 110.0576, 40.8343), label='Baotou (UTM 49N)'),
    'barcelona': dict(
        bounds=(411238, 4561687, 451774, 4602066), crs=32631,
        lonlat=(1.9413, 41.2051, 2.4216, 41.5654), label='Barcelona (UTM 31N)'),
    'barranquilla': dict(
        bounds=(501594, 1191853, 541718, 1231718), crs=32618,
        lonlat=(-74.9854, 10.7818, -74.6184, 11.1421), label='Barranquilla (UTM 18N)'),
    'basseterre': dict(
        bounds=(518034, 1901013, 542129, 1924967), crs=32620,
        lonlat=(-62.8302, 17.1939, -62.6038, 17.4101), label='Basseterre (UTM 20N)'),
    'baton_rouge': dict(
        bounds=(666334, 3358822, 690821, 3383194), crs=32615,
        lonlat=(-91.2656, 30.3498, -91.0147, 30.5661), label='Baton Rouge (UTM 15N)'),
    'beijing': dict(
        bounds=(417126, 4390121, 477770, 4450525), crs=32650,
        lonlat=(116.0339, 39.6606, 116.7388, 40.2011), label='Beijing (UTM 50N)'),
    'beirut': dict(
        bounds=(711428, 3730550, 752566, 3771505), crs=32636,
        lonlat=(35.2907, 33.6937, 35.7248, 34.0541), label='Beirut (UTM 36N)'),
    'bekasi': dict(
        bounds=(698103, 9292367, 738377, 9332368), crs=32748,
        lonlat=(106.7911, -6.3974, 107.1536, -6.0371), label='Bekasi (UTM 48S)'),
    'belem': dict(
        bounds=(760107, 9819833, 800292, 9859741), crs=32722,
        lonlat=(-48.6622, -1.6282, -48.3017, -1.2679), label='Belem (UTM 22S)'),
    'belgrade': dict(
        bounds=(437506, 4943043, 477940, 4983341), crs=32634,
        lonlat=(20.212, 44.6404, 20.7201, 45.0008), label='Belgrade (UTM 34N)'),
    'belmopan': dict(
        bounds=(299978, 1896223, 324273, 1920370), crs=32616,
        lonlat=(-88.8803, 17.1439, -88.6539, 17.3601), label='Belmopan (UTM 16N)'),
    'belo_horizonte': dict(
        bounds=(583129, 7767632, 643693, 7827851), crs=32723,
        lonlat=(-44.2044, -20.1834, -43.6295, -19.6428), label='Belo Horizonte (UTM 23S)'),
    'bengaluru': dict(
        bounds=(747139, 1405233, 807952, 1465663), crs=32643,
        lonlat=(77.2807, 12.7017, 77.8354, 13.2422), label='Bengaluru (UTM 43N)'),
    'benoni': dict(
        bounds=(612507, 7087111, 653042, 7127429), crs=32735,
        lonlat=(28.1273, -26.3278, 28.5287, -25.9675), label='Benoni (UTM 35S)'),
    'berlin': dict(
        bounds=(370809, 5800056, 411872, 5841020), crs=32633,
        lonlat=(13.1035, 52.3436, 13.6957, 52.7039), label='Berlin (UTM 33N)'),
    'bern': dict(
        bounds=(370961, 5184812, 395529, 5209308), crs=32632,
        lonlat=(7.3087, 46.8086, 7.6252, 47.0248), label='Bern (UTM 32N)'),
    'bhopal': dict(
        bounds=(725990, 2553183, 766805, 2593760), crs=32643,
        lonlat=(77.2119, 23.0718, 77.6042, 23.4321), label='Bhopal (UTM 43N)'),
    'birmingham': dict(
        bounds=(552908, 5794327, 593692, 5835013), crs=32630,
        lonlat=(-2.2178, 52.2967, -1.6261, 52.6571), label='Birmingham (UTM 30N)'),
    'birmingham_us': dict(
        bounds=(496176, 3690067, 536400, 3730093), crs=32616,
        lonlat=(-87.0411, 33.3498, -86.6088, 33.7102), label='Birmingham (UTM 16N)'),
    'bishkek': dict(
        bounds=(453831, 4734965, 478052, 4759093), crs=32643,
        lonlat=(74.4357, 42.7669, 74.7308, 42.9831), label='Bishkek (UTM 43N)'),
    'bissau': dict(
        bounds=(422770, 1299718, 446882, 1323678), crs=32628,
        lonlat=(-15.7088, 11.7569, -15.4879, 11.9731), label='Bissau (UTM 28N)'),
    'bloemfontein': dict(
        bounds=(412952, 6766412, 437188, 6790524), crs=32735,
        lonlat=(26.1062, -29.2281, 26.3537, -29.0119), label='Bloemfontein (UTM 35S)'),
    'bogota': dict(
        bounds=(571361, 478424, 631597, 538255), crs=32618,
        lonlat=(-74.3564, 4.3281, -73.8141, 4.8686), label='Bogota (UTM 18N)'),
    'boise': dict(
        bounds=(550202, 4816586, 574522, 4840821), crs=32611,
        lonlat=(-116.3768, 43.5005, -116.0782, 43.7167), label='Boise (UTM 11N)'),
    'boston': dict(
        bounds=(308670, 4668237, 349811, 4709235), crs=32619,
        lonlat=(-71.3157, 42.1517, -70.8282, 42.5121), label='Boston (UTM 19N)'),
    'brasilia': dict(
        bounds=(166978, 8232833, 207677, 8273299), crs=32723,
        lonlat=(-48.1052, -15.9616, -47.7308, -15.6012), label='Brasilia (UTM 23S)'),
    'bratislava': dict(
        bounds=(645095, 5322806, 669861, 5347498), crs=32633,
        lonlat=(16.9549, 48.0419, 17.279, 48.2581), label='Bratislava (UTM 33N)'),
    'brazzaville': dict(
        bounds=(511327, 9509508, 551436, 9549350), crs=32733,
        lonlat=(15.1021, -4.4374, 15.4634, -4.0771), label='Brazzaville (UTM 33S)'),
    'bridgetown': dict(
        bounds=(204118, 1437798, 228452, 1461979), crs=32621,
        lonlat=(-59.7275, 12.9939, -59.5055, 13.2101), label='Bridgetown (UTM 21N)'),
    'brisbane': dict(
        bounds=(483184, 6943396, 523378, 6983338), crs=32756,
        lonlat=(152.8301, -27.6333, 153.2362, -27.2729), label='Brisbane (UTM 56S)'),
    'brussels': dict(
        bounds=(573379, 5611994, 614279, 5652781), crs=32631,
        lonlat=(4.0461, 50.6551, 4.6167, 51.0154), label='Brussels (UTM 31N)'),
    'bucharest': dict(
        bounds=(407852, 4900412, 448452, 4940887), crs=32635,
        lonlat=(25.8457, 44.2551, 26.3503, 44.6155), label='Bucharest (UTM 35N)'),
    'budapest': dict(
        bounds=(334865, 5242249, 376022, 5283281), crs=32634,
        lonlat=(18.8147, 47.3218, 19.3481, 47.6821), label='Budapest (UTM 34N)'),
    'buenos_aires': dict(
        bounds=(341053, 6139923, 402105, 6200701), crs=32721,
        lonlat=(-58.7278, -34.8708, -58.0711, -34.3303), label='Buenos Aires (UTM 21S)'),
    'buffalo': dict(
        bounds=(652448, 4729389, 693612, 4770423), crs=32617,
        lonlat=(-79.1278, 42.7017, -78.6361, 43.0621), label='Buffalo (UTM 17N)'),
    'bujumbura': dict(
        bounds=(750162, 9614527, 774301, 9638503), crs=32735,
        lonlat=(29.2517, -3.4842, 29.4683, -3.268), label='Bujumbura (UTM 35S)'),
    'bursa': dict(
        bounds=(655517, 4431785, 696615, 4472716), crs=32635,
        lonlat=(28.8321, 40.0218, 29.304, 40.3821), label='Bursa (UTM 35N)'),
    'busan': dict(
        bounds=(480626, 3863817, 520854, 3903808), crs=32652,
        lonlat=(128.7879, 34.9168, 129.2283, 35.2772), label='Busan (UTM 52N)'),
    'cairo': dict(
        bounds=(300456, 3295454, 361588, 3356294), crs=32636,
        lonlat=(30.9358, 29.7816, 31.5603, 30.3222), label='Cairo (UTM 36N)'),
    'cali': dict(
        bounds=(313058, 356194, 353234, 396106), crs=32618,
        lonlat=(-76.6824, 3.2217, -76.3214, 3.5821), label='Cali (UTM 18N)'),
    'campinas': dict(
        bounds=(264007, 7445973, 304717, 7486450), crs=32723,
        lonlat=(-47.2976, -23.0782, -46.9064, -22.7179), label='Campinas (UTM 23S)'),
    'canberra': dict(
        bounds=(681334, 6081236, 705944, 6105736), crs=32755,
        lonlat=(148.9966, -35.3911, 149.2615, -35.1749), label='Canberra (UTM 55S)'),
    'cape_town': dict(
        bounds=(242068, 6223456, 283228, 6264424), crs=32734,
        lonlat=(18.2159, -34.0982, 18.6502, -33.7379), label='Cape Town (UTM 34S)'),
    'caracas': dict(
        bounds=(707577, 1141720, 747969, 1181848), crs=32619,
        lonlat=(-67.1022, 10.3228, -66.7357, 10.6831), label='Caracas (UTM 19N)'),
    'casablanca': dict(
        bounds=(607899, 3698645, 648572, 3739144), crs=32629,
        lonlat=(-7.8346, 33.4217, -7.402, 33.7821), label='Casablanca (UTM 29N)'),
    'castries': dict(
        bounds=(703887, 1536800, 728162, 1560927), crs=32620,
        lonlat=(-61.1114, 13.8939, -60.8886, 14.1101), label='Castries (UTM 20N)'),
    'changchun': dict(
        bounds=(667280, 4839206, 708582, 4880361), crs=32651,
        lonlat=(125.0881, 43.6868, 125.588, 44.0471), label='Changchun (UTM 51N)'),
    'changde': dict(
        bounds=(545879, 3191658, 586245, 3231809), crs=32649,
        lonlat=(111.472, 28.8518, 111.8842, 29.2121), label='Changde (UTM 49N)'),
    'changsha': dict(
        bounds=(672801, 3100860, 713592, 3141450), crs=32649,
        lonlat=(112.7636, 28.0217, 113.1725, 28.3821), label='Changsha (UTM 49N)'),
    'changzhou': dict(
        bounds=(760480, 3499586, 801730, 3540653), crs=32650,
        lonlat=(119.7561, 31.6017, 120.18, 31.9621), label='Changzhou (UTM 50N)'),
    'charleston': dict(
        bounds=(582234, 3616631, 606547, 3640829), crs=32617,
        lonlat=(-80.1207, 32.6843, -79.8635, 32.9005), label='Charleston (UTM 17N)'),
    'charlotte': dict(
        bounds=(503245, 3884000, 527370, 3908018), crs=32617,
        lonlat=(-80.9643, 35.0988, -80.6997, 35.315), label='Charlotte (UTM 17N)'),
    'chattogram': dict(
        bounds=(355965, 2449973, 396410, 2490182), crs=32646,
        lonlat=(91.6032, 22.1518, 91.9928, 22.5121), label='Chattogram (UTM 46N)'),
    'chengdu': dict(
        bounds=(390458, 3373505, 430928, 3413769), crs=32648,
        lonlat=(103.8586, 30.4918, 104.2776, 30.8521), label='Chengdu (UTM 48N)'),
    'chennai': dict(
        bounds=(391542, 1417460, 451868, 1477409), crs=32644,
        lonlat=(80.0006, 12.8217, 80.5555, 13.3622), label='Chennai (UTM 44N)'),
    'chicago': dict(
        bounds=(407052, 4601187, 467811, 4661725), crs=32616,
        lonlat=(-88.1147, 41.5617, -87.3893, 42.1022), label='Chicago (UTM 16N)'),
    'chifeng': dict(
        bounds=(640161, 4661374, 681238, 4702301), crs=32650,
        lonlat=(118.7045, 42.0918, 119.1915, 42.4521), label='Chifeng (UTM 50N)'),
    'chisinau': dict(
        bounds=(628908, 5195109, 653575, 5219705), crs=32635,
        lonlat=(28.6992, 46.8969, 29.0162, 47.1131), label='Chisinau (UTM 35N)'),
    'chongqing': dict(
        bounds=(623881, 3241540, 684916, 3302267), crs=32648,
        lonlat=(106.2823, 29.2967, 106.9038, 29.8372), label='Chongqing (UTM 48N)'),
    'cincinnati': dict(
        bounds=(698953, 4317503, 740237, 4358621), crs=32616,
        lonlat=(-84.6913, 38.9837, -84.2265, 39.344), label='Cincinnati (UTM 16N)'),
    'ciudad_juarez': dict(
        bounds=(338210, 3487068, 378907, 3527569), crs=32613,
        lonlat=(-106.7038, 31.5121, -106.2802, 31.8725), label='Ciudad Juarez (UTM 13N)'),
    'cleveland': dict(
        bounds=(421511, 4571251, 461988, 4611575), crs=32617,
        lonlat=(-81.9374, 41.2918, -81.4565, 41.6521), label='Cleveland (UTM 17N)'),
    'coimbatore': dict(
        bounds=(692670, 1196823, 733053, 1236954), crs=32643,
        lonlat=(76.7645, 10.8217, 77.1316, 11.1821), label='Coimbatore (UTM 43N)'),
    'colombo': dict(
        bounds=(361742, 754404, 385875, 778367), crs=32644,
        lonlat=(79.7488, 6.8239, 79.9667, 7.0401), label='Colombo (UTM 44N)'),
    'colorado_springs': dict(
        bounds=(505995, 4289580, 530142, 4313627), crs=32613,
        lonlat=(-104.9308, 38.7549, -104.6531, 38.9711), label='Colorado Springs (UTM 13N)'),
    'columbus': dict(
        bounds=(309338, 4407224, 350391, 4448127), crs=32617,
        lonlat=(-83.2271, 39.8017, -82.7568, 40.1621), label='Columbus (UTM 17N)'),
    'conakry': dict(
        bounds=(624517, 1034113, 664781, 1074111), crs=32628,
        lonlat=(-13.8649, 9.3533, -13.4995, 9.7136), label='Conakry (UTM 28N)'),
    'cordoba': dict(
        bounds=(367094, 6505472, 407665, 6545852), crs=32720,
        lonlat=(-64.3953, -31.5782, -63.9731, -31.2178), label='Cordoba (UTM 20S)'),
    'cotonou': dict(
        bounds=(434662, 695697, 458738, 719631), crs=32631,
        lonlat=(2.4093, 6.2938, 2.6268, 6.5101), label='Cotonou (UTM 31N)'),
    'curitiba': dict(
        bounds=(648485, 7167477, 689113, 7207890), crs=32722,
        lonlat=(-49.5214, -25.5982, -49.1225, -25.2379), label='Curitiba (UTM 22S)'),
    'daegu': dict(
        bounds=(444146, 3949424, 484454, 3989548), crs=32652,
        lonlat=(128.3827, 35.6886, 128.8274, 36.0489), label='Daegu (UTM 52N)'),
    'daejeon': dict(
        bounds=(338031, 4002244, 378838, 4042879), crs=32652,
        lonlat=(127.1994, 36.1573, 127.6468, 36.5177), label='Daejeon (UTM 52N)'),
    'dakar': dict(
        bounds=(213183, 1608421, 253763, 1648755), crs=32628,
        lonlat=(-17.6614, 14.5376, -17.2888, 14.898), label='Dakar (UTM 28N)'),
    'dalian': dict(
        bounds=(360631, 4289053, 401387, 4329653), crs=32651,
        lonlat=(121.3963, 38.7446, 121.8595, 39.105), label='Dalian (UTM 51N)'),
    'dallas': dict(
        bounds=(681573, 3613246, 722538, 3654031), crs=32614,
        lonlat=(-97.0564, 32.6418, -96.6276, 33.0022), label='Dallas (UTM 14N)'),
    'damascus': dict(
        bounds=(228360, 3689721, 269553, 3730742), crs=32637,
        lonlat=(36.082, 33.3218, 36.5141, 33.6822), label='Damascus (UTM 37N)'),
    'daqing': dict(
        bounds=(632556, 5140145, 673740, 5181208), crs=32651,
        lonlat=(124.7359, 46.4017, 125.2602, 46.7621), label='Daqing (UTM 51N)'),
    'dar_es_salaam': dict(
        bounds=(509378, 9228638, 549511, 9268490), crs=32737,
        lonlat=(39.0849, -6.9782, 39.4479, -6.6179), label='Dar es Salaam (UTM 37S)'),
    'datong': dict(
        bounds=(675394, 4418894, 716581, 4459928), crs=32649,
        lonlat=(113.0626, 39.9018, 113.5335, 40.2621), label='Datong (UTM 49N)'),
    'davao': dict(
        bounds=(770127, 766906, 810490, 807006), crs=32651,
        lonlat=(125.4465, 6.9318, 125.8096, 7.2921), label='Davao (UTM 51N)'),
    'dayton': dict(
        bounds=(727319, 4391643, 752164, 4416397), crs=32616,
        lonlat=(-84.3425, 39.6442, -84.0613, 39.8604), label='Dayton (UTM 16N)'),
    'delhi': dict(
        bounds=(687131, 3143201, 748470, 3204231), crs=32643,
        lonlat=(76.92, 28.4017, 77.5361, 28.9422), label='Delhi (UTM 43N)'),
    'denver': dict(
        bounds=(481075, 4379034, 521339, 4419052), crs=32613,
        lonlat=(-105.2203, 39.561, -104.7516, 39.9213), label='Denver (UTM 13N)'),
    'des_moines': dict(
        bounds=(436165, 4591254, 460428, 4615430), crs=32615,
        lonlat=(-93.7645, 41.4719, -93.4755, 41.6881), label='Des Moines (UTM 15N)'),
    'detroit': dict(
        bounds=(307844, 4668255, 348981, 4709258), crs=32617,
        lonlat=(-83.3257, 42.1517, -82.8383, 42.5121), label='Detroit (UTM 17N)'),
    'dhaka': dict(
        bounds=(204887, 2595731, 266207, 2656721), crs=32646,
        lonlat=(90.1114, 23.4547, 90.7019, 23.9953), label='Dhaka (UTM 46N)'),
    'dhanbad': dict(
        bounds=(420546, 2612453, 460836, 2652508), crs=32645,
        lonlat=(86.2211, 23.6222, 86.615, 23.9825), label='Dhanbad (UTM 45N)'),
    'dili': dict(
        bounds=(771823, 9040861, 796073, 9064948), crs=32751,
        lonlat=(125.4701, -8.6675, 125.6888, -8.4513), label='Dili (UTM 51S)'),
    'djibouti': dict(
        bounds=(285941, 1270392, 310179, 1294467), crs=32638,
        lonlat=(43.0376, 11.4869, 43.2584, 11.7031), label='Djibouti (UTM 38N)'),
    'doha': dict(
        bounds=(533545, 2776771, 573824, 2816829), crs=32639,
        lonlat=(51.3337, 25.1064, 51.7322, 25.4667), label='Doha (UTM 39N)'),
    'dongguan': dict(
        bounds=(760634, 2531465, 801531, 2572135), crs=32649,
        lonlat=(113.547, 22.8707, 113.9386, 23.231), label='Dongguan (UTM 49N)'),
    'douala': dict(
        bounds=(558528, 429125, 598669, 468989), crs=32632,
        lonlat=(9.5274, 3.8822, 9.8887, 4.2425), label='Douala (UTM 32N)'),
    'dubai': dict(
        bounds=(306190, 2771548, 346840, 2811971), crs=32640,
        lonlat=(55.0788, 25.0518, 55.4772, 25.4121), label='Dubai (UTM 40N)'),
    'dublin': dict(
        bounds=(662270, 5892289, 703987, 5933916), crs=32629,
        lonlat=(-6.5526, 53.1548, -5.9491, 53.5152), label='Dublin (UTM 29N)'),
    'duisburg': dict(
        bounds=(322810, 5679431, 364216, 5720739), crs=32632,
        lonlat=(6.461, 51.2498, 7.039, 51.6102), label='Duisburg (UTM 32N)'),
    'durban': dict(
        bounds=(284242, 6674333, 325086, 6714975), crs=32736,
        lonlat=(30.7703, -30.0432, 31.1858, -29.6829), label='Durban (UTM 36S)'),
    'dushanbe': dict(
        bounds=(460128, 4247967, 500376, 4288046), crs=32642,
        lonlat=(68.5435, 38.3799, 69.0043, 38.7402), label='Dushanbe (UTM 42N)'),
    'dusseldorf': dict(
        bounds=(324200, 5656072, 365588, 5697359), crs=32632,
        lonlat=(6.4923, 51.0402, 7.0677, 51.4006), label='Dusseldorf (UTM 32N)'),
    'el_giza': dict(
        bounds=(305007, 3301002, 345786, 3341582), crs=32636,
        lonlat=(30.9819, 29.8298, 31.3981, 30.1902), label='El Giza (UTM 36N)'),
    'el_paso': dict(
        bounds=(344623, 3505113, 369037, 3529415), crs=32613,
        lonlat=(-106.6391, 31.6738, -106.3848, 31.89), label='El Paso (UTM 13N)'),
    'essen': dict(
        bounds=(341475, 5681192, 382733, 5722355), crs=32632,
        lonlat=(6.7275, 51.2698, 7.3057, 51.6302), label='Essen (UTM 32N)'),
    'faisalabad': dict(
        bounds=(299691, 3456498, 340530, 3497144), crs=32643,
        lonlat=(72.8969, 31.2317, 73.3192, 31.5921), label='Faisalabad (UTM 43N)'),
    'faridabad': dict(
        bounds=(706276, 3127266, 747192, 3167983), crs=32643,
        lonlat=(77.1098, 28.2551, 77.5196, 28.6155), label='Faridabad (UTM 43N)'),
    'florence': dict(
        bounds=(660489, 4829369, 701749, 4870490), crs=32632,
        lonlat=(11.0004, 43.5998, 11.4996, 43.9602), label='Florence (UTM 32N)'),
    'fort_lauderdale': dict(
        bounds=(565625, 2870965, 606016, 2911135), crs=32617,
        lonlat=(-80.3425, 25.9559, -79.9411, 26.3162), label='Fort Lauderdale (UTM 17N)'),
    'fortaleza': dict(
        bounds=(526366, 9565774, 566482, 9605631), crs=32724,
        lonlat=(-38.7625, -3.9283, -38.4014, -3.5679), label='Fortaleza (UTM 24S)'),
    'frankfurt': dict(
        bounds=(456505, 5529715, 496858, 5569961), crs=32632,
        lonlat=(8.3941, 49.9198, 8.9559, 50.2802), label='Frankfurt (UTM 32N)'),
    'freetown': dict(
        bounds=(682082, 924900, 706262, 948933), crs=32628,
        lonlat=(-13.3455, 8.3638, -13.1269, 8.5801), label='Freetown (UTM 28N)'),
    'fresno': dict(
        bounds=(240018, 4058135, 264806, 4082824), crs=32611,
        lonlat=(-119.9079, 36.6396, -119.6381, 36.8558), label='Fresno (UTM 11N)'),
    'ft_worth': dict(
        bounds=(635188, 3603410, 675957, 3643998), crs=32614,
        lonlat=(-97.5542, 32.5598, -97.1258, 32.9202), label='Ft. Worth (UTM 14N)'),
    'fukuoka': dict(
        bounds=(610350, 3698130, 651035, 3738627), crs=32652,
        lonlat=(130.1918, 33.4168, 130.6244, 33.7771), label='Fukuoka (UTM 52N)'),
    'funafuti': dict(
        bounds=(731906, 9045849, 756129, 9069917), crs=32760,
        lonlat=(179.1073, -8.6248, 179.326, -8.4085), label='Funafuti (UTM 60S)'),
    'fushun': dict(
        bounds=(551814, 4615237, 592377, 4655646), crs=32651,
        lonlat=(123.6261, 41.6872, 124.11, 42.0475), label='Fushun (UTM 51N)'),
    'fuzhou': dict(
        bounds=(709474, 2866489, 750321, 2907119), crs=32650,
        lonlat=(119.0975, 25.9018, 119.4987, 26.2621), label='Fuzhou (UTM 50N)'),
    'gaborone': dict(
        bounds=(377745, 7261708, 402012, 7285840), crs=32735,
        lonlat=(25.793, -24.7544, 26.0309, -24.5382), label='Gaborone (UTM 35S)'),
    'ganzhou': dict(
        bounds=(274265, 2848170, 315034, 2888725), crs=32650,
        lonlat=(114.7497, 25.7398, 115.1504, 26.1002), label='Ganzhou (UTM 50N)'),
    'geneva': dict(
        bounds=(258522, 5100644, 300130, 5142135), crs=32632,
        lonlat=(5.8797, 46.0298, 6.4004, 46.3902), label='Geneva (UTM 32N)'),
    'george_town': dict(
        bounds=(627192, 578580, 667393, 618518), crs=32647,
        lonlat=(100.1484, 5.2334, 100.5104, 5.5938), label='George Town (UTM 47N)'),
    'georgetown': dict(
        bounds=(358974, 740037, 383092, 764000), crs=32621,
        lonlat=(-58.2759, 6.6939, -58.0582, 6.9101), label='Georgetown (UTM 21N)'),
    'ghaziabad': dict(
        bounds=(714740, 3152583, 755695, 3193339), crs=32643,
        lonlat=(77.2011, 28.4821, 77.6118, 28.8425), label='Ghaziabad (UTM 43N)'),
    'goiania': dict(
        bounds=(660821, 8130734, 701289, 8170959), crs=32722,
        lonlat=(-49.4901, -16.8983, -49.1138, -16.5379), label='Goiania (UTM 22S)'),
    'grand_rapids': dict(
        bounds=(596256, 4745458, 620736, 4769848), crs=32616,
        lonlat=(-85.8177, 42.8556, -85.5222, 43.0718), label='Grand Rapids (UTM 16N)'),
    'guadalajara': dict(
        bounds=(653500, 2266599, 694043, 2306898), crs=32613,
        lonlat=(-103.5246, 20.4918, -103.1394, 20.8521), label='Guadalajara (UTM 13N)'),
    'guangzhou': dict(
        bounds=(707331, 2531304, 768503, 2592133), crs=32649,
        lonlat=(113.0291, 22.8767, 113.617, 23.4172), label='Guangzhou (UTM 49N)'),
    'guatemala': dict(
        bounds=(745924, 1597935, 786496, 1638266), crs=32615,
        lonlat=(-90.7151, 14.4429, -90.3427, 14.8033), label='Guatemala (UTM 15N)'),
    'guayaquil': dict(
        bounds=(599809, 9734850, 639941, 9774722), crs=32717,
        lonlat=(-80.1023, -2.3983, -79.7417, -2.0379), label='Guayaquil (UTM 17S)'),
    'guiyang': dict(
        bounds=(650796, 2921072, 691473, 2961537), crs=32648,
        lonlat=(106.5166, 26.4018, 106.9196, 26.7622), label='Guiyang (UTM 48N)'),
    'gujranwala': dict(
        bounds=(402709, 3538622, 443152, 3578877), crs=32643,
        lonlat=(73.9702, 31.9822, 74.3959, 32.3426), label='Gujranwala (UTM 43N)'),
    'gwangju': dict(
        bounds=(288997, 3873833, 329986, 3914655), crs=32652,
        lonlat=(126.6881, 34.9927, 127.1289, 35.3531), label='Gwangju (UTM 52N)'),
    'haikou': dict(
        bounds=(408728, 2197147, 449006, 2237195), crs=32649,
        lonlat=(110.1282, 19.8698, 110.5118, 20.2302), label='Haikou (UTM 49N)'),
    'haiphong': dict(
        bounds=(654376, 2284320, 694914, 2324625), crs=32648,
        lonlat=(106.4854, 20.6518, 106.8709, 21.0121), label='Haiphong (UTM 48N)'),
    'hamburg': dict(
        bounds=(545835, 5913856, 586577, 5954513), crs=32632,
        lonlat=(9.6948, 53.3718, 10.3013, 53.7322), label='Hamburg (UTM 32N)'),
    'hamilton': dict(
        bounds=(319771, 3562266, 344259, 3586637), crs=32620,
        lonlat=(-64.9118, 32.1861, -64.656, 32.4023), label='Hamilton (UTM 20N)'),
    'handan': dict(
        bounds=(253716, 4030957, 294918, 4072000), crs=32650,
        lonlat=(114.2537, 36.4017, 114.7024, 36.7621), label='Handan (UTM 50N)'),
    'hangzhou': dict(
        bounds=(206893, 3329632, 248049, 3370594), crs=32651,
        lonlat=(119.9595, 30.0717, 120.3767, 30.4321), label='Hangzhou (UTM 51N)'),
    'hanoi': dict(
        bounds=(567976, 2306250, 608309, 2346353), crs=32648,
        lonlat=(105.655, 20.8551, 106.0411, 21.2155), label='Hanoi (UTM 48N)'),
    'haora': dict(
        bounds=(616505, 2477565, 656990, 2517825), crs=32645,
        lonlat=(88.1348, 22.4002, 88.5251, 22.7606), label='Haora (UTM 45N)'),
    'harare': dict(
        bounds=(272260, 8008941, 312806, 8049241), crs=32736,
        lonlat=(30.8535, -17.996, 31.232, -17.6357), label='Harare (UTM 36S)'),
    'harbin': dict(
        bounds=(296334, 5048594, 337683, 5089816), crs=32652,
        lonlat=(126.3898, 45.5717, 126.9063, 45.9321), label='Harbin (UTM 52N)'),
    'hargeysa': dict(
        bounds=(385357, 1044921, 409480, 1068892), crs=32638,
        lonlat=(43.9557, 9.4519, 44.1749, 9.6681), label='Hargeysa (UTM 38N)'),
    'hartford': dict(
        bounds=(680313, 4614742, 705057, 4639397), crs=32618,
        lonlat=(-72.8269, 41.6639, -72.537, 41.8801), label='Hartford (UTM 18N)'),
    'havana': dict(
        bounds=(339845, 2538872, 380350, 2579153), crs=32617,
        lonlat=(-82.5621, 22.9537, -82.1702, 23.3141), label='Havana (UTM 17N)'),
    'hechi': dict(
        bounds=(184311, 2713970, 225314, 2754753), crs=32649,
        lonlat=(107.8847, 24.5151, 108.2813, 24.8754), label='Hechi (UTM 49N)'),
    'hefei': dict(
        bounds=(506222, 3504060, 546466, 3544110), crs=32650,
        lonlat=(117.0659, 31.6718, 117.4902, 32.0322), label='Hefei (UTM 50N)'),
    'helsinki': dict(
        bounds=(364464, 6652337, 405907, 6693720), crs=32635,
        lonlat=(24.5699, 59.9973, 25.2945, 60.3577), label='Helsinki (UTM 35N)'),
    'heze': dict(
        bounds=(338349, 3879587, 379125, 3920188), crs=32650,
        lonlat=(115.2275, 35.0517, 115.6687, 35.4121), label='Heze (UTM 50N)'),
    'hiroshima': dict(
        bounds=(244108, 3787874, 285274, 3828865), crs=32653,
        lonlat=(132.2226, 34.2096, 132.6593, 34.57), label='Hiroshima (UTM 53N)'),
    'ho_chi_minh_city': dict(
        bounds=(654895, 1162339, 715400, 1222459), crs=32648,
        lonlat=(106.418, 10.5117, 106.9682, 11.0522), label='Ho Chi Minh City (UTM 48N)'),
    'hohhot': dict(
        bounds=(535322, 4499068, 575770, 4539377), crs=32649,
        lonlat=(111.42, 40.6417, 111.8961, 41.0021), label='Hohhot (UTM 49N)'),
    'hong_kong': dict(
        bounds=(179045, 2439044, 240396, 2500050), crs=32650,
        lonlat=(113.8909, 22.0367, 114.4752, 22.5772), label='Hong Kong (UTM 50N)'),
    'honiara': dict(
        bounds=(592210, 8944593, 616342, 8968565), crs=32757,
        lonlat=(159.8402, -9.5461, 160.0594, -9.3299), label='Honiara (UTM 57S)'),
    'honolulu': dict(
        bounds=(606129, 2344703, 630377, 2368809), crs=32604,
        lonlat=(-157.976, 21.2007, -157.7439, 21.4169), label='Honolulu (UTM 4N)'),
    'houston': dict(
        bounds=(253177, 3280987, 294142, 3321755), crs=32615,
        lonlat=(-95.5496, 29.6417, -95.1342, 30.0021), label='Houston (UTM 15N)'),
    'huainan': dict(
        bounds=(477828, 3590505, 518046, 3630481), crs=32650,
        lonlat=(116.7641, 32.4517, 117.192, 32.8121), label='Huainan (UTM 50N)'),
    'huaiyin': dict(
        bounds=(667792, 3697299, 708719, 3738042), crs=32650,
        lonlat=(118.8118, 33.4018, 119.2443, 33.7621), label='Huaiyin (UTM 50N)'),
    'huzhou': dict(
        bounds=(201898, 3398573, 243102, 3439583), crs=32651,
        lonlat=(119.8881, 30.6921, 120.308, 31.0525), label='Huzhou (UTM 51N)'),
    'hyderabad': dict(
        bounds=(201516, 1895496, 262511, 1956133), crs=32644,
        lonlat=(78.1948, 17.1317, 78.7612, 17.6722), label='Hyderabad (UTM 44N)'),
    'hyderabad_pk': dict(
        bounds=(416743, 2787361, 457059, 2827449), crs=32642,
        lonlat=(68.1736, 25.2018, 68.5725, 25.5621), label='Hyderabad (UTM 42N)'),
    'ibadan': dict(
        bounds=(582329, 796125, 622522, 836055), crs=32631,
        lonlat=(3.7463, 7.2018, 4.1097, 7.5622), label='Ibadan (UTM 31N)'),
    'incheon': dict(
        bounds=(270717, 4130053, 311878, 4171052), crs=32652,
        lonlat=(126.4132, 37.2979, 126.8673, 37.6583), label='Incheon (UTM 52N)'),
    'indianapolis': dict(
        bounds=(550720, 4380401, 591251, 4420761), crs=32616,
        lonlat=(-86.4064, 39.5718, -85.9376, 39.9321), label='Indianapolis (UTM 16N)'),
    'indore': dict(
        bounds=(568480, 2492398, 608837, 2532530), crs=32643,
        lonlat=(75.6677, 22.5368, 76.0584, 22.8972), label='Indore (UTM 43N)'),
    'irvine': dict(
        bounds=(402794, 3706910, 443261, 3747192), crs=32611,
        lonlat=(-118.0465, 33.5002, -117.6134, 33.8606), label='Irvine (UTM 11N)'),
    'isfahan': dict(
        bounds=(545274, 3598379, 585672, 3638585), crs=32639,
        lonlat=(51.484, 32.5218, 51.9122, 32.8821), label='Isfahan (UTM 39N)'),
    'islamabad': dict(
        bounds=(317636, 3718422, 342151, 3742837), crs=32643,
        lonlat=(73.0347, 33.5938, 73.2946, 33.8101), label='Islamabad (UTM 43N)'),
    'istanbul': dict(
        bounds=(637926, 4521939, 699550, 4583334), crs=32635,
        lonlat=(28.6494, 40.8367, 29.3668, 41.3772), label='Istanbul (UTM 35N)'),
    'izmir': dict(
        bounds=(492983, 4234430, 533237, 4274489), crs=32635,
        lonlat=(26.9198, 38.2579, 27.3799, 38.6183), label='Izmir (UTM 35N)'),
    'jabalpur': dict(
        bounds=(372616, 2543419, 413027, 2583609), crs=32644,
        lonlat=(79.7571, 22.9968, 80.1491, 23.3572), label='Jabalpur (UTM 44N)'),
    'jacksonville': dict(
        bounds=(423284, 3343722, 447506, 3367823), crs=32617,
        lonlat=(-81.7972, 30.2239, -81.5467, 30.4401), label='Jacksonville (UTM 17N)'),
    'jaipur': dict(
        bounds=(560070, 2958103, 600447, 2998279), crs=32643,
        lonlat=(75.606, 26.7429, 76.0101, 27.1033), label='Jaipur (UTM 43N)'),
    'jakarta': dict(
        bounds=(672034, 9287381, 732421, 9347366), crs=32748,
        lonlat=(106.5556, -6.4427, 107.0993, -5.9022), label='Jakarta (UTM 48S)'),
    'jamshedpur': dict(
        bounds=(397232, 2500398, 437574, 2540515), crs=32645,
        lonlat=(86.0001, 22.6093, 86.391, 22.9697), label='Jamshedpur (UTM 45N)'),
    'jeddah': dict(
        bounds=(502441, 2359633, 542608, 2399567), crs=32637,
        lonlat=(39.0236, 21.3387, 39.4109, 21.699), label='Jeddah (UTM 37N)'),
    'jerusalem': dict(
        bounds=(688512, 3497627, 729471, 3538401), crs=32636,
        lonlat=(34.9947, 31.5982, 35.4186, 31.9586), label='Jerusalem (UTM 36N)'),
    'jianmen': dict(
        bounds=(686367, 3372680, 727284, 3413404), crs=32649,
        lonlat=(112.9486, 30.4718, 113.3675, 30.8322), label='Jianmen (UTM 49N)'),
    'jilin': dict(
        bounds=(282191, 4837766, 323550, 4878987), crs=32652,
        lonlat=(126.2982, 43.6717, 126.798, 44.0321), label='Jilin (UTM 52N)'),
    'jinan': dict(
        bounds=(479258, 4039042, 519497, 4079045), crs=32650,
        lonlat=(116.7684, 36.4967, 117.2177, 36.8571), label='Jinan (UTM 50N)'),
    'jinxi': dict(
        bounds=(296018, 4493055, 337169, 4534060), crs=32651,
        lonlat=(120.5902, 40.5721, 121.0659, 40.9325), label='Jinxi (UTM 51N)'),
    'johannesburg': dict(
        bounds=(582550, 7085160, 622994, 7125394), crs=32735,
        lonlat=(27.8273, -26.3483, 28.2288, -25.9879), label='Johannesburg (UTM 35S)'),
    'kabul': dict(
        bounds=(496560, 3799689, 536792, 3839715), crs=32642,
        lonlat=(68.9626, 34.3385, 69.4, 34.6988), label='Kabul (UTM 42N)'),
    'kaduna': dict(
        bounds=(308909, 1143530, 349223, 1183581), crs=32632,
        lonlat=(7.2548, 10.3418, 7.6213, 10.7021), label='Kaduna (UTM 32N)'),
    'kalyan': dict(
        bounds=(286323, 2109400, 326875, 2149709), crs=32643,
        lonlat=(72.9693, 19.0701, 73.351, 19.4304), label='Kalyan (UTM 43N)'),
    'kampala': dict(
        bounds=(433368, 15297, 473473, 55135), crs=32636,
        lonlat=(32.4012, 0.1384, 32.7616, 0.4988), label='Kampala (UTM 36N)'),
    'kano': dict(
        bounds=(427440, 1306855, 467614, 1346779), crs=32632,
        lonlat=(8.3339, 11.8217, 8.7023, 12.1821), label='Kano (UTM 32N)'),
    'kanpur': dict(
        bounds=(411821, 2906981, 452157, 2947099), crs=32644,
        lonlat=(80.1168, 26.2818, 80.5193, 26.6421), label='Kanpur (UTM 44N)'),
    'kansas_city': dict(
        bounds=(340662, 4309784, 381524, 4350481), crs=32615,
        lonlat=(-94.8382, 38.9289, -94.3738, 39.2892), label='Kansas City (UTM 15N)'),
    'kaohsiung': dict(
        bounds=(198584, 2485191, 239465, 2525855), crs=32651,
        lonlat=(120.0714, 22.4531, 120.4618, 22.8135), label='Kaohsiung (UTM 51N)'),
    'karachi': dict(
        bounds=(266131, 2721927, 327224, 2782688), crs=32642,
        lonlat=(66.6902, 24.6017, 67.286, 25.1422), label='Karachi (UTM 42N)'),
    'karaj': dict(
        bounds=(476990, 3942035, 517224, 3982037), crs=32639,
        lonlat=(50.7459, 35.6221, 51.1902, 35.9825), label='Karaj (UTM 39N)'),
    'kathmandu': dict(
        bounds=(321640, 3055032, 346046, 3079316), crs=32645,
        lonlat=(85.1926, 27.6105, 85.4368, 27.8267), label='Kathmandu (UTM 45N)'),
    'katowice': dict(
        bounds=(338181, 5548929, 379421, 5590065), crs=32634,
        lonlat=(18.7382, 50.0802, 19.3019, 50.4406), label='Katowice (UTM 34N)'),
    'kawasaki': dict(
        bounds=(362212, 3912364, 402884, 3952866), crs=32654,
        lonlat=(139.4836, 35.3498, 139.9264, 35.7102), label='Kawasaki (UTM 54N)'),
    'kharkiv': dict(
        bounds=(281905, 5521737, 323547, 5563279), crs=32637,
        lonlat=(35.9678, 49.8217, 36.5284, 50.1821), label='Kharkiv (UTM 37N)'),
    'khartoum': dict(
        bounds=(429739, 1703672, 469933, 1743625), crs=32636,
        lonlat=(32.3452, 15.4098, 32.7193, 15.7702), label='Khartoum (UTM 36N)'),
    'khulna': dict(
        bounds=(742121, 2508011, 782968, 2548620), crs=32645,
        lonlat=(89.3625, 22.6618, 89.7536, 23.0221), label='Khulna (UTM 45N)'),
    'kiev': dict(
        bounds=(302677, 5569315, 344192, 5610726), crs=32636,
        lonlat=(30.2318, 50.2551, 30.7976, 50.6155), label='Kiev (UTM 36N)'),
    'kigali': dict(
        bounds=(160669, 9772004, 184809, 9795986), crs=32736,
        lonlat=(29.9504, -2.0598, 30.1668, -1.8435), label='Kigali (UTM 36S)'),
    'kingston': dict(
        bounds=(300686, 1976468, 324989, 2000626), crs=32618,
        lonlat=(-76.8811, 17.869, -76.6538, 18.0852), label='Kingston (UTM 18N)'),
    'kingstown': dict(
        bounds=(681680, 1442182, 705931, 1466272), crs=32620,
        lonlat=(-61.3231, 13.0402, -61.101, 13.2564), label='Kingstown (UTM 20N)'),
    'kinshasa': dict(
        bounds=(504659, 9491745, 564835, 9551517), crs=32733,
        lonlat=(15.042, -4.598, 15.5841, -4.0575), label='Kinshasa (UTM 33S)'),
    'knoxville': dict(
        bounds=(224266, 3972217, 249077, 3996928), crs=32617,
        lonlat=(-84.0536, 35.8619, -83.7865, 36.0781), label='Knoxville (UTM 17N)'),
    'kobe': dict(
        bounds=(495492, 3817575, 535721, 3857609), crs=32653,
        lonlat=(134.9509, 34.4998, 135.3891, 34.8602), label='Kobe (UTM 53N)'),
    'kobenhavn': dict(
        bounds=(325801, 6152507, 367385, 6194016), crs=32633,
        lonlat=(12.242, 55.5003, 12.8811, 55.8607), label='Kobenhavn (UTM 33N)'),
    'kochi': dict(
        bounds=(613806, 1087545, 654062, 1127539), crs=32643,
        lonlat=(76.039, 9.8368, 76.4049, 10.1971), label='Kochi (UTM 43N)'),
    'kolkata': dict(
        bounds=(605761, 2458270, 666482, 2518654), crs=32645,
        lonlat=(88.0302, 22.2266, 88.6153, 22.7672), label='Kolkata (UTM 45N)'),
    'kuala_lumpur': dict(
        bounds=(779761, 330632, 820007, 370617), crs=32647,
        lonlat=(101.5176, 2.9884, 101.8785, 3.3488), label='Kuala Lumpur (UTM 47N)'),
    'kumasi': dict(
        bounds=(631106, 719935, 671336, 759887), crs=32630,
        lonlat=(-1.8134, 6.5118, -1.4505, 6.8721), label='Kumasi (UTM 30N)'),
    'kunming': dict(
        bounds=(245341, 2754629, 286177, 2795247), crs=32648,
        lonlat=(102.4791, 24.8917, 102.877, 25.2521), label='Kunming (UTM 48N)'),
    'kuwait_city': dict(
        bounds=(768368, 3232379, 809545, 3273352), crs=32638,
        lonlat=(47.7696, 29.1915, 48.1831, 29.5518), label='Kuwait City (UTM 38N)'),
    'kyoto': dict(
        bounds=(548056, 3856736, 588501, 3896994), crs=32653,
        lonlat=(135.528, 34.8518, 135.9681, 35.2121), label='Kyoto (UTM 53N)'),
    'la_paz': dict(
        bounds=(570395, 8155978, 610673, 8196018), crs=32719,
        lonlat=(-68.3398, -16.6762, -67.964, -16.3158), label='La Paz (UTM 19S)'),
    'lagos': dict(
        bounds=(512997, 682540, 573197, 742348), crs=32631,
        lonlat=(3.1176, 6.1749, 3.6616, 6.7155), label='Lagos (UTM 31N)'),
    'lahore': dict(
        bounds=(407765, 3461968, 468328, 3522246), crs=32643,
        lonlat=(74.0309, 31.2916, 74.6653, 31.8322), label='Lahore (UTM 43N)'),
    'lanzhou': dict(
        bounds=(370662, 3970845, 411302, 4011321), crs=32648,
        lonlat=(103.5672, 35.8778, 104.0129, 36.2382), label='Lanzhou (UTM 48N)'),
    'las_vegas': dict(
        bounds=(639430, 3988594, 680320, 4029300), crs=32611,
        lonlat=(-115.4453, 36.0318, -114.9986, 36.3921), label='Las Vegas (UTM 11N)'),
    'leeds': dict(
        bounds=(572914, 5945357, 613897, 5986242), crs=32630,
        lonlat=(-1.8873, 53.6518, -1.2767, 54.0121), label='Leeds (UTM 30N)'),
    'leon': dict(
        bounds=(198953, 2321072, 239783, 2361665), crs=32614,
        lonlat=(-101.8952, 20.9718, -101.5088, 21.3321), label='Leon (UTM 14N)'),
    'libreville': dict(
        bounds=(538934, 30651, 562993, 54549), crs=32632,
        lonlat=(9.3499, 0.2773, 9.5661, 0.4935), label='Libreville (UTM 32N)'),
    'lilongwe': dict(
        bounds=(572532, 8441982, 596675, 8465974), crs=32736,
        lonlat=(33.6719, -14.0914, 33.8947, -13.8752), label='Lilongwe (UTM 36S)'),
    'lima': dict(
        bounds=(246260, 8637372, 306899, 8697626), crs=32718,
        lonlat=(-77.3284, -12.3163, -76.7757, -11.7758), label='Lima (UTM 18S)'),
    'linyi': dict(
        bounds=(600767, 3862713, 641450, 3903208), crs=32650,
        lonlat=(118.1078, 34.9018, 118.5482, 35.2621), label='Linyi (UTM 50N)'),
    'lisbon': dict(
        bounds=(467077, 4266231, 507329, 4306280), crs=32629,
        lonlat=(-9.3778, 38.5445, -8.9159, 38.9048), label='Lisbon (UTM 29N)'),
    'liupanshui': dict(
        bounds=(463093, 2921777, 503281, 2961746), crs=32648,
        lonlat=(104.6299, 26.4162, 105.0329, 26.7766), label='Liupanshui (UTM 48N)'),
    'liuzhou': dict(
        bounds=(301835, 2666372, 342467, 2706782), crs=32649,
        lonlat=(109.0504, 24.1018, 109.4457, 24.4621), label='Liuzhou (UTM 49N)'),
    'ljubljana': dict(
        bounds=(450334, 5088234, 474584, 5112402), crs=32633,
        lonlat=(14.3592, 45.9472, 14.6708, 46.1634), label='Ljubljana (UTM 33N)'),
    'lome': dict(
        bounds=(282984, 658342, 323232, 698334), crs=32631,
        lonlat=(1.0396, 5.9537, 1.402, 6.3141), label='Lome (UTM 31N)'),
    'london': dict(
        bounds=(668841, 5678437, 731459, 5740904), crs=32630,
        lonlat=(-0.5528, 51.2317, 0.3155, 51.7722), label='London (UTM 30N)'),
    'long_beach': dict(
        bounds=(372453, 3718957, 413047, 3759360), crs=32611,
        lonlat=(-118.3748, 33.6068, -117.9412, 33.9671), label='Long Beach (UTM 11N)'),
    'los_angeles': dict(
        bounds=(360290, 3731628, 421193, 3792257), crs=32611,
        lonlat=(-118.5079, 33.7217, -117.856, 34.2622), label='Los Angeles (UTM 11N)'),
    'louisville': dict(
        bounds=(597160, 4219591, 621581, 4243906), crs=32616,
        lonlat=(-85.8883, 38.1189, -85.613, 38.3351), label='Louisville (UTM 16N)'),
    'luan': dict(
        bounds=(430368, 3493049, 470697, 3533189), crs=32650,
        lonlat=(116.2662, 31.5721, 116.69, 31.9325), label='Luan (UTM 50N)'),
    'luanda': dict(
        bounds=(275359, 8992730, 335819, 9052799), crs=32733,
        lonlat=(12.959, -9.1066, 13.506, -8.5661), label='Luanda (UTM 33S)'),
    'lubumbashi': dict(
        bounds=(532032, 8689030, 572208, 8728950), crs=32735,
        lonlat=(27.2941, -11.8583, 27.6621, -11.4979), label='Lubumbashi (UTM 35S)'),
    'lucknow': dict(
        bounds=(471255, 2950639, 511442, 2990588), crs=32644,
        lonlat=(80.7111, 26.6768, 81.115, 27.0372), label='Lucknow (UTM 44N)'),
    'ludhiana': dict(
        bounds=(562964, 3402027, 603413, 3442284), crs=32643,
        lonlat=(75.6603, 30.7495, 76.0804, 31.1099), label='Ludhiana (UTM 43N)'),
    'luoyang': dict(
        bounds=(614167, 3818505, 654895, 3859049), crs=32649,
        lonlat=(112.249, 34.5018, 112.6872, 34.8621), label='Luoyang (UTM 49N)'),
    'lusaka': dict(
        bounds=(617341, 8275331, 657695, 8315442), crs=32735,
        lonlat=(28.0945, -15.5949, 28.4683, -15.2345), label='Lusaka (UTM 35S)'),
    'luxembourg': dict(
        bounds=(280145, 5486953, 305158, 5511905), crs=32632,
        lonlat=(5.9632, 49.5036, 6.2968, 49.7198), label='Luxembourg (UTM 32N)'),
    'luzhou': dict(
        bounds=(516772, 3174949, 557025, 3215006), crs=32648,
        lonlat=(105.1723, 28.7017, 105.5838, 29.0621), label='Luzhou (UTM 48N)'),
    'lyon': dict(
        bounds=(621658, 5049894, 662738, 5090842), crs=32631,
        lonlat=(4.5698, 45.5918, 5.0864, 45.9521), label='Lyon (UTM 31N)'),
    'maanshan': dict(
        bounds=(619909, 3491278, 660595, 3531775), crs=32650,
        lonlat=(118.2682, 31.5502, 118.6919, 31.9106), label='Maanshan (UTM 50N)'),
    'madrid': dict(
        bounds=(411379, 4442427, 472076, 4502889), crs=32630,
        lonlat=(-4.0402, 40.1317, -3.3304, 40.6722), label='Madrid (UTM 30N)'),
    'madurai': dict(
        bounds=(163698, 1078046, 204191, 1118289), crs=32644,
        lonlat=(77.9352, 9.7418, 78.301, 10.1022), label='Madurai (UTM 44N)'),
    'majuro': dict(
        bounds=(529930, 773194, 554001, 797114), crs=32659,
        lonlat=(171.2711, 6.9949, 171.4889, 7.2111), label='Majuro (UTM 59N)'),
    'makassar': dict(
        bounds=(749273, 9411551, 789559, 9451577), crs=32750,
        lonlat=(119.2492, -5.3182, 119.611, -4.9578), label='Makassar (UTM 50S)'),
    'makkah': dict(
        bounds=(564636, 2350143, 604971, 2390233), crs=32637,
        lonlat=(39.6245, 21.2518, 40.0117, 21.6121), label='Makkah (UTM 37N)'),
    'malabo': dict(
        bounds=(463897, 402546, 487965, 426449), crs=32632,
        lonlat=(8.6749, 3.6419, 8.8916, 3.8581), label='Malabo (UTM 32N)'),
    'male': dict(
        bounds=(321448, 448738, 345551, 472689), crs=32643,
        lonlat=(73.3916, 4.0586, 73.6083, 4.2748), label='Male (UTM 43N)'),
    'managua': dict(
        bounds=(567319, 1331816, 591440, 1355789), crs=32616,
        lonlat=(-86.381, 12.0469, -86.1599, 12.2631), label='Managua (UTM 16N)'),
    'manama': dict(
        bounds=(446266, 2889893, 470424, 2913914), crs=32639,
        lonlat=(50.4625, 26.128, 50.7036, 26.3442), label='Manama (UTM 39N)'),
    'manaus': dict(
        bounds=(813145, 9637085, 853414, 9677089), crs=32720,
        lonlat=(-60.1824, -3.2783, -59.8215, -2.9179), label='Manaus (UTM 20S)'),
    'manchester': dict(
        bounds=(529530, 5908207, 570139, 5948712), crs=32630,
        lonlat=(-2.5529, 53.3222, -1.947, 53.6825), label='Manchester (UTM 30N)'),
    'mandalay': dict(
        bounds=(178277, 2412267, 219195, 2452950), crs=32647,
        lonlat=(95.8888, 21.7918, 96.2774, 22.1521), label='Mandalay (UTM 47N)'),
    'manila': dict(
        bounds=(252035, 1585571, 312762, 1645930), crs=32651,
        lonlat=(120.701, 14.3358, 121.2596, 14.8764), label='Manila (UTM 51N)'),
    'mannheim': dict(
        bounds=(441325, 5463083, 481784, 5503432), crs=32632,
        lonlat=(8.1926, 49.3202, 8.7475, 49.6806), label='Mannheim (UTM 32N)'),
    'maoming': dict(
        bounds=(466475, 2404071, 506639, 2443998), crs=32649,
        lonlat=(110.6758, 21.7402, 111.0642, 22.1006), label='Maoming (UTM 49N)'),
    'maputo': dict(
        bounds=(438514, 7109387, 478768, 7149415), crs=32736,
        lonlat=(32.3868, -26.1335, 32.7876, -25.7732), label='Maputo (UTM 36S)'),
    'maracaibo': dict(
        bounds=(188588, 1167486, 229080, 1207723), crs=32619,
        lonlat=(-71.8453, 10.5517, -71.4785, 10.9121), label='Maracaibo (UTM 19N)'),
    'marseille': dict(
        bounds=(671916, 4775412, 713218, 4816579), crs=32631,
        lonlat=(5.1255, 43.1117, 5.6206, 43.4721), label='Marseille (UTM 31N)'),
    'maseru': dict(
        bounds=(534852, 6744795, 559032, 6768849), crs=32735,
        lonlat=(27.3593, -29.4248, 27.6073, -29.2086), label='Maseru (UTM 35S)'),
    'mashhad': dict(
        bounds=(710118, 3996676, 751330, 4037718), crs=32640,
        lonlat=(59.3446, 36.0918, 59.7915, 36.4521), label='Mashhad (UTM 40N)'),
    'mbabane': dict(
        bounds=(301444, 7075741, 325871, 7100052), crs=32736,
        lonlat=(31.0127, -26.4248, 31.2539, -26.2085), label='Mbabane (UTM 36S)'),
    'mbuji_mayi': dict(
        bounds=(767379, 9299682, 807720, 9339757), crs=32734,
        lonlat=(23.4168, -6.3283, 23.7793, -5.9679), label='Mbuji-Mayi (UTM 34S)'),
    'medan': dict(
        bounds=(440860, 375996, 480968, 415849), crs=32647,
        lonlat=(98.4676, 3.4017, 98.8286, 3.7621), label='Medan (UTM 47N)'),
    'medellin': dict(
        bounds=(416103, 673921, 456246, 713796), crs=32618,
        lonlat=(-75.7582, 6.0968, -75.3957, 6.4571), label='Medellin (UTM 18N)'),
    'meerut': dict(
        bounds=(742333, 3190838, 783407, 3231697), crs=32643,
        lonlat=(77.492, 28.8222, 77.9041, 29.1825), label='Meerut (UTM 43N)'),
    'melbourne': dict(
        bounds=(301034, 5791974, 342060, 5832838), crs=32755,
        lonlat=(144.745, -37.9983, 145.2012, -37.6379), label='Melbourne (UTM 55S)'),
    'melekeok': dict(
        bounds=(446751, 815688, 470835, 839610), crs=32653,
        lonlat=(134.5175, 7.3793, 134.7356, 7.5955), label='Melekeok (UTM 53N)'),
    'memphis': dict(
        bounds=(752577, 3870113, 793937, 3911298), crs=32615,
        lonlat=(-90.2222, 34.9418, -89.7817, 35.3021), label='Memphis (UTM 15N)'),
    'mexico_city': dict(
        bounds=(455883, 2120091, 516160, 2179965), crs=32614,
        lonlat=(-99.4196, 19.1741, -98.8463, 19.7147), label='Mexico City (UTM 14N)'),
    'miami': dict(
        bounds=(547393, 2822536, 607931, 2882749), crs=32617,
        lonlat=(-80.5262, 25.5193, -79.9259, 26.0598), label='Miami (UTM 17N)'),
    'mianyang': dict(
        bounds=(457819, 3461931, 498037, 3501960), crs=32648,
        lonlat=(104.5568, 31.2917, 104.9793, 31.6521), label='Mianyang (UTM 48N)'),
    'milan': dict(
        bounds=(495773, 5015355, 536071, 5055497), crs=32632,
        lonlat=(8.9461, 45.2917, 9.46, 45.6521), label='Milan (UTM 32N)'),
    'milwaukee': dict(
        bounds=(404563, 4747088, 445167, 4787552), crs=32616,
        lonlat=(-88.1685, 42.8744, -87.6753, 43.2348), label='Milwaukee (UTM 16N)'),
    'minneapolis': dict(
        bounds=(459784, 4960922, 500079, 5001084), crs=32615,
        lonlat=(-93.5085, 44.8017, -92.999, 45.1621), label='Minneapolis (UTM 15N)'),
    'minsk': dict(
        bounds=(516937, 5952590, 557440, 5993007), crs=32635,
        lonlat=(27.2589, 53.7217, 27.8705, 54.0821), label='Minsk (UTM 35N)'),
    'mogadishu': dict(
        bounds=(520503, 208727, 560613, 248573), crs=32638,
        lonlat=(45.1844, 1.8884, 45.545, 2.2488), label='Mogadishu (UTM 38N)'),
    'monaco': dict(
        bounds=(359418, 4831958, 383973, 4856445), crs=32632,
        lonlat=(7.2573, 43.6315, 7.5565, 43.8478), label='Monaco (UTM 32N)'),
    'monrovia': dict(
        bounds=(280786, 678332, 321037, 718330), crs=32629,
        lonlat=(-10.9809, 6.1344, -10.6184, 6.4948), label='Monrovia (UTM 29N)'),
    'monterrey': dict(
        bounds=(346036, 2819890, 386566, 2860200), crs=32614,
        lonlat=(-100.5318, 25.4918, -100.132, 25.8521), label='Monterrey (UTM 14N)'),
    'montevideo': dict(
        bounds=(555402, 6122431, 595881, 6162730), crs=32721,
        lonlat=(-56.3926, -35.0363, -55.9534, -34.6759), label='Montevideo (UTM 21S)'),
    'montreal': dict(
        bounds=(590155, 5019351, 631024, 5060087), crs=32618,
        lonlat=(-73.8423, 45.3218, -73.3282, 45.6821), label='Montreal (UTM 18N)'),
    'moroni': dict(
        bounds=(296080, 8693521, 320302, 8717599), crs=32738,
        lonlat=(43.1298, -11.8123, 43.3506, -11.596), label='Moroni (UTM 38S)'),
    'moscow': dict(
        bounds=(382038, 6149004, 443519, 6210367), crs=32637,
        lonlat=(37.1333, 55.4838, 38.0938, 56.0244), label='Moscow (UTM 37N)'),
    'mosul': dict(
        bounds=(312864, 4003685, 353784, 4044428), crs=32638,
        lonlat=(42.9194, 36.1668, 43.3668, 36.5271), label='Mosul (UTM 38N)'),
    'mudangiang': dict(
        bounds=(526522, 4916001, 566975, 4956312), crs=32652,
        lonlat=(129.3351, 44.3968, 129.841, 44.7571), label='Mudangiang (UTM 52N)'),
    'multan': dict(
        bounds=(715679, 3323312, 756686, 3364132), crs=32642,
        lonlat=(71.2446, 30.0217, 71.6615, 30.3821), label='Multan (UTM 42N)'),
    'mumbai': dict(
        bounds=(243716, 2074044, 304643, 2134618), crs=32643,
        lonlat=(72.5692, 18.7487, 73.1409, 19.2892), label='Mumbai (UTM 43N)'),
    'munich': dict(
        bounds=(670757, 5313499, 712264, 5354897), crs=32632,
        lonlat=(11.3031, 47.9517, 11.843, 48.3121), label='Munich (UTM 32N)'),
    'muscat': dict(
        bounds=(650387, 2600221, 674735, 2624430), crs=32640,
        lonlat=(58.4753, 23.5052, 58.7113, 23.7214), label='Muscat (UTM 40N)'),
    'nagoya': dict(
        bounds=(653833, 3871779, 694747, 3912516), crs=32653,
        lonlat=(136.6927, 34.9768, 137.1334, 35.3371), label='Nagoya (UTM 53N)'),
    'nagpur': dict(
        bounds=(281160, 2322189, 321780, 2362577), crs=32644,
        lonlat=(78.8948, 20.9917, 79.2813, 21.3521), label='Nagpur (UTM 44N)'),
    'nairobi': dict(
        bounds=(236752, 9838310, 276909, 9878209), crs=32737,
        lonlat=(36.6345, -1.4616, 36.9949, -1.1012), label='Nairobi (UTM 37S)'),
    'nanchang': dict(
        bounds=(370100, 3153134, 410603, 3193432), crs=32650,
        lonlat=(115.6727, 28.5018, 116.0834, 28.8621), label='Nanchang (UTM 50N)'),
    'nanchong': dict(
        bounds=(587702, 3385876, 628247, 3426225), crs=32648,
        lonlat=(105.9183, 30.6022, 106.3378, 30.9626), label='Nanchong (UTM 48N)'),
    'nanjing': dict(
        bounds=(647495, 3527294, 688301, 3567901), crs=32650,
        lonlat=(118.5654, 31.8718, 118.9906, 32.2321), label='Nanjing (UTM 50N)'),
    'nanning': dict(
        bounds=(204259, 2506003, 245138, 2546648), crs=32649,
        lonlat=(108.1226, 22.6418, 108.5136, 23.0021), label='Nanning (UTM 49N)'),
    'nanyang': dict(
        bounds=(622425, 3632340, 663151, 3672872), crs=32649,
        lonlat=(112.3132, 32.8222, 112.7429, 33.1825), label='Nanyang (UTM 49N)'),
    'naples': dict(
        bounds=(415884, 4501344, 456383, 4541698), crs=32633,
        lonlat=(14.0049, 40.6618, 14.4812, 41.0222), label='Naples (UTM 33N)'),
    'nashville': dict(
        bounds=(507553, 3991028, 531699, 4015063), crs=32616,
        lonlat=(-86.9159, 36.0638, -86.648, 36.28), label='Nashville (UTM 16N)'),
    'nasik': dict(
        bounds=(351943, 2192135, 392356, 2232306), crs=32643,
        lonlat=(73.5863, 19.8222, 73.9698, 20.1825), label='Nasik (UTM 43N)'),
    'nassau': dict(
        bounds=(250714, 2764064, 275214, 2788436), crs=32618,
        lonlat=(-77.4694, 24.9753, -77.2307, 25.1915), label='Nassau (UTM 18N)'),
    'naypyidaw': dict(
        bounds=(185594, 2176263, 210099, 2200621), crs=32647,
        lonlat=(96.0018, 19.6604, 96.2316, 19.8766), label='Naypyidaw (UTM 47N)'),
    'ndjamena': dict(
        bounds=(493099, 1327317, 517178, 1351240), crs=32633,
        lonlat=(14.9366, 12.0069, 15.1578, 12.2232), label='Ndjamena (UTM 33N)'),
    'neijiang': dict(
        bounds=(484564, 3252536, 524769, 3292496), crs=32648,
        lonlat=(104.8409, 29.4021, 105.2553, 29.7625), label='Neijiang (UTM 48N)'),
    'new_delhi': dict(
        bounds=(702870, 3153453, 727402, 3177857), crs=32643,
        lonlat=(77.0768, 28.4919, 77.3231, 28.7081), label='New Delhi (UTM 43N)'),
    'new_orleans': dict(
        bounds=(773033, 3309837, 797742, 3334443), crs=32615,
        lonlat=(-90.1667, 29.8888, -89.9171, 30.1051), label='New Orleans (UTM 15N)'),
    'new_taipei': dict(
        bounds=(324789, 2747073, 365368, 2787444), crs=32651,
        lonlat=(121.2662, 24.8326, 121.6638, 25.193), label='New Taipei (UTM 51N)'),
    'new_york': dict(
        bounds=(555598, 4481433, 616526, 4542132), crs=32618,
        lonlat=(-74.3387, 40.4817, -73.6252, 41.0222), label='New York (UTM 18N)'),
    'niamey': dict(
        bounds=(392114, 1482663, 416269, 1506674), crs=32631,
        lonlat=(2.0035, 13.4105, 2.2259, 13.6268), label='Niamey (UTM 31N)'),
    'nicosia': dict(
        bounds=(521318, 3879567, 545493, 3903632), crs=32636,
        lonlat=(33.2344, 35.0586, 33.4989, 35.2748), label='Nicosia (UTM 36N)'),
    'ningbo': dict(
        bounds=(339438, 3286380, 380079, 3326828), crs=32651,
        lonlat=(121.3403, 29.7017, 121.7559, 30.0621), label='Ningbo (UTM 51N)'),
    'niteroi': dict(
        bounds=(674584, 7446063, 715235, 7486492), crs=32723,
        lonlat=(-43.2956, -23.0802, -42.9044, -22.7198), label='Niteroi (UTM 23S)'),
    'nizhny_novgorod': dict(
        bounds=(417578, 6223515, 458353, 6264200), crs=32638,
        lonlat=(43.6731, 56.1548, 44.3232, 56.5151), label='Nizhny Novgorod (UTM 38N)'),
    'norfolk': dict(
        bounds=(365492, 4058761, 406184, 4099271), crs=32618,
        lonlat=(-76.5052, 36.6698, -76.0548, 37.0301), label='Norfolk (UTM 18N)'),
    'nouakchott': dict(
        bounds=(384685, 1987996, 408887, 2012046), crs=32628,
        lonlat=(-16.0891, 17.9783, -15.8616, 18.1945), label='Nouakchott (UTM 28N)'),
    'novosibirsk': dict(
        bounds=(604585, 6079527, 645891, 6120749), crs=32644,
        lonlat=(82.6437, 54.8517, 83.2725, 55.2121), label='Novosibirsk (UTM 44N)'),
    'nukualofa': dict(
        bounds=(672618, 7649381, 696963, 7673587), crs=32701,
        lonlat=(-175.3365, -21.2466, -175.1047, -21.0304), label='Nukualofa (UTM 1S)'),
    'oakland': dict(
        bounds=(548410, 4160325, 588887, 4200646), crs=32610,
        lonlat=(-122.449, 37.5887, -121.9932, 37.9491), label='Oakland (UTM 10N)'),
    'oklahoma_city': dict(
        bounds=(622010, 3914232, 646459, 3938573), crs=32614,
        lonlat=(-97.6534, 35.3639, -97.3879, 35.5801), label='Oklahoma City (UTM 14N)'),
    'omaha': dict(
        bounds=(738119, 4557304, 763049, 4582140), crs=32614,
        lonlat=(-96.1538, 41.1319, -95.8662, 41.3481), label='Omaha (UTM 14N)'),
    'omdurman': dict(
        bounds=(424136, 1706633, 464347, 1746597), crs=32636,
        lonlat=(32.2929, 15.4365, 32.6671, 15.7969), label='Omdurman (UTM 36N)'),
    'orlando': dict(
        bounds=(442462, 3133960, 482719, 3174015), crs=32617,
        lonlat=(-81.587, 28.3317, -81.1769, 28.6921), label='Orlando (UTM 17N)'),
    'osaka': dict(
        bounds=(511796, 3815576, 572275, 3875799), crs=32653,
        lonlat=(135.1293, 34.4817, 135.7871, 35.0223), label='Osaka (UTM 53N)'),
    'oslo': dict(
        bounds=(585394, 6631301, 610146, 6656009), crs=32632,
        lonlat=(10.5323, 59.8105, 10.9637, 60.0267), label='Oslo (UTM 32N)'),
    'ottawa': dict(
        bounds=(424754, 5009542, 465271, 5049920), crs=32618,
        lonlat=(-75.9587, 45.2385, -75.4453, 45.5988), label='Ottawa (UTM 18N)'),
    'ouagadougou': dict(
        bounds=(640027, 1348129, 680361, 1388205), crs=32630,
        lonlat=(-1.7111, 12.1921, -1.3422, 12.5524), label='Ouagadougou (UTM 30N)'),
    'palembang': dict(
        bounds=(451952, 9650899, 492056, 9690745), crs=32748,
        lonlat=(104.5677, -3.1583, 104.9285, -2.7979), label='Palembang (UTM 48S)'),
    'palikir': dict(
        bounds=(394041, 752642, 418147, 776599), crs=32657,
        lonlat=(158.0411, 6.8085, 158.2589, 7.0248), label='Palikir (UTM 57N)'),
    'panama_city': dict(
        bounds=(640928, 971854, 681200, 1011861), crs=32617,
        lonlat=(-79.7174, 8.7898, -79.3526, 9.1501), label='Panama City (UTM 17N)'),
    'paramaribo': dict(
        bounds=(690875, 633302, 715021, 657292), crs=32621,
        lonlat=(-55.2757, 5.7269, -55.0584, 5.9431), label='Paramaribo (UTM 21N)'),
    'paris': dict(
        bounds=(420410, 5382845, 481204, 5443458), crs=32631,
        lonlat=(1.9205, 48.5984, 2.7423, 49.1389), label='Paris (UTM 31N)'),
    'patna': dict(
        bounds=(291679, 2815466, 332373, 2855957), crs=32645,
        lonlat=(84.9283, 25.4467, 85.3279, 25.8071), label='Patna (UTM 45N)'),
    'perth': dict(
        bounds=(369865, 6443973, 410433, 6484344), crs=32750,
        lonlat=(115.6257, -32.1332, 116.0504, -31.7729), label='Perth (UTM 50S)'),
    'peshawar': dict(
        bounds=(713401, 3745364, 754545, 3786322), crs=32642,
        lonlat=(71.3157, 33.8268, 71.7504, 34.1871), label='Peshawar (UTM 42N)'),
    'philadelphia': dict(
        bounds=(455028, 4397968, 515502, 4458102), crs=32618,
        lonlat=(-75.5248, 39.7316, -74.8191, 40.2722), label='Philadelphia (UTM 18N)'),
    'phnom_penh': dict(
        bounds=(470629, 1257090, 510758, 1296955), crs=32648,
        lonlat=(104.7308, 11.3718, 105.0986, 11.7322), label='Phnom Penh (UTM 48N)'),
    'phoenix': dict(
        bounds=(380160, 3691713, 420711, 3732089), crs=32612,
        lonlat=(-112.2881, 33.3617, -111.8558, 33.7221), label='Phoenix (UTM 12N)'),
    'pittsburgh': dict(
        bounds=(564409, 4455983, 605015, 4496431), crs=32617,
        lonlat=(-80.2386, 40.2518, -79.7652, 40.6121), label='Pittsburgh (UTM 17N)'),
    'podgorica': dict(
        bounds=(345159, 4690735, 369747, 4715234), crs=32634,
        lonlat=(19.1198, 42.3579, 19.4129, 42.5741), label='Podgorica (UTM 34N)'),
    'port_au_prince': dict(
        bounds=(760670, 2032101, 801408, 2072605), crs=32618,
        lonlat=(-72.528, 18.3628, -72.1479, 18.7232), label='Port-au-Prince (UTM 18N)'),
    'port_louis': dict(
        bounds=(540182, 7758000, 564330, 7781999), crs=32740,
        lonlat=(57.3848, -20.2747, 57.6152, -20.0585), label='Port Louis (UTM 40S)'),
    'port_moresby': dict(
        bounds=(509097, 8941805, 533172, 8965722), crs=32755,
        lonlat=(147.0829, -9.5728, 147.3021, -9.3566), label='Port Moresby (UTM 55S)'),
    'port_of_spain': dict(
        bounds=(650125, 1165878, 674309, 1189906), crs=32620,
        lonlat=(-61.627, 10.5439, -61.407, 10.7601), label='Port-of-Spain (UTM 20N)'),
    'port_vila': dict(
        bounds=(203197, 8025132, 227627, 8049427), crs=32759,
        lonlat=(168.2031, -17.8415, 168.4301, -17.6252), label='Port Vila (UTM 59S)'),
    'portland': dict(
        bounds=(504741, 5020923, 545064, 5061122), crs=32610,
        lonlat=(-122.9391, 45.3418, -122.4248, 45.7022), label='Portland (UTM 10N)'),
    'porto': dict(
        bounds=(511615, 4535636, 551938, 4575808), crs=32629,
        lonlat=(-8.8612, 40.9718, -8.3827, 41.3321), label='Porto (UTM 29N)'),
    'porto_alegre': dict(
        bounds=(460394, 6655857, 500599, 6695852), crs=32722,
        lonlat=(-51.4101, -30.2282, -50.9938, -29.8679), label='Porto Alegre (UTM 22S)'),
    'prague': dict(
        bounds=(441348, 5528111, 481811, 5568470), crs=32633,
        lonlat=(14.1832, 49.9051, 14.7448, 50.2655), label='Prague (UTM 33N)'),
    'praia': dict(
        bounds=(217069, 1638545, 241430, 1662751), crs=32627,
        lonlat=(-23.6286, 14.8086, -23.4048, 15.0248), label='Praia (UTM 27N)'),
    'pretoria': dict(
        bounds=(602933, 7136252, 643440, 7176541), crs=32735,
        lonlat=(28.0275, -25.8852, 28.4275, -25.5248), label='Pretoria (UTM 35S)'),
    'pristina': dict(
        bounds=(501554, 4711800, 525694, 4735855), crs=32634,
        lonlat=(21.019, 42.5586, 21.313, 42.7748), label='Pristina (UTM 34N)'),
    'providence': dict(
        bounds=(278577, 4612414, 319868, 4653551), crs=32619,
        lonlat=(-71.6587, 41.6429, -71.1751, 42.0032), label='Providence (UTM 19N)'),
    'puebla': dict(
        bounds=(563841, 2086741, 604134, 2126804), crs=32614,
        lonlat=(-98.3926, 18.8717, -98.0114, 19.2321), label='Puebla (UTM 14N)'),
    'pune': dict(
        bounds=(358206, 2029377, 398583, 2069504), crs=32643,
        lonlat=(73.658, 18.3518, 74.0381, 18.7121), label='Pune (UTM 43N)'),
    'pyongyang': dict(
        bounds=(717674, 4302174, 759052, 4343394), crs=32651,
        lonlat=(125.5208, 38.8412, 125.9847, 39.2016), label='Pyongyang (UTM 51N)'),
    'qingdao': dict(
        bounds=(238764, 3976926, 280018, 4018018), crs=32651,
        lonlat=(120.1051, 35.9117, 120.551, 36.2721), label='Qingdao (UTM 51N)'),
    'qiqihar': dict(
        bounds=(554356, 5223942, 595039, 5264504), crs=32651,
        lonlat=(123.7221, 47.1667, 124.254, 47.5271), label='Qiqihar (UTM 51N)'),
    'quanzhou': dict(
        bounds=(639111, 2734846, 679707, 2775217), crs=32650,
        lonlat=(118.3794, 24.7218, 118.7767, 25.0821), label='Quanzhou (UTM 50N)'),
    'quezon_city': dict(
        bounds=(267572, 1600488, 308051, 1640708), crs=32651,
        lonlat=(120.8437, 14.4703, 121.2162, 14.8306), label='Quezon City (UTM 51N)'),
    'quito': dict(
        bounds=(757978, 9956492, 798126, 9996361), crs=32717,
        lonlat=(-78.6822, -0.3932, -78.3218, -0.0329), label='Quito (UTM 17S)'),
    'rabat': dict(
        bounds=(679306, 3746685, 720303, 3787505), crs=32629,
        lonlat=(-7.0538, 33.8451, -6.619, 34.2055), label='Rabat (UTM 29N)'),
    'rajkot': dict(
        bounds=(664932, 2448287, 705538, 2488661), crs=32642,
        lonlat=(70.6033, 22.1318, 70.9928, 22.4921), label='Rajkot (UTM 42N)'),
    'raleigh': dict(
        bounds=(692268, 3945961, 733385, 3986910), crs=32617,
        lonlat=(-78.8669, 35.6386, -78.4225, 35.999), label='Raleigh (UTM 17N)'),
    'rangoon': dict(
        bounds=(177376, 1837735, 218099, 1878219), crs=32647,
        lonlat=(95.9765, 16.6051, 96.3529, 16.9655), label='Rangoon (UTM 47N)'),
    'rawalpindi': dict(
        bounds=(297476, 3699400, 338378, 3740128), crs=32643,
        lonlat=(72.8218, 33.4217, 73.2544, 33.7821), label='Rawalpindi (UTM 43N)'),
    'recife': dict(
        bounds=(268527, 9087024, 308830, 9127078), crs=32725,
        lonlat=(-35.0995, -8.2539, -34.7356, -7.8935), label='Recife (UTM 25S)'),
    'reno': dict(
        bounds=(245178, 4367024, 270034, 4391780), crs=32611,
        lonlat=(-119.9602, 39.4219, -119.6798, 39.6381), label='Reno (UTM 11N)'),
    'reykjavik': dict(
        bounds=(441494, 7101872, 465976, 7126322), crs=32627,
        lonlat=(-22.198, 64.0419, -21.7021, 64.2581), label='Reykjavik (UTM 27N)'),
    'richmond': dict(
        bounds=(271028, 4146635, 295748, 4171255), crs=32618,
        lonlat=(-77.5883, 37.4439, -77.3156, 37.6601), label='Richmond (UTM 18N)'),
    'riga': dict(
        bounds=(311024, 6303041, 336152, 6328118), crs=32635,
        lonlat=(23.9017, 56.8419, 24.2982, 57.0581), label='Riga (UTM 35N)'),
    'rio_de_janeiro': dict(
        bounds=(651427, 7433579, 712348, 7494158), crs=32723,
        lonlat=(-43.5204, -23.1933, -42.9335, -22.6528), label='Rio de Janeiro (UTM 23S)'),
    'riyadh': dict(
        bounds=(658930, 2706350, 699573, 2746785), crs=32638,
        lonlat=(46.5726, 24.4626, 46.969, 24.823), label='Riyadh (UTM 38N)'),
    'rochester': dict(
        bounds=(274443, 4770925, 299289, 4795691), crs=32618,
        lonlat=(-77.7701, 43.0643, -77.4737, 43.2805), label='Rochester (UTM 18N)'),
    'rome': dict(
        bounds=(270339, 4620938, 311683, 4662136), crs=32633,
        lonlat=(12.2392, 41.7177, 12.7234, 42.0781), label='Rome (UTM 33N)'),
    'rosario': dict(
        bounds=(697490, 6331481, 738534, 6372338), crs=32720,
        lonlat=(-60.883, -33.1294, -60.4535, -32.769), label='Rosario (UTM 20S)'),
    'roseau': dict(
        bounds=(661063, 1680215, 685318, 1704316), crs=32620,
        lonlat=(-61.4991, 15.1929, -61.2749, 15.4091), label='Roseau (UTM 20N)'),
    'sacramento': dict(
        bounds=(612738, 4250639, 653562, 4291290), crs=32610,
        lonlat=(-121.7025, 38.3968, -121.2415, 38.7571), label='Sacramento (UTM 10N)'),
    'saidu': dict(
        bounds=(236770, 3828033, 277983, 3869062), crs=32643,
        lonlat=(72.1307, 34.5699, 72.5693, 34.9302), label='Saidu (UTM 43N)'),
    'saint_george_s': dict(
        bounds=(624882, 1320676, 649061, 1344698), crs=32620,
        lonlat=(-61.8522, 11.9445, -61.6311, 12.1607), label="Saint George's (UTM 20N)"),
    'saint_john_s': dict(
        bounds=(610232, 1880942, 634450, 1905006), crs=32620,
        lonlat=(-61.9632, 17.0099, -61.7369, 17.2261), label="Saint John's (UTM 20N)"),
    'salt_lake_city': dict(
        bounds=(409156, 4502302, 433512, 4526558), crs=32612,
        lonlat=(-112.0748, 40.6689, -111.7892, 40.8851), label='Salt Lake City (UTM 12N)'),
    'salvador': dict(
        bounds=(536111, 8546369, 576300, 8586307), crs=32724,
        lonlat=(-38.6668, -13.1482, -38.297, -12.7878), label='Salvador (UTM 24S)'),
    'san_antonio': dict(
        bounds=(527462, 3242265, 567754, 3282368), crs=32614,
        lonlat=(-98.7162, 29.3091, -98.3023, 29.6695), label='San Antonio (UTM 14N)'),
    'san_bernardino': dict(
        bounds=(451983, 3755739, 492243, 3795820), crs=32611,
        lonlat=(-117.5196, 33.9421, -117.0843, 34.3025), label='San Bernardino (UTM 11N)'),
    'san_diego': dict(
        bounds=(462830, 3611578, 503048, 3651601), crs=32611,
        lonlat=(-117.3963, 32.6418, -116.9675, 33.0022), label='San Diego (UTM 11N)'),
    'san_francisco': dict(
        bounds=(531181, 4160275, 571582, 4200511), crs=32610,
        lonlat=(-122.6451, 37.589, -122.1892, 37.9494), label='San Francisco (UTM 10N)'),
    'san_jose': dict(
        bounds=(581481, 4108770, 622115, 4149243), crs=32610,
        lonlat=(-122.0784, 37.1217, -121.6254, 37.4821), label='San Jose (UTM 10N)'),
    'san_jose_cr': dict(
        bounds=(799316, 1079735, 839813, 1119972), crs=32616,
        lonlat=(-84.2689, 9.7568, -83.9031, 10.1171), label='San Jose (UTM 16N)'),
    'san_juan': dict(
        bounds=(782797, 2021007, 823586, 2061559), crs=32619,
        lonlat=(-66.3199, 18.2598, -65.94, 18.6202), label='San Juan (UTM 19N)'),
    'san_marino': dict(
        bounds=(282227, 4855588, 307074, 4880349), crs=32633,
        lonlat=(12.2916, 43.828, 12.5919, 44.0442), label='San Marino (UTM 33N)'),
    'san_salvador': dict(
        bounds=(241267, 1496848, 281771, 1537088), crs=32616,
        lonlat=(-89.3905, 13.5318, -89.0195, 13.8921), label='San Salvador (UTM 16N)'),
    'sanaa': dict(
        bounds=(394489, 1677939, 434747, 1717953), crs=32638,
        lonlat=(44.0178, 15.1765, 44.3915, 15.5369), label='Sanaa (UTM 38N)'),
    'santa_cruz': dict(
        bounds=(455739, 8017266, 495891, 8057187), crs=32720,
        lonlat=(-63.4171, -17.9322, -63.0388, -17.5718), label='Santa Cruz (UTM 20S)'),
    'santiago': dict(
        bounds=(314191, 6267294, 375370, 6328196), crs=32719,
        lonlat=(-70.9929, -33.7183, -70.3451, -33.1778), label='Santiago (UTM 19S)'),
    'santiago_do': dict(
        bounds=(304459, 2136877, 344976, 2177160), crs=32619,
        lonlat=(-70.8612, 19.3198, -70.4789, 19.6802), label='Santiago (UTM 19N)'),
    'santo_domingo': dict(
        bounds=(384584, 2022616, 424896, 2062696), crs=32619,
        lonlat=(-70.092, 18.2918, -69.7121, 18.6522), label='Santo Domingo (UTM 19N)'),
    'santos': dict(
        bounds=(343892, 7330308, 384397, 7370596), crs=32723,
        lonlat=(-46.532, -24.132, -46.1377, -23.7716), label='Santos (UTM 23S)'),
    'sao_paulo': dict(
        bounds=(303444, 7363600, 364322, 7424138), crs=32723,
        lonlat=(-46.9218, -23.827, -46.3321, -23.2865), label='Sao Paulo (UTM 23S)'),
    'sao_tome': dict(
        bounds=(235670, 24920, 259752, 48841), crs=32632,
        lonlat=(6.6252, 0.2253, 6.8414, 0.4415), label='Sao Tome (UTM 32N)'),
    'sapporo': dict(
        bounds=(507419, 4749348, 547746, 4789532), crs=32654,
        lonlat=(141.0914, 42.8967, 141.5848, 43.2571), label='Sapporo (UTM 54N)'),
    'sarajevo': dict(
        bounds=(277203, 4846165, 302061, 4870941), crs=32634,
        lonlat=(18.2331, 43.7419, 18.5329, 43.9581), label='Sarajevo (UTM 34N)'),
    'savannah': dict(
        bounds=(477546, 3530792, 501655, 3554781), crs=32617,
        lonlat=(-81.2375, 31.913, -80.9825, 32.1292), label='Savannah (UTM 17N)'),
    'seattle': dict(
        bounds=(529306, 5248779, 569815, 5289162), crs=32610,
        lonlat=(-122.609, 47.3918, -122.0749, 47.7521), label='Seattle (UTM 10N)'),
    'semarang': dict(
        bounds=(415641, 9210163, 455789, 9250056), crs=32749,
        lonlat=(110.2366, -7.1449, 110.5996, -6.7845), label='Semarang (UTM 49S)'),
    'sendai': dict(
        bounds=(481607, 4217898, 521855, 4257905), crs=32654,
        lonlat=(140.7902, 38.1089, 141.2493, 38.4692), label='Sendai (UTM 54N)'),
    'seoul': dict(
        bounds=(292312, 4129225, 353827, 4190495), crs=32652,
        lonlat=(126.6568, 37.298, 127.3388, 37.8386), label='Seoul (UTM 52N)'),
    'seville': dict(
        bounds=(215474, 4123363, 256897, 4164630), crs=32630,
        lonlat=(-6.2068, 37.2248, -5.7532, 37.5852), label='Seville (UTM 30N)'),
    'shanghai': dict(
        bounds=(320268, 3424519, 381333, 3485297), crs=32651,
        lonlat=(121.1185, 30.9481, 121.7506, 31.4887), label='Shanghai (UTM 51N)'),
    'shangqiu': dict(
        bounds=(355435, 3792919, 396113, 3833411), crs=32650,
        lonlat=(115.4296, 34.2722, 115.8666, 34.6325), label='Shangqiu (UTM 50N)'),
    'shantou': dict(
        bounds=(445946, 2564759, 486161, 2604739), crs=32650,
        lonlat=(116.4718, 23.1918, 116.8644, 23.5521), label='Shantou (UTM 50N)'),
    'sheffield': dict(
        bounds=(579377, 5893687, 620406, 5934622), crs=32630,
        lonlat=(-1.802, 53.1865, -1.198, 53.5469), label='Sheffield (UTM 30N)'),
    'shenyeng': dict(
        bounds=(517089, 4608351, 557460, 4648574), crs=32651,
        lonlat=(123.2063, 41.6267, 123.6898, 41.9871), label='Shenyeng (UTM 51N)'),
    'shenzhen': dict(
        bounds=(173068, 2466546, 234455, 2527603), crs=32650,
        lonlat=(113.8275, 22.284, 114.4128, 22.8246), label='Shenzhen (UTM 50N)'),
    'shijiazhuang': dict(
        bounds=(258044, 4194070, 299287, 4235146), crs=32650,
        lonlat=(114.2492, 37.8718, 114.7068, 38.2321), label='Shijiazhuang (UTM 50N)'),
    'shiraz': dict(
        bounds=(631499, 3258802, 672181, 3299287), crs=32639,
        lonlat=(52.3608, 29.4517, 52.7754, 29.8121), label='Shiraz (UTM 39N)'),
    'shuyang': dict(
        bounds=(643148, 3757669, 683987, 3798324), crs=32650,
        lonlat=(118.5557, 33.9497, 118.991, 34.31), label='Shuyang (UTM 50N)'),
    'singapore': dict(
        bounds=(342388, 113274, 402578, 173055), crs=32648,
        lonlat=(103.5835, 1.0247, 104.1242, 1.5652), label='Singapore (UTM 48N)'),
    'skopje': dict(
        bounds=(523811, 4637814, 548024, 4661941), crs=32634,
        lonlat=(21.288, 41.8919, 21.5789, 42.1081), label='Skopje (UTM 34N)'),
    'sofia': dict(
        bounds=(669060, 4707935, 710318, 4749058), crs=32634,
        lonlat=(23.0696, 42.5051, 23.5598, 42.8655), label='Sofia (UTM 34N)'),
    'spokane': dict(
        bounds=(456332, 5267642, 480567, 5291801), crs=32611,
        lonlat=(-117.5805, 47.5619, -117.2594, 47.7781), label='Spokane (UTM 11N)'),
    'st_louis': dict(
        bounds=(719430, 4259526, 760791, 4300725), crs=32615,
        lonlat=(-90.4726, 38.4568, -90.0113, 38.8171), label='St. Louis (UTM 15N)'),
    'st_petersburg': dict(
        bounds=(328938, 6627065, 370742, 6668799), crs=32636,
        lonlat=(29.9544, 59.7608, 30.6738, 60.1211), label='St. Petersburg (UTM 36N)'),
    'stockholm': dict(
        bounds=(313799, 6562048, 355728, 6603910), crs=32634,
        lonlat=(17.7419, 59.1725, 18.4489, 59.5329), label='Stockholm (UTM 34N)'),
    'stuttgart': dict(
        bounds=(494588, 5382969, 534902, 5423139), crs=32632,
        lonlat=(8.9266, 48.5998, 9.4734, 48.9602), label='Stuttgart (UTM 32N)'),
    'sucre': dict(
        bounds=(249990, 7880977, 274378, 7905224), crs=32720,
        lonlat=(-65.3739, -19.1491, -65.1452, -18.9329), label='Sucre (UTM 20S)'),
    'suining': dict(
        bounds=(530850, 3358178, 571174, 3398307), crs=32648,
        lonlat=(105.3222, 30.3551, 105.7406, 30.7155), label='Suining (UTM 48N)'),
    'surabaya': dict(
        bounds=(672969, 9178528, 713234, 9218542), crs=32749,
        lonlat=(112.5673, -7.4275, 112.9305, -7.0671), label='Surabaya (UTM 49S)'),
    'surat': dict(
        bounds=(255211, 2325810, 295901, 2366267), crs=32643,
        lonlat=(72.6448, 21.0217, 73.0314, 21.3821), label='Surat (UTM 43N)'),
    'suva': dict(
        bounds=(640393, 7982441, 664663, 8006556), crs=32760,
        lonlat=(178.3279, -18.2411, 178.5555, -18.0249), label='Suva (UTM 60S)'),
    'suzhou': dict(
        bounds=(477743, 3702052, 517973, 3742024), crs=32650,
        lonlat=(116.7605, 33.4579, 117.1934, 33.8182), label='Suzhou (UTM 50N)'),
    'suzhou_cn': dict(
        bounds=(252752, 3445173, 293762, 3486000), crs=32651,
        lonlat=(120.4072, 31.1222, 120.8289, 31.4826), label='Suzhou (UTM 51N)'),
    'sydney': dict(
        bounds=(311584, 6225088, 352443, 6265756), crs=32756,
        lonlat=(150.9661, -34.0982, 151.4004, -33.7379), label='Sydney (UTM 56S)'),
    'syracuse': dict(
        bounds=(394102, 4754850, 418530, 4779189), crs=32618,
        lonlat=(-76.298, 42.9419, -76.0021, 43.1581), label='Syracuse (UTM 18N)'),
    'tabriz': dict(
        bounds=(593636, 4196159, 634343, 4236700), crs=32638,
        lonlat=(46.0704, 37.9081, 46.5282, 38.2684), label='Tabriz (UTM 38N)'),
    'taian': dict(
        bounds=(490521, 3986366, 530761, 4026384), crs=32650,
        lonlat=(116.8948, 36.0218, 117.3414, 36.3821), label='Taian (UTM 50N)'),
    'taichung': dict(
        bounds=(244004, 2652739, 284806, 2693328), crs=32651,
        lonlat=(120.4842, 23.9719, 120.8791, 24.3323), label='Taichung (UTM 51N)'),
    'tainan': dict(
        bounds=(192491, 2525929, 233409, 2566626), crs=32651,
        lonlat=(120.0043, 22.8198, 120.3958, 23.1802), label='Tainan (UTM 51N)'),
    'taipei': dict(
        bounds=(325079, 2739468, 385909, 2799967), crs=32651,
        lonlat=(121.27, 24.7656, 121.8666, 25.3061), label='Taipei (UTM 51N)'),
    'taiyuan': dict(
        bounds=(615349, 4172989, 656169, 4213632), crs=32649,
        lonlat=(112.3148, 37.6968, 112.7714, 38.0571), label='Taiyuan (UTM 49N)'),
    'tallahassee': dict(
        bounds=(748872, 3359526, 773542, 3384077), crs=32616,
        lonlat=(-84.4054, 30.3419, -84.1546, 30.5581), label='Tallahassee (UTM 16N)'),
    'tallinn': dict(
        bounds=(358632, 6578143, 383560, 6603032), crs=32635,
        lonlat=(24.5155, 59.3258, 24.9406, 59.542), label='Tallinn (UTM 35N)'),
    'tampa': dict(
        bounds=(335981, 3072224, 376590, 3112623), crs=32617,
        lonlat=(-82.6645, 27.7688, -82.2566, 28.1291), label='Tampa (UTM 17N)'),
    'tangshan': dict(
        bounds=(582048, 4366721, 622738, 4407254), crs=32650,
        lonlat=(117.9585, 39.4461, 118.4264, 39.8065), label='Tangshan (UTM 50N)'),
    'tarawa': dict(
        bounds=(712441, 136039, 736538, 159971), crs=32659,
        lonlat=(172.9094, 1.2301, 173.1257, 1.4463), label='Tarawa (UTM 59N)'),
    'tashkent': dict(
        bounds=(504432, 4553578, 544727, 4593713), crs=32642,
        lonlat=(69.0531, 41.1335, 69.5329, 41.4938), label='Tashkent (UTM 42N)'),
    'tbilisi': dict(
        bounds=(462252, 4599460, 502527, 4639561), crs=32638,
        lonlat=(44.5474, 41.5468, 45.0303, 41.9071), label='Tbilisi (UTM 38N)'),
    'tegucigalpa': dict(
        bounds=(464265, 1547276, 488348, 1571209), crs=32616,
        lonlat=(-87.3309, 13.9959, -87.108, 14.2121), label='Tegucigalpa (UTM 16N)'),
    'tehran': dict(
        bounds=(508090, 3917806, 568565, 3978023), crs=32639,
        lonlat=(51.0897, 35.4036, 51.7551, 35.9442), label='Tehran (UTM 39N)'),
    'tel_aviv_yafo': dict(
        bounds=(646504, 3530606, 687303, 3571210), crs=32636,
        lonlat=(34.5554, 31.9018, 34.9807, 32.2621), label='Tel Aviv-Yafo (UTM 36N)'),
    'the_hague': dict(
        bounds=(566668, 5750356, 607544, 5791130), crs=32631,
        lonlat=(3.9768, 51.8999, 4.5631, 52.2602), label='The Hague (UTM 31N)'),
    'thimphu': dict(
        bounds=(748494, 3029367, 773095, 3053842), crs=32645,
        lonlat=(89.5172, 27.3649, 89.7609, 27.5811), label='Thimphu (UTM 45N)'),
    'tianjin': dict(
        bounds=(486951, 4301429, 547415, 4361553), crs=32650,
        lonlat=(116.8496, 38.8617, 117.5465, 39.4022), label='Tianjin (UTM 50N)'),
    'tianshui': dict(
        bounds=(563963, 3809147, 604471, 3849468), crs=32648,
        lonlat=(105.6991, 34.4218, 106.1369, 34.7821), label='Tianshui (UTM 48N)'),
    'tijuana': dict(
        bounds=(472177, 3576106, 512396, 3616084), crs=32611,
        lonlat=(-117.2956, 32.3218, -116.8683, 32.6821), label='Tijuana (UTM 11N)'),
    'tirana': dict(
        bounds=(388926, 4563632, 413346, 4587962), crs=32634,
        lonlat=(19.6749, 41.2194, 19.9628, 41.4356), label='Tirana (UTM 34N)'),
    'tokyo': dict(
        bounds=(356251, 3919642, 417235, 3980360), crs=32654,
        lonlat=(139.4167, 35.4167, 140.0822, 35.9572), label='Tokyo (UTM 54N)'),
    'toluca': dict(
        bounds=(409260, 2117727, 449531, 2157764), crs=32614,
        lonlat=(-99.8629, 19.1521, -99.481, 19.5125), label='Toluca (UTM 14N)'),
    'toronto': dict(
        bounds=(596591, 4809460, 657980, 4870637), crs=32617,
        lonlat=(-79.7958, 43.4317, -79.0481, 43.9722), label='Toronto (UTM 17N)'),
    'tripoli': dict(
        bounds=(309305, 3620533, 350145, 3661187), crs=32633,
        lonlat=(12.9654, 32.7123, 13.3946, 33.0727), label='Tripoli (UTM 33N)'),
    'tucson': dict(
        bounds=(498141, 3551387, 522249, 3575387), crs=32612,
        lonlat=(-111.0197, 32.0988, -110.7642, 32.3151), label='Tucson (UTM 12N)'),
    'tulsa': dict(
        bounds=(223859, 3988886, 248683, 4013603), crs=32615,
        lonlat=(-96.0639, 36.0119, -95.7962, 36.2281), label='Tulsa (UTM 15N)'),
    'tunis': dict(
        bounds=(584965, 4053432, 625603, 4093909), crs=32632,
        lonlat=(9.9547, 36.6226, 10.4047, 36.983), label='Tunis (UTM 32N)'),
    'turin': dict(
        bounds=(374678, 4971539, 415505, 5012226), crs=32632,
        lonlat=(7.4129, 44.8922, 7.9232, 45.2525), label='Turin (UTM 32N)'),
    'ulaanbaatar': dict(
        bounds=(630726, 5296727, 655423, 5321353), crs=32648,
        lonlat=(106.7534, 47.8105, 107.076, 48.0267), label='Ulaanbaatar (UTM 48N)'),
    'urumqi': dict(
        bounds=(525934, 4830475, 566371, 4870769), crs=32645,
        lonlat=(87.3234, 43.6268, 87.8227, 43.9871), label='Urumqi (UTM 45N)'),
    'vadodara': dict(
        bounds=(291996, 2448313, 332609, 2488694), crs=32643,
        lonlat=(72.9833, 22.1318, 73.3728, 22.4921), label='Vadodara (UTM 43N)'),
    'vaduz': dict(
        bounds=(527079, 5208071, 551338, 5232256), crs=32632,
        lonlat=(9.3578, 47.0256, 9.6756, 47.2418), label='Vaduz (UTM 32N)'),
    'valencia': dict(
        bounds=(591387, 1111245, 631623, 1151225), crs=32619,
        lonlat=(-68.1651, 10.0517, -67.7989, 10.4121), label='Valencia (UTM 19N)'),
    'valletta': dict(
        bounds=(444091, 3960891, 468296, 3984991), crs=32633,
        lonlat=(14.3813, 35.7916, 14.6482, 36.0078), label='Valletta (UTM 33N)'),
    'vancouver': dict(
        bounds=(470813, 5438039, 511140, 5478170), crs=32610,
        lonlat=(-123.3998, 49.0952, -122.8474, 49.4555), label='Vancouver (UTM 10N)'),
    'varanasi': dict(
        bounds=(680766, 2782966, 721501, 2823478), crs=32644,
        lonlat=(82.7987, 25.1518, 83.1974, 25.5121), label='Varanasi (UTM 44N)'),
    'vatican_city': dict(
        bounds=(276340, 4629825, 301151, 4654548), crs=32633,
        lonlat=(12.3081, 41.7952, 12.5986, 42.0114), label='Vatican City (UTM 33N)'),
    'victoria': dict(
        bounds=(315992, 9477547, 340122, 9501507), crs=32740,
        lonlat=(55.3415, -4.7247, 55.5585, -4.5085), label='Victoria (UTM 40S)'),
    'vienna': dict(
        bounds=(581032, 5319303, 621911, 5360062), crs=32633,
        lonlat=(16.0944, 48.0218, 16.635, 48.3821), label='Vienna (UTM 33N)'),
    'vientiane': dict(
        bounds=(233612, 1976025, 258006, 2000273), crs=32648,
        lonlat=(102.4863, 17.8586, 102.7136, 18.0748), label='Vientiane (UTM 48N)'),
    'vila_velha': dict(
        bounds=(455874, 335628, 495979, 375462), crs=32622,
        lonlat=(-51.3971, 3.0365, -51.0362, 3.3968), label='Vila Velha (UTM 22N)'),
    'vila_velha_br': dict(
        bounds=(342199, 7727169, 382642, 7767382), crs=32724,
        lonlat=(-40.5102, -20.5478, -40.1258, -20.1874), label='Vila Velha (UTM 24S)'),
    'vilnius': dict(
        bounds=(379100, 6048562, 403785, 6073194), crs=32635,
        lonlat=(25.1296, 54.5753, 25.5036, 54.7915), label='Vilnius (UTM 35N)'),
    'virginia_beach': dict(
        bounds=(392291, 4059086, 432845, 4099468), crs=32618,
        lonlat=(-76.2054, 36.675, -75.7551, 37.0353), label='Virginia Beach (UTM 18N)'),
    'vishakhapatnam': dict(
        bounds=(723935, 1941848, 764558, 1982228), crs=32644,
        lonlat=(83.1139, 17.5518, 83.4922, 17.9121), label='Vishakhapatnam (UTM 44N)'),
    'vitoria': dict(
        bounds=(336937, 7732168, 377393, 7772382), crs=32724,
        lonlat=(-40.5601, -20.5022, -40.1758, -20.1419), label='Vitoria (UTM 24S)'),
    'wanzhou': dict(
        bounds=(230704, 3392120, 271786, 3433014), crs=32649,
        lonlat=(108.1902, 30.6398, 108.6098, 31.0002), label='Wanzhou (UTM 49N)'),
    'warangal': dict(
        bounds=(329433, 1971784, 369868, 2011975), crs=32644,
        lonlat=(79.3905, 17.8298, 79.7695, 18.1902), label='Warangal (UTM 44N)'),
    'warsaw': dict(
        bounds=(479691, 5769024, 520041, 5809141), crs=32634,
        lonlat=(20.7037, 52.0718, 21.2924, 52.4321), label='Warsaw (UTM 34N)'),
    'washington_d_c': dict(
        bounds=(305013, 4287353, 346056, 4328238), crs=32618,
        lonlat=(-77.2429, 38.7213, -76.7798, 39.0817), label='Washington, D.C. (UTM 18N)'),
    'weifang': dict(
        bounds=(666909, 4045722, 707941, 4086577), crs=32650,
        lonlat=(118.8734, 36.5422, 119.323, 36.9025), label='Weifang (UTM 50N)'),
    'wellington': dict(
        bounds=(302039, 5413249, 326749, 5437869), crs=32760,
        lonlat=(174.6394, -41.4081, 174.9272, -41.1919), label='Wellington (UTM 60S)'),
    'wenzhou': dict(
        bounds=(248274, 3081519, 289200, 3122236), crs=32651,
        lonlat=(120.444, 27.8417, 120.8523, 28.2021), label='Wenzhou (UTM 51N)'),
    'west_palm_beach': dict(
        bounds=(566976, 2938411, 607376, 2978607), crs=32617,
        lonlat=(-80.3254, 26.5648, -79.9219, 26.9252), label='West Palm Beach (UTM 17N)'),
    'wichita': dict(
        bounds=(634937, 4163861, 659463, 4188280), crs=32614,
        lonlat=(-97.4667, 37.6119, -97.1933, 37.8281), label='Wichita (UTM 14N)'),
    'windhoek': dict(
        bounds=(702033, 7490439, 726445, 7514719), crs=32733,
        lonlat=(16.9665, -22.6781, 17.2006, -22.4619), label='Windhoek (UTM 33S)'),
    'wuhan': dict(
        bounds=(207084, 3355802, 268775, 3417195), crs=32650,
        lonlat=(113.9541, 30.3117, 114.582, 30.8522), label='Wuhan (UTM 50N)'),
    'wuxi': dict(
        bounds=(222978, 3476815, 264120, 3517762), crs=32651,
        lonlat=(120.0865, 31.4018, 120.5095, 31.7621), label='Wuxi (UTM 51N)'),
    'xiamen': dict(
        bounds=(589076, 2684601, 629516, 2724811), crs=32650,
        lonlat=(117.8801, 24.2718, 118.276, 24.6321), label='Xiamen (UTM 50N)'),
    'xian': dict(
        bounds=(285513, 3774500, 326493, 3815304), crs=32649,
        lonlat=(108.675, 34.0968, 109.1111, 34.4572), label='Xian (UTM 49N)'),
    'xiangtan': dict(
        bounds=(666741, 3061831, 707504, 3102377), crs=32649,
        lonlat=(112.6962, 27.6703, 113.1038, 28.0306), label='Xiangtan (UTM 49N)'),
    'xiantao': dict(
        bounds=(713836, 3342183, 754839, 3382992), crs=32649,
        lonlat=(113.2293, 30.1922, 113.6469, 30.5525), label='Xiantao (UTM 49N)'),
    'xinyang': dict(
        bounds=(202746, 3538355, 243990, 3579421), crs=32650,
        lonlat=(113.8553, 31.9521, 114.2808, 32.3125), label='Xinyang (UTM 50N)'),
    'xuzhou': dict(
        bounds=(496310, 3773443, 536535, 3813466), crs=32650,
        lonlat=(116.96, 34.1018, 117.3961, 34.4621), label='Xuzhou (UTM 50N)'),
    'yamoussoukro': dict(
        bounds=(236407, 742253, 260604, 766285), crs=32630,
        lonlat=(-5.3844, 6.7103, -5.1666, 6.9265), label='Yamoussoukro (UTM 30N)'),
    'yantai': dict(
        bounds=(338000, 4134830, 378832, 4175492), crs=32651,
        lonlat=(121.1709, 37.3522, 121.6253, 37.7125), label='Yantai (UTM 51N)'),
    'yaounde': dict(
        bounds=(759154, 408036, 799413, 448019), crs=32632,
        lonlat=(11.3341, 3.6885, 11.6953, 4.0488), label='Yaounde (UTM 32N)'),
    'yekaterinburg': dict(
        bounds=(332634, 6282773, 374220, 6324286), crs=32641,
        lonlat=(60.2685, 56.6718, 60.9275, 57.0322), label='Yekaterinburg (UTM 41N)'),
    'yerevan': dict(
        bounds=(438185, 4428110, 478553, 4468333), crs=32638,
        lonlat=(44.2758, 40.0029, 44.7474, 40.3633), label='Yerevan (UTM 38N)'),
    'yiyang': dict(
        bounds=(609602, 3144486, 650176, 3184853), crs=32649,
        lonlat=(112.1229, 28.4222, 112.5333, 28.7825), label='Yiyang (UTM 49N)'),
    'yokohama': dict(
        bounds=(352700, 3901458, 393417, 3941989), crs=32654,
        lonlat=(139.3809, 35.2505, 139.8232, 35.6108), label='Yokohama (UTM 54N)'),
    'zagreb': dict(
        bounds=(565531, 5060164, 589937, 5084486), crs=32633,
        lonlat=(15.8449, 45.6919, 16.1551, 45.9081), label='Zagreb (UTM 33N)'),
    'zaozhuang': dict(
        bounds=(531768, 3840030, 572140, 3880213), crs=32650,
        lonlat=(117.3484, 34.7018, 117.7877, 35.0621), label='Zaozhuang (UTM 50N)'),
    'zhangzhou': dict(
        bounds=(547745, 2691974, 588064, 2732074), crs=32650,
        lonlat=(117.472, 24.3402, 117.8681, 24.7006), label='Zhangzhou (UTM 50N)'),
    'zhanjiang': dict(
        bounds=(415287, 2324609, 455560, 2364656), crs=32649,
        lonlat=(110.1848, 21.0217, 110.5713, 21.3821), label='Zhanjiang (UTM 49N)'),
    'zhengzhou': dict(
        bounds=(723195, 3828827, 764420, 3869862), crs=32649,
        lonlat=(113.4438, 34.5768, 113.8825, 34.9371), label='Zhengzhou (UTM 49N)'),
    'zhongli': dict(
        bounds=(299624, 2742048, 340284, 2782494), crs=32651,
        lonlat=(121.018, 24.7848, 121.4155, 25.1452), label='Zhongli (UTM 51N)'),
    'zibo': dict(
        bounds=(573245, 4053234, 613837, 4093644), crs=32650,
        lonlat=(117.823, 36.6218, 118.2731, 36.9821), label='Zibo (UTM 50N)'),
}
