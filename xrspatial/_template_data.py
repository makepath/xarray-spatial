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
    # The default (non-preserve) world grid spans the full +/-90 in EPSG:4326.
    # The preserve path uses a +/-85 latitude band (the conventional Web
    # Mercator limit) so 'shape' (World Mercator) does not diverge at the poles.
    'world': dict(bounds=(-180.0, -90.0, 180.0, 90.0), crs=4326,
                  default_resolution=0.5, label='World (WGS84)',
                  lonlat=(-180.0, -85.0, 180.0, 85.0), area_epsg=8857,
                  shape_epsg=3395),
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
