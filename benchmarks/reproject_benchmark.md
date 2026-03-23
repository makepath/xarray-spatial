# Reproject Module: Comprehensive Benchmarks

Generated: 2026-03-22

Hardware: AMD Ryzen / NVIDIA A6000 GPU, PCIe Gen4, NVMe SSD

Python 3.14, NumPy, Numba, CuPy, Dask, pyproj, rioxarray (GDAL)

---

## 1. Full Pipeline Benchmark (read -> reproject -> write)

Source file: Copernicus DEM COG (`Copernicus_DSM_COG_10_N40_00_W075_00_DEM.tif`), 3600x3600, WGS84, deflate+floating-point predictor. Reprojected to Web Mercator (EPSG:3857). Median of 3 runs after warmup.

```python
from xrspatial.geotiff import read_geotiff, write_geotiff
from xrspatial.reproject import reproject

dem = read_geotiff('Copernicus_DSM_COG_10_N40_00_W075_00_DEM.tif')
dem_merc = reproject(dem, 'EPSG:3857')
write_geotiff(dem_merc, 'output.tif')
```

All times measured with warm Numba/CUDA kernels (first call incurs ~4.5s JIT compilation).

| Backend | End-to-end | Reproject only | vs rioxarray (reproject) |
|:--------|----------:|--------------:|:------------------------|
| CuPy GPU | 747 ms | 73 ms | **2.0x faster** |
| Dask+CuPy GPU | 782 ms | ~80 ms | ~1.8x faster |
| rioxarray (GDAL) | 411 ms | 144 ms | 1.0x |
| NumPy | 2,907 ms | 413 ms | 0.3x |

The CuPy reproject is 2x faster than rioxarray for the coordinate transform + resampling. The end-to-end gap is due to I/O: rioxarray uses rasterio's C-level compressed read/write, while our geotiff reader is pure Python/Numba. For reproject-only workloads (data already in memory), CuPy is the clear winner.

**Note on JIT warmup**: The first `reproject()` call compiles the Numba kernels (~4.5s). All subsequent calls run at full speed. For long-running applications or batch processing, this is amortized over many calls.

---

## 2. Projection Coverage and Accuracy

Each projection was tested with 5 geographically appropriate points. "Max error vs pyproj" measures the maximum positional difference between the Numba JIT inverse transform and pyproj's reference implementation. Errors are measured as approximate ground distance.

| Projection | EPSG examples | Max error vs pyproj | CPU Numba | CUDA GPU |
|:-----------|:-------------|--------------------:|:---------:|:--------:|
| Web Mercator | 3857 | < 0.001 mm | yes | yes |
| UTM / Transverse Mercator | 326xx, 327xx | < 0.001 mm | yes | yes |
| Ellipsoidal Mercator | 3395 | < 0.001 mm | yes | yes |
| Lambert Conformal Conic | 2154, 2229 | 0.003 mm | yes | yes |
| Albers Equal Area | 5070 | 3.5 m | yes | yes |
| Cylindrical Equal Area | 6933 | 4.8 m | yes | yes |
| Sinusoidal | MODIS | 0.001 mm | yes | yes |
| Lambert Azimuthal Equal Area | 3035 | see note | yes | yes |
| Polar Stereographic (Antarctic) | 3031 | < 0.001 mm | yes | yes |
| Polar Stereographic (Arctic) | 3413 | < 0.001 mm | yes | yes |
| Oblique Stereographic | custom WGS84 | < 0.001 mm | yes | fallback |
| Oblique Mercator (Hotine) | 3375 | N/A | disabled | fallback |
| State Plane (tmerc) | 26983 | 43 cm | yes | yes |
| State Plane (LCC, ftUS) | 2229 | 19 cm | yes | yes |

**Notes:**
- LAEA Europe (3035): The current implementation has a known latitude bias (~700m near Paris, larger at the projection's edges). This is an area for future improvement; for high-accuracy LAEA work, the pyproj fallback is used for unsupported ellipsoids.
- Albers and CEA: Errors of 3-5m stem from the authalic latitude series approximation. Acceptable for most raster reprojection at typical DEM resolutions (30m+).
- State Plane: Sub-metre accuracy in both tmerc and LCC variants. Unit conversion (US survey feet) is handled internally.
- Oblique Stereographic: The Numba kernel exists and works for WGS84-based CRS definitions. EPSG:28992 (RD New) uses the Bessel ellipsoid without a registered datum, so it falls back to pyproj.
- Oblique Mercator: Kernel implemented but disabled pending alignment with PROJ's omerc.cpp variant handling. Falls back to pyproj.

### Reproject-only timing (1024x1024, bilinear)

| Transform | xrspatial | rioxarray |
|:-----------|----------:|----------:|
| WGS84 -> Web Mercator | 23 ms | 14 ms |
| WGS84 -> UTM 33N | 24 ms | 18 ms |
| WGS84 -> Albers CONUS | 41 ms | 33 ms |
| WGS84 -> LAEA Europe | 57 ms | 17 ms |
| WGS84 -> Polar Stere S | 44 ms | 38 ms |
| WGS84 -> LCC France | 44 ms | 25 ms |
| WGS84 -> Ellipsoidal Merc | 27 ms | 14 ms |
| WGS84 -> CEA EASE-Grid | 24 ms | 15 ms |

At 1024x1024, rioxarray (GDAL) is generally faster than the NumPy backend for reproject-only workloads. The GPU backend closes this gap and pulls ahead for larger rasters (see Section 1). The xrspatial advantage is its pure-Python stack with no GDAL dependency, four-backend dispatch (NumPy/CuPy/Dask/Dask+CuPy), and integrated vertical/datum handling.

### Merge timing (4 overlapping same-CRS tiles)

| Tile size | xrspatial | rioxarray | Speedup |
|:----------|----------:|----------:|--------:|
| 512x512 | 16 ms | 29 ms | 1.8x |
| 1024x1024 | 52 ms | 76 ms | 1.5x |
| 2048x2048 | 361 ms | 280 ms | 0.8x |

Same-CRS merge skips reprojection and places tiles by coordinate alignment. xrspatial is faster at small to medium sizes; rioxarray catches up at larger sizes due to its C-level copy routines.

---

## 3. Datum Shift Coverage

The reproject module handles horizontal datum shifts for non-WGS84 source CRS. It first tries grid-based shifts (downloaded from the PROJ CDN on first use), falling back to 7-parameter Helmert transforms when no grid is available.

### Grid-based shifts (sub-metre accuracy)

| Registry key | Grid file | Coverage | Description |
|:-------------|:----------|:---------|:------------|
| NAD27_CONUS | us_noaa_conus.tif | CONUS | NAD27 -> NAD83 (NADCON) |
| NAD27_NADCON5_CONUS | us_noaa_nadcon5_nad27_nad83_1986_conus.tif | CONUS | NAD27 -> NAD83 (NADCON5, preferred) |
| NAD27_ALASKA | us_noaa_alaska.tif | Alaska | NAD27 -> NAD83 (NADCON) |
| NAD27_HAWAII | us_noaa_hawaii.tif | Hawaii | Old Hawaiian -> NAD83 |
| NAD27_PRVI | us_noaa_prvi.tif | PR/USVI | NAD27 -> NAD83 |
| OSGB36_UK | uk_os_OSTN15_NTv2_OSGBtoETRS.tif | UK | OSGB36 -> ETRS89 (OSTN15) |
| AGD66_GDA94 | au_icsm_A66_National_13_09_01.tif | Australia NT | AGD66 -> GDA94 |
| DHDN_ETRS89_DE | de_adv_BETA2007.tif | Germany | DHDN -> ETRS89 |
| MGI_ETRS89_AT | at_bev_AT_GIS_GRID.tif | Austria | MGI -> ETRS89 |
| ED50_ETRS89_ES | es_ign_SPED2ETV2.tif | Spain (E coast) | ED50 -> ETRS89 |
| RD_ETRS89_NL | nl_nsgi_rdcorr2018.tif | Netherlands | RD -> ETRS89 |
| BD72_ETRS89_BE | be_ign_bd72lb72_etrs89lb08.tif | Belgium | BD72 -> ETRS89 |
| CH1903_ETRS89_CH | ch_swisstopo_CHENyx06_ETRS.tif | Switzerland | CH1903 -> ETRS89 |
| D73_ETRS89_PT | pt_dgt_D73_ETRS89_geo.tif | Portugal | D73 -> ETRS89 |

Grids are downloaded from `cdn.proj.org` on first use and cached in `~/.cache/xrspatial/proj_grids/`. Bilinear interpolation within the grid is done via Numba JIT.

### Helmert fallback (1-5m accuracy)

When no grid covers the area, a 7-parameter (or 3-parameter) geocentric Helmert transform is applied:

| Datum / Ellipsoid | Type | Parameters (dx, dy, dz, rx, ry, rz, ds) |
|:------------------|:-----|:-----------------------------------------|
| NAD27 / Clarke 1866 | 3-param | (-8, 160, 176, 0, 0, 0, 0) |
| OSGB36 / Airy | 7-param | (446.4, -125.2, 542.1, 0.15, 0.25, 0.84, -20.5) |
| DHDN / Bessel | 7-param | (598.1, 73.7, 418.2, 0.20, 0.05, -2.46, 6.7) |
| MGI / Bessel | 7-param | (577.3, 90.1, 463.9, 5.14, 1.47, 5.30, 2.42) |
| ED50 / Intl 1924 | 7-param | (-87, -98, -121, 0, 0, 0.81, -0.38) |
| BD72 / Intl 1924 | 7-param | (-106.9, 52.3, -103.7, 0.34, -0.46, 1.84, -1.27) |
| CH1903 / Bessel | 3-param | (674.4, 15.1, 405.3, 0, 0, 0, 0) |
| D73 / Intl 1924 | 3-param | (-239.7, 88.2, 30.5, 0, 0, 0, 0) |
| AGD66 / ANS | 3-param | (-133, -48, 148, 0, 0, 0, 0) |
| Tokyo / Bessel | 3-param | (-146.4, 507.3, 680.5, 0, 0, 0, 0) |

Grid-based accuracy is typically 0.01-0.1m; Helmert fallback accuracy is 1-5m depending on the datum.

---

## 4. Vertical Datum Support

The module provides geoid undulation lookup from EGM96 (vendored, 15-arcminute global grid, 2.6MB) and optionally EGM2008 (25-arcminute, 77MB, downloaded on first use).

### API

```python
from xrspatial.reproject import geoid_height, ellipsoidal_to_orthometric

# Single point
N = geoid_height(-74.0, 40.7)           # New York: -32.86m

# Convert GPS height to map height
H = ellipsoidal_to_orthometric(100.0, -74.0, 40.7)  # 132.86m

# Batch (array)
N = geoid_height(lon_array, lat_array)

# Raster grid
from xrspatial.reproject import geoid_height_raster
N_grid = geoid_height_raster(dem)
```

### Accuracy vs pyproj geoid

| Location | xrspatial EGM96 (m) | pyproj EGM96 (m) | Difference |
|:---------|---------------------:|------------------:|-----------:|
| New York (-74.0, 40.7) | -32.86 | -32.77 | 0.09 m |
| Paris (2.35, 48.85) | 44.59 | 44.57 | 0.02 m |
| Tokyo (139.7, 35.7) | 35.75 | 36.80 | 1.06 m |
| Null Island (0.0, 0.0) | 17.15 | 17.16 | 0.02 m |
| Rio (-43.2, -22.9) | -5.59 | -5.43 | 0.16 m |

The 1.06m Tokyo difference is due to the 15-arcminute grid resolution in EGM96; the steep geoid gradient near Japan amplifies interpolation differences. Roundtrip accuracy (`ellipsoidal_to_orthometric` then `orthometric_to_ellipsoidal`) is exact (0.0 error).

### Integration with reproject

The `reproject` function accepts a `vertical_crs` parameter to apply vertical datum shifts during reprojection:

```python
from xrspatial.reproject import reproject

# Reproject and convert ellipsoidal heights to orthometric (MSL)
dem_merc = reproject(
    dem, 'EPSG:3857',
    src_vertical_crs='ellipsoidal',
    tgt_vertical_crs='EGM96',
)
```

---

## 5. ITRF Frame Support

Time-dependent transformations between International Terrestrial Reference Frames using 14-parameter Helmert transforms (7 static + 7 rates) from PROJ data files.

### Available frames

- ITRF2000
- ITRF2008
- ITRF2014
- ITRF2020

### Example

```python
from xrspatial.reproject import itrf_transform, itrf_frames

print(itrf_frames())  # ['ITRF2000', 'ITRF2008', 'ITRF2014', 'ITRF2020']

lon2, lat2, h2 = itrf_transform(
    -74.0, 40.7, 10.0,
    src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
)
# -> (-73.9999999782, 40.6999999860, 9.996897)
# Horizontal shift: 2.4 mm, vertical shift: -3.1 mm
```

### All frame-pair shifts (at epoch 2020.0, location 0E 45N)

| Source | Target | Horizontal shift | Vertical shift |
|:-------|:-------|:----------------:|:--------------:|
| ITRF2000 | ITRF2008 | 33.0 mm | 32.8 mm |
| ITRF2000 | ITRF2014 | 33.2 mm | 30.7 mm |
| ITRF2000 | ITRF2020 | 30.5 mm | 30.0 mm |
| ITRF2008 | ITRF2014 | 1.9 mm | -2.1 mm |
| ITRF2008 | ITRF2020 | 2.6 mm | -2.8 mm |
| ITRF2014 | ITRF2020 | 3.0 mm | -0.7 mm |

Shifts between recent frames (ITRF2014/2020) are at the mm level. Older frames (ITRF2000) show larger shifts (~30mm) due to accumulated tectonic motion.

---

## 6. pyproj Usage

The reproject module uses pyproj for metadata operations only. The heavy per-pixel work is done in Numba JIT or CUDA.

### What pyproj does (runs once per reproject call)

| Task | Cost | Description |
|:-----|:-----|:------------|
| CRS metadata parsing | ~1 ms | `CRS.from_user_input()`, `CRS.to_dict()`, extract projection parameters |
| EPSG code lookup | ~0.1 ms | `CRS.to_epsg()` to check for known fast paths |
| Output grid estimation | ~1 ms | `Transformer.transform()` on ~500 boundary points to determine output extent |
| Fallback transform | per-pixel | Only used for CRS pairs without a built-in Numba/CUDA kernel |

### What Numba/CUDA does (the per-pixel bottleneck)

| Task | Implementation | Notes |
|:-----|:---------------|:------|
| Coordinate transforms | Numba `@njit(parallel=True)` / CUDA `@cuda.jit` | Per-pixel forward/inverse projection |
| Bilinear resampling | Numba `@njit` / CUDA `@cuda.jit` | Source pixel interpolation |
| Nearest-neighbor resampling | Numba `@njit` / CUDA `@cuda.jit` | Source pixel lookup |
| Cubic resampling | `scipy.ndimage.map_coordinates` | CPU only (no Numba/CUDA kernel yet) |
| Datum grid interpolation | Numba `@njit(parallel=True)` | Bilinear interp of NTv2/NADCON grids |
| Geoid undulation interpolation | Numba `@njit(parallel=True)` | Bilinear interp of EGM96/EGM2008 grid |
| 7-param Helmert datum shift | Numba `@njit(parallel=True)` | Geocentric ECEF transform |
| 14-param ITRF transform | Numba `@njit(parallel=True)` | Time-dependent Helmert in ECEF |
