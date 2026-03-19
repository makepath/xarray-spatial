# Reproject Benchmarks

Generated: 2026-03-19 20:50 UTC

Compares xrspatial reproject (numpy backend, numba JIT) against rioxarray (GDAL warp) across 4 raster sizes, 2 CRS transforms, and 3 resampling methods. The **fastest** time in each row is bold.

- **Test data:** synthetic terrain (sin/cos + Gaussian + noise) in EPSG:4326
- **Transform:** identity (4326 to 4326) and geographic to Web Mercator (4326 to 3857)
- **Resampling:** nearest, bilinear, cubic
- **Consistency:** sampled at matching geographic coordinates (interior 90%), correlation and RMSE reported

## identity

### Timings

| size | xrs-nearest | rio-nearest | xrs-bilinear | rio-bilinear | xrs-cubic | rio-cubic |
|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | **1.6 ms** | 3.4 ms | **1.7 ms** | 4.1 ms | **2.4 ms** | 6.0 ms |
| 512x512 | **6.8 ms** | 7.0 ms | **9.0 ms** | 11 ms | **12 ms** | 19 ms |
| 1024x1024 | 31 ms | **23 ms** | **31 ms** | 38 ms | **40 ms** | 69 ms |
| 2048x2048 | 141 ms | **88 ms** | **145 ms** | 158 ms | **181 ms** | 277 ms |

### Relative to rioxarray

Values below 1.0 mean xrspatial is faster.

| size | nearest | bilinear | cubic |
|---:|---:|---:|---:|
| 256x256 | 0.48x | 0.41x | 0.40x |
| 512x512 | 0.97x | 0.79x | 0.63x |
| 1024x1024 | 1.36x | 0.80x | 0.58x |
| 2048x2048 | 1.60x | 0.91x | 0.65x |

### Consistency

| size | method | correlation | RMSE | max |Δ| | median rel |
|---:|:------|---:|---:|---:|---:|
| 256x256 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 256x256 | bilinear | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 256x256 | cubic | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 512x512 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 512x512 | bilinear | 1.000000 | 0.0000 | 0.0000 | 1.24e-16 |
| 512x512 | cubic | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 1024x1024 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 1024x1024 | bilinear | 1.000000 | 0.0000 | 0.0000 | 1.92e-16 |
| 1024x1024 | cubic | 1.000000 | 0.0000 | 0.0000 | 1.64e-16 |
| 2048x2048 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 2048x2048 | bilinear | 1.000000 | 0.0000 | 0.0000 | 2.63e-16 |
| 2048x2048 | cubic | 1.000000 | 0.0000 | 0.0000 | 1.79e-16 |

## 4326 to 3857

### Timings

| size | xrs-nearest | rio-nearest | xrs-bilinear | rio-bilinear | xrs-cubic | rio-cubic |
|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | **2.0 ms** | 3.4 ms | **2.1 ms** | 4.6 ms | **2.8 ms** | 6.6 ms |
| 512x512 | **3.9 ms** | 5.7 ms | **4.3 ms** | 10 ms | **7.0 ms** | 17 ms |
| 1024x1024 | 25 ms | **18 ms** | **33 ms** | 35 ms | **41 ms** | 64 ms |
| 2048x2048 | 123 ms | **68 ms** | **129 ms** | 141 ms | **164 ms** | 254 ms |

### Relative to rioxarray

Values below 1.0 mean xrspatial is faster.

| size | nearest | bilinear | cubic |
|---:|---:|---:|---:|
| 256x256 | 0.59x | 0.45x | 0.42x |
| 512x512 | 0.69x | 0.43x | 0.40x |
| 1024x1024 | 1.41x | 0.95x | 0.65x |
| 2048x2048 | 1.79x | 0.92x | 0.64x |

### Consistency

| size | method | correlation | RMSE | max |Δ| | median rel |
|---:|:------|---:|---:|---:|---:|
| 256x256 | nearest | 0.999844 | 2.6289 | 11.6941 | 3.37e-03 |
| 256x256 | bilinear | 0.999997 | 0.3374 | 2.2414 | 3.07e-04 |
| 256x256 | cubic | 0.999996 | 0.4388 | 3.0041 | 4.08e-04 |
| 512x512 | nearest | 0.999951 | 1.4678 | 7.7334 | 1.50e-03 |
| 512x512 | bilinear | 0.999997 | 0.3659 | 2.5772 | 3.51e-04 |
| 512x512 | cubic | 0.999995 | 0.4613 | 3.0562 | 4.58e-04 |
| 1024x1024 | nearest | 0.999966 | 1.2208 | 7.1378 | 1.30e-03 |
| 1024x1024 | bilinear | 0.999996 | 0.3939 | 3.3103 | 3.49e-04 |
| 1024x1024 | cubic | 0.999995 | 0.4623 | 3.7910 | 4.52e-04 |
| 2048x2048 | nearest | 0.999979 | 0.9678 | 6.3667 | 1.05e-03 |
| 2048x2048 | bilinear | 0.999995 | 0.4543 | 3.9038 | 4.68e-04 |
| 2048x2048 | cubic | 0.999994 | 0.5153 | 3.5623 | 5.55e-04 |

## merge()

Two overlapping tiles merged into a single mosaic. Tiles overlap by ~10% in the center.

### Timings

| size | strategy | xrspatial | rioxarray |
|---:|:------|---:|---:|
| 256x256 | first | **3.1 ms** | 9.6 ms |
| 512x512 | first | **8.5 ms** | 13 ms |
| 1024x1024 | first | 28 ms | **27 ms** |
| 2048x2048 | first | 299 ms | **124 ms** |

## Dask out-of-core

Reproject with dask-backed input (chunk_size=512). Measures graph construction time (lazy) and full compute time.

| size | method | graph build | compute | total |
|---:|:------|---:|---:|---:|
| 1024x1024 | bilinear | 123 ms | 28 ms | 151 ms |
| 2048x2048 | bilinear | 7.8 ms | 89 ms | 96 ms |
| 4096x4096 | bilinear | 95 ms | 984 ms | 1.08 s |

## Where xrspatial wins and loses

| Resampling | Small (256) | Medium (512) | Large (1024) | XL (2048) |
|:-----------|:------------|:-------------|:-------------|:----------|
| bilinear (4326 to 3857) | **xrs 2.2x** | **xrs 2.3x** | **xrs 2.2x** | **xrs 2.0x** |
| cubic (4326 to 3857) | **xrs 2.3x** | **xrs 2.5x** | **xrs 2.7x** | **xrs 2.0x** |
| nearest (4326 to 3857) | **xrs 1.5x** | ~tied | ~tied | ~tied |
| bilinear (identity) | **xrs 2.1x** | **xrs 2.1x** | **xrs 2.1x** | **xrs 2.0x** |
| cubic (identity) | **xrs 2.6x** | **xrs 2.8x** | **xrs 2.6x** | **xrs 2.4x** |
| nearest (identity) | **xrs 2.0x** | **xrs 1.4x** | ~tied | ~tied |

Bold = winner by >20%. 'xrs 2.0x' means xrspatial is 2x faster. 'rio 1.5x' means rioxarray is 1.5x faster.
