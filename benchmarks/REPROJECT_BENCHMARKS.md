# Reproject Benchmarks

Generated: 2026-03-19 20:23 UTC

Compares xrspatial reproject (numpy backend, numba JIT) against rioxarray (GDAL warp) across 4 raster sizes, 2 CRS transforms, and 3 resampling methods. The **fastest** time in each row is bold.

- **Test data:** synthetic terrain (sin/cos + Gaussian + noise) in EPSG:4326
- **Transform:** identity (4326 to 4326) and geographic to Web Mercator (4326 to 3857)
- **Resampling:** nearest, bilinear, cubic
- **Consistency:** sampled at matching geographic coordinates (interior 90%), correlation and RMSE reported

## identity

### Timings

| size | xrs-nearest | rio-nearest | xrs-bilinear | rio-bilinear | xrs-cubic | rio-cubic |
|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | **1.6 ms** | 3.3 ms | **1.4 ms** | 4.1 ms | **3.1 ms** | 5.8 ms |
| 512x512 | **4.5 ms** | 6.7 ms | **6.2 ms** | 11 ms | **6.4 ms** | 19 ms |
| 1024x1024 | **22 ms** | 24 ms | **26 ms** | 39 ms | **24 ms** | 68 ms |
| 2048x2048 | 105 ms | **86 ms** | **103 ms** | 146 ms | **105 ms** | 266 ms |

### Relative to rioxarray

Values below 1.0 mean xrspatial is faster.

| size | nearest | bilinear | cubic |
|---:|---:|---:|---:|
| 256x256 | 0.49x | 0.34x | 0.54x |
| 512x512 | 0.68x | 0.56x | 0.34x |
| 1024x1024 | 0.92x | 0.66x | 0.35x |
| 2048x2048 | 1.22x | 0.70x | 0.40x |

### Consistency

| size | method | correlation | RMSE | max |Δ| | median rel |
|---:|:------|---:|---:|---:|---:|
| 256x256 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 256x256 | bilinear | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 256x256 | cubic | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 512x512 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 512x512 | bilinear | 1.000000 | 0.0000 | 0.0000 | 1.22e-16 |
| 512x512 | cubic | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 1024x1024 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 1024x1024 | bilinear | 1.000000 | 0.0000 | 0.0000 | 1.91e-16 |
| 1024x1024 | cubic | 1.000000 | 0.0000 | 0.0000 | 1.63e-16 |
| 2048x2048 | nearest | 1.000000 | 0.0000 | 0.0000 | 0.00e+00 |
| 2048x2048 | bilinear | 1.000000 | 0.0000 | 0.0000 | 2.53e-16 |
| 2048x2048 | cubic | 1.000000 | 0.0000 | 0.0000 | 1.79e-16 |

## 4326 to 3857

### Timings

| size | xrs-nearest | rio-nearest | xrs-bilinear | rio-bilinear | xrs-cubic | rio-cubic |
|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | **1.6 ms** | 3.3 ms | **1.7 ms** | 4.3 ms | **2.1 ms** | 6.1 ms |
| 512x512 | **2.2 ms** | 5.4 ms | **2.3 ms** | 9.3 ms | **4.2 ms** | 17 ms |
| 1024x1024 | 20 ms | **17 ms** | **22 ms** | 34 ms | **26 ms** | 61 ms |
| 2048x2048 | 106 ms | **89 ms** | **108 ms** | 155 ms | **111 ms** | 272 ms |

### Relative to rioxarray

Values below 1.0 mean xrspatial is faster.

| size | nearest | bilinear | cubic |
|---:|---:|---:|---:|
| 256x256 | 0.49x | 0.39x | 0.34x |
| 512x512 | 0.41x | 0.25x | 0.25x |
| 1024x1024 | 1.14x | 0.65x | 0.42x |
| 2048x2048 | 1.18x | 0.70x | 0.41x |

### Consistency

| size | method | correlation | RMSE | max |Δ| | median rel |
|---:|:------|---:|---:|---:|---:|
| 256x256 | nearest | 0.999929 | 1.7851 | 10.3843 | 2.15e-03 |
| 256x256 | bilinear | 0.999996 | 0.4451 | 3.3809 | 4.12e-04 |
| 256x256 | cubic | 0.999995 | 0.4599 | 3.2766 | 4.70e-04 |
| 512x512 | nearest | 0.999949 | 1.5032 | 8.3315 | 1.87e-03 |
| 512x512 | bilinear | 0.999995 | 0.4683 | 3.7009 | 4.49e-04 |
| 512x512 | cubic | 0.999994 | 0.5055 | 3.8156 | 5.15e-04 |
| 1024x1024 | nearest | 0.999974 | 1.0722 | 6.9123 | 1.17e-03 |
| 1024x1024 | bilinear | 0.999995 | 0.4719 | 3.5555 | 4.46e-04 |
| 1024x1024 | cubic | 0.999995 | 0.4944 | 3.8284 | 5.07e-04 |
| 2048x2048 | nearest | 0.999980 | 0.9508 | 6.8803 | 1.04e-03 |
| 2048x2048 | bilinear | 0.999995 | 0.4739 | 3.8786 | 4.65e-04 |
| 2048x2048 | cubic | 0.999994 | 0.5038 | 3.5338 | 5.28e-04 |

## merge()

Two overlapping tiles merged into a single mosaic. Tiles overlap by ~10% in the center.

### Timings

| size | strategy | xrspatial | rioxarray |
|---:|:------|---:|---:|
| 256x256 | first | **4.7 ms** | 9.9 ms |
| 256x256 | mean | **3.3 ms** | 9.6 ms |
| 512x512 | first | **8.6 ms** | 14 ms |
| 512x512 | mean | **9.6 ms** | 16 ms |
| 1024x1024 | first | **33 ms** | 39 ms |
| 1024x1024 | mean | 42 ms | **40 ms** |
| 2048x2048 | first | 252 ms | **149 ms** |
| 2048x2048 | mean | 375 ms | **150 ms** |

## Dask out-of-core

Reproject with dask-backed input (chunk_size=512). Measures graph construction time (lazy) and full compute time.

| size | method | graph build | compute | total |
|---:|:------|---:|---:|---:|
| 1024x1024 | bilinear | 71 ms | 30 ms | 101 ms |
| 2048x2048 | bilinear | 10 ms | 118 ms | 128 ms |
| 4096x4096 | bilinear | 127 ms | 1.15 s | 1.28 s |

## Where xrspatial wins and loses

| Resampling | Small (256) | Medium (512) | Large (1024) | XL (2048) |
|:-----------|:------------|:-------------|:-------------|:----------|
| bilinear (4326 to 3857) | **xrs 2.0x** | **xrs 2.7x** | **xrs 2.5x** | **xrs 1.8x** |
| cubic (4326 to 3857) | **xrs 2.7x** | **xrs 4.5x** | **xrs 5.7x** | **xrs 2.2x** |
| nearest (4326 to 3857) | **xrs 1.6x** | **xrs 1.4x** | **xrs 1.4x** | rio 1.3x |
| bilinear (identity) | **xrs 1.9x** | **xrs 2.3x** | **xrs 2.5x** | **xrs 2.9x** |
| cubic (identity) | **xrs 2.8x** | **xrs 4.0x** | **xrs 5.3x** | **xrs 5.8x** |
| nearest (identity) | **xrs 1.4x** | **xrs 1.3x** | **xrs 1.3x** | **xrs 1.3x** |

Bold = winner by >20%. 'xrs 2.0x' means xrspatial is 2x faster. 'rio 1.5x' means rioxarray is 1.5x faster.
