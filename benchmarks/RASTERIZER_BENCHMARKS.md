# Rasterizer Benchmarks

Generated: 2026-03-14 04:21 UTC

Compares xarray-spatial (numpy and cupy backends) against datashader, geocube, and rasterio across 10 geometry types, 3 feature counts (50/200/1000), and 5 output resolutions (100-4000 px wide).

- **Polygon types** (circles, irregular, rectangles, stars, donuts, multipolygons): all 5 rasterizers compared
- **Line types** (lines, multilines): xrspatial, geocube, rasterio (datashader uses a different API for lines)
- **Point types** (points, multipoints): xrspatial, geocube, rasterio (datashader uses a different API for points)

## Circles (64 vertices)

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.59 ms | 1.64 ms | 0.42 ms | 3.62 ms | 1.16 ms |
| 50 | 500x250 | 0.83 ms | 1.57 ms | 1.30 ms | 4.67 ms | 1.50 ms |
| 50 | 1000x500 | 1.97 ms | 1.73 ms | 3.83 ms | 7.10 ms | 3.10 ms |
| 50 | 2000x1000 | 7.35 ms | 2.07 ms | 13.7 ms | 18.0 ms | 10.1 ms |
| 50 | 4000x2000 | 49.8 ms | 2.86 ms | 65.5 ms | 94.2 ms | 59.9 ms |
| 200 | 100x50 | 0.98 ms | 1.95 ms | 0.60 ms | 7.18 ms | 4.15 ms |
| 200 | 500x250 | 1.47 ms | 2.61 ms | 3.79 ms | 7.69 ms | 4.49 ms |
| 200 | 1000x500 | 2.49 ms | 2.91 ms | 12.9 ms | 10.2 ms | 6.12 ms |
| 200 | 2000x1000 | 8.65 ms | 3.80 ms | 48.9 ms | 19.5 ms | 13.9 ms |
| 200 | 4000x2000 | 53.0 ms | 33.3 ms | 200 ms | 82.0 ms | 65.4 ms |
| 1000 | 100x50 | 4.61 ms | 9.58 ms | 1.93 ms | 32.9 ms | 21.5 ms |
| 1000 | 500x250 | 5.62 ms | 16.5 ms | 18.0 ms | 28.2 ms | 23.9 ms |
| 1000 | 1000x500 | 8.84 ms | 29.7 ms | 61.0 ms | 31.4 ms | 26.6 ms |
| 1000 | 2000x1000 | 17.9 ms | 54.3 ms | 230 ms | 45.5 ms | 37.8 ms |
| 1000 | 4000x2000 | 79.9 ms | 39.6 ms | 912 ms | 142 ms | 106 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 2.76x | 0.70x | 6.10x | 1.94x |
| 50 | 500x250 | 1.89x | 1.56x | 5.63x | 1.80x |
| 50 | 1000x500 | 0.88x | 1.94x | 3.60x | 1.57x |
| 50 | 2000x1000 | 0.28x | 1.87x | 2.45x | 1.37x |
| 50 | 4000x2000 | 0.06x | 1.32x | 1.89x | 1.20x |
| 200 | 100x50 | 2.00x | 0.62x | 7.36x | 4.25x |
| 200 | 500x250 | 1.78x | 2.58x | 5.23x | 3.05x |
| 200 | 1000x500 | 1.17x | 5.19x | 4.08x | 2.45x |
| 200 | 2000x1000 | 0.44x | 5.65x | 2.25x | 1.60x |
| 200 | 4000x2000 | 0.63x | 3.78x | 1.55x | 1.24x |
| 1000 | 100x50 | 2.08x | 0.42x | 7.14x | 4.67x |
| 1000 | 500x250 | 2.94x | 3.21x | 5.02x | 4.25x |
| 1000 | 1000x500 | 3.36x | 6.90x | 3.55x | 3.00x |
| 1000 | 2000x1000 | 3.04x | 12.85x | 2.54x | 2.11x |
| 1000 | 4000x2000 | 0.49x | 11.41x | 1.78x | 1.32x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.1463 | 0.9166 | 34.9 - 44.4 |
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.1463 | 0.9166 | 34.9 - 44.4 |
| datashader vs rasterio | 0.1463 | 0.9166 | 34.9 - 44.4 |
| datashader vs xrs-cupy | 0.1463 | 0.9166 | 34.9 - 44.4 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |

## Irregular (128 vertices)

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.74 ms | 1.72 ms | 0.51 ms | 4.08 ms | 1.64 ms |
| 50 | 500x250 | 1.19 ms | 2.19 ms | 3.13 ms | 4.48 ms | 2.03 ms |
| 50 | 1000x500 | 1.97 ms | 2.27 ms | 10.1 ms | 5.82 ms | 2.72 ms |
| 50 | 2000x1000 | 4.60 ms | 2.60 ms | 35.6 ms | 10.9 ms | 6.85 ms |
| 50 | 4000x2000 | 49.2 ms | 23.1 ms | 150 ms | 89.9 ms | 61.7 ms |
| 200 | 100x50 | 2.37 ms | 9.49 ms | 1.39 ms | 9.06 ms | 6.17 ms |
| 200 | 500x250 | 3.61 ms | 18.9 ms | 12.7 ms | 10.2 ms | 7.35 ms |
| 200 | 1000x500 | 5.83 ms | 18.5 ms | 43.8 ms | 13.8 ms | 9.82 ms |
| 200 | 2000x1000 | 13.4 ms | 20.8 ms | 158 ms | 27.7 ms | 18.0 ms |
| 200 | 4000x2000 | 68.5 ms | 47.1 ms | 640 ms | 111 ms | 77.6 ms |
| 1000 | 100x50 | 9.53 ms | 15.5 ms | 5.77 ms | 37.7 ms | 34.2 ms |
| 1000 | 500x250 | 18.2 ms | 51.1 ms | 65.3 ms | 42.5 ms | 38.0 ms |
| 1000 | 1000x500 | 30.4 ms | 54.3 ms | 223 ms | 52.0 ms | 47.4 ms |
| 1000 | 2000x1000 | 60.7 ms | 45.3 ms | 813 ms | 79.6 ms | 63.5 ms |
| 1000 | 4000x2000 | 141 ms | 55.6 ms | 3095 ms | 169 ms | 140 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 2.32x | 0.69x | 5.50x | 2.22x |
| 50 | 500x250 | 1.84x | 2.63x | 3.77x | 1.71x |
| 50 | 1000x500 | 1.15x | 5.12x | 2.95x | 1.38x |
| 50 | 2000x1000 | 0.57x | 7.73x | 2.36x | 1.49x |
| 50 | 4000x2000 | 0.47x | 3.05x | 1.83x | 1.25x |
| 200 | 100x50 | 4.01x | 0.59x | 3.83x | 2.61x |
| 200 | 500x250 | 5.23x | 3.52x | 2.83x | 2.03x |
| 200 | 1000x500 | 3.17x | 7.51x | 2.37x | 1.69x |
| 200 | 2000x1000 | 1.55x | 11.84x | 2.07x | 1.35x |
| 200 | 4000x2000 | 0.69x | 9.34x | 1.62x | 1.13x |
| 1000 | 100x50 | 1.63x | 0.61x | 3.96x | 3.59x |
| 1000 | 500x250 | 2.80x | 3.58x | 2.33x | 2.08x |
| 1000 | 1000x500 | 1.78x | 7.33x | 1.71x | 1.56x |
| 1000 | 2000x1000 | 0.75x | 13.40x | 1.31x | 1.05x |
| 1000 | 4000x2000 | 0.39x | 21.95x | 1.20x | 1.00x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.2268 | 0.9321 | 34.7 - 42.3 |
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.2268 | 0.9321 | 34.7 - 42.3 |
| datashader vs rasterio | 0.2268 | 0.9321 | 34.7 - 42.3 |
| datashader vs xrs-cupy | 0.2268 | 0.9321 | 34.7 - 42.3 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |

## Rectangles

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.41 ms | 1.34 ms | 0.27 ms | 3.22 ms | 0.73 ms |
| 50 | 500x250 | 0.57 ms | 1.56 ms | 0.39 ms | 3.45 ms | 0.88 ms |
| 50 | 1000x500 | 0.99 ms | 1.40 ms | 0.68 ms | 4.19 ms | 1.34 ms |
| 50 | 2000x1000 | 2.97 ms | 1.78 ms | 2.24 ms | 9.05 ms | 5.04 ms |
| 50 | 4000x2000 | 46.5 ms | 12.5 ms | 25.7 ms | 91.5 ms | 59.2 ms |
| 200 | 100x50 | 0.70 ms | 3.50 ms | 0.41 ms | 5.16 ms | 2.56 ms |
| 200 | 500x250 | 1.02 ms | 5.62 ms | 0.67 ms | 5.38 ms | 2.71 ms |
| 200 | 1000x500 | 1.76 ms | 8.05 ms | 1.66 ms | 6.38 ms | 3.33 ms |
| 200 | 2000x1000 | 6.02 ms | 21.7 ms | 6.15 ms | 17.5 ms | 8.44 ms |
| 200 | 4000x2000 | 54.0 ms | 47.0 ms | 39.4 ms | 98.0 ms | 64.3 ms |
| 1000 | 100x50 | 1.38 ms | 3.75 ms | 0.58 ms | 14.6 ms | 11.2 ms |
| 1000 | 500x250 | 3.34 ms | 7.17 ms | 2.19 ms | 15.4 ms | 18.6 ms |
| 1000 | 1000x500 | 12.8 ms | 8.03 ms | 6.47 ms | 17.3 ms | 13.4 ms |
| 1000 | 2000x1000 | 15.7 ms | 28.9 ms | 24.2 ms | 25.4 ms | 21.3 ms |
| 1000 | 4000x2000 | 79.5 ms | 36.1 ms | 107 ms | 116 ms | 90.2 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 3.24x | 0.66x | 7.78x | 1.77x |
| 50 | 500x250 | 2.76x | 0.69x | 6.08x | 1.56x |
| 50 | 1000x500 | 1.42x | 0.69x | 4.25x | 1.36x |
| 50 | 2000x1000 | 0.60x | 0.75x | 3.05x | 1.70x |
| 50 | 4000x2000 | 0.27x | 0.55x | 1.97x | 1.27x |
| 200 | 100x50 | 5.00x | 0.59x | 7.37x | 3.66x |
| 200 | 500x250 | 5.49x | 0.65x | 5.25x | 2.65x |
| 200 | 1000x500 | 4.58x | 0.94x | 3.62x | 1.89x |
| 200 | 2000x1000 | 3.60x | 1.02x | 2.91x | 1.40x |
| 200 | 4000x2000 | 0.87x | 0.73x | 1.82x | 1.19x |
| 1000 | 100x50 | 2.72x | 0.42x | 10.55x | 8.13x |
| 1000 | 500x250 | 2.14x | 0.65x | 4.62x | 5.57x |
| 1000 | 1000x500 | 0.63x | 0.51x | 1.35x | 1.05x |
| 1000 | 2000x1000 | 1.84x | 1.53x | 1.61x | 1.35x |
| 1000 | 4000x2000 | 0.45x | 1.34x | 1.46x | 1.13x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.1280 | 0.9433 | 32.5 - 42.9 |
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.1280 | 0.9433 | 32.5 - 42.9 |
| datashader vs rasterio | 0.1280 | 0.9433 | 32.5 - 42.9 |
| datashader vs xrs-cupy | 0.1280 | 0.9433 | 32.5 - 42.9 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |

## Stars (5-point, concave)

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.45 ms | 1.39 ms | 0.29 ms | 3.07 ms | 0.74 ms |
| 50 | 500x250 | 0.63 ms | 1.42 ms | 0.66 ms | 3.33 ms | 0.92 ms |
| 50 | 1000x500 | 1.18 ms | 1.59 ms | 1.52 ms | 4.52 ms | 1.43 ms |
| 50 | 2000x1000 | 3.01 ms | 1.71 ms | 4.95 ms | 11.9 ms | 6.22 ms |
| 50 | 4000x2000 | 50.7 ms | 9.68 ms | 34.9 ms | 94.3 ms | 59.8 ms |
| 200 | 100x50 | 0.78 ms | 6.69 ms | 0.47 ms | 5.33 ms | 2.58 ms |
| 200 | 500x250 | 1.49 ms | 7.82 ms | 1.99 ms | 5.60 ms | 3.15 ms |
| 200 | 1000x500 | 2.28 ms | 8.98 ms | 5.49 ms | 6.95 ms | 3.70 ms |
| 200 | 2000x1000 | 11.0 ms | 16.5 ms | 18.6 ms | 20.4 ms | 9.60 ms |
| 200 | 4000x2000 | 54.5 ms | 35.5 ms | 85.6 ms | 119 ms | 66.0 ms |
| 1000 | 100x50 | 1.89 ms | 13.3 ms | 1.16 ms | 15.9 ms | 24.4 ms |
| 1000 | 500x250 | 5.40 ms | 28.0 ms | 8.12 ms | 17.0 ms | 12.6 ms |
| 1000 | 1000x500 | 10.0 ms | 21.6 ms | 25.2 ms | 20.6 ms | 15.3 ms |
| 1000 | 2000x1000 | 23.2 ms | 20.7 ms | 84.7 ms | 31.1 ms | 24.1 ms |
| 1000 | 4000x2000 | 93.4 ms | 27.9 ms | 329 ms | 124 ms | 84.0 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 3.10x | 0.64x | 6.85x | 1.65x |
| 50 | 500x250 | 2.26x | 1.06x | 5.30x | 1.46x |
| 50 | 1000x500 | 1.35x | 1.29x | 3.84x | 1.22x |
| 50 | 2000x1000 | 0.57x | 1.65x | 3.95x | 2.07x |
| 50 | 4000x2000 | 0.19x | 0.69x | 1.86x | 1.18x |
| 200 | 100x50 | 8.52x | 0.60x | 6.79x | 3.29x |
| 200 | 500x250 | 5.26x | 1.34x | 3.77x | 2.12x |
| 200 | 1000x500 | 3.95x | 2.41x | 3.05x | 1.62x |
| 200 | 2000x1000 | 1.50x | 1.69x | 1.85x | 0.87x |
| 200 | 4000x2000 | 0.65x | 1.57x | 2.18x | 1.21x |
| 1000 | 100x50 | 7.04x | 0.61x | 8.38x | 12.88x |
| 1000 | 500x250 | 5.19x | 1.50x | 3.16x | 2.33x |
| 1000 | 1000x500 | 2.15x | 2.51x | 2.06x | 1.53x |
| 1000 | 2000x1000 | 0.89x | 3.64x | 1.34x | 1.03x |
| 1000 | 4000x2000 | 0.30x | 3.52x | 1.33x | 0.90x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.1084 | 0.9001 | 25.3 - 41.2 |
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.1084 | 0.9001 | 25.3 - 41.2 |
| datashader vs rasterio | 0.1084 | 0.9001 | 25.3 - 41.2 |
| datashader vs xrs-cupy | 0.1084 | 0.9001 | 25.3 - 41.2 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |

## Donuts (polygon + hole)

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.70 ms | 1.65 ms | 0.52 ms | 4.51 ms | 1.94 ms |
| 50 | 500x250 | 1.04 ms | 1.98 ms | 3.77 ms | 4.99 ms | 2.32 ms |
| 50 | 1000x500 | 1.74 ms | 2.10 ms | 13.7 ms | 6.27 ms | 3.08 ms |
| 50 | 2000x1000 | 3.97 ms | 2.72 ms | 51.3 ms | 11.7 ms | 8.71 ms |
| 50 | 4000x2000 | 50.4 ms | 36.3 ms | 209 ms | 91.9 ms | 62.8 ms |
| 200 | 100x50 | 1.62 ms | 2.96 ms | 1.31 ms | 9.92 ms | 7.79 ms |
| 200 | 500x250 | 2.64 ms | 3.96 ms | 14.2 ms | 11.7 ms | 8.29 ms |
| 200 | 1000x500 | 3.82 ms | 4.70 ms | 50.6 ms | 13.9 ms | 11.3 ms |
| 200 | 2000x1000 | 10.0 ms | 21.7 ms | 192 ms | 27.8 ms | 18.1 ms |
| 200 | 4000x2000 | 60.9 ms | 36.4 ms | 762 ms | 117 ms | 78.3 ms |
| 1000 | 100x50 | 7.91 ms | 11.7 ms | 5.38 ms | 44.9 ms | 40.6 ms |
| 1000 | 500x250 | 13.2 ms | 34.4 ms | 72.9 ms | 48.9 ms | 43.3 ms |
| 1000 | 1000x500 | 19.6 ms | 47.2 ms | 261 ms | 56.1 ms | 53.3 ms |
| 1000 | 2000x1000 | 39.7 ms | 39.1 ms | 991 ms | 78.9 ms | 68.6 ms |
| 1000 | 4000x2000 | 97.8 ms | 49.1 ms | 3861 ms | 178 ms | 149 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 2.34x | 0.74x | 6.40x | 2.76x |
| 50 | 500x250 | 1.91x | 3.64x | 4.82x | 2.24x |
| 50 | 1000x500 | 1.21x | 7.89x | 3.60x | 1.77x |
| 50 | 2000x1000 | 0.69x | 12.93x | 2.96x | 2.20x |
| 50 | 4000x2000 | 0.72x | 4.14x | 1.82x | 1.24x |
| 200 | 100x50 | 1.82x | 0.80x | 6.10x | 4.79x |
| 200 | 500x250 | 1.50x | 5.36x | 4.43x | 3.14x |
| 200 | 1000x500 | 1.23x | 13.24x | 3.63x | 2.96x |
| 200 | 2000x1000 | 2.17x | 19.18x | 2.78x | 1.81x |
| 200 | 4000x2000 | 0.60x | 12.52x | 1.92x | 1.29x |
| 1000 | 100x50 | 1.48x | 0.68x | 5.68x | 5.13x |
| 1000 | 500x250 | 2.61x | 5.54x | 3.71x | 3.29x |
| 1000 | 1000x500 | 2.41x | 13.33x | 2.86x | 2.72x |
| 1000 | 2000x1000 | 0.99x | 25.00x | 1.99x | 1.73x |
| 1000 | 4000x2000 | 0.50x | 39.47x | 1.82x | 1.52x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.2476 | 0.9283 | 39.4 - 43.9 |
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.2476 | 0.9283 | 39.4 - 43.9 |
| datashader vs rasterio | 0.2476 | 0.9283 | 39.4 - 43.9 |
| datashader vs xrs-cupy | 0.2476 | 0.9283 | 39.4 - 43.9 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |

## MultiPolygons (2-4 parts)

### Timings

| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.64 ms | 1.61 ms | 0.44 ms | 4.66 ms | 2.19 ms |
| 50 | 500x250 | 0.95 ms | 1.84 ms | 1.00 ms | 5.82 ms | 2.49 ms |
| 50 | 1000x500 | 1.44 ms | 1.87 ms | 2.36 ms | 6.22 ms | 3.07 ms |
| 50 | 2000x1000 | 4.10 ms | 4.39 ms | 7.39 ms | 12.2 ms | 7.27 ms |
| 50 | 4000x2000 | 46.9 ms | 11.3 ms | 43.1 ms | 95.3 ms | 61.7 ms |
| 200 | 100x50 | 1.38 ms | 5.43 ms | 0.69 ms | 11.1 ms | 8.37 ms |
| 200 | 500x250 | 2.14 ms | 9.03 ms | 2.50 ms | 12.6 ms | 17.8 ms |
| 200 | 1000x500 | 3.37 ms | 12.9 ms | 7.18 ms | 13.2 ms | 10.5 ms |
| 200 | 2000x1000 | 7.24 ms | 20.7 ms | 22.9 ms | 22.1 ms | 22.2 ms |
| 200 | 4000x2000 | 29.0 ms | 31.9 ms | 101 ms | 85.9 ms | 61.9 ms |
| 1000 | 100x50 | 6.15 ms | 16.6 ms | 2.29 ms | 49.7 ms | 44.7 ms |
| 1000 | 500x250 | 11.9 ms | 34.9 ms | 11.6 ms | 59.3 ms | 47.0 ms |
| 1000 | 1000x500 | 14.2 ms | 33.5 ms | 33.7 ms | 57.2 ms | 57.4 ms |
| 1000 | 2000x1000 | 25.5 ms | 40.2 ms | 112 ms | 64.7 ms | 60.3 ms |
| 1000 | 4000x2000 | 64.4 ms | 53.5 ms | 428 ms | 143 ms | 120 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | datashader | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 2.52x | 0.70x | 7.28x | 3.43x |
| 50 | 500x250 | 1.93x | 1.05x | 6.12x | 2.62x |
| 50 | 1000x500 | 1.30x | 1.63x | 4.31x | 2.13x |
| 50 | 2000x1000 | 1.07x | 1.80x | 2.99x | 1.77x |
| 50 | 4000x2000 | 0.24x | 0.92x | 2.03x | 1.31x |
| 200 | 100x50 | 3.94x | 0.50x | 8.07x | 6.08x |
| 200 | 500x250 | 4.22x | 1.17x | 5.88x | 8.31x |
| 200 | 1000x500 | 3.84x | 2.13x | 3.93x | 3.12x |
| 200 | 2000x1000 | 2.86x | 3.16x | 3.05x | 3.06x |
| 200 | 4000x2000 | 1.10x | 3.47x | 2.96x | 2.13x |
| 1000 | 100x50 | 2.69x | 0.37x | 8.08x | 7.26x |
| 1000 | 500x250 | 2.92x | 0.98x | 4.97x | 3.93x |
| 1000 | 1000x500 | 2.35x | 2.37x | 4.02x | 4.03x |
| 1000 | 2000x1000 | 1.57x | 4.38x | 2.54x | 2.36x |
| 1000 | 4000x2000 | 0.83x | 6.64x | 2.22x | 1.87x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs datashader | 0.1543 | 0.9513 | 34.9 - 40.9 |
| xrs-numpy vs geocube | 0.9961 | 1.0000 | 0.0 - 1.2 |
| xrs-numpy vs rasterio | 0.9961 | 1.0000 | 0.0 - 1.2 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 0.0 |
| datashader vs geocube | 0.1547 | 0.9518 | 35.0 - 40.9 |
| datashader vs rasterio | 0.1547 | 0.9518 | 35.0 - 40.9 |
| datashader vs xrs-cupy | 0.1543 | 0.9513 | 34.9 - 40.9 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 0.9961 | 1.0000 | 0.0 - 1.2 |
| rasterio vs xrs-cupy | 0.9961 | 1.0000 | 0.0 - 1.2 |

## LineStrings

### Timings

| n | size | xrs-numpy | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.53 ms | 1.51 ms | 2.94 ms | 0.33 ms |
| 50 | 500x250 | 0.56 ms | 1.43 ms | 3.05 ms | 0.45 ms |
| 50 | 1000x500 | 0.81 ms | 1.49 ms | 3.95 ms | 0.92 ms |
| 50 | 2000x1000 | 3.97 ms | 1.59 ms | 14.2 ms | 5.24 ms |
| 50 | 4000x2000 | 45.7 ms | 7.66 ms | 91.1 ms | 59.2 ms |
| 200 | 100x50 | 0.57 ms | 1.67 ms | 3.46 ms | 1.00 ms |
| 200 | 500x250 | 0.71 ms | 1.95 ms | 3.65 ms | 1.11 ms |
| 200 | 1000x500 | 1.02 ms | 2.62 ms | 4.60 ms | 1.67 ms |
| 200 | 2000x1000 | 2.65 ms | 3.68 ms | 14.9 ms | 6.45 ms |
| 200 | 4000x2000 | 46.8 ms | 12.2 ms | 93.8 ms | 61.1 ms |
| 1000 | 100x50 | 1.27 ms | 2.61 ms | 8.00 ms | 4.26 ms |
| 1000 | 500x250 | 1.46 ms | 2.67 ms | 17.4 ms | 4.42 ms |
| 1000 | 1000x500 | 1.90 ms | 3.67 ms | 9.71 ms | 12.9 ms |
| 1000 | 2000x1000 | 4.13 ms | 6.53 ms | 16.3 ms | 11.1 ms |
| 1000 | 4000x2000 | 48.5 ms | 12.8 ms | 89.5 ms | 64.5 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|
| 50 | 100x50 | 2.83x | 5.52x | 0.62x |
| 50 | 500x250 | 2.54x | 5.41x | 0.79x |
| 50 | 1000x500 | 1.84x | 4.88x | 1.13x |
| 50 | 2000x1000 | 0.40x | 3.58x | 1.32x |
| 50 | 4000x2000 | 0.17x | 1.99x | 1.29x |
| 200 | 100x50 | 2.93x | 6.08x | 1.75x |
| 200 | 500x250 | 2.73x | 5.10x | 1.55x |
| 200 | 1000x500 | 2.56x | 4.51x | 1.64x |
| 200 | 2000x1000 | 1.39x | 5.62x | 2.43x |
| 200 | 4000x2000 | 0.26x | 2.00x | 1.31x |
| 1000 | 100x50 | 2.06x | 6.32x | 3.37x |
| 1000 | 500x250 | 1.83x | 11.92x | 3.03x |
| 1000 | 1000x500 | 1.93x | 5.11x | 6.76x |
| 1000 | 2000x1000 | 1.58x | 3.96x | 2.69x |
| 1000 | 4000x2000 | 0.26x | 1.85x | 1.33x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs geocube | 0.3514 | 0.9255 | 2.6 - 23.7 |
| xrs-numpy vs rasterio | 0.3514 | 0.9255 | 2.6 - 23.7 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.4 - 32.4 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 0.3514 | 0.9255 | 2.7 - 34.7 |
| rasterio vs xrs-cupy | 0.3514 | 0.9255 | 2.7 - 34.7 |

## MultiLineStrings

### Timings

| n | size | xrs-numpy | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.49 ms | 2.75 ms | 3.34 ms | 0.90 ms |
| 50 | 500x250 | 0.64 ms | 1.80 ms | 3.50 ms | 1.06 ms |
| 50 | 1000x500 | 0.90 ms | 1.79 ms | 4.30 ms | 1.52 ms |
| 50 | 2000x1000 | 3.00 ms | 4.34 ms | 15.4 ms | 5.90 ms |
| 50 | 4000x2000 | 45.1 ms | 11.3 ms | 86.7 ms | 58.0 ms |
| 200 | 100x50 | 0.70 ms | 1.82 ms | 6.10 ms | 3.26 ms |
| 200 | 500x250 | 0.84 ms | 2.51 ms | 6.08 ms | 3.58 ms |
| 200 | 1000x500 | 1.20 ms | 2.14 ms | 7.59 ms | 4.21 ms |
| 200 | 2000x1000 | 3.04 ms | 4.89 ms | 18.1 ms | 8.61 ms |
| 200 | 4000x2000 | 45.8 ms | 12.9 ms | 97.8 ms | 80.2 ms |
| 1000 | 100x50 | 2.26 ms | 3.04 ms | 26.1 ms | 15.2 ms |
| 1000 | 500x250 | 2.09 ms | 3.27 ms | 19.9 ms | 15.6 ms |
| 1000 | 1000x500 | 10.1 ms | 4.33 ms | 32.0 ms | 17.5 ms |
| 1000 | 2000x1000 | 6.61 ms | 5.32 ms | 27.9 ms | 23.3 ms |
| 1000 | 4000x2000 | 61.9 ms | 13.9 ms | 142 ms | 82.3 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|
| 50 | 100x50 | 5.64x | 6.84x | 1.85x |
| 50 | 500x250 | 2.81x | 5.47x | 1.66x |
| 50 | 1000x500 | 1.99x | 4.78x | 1.69x |
| 50 | 2000x1000 | 1.45x | 5.14x | 1.97x |
| 50 | 4000x2000 | 0.25x | 1.92x | 1.29x |
| 200 | 100x50 | 2.60x | 8.68x | 4.64x |
| 200 | 500x250 | 2.98x | 7.23x | 4.25x |
| 200 | 1000x500 | 1.79x | 6.34x | 3.52x |
| 200 | 2000x1000 | 1.61x | 5.97x | 2.84x |
| 200 | 4000x2000 | 0.28x | 2.13x | 1.75x |
| 1000 | 100x50 | 1.34x | 11.53x | 6.70x |
| 1000 | 500x250 | 1.57x | 9.54x | 7.49x |
| 1000 | 1000x500 | 0.43x | 3.17x | 1.73x |
| 1000 | 2000x1000 | 0.81x | 4.22x | 3.53x |
| 1000 | 4000x2000 | 0.22x | 2.29x | 1.33x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs geocube | 0.3541 | 0.9586 | 3.4 - 23.9 |
| xrs-numpy vs rasterio | 0.3541 | 0.9586 | 3.4 - 23.9 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 2.0 - 33.5 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 0.3541 | 0.9586 | 3.8 - 35.1 |
| rasterio vs xrs-cupy | 0.3541 | 0.9586 | 3.8 - 35.1 |

## Points

### Timings

| n | size | xrs-numpy | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.51 ms | 1.62 ms | 2.69 ms | 0.28 ms |
| 50 | 500x250 | 0.43 ms | 1.70 ms | 2.90 ms | 0.40 ms |
| 50 | 1000x500 | 0.75 ms | 2.41 ms | 3.91 ms | 0.85 ms |
| 50 | 2000x1000 | 2.87 ms | 4.29 ms | 7.83 ms | 3.72 ms |
| 50 | 4000x2000 | 45.4 ms | 12.9 ms | 85.2 ms | 55.5 ms |
| 200 | 100x50 | 0.39 ms | 1.43 ms | 3.15 ms | 0.66 ms |
| 200 | 500x250 | 0.49 ms | 1.99 ms | 3.74 ms | 0.78 ms |
| 200 | 1000x500 | 0.84 ms | 1.63 ms | 4.47 ms | 1.34 ms |
| 200 | 2000x1000 | 3.03 ms | 3.40 ms | 9.55 ms | 4.47 ms |
| 200 | 4000x2000 | 45.2 ms | 11.3 ms | 91.2 ms | 61.6 ms |
| 1000 | 100x50 | 0.66 ms | 1.55 ms | 5.96 ms | 2.90 ms |
| 1000 | 500x250 | 0.75 ms | 1.58 ms | 5.95 ms | 3.25 ms |
| 1000 | 1000x500 | 1.06 ms | 2.00 ms | 6.95 ms | 3.76 ms |
| 1000 | 2000x1000 | 3.30 ms | 4.52 ms | 12.5 ms | 8.73 ms |
| 1000 | 4000x2000 | 45.3 ms | 12.9 ms | 93.8 ms | 63.3 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|
| 50 | 100x50 | 3.19x | 5.30x | 0.56x |
| 50 | 500x250 | 3.94x | 6.72x | 0.92x |
| 50 | 1000x500 | 3.20x | 5.18x | 1.12x |
| 50 | 2000x1000 | 1.49x | 2.73x | 1.29x |
| 50 | 4000x2000 | 0.28x | 1.87x | 1.22x |
| 200 | 100x50 | 3.70x | 8.17x | 1.71x |
| 200 | 500x250 | 4.09x | 7.70x | 1.62x |
| 200 | 1000x500 | 1.94x | 5.33x | 1.60x |
| 200 | 2000x1000 | 1.12x | 3.15x | 1.48x |
| 200 | 4000x2000 | 0.25x | 2.02x | 1.36x |
| 1000 | 100x50 | 2.34x | 8.98x | 4.38x |
| 1000 | 500x250 | 2.11x | 7.96x | 4.35x |
| 1000 | 1000x500 | 1.89x | 6.57x | 3.55x |
| 1000 | 2000x1000 | 1.37x | 3.78x | 2.64x |
| 1000 | 4000x2000 | 0.28x | 2.07x | 1.40x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 11.3 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 11.3 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 11.3 |

## MultiPoints (3-8 pts)

### Timings

| n | size | xrs-numpy | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|---:|
| 50 | 100x50 | 0.35 ms | 1.24 ms | 3.57 ms | 1.14 ms |
| 50 | 500x250 | 0.48 ms | 1.65 ms | 3.84 ms | 1.26 ms |
| 50 | 1000x500 | 0.74 ms | 1.67 ms | 4.60 ms | 1.69 ms |
| 50 | 2000x1000 | 2.52 ms | 4.16 ms | 13.5 ms | 5.92 ms |
| 50 | 4000x2000 | 44.6 ms | 9.95 ms | 93.4 ms | 60.2 ms |
| 200 | 100x50 | 0.45 ms | 3.51 ms | 7.00 ms | 4.67 ms |
| 200 | 500x250 | 0.59 ms | 1.37 ms | 7.36 ms | 4.74 ms |
| 200 | 1000x500 | 0.86 ms | 1.54 ms | 8.31 ms | 5.10 ms |
| 200 | 2000x1000 | 2.73 ms | 3.52 ms | 18.4 ms | 9.61 ms |
| 200 | 4000x2000 | 44.4 ms | 10.1 ms | 98.8 ms | 63.7 ms |
| 1000 | 100x50 | 1.00 ms | 2.50 ms | 24.7 ms | 21.0 ms |
| 1000 | 500x250 | 1.12 ms | 2.27 ms | 24.8 ms | 21.6 ms |
| 1000 | 1000x500 | 7.26 ms | 9.20 ms | 26.4 ms | 22.7 ms |
| 1000 | 2000x1000 | 3.44 ms | 6.28 ms | 36.1 ms | 28.6 ms |
| 1000 | 4000x2000 | 46.9 ms | 13.2 ms | 106 ms | 78.6 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster than xrs-numpy.

| n | size | xrs-cupy | geocube | rasterio |
|---:|---:|---:|---:|---:|
| 50 | 100x50 | 3.57x | 10.29x | 3.28x |
| 50 | 500x250 | 3.44x | 8.01x | 2.64x |
| 50 | 1000x500 | 2.25x | 6.21x | 2.28x |
| 50 | 2000x1000 | 1.65x | 5.34x | 2.34x |
| 50 | 4000x2000 | 0.22x | 2.09x | 1.35x |
| 200 | 100x50 | 7.80x | 15.53x | 10.36x |
| 200 | 500x250 | 2.31x | 12.39x | 7.98x |
| 200 | 1000x500 | 1.77x | 9.60x | 5.89x |
| 200 | 2000x1000 | 1.29x | 6.74x | 3.51x |
| 200 | 4000x2000 | 0.23x | 2.22x | 1.43x |
| 1000 | 100x50 | 2.50x | 24.63x | 20.99x |
| 1000 | 500x250 | 2.03x | 22.14x | 19.34x |
| 1000 | 1000x500 | 1.27x | 3.63x | 3.12x |
| 1000 | 2000x1000 | 1.83x | 10.49x | 8.33x |
| 1000 | 4000x2000 | 0.28x | 2.27x | 1.68x |

### Consistency

| pair | IoU min | IoU max | RMSE range |
|:-----|--------:|--------:|-----------:|
| xrs-numpy vs geocube | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 23.4 |
| geocube vs rasterio | 1.0000 | 1.0000 | 0.0 - 0.0 |
| geocube vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 23.4 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 0.0 - 23.4 |

## Where xrs-numpy is slower

Top 10 configurations where another rasterizer beats xrs-numpy.

| # | geometry | n | size | faster lib | xrs-numpy | other | xrs slower by |
|--:|:---------|--:|-----:|:-----------|----------:|------:|--------------:|
| 1 | multipolygons | 1000 | 100x50 | datashader | 6.2 ms | 2.3 ms | 2.7x |
| 2 | circles_64v | 1000 | 100x50 | datashader | 4.6 ms | 1.9 ms | 2.4x |
| 3 | rectangles | 1000 | 100x50 | datashader | 1.4 ms | 0.6 ms | 2.4x |
| 4 | multipolygons | 200 | 100x50 | datashader | 1.4 ms | 0.7 ms | 2.0x |
| 5 | rectangles | 1000 | 1000x500 | datashader | 12.8 ms | 6.5 ms | 2.0x |
| 6 | rectangles | 50 | 4000x2000 | datashader | 46.5 ms | 25.7 ms | 1.8x |
| 7 | points | 50 | 100x50 | rasterio | 0.5 ms | 0.3 ms | 1.8x |
| 8 | irregular_128v | 200 | 100x50 | datashader | 2.4 ms | 1.4 ms | 1.7x |
| 9 | rectangles | 200 | 100x50 | datashader | 0.7 ms | 0.4 ms | 1.7x |
| 10 | stars_5pt | 200 | 100x50 | datashader | 0.8 ms | 0.5 ms | 1.7x |
