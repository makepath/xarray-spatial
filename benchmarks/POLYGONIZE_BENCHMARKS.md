# Polygonize benchmarks

Generated: 2026-03-14 05:06 UTC

Compares xrspatial polygonize (numpy and cupy backends) against rasterio.features.shapes (GDAL) across 5 region patterns and 4 raster sizes. The **fastest** time in each row is bold.

## Few large regions

### Timings

| size | xrs-numpy | xrs-cupy | rasterio |
|---:|---:|---:|---:|
| 100x50 | **0.23 ms** | 7.10 ms | 0.63 ms |
| 500x250 | **8.03 ms** | 16.2 ms | 15.3 ms |
| 1000x500 | **36.4 ms** | 43.5 ms | 71.3 ms |
| 2000x1000 | **150 ms** | 231 ms | 255 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster.

| size | xrs-cupy | rasterio |
|---:|---:|---:|
| 100x50 | 30.75x | 2.71x |
| 500x250 | 2.01x | 1.91x |
| 1000x500 | 1.19x | 1.96x |
| 2000x1000 | 1.55x | 1.70x |

### Consistency

| pair | area agreement min | area agreement max | count ratio range |
|:-----|---:|---:|---:|
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 1.00 - 1.00 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |

## Many small regions

### Timings

| size | xrs-numpy | xrs-cupy | rasterio |
|---:|---:|---:|---:|
| 100x50 | **28.0 ms** | 45.5 ms | 43.8 ms |
| 500x250 | **756 ms** | 928 ms | 1186 ms |
| 1000x500 | **3149 ms** | 3801 ms | 5111 ms |
| 2000x1000 | **13122 ms** | 15581 ms | 22266 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster.

| size | xrs-cupy | rasterio |
|---:|---:|---:|
| 100x50 | 1.63x | 1.57x |
| 500x250 | 1.23x | 1.57x |
| 1000x500 | 1.21x | 1.62x |
| 2000x1000 | 1.19x | 1.70x |

### Consistency

| pair | area agreement min | area agreement max | count ratio range |
|:-----|---:|---:|---:|
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 1.00 - 1.00 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |

## Checkerboard

### Timings

| size | xrs-numpy | xrs-cupy | rasterio |
|---:|---:|---:|---:|
| 100x50 | **46.2 ms** | 50.1 ms | 82.2 ms |
| 500x250 | **797 ms** | 893 ms | 1278 ms |
| 1000x500 | **3350 ms** | 3629 ms | 5411 ms |
| 2000x1000 | **13644 ms** | 14809 ms | 22528 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster.

| size | xrs-cupy | rasterio |
|---:|---:|---:|
| 100x50 | 1.09x | 1.78x |
| 500x250 | 1.12x | 1.60x |
| 1000x500 | 1.08x | 1.62x |
| 2000x1000 | 1.09x | 1.65x |

### Consistency

| pair | area agreement min | area agreement max | count ratio range |
|:-----|---:|---:|---:|
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 1.00 - 1.00 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |

## Concentric rings (holes)

### Timings

| size | xrs-numpy | xrs-cupy | rasterio |
|---:|---:|---:|---:|
| 100x50 | **0.27 ms** | 18.2 ms | 1.33 ms |
| 500x250 | **2.12 ms** | 34.3 ms | 4.94 ms |
| 1000x500 | **7.12 ms** | 47.0 ms | 11.0 ms |
| 2000x1000 | **28.3 ms** | 47.9 ms | 29.9 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster.

| size | xrs-cupy | rasterio |
|---:|---:|---:|
| 100x50 | 68.18x | 4.98x |
| 500x250 | 16.18x | 2.33x |
| 1000x500 | 6.61x | 1.54x |
| 2000x1000 | 1.70x | 1.06x |

### Consistency

| pair | area agreement min | area agreement max | count ratio range |
|:-----|---:|---:|---:|
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 1.00 - 1.00 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |

## Large regions + 15% mask

### Timings

| size | xrs-numpy | xrs-cupy | rasterio |
|---:|---:|---:|---:|
| 100x50 | **1.21 ms** | 4.52 ms | 3.03 ms |
| 500x250 | **35.7 ms** | 45.5 ms | 82.1 ms |
| 1000x500 | **162 ms** | 216 ms | 369 ms |
| 2000x1000 | **744 ms** | 923 ms | 1692 ms |

### Relative to xrs-numpy

Values below 1.0 mean the competitor is faster.

| size | xrs-cupy | rasterio |
|---:|---:|---:|
| 100x50 | 3.74x | 2.51x |
| 500x250 | 1.27x | 2.30x |
| 1000x500 | 1.33x | 2.28x |
| 2000x1000 | 1.24x | 2.27x |

### Consistency

| pair | area agreement min | area agreement max | count ratio range |
|:-----|---:|---:|---:|
| xrs-numpy vs rasterio | 1.0000 | 1.0000 | 1.00 - 1.00 |
| xrs-numpy vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
| rasterio vs xrs-cupy | 1.0000 | 1.0000 | 1.00 - 1.00 |
