Benchmarking Results
============

Windows 10
--------------
CPU: AMD Ryzen 5 1600

Cores: 12

GPU: GeForce RTX 3060

RAM: 32GB

```
 hillshade.Hillshade.time_hillshade
```
nx | numpy | cupy | rtxpy 
--- | --- | --- | --- 
100 | 564±9μs | 1.33±0.07ms | 6.76±0.2ms 
300 | 2.70±0.1ms | 1.30±0.04ms | 9.36±0.8ms 
1000 | 38.0±2ms | 1.56±0.06ms | 26.6±2ms
3000 | 352±30ms | 2.13±1ms | 172±1ms

```
polygonize.Polygonize.time_polygonize
```
nx | numpy | geopandas | spatialpandas | rasterio-geopandas
--- | --- | --- | --- | ---
100 | 3.74±0.8ms | failed | failed  | failed
300 | 42.7±0.4ms | failed | failed  | failed
1000 | 492±4ms | failed | failed | failed

```
slope.Slope.time_slope
```
nx | numpy | cupy 
--- | --- | --- 
100 | 784±50μs | 2.70±0.06ms
300 | 1.83±0.1ms | 2.61±0.1ms
1000 | 17.9±0.2ms | 2.70±0.08ms
3000 | 171±1ms | 4.61±1ms
10000 | 1.62±0.02s | 105±100ms

```
viewshed.Viewshed.time_viewshed
```
nx | numpy | rtxpy 
--- | --- | --- 
100 | 7.24±0.01ms | 8.18±0.2ms
300 | 53.1±0.3ms | 10.4±0.2ms
1000 | 657±0.07ms | 27.1±0.3ms
3000 | 7.24±0.04s | 170±1ms
