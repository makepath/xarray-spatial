## Xarray-Spatial Changelog
-----------


### Version 0.10.12 - 2026-06-25

#### New features
- templates: add from_template() for empty study-area DataArrays over common regions (#3484) (#3487)
- accessor: add .xrs.validate() to check a raster against the xarray-spatial contract (#3486)
- accessor: add a categorized repr to the .xrs accessor (#3476) (#3477)
- accessor: forward per-tool repr to the accessed tool (#3478) (#3479)
- interpolation: accept GeoDataFrames and expose on the .xrs accessor with coregister (#3481)
- rasterize: support string/categorical columns with QGIS-visible labels (#3483)

#### Bug fixes and improvements
- proximity: fix EUCLIDEAN proximity/allocation/direction dropping targets at exact max_distance (#3443)
- perlin: preserve input x/y coordinates on the output (#3470)
- perlin: preserve input dtype on the dask backends (#3471)
- perlin: fix dask OOM by dropping a redundant dask.persist (#3469) (#3473)
- perlin: document the name parameter in the docstring (#3468)
- perlin: document name param, float-dtype requirement, and ValueError (#3467)
- perlin: drop unused not_implemented_func import, fix isort wrap (#3466)
- perlin: add test coverage for params, degenerate shapes, and metadata (#3472)
- perlin: fix stale test_perlin coords assertion after coords preservation (#3489)
- generate_terrain: preserve caller coords/res/crs (#3474) (#3475)
- least_cost_corridor: propagate friction geo-attrs (#3446) (#3449)
- corridor: validate matching shapes for precomputed surfaces (#3447)
- corridor: add strip, cross-backend, param, and metadata test coverage (#3448)
- fire: add test coverage for dask+cupy dispatch and metadata preservation (#3444)


### Version 0.10.11 - 2026-06-21

#### Bug fixes and improvements
- classify: align natural_breaks parameter order with sibling classifiers (#3418)
- multispectral: propagate input dims/coords through true_color() (#3429) (#3434)
- multispectral: fix nbr() docstring parameter name to match signature (swir2_agg) (#3435)
- multispectral: remove unused import and fix import ordering (#3428) (#3432)
- multispectral: add test coverage for true_color edges and evi/savi validation (#3436)
- diffusion: drop unused import, sort imports (#3421) (#3423)
- diffusion: add test coverage for cupy/dask+cupy, reflect boundary, and edge shapes (#3424)
- disaggregate: fix limiting_variable silently losing a zone's value (#3403) (#3414)
- disaggregate: vectorize the zone-membership loop (#3408) (#3420)
- dasymetric: add test coverage for metadata, 1x1, Inf weight, 3-class limiting_variable (#3419)
- dasymetric: clean up lint (F401, E128, isort) (#3411)
- classify: keep the large-array sampler O(num_sample) in host memory (#3416)
- classify: add edge-case and error-path test coverage (#3417)
- classify: remove unused import and fix isort ordering (#3405)
- morphology: remove unused imports and dead local, fix import order (#3413)
- morphology: skip the memory guard for lazy dask inputs (#3401) (#3410)
- morphology: add missing accessor methods (#3409)
- morphology: add test coverage for edge inputs (Inf, all-NaN, strip, integer) (#3415)
- viewshed: add tests for 1x1, all-NaN, and Inf-terrain edge cases (#3425) (#3426)
- proximity: match GPU max_distance precision to the CPU brute-force (#3389) (#3391)
- proximity: fix isort import-ordering drift (#3393)
- proximity: cover Dataset input for proximity/allocation/direction (#3390)
- fire: declare float32 dtype on the dask+numpy backend (#3394) (#3396)
- fire: drop dead code and fix import ordering (#3395) (#3397)
- aspect: sort imports per isort config (#3437) (#3438)
- aspect: test Inf and all-NaN elevation input across backends (#3439) (#3440)
- engine: carry the coregister/auto_reproject target on the parameter value (#3380)
- rasterize: match dask all_touched polygon boundary to eager on grid lines (#3384) (#3386)
- rasterize: add validation-guard test coverage (#3383) (#3385)
- zonal: vectorize _sort_and_stride 3D reindex, drop deepcopy (#3381) (#3382)


### Version 0.10.10 - 2026-06-17

#### New features
- geotiff: expose the GeoTIFF loader as an xarray backend engine (#3365) (#3375)
- geotiff: expose coregistered reads through the xarray engine via like= (#3376) (#3377)

#### Bug fixes and improvements
- geotiff: read and parse a local file once on the chunked GPU read path (#3373) (#3374)
- geotiff: reject predictor with jpeg2000/lerc/jpeg compression (#3371) (#3372)
- cost_distance: fix Dijkstra heap overflow on non-uniform friction (#3370)
- cost_distance: add test for the dask all-impassable early return (#3367) (#3368)
- ci: add free-threaded Python 3.14 (3.14t) to the test matrix (#3360)
- ci: make test_chunks_is_lazy counter thread-safe for free-threaded CI (#3363) (#3364)
- ci: scope the 'performance' labeler to benchmarked modules (#3357)
- remove AI-assistant tooling definitions (#3362)


### Version 0.10.9 - 2026-06-15

#### Bug fixes and improvements
- cost_distance: route bounded large-radius dask path to iterative Dijkstra (#3345)
- cost_distance: guard the numpy dask chunk path with _check_memory (#3346)
- cost_distance: fix leaking dask task name as output .name (#3349)
- cost_distance: replace mutable default target_values=[] with a None sentinel (#3348)
- cost_distance: add tests for 1x1 raster, Inf friction, and metadata (#3347)
- cost_distance: fix flake8/isort findings (#3350)
- contour: batch dask chunk compute to bound client memory (#3334)
- contour: remove redundant np.empty_like allocation in the coordinate transform (#3354)
- contour: add test coverage for all-equal auto-levels and empty explicit levels (#3353)
- geotiff: reject a zero or non-finite ModelPixelScale on read (#3332)
- geotiff: fix isort import ordering in _writers/eager.py (#3330)
- docs: fix file paths in the deep-sweep documentation (#3336)

#### Thanks
Thanks to @Melissari1997 for code and test contributions to this release.


### Version 0.10.8 - 2026-06-14

#### Bug fixes and improvements
- geotiff: gate dict gdal_metadata behind the experimental rich-tag opt-in (#3327)
- name dask tasks for remaining xrspatial tool modules (#3326)
- geotiff: don't flip masked_nodata True on a caller dtype= cast of an unmasked buffer (#3325)
- to_geotiff: type-check compression_level (#3324)
- rasterize: burn all_touched lines with the supercover walk (#3322)
- focal: handle empty rasters consistently across backends (#3319)
- reproject: promote large-output in-memory cupy inputs to dask (#3318)
- reproject: serialize all parallel=True kernel launches behind one lock (#3317)
- docs: add an examples-and-data getting-started page and landing index (#3316)


### Version 0.10.7 - 2026-06-13

#### New features
- to_geotiff: add pack=True, the inverse of mask_and_scale (#3064) (#3065)
- open_geotiff: add coregister=True (mask_and_scale + reproject + match grid) (#3069) (#3070) (#3072)
- open_geotiff: pick reproject resampling by raster type under auto_reproject=True (#3067) (#3068)
- open_geotiff: rename mask_and_scale to unpack and add GPU / dask+GPU support (#3075)

#### Bug fixes and improvements
- polygonize: reject invalid return_type before running the computation (#3312)
- polygonize: document that column_name is also used for geojson output (#3309)
- kriging: keep template backend and variance name in the singular-matrix fallback (#3294)
- rasterize: drop non-finite-coordinate geometries and clamp row casts (#3310)
- merge: add GPU tests for mixed-type first/last ordering (#3296) (#3302)
- interpolate: add edge-case and parameter test coverage (#3300)
- polygonize: honor _xrspatial_no_georef in CRS detection (#3293) (#3308)
- polygonize: jit the float-chunk region ranges scan (#3303) (#3314)
- rasterize: burn each geometry once per pixel under merge='sum'/'count' (#3313)
- interpolate: cut redundant work in the backends (#3311)
- rasterize: widen the dtype annotation to the accepted forms (#3291) (#3297)
- kriging: name the singular-fallback variance '{name}_variance' (#3289)
- polygonize: cover the dask mask-rechunk path (#3299) (#3305)
- polygonize: raise on uint32 region overflow in the cupy integer backend (#3292) (#3301)
- interpolate: drop unused math import and fix isort drift (#3286) (#3287)
- docs: add a why-xarray-spatial page, retire raster_huh (#3249) (#3284)
- geotiff: fill pack=True nodata holes at the integer dtype's native width (#3277)
- ci: add a docs-build job (#3252)
- projection: read ITRF Helmert scale as ppm to match PROJ's data files (#3283)
- projection: reject non-WGS84 ellipsoids in the fast-path dispatch (#3282)
- projection: fix LAEA inverse rq normalization and authalic inverse series (#3280)
- reproject: upload shared coordinate arrays once per chunk in multi-band cupy (#3279)
- reproject: promote to the dask path when the output is large (#3278)
- geotiff: export VRTUnsupportedError, CloudSizeLimitError, PixelSafetyLimitError publicly (#3265) (#3273)
- geotiff: refuse pack=True values the packed integer dtype cannot hold (#3272)
- merge: keep the shared integer input dtype instead of promoting to float64 (#3271)
- geotiff: add gpu and dask+gpu legs to the pack=True feature tests (#3270)
- geotiff: correct the unpack docstring (masking still runs without scale/offset) (#3263) (#3269)
- geotiff: rewrap _attrs import in _writers/gpu.py to isort line_length=100 (#3259) (#3261)
- docs: replace the usage page with a runnable quickstart (#3249) (#3258)
- docs: rewrite the installation page (environments, extras, GPU and cloud notes) (#3257)
- dask: name compute tasks xrspatial.<tool> for readable task graphs (#3255)
- docs: fix stale Datashader claim and hardcoded links in getting-started (#3249) (#3251)
- geotiff: coregister=True reads COG overviews and fails fast on bad templates (#3254)
- geotiff: document coregister=True and register it as an experimental feature (#3248)
- geotiff: warn when coregister=True covers only a small fraction of the caller grid (#3247)
- proximity: replace the numpy line-sweep with an exact cKDTree query (#3245)
- projection: add dask end-to-end warning test, lock note on _crs_to_dict (#3242) (#3244)
- projection: silence the pyproj lossy-PROJ-string warning in param extractors (#3242) (#3243)
- geotiff: stream dask input through the GPU writer one tile-row band at a time (#3241)
- geotiff: fix to_geotiff(pack=True) crash on cupy and dask+cupy input (#3240)
- geotiff: keep float32 sources float32 through to_geotiff(pack=True) (#3080) (#3239)
- hotspots: raise instead of returning all zeros when the raster contains Inf (#3238)
- geotiff: defer the pack=True NaN guard into the dask graph so chunks compute once (#3237)
- geotiff: forward caller chunking to the coregister/auto_reproject output (#3236)
- visibility: add cupy backend, NaN, and Fresnel-blocked coverage (#3202)
- visibility: fix the cupy backend crash in cumulative_viewshed/visibility_frequency (#3205)
- zonal: add GPU backend coverage for regions and crosstab (#3203)
- focal: make the kernel memory guard dtype-aware (#3223) (#3232)
- focal: resolve apply() default func per backend (#3215) (#3224)
- focal: add edge-case and parameter tests from the test-coverage sweep (#3230)
- hotspots: fix the docstring backend list and kernel weight semantics (#3216) (#3227)
- clip_polygon: align dask+cupy mask chunks before map_blocks (#3186) (#3204)
- focal: make output dtype consistent across backends (#3226)
- clip_polygon: use the largest chunk for the rasterize mask on dask (#3191) (#3201)
- visibility: compute the dask source once in cumulative_viewshed (#3185) (#3199)
- polygon_clip: remove unused imports and fix import order (#3198)
- visibility: align output name and frequency_mhz hint with viewshed (#3183) (#3195)
- geotiff: fix to_geotiff dask+cupy crash in the CPU streaming writer (gpu=False) (#3171)
- clip_polygon: make nodata dtype consistent across backends (#3208)
- mcda: fix the owa() dask path and batch wpm() validation into one scheduler pass (#3158)
- hypsometric_integral: validate raster inputs (#3211)
- regions: vectorize the relabel loop (#3207) (#3212)
- mcda: fix cupy and dask backend failures in owa, standardize, and sensitivity (#3146) (#3157)
- mcda: fix API consistency (owa weights param, boolean_overlay Dataset input, ahp_weights docs, ConsistencyResult export) (#3148) (#3155)
- mcda: add cupy, dask, metadata, and edge-case test coverage (#3149) (#3156)
- mcda: guard monte_carlo dask materialization against OOM (#3145) (#3153)
- proximity: add coverage tests (int rasters, bounded GPU metrics, empty input) (#3139) (#3140)
- geotiff: add reader.unpack / writer.pack to SUPPORTED_FEATURES at experimental tier (#3172)
- focal: preserve input float dtype in mean() across all backends (#3214) (#3221)
- proximity: fix stale proximity/allocation/direction docstrings (#3091) (#3138)
- visibility: fix isort import-ordering (#3182) (#3189)
- geotiff: thread the nodata kwarg into _pack's NaN fill so pack=True holes match GDAL_NODATA (#3168) (#3174)
- mcda: remove unused warnings import, fix isort drift (#3143) (#3144)
- geotiff: rewrite stale per-band SCALE/OFFSET in to_geotiff(pack=True) after band-subset reads (#3175)
- geotiff: validate compression_level before GPU dispatch and pass it to fallback codecs (#3167) (#3176)
- vrt: place non-georeferenced tiles by explicit pixel offsets (#3116) (#3135)
- projection: document actual return types of vertical datum functions (#3097) (#3134)
- mcda: keep standardize piecewise/categorical on the device for cupy inputs (#3159)
- geotiff: fix the compression corpus fixture path so oracle tests actually run (#3164) (#3169)
- geotiff: warn when the GPU writer materializes dask input; scope streaming docs to the CPU path (#3173)
- focal: hoist the input cast out of the cupy focal_stats loop (#3231) (#3233)
- focal: make the memory guard backend-aware for dask input (#3218) (#3228)
- clip_polygon: add backend/edge-case/metadata test coverage (#3197) (#3206)
- docs: fix the README to_geotiff example that used the gated JPEG codec (#3162) (#3170)
- zonal: tighten crosstab zone_ids/cat_ids type hints and fix stale nodata docstring (#3196)
- mcda: preserve input attrs in constrain() (#3147) (#3154)
- reproject: keep integer dtype for empty chunks (#3096) (#3133)
- mcda: make monte_carlo sensitivity chunk-bounded on dask and drop constrain() deep copy (#3160)
- rasterize: drop two avoidable full-raster allocations in the backends (#3107) (#3131)
- proximity: fix bounded dask GREAT_CIRCLE missing antimeridian targets (#3108) (#3130)
- geotiff: mask 64-bit integer sentinels at native width in eager reads (#3098) (#3128)
- reproject: accept cupy and dask+cupy inputs in merge() (#3095) (#3125)
- geotiff: add xfail'd GPU and dask+GPU pack=True round-trip tests (#3114) (#3127)
- proximity: sort dask KDTree targets row-major so the tie-break matches numpy (#3090) (#3124)
- proximity: hoist the line-sweep kernel to module level to stop per-call recompiles (#3103) (#3126)
- rasterize: add all_touched-line and non-iterable-input coverage tests (#3120)
- rasterize: rename use_cuda to gpu with a deprecation shim (#3089) (#3122)
- geotiff: reject zero and non-finite SCALE/OFFSET under unpack=True; guard pack=True against divide-by-zero (#3104) (#3129)
- reproject: gate the CUDA fast path on WGS84-compatible datums (#3094) (#3119)
- geotiff: group streaming-write rows into bands so source chunks compute once (#3117) (#3136)
- reproject: skip the duplicate try_numba_transform probe in the numpy chunk worker (#3123)
- reproject: remove the obsolete strict xfail on the streaming 3-D test (#3100) (#3181)
- reproject: add tests for the streaming fallback path (#3101) (#3118)
- reproject: match streaming output dtype to the other backends (#3093) (#3111)
- geotiff: name unpack instead of the deprecated mask_and_scale alias in docs and errors (#3115)
- geotiff: keep regression tests and sweep state after #3109 landed the guard (#3088) (#3110)
- proximity: build dask coordinate grids with chunk-aligned broadcast (#3132) (#3137)
- rasterize: propagate GeoDataFrame CRS to output when like has none (#3087) (#3113)
- rasterize: reject non-finite burn values against integer and bool output dtypes (#3085) (#3109)
- reproject: remove unused import and dead local, fix import order in __init__.py (#3083) (#3092)
- geotiff: fix isort import-ordering drift (#3082) (#3084)
- merge: make the output-size guard backend-aware (#3048) (#3081)
- geotiff: restore the nodata sentinel for float rasters in to_geotiff(pack=True) (#3078) (#3079)
- projection: silence the pyproj PROJ-string warning in _get_datum_params (#3076) (#3077)
- resample: add dask+numpy nodata-masking value-parity tests (#3073) (#3074)
- rasterize: treat a LinearRing as a line; warn on dropped geometries (#3055) (#3063)
- rasterize: guard against geometry/like CRS mismatch (#3058) (#3062)
- rasterize: reject large integer burn values that float64 cannot hold exactly (#3056) (#3060)
- geotiff: warn on non-atomic custom GPU merge over overlap (#3061)
- rasterize: reject fills the output dtype cannot represent exactly (#3059)

#### Contributors
- Thanks to @brendancol for contributions to this release.


### Version 0.10.6 - 2026-06-08

#### New features
- emerging_hotspots: compute the Getis-Ord Gi* statistic (#3040)
- re-export open_geotiff and to_geotiff from top-level xrspatial (#3005) (#3006)
- add Cursor IDE support (.cursorrules and .cursor/rules/) (#3019)

#### Bug fixes and improvements
- reproject: fix flake8/isort style issues (#3052)
- reproject: parametrize dask+cupy parity over resampling modes (#3050) (#3051)
- reproject: make the output-size guard backend-aware (#3046) (#3047)
- contours: add a cross-backend NaN-input parity test (#3045)
- sweep: drop merge=union for state CSVs (#2754) (#3043)
- flow_path_dinf: fix a hang on cyclic D-inf flow directions (#2796) (#3042)
- viewshed: fix the broken bounds-check guard in _calculate_event_row_col() (#2793) (#3035)
- viewshed: validate observer_elev and target_elev before dispatch (#2794) (#3037)
- warnings: name the calling function in the unit-mismatch warning (#2782) (#3036)
- stream_link_mfd / stream_order_mfd: keep the dask assembly lazy (#2885) (#3039)
- remove dead datum-shift machinery (#2659) (#3038)
- polygonize: fix dask 8-connectivity float divergence at large rtol (#2677) (#3041)
- viewshed: validate regular grid spacing (#2789) (#3034)
- geotiff: reject to_geotiff(compression=None) with a clear TypeError (#2978) (#3029)
- geotiff: fix UnboundLocalError on restore_sentinel in plain-ndarray VRT tiled write (#2969) (#3028)
- contours: add tests for the all-NaN auto-levels RuntimeWarning regression (#2795) (#3009)
- geotiff: thread sidecar_origin through the HTTP/fsspec sidecar georef path (#3027)
- geotiff: remove stale external-overview skips on dask/fsspec/GPU golden-corpus backends (#3026)
- geotiff: route remote sidecar parse failures through the shared policy (#3022) (#3025)
- geotiff: size the streaming write budget from source chunks, not output tiles (#3008)
- contours: fix the border collar on integer dask input (#3020) (#3021)
- contours: fix return_type validation to fail fast on invalid values (#3010)

#### Contributors
- Thanks to @brendancol and @Melissari1997 for their contributions to this release.


### Version 0.10.5 - 2026-06-06

#### New features
- add Kilo CLI command definitions (#3003)

#### Bug fixes and improvements
- contours: add geometry deduplication (#2790) (#3001)
- geotiff: cover compress()/decompress() dispatcher error branches (#2995) (#2996)
- geotiff: fail closed on malformed GDAL_METADATA XML under mask_and_scale (#2999)
- geotiff: enforce tile_size positivity in array-level writers (#2997) (#3000)
- geotiff: reject mixed per-band SCALE/OFFSET under mask_and_scale (#2988) (#2993)
- geotiff: promote int to float on masked read regardless of sentinel hit (#2990) (#2994)
- geotiff: reject malformed SCALE/OFFSET under mask_and_scale (#2992)
- vrt: surface chunked decode-time hole gap under missing_sources='warn' (#2989) (#2991)
- geotiff: cover GPU writer degenerate shapes (1x1/1xN/Nx1) (#2986)
- geotiff: accumulate GPU overview mean in float64 for CPU byte parity (#2983) (#2985)
- accessor: surface standalone docstrings on .xrs methods; fix broken hydro delegations (#2982)
- geotiff: align direct backend mask_nodata default with open_geotiff (#2976) (#2980)
- vrt: privatize build_vrt; route to_geotiff's VRT path through it (#2979)
- geotiff: reject non-string compression at the to_geotiff boundary (#2975) (#2977)
- geotiff: cover degenerate-shape reads on dask and GPU backends (#2973)
- geotiff: fix flake8/isort style drift in tests and two source imports (#2970) (#2972)
- vrt: stop build_vrt fabricating a GeoTransform for non-georeferenced sources (#2966) (#2971)
- geotiff: clean up VRT staging dir on early validation failures (#2964) (#2968)
- geotiff: write VRT index atomically via temp-then-rename (#2965) (#2967)
- geotiff: align open_geotiff parameters with rioxarray's open_rasterio (#2961) (#2963)
- geotiff: consolidate public API on open_geotiff / to_geotiff (#2962)

#### Contributors
- Thanks to @brendancol and @Melissari1997 for their contributions to this release.


### Version 0.10.4 - 2026-06-04

#### Bug fixes and improvements
- kriging: guard single-point input in the variogram (#2924)
- kriging: zero the matrix diagonal so the nugget is not placed on it (#2922)
- polygonize: cast the float mask to bool before combining it with nan_mask (#2623) (#2913)
- spline: upload invariant point/weight data to the GPU once on the dask+cupy path (#2929)
- kriging: skip the full-grid k0 memory term for dask templates (#2923) (#2926)
- kriging: fix flake8 E128 and isort drift in interpolate/_kriging.py (#2916) (#2918)
- geotiff: extract a shared GeoTIFF fixture builder (#2912)
- Add GPU and edge-case test coverage for spline() (#2927)
- Add kriging test coverage for dask+numpy variance, nlags, and edge cases (#2921) (#2928)
- Add GPU backend test coverage for idw() (#2925)
- Remove a flaky wall-clock visvalingam-whyatt regression test (#2914)
- docs: add a 1.0.0 stability policy and LTS commitment page (#2958)

#### New contributors
- Thanks to @SAY-5 (Sai Asish Y) for their first contribution (#2913)


### Version 0.10.3 - 2026-06-03

#### New features
- focal: implement the Getis-Ord Gi* statistic in hotspots() (#2803)

#### Bug fixes and improvements
- slope: validate planar cell size (#2897) (#2901)
- contours: validate return_type before doing data work (#2891) (#2898)
- viewshed: build the GPU rtx mesh with real x/y resolution (#2861) (#2887)
- hydro: validate MFD fraction values in public consumers (#2873) (#2883)
- viewshed: warn and document dask Tier C divergence from the exact sweep (#2880)
- viewshed: validate max_distance, rejecting negative and non-finite values (#2874)
- contours: validate the n_levels contract (#2895) (#2899)
- proximity: fix bounded dask dropping in-range targets on irregular coords (#2908) (#2909)
- contours: drop zero-length geometry at exact-level corners (#2902)
- viewshed: stop mutating the caller's input raster (#2852) (#2871)
- proximity: fix bounded-dask crash on skinny rasters (#2877)
- contours: resolve output CRS the same way as polygonize (#2893) (#2900)
- geotiff: gate remote sources in read_geotiff_gpu under stable_only (#2867) (#2878)
- proximity: reject non-finite target_values (#2868)
- focal: reject non-binary kernels in apply() and focal_stats() (#2862)
- terrain: resolve geodesic coords from real lat/lon over numeric y/x (#2903)
- slope: set output units to degrees instead of leaking input units (#2904)
- proximity: reject non-monotonic 1D coords in proximity/allocation/direction (#2875)
- focal: fix hotspots() silent all-zeros on degenerate dask inputs (#2860)
- focal: raise a clear error for non-2D custom kernels (#2842) (#2853)
- aspect: warn on degree/meter unit mismatch in the planar path (#2839) (#2844)
- contours: fix docstring overstating the Dask peak-memory guarantee (#2906)
- viewshed: fix crash on NaN cells, marking them INVISIBLE (#2857) (#2876)
- proximity: stop the dask fallback mutating the caller's input chunks (#2847) (#2864)
- terrain: list accepted unit names in the invalid z_unit error (#2870)
- hydro: reject cyclic MFD fraction grids instead of returning wrong results or hanging (#2884)
- polygonize: replace the flaky visvalingam timing test with deterministic checks (#2889)
- proximity: make allocation/direction tie-breaking deterministic across backends (#2881)
- hydro: validate the MFD companion raster shape against the grid (#2863) (#2879)
- geotiff: gate stable_only before the bbox metadata read on remote/VRT sources (#2882)
- hydro: keep MFD dask output assembly lazy (#2865) (#2886)
- viewshed: call has_cuda_and_cupy() in has_rtx() so GPU dispatch gates correctly (#2849) (#2859)
- aspect: guard dask geodesic against single-chunk OOM (#2840) (#2846)
- aspect: fix .name leak on dask backends when name=None (#2845)
- slope: cover planar boundary modes on cupy and dask+cupy backends (#2832)
- slope: fix .name leak on dask backends when name=None (#2838)
- proximity: pin the dim-order error path and all-NaN raster across backends (#2836)
- aspect: cover the northness/eastness method='geodesic' path (#2829) (#2835)
- focal: fix GPU std/var precision on large-offset rasters (#2834)
- aspect: make the planar cupy backend match numpy in [0, 360) (#2827) (#2833)
- geotiff: enforce stable_only for HTTP/fsspec sources on non-VRT read paths (#2826)
- geotiff: synthesize x/y coords for no-georef VRT reads (#2824)
- geotiff: validate the gpu dispatch flag as bool and fail closed (#2819) (#2823)
- geotiff: correct the stable_only docstring to state HTTP/fsspec rejection (#2822)
- polygonize: optimize visvalingam_whyatt with a min-heap (#2817)
- aspect: fix the compass-direction table in the docstring (#2825) (#2828)
- proximity: raise on invalid distance_metric (#2814)
- focal: make hotspots() lazy for Dask input (#2802)
- aspect: apply cell-size correction to the planar path (#2760) (#2780)
- slope: warn on degree coords with meter elevation in the planar path (#2764) (#2777)
- focal: fix GPU variety undercount for kernels larger than 5x5 (#2800)
- aspect: add an independent rectangular-cell oracle for the planar path (#2781)
- proximity: validate negative max_distance (#2813)
- focal: preserve input float dtype in apply() and focal_stats() (#2769) (#2805)
- focal: validate the hotspots() kernel like apply() and focal_stats() (#2771) (#2799)
- terrain: warn when planar resolution averages irregular coordinates (#2766) (#2784)
- slope: propagate the NaN center cell in the planar path (#2761) (#2773)
- proximity: fix GREAT_CIRCLE on the CPU line-sweep backend (#2816)
- proximity: fix dask halo depth on irregular coords and 1xN/Nx1 rasters (#2815)
- contours: fix auto-levels silently dropping all contours with +/-inf (#2801)
- proximity: fix Great Circle docstrings to say meters, not kilometers (#2811)
- focal: validate focal_stats() stats_funcs before dispatch (#2798)
- slope: keep dask+cupy geodesic lat/lon lazy to avoid GPU OOM (#2776)
- aspect: skip the full-raster geodesic memory guard for dask (#2774)
- repo: add Codex command mirrors (#2806)
- terrain: make the geodesic memory guard backend-aware for dask (#2765) (#2779)


### Version 0.10.2 - 2026-06-01

#### New contributors
- @Melissari1997 made their first contribution in #2749

#### Bug fixes and improvements
- contours: handle infinite NaN coords (#2749)
- contours: validate input raster, reject complex dtype (#2711)
- contours: keep CRS on empty geopandas result (#2708)
- focal: pin result .name across backends in focal_stats and hotspots (#2735)
- focal: honour boundary on the cupy backend (#2736)
- focal: keep hotspots dask+cupy classification on the GPU (#2739)
- focal: align signatures across mean/apply/focal_stats/hotspots (#2699)
- aspect: keep lat/lon chunked in dask+cupy geodesic path (#2707)
- aspect: planar dask backends report float32 dtype matching computed data (#2741)
- slope: planar dask backends declare float32 meta dtype (#2694)
- slope: widen name type hint to Optional[str] for terrain-family parity (#2687)
- viewshed: consistent output name and dtype across backends (#2747)
- viewshed: guard degenerate-axis resolution in the CPU sweep (#2745)
- viewshed: size max_distance window per axis on anisotropic rasters (#2702)
- proximity: consistent output dtype and .name across allocation and direction backends (#2728)
- proximity: fix bounded GREAT_CIRCLE crash on dask backends (#2722)
- proximity: batch per-chunk-row compute in KDTree count pass (#2726)
- hydro: cap flow_path downstream walk to break cycles (#2718)
- hydro: stream_order_d8 accepts method alias; basins_d8 emits DeprecationWarning (#2716)
- resample: reject out-of-range integer nodata sentinels (#2671)
- resample: reject empty rasters with a clear ValueError (#2665)
- resample: reject irregular/non-monotonic spatial coordinates (#2667)
- resample: refresh nodata/nodatavals on identity fast path (#2670)
- resample: fix dask nearest/bilinear chunk-seam values on downsampling (#2627)
- reproject: make transform_precision=0 force exact pyproj transforms (#2654)
- reproject: account for datum shifts in output bounds estimation (#2655)
- reproject: disable numba fast path for non-WGS84 datums (#2657)
- reproject: filter footprint coordinate pairs by joint finite mask (#2652)
- reproject: fix per-band nodata masking for multi-band inputs (#2658)
- reproject: fix cupy backend divergence from numpy on pyproj-fallback CRS pairs (#2629)
- reproject: rename vertical CRS kwargs to source_/target_vertical_crs (#2626)
- zonal: raise clear ValueError for empty Dataset in stats() (#2642)
- zonal: count returns 0 for empty zones, other stats stay NaN (#2656)
- zonal: validate backend compatibility on the crosstab 3D path (#2653)
- zonal: validate backend compatibility for 3D apply() inputs (#2648)
- zonal: make apply() output .name deterministic across backends (#2622)
- zonal: fix int32 overflow in _strides() that corrupts stats/crosstab on huge rasters (#2617)
- crop: validate zones/values shapes match instead of returning nonsense (#2645)
- geotiff: forward allow_rotated/allow_invalid_nodata to VRT per-source reads (#2676)
- geotiff: make open_geotiff bbox= work for .vrt sources (#2674)
- geotiff: make VRT tiled writes atomic to avoid poisoned partial output (#2678)
- polygonize: batch dask per-chunk compute instead of serial loop (#2673)
- polygonize: batch dask compute per chunk row to recover parallelism (#2632)
- polygonize: make dask float output chunk-invariant for rtol (#2675)
- polygonize: keep diagonal-notch hole in dask 8-conn merge (#2633)
- polygonize: derive transform from x/y coords so geopandas CRS matches geometry (#2630)


### Version 0.10.1 - 2026-05-29

#### New features
- accessor: backend-aware .xrs.open_geotiff on DataArray and Dataset (#2598)
- geotiff: add bbox= to open_geotiff for geographic-space windowed reads (#2556)

#### Bug fixes and improvements
- rasterize: honor resolution= exactly when bounds don't divide evenly (#2597)
- reproject: fix vertical-datum shift crash on integer DEMs (#2596)
- resample: clamp dask interp overlap depth per axis (#2547) (#2599)
- zonal: fix trim() with NaN sentinel on numpy and cupy backends (#2559) (#2594)
- reproject: reject explicit out-of-range nodata for integer dtypes (#2591)
- zonal: fix cupy backend dropping all-NaN zones and desyncing zone_ids (#2562) (#2589)
- resample: refresh transform to match actual coord orientation (#2571) (#2587)
- rasterize: validate resolution= shape and element type (#2576) (#2586)
- resample: preserve integer precision in nodata mask comparison (#2570) (#2592)
- zonal: keep hypsometric_integral dask path fully lazy (#2563) (#2593)
- geotiff: accept projected GeoTIFFs that carry a base geographic CRS (#2603)
- polygonize: reject non-finite simplify_tolerance (#2575) (#2584)
- reproject: unit-aware bounds_policy="auto" blow-up detection (#2600)
- zonal: validate return_type at entry to stats() (#2558) (#2578)
- resample: validate target_resolution and scale_factor up front (#2574) (#2590)
- rasterize: support descending-x in like= templates (#2568) (#2595)
- rasterize: reject zig-zag duplicate-coord patterns in _check_uniform_axis (#2585)
- polygonize: fix dask chunk-dependent merging and DN corruption (#2601)
- zonal: make crop() return (0, 0) on all backends when no zone matches (#2580)
- rasterize: reject partial width/height overrides instead of silent ignore (#2579)
- zonal: fix crosstab cat_ids overcount when earlier categories are filtered out (#2560) (#2577)
- resample: preserve scalar non-dim coords and refresh stale nodata attr (#2542) (#2549)
- zonal: fix hypsometric_integral dask+cupy backend (#2525) (#2529)
- geotiff: include GB allocation estimate in max_pixels error message (#2554)
- zonal: raise on 'majority' in zonal_stats dask path instead of silent drop (#2528) (#2531)
- reproject: fix half-pixel offset in geoid and datum-shift grid lookups (#2508) (#2516)
- rasterize: reject NaN fill against integer dtype (#2504) (#2512)
- geotiff: reject distinct per-band nodatavals on write (#2514) (#2519)
- zonal: guard unbounded allocation in stats(return_type='xarray.DataArray') (#2523) (#2533)
- zonal: rechunk apply 3D dask output along stacked axis (#2526) (#2532)
- rasterize: hoist poly_props/poly_global cupy.asarray above all_touched (#2506) (#2510)
- polygonize: auto-detect attrs['transform'] for output georeferencing (#2536) (#2541)
- geotiff: reconcile COG write stability across registry, contract, and docstring (#2520)
- zonal: rename crop(zones_ids=) to crop(zone_ids=) with deprecation shim (#2521) (#2527)
- reproject: preserve integer dtype in dask+cupy fast path (#2505) (#2509)
- geotiff: scope max_pixels to the chunk when chunks= is supplied (#2501) (#2502)


### Version 0.10.0 - 2026-05-27

#### GeoTIFF release contract

Tiers what the public GeoTIFF and COG surface promises for this release.
`xrspatial.geotiff.SUPPORTED_FEATURES` is the source of truth; the bullets
below mirror it. See the reference page at `docs/source/reference/geotiff.rst`
for the per-tier semantics and the audit trail.

**Stable**
- Local-file GeoTIFF read (`open_geotiff`) on the eager numpy backend, including windowed reads via `window=`. Covered by the window-read suite in `xrspatial/geotiff/tests/`. (epic #2340)
- Dask reads (`read_geotiff_dask`). Cross-backend parity against the eager numpy reader is gated by `test_backend_parity_matrix.py` and `test_backend_full_parity_2211.py`. (epic #2341)
- Local-file GeoTIFF write (`to_geotiff`) on the CPU writer. (epic #2340)
- Local COG read and write (`reader.local_cog`, `writer.cog`) for axis-aligned 2D / 3D rasters with the lossless codecs `none`, `deflate`, `lzw`, `zstd`, `packbits`, internal overviews only, and normal CRS / transform / dtype / nodata / band / pixel-is-area / pixel-is-point round-trip. Backed by the writer compliance suite (#2292), the cross-backend parity gate (#2293), and the per-tile byte-budget contract (#2294 / #2298). (epic #2286)
- Lossless stable codecs: `none`, `deflate`, `lzw`, `zstd`, `packbits`. Integer and float byte-for-byte round-trip. (epic #2340)

**Advanced**
- fsspec-routed reads (`reader.fsspec`) and HTTPS reads (`reader.http`). The transport layer works but the redirect / retry / cache surface is not contracted at the stable bar. (epic #2344)
- HTTP COG reads (`reader.http_cog`). Range fetching, range coalescing, the SSRF / private-host filter, and the per-tile byte cap are pinned by the byte-budget contract (#2294 / #2298), but the broader transport surface is tracked separately for stable promotion. (epic #2344)
- VRT reads (`reader.vrt`). Limited to simple GDAL VRT mosaics over GeoTIFF sources that agree on CRS, transform orientation, pixel size, dtype, and band count. See the VRT support matrix in `docs/source/reference/geotiff.rst`. (epic #2342, PR #2321)
- External `.tif.ovr` sidecar reads (`reader.sidecar_ovr`). (epic #2340)
- Writer overviews and BigTIFF (`writer.overviews`, `writer.bigtiff`). (epic #2340)
- BigTIFF COG writes (`writer.bigtiff_cog`). The external-interop gate lives in `test_bigtiff_cog_compliance_2286.py`; promotion to stable follows the same release-cycle soak rule as the rest of the COG surface. (epic #2286)

**Experimental**
- GPU reads and writes (`reader.gpu`, `writer.gpu`). Cross-backend numerical parity is not claimed at this tier. (epic #2341)
- Experimental codecs: `lerc`, `jpeg2000`, `j2k`, `lz4`. Require `allow_experimental_codecs=True` on both read and write paths. Reader support across GDAL versions is uneven. (epic #2340)
- Permissive read escape hatches: `allow_rotated=True` (`reader.allow_rotated`) and `allow_unparseable_crs=True` (`reader.allow_unparseable_crs`). Both bypass a read-side check that the writer normally rejects. (epic #2340)
- Rich-tag writes: `writer.gdal_metadata_xml` and `writer.extra_tags`. Free-form payloads are written verbatim; interop with rasterio, libtiff, and GDAL depends on the payload. Gated by `allow_experimental_codecs=True` on writes from a fresh DataArray; round-tripped attrs from a previous read are exempt. (epic #2340)

**Internal-only**
- `codec.jpeg` (JPEG-in-TIFF). The encoder writes self-contained JFIF tiles without the TIFF JPEGTables tag (347), so the on-disk output is not interoperable with libtiff, GDAL, or rasterio. Reads and writes require the dedicated `allow_internal_only_jpeg=True` opt-in; `allow_experimental_codecs=True` does not unlock this path. (epic #2340)

**Unsupported for this release** (these combinations fail closed: the writer or reader raises rather than silently producing a wrong result)
- Warped VRTs (`<VRTDataset subClass="VRTWarpedDataset">`). (PR #2321)
- Nested VRTs (a `<SourceFilename>` pointing at another `.vrt`). (PR #2321)
- VRT mosaics whose sources disagree on CRS, pixel size, dtype, or band count without an explicit opt-in. The default raises `MixedBandMetadataError`. (PR #2321)
- `to_geotiff(..., cog=True, tiled=False)`. COG output requires the tiled layout. (epic #2286)
- Rotated transforms on the write path. The reader has an `allow_rotated=True` escape hatch in the experimental tier; the writer has no equivalent. (epic #2340)
- 1xN / Nx1 writes without an explicit `attrs['transform']`. The writer used to borrow the non-degenerate axis's spacing and silently invent the wrong pixel size; the writer now raises. (epic #2340)


#### Bug fixes and improvements
- Move shapely from a required dependency to an optional `vector` extra. shapely (and GEOS) used to load on every `import xrspatial` because `rasterize.py` imported it at module top level. It is now imported lazily, so a plain `pip install xarray-spatial` neither installs nor loads shapely. The only paths that need it are the vector-to-raster functions `rasterize` and `polygonize`; install `pip install xarray-spatial[vector]` to use them. Calling `rasterize` without shapely raises a clear ImportError pointing at the extra. Follows the same pattern as the matplotlib change (#2494). (#2496)
- Move matplotlib from a required dependency to an optional `plot` extra. A plain `pip install xarray-spatial` no longer pulls in matplotlib (and its pillow / fonttools / kiwisolver / contourpy / cycler / pyparsing chain), since no compute path imports it -- every matplotlib use is a lazy import inside the `.xrs.plot` accessor helpers. Install `pip install xarray-spatial[plot]` to use plotting; calling `.xrs.plot()` without it now raises a clear ImportError pointing at the extra instead of a bare `ModuleNotFoundError`. pandas stays required because xarray depends on it and several core modules import it directly. (#2494)
- Reconcile `xrspatial.geotiff.SUPPORTED_FEATURES` with the GeoTIFF release-contract tiering proposed in epic #2340. Adds `reader.windowed` at `stable` (covered by the existing window-read suite) and `reader.dask` at `stable` (covered by the cross-backend parity matrix in `test_backend_parity_matrix.py` and `test_backend_full_parity_2211.py`). Demotes `reader.allow_rotated` and `reader.allow_unparseable_crs` from `advanced` to `experimental` to match the epic's placement of permissive read-side escape hatches in the Experimental tier. A new shape test (`test_supported_features_shape_2348.py`) pins the structural invariants of the mapping (every entry carries a tier label; the tier set is closed at `{stable, advanced, experimental, internal_only}`; the dict literal contains no duplicate keys) so future drift fails CI. Runtime behaviour of every read and write path is unchanged; this is metadata-only. Callers gating on the exact string value of these two demoted entries will need to update their checks. (#2348)
- Promote the local COG read and write paths to the `stable` tier in `xrspatial.geotiff.SUPPORTED_FEATURES`. `SUPPORTED_FEATURES['writer.cog']` and `SUPPORTED_FEATURES['reader.local_cog']` now report `stable`; `reader.http_cog` stays `advanced` while the HTTP transport surface is contracted separately. The stable COG contract covers axis-aligned 2D / 3D rasters, the CPU writer and CPU reader, the lossless codecs (`none`, `deflate`, `lzw`, `zstd`, `packbits`), internal overviews, and normal CRS / transform / dtype / nodata / band / pixel-is-area / pixel-is-point round-trip. GPU COG paths, experimental codecs, rotated transforms, external `.tif.ovr` sidecars, file-like destinations with `cog=True`, BigTIFF COG, and HTTP COG remain outside the contract. Backed by the writer compliance suite (#2292), the cross-backend parity gate (#2293), and the per-tile byte-budget contract (#2294 / #2298). The reference docs (`docs/source/reference/geotiff.rst`) and the COG overview notebook spell out the full contract. (#2300)
- Resolve the GeoTIFF writer's `GeographicTypeGeoKey` / `ProjectedCSTypeGeoKey` decision via pyproj instead of an EPSG number range. The legacy heuristic (4326 + 4000-4999 -> geographic, else projected) silently mis-tagged geographic CRSes registered outside the 4000-4999 block (NAD83(2011) = 6318, GDA2020 = 7844, WGS 84 (G2139) = 9057, etc.) as projected and projected codes inside the block (4087 / 4088 / 4499) as geographic, corrupting the CRS at write time. The writer now calls `pyproj.CRS.from_epsg(...).is_geographic`. When pyproj can't classify a code (uninstalled, or installed but the local PROJ database lacks the entry), the writer raises the new `UnknownCRSModelTypeError` rather than guessing -- a small vetted allowlist (4326, 4269, 4267, 4258, 4283, 4322, 4230, 4019, 4047) is still honoured for the pyproj-missing case. `pyproj` is now listed under the `geotiff` extra. (#2277)
- Shut down the per-tile compression `ThreadPoolExecutor` on every exit path of the streaming tiled-write code in `to_geotiff`. The old code only called `shutdown(wait=True)` after the tile-row loop completed, so any mid-stream raise (compression failure, dask compute failure, file write failure) bypassed shutdown and leaked worker threads. The loop now runs inside `try/finally` and the finally calls `shutdown(wait=True, cancel_futures=True)` so queued tiles get dropped on the error path instead of blocking the unwind. The pool's workers carry an `xrspatial-geotiff-tile-compress` `thread_name_prefix` so leak-detection tests can tell them apart from dask's own offload/scheduler pools. (#2276)
- Remove read-side emission of the 13 deprecated GeoTIFF attrs (`crs_name`, `geog_citation`, `datum_code`, `angular_units`, `semi_major_axis`, `inv_flattening`, `linear_units`, `projection_code`, `vertical_crs`, `vertical_citation`, `vertical_units`, `colormap_rgba`, `cmap`) and bump `attrs['_xrspatial_geotiff_contract']` from 1 to 2. Downstream code that read these via `attrs[key]` now sees `KeyError`; migrate to `attrs.get(key)` or derive the value from `attrs['crs']` / `attrs['crs_wkt']` with pyproj. The `.xrs.plot()` accessor still surfaces palette colormaps by building a `ListedColormap` from the canonical `attrs['colormap']`. (#2016)
- Accept numpy integer scalars as the `crs=` argument to `to_geotiff` / `write_geotiff_gpu`. The validator already allowed `numbers.Integral`, but the writers gated EPSG assignment on `isinstance(crs, int)`, so `np.int32` / `np.int64` / `np.uint16` values passed validation then silently fell through with no EPSG written. (#2082)
- Tighten the writer's no-georef sentinel for integer x/y coords. The pre-fix check treated any integer dtype on either axis as the read-side no-georef placeholder and skipped transform inference, which also caught user-authored projected grids with integer-spaced coords (e.g. `x=[100,101,102], y=[200,199]`) and silently stripped their georef on write. The sentinel now matches only the exact reader pattern: `int64` ascending contiguous-step-1 arange on both axes. User-authored integer-coord grids that don't match (descending, non-unit step, non-uniform, or non-`int64`) now produce a real transform or raise `NonUniformCoordsError`. Coord values round-trip exactly through the new path; dtype flips int->float on subsequent reads. (#2087)
- Deprecate read-side emission of geographic-CRS GeoKey attrs (crs_name, geog_citation, datum_code, angular_units, semi_major_axis, inv_flattening); the writer cannot reconstruct them so they do not round-trip. These attrs still emit for now but trigger a DeprecationWarning. Removal planned for a future release. (#1984)
- Deprecate read-side emission of projected-CRS GeoKey attrs (linear_units, projection_code); the writer cannot reconstruct them so they do not round-trip. These attrs still emit for now but trigger a DeprecationWarning. Removal planned for a future release. (#1984)
- Deprecate read-side emission of vertical-CRS GeoKey attrs (vertical_crs, vertical_citation, vertical_units); the writer does not emit vertical GeoKeys so they do not round-trip. These attrs still emit for now but trigger a DeprecationWarning. Removal planned for a future release. (#1984)
- Refresh the geotiff mmap cache when a file is replaced under the same path so re-reads after an atomic-rename overwrite no longer return stale bytes
- Decode TIFF predictor=3 un-transpose by file byte order so big-endian floating-point TIFFs read back exactly
- Default internal `_vrt.read_vrt` `missing_sources` to `'raise'` so an unreadable VRT source no longer produces a silent zero-fill hole on integer rasters; pass `missing_sources='warn'` to opt back into the previous lenient behaviour (#1843)
- Deprecate read-side emission of matplotlib colormap-derived attrs (cmap, colormap_rgba) on palette TIFFs; the writer cannot set Photometric=3 so they do not round-trip. Construct ListedColormap from attrs['colormap'] in caller code. These attrs still emit for now but trigger a DeprecationWarning. Removal planned for a future release. (#1984)
- Reject writes whose `attrs['crs']` and `attrs['crs_wkt']` resolve to different CRSes (after pyproj canonicalisation) instead of silently emitting the EPSG and dropping the WKT. The new `ConflictingCRSError` (subclass of `ValueError` and `GeoTIFFAmbiguousMetadataError`) names the offending attrs; pass `crs=` explicitly to override both attrs and bypass the check. Read-back DataArrays carrying both attrs continue to round-trip because the reader's two attrs derive from the same on-disk CRS. (#1987)
- Reject reads whose CRS string cannot be parsed by pyproj instead of emitting it verbatim in `attrs['crs_wkt']` and letting downstream code crash on first use. Raises `UnparseableCRSError` (subclass of `ValueError`); pass `allow_unparseable_crs=True` to `open_geotiff` / `read_geotiff_dask` / `read_geotiff_gpu` / `read_vrt` to keep the legacy behaviour. The existing write-side raise from `_validate_crs_fallback` was retyped from plain `ValueError` to the new `UnparseableCRSError` subclass (no behaviour change). (#1987)
- Reject reads whose affine transform has non-zero rotation/shear terms instead of returning an axis-misaligned grid that downstream xrspatial ops (slope, aspect, hillshade, proximity, zonal) silently compute wrong results on. Raises `RotatedTransformError`; pass `allow_rotated=True` to `open_geotiff` / `read_geotiff_dask` / `read_geotiff_gpu` / `read_vrt` to read the pixel grid without the axis-aligned-grid assumption. (#1987)
- Extend `RotatedTransformError` to the plain GeoTIFF read path. Previously the VRT path raised the typed error via `_check_read_rotated_transform`, but a rotated `ModelTransformationTag` on a plain GeoTIFF raised a bare `NotImplementedError` from the parser, so `except RotatedTransformError` (or `except GeoTIFFAmbiguousMetadataError`) only caught the VRT case. Both entry points now raise the same typed error. `RotatedTransformError` subclasses `ValueError`, so callers catching `ValueError` are unaffected; callers that caught `NotImplementedError` specifically for the rotated branch should switch to `RotatedTransformError` (or `ValueError`). (#2267)
- Reject writes whose `coords['y']` or `coords['x']` are not uniformly spaced instead of silently using the first two values as the pixel size and misrepresenting the rest of the axis. Raises `NonUniformCoordsError`; the existing int-dtype sentinel convention from #1969 (used by the no-georef coord fallback) is exempted. (#1987)
- Reject writes whose `attrs['nodata']` disagrees with every concrete entry in `attrs['nodatavals']` instead of silently picking the canonical scalar and dropping the rioxarray tuple. Raises `ConflictingNodataError`; pass `nodata=` explicitly to the writer to override both attrs and bypass the check. `_FillValue` continues to be deprioritised per the existing resolver convention. (#1987)


### Version 0.9.9 - 2026-05-05

#### Bug fixes and improvements
- Add geotiff edge-case tests and integer-coord fallback (#1482) (#1492)
- Use synchronous dask scheduler for different-CRS merge parity test (#1495)
- Fetch COG tiles concurrently in HTTP path to mask RTT (#1487)
- Geotiff polish: validation, caching caps, parallelism thresholds, memory guards (#1488) (#1493)
- Round-trip transform, crs, and tag metadata through to_geotiff/open_geotiff (#1484) (#1494)
- Tighten geotiff reader: partial-tile shape check, ModelTransformation rotation guard (#1486) (#1491)
- Add geotiff writer test matrix (#1483) (#1490)
- Stream tile writes per dask chunk segment to bound peak memory in to_geotiff (#1485) (#1489)
- Add regression test for GPU pred=3 multi-sample TIFFs (#1479) (#1481)
- Make cubic resample prefilter explicit so chunk seams stay sub-eps (#1464) (#1478)
- Preserve input float dtype through resample() (#1467) (#1476)
- Resample polish: fix cubic depth comment, tighten NaN threshold, add edge tests (#1475)
- Fix dask aggregate boundary contamination and clean up bookkeeping (#1469) (#1477)
- Support 3D rasters, expose nodata, document target_resolution tuple in resample (#1466) (#1474)
- Cover cupy median/mode, dask+cupy, integer input, target_resolution tuple in resample tests (#1470) (#1473)
- Refresh transform and nodata attrs on resample output (#1465) (#1472)
- Inline dask aggregate kernel in resample (#1463) (#1468)
- Emit fresh grid metadata and propagate _FillValue in reproject (#1458) (#1462)
- Polish reproject/merge docstrings and cover Inf and parameter edge cases (#1459) (#1461)
- Batch CuPy reductions and drop redundant copies in reproject (#1460)
- Add transform_precision parameter to merge() (#1452) (#1456)
- Preserve non-spatial coords through reproject() and merge() (#1455)
- Guard _apply_vertical_shift against non-finite coords; add vertical CRS tests (#1453)
- Honor per-raster nodata sentinels in merge() (#1448) (#1449)
- Preserve input attrs through reproject() and merge() (#1446)
- Fix dask reproject dtype and same-CRS dask merge (#1450)
- Validate raster inputs in reproject public APIs (#1431) (#1432)
- Add memory guard and scalar validation to generate_terrain (#1443) (#1444)
- Validate scalar parameters in hydro public APIs (#1427) (#1428)
- Validate mannings_n DataArray values in flood (#1437) (#1438)
- Add _validate_raster on secondary DataArray args in hydro (#1425) (#1426)
- Validate grid/bounds/precision params in reproject (#1433) (#1434)
- Validate cellsize in hydro public APIs (#1429) (#1430)
- Validate pathfinding inputs and cap waypoints (#1439) (#1440)
- Validate raster/mask inputs in polygonize (#1441) (#1442)
- Add memory guard to flow_direction_mfd numpy/cupy backends (#1423) (#1424)
- Reject NaN/Inf in reproject scalar inputs (#1435) (#1436)
- Lazy assembly for hand_mfd dask path (#1416) (#1417)
- Return NaN from glcm_texture angle=None when no angle has valid pairs (#1408) (#1409)
- Use ground distance for sky_view_factor horizon angle (#1407) (#1410)
- Exclude centre cell from morph_erode/dilate when kernel[centre]==0 (#1397) (#1398)
- Use sized slice in dask morph chunk writeback (#1399) (#1400)
- Merge sink_d8 labels across dask tile boundaries (#1394) (#1395)
- Reject complex dtypes in _validate_raster() (#1384) (#1387)
- Reject underflowing sigma_spatial in bilateral() (#1390) (#1392)
- Use int64 row_ptr in _build_row_csr_numba (#1388) (#1391)
- Reject mixed-backend arrays in validate_arrays() (#1383) (#1386)
- Reject non-positive inputs in wpm() (#1382) (#1385)
- Guard _idw knearest against unbounded grid x k allocation (#1377) (#1380)
- Validate raster dtype in convolve_2d() (#1389) (#1393)
- Reject all-zero rasters in mesh_utils.create_triangulation (#1378) (#1381)
- Validate dtypes per band in mahalanobis() (#1376) (#1379)
- Guard owa() against unbounded criteria stack (#1370) (#1375)
- Guard spline() TPS against unbounded memory allocations (#1372) (#1374)
- Reject n_samples<=0 in mcda sensitivity (#1371) (#1373)
- Guard flow_path_d8() against unbounded memory allocations (#1364) (#1368)
- Guard flow_path_mfd() against unbounded memory allocations (#1365) (#1369)
- Guard flow_path_dinf() against unbounded memory allocations (#1363) (#1367)
- Guard snap_pour_point_d8() against unbounded memory allocations (#1362) (#1366)
- Guard sink_d8() against unbounded memory allocations (#1356) (#1361)


### Version 0.9.8 - 2026-04-29

#### Bug fixes and improvements
- Guard stream_order_dinf() against unbounded memory allocations (#1350) (#1354)
- Guard basin_d8() against unbounded memory allocations (#1357) (#1359)
- Guard flow_length_dinf() against unbounded memory allocations (#1355) (#1358)
- Guard flow_length_mfd() against unbounded memory allocations (#1351) (#1353)
- Guard stream_order_mfd() against unbounded memory allocations (#1349) (#1352)
- Guard watershed_dinf() against unbounded memory allocations (#1345) (#1348)
- Guard stream_link_dinf() against unbounded memory allocations (#1343) (#1347)
- Guard hand_dinf() against unbounded memory allocations (#1344) (#1346)
- Guard stream_link_mfd() against unbounded memory allocations (#1337) (#1341)
- Guard flow_length_d8() against unbounded memory allocations (#1327) (#1332)
- Guard hand_d8() against unbounded memory allocations (#1323) (#1326)
- Guard hand_mfd() against unbounded memory allocations (#1338) (#1340)
- Guard watershed_mfd() against unbounded memory allocations (#1339) (#1342)
- Guard watershed_d8() against unbounded memory allocations (#1329) (#1330)
- Guard fill_d8() against unbounded eager allocations (#1334) (#1336)
- Guard stream_link_d8() against unbounded memory allocations (#1333) (#1335)
- Guard stream_order_d8() against unbounded memory allocations (#1328) (#1331)
- Guard flow_accumulation_dinf() against unbounded memory allocations (#1322) (#1325)
- Guard flow_accumulation_mfd() against unbounded memory allocations (#1321) (#1324)
- Guard gpu_rtx hillshade and viewshed against unbounded GPU allocations (#1308) (#1310)
- Guard min_observable_height() against unbounded boolean stack (#1317) (#1320)
- Guard flow_accumulation() against unbounded eager allocations (#1318) (#1319)
- Reject NaN/Inf weights in mcda combine and ahp_weights (#1311) (#1312)
- Support TIFF predictor=3 on the CPU write path (#1313) (#1314)
- Guard kriging() against unbounded N x N and grid x N allocations (#1307) (#1309)
- Guard surface_distance() against unbounded eager allocations (#1303) (#1305)
- Guard landforms() against unbounded kernel allocations (#1302) (#1304)
- Enforce equal band shapes in true_color() (#1293) (#1301)
- Guard sky_view_factor() against unbounded eager allocations (#1299) (#1300)
- Guard resample() against unbounded output allocations (#1295) (#1297)
- Guard sieve() numpy and cupy backends against oversized rasters (#1296) (#1298)
- Guard kde and line_density against oversize output grids (#1287) (#1289)
- Reject oversize focal kernels before allocation (#1284) (#1286)
- Guard mahalanobis allocations against oversized rasters (#1288) (#1290)
- Guard true_color() eager backends against oversized rasters (#1291) (#1292)
- Add memory guard to geodesic slope/aspect (#1283) (#1285)
- Validate dt against CFL stability bound in diffuse() (#1281) (#1282)
- Cap erode() parameters and run memory guard on every backend (#1275) (#1277)
- Validate cellsize in curvature() (#1270) (#1272)
- Add memory guard to emerging_hotspots public API (#1274) (#1276)
- Validate input rasters in edge_detection public API (#1271) (#1273)
- Guard diffuse() against unbounded allocations and steps loop (#1267) (#1268)
- Add memory guards to dasymetric public API (#1261) (#1263)
- Guard cost_distance CuPy backend against oversize rasters (#1262) (#1264)
- Guard morphology kernel allocations against oversized kernels (#1256) (#1258)
- Cap window_size and distance in glcm_texture (#1257) (#1259)
- Guard cost_distance numpy path against oversize rasters (#1252) (#1253)


### Version 0.9.7 - 2026-04-23

#### New features
- Add kernel density estimation (KDE) (#1170)
- Add standalone raster resampling (#1152) (#1172)
- Add GPU COG overview support (#1150) (#1174)
- Add geometry simplification to polygonize() (#1176)
- Add polygon clipping function (#1144) (#1173)
- Add hypsometric_integral to zonal module (#1073)

#### Bug fixes and improvements
- Fix dask OOM in visibility and viewshed modules (#1167)
- QA/QC: align sieve with GDAL and add parity tests (#1169)
- Numba-ize visibility module loops (#1177) (#1179)
- Fix hillshade gradient to use Horn's method (#1175) (#1178)
- Fix duplicate notebook numbers and restructure to standard format (#1181)
- Fix OOM in geotiff dask read, sieve memory, and reproject GPU fallback (#1183)
- Add memory guards to reproject CuPy paths and output grid (#1186, #1187) (#1188)
- Fix NaN pixels producing spurious polygons in numpy/dask backends (#1190) (#1194)
- Fix CuPy Bellman-Ford iteration limit in cost_distance (#1192)
- Fix geotiff unbounded allocation DoS and VRT path traversal (#1189)
- Fix KDE all-zero output with descending-coordinate templates (#1199)
- Fix crop=True dropping boundary pixels when all_touched=True (#1197) (#1200)
- Add allocation guards to GPU read and VRT read paths (#1196)
- Fix resample interpolation coordinate mapping (#1202) (#1204)
- Fix float32 truncation in balanced_allocation iteration loop (#1203) (#1205)
- Fix bump OOM with int32 coords, default-count cap, per-chunk dask partitioning (#1206) (#1208)
- Pass dask chunks to rasterize in clip_polygon to keep mask lazy (#1207) (#1209)
- Cut head_tail_breaks and box_plot dask re-scans (#1213)
- Fuse hypsometric_integral dask path to a single graph evaluation (#1212)
- Cap dask graph size in read_geotiff_dask and batch adler32 transfers (#1211)
- Bound per-tile allocations in TIFF reader (#1215) (#1216)
- Fix GPU predictor kernel stride for multi-sample tiled TIFFs (#1220) (#1222)
- Reject TIFFs whose declared tile grid exceeds TileOffsets length (#1219) (#1221)
- Reject oversize rasterize outputs before allocation (#1223) (#1224)
- Fix cupy zonal_stats silently ignoring nodata_values=0 (#1227) (#1228)
- Guard viewshed numpy path against oversize rasters (#1229) (#1230)
- Reject integer-dtyped input in perlin() (#1232) (#1233)
- Reject oversize bump output rasters before allocation (#1231) (#1234)
- Clamp bilateral kernel radius to raster extent (#1236) (#1238)
- Fix CPU fp_predictor_decode for multi-band predictor=3 TIFFs (#1247) (#1250)
- Guard natural_breaks against oversize Jenks matrices (#1246) (#1248)
- Handle degenerate input in equal_interval (#1244) (#1245)
- Reject oversize convolution kernels before allocation (#1241) (#1243)
- Guard contour segment buffers against oversize allocation (#1240) (#1242)


### Version 0.9.6 - 2026-04-05

#### New features
- Multi-observer viewshed and line-of-sight profiles (#1145) (#1160)
- Sieve filter for removing small raster clumps (#1159)

#### Bug fixes and improvements
- Faster sieve labeling, with convergence warning (#1163)
- Dask laziness docs and regression tests (#1164) (#1165)
- Fix README feature matrix to match codebase (#1155) (#1158)
- Fix 196 test-suite warnings (#1148) (#1157)
- ASV benchmarks for 6 modules changed in v0.9.5 (#1156)
- Caveat and assumption admonitions in docs (#1134)
- Miscellaneous code sweeps (#1161)


### Version 0.9.5 - 2026-03-31

#### Bug fixes and improvements
- Add GPU memory guard to reproject dask+cupy path (#1131)
- Add memory guard to surface_distance geodesic dd_grid (#1129)
- Add memory guard to dasymetric validate_disaggregation (#1127)
- Replace boolean indexing with lazy reductions in normalize dask paths (#1125)
- Keep northness/eastness lazy on dask arrays (#1123)
- Add memory guard to erosion dask paths (#1121)
- Add memory guard to cost_distance iterative Dijkstra and da.block assembly (#1119)
- Fix diffusion dask OOM by passing scalar diffusivity directly to chunks (#1117)
- Fix balanced_allocation OOM with lazy source extraction and memory guard (#1115)
- Fix zonal dask memory guards and stats filtering (#1112)


### Version 0.9.4 - 2026-03-30

#### New features
- Streaming TIFF write for dask inputs (#1084) (#1108)
- Add dtype, compression_level, and VRT output to geotiff I/O (#1083) (#1085)

#### Bug fixes and improvements
- Use float64 in Jenks natural breaks internals (#1100) (#1101)
- Fix multi-metric output mislabeling in glcm_texture (#1106) (#1107)
- Fix inverted decay and off-by-one in bump spread (#1102) (#1103)
- Propagate NaN from curve_number in curve_number_runoff (#1104) (#1105)
- Fix target_elev contaminating horizon in dask viewshed distance sweep (#1098) (#1099)
- Preserve float64 precision in convolve_2d (#1096) (#1097)
- Fix SAVI formula: (1+L) was in denominator instead of numerator (#1094) (#1095)
- Fix NaN handling in focal_stats CUDA kernels (#1092) (#1093)
- Fix three accuracy bugs in zonal stats dask backend (#1090) (#1091)
- Add longitude normalization to CUDA inverse projection kernels (#1089)
- Fix three accuracy bugs in reproject resampling kernels (#1087)
- Fix three accuracy bugs in open_geotiff/to_geotiff (#1081) (#1082)


### Version 0.9.3 - 2026-03-27

#### New features
- GeoTIFF/COG reader and writer with GPU acceleration and datum transforms (#1035) (#1062)
- nvJPEG GPU acceleration for JPEG-compressed GeoTIFFs (#1050) (#1055)
- LZ4 and LERC compression codecs for GeoTIFF (#1063)
- Dask-native out-of-core reproject and merge module (#1031)
- MCDA module for spatial multi-criteria decision analysis (#1030) (#1058)
- Edge detection filters: Sobel, Laplacian, Prewitt (#1042)
- Focal variety statistic (#1040) (#1043)
- NDSI, NDBI, BAI, MSAVI2, and OSAVI spectral indices (#1044)
- Northness and eastness aspect functions (#1039) (#1041)
- Rescale and standardize normalization utilities (#1028)
- Morphological gradient, white top-hat, and black top-hat operators (#1026)
- D-inf and MFD variants of flow_path, watershed, and HAND (#1020)
- D-infinity flow length computation (#1012) (#1013)
- MFD flow length computation (#1011)
- Vegetation-aware flood modeling: roughness, curve number, and depth from land cover (#1029)
- Lightweight CRS parser, removes hard pyproj dependency (#1072)
- plot() accessors for DataArray and Dataset (#1074)
- fused_overlap and multi_overlap dask graph utilities (#1071)
- rechunk_no_shuffle utility (#1068)

#### Bug fixes and improvements
- Fix D-inf flow direction odd facet decomposition per Tarboton 1997 (#1005)
- Fix CuPy uint8 overflow and CUDA cubic NaN fallback (#1054) (#1064)
- Fix stale full pipeline benchmark numbers in README (#1077) (#1079)
- Fix duplicate prefix numbers in user guide notebooks (#1076) (#1078)
- Memory-safe rechunk, preview chunk budget, plot improvements (#1075)
- Reduce dask graph size for large raster reprojection (#1065)
- Speed up cost_distance iterative tile Dijkstra 2-4x (#1023)
- Faster polygonize: single-pass tracing, JIT merge helpers, batch shapely (#1010)
- Rename GeoTIFF API to xarray conventions (#1047) (#1061)
- Section index in README feature matrix (#1033) (#1034)
- Dask example for reproject (#1066)
- Porto taxi trajectory rasterization example (#1022)
- Polygonize benchmark comparing xrspatial vs rasterio (#1006) (#1007)


### Version 0.9.2 - 2026-03-13

#### New features
- Vector rasterize function for polygons, lines, and points (#989) (#990)
- preview() for memory-safe raster downsampling (#987)
- D-inf and MFD stream ordering and link segmentation (#983) (#984)
- Zonal functions accept vector zones directly (#999)
- Dask backends for rasterize() (#997)

#### Bug fixes and improvements
- Fix rasterize accuracy: GPU ceil, half-open fill, all_touched pairing (#995) (#996)
- Fix scanline row gaps, speed up rasterize, add resolution/like/merge params (#991)
- Rasterization fixes (#992)
- Fix cupy failures in balanced_allocation and corridor (#985)
- Refactor preview to avoid rechunking (#988)
- Replace datashader with matplotlib in user guide notebooks (#1002)
- Refactor user guide notebooks 10-31 to standard structure (#1003)
- Water/land classification example in GLCM notebook (#994)
- Source/reference column in README feature matrix (#977) (#978)
- Small notebook fixes


### Version 0.9.1 - 2026-03-04

#### New Features
- Add contour line extraction via marching squares (#964) (#974)
- Add GLCM texture metrics (#963) (#973)
- Add sky-view factor function (#962) (#972)
- Add bilateral filter for feature-preserving smoothing (#969) (#970)


### Version 0.9.0 - 2026-03-04

#### New Features
- Add least-cost corridor analysis (#965) (#968)
- Add native Dask+CuPy backends for hydrology core functions (#952) (#966)
- Add native CUDA kernel for hydraulic erosion (#961) (#967)
- Add CuPy and Dask+CuPy backends for kriging (#951) (#960)
- Add NDWI and MNDWI water indices (#959)
- Add morphological raster operators (#949) (#958)
- Add TPI-based landform classification (#950) (#957)
- Add MFD flow direction and accumulation (#946) (#956)
- Add balanced service area partitioning (#939) (#943)
- Add Dinf support to flow_accumulation() (#945)
- Add scalar diffusion solver (#940) (#944)
- Add multi_stop_search for multi-waypoint routing (#935) (#937)
- Add raster-based dasymetric mapping module (#930) (#936)
- Add interpolation tools: IDW, Kriging, and Spline (#932) (#934)

#### Bug Fixes & Improvements
- Remove esri.py and datashader from core dependencies (#953)
- Add HAND and TWI to README feature matrix (#954) (#955)
- Add missing API reference docs for ~30 undocumented functions (#941) (#942)
- Remove unnecessary array copies to reduce memory allocations (#933)


### Version 0.8.0 - 2026-03-03

#### New Features
- Add raster-based dasymetric mapping: disaggregate, pycnophylactic interpolation, and validation helper (#930)
- Add enhanced terrain generation features (#929)
- Add fire module: dNBR, RdNBR, burn severity, fireline intensity, flame length, rate of spread (Rothermel), and KBDI (#927)
- Add flood prediction tools (#926)


### Version 0.7.0 - 2026-03-02

Highlights: new hydrology, terrain metrics, and surface distance modules; 3D
multi-band support for focal functions; GPU-accelerated polygonize with
connected-component labelling; and broad dask+cupy backend coverage across the
library.

#### New Features
- Add fire module: dNBR, RdNBR, burn severity class, fireline intensity, flame length, rate of spread (Rothermel), and KBDI (#922)
- Add 3D multi-band support to focal functions (#924)
- Add foundational hydrology tools (#921)
- Add terrain metrics module: TRI, TPI, and Roughness (#920)
- Add surface_distance module: 3D terrain distance via multi-source Dijkstra (#918)
- Add CI benchmark workflow with regression detection (#917)
- Polygonize: GPU CCL, promotion, dispatcher, GeoJSON output, and benchmarks (#916)
- Add GPU (CuPy) backends for proximity, allocation, direction (#909)
- Add GPU (CuPy) backend for cost_distance (#910)
- Add out-of-core dask CPU viewshed (#897)
- Add dask+cupy backends for focal tools (#896)
- Add emerging_hotspots() for time-series trend analysis (#890)
- Replace O(n^4) regions() with scipy union-find, add dask/cupy backends (#898)

#### Bug Fixes & Improvements
- Make dask backends truly lazy, add agg parameter (#914)
- Fill remaining dask+cupy gaps (terrain, perlin, crosstab) (#913)
- Add comprehensive input validation across public API (#912)
- Fix dask zonal.stats() bug, add dask+cupy backend, edge-case tests (#911)
- Prevent OOM and unknown chunks in classify.py dask paths (#895)
- Replace np.unique/np.isfinite with dask-safe helpers in zonal.py (#894)
- Extend KDTree path to allocation/direction to prevent OOM (#893)
- Add memory guard and tiled KDTree fallback to proximity (#892)
- Add memory guard, bounded A*, and HPA* to prevent OOM (#891)


### Version 0.6.0 - 2026-02-24

Highlights: xarray accessor (`.xrs`) is now available, providing a convenient
interface for all spatial operations directly on DataArrays and Datasets.

- Fixes #880: replace single-chunk rechunk in cost_distance with iterative tile Dijkstra (#888)
- Fix test warnings: suppress expected GPU/NaN warnings, fix scalar conversion and host-array copies (#887)
- Fixes #804: add min_observable_height() to experimental (#875) (#886)
- Fixes #845: add configurable boundary mode to kernel-based spatial ops (#874)
- Fixes #587: unify project requirements around setup.cfg (#873)
- Document CPU vs GPU algorithm difference in viewshed docstring (#872)
- Fixes #114: add Mahalanobis distance metric (#871)
- Fix perlin/terrain dask backends: enable parallelism and out-of-core support (#870)
- Fixes #790: add .xrs xarray accessors (#868)
- Fixes #748: align hillshade with GDAL and add dask+cupy backend (#867)
- Added cupy support to true_color function (#866)
- Fixes #864: remove local analysis module, add migration guide (#865)
- Add GPU spill-to-CPU fallback for cost_distance (#863)
- Fixes #560: weighted A* pathfinding with friction surface (#862)


### Version 0.5.3 - 2026-02-22
- Fixes #733: add cost_distance() for weighted proximity (#859)
- Fixes #734: make noise a lazy import so datasets module is usable without it (#858)
- Fixes #774: suppress dd.concat unknown divisions warning in zonal_stats (#857)
- Fixes #392: document and test 3D time-series zonal stats via Dataset (#856)
- Fix hotspots dask performance: 2-pass streaming with O(chunk_size) memory (#855)
- Fixes #134: add xr.Dataset as input type for appropriate modules (#854)
- Warnings(17) filtering for non-GPU (#850)
- Fixes #190: GPU-enable all classification operations (#852)
- updating links and text (#849)
- fixes 766 document handling of f32 vs. f64 data when using xarray spatial (#847)
- Fix rioxarray band_as_variable DataArray extraction (#846)
- added majority as default stat for zonal operations with numpy and cupy (#844)
- fixed proximity intermediate large numpy array problem and proximity output coords are now lazy if input is dask array (#843)
- Changed regions parameter connections to neighborhood and updated docstring (#842)
- Testing and Updating Examples (#837)
- added horizontal vertical unit mismatch heuristic to help address issue #840 (#841)
- Fixes #838 update legacy repo url links (#839)
- added colorado dem to examples datasets yaml


### Version 0.5.2 - 2025-12-18
- Make dask optional (#835)
- Fixes 832 update citation info in readme (#834)
- Removed references to xarray-spatial.org (#827)


### Version 0.5.1 - 2025-12-16
- updated runner, python setup action and checkout actions for PyPI publish


### Version 0.5.0 - 2025-12-15
- Python 3.14 is now supported!
- Fixed bug in curvature dask+cupy args and added unit test for curvature(#824)
- Added dask cupy test for slope func (#824)
- Added dask cupy test for aspect func (#824)
- Added in dask-cupy convolve_2d test (#823)
- Now ensures the hash value fits into an unsigned 64-bit integer for NumPy 2.0.0 (#805)
- Added dask and pyarrow to setup.cfg test area (#822)
- Allow Negative Target Height in Viewshed Analysis (viewshed.py) (#812)
- Small Fixes while testing Cuda 13 (#818)
- Support for certain Dask+Cupy (#815)
- Update docstring for viewshed (#807)


### Version 0.4.0 - 2024-04-25
- Python 3.12 is now supported!
- Python 3.9 & 3.8 are no longer supported.
- Documentation has also been moved to https://xarray-spatial.readthedocs.io/

Pull Requests merged in this release include:
- Add links to docs to readme (#800)
- Update readthedocs configuration (#798)
- Fix typo in readthedocs config (#799)
- Move docs to readthedocs (#797)
- Bunch of CI/CD-related fixes (#796)
- chore: Remove numpy pin, pin datashader, drop Python 3.7 (#789)

### Version 0.3.7 - 2023-06-05
The 0.3.7 release is a hot fix for 0.3.6, which has problem with publishing to PyPi
as the package exceeds the limit of 100MB. In this new release, example notebooks are
cleaned up to reduce the package size.

#### Enhancements
- clear example notebook outputs (#786)

### Version 0.3.6 - 2023-06-02
With the 0.3.6 release, xarray-spatial now supports python 3.11.
This release focuses on demonstrating the reliability of the library by adding more tests
against GDAL/QGIS. 

#### Enhancements
- Test on Python 3.11 (#750)
- Pin numpy version (#780)
- zonal stats (#764)
- updated citation (#769)
- classify.binary: handle NaNs and infinite values (#763)
- Test against QGIS, GDAL (#744)
- Zonal crosstab user guide notebook enhancement (#759)
- cpu_bin with binary search (#760)
- outline for housing price feature engineering nb (#725)
- More examples to user guide (#742)
- Use setuptools (#749)
- Update contributor badge (#740)
- update pharmacy desserts notebook (#723)
- Update README.md (#719)
- viewshed gpu notebook (#720)
- Fixed feature-proposal.md config (#718)

#### Bug Fixes
- correct docs for proximity (#778)
- Zonal_crosstab 3D: Ensure content of input param `values` is preserved (#754)
- Multispectral tools: convert data to float32 dtype before doing calculations (#755)


### Version 0.3.5 - 2022-06-05
The 0.3.5 release mainly addresses the scaling issue in GPU viewshed to gain better accurate triangulation.
The GPU raytraced viewshed should now give comparable results to the CPU version.
However, the 2 versions use 2 different approaches, there can be slightly differences at some points
where a version returns visible while the other considers them as invisible.

#### Enhancements
- command to get change log (#716)
- Added Feature Proposal Template (#714)

#### Bug Fixes
- Improved viewshed rtx. Now result should match the CPU version (#715)

### Version 0.3.4 - 2022-06-01
The 0.3.4 release primarily a bug fix release but also includes a number of enhancements with a focus on GPU supports.

#### Enhancements
- NumPy zonal stats: return a data array of calculated stats (#685)
- set unit for hotspots output (#686)
- More robust cuda and cupy identification (#657)
- Remove deprecated tiles module (#698)
- Test on python 3.10, remove 3.6 (#694)
- moved all tests to github actions (#689)
- Add isort to pytest (#700)
- Add flake8 to pytest (#697)
- Remove unnecessary executable flags (#696)
- updated test hotspots gpu (#692)
- 3D numpy zonal_crosstab to support more agg methods (#687)

#### Bug Fixes
- Fix rtx viewshed rendering blank image (#711)
- Convolve_2d gpu fixes (#702)
- focal.mean(): only do data type conversion once (#699)
- Update to remote sensing notebook (#688)
- focal_stats(): gpu case (#709)
- focal apply: drop gpu support (#706)
- drop gpu support (#705)
- enabled numba.cuda.jit in hotspots cupy (#691)

#### Documentation
- Correct examples in docstrings (#703)
- Fix doc build dependencies in CI (#683)
- Fix link to Austin road network notebook (#695)

### Version 0.3.3 - 2022-03-21
- fixed ubuntu version (#681)
- Don't calculate angle when not needed (#677)
- codecov: ignore all tests at once (#674)
- add more tests to focal module (#676)
- classify: more tests (#675)
- Codecov: disable Numba; ignore tests, experimental, and gpu_rtx (#673)
- Improve linter: add isort (#672)
- removed stale test files from project root (#670)
- User guide fixes (#665)
- license year in README to include 2022 (#668)
- install dependencies specified in test config (#666)
- Pytests for CuPy zonal stats (#658)
- add Codecov badge to README
- codecov with github action (#663)
- Modernise build system (#654)
- classify tools: classify infinite values as nans, natural_breaks: classify all data points when using sub sample (#653)
- Add more benchmarks (#648)
- Stubbed out function for Analytics module (#621)
- Fix doc build failure due to Jinja2 version (#651)

### Version 0.3.2 - 2022-02-04
- Remove numpy version pin (#637)
- aspect: added benchmarks (#640)
- Clean gitignore and manifest files (#642)
- Benchmark results (#643)
- handle CLI errors #442 (#644)
- Cupy zonal (#639)
- Tests improvements (#636)

### Version 0.3.1 - 2022-01-10
- Add benchmarking framework using asv (#595)
- Fix classify bug with dask array (#599)
- polygonize function on cpu for numpy-backed xarray DataArrays (#585)
- Test python 3.9 on CI (#602)
- crosstab: speedup dask case (#596)
- Add benchmark for CPU polygonize (#605)
- Change copyright year to include 2021 (#610)
- Docs enhancement (#604, #628)
- code refactor: use array function mapper, add messages param to not_implemented_func() (#612)
- condense tests (#613)
- Multispectral fixes (#617)
- Change copyright year to 2022 (#622)
- Aspect: convert to float if int dtype input raster (#619)
- direction(), allocation(): set all NaNs at initalization (#618)
- Add rtx gpu hillshade with shadows (#608)
- Add hillshade benchmarking, for numpy, cupy and rtxpy (#625)
- Focal mean: handle nans inside kernel (#623)
- Convert to float32 if input raster is in int dtype (#629)

### Version 0.3.0 - 2021-12-01
- Added a pure numba hillshade that is 10x faster compared to numpy (#542)
- dask case proximity: process whole raster at once if max_distance exceed max possible distance (#558)
- pathfinding: `start` and `goal` in (y, x) format (#550)
- generate_terrain: cupy case, dask numpy case (#555)
- Optimize natural_break on large inputs (#562)
- Fixes in CPU version of natural_breaks. (#562) (#563)
- zonal stats, speed up numpy case (#568)
- Ensure that cupy is not None (#570)
- Use explicit cupy to numpy conversion in tests (#573)
- zonal stats: speed up dask case (#572)
- zonal_stats: ensure chunksizes of zones and values are matching (#574)
- validate_arrays: ensure chunksizes of arrays are matching (#577)
- set default value for num_sample (#580)
- Add rtx gpu viewshed and improve cpu viewshed (#588)

### Version 0.2.9 - 2021-09-01
- Refactored proximity module to avoid rechunking (#549)

### Version 0.2.8 - 2021-08-27
- Added dask support to proximity tools (#540)
- Refactored the resample utils function and changed their name to canvas_like (#539)
- Added zone_ids and cat_ids param to stats zonal function (#538)

### Version 0.2.7 - 2021-07-30
- Added Dask support for stats and crosstab zonal functions (#502)
- Ignored NaN values on classify functions (#534)
- Added agg param to crosstab zonal function (#536)

### Version 0.2.6 - 2021-06-28
- Updated the classification notebook (#489)
- Added xrspatial logo to readme (#492)
- Removed reprojection notebook old version (#494)
- Added true_color function to documentation (#494)
- Added th params to true_color function (#494)
- Added pathfinding nb data load guidance (#491)

### Version 0.2.5 - 2021-06-24
- Added reprojection notebook (#474)
- Reviewed local tools notebook (#466)
- Removed save_cogs_azure notebook (#478)
- Removed xrspatial install guidance from makepath channel (#483)
- Moved local notebook to user guide folder (#486)
- Fixed pharmacy notebook (#479)
- Fixed path-finding notebook data load guidance (#480)
- Fixed focal notebook imports (#481)
- Fixed remote-sensing notebook data load guidance (#482)
- Added output name and attrs on true_color function (#484)
- Added classify notebook (#477)

### Version 0.2.4 - 2021-06-10
- Added resample notebook (#452)
- Reviewed mosaic notebook (#454)
- Added local module (#456)

### Version 0.2.3 - 2021-06-02
- Added make terrain data function (#439)
- Added focal_stats and convolution_2d functions (#453)

### Version 0.2.2 - 2021-05-07
- Fixed conda-forge building pipeline
- Moved all examples data to Azure Storage (#424)

### Version 0.2.1 - 2021-05-06
- Added GPU and Dask support for Focal tools: mean, apply, hotspots (#238)
- Moved kernel creation functions to convolution module (#238)
- Update Code of Conduct (#391)
- Fixed manhattan distance to sum of abs (#309)
- Example notebooks running on PC Jupyter Hub (#370)
- Fixed examples download cli cmd (#349)
- Removed conda recipe (#397)
- Updated functions and classes docstrings (#302)

### Version 0.2.0 - 2021-04-28
- Test release for new github actions

### Version 0.1.9 - 2021-04-27
- Deprecated tiles module (#381)
- Added user guide on the documentation website (#376)
- Updated docs design version mapping (#378)
- Added Github Action to publish package to PyPI (#371)
- Moved Spatialpandas to core install requirements for it to work on JLabs (#372)
- Added CONTRIBUTING.md (#374)
- Updated `true_color` to return a `xr.DataArray` (#364)
- Added get_data module and example sentinel-2 data (#358)
- Added citations guidelines and reformat (#382)

### Version 0.1.8 - 2021-04-15
- Fixed pypi related error

### Version 0.1.7 - 2021-04-15
- Updated multispectral.true_color: sigmoid contrast enhancement (#339)
- Added notebook save cogs in examples directory (#307)
- Updated Focal user guide (#336)
- Added documentation step on release steps (#346)
- Updated cloudless mosaic notebook: use Dask-Gateway (#351)
- Fixed user guide notebook numbering (#333)
- Correct warnings (#350)
- Add flake8 Github Action (#331)

### Version 0.1.6 - 2021-04-12
- Cleared metadata in all examples ipynb (#327)
- Moved docs requirements to source folder (#326)
- Fixed manifest file
- Fixed travis ci (#323)
- Included yml files
- Fixed examples path in Pharmacy Deserts Noteboo
- Integrate xarray-spatial website with the documentation (#291)

### Version 0.1.5 - 2021-04-08
- CLI examples bug fixed
- Added `drop_clouds`, cloud-free mosaic from sentinel2 data example (#255)

### Version 0.1.4 - 2021-04-08
- Sphinx doc fixes
- CLI bug fixed in 0.1.5

### Version 0.1.3 - 2021-04-05
- Added band_to_img utils func
- Added download-examples CLI command for all notebooks (#241)
- Added band_to_img utils func
- Docs enhancements
- GPU and dask support for multispectral tools
- GPU and Dask support for classify module (#168)
- Fixed savi dask cupy test skip
- Moved validate_arrays to utils
- Added GPU support for hillshade (#151)
- Added CLI for examples data
- Improved Sphinx docs / theme

### Version 0.1.2 - 2020-12-01
- Added GPU support for curvature (#150)
- Added dask.Array support for curvature (#150)
- Added GPU support for aspect (#156)
- Added dask.Array support for aspect (#156)
- Added GPU support for slope (#152)
- Added dask.Array support for slope (#152)
- Fixed slope cupy: nan edge effect, remove numpy padding that cause TypeError (#160)
- Fixed aspect cupy: nan edge effect, remove numpy padding that cause TypeError(#160)
- Updated README with Supported Spatial Features Table
- Added badge for open source gis timeline
- Added GPU Support for Multispectral tools (#148)
- Added Python 3.9 to Test Suite

### Version 0.1.1 - 2020-10-21
- Added convolution module for use in focal statistics. (#131)
- Added example notebook for focal statistics and convolution modules.

### Version 0.1.0 - 2020-09-10
- Moved kernel creation to name-specific functions. (#127)
- Separated the validate and custom kernel functions. (focal)
- Added annulus focal kernel (#126) (focal)
- Added outputting of z-scores from hotspots tool (focal)
- Changed type checking to use np.floating (focal)
- Added tests for refactored focal statistics (focal)

### Version 0.0.9 - 2020-08-26
- Added A* pathfinding
- Allow all numpy float data types, not just numpy.float64 (#122)
- Broke out user-guide into individual notebooks  (examples)
- Added num_sample param option to natural_breaks (#123)
- Removed sklearn dependency

### Version 0.0.8 - 2020-07-22
- Fixed missing deps

### Version 0.0.7 - 2020-07-21
- Added 2D Crosstab (zonal)
- Added suggest_zonal_canvas (zonal)
- Added conda-forge build
- Removed Versioneer
- Updates to CI/CD

### Version 0.0.6 - 2020-07-14
- Added Proximity Direction (proximity)
- Added Proximity Allocation (proximity)
- Added Zonal Crop (zonal)
- Added Trim (zonal)
- Added ebbi (multispectral)
- Added more tests for slope (slope)
- Added image grid (readme)

### Version 0.0.5 - 2020-07-05
- Changed ndvi.py -> multispectral.py
- Added arvi (multispectral)
- Added gci (multispectral)
- Added savi (multispectral)
- Added evi (multispectral)
- Added nbr (multispectral)
- Added sipi (multispectral)
- Added `count` to default stats (zonal)
- Added regions tools (zonal)

### Version 0.0.4 - 2020-07-04
- Test Release

### Version 0.0.3 - 2020-07-04
- Test Release

### Version 0.0.2 - 2020-06-24
- Add Pixel-based Region Connectivity Tool (#52)
- Fixes to Proximity Tools (#45, #37, #36)
- Changes to slope function to allow for change x, y coordinate fields (#46)
- Added Pharmacy Desert Example Notebook
- Add natural breaks classification method
- Add equal-interval classification method
- Add quantile classification method
- Add binary membership classification method
- Fixes to zonal stats docstring (#40)
- Added experimental `query layer` from agol-pandas (will probably not be supported long term)
- Added ReadtheDocs page
- Added experimental `hotspot` analysis tool (Getis-Ord Gi*) (#27)
- Added experimental `curvature` analysis tool (Getis-Ord Gi*)
- Added support for creating WMTS tilesets (moved out of datashader)
- Added contributor code of conduct

### Version 0.0.1 - 2020-02-15
- First public release available on GitHub and PyPI.
