## Xarray-Spatial Changelog
-----------

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
