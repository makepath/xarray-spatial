## Xarray-Spatial Changelog
-----------

### Version 0.2.0 - 4/28/2021
- Test release for new github actions

### Version 0.1.9 - 4/27/2021
- Deprecated tiles module (#381)
- Added user guide on the documentation website (#376)
- Updated docs design version mapping (#378)
- Added Github Action to publish package to PyPI (#371)
- Moved Spatialpandas to core install requirements for it to work on JLabs (#372)
- Added CONTRIBUTING.md (#374)
- Updated `true_color` to return a `xr.DataArray` (#364)
- Added get_data module and example sentinel-2 data (#358)
- Added citations guidelines and reformat (#382)

### Version 0.1.8 - 4/15/2021
- Fixed pypi related error

### Version 0.1.7 - 4/15/2021
- Updated multispectral.true_color: sigmoid contrast enhancement (#339)
- Added notebook save cogs in examples directory (#307)
- Updated Focal user guide (#336)
- Added documentation step on release steps (#346)
- Updated cloudless mosaic notebook: use Dask-Gateway (#351)
- Fixed user guide notebook numbering (#333)
- Correct warnings (#350)
- Add flake8 Github Action (#331)

### Version 0.1.6 - 4/12/2021
- Cleared metadata in all examples ipynb (#327)
- Moved docs requirements to source folder (#326)
- Fixed manifest file
- Fixed travis ci (#323)
- Included yml files
- Fixed examples path in Pharmacy Deserts Noteboo
- Integrate xarray-spatial website with the documentation (#291)

### Version 0.1.5 - 4/8/2021
- CLI examples bug fixed
- Added `drop_clouds`, cloud-free mosaic from sentinel2 data example (#255)

### Version 0.1.4 - 4/8/2021
- Sphinx doc fixes
- CLI bug fixed in 0.1.5

### Version 0.1.3 - 4/5/2021
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

### Version 0.1.2 - 12/1/2020
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

### Version 0.1.1 - 10/21/2020
- Added convolution module for use in focal statistics. (#131)
- Added example notebook for focal statistics and convolution modules.

### Version 0.1.0 - 9/10/2020
- Moved kernel creation to name-specific functions. (#127)
- Separated the validate and custom kernel functions. (focal)
- Added annulus focal kernel (#126) (focal)
- Added outputting of z-scores from hotspots tool (focal)
- Changed type checking to use np.floating (focal)
- Added tests for refactored focal statistics (focal)

### Version 0.0.9 - 8/26/2020
- Added A* pathfinding
- Allow all numpy float data types, not just numpy.float64 (#122)
- Broke out user-guide into individual notebooks  (examples)
- Added num_sample param option to natural_breaks (#123)
- Removed sklearn dependency

### Version 0.0.8 - 7/22/2020
- Fixed missing deps

### Version 0.0.7 - 7/21/2020
- Added 2D Crosstab (zonal)
- Added suggest_zonal_canvas (zonal)
- Added conda-forge build
- Removed Versioneer
- Updates to CI/CD

### Version 0.0.6 - 7/14/2020
- Added Proximity Direction (proximity)
- Added Proximity Allocation (proximity)
- Added Zonal Crop (zonal)
- Added Trim (zonal)
- Added ebbi (multispectral)
- Added more tests for slope (slope)
- Added image grid (readme)

### Version 0.0.5 - 7/5/2020
- Changed ndvi.py -> multispectral.py
- Added arvi (multispectral)
- Added gci (multispectral)
- Added savi (multispectral)
- Added evi (multispectral)
- Added nbr (multispectral)
- Added sipi (multispectral)
- Added `count` to default stats (zonal)
- Added regions tools (zonal)

### Version 0.0.4 - 7/4/2020
- Test Release

### Version 0.0.3 - 7/4/2020
- Test Release

### Version 0.0.2 - 6/24/2020
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

### Version 0.0.1 - 2/15/2020
- First public release available on GitHub and PyPI.
