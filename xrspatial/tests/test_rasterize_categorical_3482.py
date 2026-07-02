"""Tests for categorical/string column support in rasterize (issue #3482).

A string or categorical ``column`` is label-encoded to integer codes on an
int32 band with a -1 nodata sentinel, and the value->label map is carried in
``attrs['category_names']`` / ``attrs['category_colors']``. ``to_geotiff``
emits a PAM ``.aux.xml`` sidecar so GDAL/QGIS show the class names, and
``open_geotiff`` reads it back for a full round-trip.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_geopandas, reason="geopandas not installed"),
]


@pytest.fixture
def landcover_gdf():
    """Three non-overlapping squares keyed by a string land-cover column."""
    return gpd.GeoDataFrame(
        {'landcover': ['water', 'forest', 'urban']},
        geometry=[box(0, 0, 5, 5), box(5, 0, 10, 5), box(0, 5, 5, 10)],
        crs='EPSG:4326',
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestCategoricalEncoding:
    def test_string_column_encodes_to_int32(self, landcover_gdf):
        result = rasterize(landcover_gdf, column='landcover',
                           width=10, height=10)
        assert result.dtype == np.int32
        # Codes are 0..N-1 plus the -1 nodata for untouched cells.
        assert set(np.unique(result.values)) <= {-1, 0, 1, 2}
        assert result.attrs['nodata'] == -1

    def test_category_names_sorted_for_plain_strings(self, landcover_gdf):
        result = rasterize(landcover_gdf, column='landcover',
                           width=10, height=10)
        # astype('category') sorts object categories lexically.
        assert result.attrs['category_names'] == ['forest', 'urban', 'water']

    def test_codes_match_category_order(self, landcover_gdf):
        result = rasterize(landcover_gdf, column='landcover',
                           width=10, height=10)
        names = result.attrs['category_names']
        # 'forest' geometry is box(5,0,10,5): right half, bottom rows.
        forest_code = names.index('forest')
        vals = result.values
        # Bottom-right quadrant should be all the forest code.
        assert np.all(vals[5:, 5:] == forest_code)

    @pytest.mark.skipif(not has_pandas, reason="pandas not installed")
    def test_ordered_categorical_preserves_order(self):
        cat = pd.Categorical(['b', 'a', 'c'],
                             categories=['c', 'b', 'a'], ordered=True)
        gdf = gpd.GeoDataFrame(
            {'k': cat},
            geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2), box(4, 0, 6, 2)],
        )
        result = rasterize(gdf, column='k', width=6, height=2)
        assert result.attrs['category_names'] == ['c', 'b', 'a']

    @pytest.mark.skipif(not has_pandas, reason="pandas not installed")
    def test_missing_category_becomes_nodata(self):
        gdf = gpd.GeoDataFrame(
            {'k': ['x', None]},
            geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2)],
        )
        result = rasterize(gdf, column='k', width=4, height=2)
        assert result.attrs['category_names'] == ['x']
        # The None geometry must not paint a real code; only -1 and 0 appear.
        assert set(np.unique(result.values)) <= {-1, 0}

    def test_explicit_fill_remaps_missing(self):
        gdf = gpd.GeoDataFrame(
            {'k': ['x', None]},
            geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2)],
        )
        result = rasterize(gdf, column='k', width=4, height=2,
                           fill=99, dtype=np.int32)
        assert result.attrs['nodata'] == 99
        assert -1 not in np.unique(result.values)

    def test_colors_one_per_category(self, landcover_gdf):
        result = rasterize(landcover_gdf, column='landcover',
                           width=10, height=10)
        colors = result.attrs['category_colors']
        assert len(colors) == len(result.attrs['category_names'])
        for r, g, b, a in colors:
            assert all(0 <= c <= 255 for c in (r, g, b, a))
            assert a == 255

    def test_numeric_column_unchanged(self):
        gdf = gpd.GeoDataFrame({'v': [5.0]}, geometry=[box(0, 0, 4, 4)])
        result = rasterize(gdf, column='v', width=4, height=4)
        assert result.dtype == np.float64
        assert 'category_names' not in result.attrs

    def test_explicit_dtype_respected(self, landcover_gdf):
        result = rasterize(landcover_gdf, column='landcover',
                           width=10, height=10, fill=0, dtype=np.uint8)
        assert result.dtype == np.uint8
        assert result.attrs['category_names'] == ['forest', 'urban', 'water']


# ---------------------------------------------------------------------------
# GeoTIFF round-trip
# ---------------------------------------------------------------------------

class TestGeoTIFFRoundTrip:
    def test_sidecar_written_and_read_back(self, landcover_gdf, tmp_path):
        pytest.importorskip('tifffile')
        from xrspatial.geotiff import to_geotiff, open_geotiff

        result = rasterize(landcover_gdf, column='landcover',
                           width=20, height=20)
        path = str(tmp_path / 'landcover_3482.tif')
        to_geotiff(result, path)

        assert os.path.exists(path + '.aux.xml')

        back = open_geotiff(path)
        assert back.attrs['category_names'] == ['forest', 'urban', 'water']
        assert (list(back.attrs['category_colors'])
                == list(result.attrs['category_colors']))

    def test_no_sidecar_for_numeric(self, tmp_path):
        pytest.importorskip('tifffile')
        from xrspatial.geotiff import to_geotiff

        gdf = gpd.GeoDataFrame({'v': [5.0]}, geometry=[box(0, 0, 4, 4)],
                               crs='EPSG:4326')
        result = rasterize(gdf, column='v', width=8, height=8)
        path = str(tmp_path / 'numeric_3482.tif')
        to_geotiff(result, path)
        assert not os.path.exists(path + '.aux.xml')

    @pytest.mark.skipif(shutil.which('gdalinfo') is None,
                        reason="gdalinfo not available")
    def test_gdalinfo_shows_categories(self, landcover_gdf, tmp_path):
        """GDAL (and therefore QGIS) reads the labels from the sidecar."""
        pytest.importorskip('tifffile')
        from xrspatial.geotiff import to_geotiff

        result = rasterize(landcover_gdf, column='landcover',
                           width=20, height=20)
        path = str(tmp_path / 'gdalinfo_3482.tif')
        to_geotiff(result, path)

        out = subprocess.run(['gdalinfo', path], capture_output=True,
                             text=True, check=True).stdout
        assert 'Categories:' in out
        for name in ('forest', 'urban', 'water'):
            assert name in out


# ---------------------------------------------------------------------------
# PAM XML helpers
# ---------------------------------------------------------------------------

class TestPamHelpers:
    def test_build_and_parse_round_trip(self):
        from xrspatial.geotiff._pam import build_pam_xml, _parse_rat
        from xrspatial.geotiff._safe_xml import safe_fromstring

        names = ['a', 'b', 'c']
        colors = [(10, 20, 30, 255), (40, 50, 60, 255), (70, 80, 90, 255)]
        xml = build_pam_xml(names, colors)
        root = safe_fromstring(xml)
        rat = root.find('.//GDALRasterAttributeTable')
        parsed_names, parsed_colors = _parse_rat(rat)
        assert parsed_names == names
        assert parsed_colors == colors

    def test_special_characters_escaped(self):
        from xrspatial.geotiff._pam import build_pam_xml
        from xrspatial.geotiff._safe_xml import safe_fromstring

        # Ampersand / angle brackets must survive XML serialization.
        xml = build_pam_xml(['a & b', 'c <d>'])
        root = safe_fromstring(xml)  # must not raise
        cats = [c.text for c in root.findall('.//Category')]
        assert cats == ['a & b', 'c <d>']

    def test_read_missing_sidecar_returns_empty(self, tmp_path):
        from xrspatial.geotiff._pam import read_pam_sidecar
        assert read_pam_sidecar(str(tmp_path / 'nope.tif')) == {}

    def test_athematic_stats_sidecar_ignored(self, tmp_path):
        """A GDAL statistics/histogram sidecar must not yield categories."""
        from xrspatial.geotiff._pam import read_pam_sidecar
        path = str(tmp_path / 'stats.tif')
        with open(path + '.aux.xml', 'w') as fh:
            fh.write(
                '<PAMDataset><PAMRasterBand band="1">'
                '<Metadata><MDI key="STATISTICS_MINIMUM">0</MDI></Metadata>'
                '<GDALRasterAttributeTable tableType="athematic">'
                '<FieldDefn index="0"><Name>Value</Name><Type>1</Type>'
                '<Usage>5</Usage></FieldDefn>'
                '<FieldDefn index="1"><Name>Count</Name><Type>0</Type>'
                '<Usage>1</Usage></FieldDefn>'
                '<Row index="0"><F>0.0</F><F>5</F></Row>'
                '<Row index="1"><F>1.0</F><F>4</F></Row>'
                '</GDALRasterAttributeTable>'
                '</PAMRasterBand></PAMDataset>')
        assert read_pam_sidecar(path) == {}

    def test_malformed_sidecar_returns_empty(self, tmp_path):
        from xrspatial.geotiff._pam import read_pam_sidecar
        path = str(tmp_path / 'bad.tif')
        with open(path + '.aux.xml', 'w') as fh:
            fh.write('<PAMDataset><PAMRasterBand band="1">'
                     '<GDALRasterAttributeTable tableType="thematic">'
                     '<FieldDefn index="0"><Name>Class</Name><Type>2</Type>'
                     '<Usage>2</Usage></FieldDefn>'
                     '<Row index="0"><F>ok</F></Row>'
                     '<Row index="0"><F>bad</F><F>extra</F></Row>'
                     'GARBAGE</GDALRasterAttributeTable>'
                     '</PAMRasterBand></PAMDataset>')
        # Must never raise; worst case returns {}.
        assert isinstance(read_pam_sidecar(path), dict)

    def test_short_row_thematic_rat_returns_empty(self, tmp_path):
        """A thematic RAT row with fewer <F> cells than the Name column
        index must not crash read_pam_sidecar.

        The Name column sits at index 1 (after a Value column at index 0),
        but the row carries a single cell, so indexing the name cell raises
        IndexError inside _parse_rat. read_pam_sidecar must swallow it and
        fall back to {} like every other malformed-sidecar case, since
        open_geotiff reads this sidecar for any local string source.
        """
        from xrspatial.geotiff._pam import read_pam_sidecar
        path = str(tmp_path / 'short_row.tif')
        with open(path + '.aux.xml', 'w') as fh:
            fh.write('<PAMDataset><PAMRasterBand band="1">'
                     '<GDALRasterAttributeTable tableType="thematic">'
                     '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
                     '<Usage>5</Usage></FieldDefn>'
                     '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
                     '<Usage>2</Usage></FieldDefn>'
                     '<Row index="0"><F>0</F></Row>'
                     '</GDALRasterAttributeTable>'
                     '</PAMRasterBand></PAMDataset>')
        # Must never raise; worst case returns {}.
        assert read_pam_sidecar(path) == {}

    def test_non_finite_rat_value_returns_empty(self, tmp_path):
        """A RAT Value cell that parses to infinity must not crash the read.

        _parse_rat converts the Value cell with int(float(text)); a cell
        such as "1e400" or "inf" makes int() raise OverflowError, which
        subclasses ArithmeticError rather than ValueError and so escaped
        the read_pam_sidecar except tuple, crashing the open_geotiff call
        that reads the sidecar for any local string source (issue #3590).
        """
        from xrspatial.geotiff._pam import read_pam_sidecar
        path = str(tmp_path / 'inf_value_3590.tif')
        with open(path + '.aux.xml', 'w') as fh:
            fh.write('<PAMDataset><PAMRasterBand band="1">'
                     '<GDALRasterAttributeTable tableType="thematic">'
                     '<FieldDefn index="0"><Name>Value</Name><Type>1</Type>'
                     '<Usage>5</Usage></FieldDefn>'
                     '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
                     '<Usage>2</Usage></FieldDefn>'
                     '<Row index="0"><F>1e400</F><F>water</F></Row>'
                     '</GDALRasterAttributeTable>'
                     '</PAMRasterBand></PAMDataset>')
        # Must never raise; worst case returns {}.
        assert read_pam_sidecar(path) == {}

    def test_non_well_formed_xml_sidecar_returns_empty(self, tmp_path):
        """A truncated / non-well-formed .aux.xml must not crash the read.

        safe_fromstring raises xml.etree.ElementTree.ParseError, which is a
        SyntaxError subclass (not an OSError/ValueError/TypeError/IndexError),
        so it would otherwise escape read_pam_sidecar and break the
        open_geotiff call that reads the sidecar for any local string source.
        """
        from xrspatial.geotiff._pam import read_pam_sidecar
        path = str(tmp_path / 'truncated.tif')
        with open(path + '.aux.xml', 'w') as fh:
            fh.write('<PAMDataset><PAMRasterBand band="1"><Category>oops')
        # Must never raise; worst case returns {}.
        assert read_pam_sidecar(path) == {}


# ---------------------------------------------------------------------------
# GPU backend parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
class TestCuPyParity:
    def test_gpu_codes_match_cpu(self, landcover_gdf):
        cpu = rasterize(landcover_gdf, column='landcover',
                        width=16, height=16)
        gpu = rasterize(landcover_gdf, column='landcover',
                        width=16, height=16, gpu=True)
        assert gpu.attrs['category_names'] == cpu.attrs['category_names']
        np.testing.assert_array_equal(gpu.data.get(), cpu.values)
