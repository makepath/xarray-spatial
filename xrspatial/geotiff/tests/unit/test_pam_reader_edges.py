"""``read_pam_sidecar`` reader edge branches.

``open_geotiff`` reads a PAM ``.aux.xml`` for any local string source. The
reader recovers category names/colors from a thematic RAT and falls back to
the ``<CategoryNames>`` element, and it must decline tables that do not
describe categories rather than invent a name list. These cases drive the
``_parse_rat`` fall-throughs and the ``CategoryNames`` fallback that the
round-trip write tests (which always emit a fully-populated RAT) never reach.
"""
from xrspatial.geotiff import _pam


def _write(tmp_path, name, body):
    path = str(tmp_path / name)
    with open(_pam.sidecar_path(path), "w", encoding="utf-8") as fh:
        fh.write('<PAMDataset><PAMRasterBand band="1">' + body
                 + "</PAMRasterBand></PAMDataset>")
    return path


def test_rat_without_name_column_is_not_categories(tmp_path):
    """A thematic RAT carrying only a Value column describes no categories,
    so the reader returns ``{}`` instead of a bogus name list."""
    path = _write(
        tmp_path, "a.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<Row index="0"><F>0</F></Row>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {}


def test_empty_rat_falls_back_to_category_names(tmp_path):
    """A named RAT with zero rows yields no categories, so the reader falls
    back to the ``<CategoryNames>`` list."""
    path = _write(
        tmp_path, "b.tif",
        '<CategoryNames><Category>water</Category>'
        '<Category>land</Category></CategoryNames>'
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {"category_names": ["water", "land"]}


def test_category_names_only_no_rat(tmp_path):
    """With no RAT at all, the category names come from ``<CategoryNames>``."""
    path = _write(
        tmp_path, "c.tif",
        '<CategoryNames><Category>a</Category><Category>b</Category>'
        '<Category>c</Category></CategoryNames>')
    assert _pam.read_pam_sidecar(path) == {"category_names": ["a", "b", "c"]}


def test_field_defn_without_usage_is_skipped(tmp_path):
    """A ``<FieldDefn>`` missing its ``<Usage>`` element is skipped; the
    remaining Value/Class columns still resolve the category names."""
    path = _write(
        tmp_path, "d.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<FieldDefn index="2"><Name>Junk</Name><Type>2</Type></FieldDefn>'
        '<Row index="0"><F>0</F><F>sea</F><F>x</F></Row>'
        '<Row index="1"><F>1</F><F>ground</F><F>y</F></Row>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {"category_names": ["sea", "ground"]}


def test_sparse_one_based_rat_is_padded_to_pixel_values(tmp_path):
    """A thematic RAT with 1-based sparse values [1, 2, 5] must pad names so
    list index equals pixel value (issue #3591)."""
    path = _write(
        tmp_path, "sparse.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<Row index="0"><F>1</F><F>water</F></Row>'
        '<Row index="1"><F>2</F><F>forest</F></Row>'
        '<Row index="2"><F>5</F><F>urban</F></Row>'
        '</GDALRasterAttributeTable>')
    names = _pam.read_pam_sidecar(path)["category_names"]
    assert names == ["", "water", "forest", "", "", "urban"]


def test_sparse_rat_with_colors_is_padded(tmp_path):
    """A thematic RAT with sparse values and RGBA columns pads colors with
    (0,0,0,0) for gaps."""
    path = _write(
        tmp_path, "sparse_colors.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<FieldDefn index="2"><Name>Red</Name><Type>0</Type>'
        '<Usage>6</Usage></FieldDefn>'
        '<FieldDefn index="3"><Name>Green</Name><Type>0</Type>'
        '<Usage>7</Usage></FieldDefn>'
        '<FieldDefn index="4"><Name>Blue</Name><Type>0</Type>'
        '<Usage>8</Usage></FieldDefn>'
        '<FieldDefn index="5"><Name>Alpha</Name><Type>0</Type>'
        '<Usage>9</Usage></FieldDefn>'
        '<Row index="0"><F>2</F><F>forest</F><F>34</F><F>139</F>'
        '<F>34</F><F>255</F></Row>'
        '<Row index="1"><F>5</F><F>urban</F><F>128</F><F>128</F>'
        '<F>128</F><F>255</F></Row>'
        '</GDALRasterAttributeTable>')
    result = _pam.read_pam_sidecar(path)
    assert result["category_names"] == [
        "", "", "forest", "", "", "urban",
    ]
    assert result["category_colors"] == [
        (0, 0, 0, 0), (0, 0, 0, 0),
        (34, 139, 34, 255), (0, 0, 0, 0), (0, 0, 0, 0),
        (128, 128, 128, 255),
    ]


def test_negative_value_rat_returns_empty(tmp_path):
    """A RAT with a negative pixel value fails closed: the sidecar is ignored
    rather than producing misaligned categories."""
    path = _write(
        tmp_path, "neg.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<Row index="0"><F>-1</F><F>bogus</F></Row>'
        '<Row index="1"><F>0</F><F>real</F></Row>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {}


def test_huge_max_value_rat_returns_empty(tmp_path):
    """A RAT whose maximum pixel value exceeds _MAX_CATEGORIES fails closed
    rather than allocating an enormous list."""
    path = _write(
        tmp_path, "huge.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<Row index="0"><F>1000000</F><F>huge</F></Row>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {}


def test_no_value_column_uses_row_index_and_pads(tmp_path):
    """When the RAT omits the Value column, row index attributes serve as
    pixel values, and the result is still padded."""
    path = _write(
        tmp_path, "no_value_col.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<Row index="1"><F>water</F></Row>'
        '<Row index="3"><F>land</F></Row>'
        '</GDALRasterAttributeTable>')
    names = _pam.read_pam_sidecar(path)["category_names"]
    assert names == ["", "water", "", "land"]


def test_dense_zero_based_rat_still_works(tmp_path):
    """A standard dense 0-based RAT produces index-aligned names, same as
    before the sparse fix."""
    path = _write(
        tmp_path, "dense.tif",
        '<GDALRasterAttributeTable tableType="thematic">'
        '<FieldDefn index="0"><Name>Value</Name><Type>0</Type>'
        '<Usage>5</Usage></FieldDefn>'
        '<FieldDefn index="1"><Name>Class</Name><Type>2</Type>'
        '<Usage>2</Usage></FieldDefn>'
        '<Row index="0"><F>0</F><F>water</F></Row>'
        '<Row index="1"><F>1</F><F>land</F></Row>'
        '<Row index="2"><F>2</F><F>snow</F></Row>'
        '</GDALRasterAttributeTable>')
    assert _pam.read_pam_sidecar(path) == {
        "category_names": ["water", "land", "snow"],
    }
