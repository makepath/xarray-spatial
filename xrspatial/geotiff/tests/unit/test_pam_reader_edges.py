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
