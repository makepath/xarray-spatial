"""PAM (``.aux.xml``) sidecar helpers for categorical rasters.

GDAL stores a band's category names and Raster Attribute Table (RAT) for a
GeoTIFF in a ``<file>.aux.xml`` PAM sidecar, not in the TIFF itself: an
embedded RAT in the GDAL_METADATA tag is silently ignored on read. QGIS
reads the sidecar to label and color discrete classes, so writing one is
what makes a categorical ``rasterize`` result show class names instead of
bare integers.

This module builds that sidecar from ``attrs['category_names']`` /
``attrs['category_colors']`` and parses it back. The XML shape matches what
``gdalinfo`` round-trips: a single ``<PAMRasterBand band="1">`` carrying a
``<CategoryNames>`` list plus a thematic ``<GDALRasterAttributeTable>``.
"""
from __future__ import annotations

import os
from xml.etree.ElementTree import ParseError
from xml.sax.saxutils import escape

from ._safe_xml import safe_fromstring

# GDAL RAT field-usage codes (GDALRATFieldUsage).
_USAGE_MINMAX = 5
_USAGE_NAME = 2
_USAGE_RED = 6
_USAGE_GREEN = 7
_USAGE_BLUE = 8
_USAGE_ALPHA = 9
# GDAL RAT field types (GDALRATFieldType): Integer=0, Real=1, String=2.
_TYPE_INT = 0
_TYPE_STRING = 2


def sidecar_path(path: str) -> str:
    """Return the PAM sidecar path GDAL expects for *path* (``<path>.aux.xml``)."""
    return path + '.aux.xml'


def build_pam_xml(category_names, category_colors=None):
    """Build a PAM ``.aux.xml`` document for a categorical band.

    Parameters
    ----------
    category_names : sequence of str
        Class labels; list index is the pixel value.
    category_colors : sequence of (r, g, b, a), optional
        One RGBA tuple (components 0-255) per category. When given, the RAT
        gains Red/Green/Blue/Alpha columns so QGIS colors each class.

    Returns
    -------
    str
        The ``<PAMDataset>`` XML document.
    """
    names = [str(n) for n in category_names]
    have_colors = category_colors is not None and len(category_colors) == len(names)

    field_defs = [
        ('Value', _TYPE_INT, _USAGE_MINMAX),
        ('Class', _TYPE_STRING, _USAGE_NAME),
    ]
    if have_colors:
        field_defs += [
            ('Red', _TYPE_INT, _USAGE_RED),
            ('Green', _TYPE_INT, _USAGE_GREEN),
            ('Blue', _TYPE_INT, _USAGE_BLUE),
            ('Alpha', _TYPE_INT, _USAGE_ALPHA),
        ]

    lines = ['<PAMDataset>', '  <PAMRasterBand band="1">']

    lines.append('    <CategoryNames>')
    for name in names:
        lines.append(f'      <Category>{escape(name)}</Category>')
    lines.append('    </CategoryNames>')

    lines.append('    <GDALRasterAttributeTable tableType="thematic">')
    for i, (fname, ftype, usage) in enumerate(field_defs):
        lines.append(f'      <FieldDefn index="{i}">')
        lines.append(f'        <Name>{fname}</Name>')
        lines.append(f'        <Type>{ftype}</Type>')
        lines.append(f'        <Usage>{usage}</Usage>')
        lines.append('      </FieldDefn>')
    for value, name in enumerate(names):
        cells = [str(value), escape(name)]
        if have_colors:
            r, g, b, a = category_colors[value]
            cells += [str(int(r)), str(int(g)), str(int(b)), str(int(a))]
        row = ''.join(f'<F>{c}</F>' for c in cells)
        lines.append(f'      <Row index="{value}">{row}</Row>')
    lines.append('    </GDALRasterAttributeTable>')

    lines.append('  </PAMRasterBand>')
    lines.append('</PAMDataset>')
    return '\n'.join(lines) + '\n'


def write_pam_sidecar(path, category_names, category_colors=None):
    """Write the PAM sidecar for *path* and return the sidecar path."""
    xml = build_pam_xml(category_names, category_colors)
    out = sidecar_path(path)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(xml)
    return out


def build_stats_pam_xml(stats):
    """Build a PAM ``.aux.xml`` document carrying band statistics.

    *stats* maps GDAL ``STATISTICS_*`` keys to numeric values. GDAL tools and
    QGIS read these from the band ``<Metadata>`` to drive a default min/max
    stretch on a continuous raster, the way :func:`build_pam_xml` drives class
    coloring on a categorical one.
    """
    lines = ['<PAMDataset>', '  <PAMRasterBand band="1">', '    <Metadata>']
    for key, value in stats.items():
        text = escape('{:.10g}'.format(float(value)))
        lines.append(f'      <MDI key="{escape(str(key))}">{text}</MDI>')
    lines.append('    </Metadata>')
    lines.append('  </PAMRasterBand>')
    lines.append('</PAMDataset>')
    return '\n'.join(lines) + '\n'


def write_stats_pam_sidecar(path, stats):
    """Write the statistics PAM sidecar for *path* and return its path."""
    xml = build_stats_pam_xml(stats)
    out = sidecar_path(path)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(xml)
    return out


def read_pam_sidecar(path):
    """Read ``category_names`` / ``category_colors`` from *path*'s sidecar.

    Returns a dict with whatever it could recover (``category_names`` and,
    when the RAT carries color columns, ``category_colors``). Returns an
    empty dict when no sidecar exists or it cannot be parsed -- a missing or
    malformed sidecar is non-fatal auxiliary metadata, not a read error.
    """
    aux = sidecar_path(path)
    if not os.path.exists(aux):
        return {}
    try:
        with open(aux, 'r', encoding='utf-8') as fh:
            root = safe_fromstring(fh.read())

        band = root.find('.//PAMRasterBand')
        if band is None:
            return {}

        names = None
        colors = None
        # Only a thematic RAT with a Name column describes categories. GDAL
        # writes an athematic histogram/statistics RAT next to many ordinary
        # rasters; it must not masquerade as category names. Prefer the RAT
        # (it carries colors); fall back to the <CategoryNames> element,
        # which GDAL only writes for real categories.
        rat = band.find('GDALRasterAttributeTable')
        if rat is not None and rat.get('tableType') == 'thematic':
            names, colors = _parse_rat(rat)
        if names is None:
            cat_el = band.find('CategoryNames')
            if cat_el is not None:
                names = [c.text or '' for c in cat_el.findall('Category')]

        out = {}
        if names is not None:
            out['category_names'] = names
        if colors is not None:
            out['category_colors'] = colors
        return out
    except (OSError, ValueError, TypeError, IndexError, ParseError):
        # A missing, malformed, or foreign sidecar is non-fatal auxiliary
        # metadata, not a read error -- never let it break open_geotiff.
        # IndexError covers a thematic RAT whose <Row> carries fewer <F>
        # cells than its highest column index (e.g. a Name column at index
        # 1 paired with a single-cell row), which would otherwise escape
        # _parse_rat and crash the open_geotiff call that reads the
        # adjacent sidecar. ParseError covers a truncated or otherwise
        # non-well-formed sidecar: safe_fromstring raises it (a SyntaxError
        # subclass, so not covered by the types above) and it would
        # likewise escape and crash the read.
        return {}


def _parse_rat(rat):
    """Return (names, colors) from a thematic ``<GDALRasterAttributeTable>``.

    Returns ``(None, None)`` when the table has no Name column, i.e. it does
    not describe categories. ``names`` is ordered by the row's Value column;
    ``colors`` is the RGBA list when the RAT defines color columns, else
    ``None``.
    """
    usage_to_col = {}
    for fd in rat.findall('FieldDefn'):
        usage_el = fd.find('Usage')
        if fd.get('index') is None or usage_el is None \
                or usage_el.text is None:
            continue
        usage_to_col[int(usage_el.text)] = int(fd.get('index'))

    name_col = usage_to_col.get(_USAGE_NAME)
    if name_col is None:
        return None, None
    value_col = usage_to_col.get(_USAGE_MINMAX)
    rgba_cols = [usage_to_col.get(u) for u in
                 (_USAGE_RED, _USAGE_GREEN, _USAGE_BLUE, _USAGE_ALPHA)]
    have_colors = all(c is not None for c in rgba_cols)

    rows = []
    for row in rat.findall('Row'):
        fields = [f.text or '' for f in row.findall('F')]
        # Tolerate a Real-typed value cell ("0.0") as well as an integer.
        value = int(float(fields[value_col])) if value_col is not None \
            else int(row.get('index'))
        name = fields[name_col]
        color = None
        if have_colors:
            color = tuple(int(fields[c]) for c in rgba_cols)
        rows.append((value, name, color))

    if not rows:
        return None, None

    rows.sort(key=lambda r: r[0])
    names = [name for _, name, _ in rows]
    colors = [color for _, _, color in rows] if have_colors else None
    return names, colors
