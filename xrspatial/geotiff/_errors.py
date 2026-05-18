"""Typed errors for ambiguous GeoTIFF metadata (issue #1987).

The reader and writer used to "guess and continue" when geospatial
metadata was ambiguous: invalid CRS codes, unparseable CRS strings,
rotated transforms, non-uniform coords, mixed band metadata, conflicting
``crs`` vs ``crs_wkt`` attrs, conflicting nodata aliases. Each case
becomes a hard error by default with a per-case typed subclass so
callers can ``except`` the family or a specific case.

This module provides the error class hierarchy only. The validator
hooks in ``_validation.py`` decide when each one fires; the per-case
PRs (issue #1987 PRs 1-7) wire them up at the read/write entry points.

Hierarchy::

    Exception
    └── GeoTIFFAmbiguousMetadataError
        ├── InvalidCRSCodeError        (PR 1 / #1971)
        ├── UnparseableCRSError        (PR 2)
        ├── RotatedTransformError      (PR 3)
        ├── NonUniformCoordsError      (PR 4)
        ├── MixedBandMetadataError     (PR 5)
        ├── ConflictingCRSError        (PR 6, blocked on #1984)
        └── ConflictingNodataError     (PR 7, blocked on #1988)
"""
from __future__ import annotations


class GeoTIFFAmbiguousMetadataError(ValueError):
    """Base class for ambiguous GeoTIFF metadata failures (#1987).

    Subclasses ``ValueError`` so existing ``except ValueError`` callers
    keep catching these. Catch this class directly to handle the whole
    family, or one of the per-case subclasses to handle a single
    ambiguity type.
    """


class InvalidCRSCodeError(GeoTIFFAmbiguousMetadataError):
    """Invalid EPSG / authority code on read or write (#1971, PR 1).

    Raised when a CRS code does not resolve to a known authority entry
    (e.g. ``to_geotiff(crs=True)`` formerly wrote ``EPSG=1`` silently).
    """


class UnparseableCRSError(GeoTIFFAmbiguousMetadataError):
    """CRS string cannot be parsed as WKT or recognised authority code (PR 2).

    Partial WKT or malformed input that the legacy path would have
    emitted unchanged, producing mismatched ``crs`` vs ``crs_wkt``
    attrs downstream.
    """


class RotatedTransformError(GeoTIFFAmbiguousMetadataError):
    """Affine transform has non-zero rotation/shear terms (PR 3).

    Downstream xrspatial functions assume axis-aligned rasters and
    would otherwise produce wrong results on a rotated grid. The
    read entry points raise this by default; pass ``allow_rotated=True``
    to retain the existing attr-flag behaviour and read the pixel
    grid without the geospatial assumption.
    """


class NonUniformCoordsError(GeoTIFFAmbiguousMetadataError):
    """DataArray coords disagree with the implied transform on write (PR 4).

    ``to_geotiff`` accepts coords that imply a non-uniform pixel grid
    (variable cell size, gaps); the writer would otherwise pick the
    first two coord values as the transform and silently truncate the
    rest. The existing sentinel exemption for int-dtype coords stays
    (#1969).
    """


class MixedBandMetadataError(GeoTIFFAmbiguousMetadataError):
    """VRT bands declare conflicting per-band metadata (PR 5).

    Most often disagreeing nodata sentinels across bands. The legacy
    read path flattened to one value silently. Pass
    ``band_nodata='first'`` to keep the legacy behaviour explicitly.
    """


class ConflictingCRSError(GeoTIFFAmbiguousMetadataError):
    """``attrs['crs']`` and ``attrs['crs_wkt']`` disagree on write (PR 6).

    Both keys set to CRS strings that do not canonicalise to the same
    WKT (after EPSG → WKT lookup). The writer would otherwise pick one
    and emit it, silently dropping the other.
    """


class ConflictingNodataError(GeoTIFFAmbiguousMetadataError):
    """Nodata sentinel aliases disagree on write (PR 7).

    ``attrs['nodata']`` and ``attrs['nodatavals']`` set to different
    values. ``_resolve_nodata_attr`` formerly picked one and ignored
    the other. ``_FillValue`` is a CF alias and remains deprioritised
    per the existing convention.
    """


__all__ = [
    "GeoTIFFAmbiguousMetadataError",
    "InvalidCRSCodeError",
    "UnparseableCRSError",
    "RotatedTransformError",
    "NonUniformCoordsError",
    "MixedBandMetadataError",
    "ConflictingCRSError",
    "ConflictingNodataError",
]
