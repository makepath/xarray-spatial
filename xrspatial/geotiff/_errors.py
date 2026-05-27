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
        ├── ConflictingNodataError     (PR 7, blocked on #1988)
        └── InconsistentGeoKeysError   (#2417)
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


class VRTUnsupportedError(GeoTIFFAmbiguousMetadataError):
    """A parsed VRT declares a feature the read pipeline does not honour (#2329).

    Raised by the centralised VRT validator at graph-build / eager-read
    setup time, before any source bytes are decoded. Covers CRS / dtype
    / band / nodata / transform / pixel-size / source-window /
    destination-window / resampling mismatches that the VRT read path
    cannot serve correctly. The message names the offending source path
    and field so a caller can locate the bad source without re-parsing
    the VRT XML themselves.

    Subclasses ``GeoTIFFAmbiguousMetadataError`` (and therefore
    ``ValueError``) so existing ``except ValueError`` callers keep
    catching VRT-capability failures alongside the older ambiguous-
    metadata family.
    """


class InconsistentGeoKeysError(GeoTIFFAmbiguousMetadataError):
    """GeoKey set is internally contradictory on read (#2417).

    Raised when a GeoTIFF source ships a GeoKey directory whose
    ``ModelTypeGeoKey`` does not agree with the type-specific keys
    actually populated, or whose ``ProjectedCSTypeGeoKey`` and
    ``GeographicTypeGeoKey`` resolve to different EPSG codes.

    The legacy reader took ``ProjectedCSTypeGeoKey`` first, then fell
    back to ``GeographicTypeGeoKey``, without ever cross-checking
    either against ``ModelTypeGeoKey``. A malformed or hostile input
    could declare ``ModelTypeGeoKey = geographic`` while stashing an
    EPSG under the projected key (or vice versa) and the reader would
    publish a trustworthy-looking ``attrs['crs']`` / ``attrs['crs_wkt']``
    built from inconsistent inputs. Silent acceptance of contradictory
    geospatial metadata is the failure mode the rest of the #1987
    series already rejects; this check extends the same contract to
    the GeoKey shape itself.

    Pass ``allow_inconsistent_geokeys=True`` on the public read entry
    points to keep the legacy permissive behaviour for files known to
    carry quirky-but-trusted GeoKey layouts.
    """


class InvalidIntegerNodataError(GeoTIFFAmbiguousMetadataError):
    """Integer-dtype source carries a non-finite or fractional GDAL_NODATA (#1774 follow-up).

    Raised when a GeoTIFF whose pixel buffer is an integer dtype declares
    a ``GDAL_NODATA`` value the integer buffer cannot represent: NaN, +Inf,
    -Inf, or a fractional float such as ``"3.5"`` on a ``uint16`` file.
    The original #1774 fix parsed the sentinel into ``attrs['nodata']``
    and silently skipped the masking step, so callers had no way to tell
    a silently-ignored sentinel from a missing one. The release contract
    (see ``test_release_gate_negative_integer_nodata_float_promoted``)
    upgrades that no-op to a typed rejection so the silent-coercion risk
    surfaces at the read boundary.

    Pass ``allow_invalid_nodata=True`` on the public read entry points to
    restore the pre-rejection no-op behaviour for files known to carry
    such sentinels.
    """


class VRTStableSourcesOnlyError(GeoTIFFAmbiguousMetadataError):
    """VRT source opened under ``stable_only=True`` (epic #2342).

    Raised when a caller opens a ``.vrt`` file with ``stable_only=True``.
    The VRT reader (``reader.vrt``) and its child sources sit at the
    ``advanced`` / ``experimental`` tiers in
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES`, so a request for
    stable-only sources cannot be served from a VRT mosaic without an
    explicit opt-in. The message names the offending VRT path and the
    matching opt-in flag (``allow_experimental_codecs``) so the caller
    learns the unlock at the boundary rather than from the docs.

    Pass ``stable_only=False`` (the default) to keep the legacy
    behaviour, or pass ``allow_experimental_codecs=True`` to opt into
    the broader tier set explicitly. See the release contract document
    at ``docs/source/reference/release_gate_geotiff.rst`` and epic
    #2342 for the full rationale.
    """


class UnknownCRSModelTypeError(GeoTIFFAmbiguousMetadataError):
    """Can't classify an EPSG as geographic or projected on write (#2277).

    Raised by the GeoTIFF writer when the caller supplies an EPSG code
    that pyproj cannot resolve, or when pyproj isn't installed and the
    code falls outside the hard-coded geographic fallback set
    (EPSG 4326 plus the 4000-4999 block). The legacy heuristic at the
    same site guessed any code outside that window was projected, which
    silently mis-tagged geographic codes like 6318 (NAD83(2011)),
    7844 (GDA2020), and 9057 (WGS 84 (G2139)) as
    ``ProjectedCSTypeGeoKey``. Silent CRS corruption is worse than an
    explicit error.
    """


class NonRepresentableEPSGCRSError(GeoTIFFAmbiguousMetadataError):
    """EPSG code resolves, but the GeoTIFF writer cannot represent it (#2418).

    Raised by the GeoTIFF writer when the caller supplies an integer
    EPSG code that pyproj resolves to a compound CRS (horizontal +
    vertical). The writer can only emit ``GeographicTypeGeoKey`` (2048)
    or ``ProjectedCSTypeGeoKey`` (3072); storing a compound EPSG in
    either slot produces a GeoTIFF that rasterio / GDAL reads back as
    a different CRS (or no CRS at all), because the GeoKey value
    declares "this is a 2D horizontal CRS" but the EPSG database
    registers it as compound.

    Callers who need to preserve a compound CRS should either pass the
    horizontal sub-CRS as the integer EPSG (the vertical component is
    not representable in a single GeoTIFF without additional GeoKeys
    that this release does not emit), or pass the full compound WKT as
    ``crs=`` so the writer takes the ``ProjectedCSType=32767`` /
    ``GTCitationGeoKey`` fallback path. The WKT fallback path round-trips
    through libgeotiff and GDAL but not all readers; see issue #1768.
    """


class DuplicateIFDTagError(ValueError):
    """An IFD declares the same tag id more than once (#2483).

    TIFF 6.0 section 2 ("The TIFF Structure") requires IFD entries to be
    sorted in ascending order by tag id with no duplicates. The legacy
    parser stored entries in a dict keyed by tag and let the last write
    win, so a file with two ``ImageWidth`` (or any other) entries parsed
    silently to whichever value happened to come second. That is the
    same silent-acceptance failure mode the rest of the
    ``GeoTIFFAmbiguousMetadataError`` family rejects: a malformed or
    adversarial file changes interpretation at the read boundary with no
    signal to the caller.

    The exception names the duplicated tag id and the byte offsets of
    the two conflicting entries (the prior entry and the current one)
    so a caller can locate the bad bytes without re-parsing the IFD.
    The same check fires in the dimension pre-scan used to bound
    pixel-array tag counts, so the early-exit path cannot silently
    accept a malformed ``ImageWidth`` / ``ImageLength`` either.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers
    keep catching the case; new code can ``except`` this class directly
    to distinguish duplicate-tag rejection from other parse failures.

    Raised at the public read entry points
    (:func:`xrspatial.geotiff.open_geotiff` and the chunked / GPU /
    VRT readers that share the same IFD parser), so a caller does not
    need to invoke ``parse_ifd`` directly to see the failure.
    """


class UnsupportedGeoTIFFFeatureError(ValueError):
    """Caller asked for a feature this release does not implement (#2349).

    Raised at the read or write entry point when the input declares a
    feature the GeoTIFF module does not support (warped / reprojection
    VRTs, pansharpened / processed / derived VRT subclasses, unknown
    VRT band children, source transforms with non-zero skew on a VRT
    mosaic). The message names the feature and points the caller at
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier map.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers
    keep catching the case; new code can ``except`` this class to
    distinguish "we refuse this input" from "the input is malformed".
    """


__all__ = [
    "ConflictingCRSError",
    "ConflictingNodataError",
    "DuplicateIFDTagError",
    "GeoTIFFAmbiguousMetadataError",
    "InconsistentGeoKeysError",
    "InvalidCRSCodeError",
    "InvalidIntegerNodataError",
    "MixedBandMetadataError",
    "NonRepresentableEPSGCRSError",
    "NonUniformCoordsError",
    "RotatedTransformError",
    "UnknownCRSModelTypeError",
    "UnparseableCRSError",
    "UnsupportedGeoTIFFFeatureError",
    "VRTStableSourcesOnlyError",
    "VRTUnsupportedError",
]
