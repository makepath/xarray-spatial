"""COG writer rejects unsupported inputs with typed, actionable errors (#2301).

Production-ready means predictable failure modes. The rows below cover the
input combinations the parent issue (#2286) flagged as ambiguous on the
``to_geotiff(..., cog=True)`` surface: experimental codecs without the
opt-in, internal-only JPEG without the opt-in, rotated transforms, file-like
destinations, object-dtype arrays, and conflicting CRS attrs. Each row
asserts the exception type AND a substring of the message that names the
violated constraint, so a message rewrite cannot silently turn an actionable
error into a vague one.

Most rows pin behaviour the writer already enforced. The rotated
``attrs['transform']`` Affine row is the one writer-side change in this PR:
a rasterio ``Affine`` iterates as a 9-element augmented matrix and used to
slip past the 6-tuple rotation gate in ``transform_from_attr``, silently
producing an axis-aligned GeoTIFF that dropped the rotation. ``to_geotiff``
now detects that shape via the ``Affine.b`` / ``Affine.d`` attrs and raises
the same diagnostic the 6-tuple branch already produced.

The CuPy + ``cog=True`` row is intentionally a no-op pin: the GPU writer
currently produces a valid COG and is already documented as Experimental
in the docstring tier map. Promoting that to a typed rejection is a tier
decision tracked under the parent issue, not a #2301 deliverable.
"""
from __future__ import annotations

import importlib.util
import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._errors import ConflictingCRSError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_da(shape=(8, 8)):
    """A small float32 DataArray suitable for COG writes."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=('y', 'x')
    )


def _uint8_da(shape=(8, 8)):
    """A small uint8 DataArray (JPEG is uint8-only)."""
    return xr.DataArray(
        np.zeros(shape, dtype=np.uint8), dims=('y', 'x')
    )


# ---------------------------------------------------------------------------
# Row 1: Experimental codec without ``allow_experimental_codecs=True``
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('codec', ['lerc', 'lz4', 'jpeg2000', 'j2k'])
def test_experimental_codec_without_opt_in_raises(tmp_path, codec):
    """Experimental codecs are gated; the message names the codec and
    the opt-in flag, and mentions the experimental tier so the caller
    knows why the default refuses the input."""
    da = _float_da()
    p = tmp_path / f'cog_exp_codec_{codec}_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, compression=codec)

    msg = str(exc.value)
    assert codec in msg, msg
    assert 'allow_experimental_codecs' in msg, msg
    assert 'experimental' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Row 2: Internal-only JPEG without ``allow_internal_only_jpeg=True``
# ---------------------------------------------------------------------------

def test_internal_only_jpeg_without_opt_in_raises(tmp_path):
    """``compression='jpeg'`` is rejected by default; the message names
    the codec, the opt-in flag, and explains the interop break."""
    da = _uint8_da()
    p = tmp_path / 'cog_jpeg_no_optin_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True, compression='jpeg')

    msg = str(exc.value)
    assert 'jpeg' in msg.lower(), msg
    assert 'allow_internal_only_jpeg' in msg, msg


def test_internal_only_jpeg_not_covered_by_experimental_flag(tmp_path):
    """``allow_experimental_codecs=True`` does not cover JPEG. The two
    flags are deliberately separate (internal-only is stricter than
    experimental) so a caller cannot reach the JFIF path by toggling
    only the experimental switch."""
    da = _uint8_da()
    p = tmp_path / 'cog_jpeg_exp_flag_only_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True,
                   compression='jpeg',
                   allow_experimental_codecs=True)

    msg = str(exc.value)
    assert 'jpeg' in msg.lower(), msg
    assert 'allow_internal_only_jpeg' in msg, msg


# ---------------------------------------------------------------------------
# Row 3: Rotated transform on input DataArray
# ---------------------------------------------------------------------------

def test_rotated_affine_attr_without_drop_rotation_raises(tmp_path):
    """The reader stamps ``attrs['rotated_affine']`` when called with
    ``allow_rotated=True``. Writing such a DataArray without
    ``drop_rotation=True`` would silently produce an identity-affine
    output (#2216), so the entry point refuses up front."""
    da = _float_da()
    da.attrs['rotated_affine'] = (1.0, 0.5, 0.0, 0.0, 0.5, 1.0)
    p = tmp_path / 'cog_rotated_affine_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotated_affine' in msg, msg
    assert 'drop_rotation' in msg, msg


def test_rotated_affine_attr_drop_rotation_opt_in_succeeds(tmp_path):
    """The opt-in path lets the write proceed (lossy but explicit).
    Pinned here so the rejection-message test cannot be 'fixed' by
    removing the opt-in entirely."""
    da = _float_da()
    da.attrs['rotated_affine'] = (1.0, 0.5, 0.0, 0.0, 0.5, 1.0)
    p = tmp_path / 'cog_rotated_affine_optin_2301.tif'

    to_geotiff(da, str(p), cog=True, drop_rotation=True)
    assert p.exists()
    assert p.stat().st_size > 0


def test_rotated_transform_tuple_attr_raises(tmp_path):
    """``attrs['transform']`` as a 6-tuple ``(a, b, c, d, e, f)`` with
    non-zero rotation/shear (``b`` or ``d``) is refused by
    ``transform_from_attr``. The message names the rotation/shear
    constraint and the axis-aligned requirement."""
    da = _float_da()
    da.attrs['transform'] = (1.0, 0.5, 0.0, 0.0, -1.0, 4.0)  # b = 0.5
    p = tmp_path / 'cog_rotated_tuple_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg
    assert 'axis-aligned' in msg, msg


def test_rotated_transform_affine_attr_raises(tmp_path):
    """``attrs['transform']`` as a rasterio ``Affine`` with non-zero
    rotation/shear used to slip past the 6-tuple gate because
    ``Affine`` iterates as a 9-element augmented matrix. The #2301
    validation hook detects the Affine duck-type and raises the same
    diagnostic the 6-tuple branch already produced."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da()
    da.attrs['transform'] = Affine(1.0, 0.5, 0.0, 0.0, -1.0, 4.0)  # b = 0.5
    p = tmp_path / 'cog_rotated_affine_obj_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg
    assert 'axis-aligned' in msg, msg


def test_skewed_transform_affine_attr_raises(tmp_path):
    """The ``d`` shear term (Affine's third row, first column) is also
    rejected. Same validator path as ``b != 0``; pinned separately so a
    refactor that only covers ``b`` is caught."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da()
    da.attrs['transform'] = Affine(1.0, 0.0, 0.0, 0.3, -1.0, 4.0)  # d = 0.3
    p = tmp_path / 'cog_skewed_affine_obj_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'rotation/shear' in msg, msg


def test_affine_attr_with_unconvertable_b_d_raises(tmp_path):
    """An attrs['transform'] object that quacks like an Affine (has
    ``.b`` and ``.d``) but carries non-numeric values for them is
    refused with a clear ``ValueError``. The fail-closed branch
    prevents a malformed input from bypassing the rotation/shear gate
    and falling through to the no-georef path."""
    class _BogusAffine:
        b = "not a number"
        d = 0.0
    da = _float_da()
    da.attrs['transform'] = _BogusAffine()
    p = tmp_path / 'cog_bogus_affine_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'unconvertable' in msg or 'rotation/shear' in msg, msg


def test_axis_aligned_affine_attr_still_writes(tmp_path):
    """Sanity guard: an axis-aligned Affine (b=d=0) must keep working.
    Without this row the #2301 hook could regress every legitimate
    Affine call site by widening the rejection bucket."""
    Affine = pytest.importorskip('affine').Affine
    da = _float_da()
    da.attrs['transform'] = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)  # b=d=0
    p = tmp_path / 'cog_axis_aligned_affine_2301.tif'

    to_geotiff(da, str(p), cog=True)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Row 4: File-like / BytesIO destination with ``cog=True``
# ---------------------------------------------------------------------------

def test_bytesio_destination_with_cog_raises():
    """COG output needs a real filesystem path because the writer runs
    a second pass to populate overview offsets. ``to_geotiff`` rejects
    file-like destinations with ``cog=True`` up front."""
    da = _float_da()
    buf = io.BytesIO()

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, buf, cog=True)

    msg = str(exc.value)
    assert 'cog' in msg.lower(), msg
    assert 'file-like' in msg or 'string path' in msg, msg


def test_bytesio_destination_without_cog_still_works():
    """Sanity guard: BytesIO is fine for plain TIFF writes. Pinned so
    the COG-only rejection cannot regress into a blanket file-like
    refusal."""
    da = _float_da()
    buf = io.BytesIO()

    to_geotiff(da, buf, cog=False)
    assert buf.tell() > 0


# ---------------------------------------------------------------------------
# Row 5: CuPy / GPU-backed array with ``cog=True``
# ---------------------------------------------------------------------------

def test_cupy_input_with_cog_currently_succeeds(tmp_path):
    """The GPU writer currently produces a valid COG for CuPy input;
    GPU COG is documented as Experimental in the docstring tier map
    but is not refused at the entry point. This row pins the
    currently-succeeds behaviour so a future tier-promotion change
    (tracked under #2286) does not silently break callers that
    already rely on the path.

    No production-side validation hook is added for #2301 because the
    constraint for this issue is 'do not change semantics on paths
    that currently succeed'."""
    if importlib.util.find_spec('cupy') is None:
        pytest.skip('cupy not installed')
    try:
        import cupy as cp
        if not cp.cuda.is_available():
            pytest.skip('CUDA device not available')
    except Exception as exc:
        pytest.skip(f'cupy import failed: {exc}')

    da = xr.DataArray(cp.zeros((8, 8), dtype=cp.float32), dims=('y', 'x'))
    p = tmp_path / 'cog_cupy_2301.tif'

    # No exception; produces a real file. If a future PR tightens the
    # GPU COG tier this assertion will start failing and the next
    # reviewer can decide whether to flip this to a ``pytest.raises``.
    to_geotiff(da, str(p), cog=True)
    assert p.exists()
    assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Row 6: Object-dtype DataArray
# ---------------------------------------------------------------------------

def test_object_dtype_with_cog_raises(tmp_path):
    """Object dtype is not a TIFF sample format. ``numpy_to_tiff_dtype``
    raises ``ValueError`` naming the dtype, so the writer surfaces a
    typed error rather than a deep struct-pack traceback."""
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=object), dims=('y', 'x'))
    p = tmp_path / 'cog_object_dtype_2301.tif'

    with pytest.raises(ValueError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    assert 'object' in msg.lower() or 'dtype' in msg.lower(), msg


# ---------------------------------------------------------------------------
# Row 7: Conflicting ``crs=`` kwarg / array CRS
# ---------------------------------------------------------------------------

def test_conflicting_attrs_crs_and_crs_wkt_raises(tmp_path):
    """When ``attrs['crs']`` and ``attrs['crs_wkt']`` resolve to
    different CRSes via pyproj, the writer refuses with
    ``ConflictingCRSError`` (#1987 PR 6). #2301 only confirms the
    message stays actionable; it does not introduce a new check."""
    pytest.importorskip('pyproj')
    wkt_3857 = (
        'PROJCS["WGS 84 / Pseudo-Mercator",'
        'GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Mercator_1SP"],'
        'PARAMETER["central_meridian",0],'
        'PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1],'
        'AUTHORITY["EPSG","3857"]]'
    )
    da = _float_da()
    da.attrs['crs'] = 4326
    da.attrs['crs_wkt'] = wkt_3857
    p = tmp_path / 'cog_conflicting_crs_2301.tif'

    with pytest.raises(ConflictingCRSError) as exc:
        to_geotiff(da, str(p), cog=True)

    msg = str(exc.value)
    # Message names both inputs and the resolution hint.
    assert "attrs['crs']" in msg, msg
    assert "attrs['crs_wkt']" in msg, msg
    # Caller-actionable: tells the user to reconcile the two attrs.
    assert 'Reconcile' in msg or 'reconcile' in msg, msg


def test_crs_kwarg_overrides_attrs_silently(tmp_path):
    """``crs=`` kwarg overrides the attrs disagreement. The
    ``_check_write_conflicting_crs`` short-circuit at the top of the
    check (``if context.get('crs_kwarg') is not None: return``) lets
    the write proceed even when the two attrs would otherwise
    disagree, so callers can intentionally use the kwarg to clobber
    stale attrs. Pinned here so a future 'stricter' rewrite of the
    conflict check that drops the short-circuit does not surprise
    those callers."""
    pytest.importorskip('pyproj')
    da = _float_da()
    da.attrs['crs'] = 4326
    # ``crs_wkt`` value is irrelevant: the check short-circuits on the
    # kwarg before pyproj parsing ever runs.
    da.attrs['crs_wkt'] = 'GEOGCS["foo"]'
    p = tmp_path / 'cog_crs_kwarg_override_2301.tif'

    to_geotiff(da, str(p), cog=True, crs=3857)
    assert p.exists()
    assert p.stat().st_size > 0
