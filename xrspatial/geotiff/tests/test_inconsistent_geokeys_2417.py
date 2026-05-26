"""Fail-closed read check for internally contradictory GeoKeys (issue #2417).

The legacy reader took ``ProjectedCSTypeGeoKey`` first, fell back to
``GeographicTypeGeoKey``, and never cross-checked either against
``ModelTypeGeoKey``. A malformed file declaring
``ModelTypeGeoKey = geographic (2)`` with an EPSG stashed under
``ProjectedCSTypeGeoKey`` would publish a trustworthy-looking
``attrs['crs']`` / ``attrs['crs_wkt']`` fabricated from contradictory
inputs. ``_check_read_inconsistent_geokeys`` rejects three structural
combinations:

1. ``ModelType=projected`` with only ``GeographicTypeGeoKey`` set.
2. ``ModelType=geographic`` with ``ProjectedCSTypeGeoKey`` set.
3. Both type keys populated with different non-user-defined EPSG codes.

Pass ``allow_inconsistent_geokeys=True`` on the public read entry
points to keep the legacy permissive behaviour.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from xrspatial.geotiff import (GeoTIFFAmbiguousMetadataError, InconsistentGeoKeysError,
                               open_geotiff)
from xrspatial.geotiff._validation import (_check_read_inconsistent_geokeys,
                                           _registered_read_metadata_checks,
                                           validate_read_metadata)


# ---------------------------------------------------------------------------
# Registry sanity.
# ---------------------------------------------------------------------------


def test_inconsistent_geokeys_check_registered():
    names = {c.__name__ for c in _registered_read_metadata_checks()}
    assert _check_read_inconsistent_geokeys.__name__ in names


def test_error_class_hierarchy():
    assert issubclass(InconsistentGeoKeysError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(InconsistentGeoKeysError, ValueError)


# ---------------------------------------------------------------------------
# Unit-level checks driven through ``validate_read_metadata``.
# ---------------------------------------------------------------------------


def test_validate_rejects_model_geographic_with_projected_key():
    """The exact scenario from issue #2417: ModelType=geographic but
    ProjectedCSTypeGeoKey=4326 set, GeographicTypeGeoKey absent."""
    with pytest.raises(InconsistentGeoKeysError) as excinfo:
        validate_read_metadata({
            'model_type': 2,
            'projected_cs_type': 4326,
            'geographic_type': None,
        })
    msg = str(excinfo.value)
    assert 'ModelTypeGeoKey=geographic' in msg
    assert '4326' in msg
    assert '#2417' in msg


def test_validate_rejects_model_projected_with_only_geographic_key():
    """ModelType=projected but only GeographicTypeGeoKey is populated."""
    with pytest.raises(InconsistentGeoKeysError) as excinfo:
        validate_read_metadata({
            'model_type': 1,
            'projected_cs_type': None,
            'geographic_type': 4326,
        })
    msg = str(excinfo.value)
    assert 'ModelTypeGeoKey=projected' in msg
    assert '4326' in msg


def test_validate_rejects_both_type_keys_with_different_epsg():
    """Both ProjectedCSTypeGeoKey and GeographicTypeGeoKey set to
    different EPSG codes."""
    with pytest.raises(InconsistentGeoKeysError) as excinfo:
        validate_read_metadata({
            'model_type': 1,
            'projected_cs_type': 32633,
            'geographic_type': 4326,
        })
    msg = str(excinfo.value)
    assert '32633' in msg
    assert '4326' in msg


def test_validate_passes_consistent_projected():
    """ModelType=projected with a projected EPSG and no geographic key."""
    validate_read_metadata({
        'model_type': 1,
        'projected_cs_type': 32633,
        'geographic_type': None,
    })


def test_validate_passes_consistent_geographic():
    """ModelType=geographic with a geographic EPSG and no projected key."""
    validate_read_metadata({
        'model_type': 2,
        'projected_cs_type': None,
        'geographic_type': 4326,
    })


def test_validate_passes_both_keys_with_same_epsg():
    """Some legitimate writers stash the same code under both slots; the
    check only rejects when the codes disagree."""
    validate_read_metadata({
        'model_type': 1,
        'projected_cs_type': 32633,
        'geographic_type': 32633,
    })


def test_validate_passes_user_defined_projected_with_geographic():
    """``32767`` (user-defined) under the projected slot means the
    projected code is not actually resolved -- the geographic key is
    then the only resolved CRS source, which is not a contradiction."""
    validate_read_metadata({
        'model_type': 2,
        'projected_cs_type': 32767,
        'geographic_type': 4326,
    })


def test_validate_passes_user_defined_geographic_with_projected():
    validate_read_metadata({
        'model_type': 1,
        'projected_cs_type': 32633,
        'geographic_type': 32767,
    })


def test_validate_passes_no_geokey_context():
    """A caller without any GeoKey context (e.g. VRT) short-circuits.
    The check is opt-in to the context it consumes."""
    validate_read_metadata({})


def test_validate_passes_when_model_type_undefined():
    """Some legitimate writers omit ModelTypeGeoKey entirely. The check
    is not strict about that on its own; only the explicit conflicts
    above raise."""
    validate_read_metadata({
        'model_type': 0,
        'projected_cs_type': 4326,
        'geographic_type': None,
    })


def test_opt_out_short_circuits_each_case():
    """``allow_inconsistent_geokeys=True`` keeps the legacy permissive
    behaviour across all three rejection cases."""
    # Case 2 (the issue's exact reproducer).
    validate_read_metadata({
        'model_type': 2,
        'projected_cs_type': 4326,
        'allow_inconsistent_geokeys': True,
    })
    # Case 1.
    validate_read_metadata({
        'model_type': 1,
        'geographic_type': 4326,
        'allow_inconsistent_geokeys': True,
    })
    # Case 3.
    validate_read_metadata({
        'model_type': 1,
        'projected_cs_type': 32633,
        'geographic_type': 4326,
        'allow_inconsistent_geokeys': True,
    })


def test_validate_tolerates_float_geokey_values():
    """``_parse_geokeys`` can stash float values for keys whose entries
    point at the doubles section; the check coerces to int."""
    with pytest.raises(InconsistentGeoKeysError):
        validate_read_metadata({
            'model_type': 2.0,
            'projected_cs_type': 4326.0,
            'geographic_type': None,
        })


def test_validate_ignores_non_numeric_geokey_values():
    """A string value in the model_type slot (e.g. a corrupt parse) is
    treated as 'not declared' rather than crashing the validator."""
    # Should not raise even though projected_cs_type is set.
    validate_read_metadata({
        'model_type': 'garbage',
        'projected_cs_type': 4326,
        'geographic_type': None,
    })


@pytest.mark.parametrize(
    "bad_value",
    [
        float('nan'),
        float('inf'),
        float('-inf'),
    ],
    ids=["nan", "inf", "-inf"],
)
def test_validate_tolerates_nan_or_inf_geokey_values(bad_value):
    """Hand-built context dicts with NaN / inf floats in a GeoKey slot
    must not crash the validator. ``int(float('nan'))`` raises
    ValueError and ``int(float('inf'))`` raises OverflowError; the
    check catches both and treats the slot as 'not declared'.

    The reader never produces these for type-code GeoKeys (those parse
    as TIFF SHORT ints), but ``validate_read_metadata`` is a public-ish
    helper that takes an arbitrary dict, so the validator stays robust
    against garbage callers. See PR review follow-up on issue #2417.
    """
    # bad model_type with an otherwise-conflict-looking projected slot.
    validate_read_metadata({
        'model_type': bad_value,
        'projected_cs_type': 4326,
        'geographic_type': None,
    })
    # bad projected_cs_type alongside a valid geographic slot.
    validate_read_metadata({
        'model_type': 2,
        'projected_cs_type': bad_value,
        'geographic_type': 4326,
    })
    # bad geographic_type.
    validate_read_metadata({
        'model_type': 1,
        'projected_cs_type': 32633,
        'geographic_type': bad_value,
    })


# ---------------------------------------------------------------------------
# Full-stack integration: the bug's exact reproducer through open_geotiff().
# ---------------------------------------------------------------------------


def _write_tiff_with_geokeys(
    path: str,
    *,
    model_type: int,
    projected_cs_type: int | None,
    geographic_type: int | None,
) -> None:
    """Hand-build a 4x4 float32 TIFF with a tiny GeoKey directory.

    Only ``ModelTypeGeoKey`` plus optional ``ProjectedCSTypeGeoKey`` and
    ``GeographicTypeGeoKey`` are written; the rest of the directory is
    minimal. Pass ``None`` to either type-key argument to omit it.

    Unique fixture for issue #2417 -- name carries the issue number so
    parallel test runs and other worktrees do not collide on tmp paths.
    """
    bo = '<'
    pixels = np.zeros((4, 4), dtype=np.float32).tobytes()

    # GeoKeyDirectory header: (version, rev_maj, rev_min, n_keys)
    n_keys = 1 + int(projected_cs_type is not None) + int(geographic_type is not None)
    gkd = [1, 1, 0, n_keys]
    # ModelTypeGeoKey: id 1024, location 0 (immediate value), count 1, value.
    gkd.extend([1024, 0, 1, model_type])
    if projected_cs_type is not None:
        gkd.extend([3072, 0, 1, projected_cs_type])
    if geographic_type is not None:
        gkd.extend([2048, 0, 1, geographic_type])

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals),
                         struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_doubles(tag, vals):
        tag_list.append((tag, 12, len(vals),
                         struct.pack(f'{bo}{len(vals)}d', *vals)))

    add_short(256, 4)
    add_short(257, 4)
    add_short(258, 32)
    add_short(259, 1)
    add_short(262, 1)
    add_short(277, 1)
    add_short(339, 3)
    add_short(278, 4)
    add_long(273, 0)
    add_long(279, len(pixels))
    add_doubles(33550, [1.0, 1.0, 0.0])
    add_doubles(33922, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add_shorts(34735, gkd)
    tag_list.sort(key=lambda t: t[0])

    n = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * n + 4
    overflow_start = ifd_start + ifd_size
    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None
    pixel_data_start = overflow_start + len(overflow_buf)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            raw = struct.pack(f'{bo}I', pixel_data_start)
        patched.append((tag, typ, count, raw))
    tag_list = patched
    out = bytearray(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', n))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{bo}I', overflow_start + tag_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixels)
    Path(path).write_bytes(bytes(out))


def test_open_geotiff_rejects_model_geographic_with_projected_key(tmp_path):
    """The exact reproducer from issue #2417. A TIFF declaring
    ``ModelTypeGeoKey=geographic`` with ``ProjectedCSTypeGeoKey=4326``
    used to surface ``attrs['crs']=4326`` silently; now it raises."""
    path = tmp_path / "tmp_2417_model_geographic.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=2,            # geographic
        projected_cs_type=4326,  # but projected key populated
        geographic_type=None,
    )
    with pytest.raises(InconsistentGeoKeysError):
        open_geotiff(str(path))


def test_open_geotiff_rejects_model_projected_with_only_geographic_key(tmp_path):
    path = tmp_path / "tmp_2417_model_projected.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=1,            # projected
        projected_cs_type=None,  # but only geographic key populated
        geographic_type=4326,
    )
    with pytest.raises(InconsistentGeoKeysError):
        open_geotiff(str(path))


def test_open_geotiff_rejects_both_keys_with_different_epsg(tmp_path):
    path = tmp_path / "tmp_2417_both_keys_disagree.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=1,
        projected_cs_type=32633,
        geographic_type=4326,
    )
    with pytest.raises(InconsistentGeoKeysError):
        open_geotiff(str(path))


def test_open_geotiff_accepts_consistent_projected(tmp_path):
    """Control: a TIFF with a consistent projected ModelType + projected
    EPSG should still read cleanly."""
    path = tmp_path / "tmp_2417_consistent_projected.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=1,
        projected_cs_type=32633,
        geographic_type=None,
    )
    da = open_geotiff(str(path))
    assert da.shape == (4, 4)
    # The reader publishes the projected EPSG verbatim.
    assert da.attrs.get('crs') == 32633


def test_open_geotiff_accepts_consistent_geographic(tmp_path):
    path = tmp_path / "tmp_2417_consistent_geographic.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=2,
        projected_cs_type=None,
        geographic_type=4326,
    )
    da = open_geotiff(str(path))
    assert da.shape == (4, 4)
    assert da.attrs.get('crs') == 4326


def test_open_geotiff_opt_out_restores_legacy_behaviour(tmp_path):
    """``allow_inconsistent_geokeys=True`` keeps the pre-#2417 silent
    acceptance for callers with known-quirky historical files."""
    path = tmp_path / "tmp_2417_opt_out.tif"
    _write_tiff_with_geokeys(
        str(path),
        model_type=2,
        projected_cs_type=4326,
        geographic_type=None,
    )
    da = open_geotiff(str(path), allow_inconsistent_geokeys=True)
    # The legacy reader takes ProjectedCSTypeGeoKey first, so the EPSG
    # surfaces in attrs['crs'] verbatim. The bug is that the surface
    # looks trustworthy; the opt-in is the documented way to keep that
    # behaviour for callers who need the legacy contract.
    assert da.attrs.get('crs') == 4326
