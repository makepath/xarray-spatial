"""Issue #2215: NonUniformCoordsError fires consistently across spatial aliases.

``to_geotiff`` documents support for spatial dim aliases (``lat``/``lon``,
``latitude``/``longitude``, ``row``/``col``). Before this fix the
ambiguous-metadata validator only looked up the literal ``coords['y']``
and ``coords['x']`` entries, so alias-named non-uniform coords slipped
past the validator. The transform-synthesis path in ``_coords.py``
still caught them downstream, but it raised plain ``ValueError`` rather
than the exported ``NonUniformCoordsError``. The exception type ended
up depending on which dim name the caller picked.

After the fix:

* The validator resolves the documented aliases via
  ``_resolve_spatial_coords`` before passing coord arrays into
  ``validate_write_metadata``.
* The transform-synthesis path raises ``NonUniformCoordsError``
  (subclass of ``ValueError``) so legacy ``except ValueError`` callers
  still catch it and new ``except NonUniformCoordsError`` callers see
  it consistently.

Tests pin:

1. Alias-named non-uniform coords raise ``NonUniformCoordsError`` and
   ``isinstance(exc, NonUniformCoordsError)`` holds for every alias in
   ``_Y_DIM_NAMES`` / ``_X_DIM_NAMES``.
2. ``y``/``x``-named non-uniform coords still raise
   ``NonUniformCoordsError`` (no regression on the existing path).
3. Uniform alias-named coords still write successfully (alias
   resolution does not break legitimate inputs).
4. The exception is a ``ValueError`` subclass, so legacy
   ``except ValueError`` callers keep working.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import NonUniformCoordsError, to_geotiff
from xrspatial.geotiff._runtime import (
    _X_DIM_NAMES,
    _Y_DIM_NAMES,
    _resolve_spatial_coords,
)


def _da_with_alias_coords(y_name, x_name, *, y_coord=None, x_coord=None,
                          shape=(4, 4)):
    """Build a 2-D DataArray with alias-named y/x dims and coords."""
    data = np.zeros(shape, dtype=np.float32)
    if y_coord is None:
        y_coord = np.linspace(3.0, 0.0, shape[0], dtype=np.float64)
    if x_coord is None:
        x_coord = np.linspace(0.0, 3.0, shape[1], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=(y_name, x_name),
        coords={y_name: y_coord, x_name: x_coord},
    )


_ALIAS_PAIRS = [
    ('y', 'x'),                  # canonical
    ('lat', 'lon'),
    ('latitude', 'longitude'),
    ('row', 'col'),
]


# ---------------------------------------------------------------------------
# Helper test: _resolve_spatial_coords picks the right coord arrays.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
def test_resolve_spatial_coords_finds_alias(y_name, x_name):
    """Each documented alias resolves to the matching coord array."""
    da = _da_with_alias_coords(y_name, x_name)
    coord_y, coord_x = _resolve_spatial_coords(da)
    assert coord_y is not None
    assert coord_x is not None
    np.testing.assert_array_equal(coord_y, da.coords[y_name].values)
    np.testing.assert_array_equal(coord_x, da.coords[x_name].values)


def test_resolve_spatial_coords_picks_canonical_first():
    """When both ``y`` and an alias exist, canonical wins.

    The alias list places ``y`` / ``x`` first so an array that happens
    to carry both names (rare, but possible after a rename + retain)
    keeps matching exactly the coord it matched before issue #2215.
    """
    data = np.zeros((4, 4), dtype=np.float32)
    y_arr = np.linspace(3.0, 0.0, 4, dtype=np.float64)
    lat_arr = np.array([99.0, 88.0, 77.0, 66.0], dtype=np.float64)
    x_arr = np.linspace(0.0, 3.0, 4, dtype=np.float64)
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': y_arr, 'x': x_arr, 'lat': ('y', lat_arr)},
    )
    coord_y, _ = _resolve_spatial_coords(da)
    np.testing.assert_array_equal(coord_y, y_arr)


def test_resolve_spatial_coords_missing_returns_none():
    """No matching coord on either axis returns ``(None, None)``."""
    data = np.zeros((4, 4), dtype=np.float32)
    da = xr.DataArray(data, dims=('foo', 'bar'))
    coord_y, coord_x = _resolve_spatial_coords(da)
    assert coord_y is None
    assert coord_x is None


def test_resolve_spatial_coords_handles_none_input():
    """Passing an object with no ``coords`` attribute returns ``(None, None)``."""
    coord_y, coord_x = _resolve_spatial_coords(object())
    assert coord_y is None
    assert coord_x is None


# ---------------------------------------------------------------------------
# Non-uniform alias coords raise NonUniformCoordsError (not plain ValueError).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
def test_non_uniform_y_alias_raises_non_uniform_coords_error(
        tmp_path, y_name, x_name):
    """Non-uniform y-axis coords trip ``NonUniformCoordsError`` for every alias.

    Without the fix, only ``y_name == 'y'`` raised the typed error;
    alias names slipped past the validator and surfaced a plain
    ``ValueError`` from the later transform-synthesis path.
    """
    da = _da_with_alias_coords(
        y_name, x_name,
        y_coord=np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
    )
    with pytest.raises(NonUniformCoordsError) as exc_info:
        to_geotiff(da, str(tmp_path / f'non_uniform_{y_name}_2215.tif'))
    # The user-facing contract: ``isinstance(exc, NonUniformCoordsError)``
    # holds regardless of which alias was used.
    assert isinstance(exc_info.value, NonUniformCoordsError)
    # And the legacy ``except ValueError`` clause still catches it,
    # because NonUniformCoordsError subclasses ValueError.
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
def test_non_uniform_x_alias_raises_non_uniform_coords_error(
        tmp_path, y_name, x_name):
    """Non-uniform x-axis coords trip ``NonUniformCoordsError`` for every alias."""
    da = _da_with_alias_coords(
        y_name, x_name,
        x_coord=np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64),
    )
    with pytest.raises(NonUniformCoordsError) as exc_info:
        to_geotiff(da, str(tmp_path / f'non_uniform_{x_name}_2215.tif'))
    assert isinstance(exc_info.value, NonUniformCoordsError)
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
def test_constant_y_alias_raises_non_uniform_coords_error(tmp_path, y_name, x_name):
    """Constant (zero-step) y-axis coords raise the typed error for every alias."""
    da = _da_with_alias_coords(
        y_name, x_name,
        y_coord=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    )
    with pytest.raises(NonUniformCoordsError) as exc_info:
        to_geotiff(da, str(tmp_path / f'constant_{y_name}_2215.tif'))
    assert isinstance(exc_info.value, NonUniformCoordsError)


# ---------------------------------------------------------------------------
# Uniform alias coords still succeed (no false positives).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('y_name,x_name', _ALIAS_PAIRS)
def test_uniform_alias_coords_write_successfully(tmp_path, y_name, x_name):
    """Alias-named coords with uniform spacing keep writing cleanly."""
    da = _da_with_alias_coords(y_name, x_name)
    out = tmp_path / f'uniform_{y_name}_{x_name}_2215.tif'
    to_geotiff(da, str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# Coverage of every alias pair against _Y_DIM_NAMES / _X_DIM_NAMES.
# ---------------------------------------------------------------------------


def test_alias_pairs_cover_every_documented_name():
    """Pin that the parametrization above covers every documented alias.

    If a new alias is added to ``_Y_DIM_NAMES`` / ``_X_DIM_NAMES``, this
    assertion fails and forces the parametrization to be updated so the
    consistency guarantee actually holds for the new name.
    """
    parametrized_y = {pair[0] for pair in _ALIAS_PAIRS}
    parametrized_x = {pair[1] for pair in _ALIAS_PAIRS}
    assert parametrized_y == set(_Y_DIM_NAMES), (
        f"Y alias coverage drift: parametrized={parametrized_y}, "
        f"_Y_DIM_NAMES={set(_Y_DIM_NAMES)}"
    )
    assert parametrized_x == set(_X_DIM_NAMES), (
        f"X alias coverage drift: parametrized={parametrized_x}, "
        f"_X_DIM_NAMES={set(_X_DIM_NAMES)}"
    )


# ---------------------------------------------------------------------------
# Backward-compat: existing y/x callers still see the same exception type.
# ---------------------------------------------------------------------------


def test_legacy_except_value_error_still_catches(tmp_path):
    """Callers using ``except ValueError`` keep working on the typed path.

    ``NonUniformCoordsError`` subclasses ``ValueError`` (via
    ``GeoTIFFAmbiguousMetadataError``), so the legacy try/except shape
    keeps working even though the concrete type changed.
    """
    da = _da_with_alias_coords(
        'y', 'x',
        y_coord=np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float64),
    )
    try:
        to_geotiff(da, str(tmp_path / 'legacy_except_2215.tif'))
    except ValueError as exc:
        assert isinstance(exc, NonUniformCoordsError)
    else:  # pragma: no cover - defensive
        pytest.fail("expected ValueError (NonUniformCoordsError)")
