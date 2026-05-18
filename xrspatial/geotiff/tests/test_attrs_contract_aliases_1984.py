"""Locking test for the compatibility-alias tier of the attrs contract
(issue #1984, PR 5 of 7).

xrspatial's canonical NoData attr is ``attrs['nodata']``. Two
ecosystem aliases are tolerated on *read* paths so foreign DataArrays
(rioxarray-style ``attrs['nodatavals']`` tuples, CF-style
``attrs['_FillValue']``) survive a round-trip through ``to_geotiff``.

This module pins both halves of that contract in one place so future
drift surfaces here:

* Read paths must accept the two aliases when ``attrs['nodata']`` is
  absent, and must prefer ``nodata`` when present.
* Write paths must not emit either alias into the resulting attrs
  after a round-trip; only ``nodata`` should appear.

A finer-grained set of regression tests for the same aliases lives in
``test_nodata_attr_aliases_1582.py``. This file is intentionally
contract-shaped: it does not duplicate those cases but locks the
high-level rules.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _resolve_nodata_attr


# Arbitrary float-castable sentinel that is distinct from any data value
# used below. The value itself does not matter; the tests assert it
# survives the alias resolver and the round-trip.
_SENTINEL = -9999.0


def _da_float(arr, **attrs):
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={"y": np.arange(arr.shape[0], dtype=np.float64),
                "x": np.arange(arr.shape[1], dtype=np.float64)},
        attrs=attrs,
    )


@pytest.fixture
def _arr_with_sentinel():
    return np.array(
        [[1.0, 2.0, _SENTINEL], [3.0, _SENTINEL, 5.0]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Read-side: aliases are honoured when ``attrs['nodata']`` is absent.
# ---------------------------------------------------------------------------


def test_read_uses_nodatavals_when_nodata_absent(tmp_path, _arr_with_sentinel):
    """Pin: a foreign DataArray carrying only ``nodatavals`` round-trips
    through ``to_geotiff`` with the sentinel preserved."""
    da = _da_float(_arr_with_sentinel, crs=4326, nodatavals=(_SENTINEL,))
    assert "nodata" not in da.attrs  # precondition: only the alias is set
    out = str(tmp_path / "alias_nodatavals.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


def test_read_uses_fill_value_when_nodata_absent(tmp_path, _arr_with_sentinel):
    """Pin: a foreign DataArray carrying only CF ``_FillValue`` round-trips
    through ``to_geotiff`` with the sentinel preserved."""
    da = _da_float(_arr_with_sentinel, crs=4326,
                   **{"_FillValue": _SENTINEL})
    assert "nodata" not in da.attrs
    out = str(tmp_path / "alias_fillvalue.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL


# ---------------------------------------------------------------------------
# Precedence: canonical ``nodata`` wins over the two aliases.
# ---------------------------------------------------------------------------


def test_canonical_nodata_wins_over_aliases_at_resolver(_arr_with_sentinel):
    """``_resolve_nodata_attr`` (the read-side chokepoint) still prefers
    canonical ``nodata`` over both aliases. This is the resolver-layer
    contract that downstream code consumes; the write-side fail-closed
    check is exercised in ``test_canonical_nodata_wins_over_aliases_at_write``
    below."""
    canonical = -8888.0
    da = _da_float(
        _arr_with_sentinel, crs=4326,
        nodata=canonical,
        nodatavals=(_SENTINEL,),
        **{"_FillValue": -7777.0},
    )
    assert _resolve_nodata_attr(dict(da.attrs)) == canonical


def test_canonical_nodata_wins_over_aliases_at_write(
        tmp_path, _arr_with_sentinel):
    """Pin: the legacy "canonical nodata silently wins on write" was
    replaced by a fail-closed raise (issue #1987 PR 7). A DataArray with
    disagreeing ``nodata`` and ``nodatavals`` attrs now refuses to write
    instead of dropping the alias. The opt-out is the explicit
    ``nodata=`` kwarg, which overrides both attrs."""
    from xrspatial.geotiff import ConflictingNodataError

    da = _da_float(
        _arr_with_sentinel, crs=4326,
        nodata=-8888.0,
        nodatavals=(_SENTINEL,),
        **{"_FillValue": -7777.0},
    )

    out = str(tmp_path / "canonical_wins.tif")
    with pytest.raises(ConflictingNodataError):
        to_geotiff(da, out)

    # Explicit ``nodata=`` bypasses the check.
    to_geotiff(da, out, nodata=-8888.0)
    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == -8888.0


# ---------------------------------------------------------------------------
# Write-side: a round-trip must emit only ``nodata``, not the aliases.
# ---------------------------------------------------------------------------


def test_write_does_not_emit_aliases_when_canonical_present(
        tmp_path, _arr_with_sentinel):
    """Pin: round-tripping a DataArray that has only ``attrs['nodata']``
    set returns attrs containing ``nodata`` and *not* the two aliases.

    Reader paths emit a single canonical key; this guards against a
    future change that copies the resolved value into the rioxarray or
    CF alias slot as well (which would silently double-write the
    sentinel and confuse downstream consumers that special-case one
    alias over another)."""
    da = _da_float(_arr_with_sentinel, crs=4326, nodata=_SENTINEL)
    out = str(tmp_path / "canonical_only.tif")
    to_geotiff(da, out)

    rd = open_geotiff(out)
    assert rd.attrs.get("nodata") == _SENTINEL
    assert "nodatavals" not in rd.attrs, (
        f"reader emitted rioxarray alias alongside canonical nodata: "
        f"attrs={sorted(rd.attrs)}"
    )
    assert "_FillValue" not in rd.attrs, (
        f"reader emitted CF alias alongside canonical nodata: "
        f"attrs={sorted(rd.attrs)}"
    )


# ---------------------------------------------------------------------------
# NaN-in-nodatavals quirk: NaN entries are skipped, not treated as sentinel.
# ---------------------------------------------------------------------------


def test_nan_in_nodatavals_is_skipped():
    """Pin: NaN entries in ``attrs['nodatavals']`` are not a sentinel.

    NaN means the float NaN is already the implicit sentinel, which
    doesn't need a GDAL_NODATA tag. The resolver must skip NaN entries
    and (in a mixed tuple) return the first concrete numeric value."""
    # Pure-NaN tuple resolves to None.
    assert _resolve_nodata_attr({"nodatavals": (float("nan"),)}) is None
    # NaN followed by a concrete sentinel: the resolver skips NaN and
    # returns the concrete value.
    assert _resolve_nodata_attr(
        {"nodatavals": (float("nan"), _SENTINEL)}) == _SENTINEL
    # None entries are also skipped (rioxarray uses None for "no sentinel
    # on this band").
    assert _resolve_nodata_attr(
        {"nodatavals": (None, _SENTINEL)}) == _SENTINEL
