"""Smoke tests for the nodata-sentinel golden-corpus fixtures (#1930, Phase 2.6).

Three fixtures exercise the three nodata conventions the manifest schema
recognises:

* ``nodata_int_sentinel_uint16``  -- explicit integer sentinel
* ``nodata_nan_float32``          -- ``nodata=NaN`` (string-encoded in YAML)
* ``nodata_miniswhite_uint8``     -- photometric=miniswhite, no tag

For each fixture we assert:

1. The file on disk is a valid TIFF that rasterio can open;
2. The nodata convention is observable on the rasterio source (an int /
   NaN tag, or the IMAGE_STRUCTURE MINISWHITE flag);
3. ``compare_to_oracle`` accepts a hand-built DataArray that mirrors what
   an xrspatial backend would emit. This proves the oracle's NaN-aware
   nodata comparison handles each convention end-to-end.

These tests do not touch any read backend -- backend wiring is deferred
to Phase 3 per the plan on #1930. The xrspatial-shaped DataArray here is
synthesised directly from the rasterio read so the oracle has something
to compare against.

TODO(#1988): When the codebase grows a "declared nodata vs masked-data
state" split, switch the candidate construction here to drive both sides
explicitly. Today the candidate's ``attrs['nodata']`` mirrors whatever
the rasterio source reports, which is the same shape the existing
xrspatial reader emits.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

rasterio = pytest.importorskip('rasterio')

from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'

FIXTURE_INT = FIXTURE_DIR / 'nodata_int_sentinel_uint16.tif'
FIXTURE_NAN = FIXTURE_DIR / 'nodata_nan_float32.tif'
FIXTURE_MINISWHITE = FIXTURE_DIR / 'nodata_miniswhite_uint8.tif'

ALL_FIXTURES = (FIXTURE_INT, FIXTURE_NAN, FIXTURE_MINISWHITE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_from_source(src) -> xr.DataArray:
    """Build the xrspatial-shaped DataArray a backend would emit.

    Mirrors ``coords_from_pixel_geometry`` (pixel-centre coords) and the
    ``attrs['transform']`` 6-tuple shape used elsewhere in xrspatial.
    """
    arr = src.read(1)
    transform = src.transform
    height, width = arr.shape
    pw = float(transform.a)
    ph = float(transform.e)
    ox = float(transform.c)
    oy = float(transform.f)
    x = ox + (np.arange(width) + 0.5) * pw
    y = oy + (np.arange(height) + 0.5) * ph
    attrs: dict = {'transform': (pw, 0.0, ox, 0.0, ph, oy)}
    epsg = src.crs.to_epsg() if src.crs is not None else None
    if epsg is not None:
        attrs['crs'] = epsg
    elif src.crs is not None:
        attrs['crs_wkt'] = src.crs.to_wkt()
    if src.nodata is not None:
        attrs['nodata'] = src.nodata
    return xr.DataArray(arr, dims=('y', 'x'), coords={'y': y, 'x': x}, attrs=attrs)


# ---------------------------------------------------------------------------
# Per-fixture parametrised TIFF validity check
# ---------------------------------------------------------------------------

# Tight budget chosen from the largest fixture today (1402 bytes for the
# float32 NaN file). The plan caps fixtures at 4 KB; tightening to 2 KB
# here catches silent bloat (accidental overviews, predictor changes)
# before it drifts toward the documented limit.
_FIXTURE_SIZE_BUDGET = 2048


@pytest.mark.parametrize('path', ALL_FIXTURES, ids=lambda p: p.name)
def test_fixture_is_a_valid_tiff(path: Path) -> None:
    """Each fixture exists, opens cleanly, and is small enough for git."""
    assert path.exists(), f'corpus fixture missing on disk: {path}'
    size = path.stat().st_size
    assert size < _FIXTURE_SIZE_BUDGET, (
        f'{path.name} exceeded {_FIXTURE_SIZE_BUDGET} byte budget: '
        f'{size} bytes')
    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.width == 16
        assert src.height == 16
        src.read(1)  # raises if the file is unreadable


# ---------------------------------------------------------------------------
# Per-convention assertions about the rasterio-observable nodata state
# ---------------------------------------------------------------------------

def test_int_sentinel_round_trips_through_rasterio() -> None:
    """rasterio reads back the integer sentinel and the planted pixels."""
    with rasterio.open(FIXTURE_INT) as src:
        assert src.dtypes[0] == 'uint16'
        # rasterio reports nodata as a float, but it represents int 0.
        assert src.nodata is not None
        assert not math.isnan(src.nodata)
        assert src.nodata == 0
        arr = src.read(1)
        # The generator stamps three deterministic positions on the sentinel.
        assert int(np.sum(arr == 0)) >= 3


def test_nan_sentinel_round_trips_through_rasterio() -> None:
    """rasterio reads back a NaN nodata and the planted NaN pixels."""
    with rasterio.open(FIXTURE_NAN) as src:
        assert src.dtypes[0] == 'float32'
        assert src.nodata is not None and math.isnan(src.nodata)
        arr = src.read(1)
        assert int(np.sum(np.isnan(arr))) >= 3


def test_miniswhite_is_visible_on_the_rasterio_source() -> None:
    """The miniswhite photometric is observable via IMAGE_STRUCTURE tags.

    rasterio does not surface miniswhite via the ``photometric`` property
    on read for a GTiff opened without a colourmap, but it is reachable
    through the IMAGE_STRUCTURE namespace tags. The oracle reads from
    the rasterio source directly, so any backend wiring that wants the
    photometric flag must read it from the same place.
    """
    with rasterio.open(FIXTURE_MINISWHITE) as src:
        assert src.dtypes[0] == 'uint8'
        assert src.nodata is None  # white-as-min carries no nodata tag
        tags = src.tags(ns='IMAGE_STRUCTURE')
        assert tags.get('MINISWHITE') == 'YES', (
            f'miniswhite flag missing from IMAGE_STRUCTURE: {tags}')
        arr = src.read(1)
        # The generator stamps three deterministic pixels at the dtype max.
        assert int(np.sum(arr == 255)) >= 3


# ---------------------------------------------------------------------------
# Oracle accepts each convention end-to-end
# ---------------------------------------------------------------------------

def test_oracle_accepts_int_sentinel_fixture() -> None:
    with rasterio.open(FIXTURE_INT) as src:
        cand = _candidate_from_source(src)
    compare_to_oracle(FIXTURE_INT, cand)


def test_oracle_accepts_nan_sentinel_fixture() -> None:
    """Confirms the oracle's NaN-aware equality path handles ``nodata=NaN``.

    A plain ``==`` comparison would fail because ``NaN != NaN``;
    ``_nodata_equal`` and ``_pixels_equal`` (with ``equal_nan=True``) are
    what makes this pass.
    """
    with rasterio.open(FIXTURE_NAN) as src:
        cand = _candidate_from_source(src)
    # Sanity check: the candidate carries the NaN sentinel and at least
    # one NaN pixel, so the test would fail if the oracle short-circuited.
    assert math.isnan(cand.attrs['nodata'])
    assert int(np.isnan(cand.values).sum()) >= 3
    compare_to_oracle(FIXTURE_NAN, cand)


def test_oracle_accepts_miniswhite_fixture() -> None:
    """Confirms the oracle accepts the miniswhite convention.

    The white-as-min file carries no nodata tag, so the oracle's nodata
    branch compares ``None`` on both sides. The photometric flag itself
    is not part of the canonical-attrs contract yet (#1984), and is read
    by callers from the rasterio source directly.
    """
    with rasterio.open(FIXTURE_MINISWHITE) as src:
        cand = _candidate_from_source(src)
        assert src.tags(ns='IMAGE_STRUCTURE').get('MINISWHITE') == 'YES'
    assert 'nodata' not in cand.attrs
    compare_to_oracle(FIXTURE_MINISWHITE, cand)
