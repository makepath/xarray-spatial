"""Validation tests for the ``resolution=`` parameter of ``rasterize`` (#2576).

Before the fix, ``resolution`` shape/type checks were absent: a 3-tuple was
silently truncated to its first two elements, a 1-tuple raised an opaque
``IndexError``, strings tripped a raw ``float()`` conversion on the first
character, and dicts raised ``KeyError: 0``.  This module pins the
clean-``ValueError`` contract for all of those.

The happy paths (scalar number, length-2 tuple) are covered elsewhere in
``test_rasterize_coverage_2026_05_17.py`` and ``test_rasterize_coverage_2026_05_21.py``;
this module focuses on the validation branch added in #2576.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

if has_shapely and has_geopandas:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not (has_shapely and has_geopandas),
    reason="shapely + geopandas required",
)


@pytest.fixture
def simple_gdf():
    """Single square polygon with a numeric column to burn."""
    return gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 10, 10)], "val": [1.0]}
    )


# --- happy paths (sanity checks so a stricter rejection rule does not break
# documented usage) ---

def test_scalar_int_resolution_accepted(simple_gdf):
    out = rasterize(simple_gdf, column="val", resolution=2,
                    bounds=(0, 0, 10, 10), fill=0)
    assert out.shape == (5, 5)


def test_scalar_float_resolution_accepted(simple_gdf):
    out = rasterize(simple_gdf, column="val", resolution=2.0,
                    bounds=(0, 0, 10, 10), fill=0)
    assert out.shape == (5, 5)


def test_length_2_tuple_accepted(simple_gdf):
    out = rasterize(simple_gdf, column="val", resolution=(2.0, 1.0),
                    bounds=(0, 0, 10, 10), fill=0)
    # extent 10 / x_res 2 = 5 cols; extent 10 / y_res 1 = 10 rows.
    assert out.shape == (10, 5)


def test_length_2_list_accepted(simple_gdf):
    out = rasterize(simple_gdf, column="val", resolution=[2.0, 1.0],
                    bounds=(0, 0, 10, 10), fill=0)
    assert out.shape == (10, 5)


def test_numpy_scalar_resolution_accepted(simple_gdf):
    """numpy scalars (np.float32, np.int32, ...) are common in pipelines."""
    out = rasterize(simple_gdf, column="val", resolution=np.float32(2.0),
                    bounds=(0, 0, 10, 10), fill=0)
    assert out.shape == (5, 5)
    out = rasterize(simple_gdf, column="val", resolution=np.int32(2),
                    bounds=(0, 0, 10, 10), fill=0)
    assert out.shape == (5, 5)


def test_numpy_1d_array_resolution_accepted(simple_gdf):
    """1-D numpy array of length 2 is treated as (x_res, y_res)."""
    out = rasterize(
        simple_gdf, column="val",
        resolution=np.array([2.0, 1.0]),
        bounds=(0, 0, 10, 10), fill=0,
    )
    assert out.shape == (10, 5)


# --- rejection paths ---

def test_three_tuple_resolution_rejected(simple_gdf):
    """3-tuple was silently truncated to its first two elements before #2576."""
    with pytest.raises(ValueError, match="length 2"):
        rasterize(simple_gdf, column="val", resolution=(1, 2, 3),
                  bounds=(0, 0, 10, 10), fill=0)


def test_one_tuple_resolution_rejected(simple_gdf):
    """1-tuple used to crash with IndexError from resolution[1]."""
    with pytest.raises(ValueError, match="length 2"):
        rasterize(simple_gdf, column="val", resolution=(1.0,),
                  bounds=(0, 0, 10, 10), fill=0)


def test_empty_tuple_resolution_rejected(simple_gdf):
    with pytest.raises(ValueError, match="length 2"):
        rasterize(simple_gdf, column="val", resolution=(),
                  bounds=(0, 0, 10, 10), fill=0)


def test_string_resolution_rejected(simple_gdf):
    """str used to leak a raw float() conversion error from resolution[0]='f'."""
    with pytest.raises(ValueError, match="number or a length-2 sequence"):
        rasterize(simple_gdf, column="val", resolution="foo",
                  bounds=(0, 0, 10, 10), fill=0)


def test_dict_resolution_rejected(simple_gdf):
    """dict used to leak a KeyError: 0 from resolution[0]."""
    with pytest.raises(ValueError, match="number or a length-2 sequence"):
        rasterize(simple_gdf, column="val", resolution={"x": 1.0, "y": 1.0},
                  bounds=(0, 0, 10, 10), fill=0)


def test_bool_resolution_rejected(simple_gdf):
    """bool is technically int, but ``resolution=True`` is almost certainly
    a caller mistake -- accepting it would silently produce a 1x1 raster.
    """
    with pytest.raises(ValueError, match="number or a length-2 sequence"):
        rasterize(simple_gdf, column="val", resolution=True,
                  bounds=(0, 0, 10, 10), fill=0)


def test_none_element_in_pair_rejected(simple_gdf):
    """``(1.0, None)`` should not slip past as a "length-2 sequence"."""
    with pytest.raises(ValueError, match="elements must be numbers"):
        rasterize(simple_gdf, column="val", resolution=(1.0, None),
                  bounds=(0, 0, 10, 10), fill=0)


def test_non_numeric_element_rejected(simple_gdf):
    with pytest.raises(ValueError, match="elements must be numbers"):
        rasterize(simple_gdf, column="val", resolution=(1.0, "bar"),
                  bounds=(0, 0, 10, 10), fill=0)


def test_2d_numpy_array_rejected(simple_gdf):
    """A 2-D numpy array should not slip past as a length-2 sequence."""
    with pytest.raises(ValueError, match="1-D"):
        rasterize(simple_gdf, column="val",
                  resolution=np.array([[1.0, 2.0], [3.0, 4.0]]),
                  bounds=(0, 0, 10, 10), fill=0)


def test_error_message_names_offending_input(simple_gdf):
    """The ValueError should include the bad input so users can spot it."""
    with pytest.raises(ValueError, match=r"\(1, 2, 3\)"):
        rasterize(simple_gdf, column="val", resolution=(1, 2, 3),
                  bounds=(0, 0, 10, 10), fill=0)
    with pytest.raises(ValueError, match="'foo'"):
        rasterize(simple_gdf, column="val", resolution="foo",
                  bounds=(0, 0, 10, 10), fill=0)
