"""Regression: rasterize() dtype annotation covers the accepted forms.

Pins the fix for issue #3291 (api-consistency sweep, Cat 3). The
implementation accepts anything ``np.dtype()`` can parse -- a dtype name
string, a numpy scalar type, or an ``np.dtype`` instance -- and the
module's own tests call it with scalar types (``dtype=np.int32``). The
old ``Optional[np.dtype]`` hint rejected two of the three working forms
under mypy/pyright.
"""
from __future__ import annotations

import typing

import numpy as np
import pytest

from xrspatial.rasterize import rasterize


def test_dtype_annotation_accepts_str_dtype_and_type():
    hints = typing.get_type_hints(rasterize)
    args = set(typing.get_args(hints["dtype"]))
    assert str in args, "dtype hint must accept dtype name strings"
    assert np.dtype in args, "dtype hint must accept np.dtype instances"
    assert type in args, "dtype hint must accept numpy scalar types"
    assert type(None) in args, "dtype hint must stay optional"


@pytest.mark.parametrize(
    "dtype_arg",
    ["int32", np.int32, np.dtype("int32")],
    ids=["str-name", "scalar-type", "dtype-instance"],
)
def test_dtype_forms_accepted_at_runtime(dtype_arg):
    pytest.importorskip("shapely")
    from shapely.geometry import box

    result = rasterize(
        [(box(0, 0, 5, 5), 1.0)],
        width=4, height=4, bounds=(0, 0, 10, 10),
        fill=0, dtype=dtype_arg,
    )
    assert result.dtype == np.int32
