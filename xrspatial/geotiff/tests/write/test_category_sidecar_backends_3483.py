"""Categorical PAM sidecar round-trips across write backends (#3483, #3518).

``to_geotiff`` emits a PAM ``<file>.aux.xml`` sidecar from
``attrs['category_names']`` / ``attrs['category_colors']`` so GDAL/QGIS show
class labels. The emit is wired into every write path -- the eager numpy
writer, the dask streaming writer, and the GPU (nvCOMP) writer each call
``_write_category_sidecar()`` after writing pixels. The original feature
tests (``xrspatial/tests/test_rasterize_categorical_3482.py``) only round-trip
the eager numpy path, so a refactor that dropped the sidecar emit on the dask
or GPU branch would lose category labels with no failing test.

These tests close that backend-coverage gap: a categorical int32 raster is
written through the dask and GPU backends and read back, asserting both the
labels and the RGBA colors survive. A colors-less variant also covers the
``category_colors=None`` build branch and the names-only read path, which the
existing suite always exercises with colors present.
"""
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .._helpers.markers import requires_gpu

pytest.importorskip("tifffile")

_NAMES = ["water", "forest", "urban"]
_COLORS = [(0, 0, 255, 255), (0, 255, 0, 255), (128, 128, 128, 255)]
# -1 is the categorical nodata sentinel rasterize uses for untouched cells.
_CODES = np.array([[-1, 0, 1], [1, 2, 0], [2, -1, 1]], dtype=np.int32)


def _categorical_da(values, *, with_colors=True):
    """An int32 categorical DataArray carrying category attrs + georef."""
    attrs = {
        "category_names": list(_NAMES),
        "nodata": -1,
        "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        "crs": 4326,
    }
    if with_colors:
        attrs["category_colors"] = list(_COLORS)
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
        attrs=attrs,
    )


def test_dask_write_round_trips_categories(tmp_path):
    """The dask streaming writer emits and round-trips the sidecar."""
    import dask.array as dsa

    da = _categorical_da(dsa.from_array(_CODES, chunks=(2, 3)))
    path = str(tmp_path / "dask_cat_3483.tif")
    to_geotiff(da, path)

    assert os.path.exists(path + ".aux.xml")
    back = open_geotiff(path)
    assert back.attrs["category_names"] == _NAMES
    assert list(back.attrs["category_colors"]) == _COLORS


@requires_gpu
def test_gpu_write_round_trips_categories(tmp_path):
    """The GPU (nvCOMP) writer emits and round-trips the sidecar."""
    import cupy

    da = _categorical_da(cupy.asarray(_CODES))
    path = str(tmp_path / "gpu_cat_3483.tif")
    to_geotiff(da, path, gpu=True)

    assert os.path.exists(path + ".aux.xml")
    back = open_geotiff(path)
    assert back.attrs["category_names"] == _NAMES
    assert list(back.attrs["category_colors"]) == _COLORS


def test_round_trips_names_without_colors(tmp_path):
    """A names-only categorical array round-trips labels with no colors.

    Exercises the ``category_colors=None`` branch of ``build_pam_xml`` (no
    Red/Green/Blue/Alpha RAT columns) and the matching names-only read path,
    which the existing suite never hits because it always supplies colors.
    """
    da = _categorical_da(_CODES, with_colors=False)
    path = str(tmp_path / "names_only_3483.tif")
    to_geotiff(da, path)

    assert os.path.exists(path + ".aux.xml")
    back = open_geotiff(path)
    assert back.attrs["category_names"] == _NAMES
    assert "category_colors" not in back.attrs
