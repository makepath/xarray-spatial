"""4-backend attrs parity tests for issue #1548.

Before the fix, ``open_geotiff`` returned a different ``attrs`` set
depending on which backend handled the read:

* numpy (eager): full set, including ``x_resolution``, ``y_resolution``,
  ``resolution_unit``, ``extra_tags``, ``image_description``,
  ``extra_samples``.
* dask: only ``crs``, ``transform``, ``raster_type``, ``nodata``.
* cupy / dask+cupy: only ``crs``, ``crs_wkt``, ``transform``, ``nodata``.

The fix factors a single ``_populate_attrs_from_geo_info`` helper and
calls it from every read path, so all four backends now emit the same
keys with the same values for the same input file.

These tests pin that contract.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

tifffile = pytest.importorskip("tifffile")


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _write_tiff_with_pass_through_tags(path):
    """Write a tiled 2-band float32 TIFF that exercises the pass-through
    TIFF tags called out in the issue.

    Uses tifffile's first-class ``resolution`` / ``resolutionunit`` /
    ``description`` kwargs (its preferred path; ``extratags`` would be
    silently dropped for the resolution rationals). ``metadata=None``
    suppresses tifffile's auto-generated shape JSON in ImageDescription
    so the fixture description survives. ``ExtraSamples`` (338) is
    auto-derived from the 2-band layout.
    """
    arr = np.random.default_rng(seed=1548).random(
        (64, 64, 2)).astype(np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        tile=(32, 32), compression='deflate',
        resolution=(300, 300), resolutionunit=2,
        description='issue-1548 parity fixture',
        metadata=None,
    )
    return arr


def _attrs_subset(attrs, keys):
    """Slice attrs down to the keys we are comparing across backends.

    ``cmap`` is a matplotlib object whose equality is brittle; ``transform``
    is a tuple of floats so handle separately.
    """
    return {k: attrs.get(k) for k in keys}


_PASS_THROUGH_KEYS = (
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    'image_description',
    'extra_samples',
)


def test_numpy_attrs_includes_pass_through_tags(tmp_path):
    """Sanity baseline: the eager numpy path emits all pass-through keys."""
    path = str(tmp_path / 'attrs_parity_1548_baseline.tif')
    _write_tiff_with_pass_through_tags(path)

    da = open_geotiff(path)
    for key in _PASS_THROUGH_KEYS:
        assert key in da.attrs, (
            f"eager numpy is the canonical reference and should always "
            f"emit '{key}'; got attrs={sorted(da.attrs.keys())}"
        )


def test_dask_attrs_match_numpy(tmp_path):
    """The dask read path now emits the same pass-through attrs as numpy."""
    path = str(tmp_path / 'attrs_parity_1548_dask.tif')
    _write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    dk_da = open_geotiff(path, chunks=32)

    np_subset = _attrs_subset(np_da.attrs, _PASS_THROUGH_KEYS)
    dk_subset = _attrs_subset(dk_da.attrs, _PASS_THROUGH_KEYS)

    assert dk_subset == np_subset, (
        f"dask attrs diverge from numpy:\n"
        f"  numpy: {np_subset}\n"
        f"  dask : {dk_subset}"
    )


@_gpu_only
def test_cupy_attrs_match_numpy(tmp_path):
    """Cupy / GPU read emits the same pass-through attrs as numpy."""
    path = str(tmp_path / 'attrs_parity_1548_cupy.tif')
    _write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    gpu_da = open_geotiff(path, gpu=True)

    np_subset = _attrs_subset(np_da.attrs, _PASS_THROUGH_KEYS)
    gpu_subset = _attrs_subset(gpu_da.attrs, _PASS_THROUGH_KEYS)

    assert gpu_subset == np_subset, (
        f"cupy attrs diverge from numpy:\n"
        f"  numpy: {np_subset}\n"
        f"  cupy : {gpu_subset}"
    )


@_gpu_only
def test_dask_cupy_attrs_match_numpy(tmp_path):
    """Combined dask+cupy read still emits the pass-through attrs."""
    path = str(tmp_path / 'attrs_parity_1548_dask_cupy.tif')
    _write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    combined = open_geotiff(path, gpu=True, chunks=32)

    np_subset = _attrs_subset(np_da.attrs, _PASS_THROUGH_KEYS)
    combined_subset = _attrs_subset(combined.attrs, _PASS_THROUGH_KEYS)

    assert combined_subset == np_subset, (
        f"dask+cupy attrs diverge from numpy:\n"
        f"  numpy     : {np_subset}\n"
        f"  dask+cupy : {combined_subset}"
    )


def test_all_backend_attrs_keysets_equal(tmp_path):
    """Strong contract: the *set* of attrs keys is identical across all
    available backends, not just the pass-through subset.

    This guards against any future read path silently dropping a
    different attr that nobody happened to test for above.
    """
    path = str(tmp_path / 'attrs_parity_1548_keysets.tif')
    _write_tiff_with_pass_through_tags(path)

    np_keys = set(open_geotiff(path).attrs.keys())
    dk_keys = set(open_geotiff(path, chunks=32).attrs.keys())

    backend_keys = {'numpy': np_keys, 'dask+numpy': dk_keys}
    if _HAS_GPU:
        backend_keys['cupy'] = set(open_geotiff(path, gpu=True).attrs.keys())
        backend_keys['dask+cupy'] = set(
            open_geotiff(path, gpu=True, chunks=32).attrs.keys())

    differences = {
        name: keys ^ np_keys
        for name, keys in backend_keys.items()
        if keys != np_keys
    }
    assert not differences, (
        f"backend attrs keysets diverge from numpy:\n"
        f"  numpy keys: {sorted(np_keys)}\n"
        f"  diffs    : "
        + "\n             ".join(
            f"{name}: symmetric_diff={sorted(diff)}"
            for name, diff in differences.items()
        )
    )
