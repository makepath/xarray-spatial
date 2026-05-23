"""GPU + sidecar georef-attr parity across backends (issue #2324).

The GPU eager path used to overwrite the local ``data`` / ``header``
references with the sidecar's buffer before calling
``extract_geo_info_with_overview_inheritance``. For the common GDAL
sidecar case where overview IFDs lack their own geokeys, the helper
then parsed the inherited base IFD against the sidecar bytes -- the
returned georef came out silently wrong, while the CPU eager and dask
paths handled it correctly.

These tests load the same ``.tif`` + ``.tif.ovr`` pair via the three
backends and assert georef attrs match. The GPU portion skips cleanly
when cupy / CUDA is unavailable so the test runs on CI without a GPU
and still guards the metadata code on a real GPU box.

Test fixture helpers (sidecar writer, NewSubfileType insertion, georef
tag stripping) are shared with ``test_sidecar_own_geokeys_2315.py`` --
imported rather than duplicated so a fixture-format change there does
not silently desync this file.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff, read_geotiff_dask

from .test_sidecar_own_geokeys_2315 import _write_pair


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def _attrs_subset(da):
    """Pull the georef attrs we want to compare across backends."""
    return {
        "transform": tuple(da.attrs.get("transform", ())),
        "crs": da.attrs.get("crs"),
    }


def test_sidecar_without_geokeys_attrs_match_cpu_vs_dask_2324(tmp_path):
    """Baseline: CPU eager and dask agree on inherited georef.

    Pre-#2324 already passes; included so the parity matrix is complete
    and a future regression on either path surfaces here too. The GPU
    half lives in ``test_sidecar_without_geokeys_gpu_matches_cpu_2324``.
    """
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=False,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    dsk = read_geotiff_dask(str(base), overview_level=1, chunks=2)

    assert _attrs_subset(cpu) == _attrs_subset(dsk)
    # And the inherited transform is what we expect from the base file
    # (pixel size rescaled by 8/4 = 2 for the 4x4 overview).
    t = cpu.attrs["transform"]
    assert t[0] == pytest.approx(20.0)
    assert t[2] == pytest.approx(100.0)
    assert t[4] == pytest.approx(-20.0)
    assert t[5] == pytest.approx(200.0)
    assert cpu.attrs.get("crs") == 4326


@_gpu_only
def test_sidecar_without_geokeys_gpu_matches_cpu_2324(tmp_path):
    """Regression for #2324: GPU eager georef matches CPU / dask.

    The sidecar is stripped of its georef tags, so the inheritance
    helper has to walk back to the base IFD. Pre-fix, the GPU path
    handed the helper the sidecar bytes (because ``data, header`` got
    swapped before the call) and parsed the base IFD's georef offsets
    against the sidecar buffer -- silently corrupt transform / crs.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=False,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    gpu = read_geotiff_gpu(str(base), overview_level=1)

    # Same georef attrs as the CPU path.
    assert _attrs_subset(gpu) == _attrs_subset(cpu)
    # And pixel data also matches (sidecar bytes are still consumed for
    # the actual decode; this confirms the fix did not break the tile
    # read path while moving the variable rename).
    np.testing.assert_array_equal(np.asarray(gpu.data.get()),
                                  np.asarray(cpu.data))


@_gpu_only
def test_sidecar_with_own_geokeys_gpu_matches_cpu_2324(tmp_path):
    """GPU path routes a sidecar-owned georef payload to sidecar bytes.

    When the sidecar IFD declares its own georef payload the helper's
    ``sidecar_origin`` argument routes the parse to the sidecar buffer.
    The CPU eager / dask paths already pass the mapping; the GPU eager
    path now passes it too. The sidecar's transform must win for both
    backends.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(
        tmp_path,
        base_geo=base_geo, sidecar_geo=side_geo,
        sidecar_has_geokeys=True,
    )

    cpu = open_geotiff(str(base), overview_level=1)
    gpu = read_geotiff_gpu(str(base), overview_level=1)

    assert _attrs_subset(gpu) == _attrs_subset(cpu)
    # Sidecar's transform won on both sides.
    t = gpu.attrs["transform"]
    assert t[0] == pytest.approx(2.5)
    assert t[2] == pytest.approx(500.0)
    assert t[4] == pytest.approx(-2.5)
    assert t[5] == pytest.approx(800.0)
    assert gpu.attrs.get("crs") == 3857
