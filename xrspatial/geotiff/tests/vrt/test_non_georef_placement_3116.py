"""Non-georeferenced VRT tile placement (issue #3116).

``to_geotiff(da, 'out.vrt', tile_size=N)`` on a non-georeferenced array
spanning more than one tile used to write a corrupt index: placement was
derived from each tile's GeoTransform, and non-georef tiles all carry
the default identity transform, so every ``DstRect`` landed at the
origin and ``rasterXSize`` / ``rasterYSize`` shrank to one tile. Reading
the VRT back silently returned a single tile's data.

The fix threads each tile's pixel offsets from ``_write_vrt_tiled``
through ``_build_vrt`` to ``write_vrt`` (``dst_offsets``), and makes
``write_vrt`` refuse multiple non-georeferenced sources without
explicit placement.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import GEOREF_STATUS_NONE
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()

# Reader backends: kwargs for open_geotiff. GPU entries skip when no
# CUDA device is present.
_BACKENDS = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 10}, id="dask-numpy"),
    pytest.param({"gpu": True}, id="cupy", marks=pytest.mark.skipif(
        not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 10}, id="dask-cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
]


def _materialise(da):
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "get"):
        data = data.get()
    return np.asarray(data)


_DATA = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)


@pytest.mark.parametrize("reader_kwargs", _BACKENDS)
def test_non_georef_multi_tile_round_trip(tmp_path, reader_kwargs):
    """A 24x32 non-georef array tiled at 16 spans a 2x2 tile grid; the
    round trip must return the full array with integer pixel coords."""
    vrt_path = str(tmp_path / "ng_multi_3116.vrt")
    to_geotiff(xr.DataArray(_DATA, dims=("y", "x")), vrt_path, tile_size=16)

    out = open_geotiff(vrt_path, **reader_kwargs)

    assert out.shape == _DATA.shape
    np.testing.assert_array_equal(_materialise(out), _DATA)
    np.testing.assert_array_equal(out.y.values, np.arange(24))
    np.testing.assert_array_equal(out.x.values, np.arange(32))
    assert out.attrs.get("georef_status") == GEOREF_STATUS_NONE
    assert "transform" not in out.attrs


def test_non_georef_index_places_tiles_by_pixel_offset(tmp_path):
    """The emitted XML must size the mosaic at the full array and give
    each tile its own DstRect offset."""
    vrt_path = str(tmp_path / "ng_xml_3116.vrt")
    to_geotiff(xr.DataArray(_DATA, dims=("y", "x")), vrt_path, tile_size=16)

    xml = pathlib.Path(vrt_path).read_text()
    assert 'rasterXSize="32" rasterYSize="24"' in xml
    for rect in ('<DstRect xOff="0" yOff="0" xSize="16" ySize="16"/>',
                 '<DstRect xOff="16" yOff="0" xSize="16" ySize="16"/>',
                 '<DstRect xOff="0" yOff="16" xSize="16" ySize="8"/>',
                 '<DstRect xOff="16" yOff="16" xSize="16" ySize="8"/>'):
        assert rect in xml, f"missing {rect}"


def test_non_georef_dask_backed_write_round_trip(tmp_path):
    """The dask streaming write tiles by chunk grid; placement must
    follow the chunk offsets."""
    da_mod = pytest.importorskip("dask.array")
    src = xr.DataArray(da_mod.from_array(_DATA, chunks=(16, 16)),
                       dims=("y", "x"))
    vrt_path = str(tmp_path / "ng_dask_3116.vrt")
    to_geotiff(src, vrt_path, tile_size=16)

    out = open_geotiff(vrt_path)
    assert out.shape == _DATA.shape
    np.testing.assert_array_equal(_materialise(out), _DATA)


def test_non_georef_plain_ndarray_write_round_trip(tmp_path):
    """A bare ndarray write takes the non-DataArray branch of
    ``_write_vrt_tiled`` and is just as non-georeferenced."""
    vrt_path = str(tmp_path / "ng_plain_3116.vrt")
    to_geotiff(xr.DataArray(_DATA, dims=("y", "x")).data, vrt_path,
               tile_size=16)

    out = open_geotiff(vrt_path)
    assert out.shape == _DATA.shape
    np.testing.assert_array_equal(_materialise(out), _DATA)


def test_non_georef_single_tile_still_works(tmp_path):
    """One tile needs no placement; the #2966 behaviour is unchanged."""
    small = _DATA[:10, :12]
    vrt_path = str(tmp_path / "ng_single_3116.vrt")
    to_geotiff(xr.DataArray(small, dims=("y", "x")), vrt_path, tile_size=16)

    out = open_geotiff(vrt_path)
    assert out.shape == small.shape
    np.testing.assert_array_equal(_materialise(out), small)
    assert out.attrs.get("georef_status") == GEOREF_STATUS_NONE


def test_georef_multi_tile_placement_unchanged(tmp_path):
    """Georeferenced tiles keep placing via their GeoTransform; the
    placement refactor must not move them."""
    h, w = 24, 32
    src = xr.DataArray(
        _DATA, dims=("y", "x"),
        coords={"y": 4000.0 - 5.0 * (np.arange(h) + 0.5),
                "x": 100.0 + 5.0 * (np.arange(w) + 0.5)},
        attrs={"crs": 32633,
               "transform": (5.0, 0.0, 100.0, 0.0, -5.0, 4000.0)})
    vrt_path = str(tmp_path / "geo_multi_3116.vrt")
    to_geotiff(src, vrt_path, tile_size=16)

    out = open_geotiff(vrt_path)
    assert out.shape == (h, w)
    np.testing.assert_array_equal(_materialise(out), _DATA)
    np.testing.assert_allclose(out.y.values, src.y.values)
    np.testing.assert_allclose(out.x.values, src.x.values)
    assert out.attrs["transform"] == (5.0, 0.0, 100.0, 0.0, -5.0, 4000.0)


# ---------------------------------------------------------------------------
# write_vrt-level contract for dst_offsets
# ---------------------------------------------------------------------------


def _write_plain_tile(tmp_path, name, arr):
    path = str(tmp_path / name)
    to_geotiff(xr.DataArray(arr, dims=("y", "x")), path)
    return path


def _write_georef_tile(tmp_path, name, arr):
    h, w = arr.shape
    path = str(tmp_path / name)
    to_geotiff(
        xr.DataArray(
            arr, dims=("y", "x"),
            coords={"y": 100.0 - (np.arange(h) + 0.5),
                    "x": np.arange(w) + 0.5},
            attrs={"crs": 4326,
                   "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 100.0)}),
        path)
    return path


def test_write_vrt_multiple_non_georef_without_offsets_raises(tmp_path):
    a = _write_plain_tile(tmp_path, "a_3116.tif", _DATA[:8, :8])
    b = _write_plain_tile(tmp_path, "b_3116.tif", _DATA[:8, 8:16])
    with pytest.raises(ValueError, match="dst_offsets"):
        _write_vrt_internal(str(tmp_path / "amb_3116.vrt"), [a, b])


def test_write_vrt_explicit_offsets_place_non_georef_sources(tmp_path):
    a = _write_plain_tile(tmp_path, "left_3116.tif", _DATA[:8, :8])
    b = _write_plain_tile(tmp_path, "right_3116.tif", _DATA[:8, 8:16])
    vrt_path = _write_vrt_internal(
        str(tmp_path / "placed_3116.vrt"), [a, b],
        dst_offsets=[(0, 0), (8, 0)])

    out = open_geotiff(vrt_path)
    assert out.shape == (8, 16)
    np.testing.assert_array_equal(_materialise(out), _DATA[:8, :16])


def test_write_vrt_dst_offsets_with_georef_source_raises(tmp_path):
    a = _write_georef_tile(tmp_path, "geo_a_3116.tif", _DATA[:8, :8])
    with pytest.raises(ValueError, match="non-georeferenced"):
        _write_vrt_internal(str(tmp_path / "geo_off_3116.vrt"), [a],
                            dst_offsets=[(0, 0)])


def test_write_vrt_dst_offsets_length_mismatch_raises(tmp_path):
    a = _write_plain_tile(tmp_path, "len_a_3116.tif", _DATA[:8, :8])
    b = _write_plain_tile(tmp_path, "len_b_3116.tif", _DATA[:8, 8:16])
    with pytest.raises(ValueError, match="entries"):
        _write_vrt_internal(str(tmp_path / "len_3116.vrt"), [a, b],
                            dst_offsets=[(0, 0)])


@pytest.mark.parametrize("bad", [
    (0,),                # wrong arity
    (-1, 0),             # negative
    (True, 0),           # bool masquerading as int
    (0.0, 0),            # float
    "00",                # not a pair at all
])
def test_write_vrt_dst_offsets_bad_pair_raises(tmp_path, bad):
    a = _write_plain_tile(tmp_path, "bad_a_3116.tif", _DATA[:8, :8])
    with pytest.raises(ValueError, match="dst_offsets"):
        _write_vrt_internal(str(tmp_path / "bad_3116.vrt"), [a],
                            dst_offsets=[bad])
