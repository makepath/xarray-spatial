"""``to_geotiff`` refuses to silently drop ``attrs['rotated_affine']``.

Issue #2216. The reader exposes the rotated 6-tuple on
``attrs['rotated_affine']`` when called with ``allow_rotated=True``
(issue #2129). The writer does not emit a ``ModelTransformationTag``
(tracked in issue #2115), so the round-trip used to write an
identity-affine output without warning. This file pins the fail-closed
contract:

* writing a rotated-affine raster without the opt-in raises
  ``ValueError`` naming the attr;
* ``drop_rotation=True`` lets the write proceed and the round-trip
  output carries no ``rotated_affine`` attr;
* writes of plain rasters with no rotated attr are unchanged;
* a read-then-write-then-read cycle on a rotated file requires the
  opt-in to succeed.

The tests target the eager ``to_geotiff`` entry point and reuse the
synthetic rotated-TIFF helper from ``test_rotated_affine_attr_2129.py``
so the two suites stay in lockstep on what the reader produces.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writers.eager import _write_vrt_tiled


_ROTATED_TUPLE = (8.66, -5.0, 100.0, 5.0, 8.66, 200.0)


def _write_rotated_tiff(path, arr, *, epsg=None):
    """Write a synthetic rotated GeoTIFF (30-degree rotation)."""
    tifffile = pytest.importorskip("tifffile")
    cos30 = 0.8660254037844387
    sin30 = 0.5
    m = (
        10.0 * cos30, -10.0 * sin30, 0.0, 100.0,
        10.0 * sin30,  10.0 * cos30, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [(34264, 12, 16, m, False)]
    if epsg is not None:
        geo_key_directory = (
            1, 1, 0, 1,
            2048, 0, 1, int(epsg),
        )
        extratags.append((34735, 3, 8, geo_key_directory, False))
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        extratags=extratags,
    )
    return m


def _rotated_dataarray():
    """Build a 2D DataArray that mimics ``open_geotiff(..., allow_rotated=True)``.

    Avoids depending on the synthetic-TIFF helper for the parsing-side
    tests so the rejection logic is exercised on the boundary regardless
    of whether the synthetic file produced the exact attrs the reader
    would emit.
    """
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(4), 'x': np.arange(5)},
        attrs={'rotated_affine': _ROTATED_TUPLE},
    )


# ---------------------------------------------------------------------------
# Default fail-closed gate.
# ---------------------------------------------------------------------------


def test_to_geotiff_rejects_rotated_affine_without_opt_in(tmp_path):
    da = _rotated_dataarray()
    out = tmp_path / "tmp_2216_reject.tif"

    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    # The error names the opt-in so the caller learns the flag.
    with pytest.raises(ValueError, match="drop_rotation=True"):
        to_geotiff(da, str(out))

    # Nothing got written.
    assert not out.exists()


def test_to_geotiff_error_message_points_at_issue(tmp_path):
    """The rejection message references issue #2216 so a grep
    ties back to this PR. The check is on the issue number, not on
    surrounding phrasing, so the wording can evolve without breaking
    the test."""
    da = _rotated_dataarray()
    out = tmp_path / "tmp_2216_issue_ref.tif"

    with pytest.raises(ValueError, match="#2216"):
        to_geotiff(da, str(out))


# ---------------------------------------------------------------------------
# Opt-in path.
# ---------------------------------------------------------------------------


def test_to_geotiff_drop_rotation_writes_axis_aligned_file(tmp_path):
    """``drop_rotation=True`` lets the write proceed and the output is a
    plain axis-aligned (non-rotated) TIFF -- the round-trip reader sees
    no ``rotated_affine`` attr."""
    da = _rotated_dataarray()
    out = tmp_path / "tmp_2216_drop.tif"

    to_geotiff(da, str(out), drop_rotation=True)

    # Sanity: the file exists and re-opens.
    assert out.exists()
    da2 = open_geotiff(str(out))

    # The rotated attr is gone on the round-trip (the writer has no
    # ModelTransformationTag emit path, so the on-disk file is
    # axis-aligned; the reader's normal path therefore sees no
    # rotated-tag and emits no rotated_affine attr).
    assert 'rotated_affine' not in da2.attrs


def test_to_geotiff_drop_rotation_preserves_pixel_values(tmp_path):
    """The opt-in only drops the rotated *georeferencing*; the pixel
    grid itself round-trips unchanged."""
    da = _rotated_dataarray()
    out = tmp_path / "tmp_2216_pixels.tif"

    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))

    np.testing.assert_array_equal(
        np.asarray(da2.data, dtype=np.float32),
        np.asarray(da.data, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Non-rotated raster baseline.
# ---------------------------------------------------------------------------


def test_to_geotiff_normal_raster_unchanged(tmp_path):
    """A DataArray with no ``rotated_affine`` attr writes the same way
    it always did -- the new gate must not change behaviour for the
    common path."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    da = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(3), 'x': np.arange(4)},
        attrs={},
    )
    out = tmp_path / "tmp_2216_normal.tif"

    to_geotiff(da, str(out))
    assert out.exists()

    # And explicitly setting ``drop_rotation=True`` on a non-rotated
    # input is a no-op; the kwarg only affects the rotated path.
    out2 = tmp_path / "tmp_2216_normal_optin.tif"
    to_geotiff(da, str(out2), drop_rotation=True)
    assert out2.exists()


def test_to_geotiff_rotated_affine_none_does_not_trigger(tmp_path):
    """A literal ``attrs['rotated_affine'] = None`` does not trigger
    the gate. The check is on the attr's truthiness so a future read
    path that pre-allocates the key with ``None`` does not break the
    common write path."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    da = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': np.arange(2), 'x': np.arange(3)},
        attrs={'rotated_affine': None},
    )
    out = tmp_path / "tmp_2216_none.tif"

    to_geotiff(da, str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# End-to-end round-trip: opening a real rotated TIFF and writing back.
# ---------------------------------------------------------------------------


def test_round_trip_rotated_tiff_requires_opt_in(tmp_path):
    """Read a rotated TIFF with ``allow_rotated=True``, then attempt to
    write it back. Without the opt-in the write raises; with the opt-in
    it succeeds and the round-trip output is axis-aligned."""
    arr = np.arange(20, dtype='<u2').reshape(4, 5)
    src = tmp_path / "tmp_2216_round_trip_src.tif"
    _write_rotated_tiff(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True)
    assert 'rotated_affine' in da.attrs

    # Fail closed without opt-in.
    out = tmp_path / "tmp_2216_round_trip_out.tif"
    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    # Opt-in succeeds; the read-back file is plain axis-aligned.
    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))
    assert 'rotated_affine' not in da2.attrs


def test_round_trip_dask_rotated_tiff_requires_opt_in(tmp_path):
    """Same contract on the dask read path -- ``allow_rotated=True``
    with ``chunks=`` lands the attr on the lazy DataArray, and the
    writer's gate fires before any tile streaming begins."""
    arr = np.arange(40, dtype='<u2').reshape(5, 8)
    src = tmp_path / "tmp_2216_round_trip_dask_src.tif"
    _write_rotated_tiff(str(src), arr, epsg=4326)

    da = open_geotiff(str(src), allow_rotated=True, chunks=4)
    assert 'rotated_affine' in da.attrs

    out = tmp_path / "tmp_2216_round_trip_dask_out.tif"
    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(out))

    to_geotiff(da, str(out), drop_rotation=True)
    da2 = open_geotiff(str(out))
    assert 'rotated_affine' not in da2.attrs


# ---------------------------------------------------------------------------
# VRT path parity.
# ---------------------------------------------------------------------------


def test_to_geotiff_vrt_path_rejects_rotated_affine(tmp_path):
    """The VRT branch of ``to_geotiff`` runs the same gate so the
    silent-loss surface is uniform across single-file and tiled-VRT
    outputs."""
    da = _rotated_dataarray()
    vrt_out = tmp_path / "tmp_2216_vrt.vrt"

    with pytest.raises(ValueError, match="rotated_affine"):
        to_geotiff(da, str(vrt_out))

    # Underlying tiles directory should not exist on the failed write.
    tiles_dir = tmp_path / "tmp_2216_vrt_tiles"
    assert not tiles_dir.exists() or not list(tiles_dir.iterdir())


def test_write_vrt_tiled_direct_call_rejects_rotated_affine(tmp_path):
    """A direct ``_write_vrt_tiled`` call bypasses the ``to_geotiff``
    wrapper but must still refuse rotated inputs. Without the explicit
    gate inside ``_write_vrt_tiled`` the helper would silently produce
    identity-affine tiles."""
    da = _rotated_dataarray()
    vrt_out = tmp_path / "tmp_2216_vrt_direct.vrt"

    with pytest.raises(ValueError, match="rotated_affine"):
        _write_vrt_tiled(da, str(vrt_out))

    # The error names the function actually running the check, not the
    # public wrapper, so a direct caller of the private helper learns
    # which entry point fired (review nit on #2216).
    with pytest.raises(ValueError, match="_write_vrt_tiled"):
        _write_vrt_tiled(da, str(vrt_out))
