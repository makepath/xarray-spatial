"""GPU follow-up to PR #1521 (TIFF Orientation tag on read).

The CPU reader applies the Orientation tag (274) post-decode so pixel
(0, 0) is always the visual top-left. The GPU read path used to skip
this remap, so reads of any file with orientation != 1 returned a
different pixel buffer than the CPU reader (#1540).
"""
from __future__ import annotations

import importlib.util
import warnings as _warnings

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


_ORIENTATIONS = [1, 2, 3, 4, 5, 6, 7, 8]


def _expected_for_orientation(stored, orientation):
    """Spec-defined remap so tests don't depend on the production helper."""
    if orientation == 1:
        return stored
    if orientation == 2:
        return stored[:, ::-1]
    if orientation == 3:
        return stored[::-1, ::-1]
    if orientation == 4:
        return stored[::-1, :]
    if orientation == 5:
        return stored.T if stored.ndim == 2 else stored.transpose(1, 0, 2)
    if orientation == 6:
        return (stored.T[:, ::-1] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[:, ::-1])
    if orientation == 7:
        return (stored.T[::-1, ::-1] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[::-1, ::-1])
    if orientation == 8:
        return (stored.T[::-1, :] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[::-1, :])
    raise AssertionError(orientation)


def _write_tiled(path, arr, orientation, tile=(16, 16), extra=None):
    """Write *arr* tiled with the requested Orientation tag."""
    extras = [(274, 'H', 1, orientation, True)]
    if extra:
        extras.extend(extra)
    tifffile.imwrite(str(path), arr, tile=tile, extratags=extras)


def _write_stripped(path, arr, orientation, extra=None):
    """Write *arr* stripped (no tile=) with the requested Orientation tag."""
    extras = [(274, 'H', 1, orientation, True)]
    if extra:
        extras.extend(extra)
    tifffile.imwrite(str(path), arr, extratags=extras)


@_gpu_only
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_gpu_tiled_matches_cpu(tmp_path, orientation):
    """Tiled GPU read of every orientation matches the spec-remapped buffer."""
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_{orientation}.tif"
    _write_tiled(path, arr, orientation)

    da = read_geotiff_gpu(str(path))
    expected = _expected_for_orientation(arr, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_gpu_stripped_matches_cpu(tmp_path, orientation):
    """Stripped GPU read also applies orientation (via CPU fallback path)."""
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / f"gpu_strip_orient_{orientation}.tif"
    _write_stripped(path, arr, orientation)

    da = read_geotiff_gpu(str(path))
    expected = _expected_for_orientation(arr, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@pytest.mark.parametrize("orientation", _ORIENTATIONS)
def test_gpu_3band_tiled_matches_cpu(tmp_path, orientation):
    """3-band tiled (planar=2) read also applies orientation per band."""
    from xrspatial.geotiff import read_geotiff_gpu

    rgb = np.arange(3 * 16 * 16, dtype=np.uint8).reshape(3, 16, 16)
    path = tmp_path / f"gpu_rgb_orient_{orientation}.tif"
    tifffile.imwrite(
        str(path), rgb, photometric='rgb', tile=(16, 16),
        extratags=[(274, 'H', 1, orientation, True)],
    )

    da = read_geotiff_gpu(str(path))
    # tifffile writes (bands, h, w) -> on disk planar=2; reader returns (y, x, band)
    stored = np.transpose(rgb, (1, 2, 0))
    expected = _expected_for_orientation(stored, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@pytest.mark.parametrize("orientation", [2, 3, 4])
def test_gpu_orient_2_3_4_coords_track_pixel_flip(tmp_path, orientation):
    """For mirror-flip orientations, GPU coord array also flips."""
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_geo_{orientation}.tif"
    _write_tiled(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        da = read_geotiff_gpu(str(path))

    # Pixel (0, 0) of the original file is value 0 at (x=100.5, y=49.5).
    # Pixel (0, 15) is value 15 at (x=115.5, y=49.5).
    # Pixel (15, 0) is value 240 at (x=100.5, y=34.5).
    targets = [
        (100.5, 49.5, 0),
        (115.5, 49.5, 15),
        (100.5, 34.5, 240),
        (115.5, 34.5, 255),
    ]
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f"orient={orientation}: GPU sel(x={x}, y={y})={got}, "
            f"expected {expected}"
        )


@_gpu_only
def test_gpu_default_orientation_unchanged(tmp_path):
    """Files without Orientation tag still decode unchanged on GPU."""
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "gpu_no_orient.tif"
    tifffile.imwrite(str(path), arr, tile=(16, 16))

    da = read_geotiff_gpu(str(path))
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_raise_on_georef(tmp_path, orientation):
    """GPU reader refuses georef'd files with axis-swap orientations.

    Mirrors the CPU behaviour added for issue #1765. ``read_geotiff_gpu``
    used to warn and return silently wrong x/y coords for these cases.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_georef_raise_1765_{orientation}.tif"
    _write_tiled(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        read_geotiff_gpu(str(path))


@_gpu_only
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_transform_only_raises(tmp_path, orientation):
    """``has_georef`` without CRS still triggers the raise on GPU.

    Files carrying ModelPixelScale + ModelTiepoint but no
    GeoKeyDirectory have ``has_georef=True``/``crs_epsg=None``. The
    pixel-size swap alone misses the per-orientation origin shift, so
    refusing is the honest contract regardless of CRS tagging.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_transform_only_1765_{orientation}.tif"
    _write_tiled(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        read_geotiff_gpu(str(path))


@_gpu_only
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_no_georef_still_swaps(tmp_path, orientation):
    """Without any geo tags, GPU orientation 5-8 still swaps axes.

    Regression guard for the #1765 GPU fix: refusing must be scoped to
    georeferenced files, not every Orientation 5-8 read.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_no_geo_1765_{orientation}.tif"
    _write_tiled(path, arr, orientation)

    da = read_geotiff_gpu(str(path))
    expected = _expected_for_orientation(arr, orientation)
    assert da.data.shape == expected.shape
    np.testing.assert_array_equal(da.data.get(), expected)
