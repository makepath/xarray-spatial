"""VRT backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 6 of the corpus plan. The VRT path in ``xrspatial.geotiff``
is reached when ``open_geotiff`` sees a ``.vrt`` source path; it
delegates to ``read_vrt`` which parses the GDAL VRT XML and stitches
the listed source GeoTIFFs back into a single DataArray.

This module synthesises a two-source horizontal mosaic from one corpus
fixture: the source ``.tif`` is rasterio-copied twice into a temp
directory with the second copy's origin shifted east by one image
width, then ``write_vrt`` builds the VRT XML, and ``open_geotiff``
reads it back. The oracle reads the same VRT through rasterio so any
divergence is in xrspatial's VRT plumbing, not in the mosaic geometry.

A separate cell uses the COG fixture as the source. Its transform is
the manifest default (``[0.001, 0, -120, 0, -0.001, 45]``); the
right-half copy is shifted by ``+pixel_width * width`` so the two
copies do not overlap, which is what ``write_vrt`` expects for a clean
mosaic. The expected mosaic has shape ``(H, 2 * W)``.

VRT-specific gaps -- if any surface -- go in ``_VRT_SKIPS``. The
shared codec / attrs parity gaps from the eager / dask / GPU modules
do not all apply here because the VRT cells use only the source
fixtures named below; the parametrize set is intentionally narrow.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("yaml")
rasterio = pytest.importorskip("rasterio")

from xrspatial.geotiff import open_geotiff, write_vrt  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import compare_to_oracle  # noqa: E402

FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


# Source fixtures the VRT cells use. These are chosen because their
# default transform and dtype combine cleanly under a horizontal
# concat: integer dtype keeps comparison bit-exact, default
# transform is well-defined, no codec or attrs surprises.
_VRT_SOURCE_IDS = ["dtype_uint8", "dtype_uint16"]


def _build_two_source_vrt(
    src_path: pathlib.Path, tmp_dir: pathlib.Path
) -> tuple[pathlib.Path, np.ndarray]:
    """Materialise two copies of ``src_path`` side-by-side and a VRT.

    Returns the ``.vrt`` path and the expected mosaiced numpy array.
    """
    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        _height, width = data.shape
        t = src.transform
    pw = float(t.a)
    ox = float(t.c)
    oy = float(t.f)
    ph = float(t.e)

    left = tmp_dir / "left.tif"
    right = tmp_dir / "right.tif"
    profile_left = profile.copy()
    profile_left["transform"] = rasterio.transform.Affine(
        pw, 0.0, ox, 0.0, ph, oy
    )
    profile_right = profile.copy()
    profile_right["transform"] = rasterio.transform.Affine(
        pw, 0.0, ox + pw * width, 0.0, ph, oy
    )
    with rasterio.open(str(left), "w", **profile_left) as dst:
        dst.write(data, 1)
    with rasterio.open(str(right), "w", **profile_right) as dst:
        dst.write(data, 1)

    vrt_path = tmp_dir / "mosaic.vrt"
    write_vrt(str(vrt_path), [str(left), str(right)])

    expected = np.concatenate([data, data], axis=1)
    return vrt_path, expected


@pytest.fixture(params=_VRT_SOURCE_IDS, ids=_VRT_SOURCE_IDS)
def vrt_source_id(request) -> str:
    return request.param


def test_vrt_two_source_horizontal_mosaic(
    vrt_source_id: str, tmp_path: pathlib.Path
) -> None:
    """``open_geotiff(vrt_path)`` matches a rasterio read of the same VRT.

    The two source copies are horizontally tiled. The VRT writer wires
    up the geotransform; the test compares the xrspatial read against
    both the rasterio read of the VRT and a manual ``np.concatenate``
    of the source pixels.
    """
    src_path = FIXTURES_DIR / f"{vrt_source_id}.tif"
    if not src_path.exists():
        pytest.skip(
            f"source fixture {vrt_source_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    vrt_path, expected = _build_two_source_vrt(src_path, tmp_path)

    # Independent sanity: the rasterio read of the VRT really does
    # produce the horizontally concatenated mosaic. If the VRT writer
    # ever drifts, this assertion catches it before the oracle does --
    # which matters because the oracle also reads through rasterio and
    # would silently agree with itself on a busted VRT.
    with rasterio.open(vrt_path) as src:
        ref = src.read(1)
    assert ref.shape == expected.shape, (
        f"rasterio VRT read shape mismatch: ref={ref.shape}, "
        f"expected={expected.shape}"
    )
    np.testing.assert_array_equal(ref, expected)

    # Parity check via the shared oracle. Compares pixels, dtype,
    # transform, CRS, and nodata against the rasterio reference read of
    # the same VRT, the same as every other phase 3 backend module.
    candidate = open_geotiff(str(vrt_path))
    compare_to_oracle(vrt_path, candidate)


def test_vrt_source_ids_exist_in_manifest() -> None:
    """Every id in ``_VRT_SOURCE_IDS`` must be a real manifest entry."""
    manifest = generate.load_manifest()
    entries = generate.validate(manifest)
    manifest_ids = {e["id"] for e in entries}
    stale = set(_VRT_SOURCE_IDS) - manifest_ids
    assert not stale, (
        f"VRT source list references unknown fixture ids: {sorted(stale)}"
    )
