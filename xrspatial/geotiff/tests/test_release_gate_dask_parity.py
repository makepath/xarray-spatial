"""Release gate: dask read parity vs eager (epic #2340).

Dask reads of a local GeoTIFF must return the same pixels and the same
canonical attrs as the eager (numpy) read. This is the
``reader.local_file`` stable promise extended to the dask backend.

The release gate locks the small, deterministic case a release engineer
can run before tagging: write a known-good file, read it both eagerly
and through the dask backend, and assert the pixel-level and attrs
parity. The wide backend matrix
(``test_backend_pixel_parity_matrix_1813.py``,
``test_backend_parity_matrix.py``) exercises every codec / chunk-size /
dtype combination -- those stay the canonical parity suite. The
release-gate test is the one-shot the release notes can quote without
caveats.

Out of scope:
* GPU / cupy parity (``reader.gpu`` is ``experimental``, not stable).
* VRT lazy reads (``reader.vrt`` is ``advanced``).
* COG dask reads (covered by ``test_release_gate_cog.py`` via the
  eager reader; the dask parity for COG is part of the canonical
  parity matrix).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write


def _write_known_good(path: str) -> np.ndarray:
    """Write a small tiled GeoTIFF and return the source array."""
    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
    gt = GeoTransform(
        origin_x=500000.0,
        origin_y=4000000.0,
        pixel_width=30.0,
        pixel_height=-30.0,
    )
    write(
        arr,
        path,
        geo_transform=gt,
        crs_epsg=32610,
        compression="deflate",
        tiled=True,
        tile_size=16,
    )
    return arr


@pytest.mark.release_gate
def test_release_gate_dask_read_matches_eager_pixels(tmp_path) -> None:
    """The dask backend returns the same pixels as the eager backend."""
    path = str(tmp_path / "release_gate_dask_parity_pixels_2340.tif")
    _write_known_good(path)

    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=8)

    # The dask backend returns a lazy DataArray; materialise it once
    # so the equality check is comparing concrete numpy arrays.
    lazy_values = np.asarray(lazy.values)
    eager_values = np.asarray(eager.values)
    np.testing.assert_array_equal(
        lazy_values,
        eager_values,
        err_msg=(
            "release gate: dask backend returned different pixels than "
            "the eager backend; the release contract promises dask read "
            "parity for the local-file stable path"
        ),
    )
    assert lazy.dtype == eager.dtype, (
        f"release gate: dask backend changed dtype from {eager.dtype!r} "
        f"to {lazy.dtype!r}"
    )
    assert lazy.shape == eager.shape, (
        f"release gate: dask backend changed shape from {eager.shape!r} "
        f"to {lazy.shape!r}"
    )


@pytest.mark.release_gate
def test_release_gate_dask_read_matches_eager_attrs(tmp_path) -> None:
    """The dask backend produces the same canonical attrs as eager."""
    path = str(tmp_path / "release_gate_dask_parity_attrs_2340.tif")
    _write_known_good(path)

    eager = open_geotiff(path)
    lazy = open_geotiff(path, chunks=8)

    # The canonical attrs the release contract pins; backend-specific
    # additive attrs (chunk shape, source URI, etc.) are allowed to
    # differ between backends and are not part of this gate.
    canonical = ("crs", "transform", "georef_status")
    for key in canonical:
        assert key in eager.attrs, (
            f"release gate: eager read is missing canonical attr "
            f"{key!r}; cannot compare backends"
        )
        assert key in lazy.attrs, (
            f"release gate: dask read is missing canonical attr "
            f"{key!r}; the release contract requires backend parity on "
            "canonical attrs"
        )
        eager_v = eager.attrs[key]
        lazy_v = lazy.attrs[key]
        if key == "transform":
            assert len(eager_v) == len(lazy_v) == 6
            for a, b in zip(eager_v, lazy_v):
                assert a == pytest.approx(b, abs=1e-12, rel=1e-12), (
                    f"release gate: transform drifted across backends: "
                    f"eager={eager_v!r} lazy={lazy_v!r}"
                )
        else:
            assert eager_v == lazy_v, (
                f"release gate: ``attrs[{key!r}]`` drifted across "
                f"backends: eager={eager_v!r} lazy={lazy_v!r}"
            )


@pytest.mark.release_gate
def test_release_gate_dask_read_is_lazy(tmp_path) -> None:
    """A ``chunks=`` read produces a dask-backed DataArray.

    Without this assertion, a regression that silently materialised
    the dask path into numpy could pass the pixel-parity test above
    without anyone noticing. The dask backend's defining property is
    laziness; pin it.
    """
    pytest.importorskip("dask")
    import dask.array as da_mod

    path = str(tmp_path / "release_gate_dask_parity_lazy_2340.tif")
    _write_known_good(path)

    lazy = open_geotiff(path, chunks=8)
    assert isinstance(lazy.data, da_mod.Array), (
        f"release gate: chunks= read returned a non-dask array of type "
        f"{type(lazy.data).__name__}; the release contract promises a "
        "dask-backed DataArray when chunks= is set"
    )
