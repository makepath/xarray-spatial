"""Regression test for issue #1923.

``_read_vrt_chunked`` previously passed the parsed ``VRTDataset`` as a
plain kwarg to each per-chunk ``dask.delayed`` call, so dask embedded
the full source list (filenames, src/dst rects, per-source nodata) into
every task graph entry. With N sources and M chunks the graph held
``N * M`` copies of the same metadata; a 1000-source VRT split into
1000 chunks built a ~57 MB driver graph and reserialised that payload
1000 times under distributed/process schedulers.

The fix wraps the dataset in ``dask.delayed(vrt, pure=True)`` once before
the per-chunk loop and passes that single shared reference into every
task, mirroring the ``http_meta_key`` pattern in ``_backends/dask.py``
and the ``meta_key`` pattern in ``_backends/gpu.py``.

These tests assert the shared-input behaviour structurally so a
regression would either flip the structural check or pump the graph
size back over the cap.
"""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_vrt, to_geotiff


def _make_tile_vrt(tmp_path, n_tiles_per_side=4):
    """Build a small multi-source VRT for testing the chunked path.

    Each source is a 64x64 tile written to a temp directory; the VRT
    stitches them into a ``(64*N, 64*N)`` mosaic. ``N=4`` keeps the
    fixture cheap while still producing a multi-source VRT whose
    embedded metadata is measurable.
    """
    tile_dir = os.path.join(tmp_path, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    tile_size = 64
    sources = []
    for r in range(n_tiles_per_side):
        for c in range(n_tiles_per_side):
            arr = np.full(
                (tile_size, tile_size),
                fill_value=r * n_tiles_per_side + c,
                dtype=np.float32,
            )
            ox = c * tile_size
            oy = -(r * tile_size)
            # GeoTransform tuple ordering matches rasterio: pixel_width
            # first, then column rotation (forced to zero), then origin_x,
            # then row rotation (forced to zero), then pixel_height,
            # then origin_y.
            da = xr.DataArray(
                arr,
                dims=["y", "x"],
                attrs={"transform": (1.0, 0.0, ox, 0.0, -1.0, oy)},
            )
            path = os.path.join(tile_dir, f"tile_{r}_{c}.tif")
            to_geotiff(da, path, compression="deflate", tiled=True,
                       tile_size=64)
            sources.append((path, r, c, tile_size))

    vrt_path = os.path.join(tmp_path, "mosaic.vrt")
    width = n_tiles_per_side * tile_size
    height = n_tiles_per_side * tile_size
    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        '<SRS></SRS>',
        '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>',
        '<VRTRasterBand dataType="Float32" band="1">',
    ]
    for (path, r, c, ts) in sources:
        lines.extend([
            "<SimpleSource>",
            f'<SourceFilename relativeToVRT="0">{path}</SourceFilename>',
            "<SourceBand>1</SourceBand>",
            f'<SrcRect xOff="0" yOff="0" xSize="{ts}" ySize="{ts}"/>',
            f'<DstRect xOff="{c * ts}" yOff="{r * ts}" xSize="{ts}" '
            f'ySize="{ts}"/>',
            "</SimpleSource>",
        ])
    lines.extend(["</VRTRasterBand>", "</VRTDataset>"])
    with open(vrt_path, "w") as f:
        f.write("\n".join(lines))
    return vrt_path, len(sources)


def test_vrt_chunked_dataset_is_shared_graph_input(tmp_path):
    """Issue #1923: parsed VRTDataset is wrapped as a single Delayed.

    Walks each ``_vrt_chunk_read`` task's kwargs dict in the dask graph
    and verifies that ``parsed_vrt`` is NOT an inline ``VRTDataset``
    instance (the pre-fix shape). With the fix, ``parsed_vrt`` is
    routed through ``dask.delayed(vrt, pure=True)`` so each task's
    ``kwargs['parsed_vrt']`` is a graph reference (an ``Alias`` /
    ``TaskRef``-style placeholder pointing into a single shared
    ``from-value`` layer) rather than a literal embedded dataset.

    Under the in-process scheduler an embedded copy still works
    correctly because Python's pickle memo deduplicates references to
    the same in-memory object. The bug surfaces under the distributed
    / multi-process scheduler where each task pickle is serialised
    independently and the full dataset is shipped once per task -- so
    the structural shape, not the in-process pickle size, is what
    matters.
    """
    from xrspatial.geotiff._vrt import VRTDataset

    vrt_path, n_sources = _make_tile_vrt(str(tmp_path), n_tiles_per_side=4)

    # Use small chunks so the grid produces multiple tasks. 4 tiles per
    # side at 64 px each is 256x256 image; chunks=32 gives 64 chunks.
    result = read_vrt(vrt_path, chunks=32)
    graph = result.__dask_graph__()

    assert n_sources == 16, "fixture build sanity check"

    chunk_task_count = 0
    embedded_vrt_count = 0
    for layer_name, layer in graph.layers.items():
        if "_vrt_chunk_read" not in layer_name:
            continue
        for _key, task in layer.items():
            # New-style dask Task: kwargs is a dict attribute.
            kwargs = getattr(task, "kwargs", None)
            if kwargs is None:
                continue
            parsed_vrt = kwargs.get("parsed_vrt")
            if parsed_vrt is None:
                continue
            chunk_task_count += 1
            if isinstance(parsed_vrt, VRTDataset):
                embedded_vrt_count += 1

    assert chunk_task_count > 1, (
        f"fixture sanity: expected multiple chunk tasks, got "
        f"{chunk_task_count}"
    )
    assert embedded_vrt_count == 0, (
        f"#1923 regression: {embedded_vrt_count} of {chunk_task_count} "
        f"_vrt_chunk_read tasks still embed an inline VRTDataset in "
        f"kwargs['parsed_vrt']. The fix wraps the dataset in "
        f"dask.delayed(vrt, pure=True) so kwargs['parsed_vrt'] should "
        f"be a TaskRef-style graph reference, not a VRTDataset."
    )


def test_vrt_chunked_decode_unchanged_after_shared_wrap(tmp_path):
    """The shared-Delayed wrap must not change decoded pixel values."""
    vrt_path, _ = _make_tile_vrt(str(tmp_path), n_tiles_per_side=3)

    # Eager read produces the baseline.
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=32).compute()

    np.testing.assert_array_equal(np.asarray(eager), np.asarray(chunked))


def test_vrt_chunked_band_kwarg_still_validates(tmp_path):
    """Wrapping the dataset must not change band validation behaviour."""
    vrt_path, _ = _make_tile_vrt(str(tmp_path), n_tiles_per_side=2)

    with pytest.raises(ValueError):
        # The fixture is single-band; band=5 should still raise the
        # same ValueError the chunked path raised pre-fix.
        read_vrt(vrt_path, chunks=32, band=5)
