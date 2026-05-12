"""Regression tests for issue #1657: dangerous tag leakage through extra_tags.

Before the fix, reading an overview level (NewSubfileType=1) or any TIFF
with a SubIFDs entry leaked those tags into ``attrs['extra_tags']`` because
they were not in ``_MANAGED_TAGS``. Writing the DataArray back via
``to_geotiff`` or ``write_geotiff_gpu`` then re-emitted them on the output
IFD, producing:

* A primary IFD wrongly marked as a reduced-resolution overview
  (``NewSubfileType=1``), so GDAL / rasterio skip it when picking the
  primary image.
* Stale absolute byte offsets in ``SubIFDs`` that point into the new file's
  pixel data, crashing readers that follow the chain.

The fix adds both tags to ``_MANAGED_TAGS`` (read-side filter) and to
``_DANGEROUS_EXTRA_TAG_IDS`` in ``_writer.py`` (write-side belt-and-braces
guard so caller-supplied ``attrs['extra_tags']`` still produces a clean
file).

These tests pin the contract on every available backend.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

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


def _make_cog(path) -> None:
    """Write a small COG with overviews so each backend can read an overview.

    Uses tile_size=64 + explicit overview_levels so the small test fixture
    actually produces a multi-IFD pyramid (auto-overview halves until the
    smallest level fits in one tile, so 256x256 with the default 256-tile
    size yields zero overviews).
    """
    import xarray as xr
    da = xr.DataArray(
        np.arange(512 * 512, dtype=np.float32).reshape(512, 512),
        dims=['y', 'x'],
        coords={'y': np.arange(512) * -0.5 + 10.0,
                'x': np.arange(512) * 0.5 - 10.0},
        attrs={'crs': 4326},
    )
    to_geotiff(da, str(path), cog=True,
               tile_size=64, overview_levels=[2, 4])


# ---------------------------------------------------------------------------
# Read side: overview reads must not leak NewSubfileType into attrs
# ---------------------------------------------------------------------------

def test_overview_read_does_not_leak_newsubfiletype_numpy(tmp_path):
    """Reading an overview level must not surface NewSubfileType in attrs."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked into attrs['extra_tags']: {extra}"
    )


def test_overview_read_does_not_leak_newsubfiletype_dask(tmp_path):
    """Same contract for the dask backend."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, chunks=32)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under dask: {extra}"
    )


@_gpu_only
def test_overview_read_does_not_leak_newsubfiletype_cupy(tmp_path):
    """Same contract for the cupy backend."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, gpu=True)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under cupy: {extra}"
    )


@_gpu_only
def test_overview_read_does_not_leak_newsubfiletype_dask_cupy(tmp_path):
    """Same contract for the dask+cupy backend."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(
        str(cog_path), overview_level=1, gpu=True, chunks=32)
    extra = ov.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 254 for t in extra), (
        f"NewSubfileType (tag 254) leaked under dask+cupy: {extra}"
    )


# ---------------------------------------------------------------------------
# Read side: SubIFDs must not leak from level-0 IFDs that carry one
# ---------------------------------------------------------------------------

def test_subifds_does_not_leak_into_attrs(tmp_path):
    """tifffile writes SubIFDs by default on multi-page TIFFs.

    Anything carrying tag 330 must not surface in ``attrs['extra_tags']``
    because the byte offsets are file-absolute and cannot be replayed
    into a rewritten file.
    """
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / 'subifd.tif'
    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(data, tile=(32, 32), subifds=1)
        tw.write(data[::2, ::2], tile=(32, 32), subfiletype=1)

    da = open_geotiff(str(path))
    extra = da.attrs.get('extra_tags')
    assert extra is None or not any(t[0] == 330 for t in extra), (
        f"SubIFDs (tag 330) leaked into attrs['extra_tags']: {extra}"
    )


# ---------------------------------------------------------------------------
# Round-trip: overview read -> write must produce a valid primary IFD
# ---------------------------------------------------------------------------

def _read_subfile_type(path) -> int | None:
    """Return the NewSubfileType value on page 0 of a TIFF, or None if absent."""
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        tag = page.tags.get('NewSubfileType')
        return None if tag is None else int(tag.value)


def test_overview_roundtrip_primary_ifd_clean_numpy(tmp_path):
    """Round-tripping an overview must not mark the output as an overview."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1)
    out = tmp_path / 'out_numpy.tif'
    to_geotiff(ov, str(out))
    sft = _read_subfile_type(out)
    assert sft in (None, 0), (
        f"Round-tripped overview produced NewSubfileType={sft} on the "
        f"primary IFD (expected None or 0)."
    )


def test_overview_roundtrip_primary_ifd_clean_dask(tmp_path):
    """Same contract for the dask read path."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, chunks=32)
    out = tmp_path / 'out_dask.tif'
    to_geotiff(ov, str(out))
    sft = _read_subfile_type(out)
    assert sft in (None, 0), (
        f"Dask round-tripped overview produced NewSubfileType={sft}."
    )


@_gpu_only
def test_overview_roundtrip_primary_ifd_clean_cupy(tmp_path):
    """Same contract for the cupy read + write path."""
    cog_path = tmp_path / 'cog.tif'
    _make_cog(cog_path)
    ov = open_geotiff(str(cog_path), overview_level=1, gpu=True)
    out = tmp_path / 'out_cupy.tif'
    to_geotiff(ov, str(out))  # auto-dispatches to GPU writer
    sft = _read_subfile_type(out)
    assert sft in (None, 0), (
        f"Cupy round-tripped overview produced NewSubfileType={sft}."
    )


# ---------------------------------------------------------------------------
# Writer belt-and-braces: caller-supplied dangerous extra_tags get filtered
# ---------------------------------------------------------------------------

def test_writer_filters_caller_supplied_newsubfiletype(tmp_path):
    """A caller passing extra_tags with NewSubfileType=1 still gets a
    clean primary IFD on output."""
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            # Tag 254 with LONG (type id 4) count 1 value 1 -- format
            # produced by the read path before the fix.
            'extra_tags': [(254, 4, 1, 1)],
        },
    )
    out = tmp_path / 'with_dangerous_extra_tag.tif'
    to_geotiff(da, str(out))
    sft = _read_subfile_type(out)
    assert sft in (None, 0), (
        f"Writer accepted dangerous extra_tags[254]={sft}, expected None/0."
    )


def test_writer_filters_caller_supplied_subifds(tmp_path):
    """A caller passing extra_tags with SubIFDs offsets must not emit
    those stale offsets onto the output file."""
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            # Garbage offsets that would crash readers if emitted.
            'extra_tags': [(330, 4, 2, (999999, 888888))],
        },
    )
    out = tmp_path / 'with_subifds.tif'
    to_geotiff(da, str(out))
    with tifffile.TiffFile(str(out)) as tf:
        sub = tf.pages[0].tags.get('SubIFDs')
        assert sub is None, (
            f"Writer emitted SubIFDs={sub.value}, should have filtered it."
        )


def test_writer_keeps_benign_extra_tags(tmp_path):
    """The filter must only drop the dangerous IDs (254, 330) and let
    other extra_tags entries through. Use Software (305) -- ASCII, no
    embedded offsets -- as a benign canary."""
    import xarray as xr
    da = xr.DataArray(
        np.zeros((32, 32), dtype=np.float32),
        dims=['y', 'x'],
        coords={'y': np.arange(32) * -0.5 + 10.0,
                'x': np.arange(32) * 0.5 - 10.0},
        attrs={
            'crs': 4326,
            'extra_tags': [
                (254, 4, 1, 1),     # dangerous, must be filtered
                (305, 2, 12, 'tifffile.py'),  # benign, must survive
            ],
        },
    )
    out = tmp_path / 'mixed_extra_tags.tif'
    to_geotiff(da, str(out))
    with tifffile.TiffFile(str(out)) as tf:
        page = tf.pages[0]
        assert page.tags.get('NewSubfileType') is None
        software = page.tags.get('Software')
        assert software is not None, (
            "Benign extra_tag (305 Software) was filtered too -- "
            "filter is too aggressive."
        )
        assert 'tifffile' in str(software.value)
