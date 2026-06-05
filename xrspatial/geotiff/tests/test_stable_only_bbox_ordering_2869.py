"""``stable_only=True`` must be enforced before ``bbox=`` does any I/O (#2869).

``open_geotiff(..., bbox=...)`` resolves the bounding box to a pixel
window by reading source geo metadata: the TIFF path reads a header (a
range GET for an HTTP / fsspec source) and the VRT path parses the VRT
XML. Before the fix, that resolution ran ahead of the stable-only gates,
so ``open_geotiff(remote, bbox=..., stable_only=True)`` triggered remote
metadata I/O and ``open_geotiff(remote_vrt, bbox=..., stable_only=True)``
parsed the VRT XML before the request was refused. The docstring on
``_validate_stable_only_remote`` requires the rejection to land "before
any range GET or decode work".

Each test makes the ordering observable by pointing at a source whose
bbox metadata read would fail with a *different* error if it ran first:

- An ``http://`` URL that does not resolve would raise a fetch /
  file-not-found error.
- A ``.vrt`` with malformed XML would raise an ``xml`` ``ParseError``.

If the gate runs first, the typed stable-only error wins. The positive
controls confirm that with ``allow_experimental_codecs=True`` the gate is
a no-op and the bbox still resolves and reads.
"""
from __future__ import annotations

import os
import uuid
from xml.etree.ElementTree import ParseError

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (RemoteStableSourcesOnlyError, VRTStableSourcesOnlyError,
                               open_geotiff, to_geotiff)
from xrspatial.geotiff.tests._helpers.tiff_builders import make_minimal_tiff

fsspec = pytest.importorskip("fsspec")

_MEMORY_URL = "memory:///stable_only_bbox_2869/sample.tif"
# A bbox inside the memory tiff's extent (-120..-120+4*0.001 lon,
# 45-4*0.001..45 lat) so the positive control resolves a real window.
_MEMORY_BBOX = (-119.9995, 44.9985, -119.9975, 44.9995)


@pytest.fixture
def memory_tiff_url():
    """Push a stable-codec GeoTIFF into fsspec memory; yield its URL."""
    payload = make_minimal_tiff(
        4, 4, np.dtype("float32"),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )
    fs = fsspec.filesystem("memory")
    with fs.open(_MEMORY_URL, "wb") as handle:
        handle.write(payload)
    try:
        yield _MEMORY_URL
    finally:
        if fs.exists(_MEMORY_URL):
            fs.rm(_MEMORY_URL)


@pytest.fixture
def malformed_vrt_path(tmp_path):
    """Write a ``.vrt`` with broken XML; yield its path.

    Parsing this for a bbox window raises an ``xml`` ``ParseError``, so a
    clean ``VRTStableSourcesOnlyError`` proves the gate ran first.
    """
    path = tmp_path / f"stable_only_bbox_2869_{uuid.uuid4().hex[:8]}.vrt"
    path.write_text('<VRTDataset rasterXSize="4" rasterYSize="4"><bad xml')
    return str(path)


@pytest.fixture
def georef_vrt_path(tmp_path):
    """Build a small EPSG:4326 georeferenced VRT mosaic; yield its path."""
    d = tmp_path / f"stable_only_bbox_2869_vrt_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    ys = np.linspace(41.0, 40.0, 256)
    xs = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": ys, "x": xs},
        attrs={"crs": 4326},
    )
    vrt_path = os.path.join(str(d), "mosaic.vrt")
    to_geotiff(raster, vrt_path, tile_size=128)
    return vrt_path


def _assert_stable_error(excinfo) -> None:
    """The rejection names the unlocking opt-in and cites the contract."""
    msg = str(excinfo.value)
    assert "stable_only" in msg or "allow_experimental_codecs" in msg, msg
    assert "release_gate_geotiff.rst" in msg or "#2342" in msg, msg


def test_http_bbox_rejected_before_metadata_read():
    """``http://`` + bbox + stable_only gates before any fetch.

    The URL does not resolve; a metadata read would raise a fetch /
    file-not-found error. A clean ``RemoteStableSourcesOnlyError`` proves
    the gate runs ahead of bbox resolution.
    """
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(
            "http://example.invalid/sample.tif",
            bbox=(-1.0, -1.0, 1.0, 1.0),
            stable_only=True,
        )
    _assert_stable_error(excinfo)


def test_fsspec_bbox_rejected_under_stable_only(memory_tiff_url):
    """An fsspec source + bbox + stable_only is rejected."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(memory_tiff_url, bbox=_MEMORY_BBOX, stable_only=True)
    _assert_stable_error(excinfo)


def test_fsspec_dask_bbox_rejected_under_stable_only(memory_tiff_url):
    """The dask dispatch path gates the fsspec + bbox combination too."""
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(
            memory_tiff_url, bbox=_MEMORY_BBOX, stable_only=True, chunks=2,
        )
    _assert_stable_error(excinfo)


def test_fsspec_bbox_allowed_with_experimental_optin(memory_tiff_url):
    """``allow_experimental_codecs=True`` unlocks the read; bbox resolves."""
    result = open_geotiff(
        memory_tiff_url, bbox=_MEMORY_BBOX, stable_only=True,
        allow_experimental_codecs=True,
    )
    assert result.ndim == 2
    assert 0 < result.shape[0] <= 4
    assert 0 < result.shape[1] <= 4


def test_vrt_bbox_rejected_before_xml_parse(malformed_vrt_path):
    """A ``.vrt`` + bbox + stable_only gates before the VRT XML is parsed.

    The XML is malformed; parsing it for a bbox window would raise an
    ``xml`` ``ParseError``. A clean ``VRTStableSourcesOnlyError`` proves
    the gate runs ahead of ``_vrt_bbox_to_window``.
    """
    with pytest.raises(VRTStableSourcesOnlyError) as excinfo:
        open_geotiff(
            malformed_vrt_path,
            bbox=(-1.0, -1.0, 1.0, 1.0),
            stable_only=True,
        )
    _assert_stable_error(excinfo)


def test_malformed_vrt_bbox_still_parses_without_stable_only(
        malformed_vrt_path):
    """Negative control: without the gate the malformed VRT still parses.

    Confirms the malformed-XML fixture is the thing the gate short-circuits
    -- ``stable_only=False`` reaches the parse and raises ``ParseError``.
    """
    with pytest.raises(ParseError):
        open_geotiff(malformed_vrt_path, bbox=(-1.0, -1.0, 1.0, 1.0))


def test_stable_only_wins_over_window_bbox_conflict(memory_tiff_url):
    """The tier gate is refused ahead of the ``window=``/``bbox=`` conflict.

    Passing both ``window=`` and ``bbox=`` is mutually exclusive, but the
    stable-only gate now runs first, so a remote source under
    ``stable_only=True`` reports the tier rejection rather than the kwarg
    conflict. Pins that precedence.
    """
    with pytest.raises(RemoteStableSourcesOnlyError) as excinfo:
        open_geotiff(
            memory_tiff_url, window=(0, 0, 2, 2), bbox=_MEMORY_BBOX,
            stable_only=True,
        )
    _assert_stable_error(excinfo)


def test_vrt_bbox_allowed_with_experimental_optin(georef_vrt_path):
    """``allow_experimental_codecs=True`` unlocks the VRT read; bbox resolves."""
    full = open_geotiff(georef_vrt_path)
    sub = open_geotiff(
        georef_vrt_path, bbox=(-105.8, 40.2, -105.2, 40.8),
        stable_only=True, allow_experimental_codecs=True,
    )
    assert 0 < sub.shape[0] < full.shape[0]
    assert 0 < sub.shape[1] < full.shape[1]
