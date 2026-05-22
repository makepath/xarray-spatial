"""Chunked ``read_vrt`` parses the VRT XML once (issue #1825).

Before the refactor each per-chunk task re-parsed the VRT XML and
re-validated source-path containment, so an N-chunk read paid an N+1
parse cost. The dispatcher now parses once and threads the parsed
``VRTDataset`` into every task via the dask graph, removing the
per-task XML parse and allowlist validation.

These tests pin the new behaviour:

* the dispatcher calls ``parse_vrt`` exactly once during construction,
  and ``.compute()`` does not parse the XML again per task;
* the parsed VRT object survives pickling, so the dask graph can ship
  it to workers under any scheduler;
* numerical results match the eager path byte-for-byte (regression
  guard for the helper extraction).
"""
from __future__ import annotations

import os
import pickle
import tempfile

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_vrt, to_geotiff
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


@pytest.fixture
def two_by_two_vrt_1825():
    """4-tile mosaic via the to_geotiff(.vrt, ...) dask path."""
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    y = np.linspace(41.0, 40.0, 256)
    x = np.linspace(-106.0, -105.0, 256)
    raster = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1825_2x2_')
    vrt_path = os.path.join(td, 'mosaic_1825.vrt')
    to_geotiff(raster, vrt_path, tile_size=128)
    yield vrt_path, arr


@pytest.fixture
def single_tile_vrt_1825():
    """One 64x64 float32 tile wrapped in a VRT."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    y = np.linspace(41.0, 40.0, 64)
    x = np.linspace(-106.0, -105.0, 64)
    raster = xr.DataArray(arr, dims=['y', 'x'],
                          coords={'y': y, 'x': x},
                          attrs={'crs': 4326})
    td = tempfile.mkdtemp(prefix='tmp_1825_single_')
    tile_path = os.path.join(td, 'tile_1825.tif')
    to_geotiff(raster, tile_path)
    vrt_path = os.path.join(td, 'single_1825.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    yield vrt_path, arr


def test_chunked_path_parses_xml_once(monkeypatch, two_by_two_vrt_1825):
    """Construction parses once, and ``.compute()`` adds zero parses.

    The previous implementation re-parsed inside every per-chunk task,
    so a 4x4 chunk grid produced 17 parses total. After #1825 the
    dispatcher parses once and threads the already-parsed VRTDataset
    through the task graph.
    """
    vrt_path, _ = two_by_two_vrt_1825

    from xrspatial.geotiff import _vrt as vrt_module

    counter = {'parses': 0}
    real_parse = vrt_module.parse_vrt

    def counting_parse(*args, **kwargs):
        counter['parses'] += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(vrt_module, 'parse_vrt', counting_parse)

    result = read_vrt(vrt_path, chunks=(64, 64))

    # Construction parses exactly once.
    assert counter['parses'] == 1, (
        f"expected 1 parse during construction, got {counter['parses']}"
    )

    computed = result.compute()

    # 4x4 chunk grid would re-parse 16 more times under the old code.
    assert counter['parses'] == 1, (
        f"expected 1 parse total (construction only); got "
        f"{counter['parses']} -- per-chunk tasks are still reparsing"
    )

    # Sanity: the computed array is the original data.
    assert computed.shape == (256, 256)
    assert computed.dtype == np.float32


def test_chunked_path_reads_xml_file_once(monkeypatch, two_by_two_vrt_1825):
    """The chunked dispatcher reads the VRT XML file exactly once.

    Pin the file-read side too: before #1825 every per-chunk task
    re-opened the .vrt file via ``_read_vrt_xml``. After the refactor
    only the dispatcher reads it.
    """
    vrt_path, _ = two_by_two_vrt_1825

    from xrspatial.geotiff import _vrt as vrt_module

    counter = {'reads': 0}
    real_read_xml = vrt_module._read_vrt_xml

    def counting_read_xml(*args, **kwargs):
        counter['reads'] += 1
        return real_read_xml(*args, **kwargs)

    monkeypatch.setattr(vrt_module, '_read_vrt_xml', counting_read_xml)

    result = read_vrt(vrt_path, chunks=(64, 64))
    assert counter['reads'] == 1, (
        f"expected 1 XML file read during construction, got "
        f"{counter['reads']}"
    )

    result.compute()
    assert counter['reads'] == 1, (
        f"expected 1 XML file read total; got {counter['reads']} -- "
        f"per-chunk tasks are still re-opening the .vrt file"
    )


def test_parsed_vrt_is_picklable(single_tile_vrt_1825):
    """The parsed VRTDataset round-trips through pickle.

    The chunked dispatcher embeds the parsed VRT into the dask graph,
    so dask must be able to serialise it for the distributed and
    process-pool schedulers. Pin picklability with the stdlib pickler
    (cloudpickle is a strict superset).
    """
    vrt_path, _ = single_tile_vrt_1825
    from xrspatial.geotiff._vrt import _read_vrt_xml, parse_vrt

    xml_str = _read_vrt_xml(vrt_path)
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    vrt = parse_vrt(xml_str, vrt_dir)

    blob = pickle.dumps(vrt)
    restored = pickle.loads(blob)

    assert restored.width == vrt.width
    assert restored.height == vrt.height
    assert len(restored.bands) == len(vrt.bands)
    assert restored.bands[0].dtype == vrt.bands[0].dtype
    assert [s.filename for s in restored.bands[0].sources] == [
        s.filename for s in vrt.bands[0].sources
    ]


def test_chunked_matches_eager_after_refactor(two_by_two_vrt_1825):
    """Byte-identical eager vs chunked results after the helper consolidation.

    The eager path uses ``_apply_integer_sentinel_mask`` /
    ``_effective_dtype_for_bands`` / ``_sentinel_for_dtype`` from
    ``_vrt`` directly; the chunked path imports the same helpers. A
    regression in either call site would surface here.
    """
    vrt_path, original = two_by_two_vrt_1825
    eager = read_vrt(vrt_path)
    chunked = read_vrt(vrt_path, chunks=(64, 64)).compute()
    assert eager.dtype == chunked.dtype
    np.testing.assert_array_equal(eager.values, chunked.values)
    np.testing.assert_array_equal(eager.values, original)


def test_no_path_containment_revalidation_per_chunk(monkeypatch,
                                                    two_by_two_vrt_1825):
    """Per-chunk tasks skip the source-path containment check.

    ``parse_vrt`` is the only place that resolves and validates source
    paths against the VRT directory / ``XRSPATIAL_VRT_ALLOWED_ROOTS``.
    Because each task now receives the already-parsed VRT, ``parse_vrt``
    must not run during ``.compute()`` even when the graph is hydrated.
    """
    vrt_path, _ = two_by_two_vrt_1825

    from xrspatial.geotiff import _vrt as vrt_module

    parse_calls = {'n': 0}
    real_parse = vrt_module.parse_vrt

    def counting_parse(*args, **kwargs):
        parse_calls['n'] += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(vrt_module, 'parse_vrt', counting_parse)

    result = read_vrt(vrt_path, chunks=(64, 64))
    parses_after_construction = parse_calls['n']

    # Compute one block via dask's sliced API; confirm parse count
    # stays at the construction baseline (no extra parses fired).
    da_arr = result.data
    if isinstance(da_arr, da.Array):
        _block = da_arr.blocks[0, 0].compute()
        assert _block.shape[0] > 0 and _block.shape[1] > 0

    assert parse_calls['n'] == parses_after_construction, (
        f"per-block compute triggered extra parses "
        f"({parse_calls['n']} vs {parses_after_construction})"
    )


def test_parsed_kwarg_does_not_mutate_caller_holes(single_tile_vrt_1825):
    """``read_vrt(parsed=...)`` must not mutate the caller's ``holes``.

    The chunked dispatcher threads a single parsed ``VRTDataset`` into
    every per-chunk task. ``read_vrt`` appends skipped-source records to
    ``vrt.holes`` when a backing file is missing; without a defensive
    copy the appends would land on the dispatcher's shared object and
    leak across tasks (racy under the threaded scheduler, and
    cumulatively across calls if a caller ever reused the parsed
    object). Pin that ``parsed.holes`` stays untouched.
    """
    vrt_path, _ = single_tile_vrt_1825
    from xrspatial.geotiff._vrt import _read_vrt_xml, parse_vrt
    from xrspatial.geotiff._vrt import read_vrt as _read_vrt_internal

    xml_str = _read_vrt_xml(vrt_path)
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    parsed = parse_vrt(xml_str, vrt_dir)

    # Point the only source at a path that does not exist so the
    # lenient ``missing_sources='warn'`` branch fires and would append
    # a record onto ``holes``.
    parsed.bands[0].sources[0].filename = os.path.join(vrt_dir, 'gone.tif')
    holes_id_before = id(parsed.holes)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arr, returned = _read_vrt_internal(
            vrt_path, parsed=parsed, missing_sources='warn',
        )

    assert parsed.holes == [], (
        f"parsed.holes was mutated across the read; got {parsed.holes!r}"
    )
    assert id(parsed.holes) == holes_id_before, (
        "parsed.holes list object was replaced -- the caller's reference "
        "is now stale"
    )
    # The returned VRTDataset is the per-call working copy and DID
    # collect the skipped-source record.
    assert len(returned.holes) == 1
    assert returned.holes[0]['source'].endswith('gone.tif')
    assert arr.shape == (64, 64)
