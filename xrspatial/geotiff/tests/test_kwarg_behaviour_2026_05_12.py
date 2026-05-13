"""Parameter-coverage gap closure for the geotiff module.

Test coverage gap sweep 2026-05-12 (pass 9). Three Cat 4 MEDIUM
parameter-coverage gaps plus one Cat 4 LOW error path closed here.

Cat 4 MEDIUM #1 -- ``write_vrt`` documented kwargs are accepted but
not exercised. ``test_polish_1488::TestC5WriteVrtKwargs`` pins the
signature (kwargs accepted, unknown kwargs rejected, docstring
present), but no test verifies the override *effect* of any of
``relative=``, ``crs_wkt=``, or ``nodata=``. A regression that ignored
the override and silently took the default-from-first-source path
would not surface against the existing smoke tests because they only
check that the function returns without raising. The fix is one test
per kwarg that calls ``write_vrt`` with a non-default value and parses
the resulting VRT XML to assert the override landed.

Cat 4 MEDIUM #2 -- ``read_geotiff_gpu(dtype=)`` cast. The eager numpy
path has ``test_dtype_read.TestDtypeEager`` with full coverage
(float64->float32, uint16->int32, uint16->uint8, float-to-int raises,
dtype=None preserves native). The dask path has ``TestDtypeDask``.
The GPU read path has no equivalent. A regression that dropped the
``arr.astype(target)`` block in ``read_geotiff_gpu`` would silently
return data in the file's native dtype, breaking any GPU pipeline
that relies on the cast.

Cat 4 MEDIUM #3 -- ``write_geotiff_gpu(bigtiff=)``. The CPU writer
covers ``bigtiff=True`` / ``False`` / ``None`` (auto) via
``test_features::test_force_bigtiff_via_public_api`` and friends.
``write_geotiff_gpu`` threads ``bigtiff=`` through to
``_assemble_tiff(force_bigtiff=...)`` but no test asserts the on-disk
header is BigTIFF when the kwarg is set on the GPU writer. A
regression dropping the kwarg from the GPU writer's _assemble_tiff
call site would silently fall back to classic-TIFF on the GPU path.

Cat 4 LOW -- ``write_vrt(source_files=[])`` error path. The validator
raises ``ValueError("source_files must not be empty")``. The error
message is not exercised by any test, so a regression dropping the
check would only surface on a downstream IndexError much further in.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_gpu,
    to_geotiff,
    write_geotiff_gpu,
    write_vrt,
)
from xrspatial.geotiff._header import parse_header
from xrspatial.geotiff._vrt import parse_vrt


# --------------------------------------------------------------------------
# GPU gating
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def source_tif(tmp_path):
    """Write a single-band float32 GeoTIFF with EPSG:4326 + nodata."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    y = np.linspace(1.0, 0.0, 8)
    x = np.linspace(0.0, 1.0, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326, 'nodata': -1.0},
    )
    p = str(tmp_path / 'src_kwbeh_2026_05_12.tif')
    to_geotiff(da, p, compression='none')
    return p


@pytest.fixture
def float64_tif(tmp_path):
    """Write a float64 GeoTIFF for GPU dtype cast tests."""
    arr = np.random.default_rng(2026_05_12).random((40, 40)).astype(np.float64)
    y = np.linspace(41.0, 40.0, 40)
    x = np.linspace(-105.0, -104.0, 40)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / 'kwbeh_2026_05_12_f64.tif')
    to_geotiff(da, p, compression='none')
    return p, arr


@pytest.fixture
def uint16_tif(tmp_path):
    """Write a uint16 GeoTIFF for GPU dtype cast tests."""
    arr = np.random.default_rng(2026_05_12).integers(
        0, 10_000, (30, 30), dtype=np.uint16
    )
    y = np.linspace(41.0, 40.0, 30)
    x = np.linspace(-105.0, -104.0, 30)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / 'kwbeh_2026_05_12_u16.tif')
    to_geotiff(da, p, compression='none')
    return p, arr


# ==========================================================================
# Cat 4 MEDIUM #1: write_vrt kwarg behaviour
# ==========================================================================


class TestWriteVrtRelativeBehaviour:
    """``relative=`` flips the ``relativeToVRT`` attribute and rewrites the
    source filename. The existing smoke test only asserts both modes are
    *accepted*, not that they actually take effect."""

    def _read_xml(self, path):
        with open(path, 'r') as fh:
            return fh.read()

    def test_relative_true_writes_relative_path(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_true.vrt')
        write_vrt(vrt_path, [source_tif], relative=True)

        xml = self._read_xml(vrt_path)
        # The on-disk text must carry the relativeToVRT="1" attribute,
        # not "0", and the SourceFilename text must not contain the
        # absolute path's tmp_path prefix.
        assert 'relativeToVRT="1"' in xml
        assert 'relativeToVRT="0"' not in xml
        # Source path is the bare filename (same directory as the VRT).
        assert os.path.basename(source_tif) in xml
        # The absolute path prefix (the tmp_path directory) is not in
        # the XML; otherwise the writer would have stored the full
        # path despite relative=True.
        assert str(tmp_path) not in xml

    def test_relative_false_writes_absolute_path(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_false.vrt')
        write_vrt(vrt_path, [source_tif], relative=False)

        xml = self._read_xml(vrt_path)
        # ``relative=False`` must flip the attribute and emit an absolute
        # path. A regression that ignored ``relative=`` would silently
        # produce the same XML as ``relative=True``.
        assert 'relativeToVRT="0"' in xml
        assert 'relativeToVRT="1"' not in xml
        # Absolute path is in the file's SourceFilename text.
        # Use realpath to handle symlinks tmp_path may carry on macOS.
        abs_src = os.path.realpath(source_tif)
        assert abs_src in xml

    def test_relative_true_parses_back_to_same_source(self, source_tif, tmp_path):
        """relative=True still round-trips: parse_vrt resolves the
        relative path back to the absolute one."""
        vrt_path = str(tmp_path / 'rel_true_rt.vrt')
        write_vrt(vrt_path, [source_tif], relative=True)
        parsed = parse_vrt(self._read_xml(vrt_path), vrt_dir=str(tmp_path))
        assert len(parsed.bands) == 1
        assert len(parsed.bands[0].sources) == 1
        # parse_vrt canonicalises with realpath, so compare against the
        # realpath of the original source.
        assert (
            os.path.realpath(parsed.bands[0].sources[0].filename)
            == os.path.realpath(source_tif)
        )

    def test_relative_false_parses_back_to_same_source(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_false_rt.vrt')
        write_vrt(vrt_path, [source_tif], relative=False)
        parsed = parse_vrt(self._read_xml(vrt_path), vrt_dir=str(tmp_path))
        assert len(parsed.bands) == 1
        assert (
            os.path.realpath(parsed.bands[0].sources[0].filename)
            == os.path.realpath(source_tif)
        )


class TestWriteVrtCrsWktBehaviour:
    """``crs=`` overrides the first source's CRS. Without an override,
    the first source's WKT is propagated. With an override, the
    override wins.

    Pre-#1715 the kwarg was named ``crs_wkt``. The new canonical name
    is ``crs`` (parity with ``to_geotiff`` / ``write_geotiff_gpu``);
    the old name is still accepted with ``DeprecationWarning``. These
    tests exercise the new path; the deprecated path is covered by
    ``test_write_vrt_crs_1715.py``.
    """

    def _read_parsed(self, vrt_path, tmp_path):
        with open(vrt_path, 'r') as fh:
            return parse_vrt(fh.read(), vrt_dir=str(tmp_path))

    def test_crs_wkt_override_wins(self, source_tif, tmp_path):
        """The supplied WKT must land in <SRS>, not the source's WKT."""
        override = (
            'PROJCS["UnitTest_Override_Sweep_2026_05_12",'
            'GEOGCS["test_datum",DATUM["d",SPHEROID["s",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],UNIT["metre",1]]'
        )
        vrt_path = str(tmp_path / 'crs_wkt_override.vrt')
        write_vrt(vrt_path, [source_tif], crs=override)
        parsed = self._read_parsed(vrt_path, tmp_path)
        assert parsed.crs_wkt == override

    def test_crs_wkt_none_falls_back_to_first_source(self, source_tif, tmp_path):
        """No override means the first source's WKT is used. Pin the
        contract: the default-VRT's parsed crs_wkt must be present,
        non-empty, and match the source TIF's own crs_wkt (no silent
        substitution, no None on the fall-back path)."""
        vrt_path = str(tmp_path / 'crs_wkt_default.vrt')
        write_vrt(vrt_path, [source_tif])
        parsed = self._read_parsed(vrt_path, tmp_path)

        source_da = open_geotiff(source_tif)
        source_wkt = source_da.attrs.get('crs_wkt')

        assert parsed.crs_wkt is not None
        assert parsed.crs_wkt != ''
        assert parsed.crs_wkt == source_wkt

    def test_crs_wkt_override_distinct_from_default(self, source_tif, tmp_path):
        """The override and default WKT must produce *different* on-disk
        XML. This is the safety-net: even if a future writer change
        normalises the WKT before emitting, the override path must
        still land a distinguishable WKT in the file."""
        marker = "UnitTest_Override_Marker_Sweep_2026_05_12"
        override = (
            f'GEOGCS["{marker}",'
            'DATUM["d",SPHEROID["s",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
        # Override path
        vrt_override = str(tmp_path / 'override.vrt')
        write_vrt(vrt_override, [source_tif], crs=override)
        # Default path
        vrt_default = str(tmp_path / 'default.vrt')
        write_vrt(vrt_default, [source_tif])

        with open(vrt_override, 'r') as fh:
            text_override = fh.read()
        with open(vrt_default, 'r') as fh:
            text_default = fh.read()

        assert marker in text_override
        assert marker not in text_default


class TestWriteVrtNodataBehaviour:
    """``nodata=`` overrides the first source's nodata sentinel.
    Source file is written with ``nodata=-1.0``; the override must land
    in every ``<NoDataValue>`` element."""

    def _bands(self, vrt_path, tmp_path):
        with open(vrt_path, 'r') as fh:
            return parse_vrt(fh.read(), vrt_dir=str(tmp_path)).bands

    def test_nodata_override_wins(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'nodata_override.vrt')
        write_vrt(vrt_path, [source_tif], nodata=-9999.0)
        bands = self._bands(vrt_path, tmp_path)
        assert len(bands) == 1
        assert bands[0].nodata == -9999.0

    def test_nodata_none_takes_first_source(self, source_tif, tmp_path):
        """No override means the first source's nodata is used. The
        source was written with ``nodata=-1.0`` -- a regression that
        silently dropped the default-from-source code path would land
        ``None`` here."""
        vrt_path = str(tmp_path / 'nodata_default.vrt')
        write_vrt(vrt_path, [source_tif])
        bands = self._bands(vrt_path, tmp_path)
        assert len(bands) == 1
        assert bands[0].nodata == -1.0

    def test_nodata_override_writes_xml_element(self, source_tif, tmp_path):
        """Raw XML check: the override sentinel value lands in a
        <NoDataValue> element."""
        vrt_path = str(tmp_path / 'nodata_xml.vrt')
        write_vrt(vrt_path, [source_tif], nodata=-12345.0)
        with open(vrt_path, 'r') as fh:
            xml = fh.read()
        assert '<NoDataValue>-12345.0</NoDataValue>' in xml


# ==========================================================================
# Cat 4 LOW: write_vrt error paths
# ==========================================================================


class TestWriteVrtEmptySourceFiles:
    """``write_vrt(source_files=[])`` raises with a clear message.
    The error path is uncovered. A regression dropping the
    pre-validation would surface much further down as an IndexError
    when computing the bounding box of zero sources."""

    def test_empty_list_raises(self, tmp_path):
        vrt_path = str(tmp_path / 'should_not_exist.vrt')
        with pytest.raises(ValueError, match="source_files must not be empty"):
            write_vrt(vrt_path, [])

    def test_empty_list_does_not_create_file(self, tmp_path):
        vrt_path = str(tmp_path / 'should_not_exist_2.vrt')
        try:
            write_vrt(vrt_path, [])
        except ValueError:
            pass
        assert not os.path.exists(vrt_path)


# ==========================================================================
# Cat 4 MEDIUM #2: read_geotiff_gpu(dtype=)
# ==========================================================================


@_gpu_only
class TestReadGeotiffGpuDtype:
    """``read_geotiff_gpu(dtype=...)`` casts on device. The eager CPU
    path has TestDtypeEager; the dask path has TestDtypeDask. The GPU
    path had no equivalent."""

    def test_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = read_geotiff_gpu(path, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.data.get(), orig.astype(np.float32), decimal=6)

    def test_float64_to_float16(self, float64_tif):
        path, _ = float64_tif
        result = read_geotiff_gpu(path, dtype=np.float16)
        assert result.dtype == np.float16

    def test_uint16_to_int32(self, uint16_tif):
        path, orig = uint16_tif
        result = read_geotiff_gpu(path, dtype='int32')
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.data.get(), orig.astype(np.int32))

    def test_uint16_to_uint8(self, uint16_tif):
        path, _ = uint16_tif
        result = read_geotiff_gpu(path, dtype='uint8')
        assert result.dtype == np.uint8

    def test_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        # The validator runs before the GPU upload; the error contract is
        # the same as the CPU path (``float`` ... ``int``).
        with pytest.raises(ValueError, match='float.*int'):
            read_geotiff_gpu(path, dtype='int32')

    def test_dtype_none_preserves_native_float64(self, float64_tif):
        path, _ = float64_tif
        result = read_geotiff_gpu(path, dtype=None)
        assert result.dtype == np.float64

    def test_dtype_none_preserves_native_uint16(self, uint16_tif):
        path, _ = uint16_tif
        result = read_geotiff_gpu(path, dtype=None)
        assert result.dtype == np.uint16


@_gpu_only
class TestOpenGeotiffGpuDispatchDtype:
    """``open_geotiff(..., gpu=True, dtype=...)`` forwards through the
    dispatcher into ``read_geotiff_gpu``. Pin the dispatch path so a
    regression dropping ``dtype=`` on the GPU branch surfaces here too."""

    def test_dispatch_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, gpu=True, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.data.get(), orig.astype(np.float32), decimal=6)

    def test_dispatch_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, gpu=True, dtype='int32')


@_gpu_only
class TestReadGeotiffGpuChunksDtype:
    """``read_geotiff_gpu(chunks=..., dtype=...)`` -- dask + GPU + dtype
    combination is a separate dispatch path through the GPU reader and
    its own ``astype`` step on the cupy array, then a ``chunk`` call.
    Cover the cast for the dask+GPU branch too."""

    def test_chunks_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = read_geotiff_gpu(path, chunks=20, dtype='float32')
        assert result.dtype == np.float32
        # ``.data`` is a dask array of cupy chunks. Compute, then
        # ``.get()`` the resulting cupy host buffer.
        computed = result.data.compute()
        np.testing.assert_array_almost_equal(
            computed.get(), orig.astype(np.float32), decimal=6)


# ==========================================================================
# Cat 4 MEDIUM #3: write_geotiff_gpu(bigtiff=)
# ==========================================================================


@_gpu_only
class TestWriteGeotiffGpuBigtiff:
    """``write_geotiff_gpu(bigtiff=)`` threads ``force_bigtiff=`` to
    ``_assemble_tiff``. The CPU writer has equivalent header-level
    bigtiff coverage; the GPU writer did not.

    Small arrays are sufficient because the BigTIFF decision is a
    width-of-offset-field switch, not a value-range one -- a forced
    BigTIFF on a 64-pixel array produces the same header magic byte
    pattern that a >4 GB file would."""

    def _read_header_is_bigtiff(self, path):
        with open(path, 'rb') as fh:
            header = parse_header(fh.read(16))
        return header.is_bigtiff

    def test_force_bigtiff_true_writes_bigtiff(self, tmp_path):
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_true.tif')
        write_geotiff_gpu(da, path, bigtiff=True, tile_size=16)
        assert self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu(bigtiff=True) should emit BigTIFF header "
            "(magic byte 43)."
        )
        # Data round-trips even with the BigTIFF header.
        rd = open_geotiff(path)
        np.testing.assert_array_equal(rd.values, arr.get())

    def test_force_bigtiff_false_writes_classic(self, tmp_path):
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_false.tif')
        write_geotiff_gpu(da, path, bigtiff=False, tile_size=16)
        assert not self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu(bigtiff=False) should emit classic TIFF."
        )

    def test_bigtiff_none_stays_classic_small_file(self, tmp_path):
        """``bigtiff=None`` (default) is auto: small files should stay
        classic. Without an explicit None test, a regression flipping
        the default to ``True`` would not be caught -- and that would
        break interop with older readers that don't accept BigTIFF."""
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_default.tif')
        write_geotiff_gpu(da, path, tile_size=16)
        assert not self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu default should auto-pick classic TIFF for "
            "tiny outputs; a default switch to BigTIFF would break "
            "older readers."
        )

    def test_to_geotiff_gpu_bigtiff_threads_through(self, tmp_path):
        """``to_geotiff(..., gpu=True, bigtiff=True)`` dispatches into
        ``write_geotiff_gpu(bigtiff=True)``. Cover the dispatcher's
        thread-through so a regression dropping ``bigtiff=`` on the GPU
        dispatch branch surfaces here too."""
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'to_gpu_bigtiff_true.tif')
        to_geotiff(da, path, gpu=True, bigtiff=True, tile_size=16)
        assert self._read_header_is_bigtiff(path), (
            "to_geotiff(gpu=True, bigtiff=True) should reach the GPU "
            "writer with force_bigtiff=True propagated through."
        )
        rd = open_geotiff(path)
        np.testing.assert_array_equal(rd.values, arr.get())
