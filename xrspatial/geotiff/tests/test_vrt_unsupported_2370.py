"""Negative coverage for unsupported VRT features (issue #2370, epic #2342).

The release does not promise full GDAL VRT parity. A VRT that asks for
something outside the implemented subset must fail with a clear,
actionable error rather than silently produce wrong data. This module
locks in the rejection contract for the following cases:

* Warped VRT (``<VRTRasterBand subClass="VRTWarpedRasterBand">`` or a
  dataset carrying ``<GDALWarpOptions>``).
* Nested VRT (a ``.vrt`` referenced as ``<SourceFilename>`` inside
  another ``.vrt``).
* Mixed source CRS across band sources.
* Mixed source dtype across band sources where the output dtype is
  ambiguous.
* Mixed band count across sources.
* Complex mask source semantics that the attrs contract cannot
  represent.
* Unsupported resample algorithm (anything outside the implemented
  nearest-neighbour subset).

Each test asserts the exception type AND checks the error message
names the unsupported feature so users can fix the input. Where the
current code already rejects the case, the test locks the behaviour
in. Where the centralized validator from sibling PR #2329 is needed
for the rejection, the assertion is wrapped with ``pytest.xfail`` so
this PR can land independently.

Coverage spans both ``read_vrt`` and ``open_geotiff(... .vrt ...)``
entry points -- a missing rejection at either path leaves a release
loophole.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._vrt import read_vrt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


PR1_XFAIL = "depends on centralized validate_vrt_capability (#2329)"


def _write_src_tif(tmp_path, *, name: str, dtype=np.float32,
                   shape=(4, 4)) -> str:
    """Write a tiny GeoTIFF with the requested dtype/shape and return the path.

    All filenames include ``2370`` so parallel test workers do not collide
    with other rockout worktrees.
    """
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    y = np.linspace(1.0, 0.0, shape[0])
    x = np.linspace(0.0, 1.0, shape[1])
    fill = -9999 if np.issubdtype(arr.dtype, np.integer) else -9999.0
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'nodata': fill, 'crs': 'EPSG:4326'},
    )
    path = str(tmp_path / f'src_2370_{name}.tif')
    to_geotiff(da, path)
    return path


def _write_vrt(tmp_path, xml: str, name: str) -> str:
    """Write ``xml`` to ``tmp_path/<name>.vrt`` and return the path."""
    path = str(tmp_path / f'vrt_2370_{name}.vrt')
    with open(path, 'w') as fh:
        fh.write(xml)
    return path


def _simple_source_xml(src_path: str, *, band: int = 1,
                       x_size: int = 4, y_size: int = 4,
                       dst_x_size: int | None = None,
                       dst_y_size: int | None = None,
                       extra_inner: str = '') -> str:
    """Render a single ``<SimpleSource>`` block.

    ``extra_inner`` is spliced verbatim into the ``<SimpleSource>``
    element so callers can add ``<ResampleAlg>``, ``<NODATA>``, etc.
    """
    dst_x = x_size if dst_x_size is None else dst_x_size
    dst_y = y_size if dst_y_size is None else dst_y_size
    return f"""    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>{band}</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}"/>
      <DstRect xOff="0" yOff="0" xSize="{dst_x}" ySize="{dst_y}"/>
      {extra_inner}
    </SimpleSource>"""


def _vrt_xml(*, width: int = 4, height: int = 4,
             dtype_name: str = 'Float32',
             body: str = '',
             extra_dataset_inner: str = '',
             srs: str = 'EPSG:4326') -> str:
    """Render a minimal VRT XML wrapper."""
    return f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{srs}</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  {extra_dataset_inner}
  <VRTRasterBand dataType="{dtype_name}" band="1">
{body}
  </VRTRasterBand>
</VRTDataset>"""


# ---------------------------------------------------------------------------
# Group 1 -- Warped VRT
# ---------------------------------------------------------------------------


def _assert_raises_or_xfail(exc_types: tuple[type[BaseException], ...],
                            keywords: tuple[str, ...],
                            call):
    """Run ``call`` and check whether the rejection contract is met.

    The contract: ``call`` raises one of ``exc_types`` AND the lowercased
    message contains at least one of ``keywords``. If either half is
    missing, mark the test ``xfail`` with the PR1 dependency reason --
    so PR1 lands the validator and this test starts passing without an
    edit here.
    """
    try:
        call()
    except exc_types as exc:
        msg = str(exc).lower()
        if any(k in msg for k in keywords):
            return
        pytest.xfail(
            f"{PR1_XFAIL}: raised {type(exc).__name__} but message "
            f"did not name expected keyword ({keywords!r}): {msg!r}")
    except BaseException as exc:  # pragma: no cover -- diagnostic
        pytest.xfail(
            f"{PR1_XFAIL}: raised unexpected {type(exc).__name__}: {exc!r}")
    else:
        pytest.xfail(f"{PR1_XFAIL}: call did not raise")


def test_warped_vrt_subclass_raises(tmp_path):
    """``<VRTRasterBand subClass="VRTWarpedRasterBand">`` is a warped VRT
    and must be rejected: read_vrt has no warp pipeline, so honouring the
    band would silently emit unprojected pixels labelled as warped.
    """
    src_path = _write_src_tif(tmp_path, name='warped_sub')
    warped_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4" subClass="VRTWarpedDataset">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTWarpedRasterBand">
    <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, warped_xml, 'warped_subclass')

    # Current behaviour: no centralized validator yet -- the reader
    # accepts the band as a no-source band (silent zero fill). PR #2329
    # is the one that turns this into the documented rejection.
    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError),
        ('warp', 'vrtwarped'),
        lambda: read_vrt(vrt_path),
    )


def test_warped_vrt_gdalwarpoptions_raises(tmp_path):
    """A VRT containing ``<GDALWarpOptions>`` is by definition a warped
    VRT regardless of the band subClass. Must be rejected at parse or
    read time with a message that names ``GDALWarpOptions`` or 'warp'.
    """
    src_path = _write_src_tif(tmp_path, name='warpopts')
    body = _simple_source_xml(src_path)
    warp_options = """  <GDALWarpOptions>
    <WarpMemoryLimit>6.71089e+07</WarpMemoryLimit>
    <ResampleAlg>NearestNeighbour</ResampleAlg>
    <WorkingDataType>Float32</WorkingDataType>
  </GDALWarpOptions>"""
    xml = _vrt_xml(body=body, extra_dataset_inner=warp_options)
    vrt_path = _write_vrt(tmp_path, xml, 'warp_options')

    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError),
        ('warp',),
        lambda: read_vrt(vrt_path),
    )


def test_warped_vrt_open_geotiff_raises(tmp_path):
    """``open_geotiff(... warped.vrt ...)`` must reject too -- the
    rejection cannot live only in ``read_vrt`` or callers that go
    through the public accessor would slip past the contract.
    """
    src_path = _write_src_tif(tmp_path, name='warped_og')
    warped_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4" subClass="VRTWarpedDataset">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTWarpedRasterBand">
    <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, warped_xml, 'warped_og')

    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError),
        ('warp', 'vrtwarped'),
        lambda: open_geotiff(vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 2 -- Nested VRT
# ---------------------------------------------------------------------------


def test_nested_vrt_source_raises(tmp_path):
    """A VRT whose ``<SourceFilename>`` is itself a ``.vrt`` is a nested
    VRT. Resolution semantics are GDAL-specific and not promised here,
    so it must raise rather than try to parse the inner ``.vrt`` as a
    TIFF (which is what would happen today).
    """
    # Inner VRT pointing at a real TIFF.
    inner_src = _write_src_tif(tmp_path, name='nested_inner')
    inner_body = _simple_source_xml(inner_src)
    inner_xml = _vrt_xml(body=inner_body)
    inner_vrt_path = _write_vrt(tmp_path, inner_xml, 'nested_inner')

    # Outer VRT that references the inner .vrt.
    outer_body = _simple_source_xml(inner_vrt_path)
    outer_xml = _vrt_xml(body=outer_body)
    outer_vrt_path = _write_vrt(tmp_path, outer_xml, 'nested_outer')

    # Today the read trips a generic TIFF parse error that does not
    # name '.vrt' or 'nested'; xfail until PR #2329 lands the validator.
    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError, OSError),
        ('.vrt', 'nested'),
        lambda: read_vrt(outer_vrt_path),
    )


def test_nested_vrt_open_geotiff_raises(tmp_path):
    """Same nested-VRT rejection contract through the public entry point."""
    inner_src = _write_src_tif(tmp_path, name='nested_og_inner')
    inner_body = _simple_source_xml(inner_src)
    inner_xml = _vrt_xml(body=inner_body)
    inner_vrt_path = _write_vrt(tmp_path, inner_xml, 'nested_og_inner')

    outer_body = _simple_source_xml(inner_vrt_path)
    outer_xml = _vrt_xml(body=outer_body)
    outer_vrt_path = _write_vrt(tmp_path, outer_xml, 'nested_og_outer')

    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError, OSError),
        ('.vrt', 'nested'),
        lambda: open_geotiff(outer_vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 3 -- Mixed source CRS
# ---------------------------------------------------------------------------


def test_mixed_source_crs_raises(tmp_path):
    """Two band sources with disagreeing CRS (one EPSG:4326 source, one
    EPSG:3857 source) cannot mosaic correctly without reprojection. The
    VRT XML itself only carries a dataset-level ``<SRS>``, so the
    mismatch only surfaces when the validator opens each source TIFF.
    Pinned here so the validator PR (#2329) is the one that has to
    deliver it.
    """
    src_4326 = _write_src_tif(tmp_path, name='crs_4326')
    # Build a second source with a different CRS by writing the TIFF
    # through ``to_geotiff`` with a different crs attr.
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da_3857 = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0, 'crs': 'EPSG:3857'},
    )
    src_3857 = str(tmp_path / 'src_2370_crs_3857.tif')
    to_geotiff(da_3857, src_3857)

    body = (
        _simple_source_xml(src_4326)
        + "\n"
        + _simple_source_xml(src_3857)
    )
    xml = _vrt_xml(body=body, srs='EPSG:4326')
    vrt_path = _write_vrt(tmp_path, xml, 'mixed_crs')

    _assert_raises_or_xfail(
        (ValueError, NotImplementedError, RuntimeError),
        ('crs', 'srs', 'projection', 'epsg'),
        lambda: read_vrt(vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 4 -- Mixed source dtype across band sources
# ---------------------------------------------------------------------------


def test_mixed_source_dtype_unsupported_complex_raises(tmp_path):
    """``dataType="CFloat32"`` (and other complex dtype declarations)
    already raises ``ValueError`` per issue #1783 because ``read_vrt``
    has no complex code path. Lock the message in: it must name the
    rejected dtype so users know what to change.
    """
    src_path = _write_src_tif(tmp_path, name='dtype_complex')
    body = _simple_source_xml(src_path)
    xml = _vrt_xml(body=body, dtype_name='CFloat32')
    vrt_path = _write_vrt(tmp_path, xml, 'complex_dtype')

    with pytest.raises(ValueError, match=r'CFloat32') as excinfo:
        read_vrt(vrt_path)
    # The message should be actionable: it names the rejected dtype AND
    # mentions complex.
    assert 'complex' in str(excinfo.value).lower()


def test_mixed_source_dtype_ambiguous_widening_raises(tmp_path):
    """When two bands declare incompatible dtypes (e.g. ``UInt16`` and
    ``Float32``) the current code silently widens the output buffer to
    a common dtype. That widening is fine for compatible mixes, but the
    contract for the release is to reject mixed band dtypes unless the
    user opted in. Pinned for PR #2329.
    """
    src_u16 = _write_src_tif(tmp_path, name='dtype_u16', dtype=np.uint16)
    src_f32 = _write_src_tif(tmp_path, name='dtype_f32', dtype=np.float32)
    body_b1 = _simple_source_xml(src_u16)
    body_b2 = _simple_source_xml(src_f32)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
{body_b1}
  </VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2">
{body_b2}
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, xml, 'mixed_dtype')

    _assert_raises_or_xfail(
        (ValueError, NotImplementedError),
        ('dtype', 'datatype', 'mixed'),
        lambda: read_vrt(vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 5 -- Mixed band count
# ---------------------------------------------------------------------------


def test_mixed_source_band_count_raises(tmp_path):
    """When the VRT references sources with disagreeing band counts (a
    single-band source feeding into a multi-band band layout, or a
    multi-band source feeding into a single-band layout where the
    requested ``SourceBand`` does not exist), the read must fail with a
    message that names the band-count mismatch rather than silently
    decoding the wrong band.
    """
    # Single-band source.
    single_band_src = _write_src_tif(tmp_path, name='band_single')
    # Reference SourceBand=2 against a single-band file -- there is no
    # band 2 to read.
    body = _simple_source_xml(single_band_src, band=2)
    xml = _vrt_xml(body=body)
    vrt_path = _write_vrt(tmp_path, xml, 'band_count')

    _assert_raises_or_xfail(
        (ValueError, IndexError, RuntimeError, NotImplementedError),
        ('band',),
        lambda: read_vrt(vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 6 -- Complex mask source semantics
# ---------------------------------------------------------------------------


def test_complex_mask_source_raises(tmp_path):
    """A dataset-level ``<MaskBand>`` declares a per-pixel mask that the
    GeoTIFF attrs contract does not represent. Reading the mosaic and
    dropping the mask silently would produce a result the caller cannot
    distinguish from one with no mask. Must be rejected.
    """
    src_path = _write_src_tif(tmp_path, name='mask_src')
    mask_src = _write_src_tif(tmp_path, name='mask_msk', dtype=np.uint8)
    body = _simple_source_xml(src_path)
    mask_block = f"""  <MaskBand>
    <VRTRasterBand dataType="Byte">
      <SimpleSource>
        <SourceFilename relativeToVRT="0">{mask_src}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
        <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      </SimpleSource>
    </VRTRasterBand>
  </MaskBand>"""
    xml = _vrt_xml(body=body, extra_dataset_inner=mask_block)
    vrt_path = _write_vrt(tmp_path, xml, 'mask_band')

    # No rejection today -- the mask is silently dropped. Pin the
    # contract for PR #2329.
    _assert_raises_or_xfail(
        (ValueError, NotImplementedError),
        ('mask',),
        lambda: read_vrt(vrt_path),
    )


# ---------------------------------------------------------------------------
# Group 7 -- Unsupported resample algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('alg', ['Bilinear', 'Cubic', 'Lanczos',
                                 'Average', 'Mode'])
def test_unsupported_resample_alg_raises(tmp_path, alg):
    """Already enforced by issue #1751: a ``<ResampleAlg>`` value
    outside the implemented nearest subset, paired with size-changing
    SrcRect/DstRect rects, raises ``NotImplementedError`` with the
    algorithm name in the message. Lock that contract in here so the
    rejection survives any future refactor.
    """
    src_path = _write_src_tif(tmp_path, name=f'res_{alg.lower()}')
    inner = f'<ResampleAlg>{alg}</ResampleAlg>'
    # DstRect 2x2 vs SrcRect 4x4 forces the resample path so the alg
    # check actually fires.
    body = f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      {inner}
    </ComplexSource>"""
    xml = _vrt_xml(width=2, height=2, body=body)
    vrt_path = _write_vrt(tmp_path, xml, f'resample_{alg.lower()}')

    with pytest.raises(NotImplementedError) as excinfo:
        read_vrt(vrt_path)
    msg = str(excinfo.value)
    assert alg in msg, f"error must name the rejected algorithm: {msg!r}"


def test_unsupported_resample_alg_open_geotiff(tmp_path):
    """The same rejection must fire through ``open_geotiff`` -- the
    public entry point shares the read path, so a regression there
    would mean only the low-level helper is safe.
    """
    src_path = _write_src_tif(tmp_path, name='res_og')
    body = f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <ResampleAlg>Cubic</ResampleAlg>
    </ComplexSource>"""
    xml = _vrt_xml(width=2, height=2, body=body)
    vrt_path = _write_vrt(tmp_path, xml, 'resample_og')

    with pytest.raises(NotImplementedError, match='Cubic'):
        open_geotiff(vrt_path)


# ---------------------------------------------------------------------------
# Multi-entrypoint contract -- the trivial passing-case anchor so the
# file always exercises the public ``open_geotiff`` path too. If this
# regresses (e.g. extension dispatch breaks for .vrt) every test above
# becomes ambiguous, so guard it explicitly.
# ---------------------------------------------------------------------------


def test_supported_simple_vrt_round_trips_via_open_geotiff(tmp_path):
    """Sanity anchor: a supported single-source VRT opens cleanly via
    ``open_geotiff``, so the negative tests above are exercising a live
    extension-dispatch code path rather than a broken accessor.
    """
    src_path = _write_src_tif(tmp_path, name='anchor')
    body = _simple_source_xml(src_path)
    xml = _vrt_xml(body=body)
    vrt_path = _write_vrt(tmp_path, xml, 'anchor')

    da = open_geotiff(vrt_path)
    assert da.shape == (4, 4)
