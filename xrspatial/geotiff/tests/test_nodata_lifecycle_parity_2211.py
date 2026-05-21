"""Cross-backend parity tests for the centralized NodataLifecycle (#2226).

PR-C of epic #2211. The lifecycle helper in ``xrspatial.geotiff._nodata``
owns the decisions (raw sentinel, post-MinIsWhite effective sentinel,
``sentinel_fits_buffer``, ``masking_occurred``, ``dtype_cast_occurred``,
``writer_restore_sentinel``) previously inlined across the geotiff
backends. These tests pin the decision contract directly AND check the
public read/write paths produce the same lifecycle attrs across eager
NumPy, Dask, GPU (skip if no CUDA), and VRT eager backends.

Covers:
* integer sentinel
* float sentinel
* NaN sentinel
* out-of-range sentinel (uint16 + -9999)
* MinIsWhite photometric inversion (single-band, photometric=0)
* ``mask_nodata=False``
* explicit ``dtype=`` kwarg
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff._nodata import NodataLifecycle


# ---------------------------------------------------------------------------
# Skip helpers (loud, not silent: a missing-cupy test reports the skip reason)
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
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
    reason="cupy + CUDA required for NodataLifecycle GPU parity (PR-C #2226)",
)


# ===========================================================================
# Direct unit tests on NodataLifecycle's decisions
# ===========================================================================

class TestRawSentinelExposure:
    """``raw_sentinel`` is the declared value, not the inverted one."""

    def test_returns_declared_when_no_photometric(self):
        lc = NodataLifecycle(declared=-9999, dtype_in=np.dtype("int16"))
        assert lc.raw_sentinel == -9999

    def test_returns_declared_under_miniswhite(self):
        # Even when MinIsWhite would invert the effective sentinel, the
        # raw sentinel must remain the original (so attrs['nodata']
        # preserves the on-disk value for write round-trip).
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        assert lc.raw_sentinel == 10

    def test_none_passes_through(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.raw_sentinel is None


class TestEffectiveSentinelUnderMinIsWhite:
    """``effective_sentinel`` mirrors ``_reader._miniswhite_inverted_nodata``."""

    def test_uint_inverts_to_iinfo_max_minus_value(self):
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        # 255 - 10 = 245 (TIFF6 MinIsWhite semantics)
        assert lc.effective_sentinel == 245

    def test_float_inverts_to_negation(self):
        lc = NodataLifecycle(
            declared=3.5,
            photometric=0,
            dtype_in=np.dtype("float32"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == -3.5

    def test_nan_passes_unchanged(self):
        lc = NodataLifecycle(
            declared=float("nan"),
            photometric=0,
            dtype_in=np.dtype("float32"),
            samples_per_pixel=1,
        )
        assert np.isnan(lc.effective_sentinel)

    def test_multiband_photometric_zero_is_not_miniswhite(self):
        # Multi-band photometric=0 is a different TIFF case from
        # single-band MinIsWhite; the reader's gate is
        # ``photometric == 0 AND samples_per_pixel == 1``.
        lc = NodataLifecycle(
            declared=10,
            photometric=0,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=3,
        )
        assert lc.effective_sentinel == 10

    def test_photometric_one_minisblack_no_inversion(self):
        lc = NodataLifecycle(
            declared=10,
            photometric=1,
            dtype_in=np.dtype("uint8"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == 10

    def test_out_of_range_sentinel_stays_unchanged(self):
        # uint16 + nodata=-9999: cannot be represented, mirror reader.
        lc = NodataLifecycle(
            declared=-9999,
            photometric=0,
            dtype_in=np.dtype("uint16"),
            samples_per_pixel=1,
        )
        assert lc.effective_sentinel == -9999


class TestSentinelFitsBuffer:
    """``sentinel_fits_buffer`` collapses finite + integer + in-range gates."""

    def test_int_in_range(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is True

    def test_int_out_of_range(self):
        # uint16 + -9999: cannot match.
        lc = NodataLifecycle(declared=-9999, dtype_in=np.dtype("uint16"))
        assert lc.sentinel_fits_buffer is False

    def test_int_fractional(self):
        # 3.5 on uint16 -> int(3.5) would truncate to 3 and false-flag
        # a real pixel value; helper must reject it.
        lc = NodataLifecycle(declared=3.5, dtype_in=np.dtype("uint16"))
        assert lc.sentinel_fits_buffer is False

    def test_int_non_finite(self):
        lc = NodataLifecycle(declared=float("nan"), dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is False
        lc2 = NodataLifecycle(declared=float("inf"), dtype_in=np.dtype("uint8"))
        assert lc2.sentinel_fits_buffer is False

    def test_float_dtype_always_fits(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.sentinel_fits_buffer is True
        lc2 = NodataLifecycle(
            declared=float("nan"), dtype_in=np.dtype("float32"),
        )
        assert lc2.sentinel_fits_buffer is True

    def test_none_does_not_fit(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("uint8"))
        assert lc.sentinel_fits_buffer is False


class TestMaskingOccurredDecision:
    """``masking_occurred`` mirrors the call sites' attr-stamping rule."""

    def test_no_sentinel_returns_false(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=True) is False

    def test_mask_nodata_false_returns_false(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=False) is False

    def test_float_buffer_with_sentinel_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.masking_occurred(mask_nodata=True) is True

    def test_integer_buffer_returns_false(self):
        # Integer buffer surviving means no promotion -> no pixels masked.
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.masking_occurred(mask_nodata=True) is False

    def test_explicit_float_cast_returns_true(self):
        # Caller cast to float, so the final buffer is float and the
        # mask-occurred attr stays True.
        lc = NodataLifecycle(
            declared=0,
            dtype_in=np.dtype("uint8"),
            dtype_request=np.dtype("float64"),
        )
        assert lc.masking_occurred(mask_nodata=True) is True


class TestDtypeCastDecision:
    """``dtype_cast_occurred`` / ``cast_dtype_name`` reflect caller intent."""

    def test_no_request_returns_false(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.dtype_cast_occurred is False
        assert lc.cast_dtype_name is None

    def test_request_records_name(self):
        lc = NodataLifecycle(
            declared=0,
            dtype_in=np.dtype("uint8"),
            dtype_request="float64",
        )
        assert lc.dtype_cast_occurred is True
        assert lc.cast_dtype_name == "float64"


class TestWriterRestoreSentinelDecision:
    """``writer_restore_sentinel`` mirrors the in-line write-side gate."""

    def test_no_declared_returns_false(self):
        lc = NodataLifecycle(declared=None, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is False

    def test_non_float_buffer_returns_false(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("uint8"),
        ) is False

    def test_nan_sentinel_returns_false(self):
        lc = NodataLifecycle(
            declared=float("nan"), dtype_in=np.dtype("float32"),
        )
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is False

    def test_finite_float_sentinel_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
        ) is True

    def test_masked_nodata_attr_false_returns_false(self):
        # Mirrors issue #1988: literal False on attrs['masked_nodata']
        # tells the writer the read path did NOT mask, so the in-buffer
        # NaN must NOT be rewritten to the sentinel.
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=False,
        ) is False

    def test_masked_nodata_attr_true_returns_true(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=True,
        ) is True

    def test_masked_nodata_attr_none_defaults_true(self):
        # Attr absent -> pre-#1988 default (True).
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            masked_nodata_attr=None,
        ) is True

    def test_restore_sentinel_false_short_circuits(self):
        # ``restore_sentinel=False`` caller opt-out (the cached attr
        # answer from ``_should_restore_nan_sentinel``) must win over
        # the otherwise-True default path.
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            restore_sentinel=False,
        ) is False

    def test_restore_sentinel_true_is_default(self):
        lc = NodataLifecycle(declared=-9999.0, dtype_in=np.dtype("float32"))
        # Explicit True matches the default (kwarg omitted).
        assert lc.writer_restore_sentinel(
            buffer_dtype=np.dtype("float32"),
            restore_sentinel=True,
        ) is True


class TestInvertForMinIsWhiteLockstep:
    """``_invert_for_miniswhite`` mirrors ``_reader._miniswhite_inverted_nodata``.

    The two functions intentionally duplicate the inversion math so the
    lifecycle helper does not pull ``_reader`` into its dependency
    graph. Sweep a wide sentinel / dtype grid here so drift between the
    two surfaces fast.
    """

    def test_grid_matches_reader_helper(self):
        from xrspatial.geotiff._nodata import _invert_for_miniswhite
        from xrspatial.geotiff._reader import _miniswhite_inverted_nodata

        class _IFDStub:
            photometric = 0
            samples_per_pixel = 1

        sentinels = [
            0, 1, 10, 127, 128, 255,        # uint8 range
            256, 1000, 32767, 65535,        # uint16 range
            -1, -9999,                      # out of unsigned range
            0.0, 1.5, 3.5, -3.5,            # float values
            float("nan"), float("inf"),     # non-finite
        ]
        dtypes = [
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("int16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ]
        for s in sentinels:
            for d in dtypes:
                ref = _miniswhite_inverted_nodata(s, _IFDStub, d)
                got = _invert_for_miniswhite(s, d)
                if isinstance(ref, float) and np.isnan(ref):
                    assert isinstance(got, float) and np.isnan(got), (
                        f"sentinel={s} dtype={d}: ref=NaN, got={got!r}"
                    )
                else:
                    assert ref == got, (
                        f"sentinel={s} dtype={d}: ref={ref!r}, got={got!r}"
                    )


class TestPixelsPresentSlot:
    """``pixels_present`` is a settable slot; constructor leaves it None."""

    def test_default_none(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        assert lc.pixels_present is None

    def test_settable(self):
        lc = NodataLifecycle(declared=0, dtype_in=np.dtype("uint8"))
        lc.pixels_present = True
        assert lc.pixels_present is True
        lc.pixels_present = False
        assert lc.pixels_present is False


# ===========================================================================
# Cross-backend parity through the public API
# ===========================================================================

# Helpers borrowed in spirit from test_nodata_lifecycle_attrs_2135.py:
# build small rasters and compare the post-read lifecycle attrs across
# the eager / dask / GPU / VRT-eager paths.

def _make_int_raster(path, sentinel=255, dtype=np.uint8, plant=True):
    """2x3 uint raster with the sentinel planted at (0, 2) when ``plant``."""
    import xarray as xr

    if plant:
        data = np.array(
            [[1, 2, sentinel], [4, 5, 6]], dtype=dtype,
        )
    else:
        data = np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=dtype,
        )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": sentinel},
    )
    from xrspatial.geotiff import to_geotiff

    to_geotiff(da, path, nodata=sentinel)
    return path


def _make_float_raster(path, sentinel=-9999.0, plant=True):
    import xarray as xr

    if plant:
        data = np.array(
            [[1.0, 2.0, sentinel], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    else:
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": sentinel},
    )
    from xrspatial.geotiff import to_geotiff

    to_geotiff(da, path, nodata=sentinel)
    return path


@pytest.fixture
def int_tif(tmp_path):
    return _make_int_raster(str(tmp_path / "int_sentinel_2226.tif"))


@pytest.fixture
def float_tif(tmp_path):
    return _make_float_raster(str(tmp_path / "float_sentinel_2226.tif"))


@pytest.fixture
def nan_tif(tmp_path):
    return _make_float_raster(
        str(tmp_path / "nan_sentinel_2226.tif"),
        sentinel=float("nan"),
        plant=False,
    )


@pytest.fixture
def oor_tif(tmp_path):
    # uint16 + -9999: classic out-of-range sentinel that cannot match
    # any pixel; the reader still surfaces it on attrs['nodata'].
    path = str(tmp_path / "uint16_oor_2226.tif")
    return _make_int_raster(
        path, sentinel=-9999, dtype=np.uint16, plant=False,
    )


# ----------------------- integer sentinel -----------------------

class TestIntegerSentinelParity:
    def test_eager_masks_int_sentinel_to_nan(self, int_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(int_tif)
        # int sentinel auto-promotes to float when at least one pixel
        # matches, leaving NaN where the sentinel was.
        assert da.dtype.kind == "f"
        assert np.isnan(da.data[0, 2])
        assert da.attrs["nodata"] == 255
        assert da.attrs["masked_nodata"] is True

    def test_dask_matches_eager(self, int_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_dask

        eager = open_geotiff(int_tif)
        lazy = read_geotiff_dask(int_tif, chunks=2)
        # Same on-disk sentinel propagated.
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]
        # Same lifecycle decision: dask graph promoted to float64.
        assert lazy.dtype.kind == "f"
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )

    @_gpu_only
    def test_gpu_matches_eager(self, int_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

        eager = open_geotiff(int_tif)
        gpu = read_geotiff_gpu(int_tif)
        assert gpu.attrs["nodata"] == eager.attrs["nodata"]
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(gpu.data.get()),
        )


# ----------------------- float sentinel -----------------------

class TestFloatSentinelParity:
    def test_eager(self, float_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(float_tif)
        assert da.dtype == np.float32
        assert np.isnan(da.data[0, 2])
        assert da.attrs["nodata"] == -9999.0
        assert da.attrs["masked_nodata"] is True

    def test_dask(self, float_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_dask

        eager = open_geotiff(float_tif)
        lazy = read_geotiff_dask(float_tif, chunks=2)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]

    @_gpu_only
    def test_gpu(self, float_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

        eager = open_geotiff(float_tif)
        gpu = read_geotiff_gpu(float_tif)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(gpu.data.get()),
        )


# ----------------------- NaN sentinel -----------------------

class TestNaNSentinelParity:
    def test_eager(self, nan_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(nan_tif)
        # NaN sentinel: float buffer keeps NaN as the sentinel; no
        # integer-to-float promotion because the source was already
        # float, and mask sees no literal pixel match.
        assert da.dtype == np.float32
        assert np.isnan(da.attrs["nodata"])

    def test_dask_matches(self, nan_tif):
        from xrspatial.geotiff import read_geotiff_dask

        lazy = read_geotiff_dask(nan_tif, chunks=2)
        out = lazy.compute()
        # Source had no NaN pixels planted, so the float buffer carries
        # the original values.
        assert out.dtype == np.float32
        assert np.isnan(out.attrs["nodata"])


# ----------------------- out-of-range integer sentinel -----------------------

class TestOutOfRangeSentinelParity:
    def test_eager_keeps_int_dtype_and_records_sentinel(self, oor_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(oor_tif)
        # uint16 + -9999 cannot match any pixel; buffer stays integer.
        assert da.dtype == np.uint16
        # Sentinel still recorded for write round-trip.
        assert int(da.attrs["nodata"]) == -9999

    def test_dask_matches(self, oor_tif):
        from xrspatial.geotiff import open_geotiff, read_geotiff_dask

        eager = open_geotiff(oor_tif)
        lazy = read_geotiff_dask(oor_tif, chunks=2)
        assert lazy.dtype == eager.dtype
        np.testing.assert_array_equal(
            eager.data, lazy.compute().data,
        )
        assert int(lazy.attrs["nodata"]) == int(eager.attrs["nodata"])


# ----------------------- MinIsWhite photometric -----------------------

class TestMinIsWhiteSentinelInversion:
    """Direct lifecycle check + downstream usage on a hand-built file.

    Building a real MinIsWhite file through ``to_geotiff`` is supported
    (``photometric='miniswhite'``); we use that to drive an end-to-end
    parity check across eager + dask. The lifecycle helper's
    ``effective_sentinel`` is what the per-chunk masking compares
    against, so a planted sentinel pixel must mask correctly after
    inversion.
    """

    def _build(self, path, sentinel=10):
        import xarray as xr

        from xrspatial.geotiff import to_geotiff

        data = np.array(
            [[1, 2, sentinel], [4, 5, 6]], dtype=np.uint8,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
            attrs={"nodata": sentinel},
        )
        to_geotiff(da, path, nodata=sentinel, photometric="miniswhite")

    def test_eager_masks_inverted_sentinel(self, tmp_path):
        from xrspatial.geotiff import open_geotiff

        path = str(tmp_path / "miw_2226.tif")
        self._build(path)
        da = open_geotiff(path)
        # The MinIsWhite writer pre-inverts both pixels AND the sentinel
        # (see ``_writer._invert_nodata_for_miniswhite``), so the on-disk
        # GDAL_NODATA tag stores 245 = 255 - 10. The reader's MinIsWhite
        # inversion runs on read; the reader's downstream mask compares
        # against the post-inversion sentinel (10 again) and rewrites the
        # planted pixel to NaN. The lifecycle helper's contract requires
        # the same effective_sentinel resolution.
        assert np.isnan(da.data[0, 2])
        # attrs['nodata'] carries the on-disk value (inverted by the
        # writer); this is the documented round-trip behaviour (#1809).
        assert int(da.attrs["nodata"]) == 245

    def test_dask_matches_eager(self, tmp_path):
        from xrspatial.geotiff import open_geotiff, read_geotiff_dask

        path = str(tmp_path / "miw_dask_2226.tif")
        self._build(path)
        eager = open_geotiff(path)
        lazy = read_geotiff_dask(path, chunks=2)
        np.testing.assert_array_equal(
            np.isnan(eager.data), np.isnan(lazy.compute().data),
        )
        assert lazy.attrs["nodata"] == eager.attrs["nodata"]

    def test_lifecycle_effective_sentinel_matches_reader_helper(self):
        from xrspatial.geotiff._reader import _miniswhite_inverted_nodata

        # Synthesize an IFD-like object for the legacy helper. Only
        # ``photometric`` and ``samples_per_pixel`` are read; build a
        # minimal stand-in rather than constructing a real IFD.
        class _IFDStub:
            photometric = 0
            samples_per_pixel = 1

        for sentinel, dtype in [
            (10, np.dtype("uint8")),
            (-9999, np.dtype("uint16")),  # out of range -> unchanged
            (3.5, np.dtype("float32")),
            (float("nan"), np.dtype("float32")),
        ]:
            ref = _miniswhite_inverted_nodata(sentinel, _IFDStub, dtype)
            lc = NodataLifecycle(
                declared=sentinel,
                photometric=0,
                dtype_in=dtype,
                samples_per_pixel=1,
            )
            if isinstance(ref, float) and np.isnan(ref):
                assert np.isnan(lc.effective_sentinel)
            else:
                assert lc.effective_sentinel == ref, (
                    f"sentinel={sentinel} dtype={dtype}: ref={ref}, "
                    f"lifecycle={lc.effective_sentinel}"
                )


# ----------------------- mask_nodata=False -----------------------

class TestMaskNodataFalseParity:
    def test_eager_keeps_literal_sentinel(self, int_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(int_tif, mask_nodata=False)
        # Buffer keeps integer dtype + literal sentinel pixel (255).
        assert da.dtype == np.uint8
        assert int(da.data[0, 2]) == 255
        # Lifecycle attr says masking did NOT occur.
        assert da.attrs["masked_nodata"] is False

    def test_dask_keeps_literal_sentinel(self, int_tif):
        from xrspatial.geotiff import read_geotiff_dask

        lazy = read_geotiff_dask(int_tif, chunks=2, mask_nodata=False)
        out = lazy.compute()
        assert out.dtype == np.uint8
        assert int(out.data[0, 2]) == 255
        assert lazy.attrs["masked_nodata"] is False

    @_gpu_only
    def test_gpu_keeps_literal_sentinel(self, int_tif):
        from xrspatial.geotiff import read_geotiff_gpu

        gpu = read_geotiff_gpu(int_tif, mask_nodata=False)
        host = gpu.data.get()
        assert host.dtype == np.uint8
        assert int(host[0, 2]) == 255
        assert gpu.attrs["masked_nodata"] is False


# ----------------------- explicit dtype= -----------------------

class TestExplicitDtypeRequestParity:
    def test_eager_records_dtype_cast(self, int_tif):
        from xrspatial.geotiff import open_geotiff

        da = open_geotiff(int_tif, dtype="float64")
        assert da.dtype == np.float64
        assert da.attrs.get("nodata_dtype_cast") == "float64"

    def test_dask_records_dtype_cast(self, int_tif):
        from xrspatial.geotiff import read_geotiff_dask

        lazy = read_geotiff_dask(int_tif, chunks=2, dtype="float64")
        assert lazy.dtype == np.float64
        assert lazy.attrs.get("nodata_dtype_cast") == "float64"


# ----------------------- VRT eager -----------------------

@pytest.fixture
def simple_vrt(tmp_path):
    """A trivial VRT wrapping a single GeoTIFF with a -9999 sentinel."""
    import os
    import xarray as xr

    from xrspatial.geotiff import to_geotiff

    src = str(tmp_path / "vrt_src_2226.tif")
    data = np.array(
        [[1.0, 2.0, -9999.0], [4.0, 5.0, 6.0]], dtype=np.float32,
    )
    da = xr.DataArray(
        data,
        coords={"y": np.array([0.5, 1.5]), "x": np.array([0.5, 1.5, 2.5])},
        dims=("y", "x"),
        attrs={"nodata": -9999.0},
    )
    to_geotiff(da, src, nodata=-9999.0)

    vrt = str(tmp_path / "vrt_eager_2226.vrt")
    xml = f"""<VRTDataset rasterXSize="3" rasterYSize="2">
  <GeoTransform>  0.0,  1.0,  0.0,  0.0,  0.0, 1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{os.path.basename(src)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="3" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""
    with open(vrt, "w") as f:
        f.write(xml)
    return vrt


class TestVRTEagerParity:
    def test_vrt_masks_sentinel_to_nan(self, simple_vrt):
        from xrspatial.geotiff import read_vrt

        da = read_vrt(simple_vrt)
        assert da.dtype.kind == "f"
        assert np.isnan(da.data[0, 2])

    def test_vrt_mask_nodata_false_keeps_literal(self, simple_vrt):
        from xrspatial.geotiff import read_vrt

        da = read_vrt(simple_vrt, mask_nodata=False)
        # Literal -9999 survives in the float buffer.
        assert da.data[0, 2] == -9999.0
        assert da.attrs.get("masked_nodata") is False


# ===========================================================================
# Writer-side restore decision parity
# ===========================================================================

class TestWriterRestoreParity:
    """Round-trip a float raster with NaN pixels through the writer.

    The lifecycle's ``writer_restore_sentinel`` controls whether NaN
    pixels are rewritten to the integer / float sentinel on disk. The
    eager and (when available) GPU writer must agree.
    """

    def test_eager_restores_nan_to_sentinel(self, tmp_path):
        import xarray as xr

        from xrspatial.geotiff import open_geotiff, to_geotiff

        path = str(tmp_path / "restore_eager_2226.tif")
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
        )
        to_geotiff(da, path, nodata=-9999.0)
        # Read back with mask_nodata=False so we can see the literal
        # on-disk byte value the restore step planted.
        readback = open_geotiff(path, mask_nodata=False)
        assert readback.data[0, 1] == -9999.0

    def test_masked_nodata_false_attr_blocks_restore(self, tmp_path):
        # Issue #1988: the reader stores masked_nodata=False to opt out
        # of the writer's NaN->sentinel rewrite. The lifecycle's
        # writer_restore_sentinel reads this through the
        # ``restore_sentinel`` kwarg the writer threads in.
        import xarray as xr

        from xrspatial.geotiff import open_geotiff, to_geotiff

        path = str(tmp_path / "no_restore_2226.tif")
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        )
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
            attrs={"nodata": -9999.0, "masked_nodata": False},
        )
        to_geotiff(da, path)
        readback = open_geotiff(path, mask_nodata=False)
        # Restore step skipped, so the NaN survives as on-disk NaN.
        assert np.isnan(readback.data[0, 1])

    @_gpu_only
    def test_gpu_writer_matches_eager(self, tmp_path):
        import xarray as xr
        import cupy

        from xrspatial.geotiff import open_geotiff, to_geotiff

        path = str(tmp_path / "restore_gpu_2226.tif")
        data = cupy.asarray(np.array(
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32,
        ))
        da = xr.DataArray(
            data,
            coords={
                "y": np.array([0.5, 1.5]),
                "x": np.array([0.5, 1.5, 2.5]),
            },
            dims=("y", "x"),
        )
        to_geotiff(da, path, nodata=-9999.0, gpu=True)
        readback = open_geotiff(path, mask_nodata=False)
        assert readback.data[0, 1] == -9999.0
