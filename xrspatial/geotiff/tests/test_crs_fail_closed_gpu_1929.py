"""GPU + dispatcher backend coverage for issue #1929.

#1929 added ``_validate_crs_fallback`` and wired
``allow_unparseable_crs`` into ``to_geotiff``, ``write_geotiff_gpu``,
and the ``to_geotiff(gpu=True)`` dispatcher. ``test_crs_fail_closed_1929``
only exercises the eager CPU writer (``to_geotiff(gpu=False, ...)``);
the GPU writer's invocation of ``_validate_crs_fallback`` at
``_writers/gpu.py:507`` and the dispatcher thread-through at
``_writers/eager.py:447`` have no targeted tests.

A regression dropping either call would let
``write_geotiff_gpu(..., crs="EPSG:4326")`` on a host without pyproj,
or any other unparseable CRS string, silently emit a garbage citation
field that non-libgeotiff readers cannot interpret. The eager test
catches the CPU path; this module closes the GPU and dispatcher gap.
"""
from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
pytestmark = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


def _make_gpu_da() -> xr.DataArray:
    """Build a tiny CuPy-backed DataArray for the GPU writer."""
    import cupy

    arr = cupy.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


def _make_cpu_da() -> xr.DataArray:
    """Numpy-backed twin used to exercise the to_geotiff(gpu=True) path."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


class TestWriteGeotiffGpuFailClosed:
    """``write_geotiff_gpu`` refuses to land an unvalidatable CRS by default."""

    def test_garbage_string_kwarg_raises(self, tmp_path):
        """A free-form non-WKT, non-PROJ string raises by default."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_garbage_kwarg_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                write_geotiff_gpu(
                    _make_gpu_da(), out, crs="absolute-garbage")

    def test_garbage_string_attr_raises(self, tmp_path):
        """Same guard fires when garbage arrives via ``attrs['crs']``."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_garbage_attr_1929.tif")
        da = _make_gpu_da()
        da.attrs["crs"] = "still-garbage"
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                write_geotiff_gpu(da, out)

    def test_opt_in_allows_garbage(self, tmp_path):
        """``allow_unparseable_crs=True`` restores the citation-only write."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_optin_1929.tif")
        with pytest.warns(Warning):
            write_geotiff_gpu(
                _make_gpu_da(),
                out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)

    def test_message_recommends_alternatives(self, tmp_path):
        """The error message points to the four recovery options."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_msg_check_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError) as exc:
                write_geotiff_gpu(_make_gpu_da(), out, crs="bogus")
        msg = str(exc.value)
        assert "EPSG" in msg
        assert "WKT" in msg
        assert "allow_unparseable_crs" in msg

    def test_epsg_int_unchanged(self, tmp_path):
        """An int EPSG kwarg never reaches the fallback path."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_epsg_int_1929.tif")
        write_geotiff_gpu(_make_gpu_da(), out, crs=4326)
        assert os.path.exists(out)

    def test_valid_wkt_unchanged(self, tmp_path):
        """A WKT-shaped string is accepted by the structural check."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_wkt_shaped_1929.tif")
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
            "6378137,298.257223563]],PRIMEM[\"Greenwich\",0],"
            "UNIT[\"degree\",0.0174532925199433]]"
        )
        write_geotiff_gpu(_make_gpu_da(), out, crs=wkt)
        assert os.path.exists(out)

    def test_no_crs_at_all_unchanged(self, tmp_path):
        """No CRS supplied means no GTCitationGeoKey; the validator is a no-op."""
        from xrspatial.geotiff import write_geotiff_gpu

        out = str(tmp_path / "gpu_no_crs_1929.tif")
        write_geotiff_gpu(_make_gpu_da(), out)
        assert os.path.exists(out)


class TestToGeotiffGpuDispatcherFailClosed:
    """``to_geotiff(gpu=True)`` threads the validator through to the GPU writer.

    The dispatcher branch at ``_writers/eager.py:447`` forwards
    ``allow_unparseable_crs`` into ``write_geotiff_gpu``. A regression
    dropping the forward (e.g. an accidental kwarg-rename or a missed
    keyword in a refactor) would silently bypass the validator while
    looking like everything still works.
    """

    def test_dispatcher_garbage_raises_with_cupy_input(self, tmp_path):
        """CuPy-backed input auto-routes to the GPU writer."""
        from xrspatial.geotiff import to_geotiff

        out = str(tmp_path / "dispatcher_garbage_gpu_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(_make_gpu_da(), out, crs="absolute-garbage")

    def test_dispatcher_gpu_kwarg_garbage_raises(self, tmp_path):
        """Explicit ``gpu=True`` on numpy data also threads through."""
        from xrspatial.geotiff import to_geotiff

        out = str(tmp_path / "dispatcher_gpu_kwarg_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(
                    _make_cpu_da(), out, gpu=True, crs="absolute-garbage")

    def test_dispatcher_opt_in_forwarded(self, tmp_path):
        """``allow_unparseable_crs=True`` is forwarded into the GPU writer."""
        from xrspatial.geotiff import to_geotiff

        out = str(tmp_path / "dispatcher_optin_gpu_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_gpu_da(),
                out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)

    def test_dispatcher_opt_in_explicit_gpu(self, tmp_path):
        """``to_geotiff(gpu=True, allow_unparseable_crs=True)`` also works."""
        from xrspatial.geotiff import to_geotiff

        out = str(tmp_path / "dispatcher_optin_gpu_explicit_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_cpu_da(),
                out,
                gpu=True,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)


class TestErrorMessageParity:
    """The GPU and eager error messages match for the same input.

    A user catching ``ValueError`` from ``to_geotiff`` should see the
    same message whether the backend is CPU or GPU. A cross-backend
    drift would force callers to special-case the error string.
    """

    def test_gpu_vs_cpu_message_matches(self, tmp_path):
        from xrspatial.geotiff import to_geotiff, write_geotiff_gpu

        out_cpu = str(tmp_path / "cpu_msg_1929.tif")
        out_gpu = str(tmp_path / "gpu_msg_1929.tif")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_cpu:
                to_geotiff(_make_cpu_da(), out_cpu, crs="absolute-garbage")
            with pytest.raises(ValueError) as exc_gpu:
                write_geotiff_gpu(
                    _make_gpu_da(), out_gpu, crs="absolute-garbage")
        assert str(exc_cpu.value) == str(exc_gpu.value)


class TestKwargDefaultParity:
    """``allow_unparseable_crs`` defaults to False on every writer.

    A drift in the default (e.g. one writer setting True for back-compat
    while another stays False) would silently re-open the #1929 hole on
    just one backend. Pin the canonical default explicitly.
    """

    def test_default_is_false_on_all_writers(self):
        import inspect

        from xrspatial.geotiff import to_geotiff, write_geotiff_gpu

        for fn in (to_geotiff, write_geotiff_gpu):
            sig = inspect.signature(fn)
            param = sig.parameters.get("allow_unparseable_crs")
            assert param is not None, (
                f"{fn.__name__} must accept allow_unparseable_crs"
            )
            assert param.default is False, (
                f"{fn.__name__}.allow_unparseable_crs default "
                f"drifted to {param.default!r}; #1929 requires fail-closed."
            )
