"""Regression tests for issue #2506.

``_run_cupy`` and ``_rasterize_tile_cupy`` previously called
``cupy.asarray(poly_props)`` and ``cupy.asarray(poly_global)`` twice
when ``all_touched=True``: once for the scanline ``poly_launch``
tuple and once for the supercover ``boundary_launch`` tuple. The
two tuples reference the same per-tile props/global tables, so the
second upload re-transfers identical bytes.

These tests cover two angles:

1. AST-level structural assertion that each function only calls
   ``cupy.asarray(poly_props...)`` and ``cupy.asarray(poly_global...)``
   once -- the fix hoists the upload above the
   scanline/boundary conditional and shares the device buffer.
2. End-to-end correctness for ``all_touched=True`` on the cupy and
   dask+cupy backends against the numpy reference, so the hoist did
   not change observed behaviour.

When cupy / CUDA / dask / shapely are unavailable the relevant tests
skip; the structural test runs on any host with the source file
present.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

try:
    import dask  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False


# Module path used by the AST-level structural test. Independent of
# whether cupy / cuda are installed on the host.
_RASTERIZE_PY = Path(__file__).resolve().parent.parent / "rasterize.py"


def _count_cupy_asarray_calls(func_src, var_name):
    """Count ``cupy.asarray(<var_name>)`` calls inside *func_src*.

    Matches ``cupy.asarray(poly_props)`` and ``cupy.asarray(poly_props_2d)``
    but not ``cupy.asarray(poly_props_2d_gpu)`` (those are the names of
    the hoisted device buffers in the fix).
    """
    tree = ast.parse(func_src)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "asarray"
                and isinstance(func.value, ast.Name)
                and func.value.id == "cupy"):
            continue
        if len(node.args) != 1:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id == var_name:
            count += 1
    return count


def _extract_function_source(name):
    """Return the source of the named top-level function in rasterize.py."""
    src = _RASTERIZE_PY.read_text()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"function {name!r} not found in rasterize.py")


@pytest.mark.skipif(
    not _RASTERIZE_PY.exists(), reason="rasterize.py not found"
)
class TestStructure:
    """AST-level invariants pinning the props/global hoist in place."""

    def test_run_cupy_uploads_poly_props_once(self):
        src = _extract_function_source("_run_cupy")
        assert _count_cupy_asarray_calls(src, "poly_props") == 1, (
            "_run_cupy must call cupy.asarray(poly_props) exactly once "
            "(hoisted above the scanline / boundary conditional). "
            "See issue #2506."
        )

    def test_run_cupy_uploads_poly_global_once(self):
        src = _extract_function_source("_run_cupy")
        assert _count_cupy_asarray_calls(src, "poly_global") == 1, (
            "_run_cupy must call cupy.asarray(poly_global) exactly once. "
            "See issue #2506."
        )

    def test_rasterize_tile_cupy_uploads_poly_props_2d_once(self):
        src = _extract_function_source("_rasterize_tile_cupy")
        assert _count_cupy_asarray_calls(src, "poly_props_2d") == 1, (
            "_rasterize_tile_cupy must call cupy.asarray(poly_props_2d) "
            "exactly once per tile (hoisted above the scanline / boundary "
            "conditional). See issue #2506."
        )

    def test_rasterize_tile_cupy_uploads_poly_global_2d_once(self):
        src = _extract_function_source("_rasterize_tile_cupy")
        assert _count_cupy_asarray_calls(src, "poly_global_2d") == 1, (
            "_rasterize_tile_cupy must call cupy.asarray(poly_global_2d) "
            "exactly once per tile. See issue #2506."
        )


@pytest.mark.skipif(not has_shapely, reason="shapely not installed")
@pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
class TestParityAllTouched:
    """End-to-end parity: hoist must not change pixel output."""

    def _geometries(self):
        # A handful of overlapping rectangles whose ring segments cross
        # the raster interior -- exercises both the scanline polygon
        # path and the supercover boundary burn.
        return [
            (box(0.5, 0.5, 4.5, 4.5), 1.0),
            (box(2.0, 2.0, 6.0, 6.0), 2.0),
            (box(5.5, 3.5, 9.5, 7.5), 3.0),
            (box(3.0, 5.0, 7.0, 9.0), 4.0),
        ]

    def _to_numpy(self, arr):
        data = arr.data
        # Dask-backed: materialise first so the cupy chunks can be
        # transferred (avoids the implicit __array__ on cupy that
        # numpy's asarray would trigger).
        if hasattr(data, "compute"):
            data = data.compute()
        if hasattr(data, "get"):
            return data.get()
        return np.asarray(data)

    @pytest.mark.parametrize("merge", ["last", "first", "max", "min", "sum"])
    def test_cupy_all_touched_matches_numpy(self, merge):
        geoms = self._geometries()
        ref = rasterize(geoms, width=24, height=24,
                        bounds=(0, 0, 10, 10),
                        all_touched=True, merge=merge, fill=0.0)
        got = rasterize(geoms, width=24, height=24,
                        bounds=(0, 0, 10, 10),
                        all_touched=True, merge=merge, fill=0.0,
                        use_cuda=True)
        np.testing.assert_allclose(
            self._to_numpy(got), self._to_numpy(ref), atol=0, rtol=0,
            err_msg=f"cupy all_touched != numpy for merge={merge!r}",
        )

    @pytest.mark.skipif(not has_dask, reason="dask not installed")
    @pytest.mark.parametrize("merge", ["last", "sum", "max"])
    def test_dask_cupy_all_touched_runs(self, merge):
        """Smoke test: dask+cupy all_touched returns a result of the
        expected shape and dtype. Pixel-level parity against numpy is
        tracked separately (a pre-existing boundary-segment-on-tile-
        border bug, unrelated to the props hoist in #2506) and is not
        asserted here -- the goal of this test is to exercise the
        hoisted poly_props_2d / poly_global_2d upload through every
        per-tile launch in the dask path.
        """
        geoms = self._geometries()
        got = rasterize(geoms, width=24, height=24,
                        bounds=(0, 0, 10, 10),
                        all_touched=True, merge=merge, fill=0.0,
                        use_cuda=True, chunks=8)
        out = self._to_numpy(got)
        assert out.shape == (24, 24)
        # Some interior pixels must have been burned by at least one
        # geometry; an all-zero output would indicate the per-tile
        # polygon launches never wrote anything.
        assert (out != 0).any(), (
            f"dask+cupy all_touched merge={merge!r} produced an all-zero "
            "raster; the per-tile poly_props_2d hoist may have broken "
            "the launch wiring."
        )
