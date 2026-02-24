from __future__ import annotations
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import has_cuda_and_cupy
from xrspatial.utils import has_dask_array
from xrspatial.utils import has_dask_dataframe

# Use this as a decorator to skip tests if do not have both CUDA and CuPy available.
cuda_and_cupy_available = pytest.mark.skipif(
    not has_cuda_and_cupy(), reason="Requires CUDA and CuPy")


# Use this as a decorator to skip tests if do not have dask array
dask_array_available = pytest.mark.skipif(
    not has_dask_array(), reason="Requires dask.Array")

# Use this as a decorator to skip tests if do not have dask array
dask_dataframe_available = pytest.mark.skipif(
    not has_dask_dataframe(), reason="Requires dask.DataFrame")


def create_test_raster(
        data,
        backend='numpy',
        name='myraster',
        dims=['y', 'x'],
        attrs={'res': (0.5, 0.5), 'crs': 'EPSG: 5070'},
        chunks=(3, 3)
):
    raster = xr.DataArray(data, name=name, dims=dims, attrs=attrs)

    # default res if none provided
    res = (0.5, 0.5)
    if attrs is not None:
        if 'res' in attrs:
            res = attrs['res']

    # set coords for test raster, 2D coords only
    raster[dims[0]] = np.linspace((data.shape[0] - 1) * res[0], 0, data.shape[0])
    raster[dims[1]] = np.linspace(0, (data.shape[1] - 1) * res[1], data.shape[1])

    # assign units to coords
    raster[dims[0]].attrs["units"] = "m"
    raster[dims[1]].attrs["units"] = "m"

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend and has_dask_array():
        raster.data = da.from_array(raster.data, chunks=chunks)

    return raster


def general_output_checks(input_agg: xr.DataArray,
                          output_agg: xr.DataArray,
                          expected_results: np.ndarray = None,
                          verify_attrs: bool = True,
                          verify_dtype: bool = False,
                          rtol=1e-06):

    # type of output is the same as of input
    assert isinstance(output_agg.data, type(input_agg.data))

    if has_dask_array() and isinstance(input_agg.data, da.Array):
        # dask case
        assert isinstance(
            output_agg.data.compute(), type(input_agg.data.compute()))

    if verify_attrs:
        # shape and other attributes remain the same
        assert output_agg.shape == input_agg.shape
        assert output_agg.dims == input_agg.dims
        assert output_agg.attrs == input_agg.attrs
        for coord in input_agg.coords:
            np.testing.assert_allclose(
                output_agg[coord].data, input_agg[coord].data, equal_nan=True
            )

    if expected_results is not None:
        get_numpy_data = lambda output: output  # noqa: E731
        get_dask_numpy_data = lambda output: output.compute()  # noqa: E731
        get_cupy_data = lambda output: output.get()  # noqa: E731
        get_dask_cupy_data = lambda output: output.compute().get()  # noqa: E731

        mapper = ArrayTypeFunctionMapping(
            numpy_func=get_numpy_data,
            dask_func=get_dask_numpy_data,
            cupy_func=get_cupy_data,
            dask_cupy_func=get_dask_cupy_data,
        )
        output_data = mapper(output_agg)(output_agg.data)
        np.testing.assert_allclose(output_data, expected_results, equal_nan=True, rtol=rtol)

        if verify_dtype:
            assert output_data.dtype == expected_results.dtype


def assert_input_data_unmodified(data_before, data_after):
    assert data_before.equals(data_after)


def assert_nan_edges_effect(result_agg):
    # nan edge effect
    edges = [
        result_agg.data[0, :],
        result_agg.data[-1, :],
        result_agg.data[:, 0],
        result_agg.data[:, -1],
    ]
    for edge in edges:
        np.testing.assert_array_equal(edge, np.nan)


def assert_boundary_mode_correctness(numpy_agg, dask_agg, func, depth=1, rtol=1e-5,
                                     nan_edges=True):
    """Verify that all boundary modes produce correct output.

    Checks:
    - 'nan' mode: edges are NaN (preserves existing behaviour) if nan_edges=True.
    - 'nearest', 'reflect', 'wrap': shape matches input and
      numpy matches dask result. When source data has no NaN,
      edge cells should also have no NaN.

    Parameters
    ----------
    nan_edges : bool
        If True, verify that boundary='nan' produces NaN at edges.
        Set to False for functions like mean/apply/hotspots that don't
        produce NaN edges even with boundary='nan'.
    """
    from xrspatial.utils import VALID_BOUNDARY_MODES
    from functools import partial as _partial

    source_has_nan = np.any(np.isnan(numpy_agg.data))

    for mode in VALID_BOUNDARY_MODES:
        _func = _partial(func, boundary=mode)
        result_np = _func(numpy_agg)

        assert result_np.shape == numpy_agg.shape, (
            f"boundary={mode!r}: shape mismatch "
            f"{result_np.shape} vs {numpy_agg.shape}"
        )

        if mode == 'nan' and nan_edges:
            assert_nan_edges_effect(result_np)
        elif mode != 'nan' and not source_has_nan:
            result_data = result_np.data
            if has_dask_array() and isinstance(result_data, da.Array):
                result_data = result_data.compute()
            assert not np.any(np.isnan(result_data)), (
                f"boundary={mode!r}: output should have no NaN values "
                f"when source data has none"
            )

        if dask_agg is not None and has_dask_array():
            result_da = _func(dask_agg)
            da_data = result_da.data
            if isinstance(da_data, da.Array):
                da_data = da_data.compute()
            np.testing.assert_allclose(
                result_np.data, da_data,
                equal_nan=True, rtol=rtol,
            )


def assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, func, nan_edges=True):
    numpy_result = func(numpy_agg)
    if nan_edges:
        assert_nan_edges_effect(numpy_result)

    dask_result = func(dask_agg)
    general_output_checks(dask_agg, dask_result)
    np.testing.assert_allclose(numpy_result.data, dask_result.data.compute(), equal_nan=True)


def assert_numpy_equals_cupy(numpy_agg, cupy_agg, func, nan_edges=True, atol=0, rtol=1e-7):
    numpy_result = func(numpy_agg)
    if nan_edges:
        assert_nan_edges_effect(numpy_result)

    cupy_result = func(cupy_agg)
    general_output_checks(cupy_agg, cupy_result)
    np.testing.assert_allclose(
        numpy_result.data, cupy_result.data.get(), equal_nan=True, atol=atol, rtol=rtol)


def assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, func,
                                  nan_edges=True, atol=0, rtol=1e-7):
    numpy_result = func(numpy_agg)
    if nan_edges:
        assert_nan_edges_effect(numpy_result)

    dask_cupy_result = func(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result)
    np.testing.assert_allclose(numpy_result.data, dask_cupy_result.data.compute().get(),
                               equal_nan=True, atol=atol, rtol=rtol)
