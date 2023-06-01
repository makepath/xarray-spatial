import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import ArrayTypeFunctionMapping, has_cuda_and_cupy

# Use this as a decorator to skip tests if do not have both CUDA and CuPy available.
cuda_and_cupy_available = pytest.mark.skipif(
    not has_cuda_and_cupy(), reason="Requires CUDA and CuPy")


def create_test_raster(
        data,
        backend='numpy',
        name='myraster',
        dims=['y', 'x'],
        attrs={'res': (0.5, 0.5), 'crs': 'EPSG: 4326'},
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

    raster[dims[0]] = np.linspace((data.shape[0] - 1)/2, 0, data.shape[0])
    raster[dims[1]] = np.linspace(0, (data.shape[1] - 1)/2, data.shape[1])

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
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

    if isinstance(input_agg.data, da.Array):
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


def assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, func, nan_edges=True):
    numpy_result = func(numpy_agg)
    if nan_edges:
        assert_nan_edges_effect(numpy_result)

    dask_cupy_result = func(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result)
    np.testing.assert_allclose(
        numpy_result.data, dask_cupy_result.data.compute().get(), equal_nan=True
    )
