import numpy as np
import dask.array as da
import xarray as xr

from xrspatial.utils import ArrayTypeFunctionMapping


def general_output_checks(input_agg: xr.DataArray,
                          output_agg: xr.DataArray,
                          expected_results: np.ndarray = None):

    # type of output is the same as of input
    assert isinstance(output_agg.data, type(input_agg.data))

    if isinstance(input_agg.data, da.Array):
        # dask case
        assert isinstance(
            output_agg.data.compute(), type(input_agg.data.compute()))

    # shape and other attributes remain the same
    assert output_agg.shape == input_agg.shape
    assert output_agg.dims == input_agg.dims
    assert output_agg.attrs == input_agg.attrs
    for coord in input_agg.coords:
        np.testing.assert_allclose(
            output_agg[coord].data, input_agg[coord].data, equal_nan=True
        )

    if expected_results is not None:
        numpy_func = lambda output, expected: np.testing.assert_allclose(
            output, expected_results, equal_nan=True
        )
        dask_func = lambda output, expected: np.testing.assert_allclose(
            output.compute(), expected_results, equal_nan=True
        )
        cupy_func = lambda output, expected: np.testing.assert_allclose(
            output.get(), expected_results, equal_nan=True
        )
        dask_cupy_func = lambda output, expected: np.testing.assert_allclose(
            output.compute().get(), expected_results, equal_nan=True
        )
        mapper = ArrayTypeFunctionMapping(
            numpy_func=numpy_func,
            dask_func=dask_func,
            cupy_func=cupy_func,
            dask_cupy_func=dask_cupy_func,
        )
        mapper(input_agg)(output_agg.data, expected_results)
