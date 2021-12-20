import numpy as np
import dask.array as da


def general_output_checks(input_agg, output_agg, expected_results=None):

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
        assert np.all(output_agg[coord] == input_agg[coord])

    if expected_results is not None:
        if isinstance(input_agg.data, da.Array):
            assert np.isclose(
                output_agg.data.compute(),
                expected_results.data,
                equal_nan=True
            ).all()
        else:
            assert np.isclose(
                output_agg.data, expected_results.data, equal_nan=True
            ).all()
