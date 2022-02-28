import xarray as xr

from xrspatial import aspect, curvature, slope


def summarize_terrain(terrain: xr.DataArray):
    """
    Calculates slope, aspect, and curvature of an elevation terrain and return a dataset
    of the computed data.

    Parameters
    ----------
    terrain: xarray.DataArray
        2D NumPy, CuPy, or Dask with NumPy-backed xarray DataArray of elevation values.

    Returns
    -------
    summarized_terrain: xarray.Dataset
        Dataset with slope, aspect, curvature variables with a naming convention of
        `terrain.name-variable_name`

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.analytics import summarize_terrain
        >>> data = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
        >>> raster = xr.DataArray(data, name='myraster', attrs={'res': (1, 1)})
        >>> summarized_terrain = summarize_terrain(raster)
        >>> summarized_terrain
        <xarray.Dataset>
        Dimensions:             (dim_0: 5, dim_1: 8)
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            myraster            (dim_0, dim_1) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0
            myraster-slope      (dim_0, dim_1) float32 nan nan nan nan ... nan nan nan
            myraster-curvature  (dim_0, dim_1) float64 nan nan nan nan ... nan nan nan
            myraster-aspect     (dim_0, dim_1) float32 nan nan nan nan ... nan nan nan
        >>> summarized_terrain['myraster-slope']
        <xarray.DataArray 'myraster-slope' (dim_0: 5, dim_1: 8)>
        array([[      nan,       nan,       nan,       nan,       nan,       nan,       nan,   nan],
               [      nan, 10.024988, 14.036243, 10.024988, 10.024988, 14.036243, 10.024988,   nan],
               [      nan, 14.036243,  0.      , 14.036243, 14.036243,  0.      , 14.036243,   nan],
               [      nan, 10.024988, 14.036243, 10.024988, 10.024988, 14.036243, 10.024988,   nan],
               [      nan,       nan,       nan,       nan,       nan,       nan,       nan,   nan]], dtype=float32)  # noqa
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (1, 1)

        >>> summarized_terrain['myraster-curvature']
        <xarray.DataArray 'myraster-curvature' (dim_0: 5, dim_1: 8)>
        array([[  nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan],
               [  nan,   -0., -100.,   -0.,   -0.,  100.,   -0.,   nan],
               [  nan, -100.,  400., -100.,  100., -400.,  100.,   nan],
               [  nan,   -0., -100.,   -0.,   -0.,  100.,   -0.,   nan],
               [  nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan]])
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (1, 1)

        >>> summarized_terrain['myraster-aspect']
        <xarray.DataArray 'myraster-aspect' (dim_0: 5, dim_1: 8)>
        array([[ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan],
               [ nan, 315.,   0.,  45., 135., 180., 225.,  nan],
               [ nan, 270.,  -1.,  90.,  90.,  -1., 270.,  nan],
               [ nan, 225., 180., 135.,  45.,   0., 315.,  nan],
               [ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (1, 1)
    """

    if terrain.name is None:
        raise NameError('Requires xr.DataArray.name property to be set')

    ds = terrain.to_dataset()
    ds[f'{terrain.name}-slope'] = slope(terrain)
    ds[f'{terrain.name}-curvature'] = curvature(terrain)
    ds[f'{terrain.name}-aspect'] = aspect(terrain)
    return ds
