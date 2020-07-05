import numpy as np
import datashader as ds

from xrspatial.utils import height_implied_by_aspect_ratio


def polygon_to_points(gdf, npoints=10000, rasterize_width=1024):
    """
    Convert a spatialpandas.GeoDataFrame of (multi)polygon geometry to
    a pandas.DataFrame containing `x`, and `y` columns.

    Parameters
    ----------
    gdf: GeoDataFrame
      spatialpandas.GeoDataFrame which contains polygon or multipolygon
      in `geometry` column.
    npoints: int, callable->int
      number of point to random distribute across the polygon area.
      defaults to 10,000 points
    rasterize_width: int
      hyperparameter which controls width of intermediate
      raster representation of the polygon which is subsequently
      sampled to get resultingg x, y locations.

    Returns
    -------
    points_df: pd.DataFrame
      pandas.DataFrame with `x`, and `y` columns. shape = (npoints x 2)

    Notes
    -----
    The return pd.DataFrame only includes x, y columns.

    If you want additional oolumns aggregated
    to the resulting points, see `polygons_to_points`.

    Example
    -------
    from xrspatial.conversion import polygon_to_points

    points = polygon_to_points(my_polygon_geodataframe, npoints=1299000)
    >>>
    x             y
    2.127132e+06  5.162847e+06
    1.158690e+07  1.562231e+05
    2.127132e+06  5.162847e+06
    1.155180e+07  1.386560e+05
    1.467580e+07  4.477730e+06
             ...           ...
    3.299768e+05  1.016735e+06
    3.500137e+05  8.963614e+05
    3.099399e+05  9.364860e+05
    4.101243e+05  1.076922e+06
    3.900874e+05  1.096985e+06

    [1299000 rows x 2 columns]
    """
    W = rasterize_width
    X = gdf['geometry'].bounds['x0'].min(), gdf['geometry'].bounds['x1'].max()
    Y = gdf['geometry'].bounds['y0'].min(), gdf['geometry'].bounds['y1'].max()
    H = height_implied_by_aspect_ratio(W, X, Y)

    cvs = ds.Canvas(plot_width=W, plot_height=H,
                    x_range=X, y_range=Y)

    points_df = (cvs.polygons(gdf, geometry='geometry')
                    .to_pandas()
                    .stack()
                    .reset_index())

    points_df = points_df[points_df[0]]
    rows = np.random.randint(low=0, high=len(points_df), size=npoints)
    return points_df.iloc[rows, :][['x', 'y']]


def polygons_to_points(gdf, npoints=1000, groupby=None, columns=None, ignore_group_errors=False):
    """
    Convert a spatialpandas.GeoDataFrame of (multi)polygon geometry to
    a pandas.DataFrame with `x`, `y`, and additionl columns.

    Additional `columns` are specified as either a list of strings
    or a dictionary<column_name:lambda group_df: pd.Series>

    Parameters
    ----------

    gdf: GeoDataFrame
      spatialpandas.GeoDataFrame which contains polygon or multipolygon-typed
      `geometry` column.

    npoints: int, callable->int
      Number of point to random distribute across the polygon area

    groupby: int, callable->int
      Number of point to random distribute across the polygon area

    rasterize_width: int
      Hyperparameter which controls width of
      intermediate raster representation of the polygon
      which is subsequently sampled to get resultingg x, y locations.

    ignore_group_errors: bool
      silence errors that occur during conversation.
      this can sometimes be helpful if there are corrupt geometry.

    Returns
    -------
    points_df: pd.DataFrame
      pandas.DataFrame with `x`, and `y` columns. shape = (npoints x 2)

    Example
    -------
    from xrspatial.conversion import polygons_to_points

    points = polygons_to_points(my_polygon_geodataframe, npoints=1299000,
                                groupby='ISO_SUB', columns=['ISO_SUB'])

    >>>
                            x             y ISO_SUB
    0        2.127132e+06  5.162847e+06      AA
    1        1.158690e+07  1.562231e+05      AA
    2        2.127132e+06  5.162847e+06      AA
    3        1.155180e+07  1.386560e+05      AA
    4        1.467580e+07  4.477730e+06      AA
    ...               ...           ...     ...
    1298995 -3.299768e+05  1.016735e+06      ZZ
    1298996 -3.500137e+05  8.963614e+05      ZZ
    1298997 -3.099399e+05  9.364860e+05      ZZ
    1298998 -4.101243e+05  1.076922e+06      ZZ
    1298999 -3.900874e+05  1.096985e+06      ZZ

    [1299000 rows x 3 columns]
    """

    if groupby is None:

        points = polygon_to_points(gdf, npoints=npoints)

        # add columns
        if isinstance(columns, (list, tuple)):
            for c in columns:
                points[c] = gdf[c].iloc[0]

        elif isinstance(columns, dict):
            for field, func in columns.items():
                points[field] = func(gdf)

        return points

    all_points = None
    for label, group in gdf.groupby([groupby]):
        try:
            points = polygon_to_points(group, npoints=npoints)
        except Exception as err:
            print("Error while convertingk {0}: {1}".format(label, err))
            if not ignore_group_errors:
                raise err

        # add columns
        if isinstance(columns, (list, tuple)):
            for c in columns:
                points[c] = group[c].iloc[0]

        elif isinstance(columns, dict):
            for field, func in columns.items():
                points[field] = func(group)

        if all_points is None:
            all_points = points
        else:
            # TODO: Figure out how to remove copy...
            # append doesn't have inplace arg
            all_points = all_points.append(points, ignore_index=True)

    return all_points
