import numpy as np
import geopandas as gpd
import spatialpandas as spd

import datashader as ds

from xrspatial.utils import height_implied_by_aspect_ratio

import math
import shapely
from shapely.geometry import MultiPoint


# def polygon_to_points(gdf, npoints=10000, rasterize_width=1024):
#     """
#     Convert a spatialpandas.GeoDataFrame of (multi)polygon geometry to
#     a pandas.DataFrame containing `x`, and `y` columns.
#     Parameters
#     ----------
#     gdf: GeoDataFrame
#       spatialpandas.GeoDataFrame which contains polygon or multipolygon
#       in `geometry` column.
#     npoints: int, callable->int
#       number of point to random distribute across the polygon area.
#       defaults to 10,000 points
#     rasterize_width: int
#       hyperparameter which controls width of intermediate
#       raster representation of the polygon which is subsequently
#       sampled to get resultingg x, y locations.
#     Returns
#     -------
#     points_df: pd.DataFrame
#       pandas.DataFrame with `x`, and `y` columns. shape = (npoints x 2)
#     Notes
#     -----
#     The return pd.DataFrame only includes x, y columns.
#     If you want additional oolumns aggregated
#     to the resulting points, see `polygons_to_points`.
#     Example
#     -------
#     from xrspatial.conversion import polygon_to_points
#     points = polygon_to_points(my_polygon_geodataframe, npoints=1299000)
#     >>>
#     x             y
#     2.127132e+06  5.162847e+06
#     1.158690e+07  1.562231e+05
#     2.127132e+06  5.162847e+06
#     1.155180e+07  1.386560e+05
#     1.467580e+07  4.477730e+06
#              ...           ...
#     3.299768e+05  1.016735e+06
#     3.500137e+05  8.963614e+05
#     3.099399e+05  9.364860e+05
#     4.101243e+05  1.076922e+06
#     3.900874e+05  1.096985e+06
#     [1299000 rows x 2 columns]
#     """
#     W = rasterize_width
#     X = gdf['geometry'].bounds['x0'].min(),
#         gdf['geometry'].bounds['x1'].max()
#     Y = gdf['geometry'].bounds['y0'].min(),
#         gdf['geometry'].bounds['y1'].max()
#     H = height_implied_by_aspect_ratio(W, X, Y)

#     cvs = ds.Canvas(plot_width=W, plot_height=H,
#                     x_range=X, y_range=Y)

#     points_df = (cvs.polygons(gdf, geometry='geometry')
#                     .to_pandas()
#                     .stack()
#                     .reset_index())

#     points_df = points_df[points_df[0]]
#     rows = np.random.randint(low=0, high=len(points_df), size=npoints)
#     return points_df.iloc[rows, :][['x', 'y']]


# def polygons_to_points(gdf, npoints=1000,
#                        groupby=None, columns=None,
#                        ignore_group_errors=False):
#     """
#     Convert a spatialpandas.GeoDataFrame of (multi)polygon geometry to
#     a pandas.DataFrame with `x`, `y`, and additionl columns.
#     Additional `columns` are specified as either a list of strings
#     or a dictionary<column_name:lambda group_df: pd.Series>
#     Parameters
#     ----------
#     gdf: GeoDataFrame
#       spatialpandas.GeoDataFrame which contains polygon or multipolygon-typed
#       `geometry` column.
#     npoints: int, callable->int
#       Number of point to random distribute across the polygon area
#     groupby: int, callable->int
#       Number of point to random distribute across the polygon area
#     rasterize_width: int
#       Hyperparameter which controls width of
#       intermediate raster representation of the polygon
#       which is subsequently sampled to get resultingg x, y locations.
#     ignore_group_errors: bool
#       Silence errors that occur during conversion.
#       This can sometimes be helpful if there are corrupt geometries.
#     Returns
#     -------
#     points_df: pd.DataFrame
#       pandas.DataFrame with `x`, and `y` columns. shape = (npoints x 2)
#     Example
#     -------
#     from xrspatial.conversion import polygons_to_points
#     points = polygons_to_points(my_polygon_geodataframe, npoints=1299000,
#                                 groupby='ISO_SUB', columns=['ISO_SUB'])
#     >>>
#                             x             y ISO_SUB
#     0        2.127132e+06  5.162847e+06      AA
#     1        1.158690e+07  1.562231e+05      AA
#     2        2.127132e+06  5.162847e+06      AA
#     3        1.155180e+07  1.386560e+05      AA
#     4        1.467580e+07  4.477730e+06      AA
#     ...               ...           ...     ...
#     1298995 -3.299768e+05  1.016735e+06      ZZ
#     1298996 -3.500137e+05  8.963614e+05      ZZ
#     1298997 -3.099399e+05  9.364860e+05      ZZ
#     1298998 -4.101243e+05  1.076922e+06      ZZ
#     1298999 -3.900874e+05  1.096985e+06      ZZ
#     [1299000 rows x 3 columns]
#     """

#     if groupby is None:

#         points = polygon_to_points(gdf, npoints=npoints)

#         # add columns
#         if isinstance(columns, (list, tuple)):
#             for c in columns:
#                 points[c] = gdf[c].iloc[0]

#         elif isinstance(columns, dict):
#             for field, func in columns.items():
#                 points[field] = func(gdf)

#         return points

#     all_points = None
#     for label, group in gdf.groupby([groupby]):
#         try:
#             points = polygon_to_points(group, npoints=npoints)
#         except Exception as err:
#             print("Error while converting {0}: {1}".format(label, err))
#             if not ignore_group_errors:
#                 raise err

#         # add columns
#         if isinstance(columns, (list, tuple)):
#             for c in columns:
#                 points[c] = group[c].iloc[0]

#         elif isinstance(columns, dict):
#             for field, func in columns.items():
#                 points[field] = func(group)

#         if all_points is None:
#             all_points = points
#         else:
#             # TODO: Figure out how to remove copy...
#             # append doesn't have inplace arg
#             all_points = all_points.append(points, ignore_index=True)

#     return all_points


def _exp_geom_recalc_agg(gdf, agg_column):
    gdf_exp = gdf.explode('geometry')
    agg_column_np = gdf_exp[agg_column].values
    indices = [index[0] for index in gdf_exp.index]
    indices_counts = {}
    for index in indices:
        indices_counts[str(index)] = indices.count(index)
    new_list = []
    for idx in indices_counts.keys():
        val = agg_column_np[int(idx)] / indices_counts[idx]
        for n in range(int(indices_counts[idx])):
            new_list.append(val)
    new_np = np.asarray(new_list).astype(np.int64)
    gdf_exp[agg_column] = new_np
    gdf_exp = gdf_exp.reset_index(drop=True)
    return gdf_exp


def _get_scaled_agg(agg_column, num_points):
    if not isinstance(agg_column, np.ndarray):
        agg_column = np.asarray(agg_column)
    max_agg, min_agg = np.max(agg_column), np.min(agg_column)
    range_agg = max_agg - min_agg
    scaled = ((agg_column - min_agg) / range_agg) + (
                min_agg / range_agg) * num_points
    return scaled


def _polygons_to_pts(gdf: spd.GeoDataFrame, agg_column, num_points):
    pass


def polygons_to_pts(gdf: spd.GeoDataFrame, agg_column=None,
                    num_points=100, single_polygons=False):
    # TODO: switch from gpd to spd gdf

    # TODO: get and test geometry column; does not work correctly yet

    try:
        _ = gdf['geometry']
    except KeyError:
        # not working correctly; works for 20 and 10 to convert fun_stuff to
        # geometry
        for column in gdf.columns:
            if isinstance(gdf[column].all(), shapely.geometry.Polygon):
                if 'geometry' in list(gdf.columns):
                    raise ValueError('2 columns found for geometry; please '
                                     'choose one column containing '
                                     'multipolygon or polygon types and '
                                     'label it geoemetry')
                else:
                    gdf['geometry'] = gdf[column]
        if 'geometry' not in list(gdf.columns):
            raise KeyError('no geometry column found')

    # get scaling column to determine num_points
    if agg_column is not None:

        # validate agg_column key
        try:
            agg_gdf_column = gdf[agg_column]
        except KeyError:
            print('agg column not found')
            return

        # explode geometry after column key is validated
        if single_polygons:
            gdf = _exp_geom_recalc_agg(gdf, agg_column)
            agg_gdf_column = gdf[agg_column]

        # set num points to scaled agg column
        num_points = _get_scaled_agg(agg_gdf_column, num_points)

    # explode and get multipoints per polygon with intersect and scale with
    # num_points
    def _get_intersect_pts(row_idx, num_points_row, num_pts_found=None):

        polygons = gdf.iloc[row_idx]['geometry']

        max_x, min_y, min_x, max_y = polygons.bounds
        range_x, range_y = max_x - min_x, max_y - min_y

        area_scale = abs(range_x * range_y) / abs(polygons.area)
        num_points_scaled = abs(area_scale * num_points_row)

        if num_pts_found is not None:
            irregular_shape_offset = num_pts_found / num_points_scaled
            num_points_scaled = num_points_scaled / irregular_shape_offset

        num_points_sqrt = math.ceil(math.sqrt(num_points_scaled))
        x = np.linspace(min_x, max_x, num=num_points_sqrt)
        y = np.linspace(min_y, max_y, num=num_points_sqrt)

        X_Y = []
        for x_pt in x:
            for y_pt in y:
                X_Y.append((x_pt, y_pt))
        points_grid = MultiPoint(X_Y)
        intersect_pts = points_grid.intersection(polygons)

        return intersect_pts

    def _get_validate_intersect_pts(row_idx, num_points_row):
        intersect_pts = _get_intersect_pts(row_idx=row_idx,
                                           num_points_row=num_points_row)

        if intersect_pts.is_empty:
            num_points_scale = 1
            while intersect_pts.is_empty:
                num_points_scale *= 1.5
                intersect_pts = _get_intersect_pts(row_idx,
                                                   (num_pts_row *
                                                    num_points_scale))

        elif isinstance(intersect_pts, shapely.geometry.MultiPoint):
            if (len(intersect_pts) < num_points_row):
                num_pts_found = len(intersect_pts)
                intersect_pts = _get_intersect_pts(row_idx, num_points_row,
                                                   num_pts_found)

        elif isinstance(intersect_pts, shapely.geometry.Point):
            if num_points_row >= 2:
                num_pts_found = 1
                intersect_pts = _get_intersect_pts(row_idx, num_pts_row,
                                                   num_pts_found)

        else:
            raise ValueError(
                f'could not validate intersect points found at row {row_idx}')

        return intersect_pts

    multipts_list = []
    for row_idx in range(len(gdf)):

        if agg_column is not None:
            num_pts_row = num_points[row_idx]
            if num_pts_row == 0:
                num_pts_row = 1
        else:
            num_pts_row = num_points

        intersect_pts = _get_validate_intersect_pts(row_idx, num_pts_row)
        multipts_list.append(intersect_pts)

    multipts_gpd_df = gpd.GeoDataFrame({'geometry': multipts_list})
    singlepts_gpd_df = multipts_gpd_df.explode('geometry').reset_index(
        drop=True)
    singlepts_spd_df = spd.GeoDataFrame(singlepts_gpd_df, geometry='geometry')

    return singlepts_spd_df


def spd_df_to_agg(gdf: spd.GeoDataFrame):
    """
    must be single points spd gdf / maybe, depends on Canvas
    """
    maxx, maxy = gdf['geometry'].bounds['x1'].max(), gdf['geometry'].bounds[
        'y1'].max(),
    minx, miny = gdf['geometry'].bounds['x0'].min(), gdf['geometry'].bounds[
        'y0'].min(),
    W = 1000
    H = height_implied_by_aspect_ratio(W, (minx, maxx), (miny, maxy))
    cvs = ds.Canvas(plot_width=W, plot_height=H, x_range=(minx, maxx),
                    y_range=(miny, maxy))
    pts_agg = cvs.points(gdf, geometry='geometry')
    return pts_agg
