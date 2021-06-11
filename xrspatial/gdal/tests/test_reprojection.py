import rioxarray
import os
from xrspatial.gdal.reprojection import reproject
import pytest  # noqa: F401


elev_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            '..',
            'datasets',
            'elevation.tif')
earth_ll = rioxarray.open_rasterio(elev_path)
dst_crs = 'EPSG:3857'
reprojected_rio = earth_ll.rio.reproject(dst_crs)


def test_reprojection_numpy():
    reprojected = reproject(earth_ll, dst_crs)

    assert reprojected.rio.crs == reprojected_rio.rio.crs
    assert (reprojected.data == reprojected_rio.data).all()
    assert reprojected.data.shape == reprojected_rio.data.shape
    assert reprojected.rio.bounds() == reprojected_rio.rio.bounds()


def test_reprojection_dask_one_chunk():
    earth_ll_chunk = earth_ll.chunk()
    reprojected = reproject(earth_ll_chunk, dst_crs)

    assert reprojected.rio.crs == reprojected_rio.rio.crs
    assert (reprojected.data == reprojected_rio.data).all()
    assert reprojected.data.shape == reprojected_rio.data.shape
    assert reprojected.rio.bounds() == reprojected_rio.rio.bounds()


# def test_reprojection_dask_chunks():
#     earth_ll_chunks = earth_ll.chunk((1, 135, 270))

#     reprojected = reproject(earth_ll_chunks, dst_crs)

#     assert reprojected.rio.crs == reprojected_rio.rio.crs
#     assert (reprojected.data == reprojected_rio.data).all()
#     assert reprojected.data.shape == reprojected_rio.data.shape
#     assert reprojected.rio.bounds() == reprojected_rio.rio.bounds()
