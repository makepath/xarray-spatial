import xarray as xr
import numpy as np

crs_attrs = { 'res': (10.0, 10.0), 'nodata': 0.0, 'grid_mapping': 'spatial_ref' }
spatial_ref_attrs_EPSG4326 = {
        'crs_wkt': 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]',
        'semi_major_axis': 6378137.0,
        'semi_minor_axis': 6356752.314245179,
        'inverse_flattening': 298.257223563,
        'reference_ellipsoid_name': 'WGS 84',
        'longitude_of_prime_meridian': 0.0,
        'prime_meridian_name': 'Greenwich',
        'geographic_crs_name': 'WGS 84',
        'grid_mapping_name': 'latitude_longitude',
        'spatial_ref': 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]'
}

spatial_ref_coords_da = xr.DataArray(0, attrs=spatial_ref_attrs_EPSG4326)

def _add_EPSG4326_crs_to_da(da):
        crs_da = xr.DataArray(np.empty_like(da.data))
        crs_da.data = da.data
        crs_da.attrs = crs_attrs
        crs_da.coords['spatial_ref'] = spatial_ref_coords_da
        return crs_da
