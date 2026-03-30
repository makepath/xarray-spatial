import numpy as np
import pytest
import xarray as xr

from xrspatial.multispectral import (arvi, bai, ebbi, evi, gci, mndwi, msavi2, nbr, nbr2, ndbi,
                                     ndmi, ndsi, ndvi, ndwi, osavi, savi, sipi, true_color)
from xrspatial.tests.general_checks import (create_test_raster, cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)


@pytest.fixture
def blue_data(backend):
    data = np.array([[0, 9167, 9198, 9589.],
                     [9154, 9122, 9202, 9475.],
                     [9029, 9052, 9145, 9385.],
                     [9153, 9134, 9157, 9389.],
                     [9254, 9178, 9174, 9453.],
                     [9348, 9222, 9201, 9413.],
                     [9393, 9278, 9251, 9347.],
                     [9486, 9293, np.nan, 9317.]])
    agg = create_test_raster(data, backend=backend)
    # to run this data on QGIS, save this raster to tif file
    # blue = green_data(backend='numpy')
    # blue.rio.to_raster('blue.tif')
    return agg


@pytest.fixture
def green_data(backend):
    data = np.array([[0, 9929, 10056, 10620.],
                     [9913, 9904, 10061, 10621.],
                     [9853, 9874, 10116, 10429.],
                     [9901, 9970, 10121, 10395.],
                     [9954, 9945, 10068, 10512.],
                     [9985, 9942, 10027, 10541.],
                     [np.nan, 9980, 10062, 10371.],
                     [10101, 9971, 10044, 10275.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def red_data(backend):
    data = np.array([[0, 10301, 10454, 11325.],
                     [10353, 10269, 10501, np.nan],
                     [10184, 10106, 10445, 10961.],
                     [10349, 10230, 10299, 10844.],
                     [10430, 10328, 10327, 10979.],
                     [10479, 10340, 10381, 11076.],
                     [10498, 10402, 10495, 10877.],
                     [10603, 10383, 10433, 10751.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def nir_data(backend):
    data = np.array([[0, 15928, 16135, 16411.],
                     [15588, 15881, 16253, 16651.],
                     [16175, 16486, 17038, 17084.],
                     [15671, 16596, 17511, 17525.],
                     [15522, 15936, 17003, 17549.],
                     [15317, 15782, 16322, 17133.],
                     [15168, 15529, 16011, 16600.],
                     [15072, 15496, 15983, 16477.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def tir_data(backend):
    data = np.array([[0, 10512, 10517, 10527.],
                     [10511, 10504, 10502, 10504.],
                     [10522, 10507, 10497, 10491.],
                     [10543, 10514, 10498, 10486.],
                     [10566, np.nan, 10509, 10490.],
                     [10592, 10558, 10527, 10504.],
                     [10629, 10598, 10567, 10536.],
                     [10664, 10639, 10612, 10587.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def swir1_data(backend):
    data = np.array([[0, np.nan, 17194, 18163.],
                     [16974, 16871, 17123, 18304.],
                     [16680, 16437, 16474, 17519.],
                     [17004, 16453, 16001, 16800.],
                     [17230, 16906, 16442, 16840.],
                     [17237, 16969, 16784, 17461.],
                     [17417, 17079, 17173, 17679.],
                     [17621, 17205, 17163, 17362.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def swir2_data(backend):
    data = np.array([[0, 13175, 13558, 14952.],
                     [13291, 13159, 13516, 15029.],
                     [12924, 12676, np.nan, 14009.],
                     [13294, 12728, 12370, 13289.],
                     [13507, 13163, 12763, 13499.],
                     [13570, 13219, 13048, 14145.],
                     [13770, 13393, 13472, 14249.],
                     [14148, 13489, 13483, 13893.]])
    agg = create_test_raster(data, backend=backend)
    return agg


@pytest.fixture
def qgis_arvi():
    # this result is obtained by using NIR, red and blue band data
    # running through QGIS Raster Calculator with formula:
    # arvi = (nir - 2*red + blue) / (nir + 2*red + blue)
    result = np.array([
        [np.nan, 0.09832155, 0.0956943, 0.0688592],
        [0.08880479, 0.09804352, 0.09585208, np.nan],
        [0.10611779, 0.1164153, 0.11244237, 0.09396376],
        [0.0906375, 0.11409396, 0.12842213, 0.10752644],
        [0.08580945, 0.09740005, 0.1179347, 0.10302287],
        [0.08125288, 0.09465021, 0.1028627, 0.09022958],
        [0.07825362, 0.08776391, 0.09236357, 0.08790172],
        [0.07324535, 0.08831083, np.nan, 0.09074763]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_evi():
    # this result is obtained by using NIR, red and blue band data
    # running through QGIS Raster Calculator with formula:
    # evi = gain * (nir - red) / (nir + c1*red -c2*blue + soil_factor)
    # with default values of gain = 2.5, c1=6, c2=7.5, and soil_factor=1
    result = np.array([
        [0., 1.5661007, 1.4382279, 1.0217365],
        [1.4458131, 1.544984, 1.4036115, np.nan],
        [1.5662745, 1.7274992, 1.4820393, 1.2281862],
        [1.4591216, 1.6802154, 1.6963824, 1.3721503],
        [1.4635549, 1.5457553, 1.6425549, 1.3112202],
        [1.4965355, 1.5713791, 1.5468937, 1.1654801],
        [1.5143654, 1.5337442, 1.4365331, 1.2165724],
        [1.4805857, 1.5785736, np.nan, 1.2888849]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_nbr():
    # this result is obtained by using NIR, and SWIR2 band data
    # running through QGIS Raster Calculator with formula:
    # nbr = (nir - swir2) / (nir + swir2)
    result = np.array([
        [np.nan, 0.09459506, 0.08678813, 0.04651979],
        [0.07953876, 0.09373278, 0.09194128, 0.0511995],
        [0.11172205, 0.13064948, np.nan, 0.09889686],
        [0.08206456, 0.1319056, 0.17204913, 0.13746998],
        [0.06941334, 0.09529537, 0.1424444, 0.13044319],
        [0.06047703, 0.08837626, 0.11147429, 0.09553041],
        [0.04831018, 0.07385381, 0.08611742, 0.07620993],
        [0.03162218, 0.06924271, 0.08484355, 0.08508396]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_nbr2():
    # this result is obtained by using SWIR1, and SWIR2 band data
    # running through QGIS Raster Calculator with formula:
    # nbr2 = (swir1 - swir2) / (swir1 + swir2)
    result = np.array([
        [np.nan, np.nan, 0.11823621, 0.09696512],
        [0.12169173, 0.12360972, 0.11772577, 0.09825099],
        [0.12687474, 0.12918627, np.nan, 0.11132962],
        [0.12245033, 0.12765156, 0.1279828, 0.11668716],
        [0.12112438, 0.12448036, 0.12597159, 0.11012229],
        [0.11903139, 0.12422155, 0.12523465, 0.10491679],
        [0.11693975, 0.12096351, 0.12077011, 0.10742921],
        [0.10932041, 0.121066, 0.12008093, 0.11099024]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_ndvi():
    # this result is obtained by using NIR, and red band data
    # running through QGIS Raster Calculator with formula:
    # ndvi = (nir - red) / (nir + red)
    result = np.array([
        [np.nan, 0.21453354, 0.21365978, 0.1833718],
        [0.20180409, 0.21460803, 0.21499589, np.nan],
        [0.2272848, 0.23992178, 0.23989375, 0.21832769],
        [0.20453498, 0.23730709, 0.25933117, 0.23550354],
        [0.19620839, 0.21352422, 0.24427369, 0.23030005],
        [0.18754846, 0.20833014, 0.22248437, 0.2147187],
        [0.18195277, 0.19771701, 0.20810382, 0.20828329],
        [0.17406037, 0.19757332, 0.21009994, 0.21029823]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_ndmi():
    # this result is obtained by using NIR, and SWIR1 band data
    # running through QGIS Raster Calculator with formula:
    # ndvi = (nir - swir1) / (nir + swir1)
    result = np.array([
        [np.nan, np.nan, -0.03177413, -0.05067392],
        [-0.04256495, -0.03022716, -0.02606663, -0.04728937],
        [-0.01537057,  0.00148832,  0.01682979, -0.01257116],
        [-0.04079571,  0.00432691,  0.04505849,  0.02112163],
        [-0.05214949, -0.02953535,  0.01677381,  0.02061706],
        [-0.05897893, -0.03624317, -0.01395517, -0.00948141],
        [-0.06901949, -0.04753435, -0.03501688, -0.031477],
        [-0.07796776, -0.0522614, -0.03560007, -0.02615326]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_savi():
    # Correct SAVI (Huete 1988):
    # savi = ((nir - red) / (nir + red + L)) * (1 + L)
    # with default value of soil_factor=1
    result = np.array([
        [0., 0.4290507, 0.4273035, 0.3667304],
        [0.4035926, 0.4291996, 0.4299757, np.nan],
        [0.4545524, 0.4798255, 0.4797700, 0.4366398],
        [0.4090542, 0.4745965, 0.5186437, 0.4709905],
        [0.3924016, 0.4270322, 0.4885295, 0.4605840],
        [0.3750824, 0.4166443, 0.4449521, 0.4294222],
        [0.3638914, 0.3954188, 0.4161919, 0.4165514],
        [0.3481072, 0.3951314, 0.4201840, 0.4205810]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_gci():
    # this result is obtained by using NIR, and green band data
    # running through QGIS Raster Calculator with formula:
    # gci = nir / green - 1
    result = np.array([
        [np.nan, 0.60418975, 0.6045147, 0.5452919],
        [0.57248056, 0.6034935, 0.6154458, 0.5677431],
        [0.64163196, 0.66963744, 0.6842626, 0.63812447],
        [0.5827694, 0.66459376, 0.730165, 0.6859067],
        [0.55937314, 0.6024133, 0.6888161, 0.6694254],
        [0.534001, 0.58740693, 0.62780493, 0.62536764],
        [np.nan, 0.55601203, 0.5912343, 0.6006171],
        [0.4921295, 0.5541069, 0.5912983, 0.603601]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_sipi():
    # this result is obtained by using NIR, red and blue band data
    # running through QGIS Raster Calculator with formula:
    # sipi = (nir - blue) / (nir - red)
    result = np.array([
        [np.nan, 1.2015283, 1.2210878, 1.3413291],
        [1.2290354, 1.2043835, 1.2258345, np.nan],
        [1.1927892, 1.1652038, 1.1971788, 1.2573901],
        [1.2247275, 1.1721647, 1.1583472, 1.2177818],
        [1.2309505, 1.2050642, 1.1727082, 1.2322679],
        [1.2337743, 1.2054392, 1.1986197, 1.2745583],
        [1.2366167, 1.2192315, 1.2255257, 1.2673423],
        [1.2499441, 1.2131821, np.nan, 1.2504367]], dtype=np.float32)
    return result


@pytest.fixture
def qgis_ebbi():
    # this result is obtained by using red, swir1 and tir band data
    # running through QGIS Raster Calculator with formula:
    # ebbi = (swir1 - red) / (10 * sqrt(swir1 + tir))
    result = np.array([
        [np.nan, np.nan, 4.0488696, 4.0370474],
        [3.9937027, 3.9902349, 3.9841716, np.nan],
        [3.9386337, 3.8569257, 3.6711047, 3.918455],
        [4.0096908, 3.7895138, 3.5027769, 3.6056597],
        [4.0786624, np.nan, 3.724852, 3.5452912],
        [4.0510664, 3.9954765, 3.8744915, 3.8181543],
        [4.131501, 4.013487, 4.009527, 4.049455],
        [4.172874, 4.08833, 4.038202, 3.954431]], dtype=np.float32)
    return result


@pytest.fixture
def data_uint_dtype_normalized_ratio(dtype):
    # test data for input data array of uint dtype
    # normalized ratio is applied with different bands for NBR, NBR2, NDVI, NDMI.
    band1 = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    band2 = xr.DataArray(np.array([[0, 2], [1, 2]], dtype=dtype))
    result = np.array([[1, -0.33333334], [0, -0.33333334]], dtype=np.float32)
    return band1, band2, result


@pytest.fixture
def data_uint_dtype_arvi(dtype):
    nir = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    red = xr.DataArray(np.array([[0, 1], [0, 2]], dtype=dtype))
    blue = xr.DataArray(np.array([[0, 2], [1, 2]], dtype=dtype))
    result = np.array([[1, 0.2], [1, -0.14285715]], dtype=np.float32)
    return nir, red, blue, result


@pytest.fixture
def data_uint_dtype_evi(dtype):
    nir = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    red = xr.DataArray(np.array([[0, 1], [0, 2]], dtype=dtype))
    blue = xr.DataArray(np.array([[0, 2], [1, 2]], dtype=dtype))
    result = np.array([[1.25, 0.], [-0.45454547, 2.5]], dtype=np.float32)
    return nir, red, blue, result


@pytest.fixture
def data_uint_dtype_savi(dtype):
    nir = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    red = xr.DataArray(np.array([[0, 1], [0, 2]], dtype=dtype))
    # Correct SAVI with L=1: ((nir-red)/(nir+red+1)) * 2
    result = np.array([[1.0, 0.], [1.0, -0.5]], dtype=np.float32)
    return nir, red, result


@pytest.fixture
def data_uint_dtype_sipi(dtype):
    nir = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    red = xr.DataArray(np.array([[0, 0], [0, 2]], dtype=dtype))
    blue = xr.DataArray(np.array([[0, 2], [1, 2]], dtype=dtype))
    result = np.array([[1, -1], [0, 1]], dtype=np.float32)
    return nir, red, blue, result


@pytest.fixture
def data_uint_dtype_ebbi(dtype):
    red = xr.DataArray(np.array([[0, 0], [0, 2]], dtype=dtype))
    swir = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    tir = xr.DataArray(np.array([[0, 2], [1, 2]], dtype=dtype))
    result = np.array([[0.1, 0.05773503], [0.07071068, -0.05773503]], dtype=np.float32)
    return red, swir, tir, result


# NDVI -------------
def test_ndvi_data_contains_valid_values():
    _x = np.mgrid[1:0:21j]
    a, b = np.meshgrid(_x, _x)
    red_data = a*b
    nir_data = (a*b)[::-1, ::-1]

    da_nir = xr.DataArray(nir_data, dims=['y', 'x'])
    da_red = xr.DataArray(red_data, dims=['y', 'x'])

    da_ndvi = ndvi(da_nir, da_red)

    assert da_ndvi.dims == da_nir.dims
    assert da_ndvi.attrs == da_nir.attrs
    for coord in da_nir.coords:
        assert np.all(da_nir[coord] == da_ndvi[coord])

    assert da_ndvi[0, 0] == -1
    assert da_ndvi[-1, -1] == 1
    assert da_ndvi[5, 10] == da_ndvi[10, 5] == -0.5
    assert da_ndvi[15, 10] == da_ndvi[10, 15] == 0.5


@pytest.mark.parametrize("backend", ["numpy"])
def test_ndvi_cpu_against_qgis(nir_data, red_data, qgis_ndvi):
    result = ndvi(nir_data, red_data)
    general_output_checks(nir_data, result, qgis_ndvi, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_ndvi_dask_cpu_against_qgis(nir_data, red_data, qgis_ndvi):
    result = ndvi(nir_data, red_data)
    general_output_checks(nir_data, result, qgis_ndvi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ndvi_uint_dtype(data_uint_dtype_normalized_ratio):
    nir_data, red_data, result_ndvi = data_uint_dtype_normalized_ratio
    result = ndvi(nir_data, red_data)
    general_output_checks(nir_data, result, result_ndvi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ndvi_gpu(nir_data, red_data, qgis_ndvi):
    result = ndvi(nir_data, red_data)
    general_output_checks(nir_data, result, qgis_ndvi, verify_dtype=True)


# SAVI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_savi_zero_soil_factor_cpu_against_qgis(nir_data, red_data, qgis_ndvi):
    # savi should be same as ndvi at soil_factor=0
    qgis_savi = savi(nir_data, red_data, soil_factor=0.0)
    general_output_checks(nir_data, qgis_savi, qgis_ndvi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_savi_zero_soil_factor_gpu(nir_data, red_data, qgis_ndvi):
    # savi should be same as ndvi at soil_factor=0
    qgis_savi = savi(nir_data, red_data, soil_factor=0.0)
    general_output_checks(nir_data, qgis_savi, qgis_ndvi, verify_dtype=True)


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_savi_cpu_against_qgis(nir_data, red_data, qgis_savi):
    # test default savi where soil_factor = 1.0
    result = savi(nir_data, red_data, soil_factor=1.0)
    general_output_checks(nir_data, result, qgis_savi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_savi_uint_dtype(data_uint_dtype_savi):
    nir_data, red_data, result_savi = data_uint_dtype_savi
    result = savi(nir_data, red_data)
    general_output_checks(nir_data, result, result_savi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_savi_gpu(nir_data, red_data, qgis_savi):
    # test default savi where soil_factor = 1.0
    result = savi(nir_data, red_data, soil_factor=1.0)
    general_output_checks(nir_data, result, qgis_savi)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "dask+numpy", "dask+cupy"])
def test_savi_formula_1094(backend):
    """Verify SAVI against the Huete (1988) formula directly.

    Regression test for #1094: (1+L) was in the denominator instead
    of the numerator.
    """
    from xrspatial.tests.general_checks import has_cuda_and_cupy, dask_array_available as _dask_avail
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    try:
        import dask.array
    except ImportError:
        if 'dask' in backend:
            pytest.skip("Requires Dask")

    nir_arr = np.array([[0.8, 0.6], [0.4, 0.0]])
    red_arr = np.array([[0.2, 0.3], [0.4, 0.0]])

    nir_agg = create_test_raster(nir_arr, backend=backend, chunks=(2, 2))
    red_agg = create_test_raster(red_arr, backend=backend, chunks=(2, 2))

    L = 0.5
    result = savi(nir_agg, red_agg, soil_factor=L)
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    data = np.asarray(data)

    # Correct: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    expected = np.where(
        (nir_arr + red_arr + L) != 0,
        ((nir_arr - red_arr) / (nir_arr + red_arr + L)) * (1 + L),
        np.nan,
    ).astype(np.float32)

    np.testing.assert_allclose(data, expected, rtol=1e-5, equal_nan=True)

    # Spot check: NIR=0.8, Red=0.2, L=0.5 -> (0.6/1.5)*1.5 = 0.6
    assert abs(float(data[0, 0]) - 0.6) < 1e-5


# arvi -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_arvi_cpu_against_qgis(nir_data, red_data, blue_data, qgis_arvi):
    result = arvi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_arvi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_arvi_uint_dtype(data_uint_dtype_arvi):
    nir_data, red_data, blue_data, result_arvi = data_uint_dtype_arvi
    result = arvi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, result_arvi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_arvi_gpu(nir_data, red_data, blue_data, qgis_arvi):
    result = arvi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_arvi)


# EVI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_evi_cpu_against_qgis(nir_data, red_data, blue_data, qgis_evi):
    result = evi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_evi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_evi_uint_dtype(data_uint_dtype_evi):
    nir_data, red_data, blue_data, result_evi = data_uint_dtype_evi
    result = evi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, result_evi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_evi_gpu(nir_data, red_data, blue_data, qgis_evi):
    result = evi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_evi)


# GCI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_gci_cpu_against_qgis(nir_data, green_data, qgis_gci):
    result = gci(nir_data, green_data)
    general_output_checks(nir_data, result, qgis_gci)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_gci_gpu(nir_data, green_data, qgis_gci):
    result = gci(nir_data, green_data)
    general_output_checks(nir_data, result, qgis_gci)


# SIPI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_sipi_cpu_against_qgis(nir_data, red_data, blue_data, qgis_sipi):
    result = sipi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_sipi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_sipi_uint_dtype(data_uint_dtype_sipi):
    nir_data, red_data, blue_data, result_sipi = data_uint_dtype_sipi
    result = sipi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, result_sipi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_sipi_gpu(nir_data, red_data, blue_data, qgis_sipi):
    result = sipi(nir_data, red_data, blue_data)
    general_output_checks(nir_data, result, qgis_sipi)


# NBR -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_nbr_cpu_against_qgis(nir_data, swir2_data, qgis_nbr):
    result = nbr(nir_data, swir2_data)
    general_output_checks(nir_data, result, qgis_nbr)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_nbr_uint_dtype(data_uint_dtype_normalized_ratio):
    nir_data, red_data, result_nbr = data_uint_dtype_normalized_ratio
    result = nbr(nir_data, red_data)
    general_output_checks(nir_data, result, result_nbr, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_nbr_gpu(nir_data, swir2_data, qgis_nbr):
    result = nbr(nir_data, swir2_data)
    general_output_checks(nir_data, result, qgis_nbr)


# NBR2 -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_nbr2_cpu_against_qgis(swir1_data, swir2_data, qgis_nbr2):
    result = nbr2(swir1_data, swir2_data)
    general_output_checks(swir1_data, result, qgis_nbr2)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_nbr2_uint_dtype(data_uint_dtype_normalized_ratio):
    nir_data, red_data, result_nbr2 = data_uint_dtype_normalized_ratio
    result = nbr2(nir_data, red_data)
    general_output_checks(nir_data, result, result_nbr2, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_nbr2_gpu(swir1_data, swir2_data, qgis_nbr2):
    result = nbr2(swir1_data, swir2_data)
    general_output_checks(swir1_data, result, qgis_nbr2)


# NDMI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_ndmi_cpu_against_qgis(nir_data, swir1_data, qgis_ndmi):
    result = ndmi(nir_data, swir1_data)
    general_output_checks(nir_data, result, qgis_ndmi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ndmi_uint_dtype(data_uint_dtype_normalized_ratio):
    nir_data, red_data, result_ndmi = data_uint_dtype_normalized_ratio
    result = ndmi(nir_data, red_data)
    general_output_checks(nir_data, result, result_ndmi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ndmi_gpu(nir_data, swir1_data, qgis_ndmi):
    result = ndmi(nir_data, swir1_data)
    general_output_checks(nir_data, result, qgis_ndmi)


# NDWI -------------
@pytest.fixture
def qgis_ndwi():
    # ndwi = (green - nir) / (green + nir) (McFeeters 1996)
    result = np.array([
        [np.nan, -0.2320068, -0.23210263, -0.21423551],
        [-0.22254029, -0.23180144, -0.235312, -0.2211059],
        [-0.24289227, -0.25083458, -0.2549164, -0.24188565],
        [-0.22563742, -0.24941655, -0.26744354, -0.2553725],
        [-0.21855864, -0.23148255, -0.2561782, -0.2507751],
        [-0.21073432, -0.22702534, -0.2389085, -0.23820192],
        [np.nan, -0.21753107, -0.22816707, -0.23095176],
        [-0.19747348, -0.21694742, -0.22818612, -0.23183313]], dtype=np.float32)
    return result


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_ndwi_cpu(green_data, nir_data, qgis_ndwi):
    result = ndwi(green_data, nir_data)
    general_output_checks(green_data, result, qgis_ndwi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ndwi_uint_dtype(data_uint_dtype_normalized_ratio):
    band1, band2, expected = data_uint_dtype_normalized_ratio
    result = ndwi(band1, band2)
    general_output_checks(band1, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ndwi_gpu(green_data, nir_data, qgis_ndwi):
    result = ndwi(green_data, nir_data)
    general_output_checks(green_data, result, qgis_ndwi, verify_dtype=True)


# MNDWI -------------
@pytest.fixture
def qgis_mndwi():
    # mndwi = (green - swir) / (green + swir) (Xu 2006)
    result = np.array([
        [np.nan, np.nan, -0.26194495, -0.26206443],
        [-0.26261762, -0.26020542, -0.25978518, -0.26561797],
        [-0.25730222, -0.2494394, -0.23911245, -0.2536854],
        [-0.26400298, -0.24535443, -0.22509761, -0.23552124],
        [-0.26765746, -0.25924546, -0.24043757, -0.23135419],
        [-0.26640218, -0.26112, -0.25202343, -0.24712521],
        [np.nan, -0.26235265, -0.26109785, -0.26053476],
        [-0.2712647, -0.26619077, -0.2616606, -0.2564316]], dtype=np.float32)
    return result


@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_mndwi_cpu(green_data, swir1_data, qgis_mndwi):
    result = mndwi(green_data, swir1_data)
    general_output_checks(green_data, result, qgis_mndwi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_mndwi_uint_dtype(data_uint_dtype_normalized_ratio):
    band1, band2, expected = data_uint_dtype_normalized_ratio
    result = mndwi(band1, band2)
    general_output_checks(band1, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_mndwi_gpu(green_data, swir1_data, qgis_mndwi):
    result = mndwi(green_data, swir1_data)
    general_output_checks(green_data, result, qgis_mndwi, verify_dtype=True)


# EBBI -------------
@pytest.mark.parametrize("backend", ["numpy", "dask+numpy"])
def test_ebbi_cpu_against_qgis(red_data, swir1_data, tir_data, qgis_ebbi):
    result = ebbi(red_data, swir1_data, tir_data)
    general_output_checks(red_data, result, qgis_ebbi)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ebbi_uint_dtype(data_uint_dtype_ebbi):
    red_data, swir_data, tir_data, result_ebbi = data_uint_dtype_ebbi
    result = ebbi(red_data, swir_data, tir_data)
    general_output_checks(red_data, result, result_ebbi, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ebbi_gpu(red_data, swir1_data, tir_data, qgis_ebbi):
    result = ebbi(red_data, swir1_data, tir_data)
    general_output_checks(red_data, result, qgis_ebbi)


# true_color ----------
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_true_color_numpy_equals_dask_numpy(random_data):
    # vanilla numpy version
    red_numpy = create_test_raster(random_data, backend="numpy")
    green_numpy = create_test_raster(random_data, backend="numpy")
    blue_numpy = create_test_raster(random_data, backend="numpy")
    numpy_result = true_color(
        red_numpy, green_numpy, blue_numpy, name='np_true_color'
    )
    assert numpy_result.name == 'np_true_color'
    general_output_checks(red_numpy, numpy_result, verify_attrs=False)

    # dask
    red_dask = create_test_raster(random_data, backend='dask')
    green_dask = create_test_raster(random_data, backend='dask')
    blue_dask = create_test_raster(random_data, backend='dask')
    dask_result = true_color(
        red_dask, green_dask, blue_dask, name='dask_true_color'
    )
    assert dask_result.name == 'dask_true_color'
    general_output_checks(red_dask, dask_result, verify_attrs=False)

    np.testing.assert_allclose(
        numpy_result.data, dask_result.compute().data, equal_nan=True
    )


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_true_color_gpu(random_data, backend):
    # numpy baseline
    red_np = create_test_raster(random_data, backend="numpy")
    green_np = create_test_raster(random_data, backend="numpy")
    blue_np = create_test_raster(random_data, backend="numpy")
    numpy_result = true_color(red_np, green_np, blue_np)

    # gpu version
    red_gpu = create_test_raster(random_data, backend=backend)
    green_gpu = create_test_raster(random_data, backend=backend)
    blue_gpu = create_test_raster(random_data, backend=backend)
    gpu_result = true_color(red_gpu, green_gpu, blue_gpu)

    general_output_checks(red_gpu, gpu_result, verify_attrs=False)

    gpu_data = gpu_result.data
    if hasattr(gpu_data, 'compute'):
        gpu_data = gpu_data.compute()
    if hasattr(gpu_data, 'get'):
        gpu_data = gpu_data.get()

    np.testing.assert_allclose(
        numpy_result.data, gpu_data, equal_nan=True
    )


# NDSI ----------
@pytest.fixture
def expected_ndsi():
    # ndsi = (green - swir1) / (green + swir1)
    result = np.array(
        [[np.nan, np.nan, -0.26194495, -0.26206443],
         [-0.26261762, -0.26020542, -0.25978518, -0.26561797],
         [-0.25730222, -0.2494394, -0.23911245, -0.2536854],
         [-0.26400298, -0.24535443, -0.22509761, -0.23552124],
         [-0.26765746, -0.25924546, -0.24043757, -0.23135419],
         [-0.26640218, -0.26112, -0.25202343, -0.24712521],
         [np.nan, -0.26235265, -0.26109785, -0.26053476],
         [-0.2712647, -0.26619077, -0.2616606, -0.2564316]],
        dtype=np.float32)
    return result


@pytest.mark.parametrize("backend", ["numpy"])
def test_ndsi_cpu(green_data, swir1_data, expected_ndsi):
    result = ndsi(green_data, swir1_data)
    general_output_checks(green_data, result, expected_ndsi, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_ndsi_dask_cpu(green_data, swir1_data, expected_ndsi):
    result = ndsi(green_data, swir1_data)
    general_output_checks(green_data, result, expected_ndsi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ndsi_uint_dtype(data_uint_dtype_normalized_ratio):
    band1, band2, expected = data_uint_dtype_normalized_ratio
    result = ndsi(band1, band2)
    general_output_checks(band1, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ndsi_gpu(green_data, swir1_data, expected_ndsi):
    result = ndsi(green_data, swir1_data)
    general_output_checks(green_data, result, expected_ndsi, verify_dtype=True)


# NDBI ----------
@pytest.fixture
def expected_ndbi():
    # ndbi = (swir1 - nir) / (swir1 + nir)
    # This is the negation of NDMI
    result = np.array(
        [[np.nan, np.nan, 0.03177413, 0.05067392],
         [0.04256495, 0.03022716, 0.02606663, 0.04728937],
         [0.01537057, -0.00148832, -0.01682979, 0.01257116],
         [0.04079571, -0.00432691, -0.04505849, -0.02112163],
         [0.05214949, 0.02953535, -0.01677381, -0.02061706],
         [0.05897893, 0.03624317, 0.01395517, 0.00948141],
         [0.06901949, 0.04753435, 0.03501688, 0.031477],
         [0.07796776, 0.0522614, 0.03560007, 0.02615326]],
        dtype=np.float32)
    return result


@pytest.mark.parametrize("backend", ["numpy"])
def test_ndbi_cpu(swir1_data, nir_data, expected_ndbi):
    result = ndbi(swir1_data, nir_data)
    general_output_checks(swir1_data, result, expected_ndbi, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_ndbi_dask_cpu(swir1_data, nir_data, expected_ndbi):
    result = ndbi(swir1_data, nir_data)
    general_output_checks(swir1_data, result, expected_ndbi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ndbi_uint_dtype(data_uint_dtype_normalized_ratio):
    band1, band2, expected = data_uint_dtype_normalized_ratio
    result = ndbi(band1, band2)
    general_output_checks(band1, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_ndbi_gpu(swir1_data, nir_data, expected_ndbi):
    result = ndbi(swir1_data, nir_data)
    general_output_checks(swir1_data, result, expected_ndbi, verify_dtype=True)


# BAI ----------
@pytest.fixture
def expected_bai():
    # bai = 1 / ((0.1 - red)^2 + (0.06 - nir)^2)
    # Test data is in DN (~10000s), so values are very small
    result = np.array(
        [[7.3529404e+01, 2.7792613e-09, 2.7054787e-09, 2.5152656e-09],
         [2.8557849e-09, 2.7959937e-09, 2.6707474e-09, np.nan],
         [2.7371723e-09, 2.6744016e-09, 2.5038351e-09, 2.4271707e-09],
         [2.8354468e-09, 2.6310520e-09, 2.4230655e-09, 2.3545275e-09],
         [2.8594838e-09, 2.7730043e-09, 2.5268809e-09, 2.3337139e-09],
         [2.9034652e-09, 2.8091214e-09, 2.6725904e-09, 2.4026150e-09],
         [2.9388227e-09, 2.8624791e-09, 2.7285640e-09, 2.5389413e-09],
         [2.9447718e-09, 2.8741514e-09, 2.7449920e-09, 2.5835027e-09]],
        dtype=np.float32)
    return result


@pytest.fixture
def data_uint_dtype_bai(dtype):
    red = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=dtype))
    nir = xr.DataArray(np.array([[0, 1], [1, 2]], dtype=dtype))
    # bai = 1 / ((0.1 - red)^2 + (0.06 - nir)^2)
    result = np.array(
        [[1.2291054, 0.5904582], [0.5904582, 0.21864615]],
        dtype=np.float32)
    return red, nir, result


@pytest.mark.parametrize("backend", ["numpy"])
def test_bai_cpu(red_data, nir_data, expected_bai):
    result = bai(red_data, nir_data)
    general_output_checks(red_data, result, expected_bai, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_bai_dask_cpu(red_data, nir_data, expected_bai):
    result = bai(red_data, nir_data)
    general_output_checks(red_data, result, expected_bai, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_bai_uint_dtype(data_uint_dtype_bai):
    red_data, nir_data, expected = data_uint_dtype_bai
    result = bai(red_data, nir_data)
    general_output_checks(red_data, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_bai_gpu(red_data, nir_data, expected_bai):
    result = bai(red_data, nir_data)
    general_output_checks(red_data, result, expected_bai, verify_dtype=True)


# MSAVI2 ----------
@pytest.fixture
def expected_msavi2():
    # msavi2 = (2*nir + 1 - sqrt((2*nir + 1)^2 - 8*(nir - red))) / 2
    result = np.array(
        [[0., 0.35253906, 0.3515625, 0.31054688],
         [0.3359375, 0.35351562, 0.3544922, np.nan],
         [0.3701172, 0.38671875, 0.38671875, 0.359375],
         [0.3388672, 0.38476562, 0.41210938, 0.3828125],
         [0.328125, 0.3515625, 0.39257812, 0.375],
         [0.31640625, 0.34472656, 0.3642578, 0.35351562],
         [0.3076172, 0.33007812, 0.34472656, 0.34570312],
         [0.296875, 0.33007812, 0.34765625, 0.34765625]],
        dtype=np.float32)
    return result


@pytest.fixture
def data_uint_dtype_msavi2(dtype):
    nir = xr.DataArray(np.array([[2, 3], [1, 4]], dtype=dtype))
    red = xr.DataArray(np.array([[1, 2], [0, 3]], dtype=dtype))
    # msavi2 = (2*nir + 1 - sqrt((2*nir + 1)^2 - 8*(nir - red))) / 2
    result = np.array(
        [[0.43844724, 0.29843783], [1., 0.22799826]],
        dtype=np.float32)
    return nir, red, result


@pytest.mark.parametrize("backend", ["numpy"])
def test_msavi2_cpu(nir_data, red_data, expected_msavi2):
    result = msavi2(nir_data, red_data)
    general_output_checks(nir_data, result, expected_msavi2, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_msavi2_dask_cpu(nir_data, red_data, expected_msavi2):
    result = msavi2(nir_data, red_data)
    general_output_checks(nir_data, result, expected_msavi2, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_msavi2_uint_dtype(data_uint_dtype_msavi2):
    nir_data, red_data, expected = data_uint_dtype_msavi2
    result = msavi2(nir_data, red_data)
    general_output_checks(nir_data, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_msavi2_gpu(nir_data, red_data, expected_msavi2):
    # GPU sqrt has slightly different precision than CPU (~0.002 for
    # large DN values), so use a looser tolerance.
    result = msavi2(nir_data, red_data)
    general_output_checks(nir_data, result, expected_msavi2,
                          verify_dtype=True, rtol=6e-3)


# OSAVI ----------
@pytest.fixture
def expected_osavi():
    # osavi = (nir - red) / (nir + red + 0.16)
    # With test DN data (~10000s), the 0.16 constant is negligible,
    # so results are very close to NDVI.
    result = np.array(
        [[0., 0.21453223, 0.2136585, 0.18337074],
         [0.20180285, 0.21460672, 0.2149946, np.nan],
         [0.22728342, 0.23992033, 0.23989235, 0.21832643],
         [0.20453371, 0.23730567, 0.25932968, 0.23550221],
         [0.19620718, 0.21352291, 0.24427226, 0.23029876],
         [0.1875473, 0.20832887, 0.22248302, 0.21471749],
         [0.18195164, 0.1977158, 0.20810257, 0.20828208],
         [0.17405929, 0.1975721, 0.21009867, 0.21029699]],
        dtype=np.float32)
    return result


@pytest.fixture
def data_uint_dtype_osavi(dtype):
    nir = xr.DataArray(np.array([[2, 3], [1, 4]], dtype=dtype))
    red = xr.DataArray(np.array([[1, 2], [0, 3]], dtype=dtype))
    # osavi = (nir - red) / (nir + red + 0.16)
    result = np.array(
        [[0.3164557, 0.19379845], [0.862069, 0.13966481]],
        dtype=np.float32)
    return nir, red, result


@pytest.mark.parametrize("backend", ["numpy"])
def test_osavi_cpu(nir_data, red_data, expected_osavi):
    result = osavi(nir_data, red_data)
    general_output_checks(nir_data, result, expected_osavi, verify_dtype=True)


@dask_array_available
@pytest.mark.parametrize("backend", ["dask+numpy"])
def test_osavi_dask_cpu(nir_data, red_data, expected_osavi):
    result = osavi(nir_data, red_data)
    general_output_checks(nir_data, result, expected_osavi, verify_dtype=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_osavi_uint_dtype(data_uint_dtype_osavi):
    nir_data, red_data, expected = data_uint_dtype_osavi
    result = osavi(nir_data, red_data)
    general_output_checks(nir_data, result, expected, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
def test_osavi_gpu(nir_data, red_data, expected_osavi):
    result = osavi(nir_data, red_data)
    general_output_checks(nir_data, result, expected_osavi, verify_dtype=True)


# Cross-index consistency checks ----------
@pytest.mark.parametrize("backend", ["numpy"])
def test_ndbi_is_negation_of_ndmi(swir1_data, nir_data, qgis_ndmi):
    # NDBI(swir1, nir) = -NDMI(nir, swir1)
    result_ndbi = ndbi(swir1_data, nir_data)
    result_data = result_ndbi.values
    np.testing.assert_allclose(result_data, -qgis_ndmi, equal_nan=True, rtol=1e-5)


@pytest.mark.parametrize("backend", ["numpy"])
def test_osavi_approaches_ndvi_for_large_dn(nir_data, red_data, qgis_ndvi):
    # With DN values >> 0.16, OSAVI and NDVI should be nearly identical.
    # Skip (0,0) where both bands are 0: NDVI=NaN (0/0), OSAVI=0 (0/0.16).
    result_osavi = osavi(nir_data, red_data)
    osavi_vals = result_osavi.values
    ndvi_vals = qgis_ndvi.copy()
    # Mask out cells where both inputs are zero
    mask = np.isnan(ndvi_vals) & ~np.isnan(osavi_vals)
    osavi_vals[mask] = np.nan
    np.testing.assert_allclose(
        osavi_vals, ndvi_vals, equal_nan=True, rtol=1e-3)
