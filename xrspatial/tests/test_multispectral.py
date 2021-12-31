import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import has_cuda
from xrspatial.utils import doesnt_have_cuda

from xrspatial.multispectral import arvi
from xrspatial.multispectral import ebbi
from xrspatial.multispectral import evi
from xrspatial.multispectral import nbr
from xrspatial.multispectral import nbr2
from xrspatial.multispectral import ndmi
from xrspatial.multispectral import ndvi
from xrspatial.multispectral import savi
from xrspatial.multispectral import gci
from xrspatial.multispectral import sipi
from xrspatial.multispectral import true_color

from xrspatial.tests.general_checks import general_output_checks


blue_data = np.array([[0, 9167, 9198, 9589.],
                      [9154, 9122, 9202, 9475.],
                      [9029, 9052, 9145, 9385.],
                      [9153, 9134, 9157, 9389.],
                      [9254, 9178, 9174, 9453.],
                      [9348, 9222, 9201, 9413.],
                      [9393, 9278, 9251, 9347.],
                      [9486, 9293, np.nan, 9317.]])

green_data = np.array([[0, 9929, 10056, 10620.],
                       [9913, 9904, 10061, 10621.],
                       [9853, 9874, 10116, 10429.],
                       [9901, 9970, 10121, 10395.],
                       [9954, 9945, 10068, 10512.],
                       [9985, 9942, 10027, 10541.],
                       [np.nan, 9980, 10062, 10371.],
                       [10101, 9971, 10044, 10275.]])

red_data = np.array([[0, 10301, 10454, 11325.],
                     [10353, 10269, 10501, np.nan],
                     [10184, 10106, 10445, 10961.],
                     [10349, 10230, 10299, 10844.],
                     [10430, 10328, 10327, 10979.],
                     [10479, 10340, 10381, 11076.],
                     [10498, 10402, 10495, 10877.],
                     [10603, 10383, 10433, 10751.]])

nir_data = np.array([[0, 15928, 16135, 16411.],
                     [15588, 15881, 16253, 16651.],
                     [16175, 16486, 17038, 17084.],
                     [15671, 16596, 17511, 17525.],
                     [15522, 15936, 17003, 17549.],
                     [15317, 15782, 16322, 17133.],
                     [15168, 15529, 16011, 16600.],
                     [15072, 15496, 15983, 16477.]])

tir_data = np.array([[0, 10512, 10517, 10527.],
                     [10511, 10504, 10502, 10504.],
                     [10522, 10507, 10497, 10491.],
                     [10543, 10514, 10498, 10486.],
                     [10566, np.nan, 10509, 10490.],
                     [10592, 10558, 10527, 10504.],
                     [10629, 10598, 10567, 10536.],
                     [10664, 10639, 10612, 10587.]])

swir1_data = np.array([[0, np.nan, 17194, 18163.],
                       [16974, 16871, 17123, 18304.],
                       [16680, 16437, 16474, 17519.],
                       [17004, 16453, 16001, 16800.],
                       [17230, 16906, 16442, 16840.],
                       [17237, 16969, 16784, 17461.],
                       [17417, 17079, 17173, 17679.],
                       [17621, 17205, 17163, 17362.]])

swir2_data = np.array([[0, 13175, 13558, 14952.],
                       [13291, 13159, 13516, 15029.],
                       [12924, 12676, np.nan, 14009.],
                       [13294, 12728, 12370, 13289.],
                       [13507, 13163, 12763, 13499.],
                       [13570, 13219, 13048, 14145.],
                       [13770, 13393, 13472, 14249.],
                       [14148, 13489, 13483, 13893.]])

arvi_expected_results = np.array([
    [np.nan, 0.09832155, 0.0956943, 0.0688592],
    [0.08880479, 0.09804352, 0.09585208, np.nan],
    [0.10611779, 0.1164153, 0.11244237, 0.09396376],
    [0.0906375, 0.11409396, 0.12842213, 0.10752644],
    [0.08580945, 0.09740005, 0.1179347, 0.10302287],
    [0.08125288, 0.09465021, 0.1028627, 0.09022958],
    [0.07825362, 0.08776391, 0.09236357, 0.08790172],
    [0.07324535, 0.08831083, np.nan, 0.09074763]], dtype=np.float32)

evi_expected_results = np.array([
    [0., 1.5661007, 1.4382279, 1.0217365],
    [1.4458131, 1.544984, 1.4036115, np.nan],
    [1.5662745, 1.7274992, 1.4820393, 1.2281862],
    [1.4591216, 1.6802154, 1.6963824, 1.3721503],
    [1.4635549, 1.5457553, 1.6425549, 1.3112202],
    [1.4965355, 1.5713791, 1.5468937, 1.1654801],
    [1.5143654, 1.5337442, 1.4365331, 1.2165724],
    [1.4805857, 1.5785736, np.nan, 1.2888849]], dtype=np.float32)

nbr_expected_results = np.array([
    [np.nan, 0.09459506, 0.08678813, 0.04651979],
    [0.07953876, 0.09373278, 0.09194128, 0.0511995],
    [0.11172205, 0.13064948, np.nan, 0.09889686],
    [0.08206456, 0.1319056, 0.17204913, 0.13746998],
    [0.06941334, 0.09529537, 0.1424444, 0.13044319],
    [0.06047703, 0.08837626, 0.11147429, 0.09553041],
    [0.04831018, 0.07385381, 0.08611742, 0.07620993],
    [0.03162218, 0.06924271, 0.08484355, 0.08508396]], dtype=np.float32)

nbr2_expected_results = np.array([
    [np.nan, np.nan, 0.11823621, 0.09696512],
    [0.12169173, 0.12360972, 0.11772577, 0.09825099],
    [0.12687474, 0.12918627, np.nan, 0.11132962],
    [0.12245033, 0.12765156, 0.1279828, 0.11668716],
    [0.12112438, 0.12448036, 0.12597159, 0.11012229],
    [0.11903139, 0.12422155, 0.12523465, 0.10491679],
    [0.11693975, 0.12096351, 0.12077011, 0.10742921],
    [0.10932041, 0.121066, 0.12008093, 0.11099024]], dtype=np.float32)

ndvi_expected_results = np.array([
    [np.nan, 0.21453354, 0.21365978, 0.1833718],
    [0.20180409, 0.21460803, 0.21499589, np.nan],
    [0.2272848, 0.23992178, 0.23989375, 0.21832769],
    [0.20453498, 0.23730709, 0.25933117, 0.23550354],
    [0.19620839, 0.21352422, 0.24427369, 0.23030005],
    [0.18754846, 0.20833014, 0.22248437, 0.2147187],
    [0.18195277, 0.19771701, 0.20810382, 0.20828329],
    [0.17406037, 0.19757332, 0.21009994, 0.21029823]], dtype=np.float32)

ndmi_expected_results = np.array([
    [np.nan, np.nan, -0.03177413, -0.05067392],
    [-0.04256495, -0.03022716, -0.02606663, -0.04728937],
    [-0.01537057,  0.00148832,  0.01682979, -0.01257116],
    [-0.04079571,  0.00432691,  0.04505849,  0.02112163],
    [-0.05214949, -0.02953535,  0.01677381,  0.02061706],
    [-0.05897893, -0.03624317, -0.01395517, -0.00948141],
    [-0.06901949, -0.04753435, -0.03501688, -0.031477],
    [-0.07796776, -0.0522614, -0.03560007, -0.02615326]], dtype=np.float32)

savi_expected_results = np.array([
    [0., 0.10726268, 0.10682587, 0.09168259],
    [0.10089815, 0.10729991, 0.10749393, np.nan],
    [0.11363809, 0.11995638, 0.11994251, 0.10915995],
    [0.10226355, 0.11864913, 0.12966092, 0.11774762],
    [0.09810041, 0.10675804, 0.12213238, 0.11514599],
    [0.09377059, 0.10416108, 0.11123802, 0.10735555],
    [0.09097284, 0.0988547, 0.10404798, 0.10413785],
    [0.0870268, 0.09878284, 0.105046, 0.10514525]], dtype=np.float32)

gci_expected_results = np.array([
    [np.nan, 0.60418975, 0.6045147, 0.5452919],
    [0.57248056, 0.6034935, 0.6154458, 0.5677431],
    [0.64163196, 0.66963744, 0.6842626, 0.63812447],
    [0.5827694, 0.66459376, 0.730165, 0.6859067],
    [0.55937314, 0.6024133, 0.6888161, 0.6694254],
    [0.534001, 0.58740693, 0.62780493, 0.62536764],
    [np.nan, 0.55601203, 0.5912343, 0.6006171],
    [0.4921295, 0.5541069, 0.5912983, 0.603601]], dtype=np.float32)

sipi_expected_results = np.array([
    [np.nan, 1.2015283, 1.2210878, 1.3413291],
    [1.2290354, 1.2043835, 1.2258345, np.nan],
    [1.1927892, 1.1652038, 1.1971788, 1.2573901],
    [1.2247275, 1.1721647, 1.1583472, 1.2177818],
    [1.2309505, 1.2050642, 1.1727082, 1.2322679],
    [1.2337743, 1.2054392, 1.1986197, 1.2745583],
    [1.2366167, 1.2192315, 1.2255257, 1.2673423],
    [1.2499441, 1.2131821, np.nan, 1.2504367]], dtype=np.float32)

ebbi_expected_results = np.array([
    [np.nan, np.nan, 4.0488696, 4.0370474],
    [3.9937027, 3.9902349, 3.9841716, np.nan],
    [3.9386337, 3.8569257, 3.6711047, 3.918455],
    [4.0096908, 3.7895138, 3.5027769, 3.6056597],
    [4.0786624, np.nan, 3.724852, 3.5452912],
    [4.0510664, 3.9954765, 3.8744915, 3.8181543],
    [4.131501, 4.013487, 4.009527, 4.049455],
    [4.172874, 4.08833, 4.038202, 3.954431]], dtype=np.float32)


def _do_gaussian_array():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X, Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian


data_gaussian = _do_gaussian_array()


def create_test_arr(arr, backend='numpy'):

    y, x = arr.shape
    raster = xr.DataArray(arr, dims=['y', 'x'])

    if backend == 'numpy':
        raster['y'] = np.linspace(0, y, y)
        raster['x'] = np.linspace(0, x, x)
        return raster

    if has_cuda() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        raster.data = da.from_array(raster.data, chunks=(3, 3))

    return raster


# NDVI -------------
def test_ndvi_numpy_contains_valid_values():
    _x = np.mgrid[1:0:21j]
    a, b = np.meshgrid(_x, _x)
    red_numpy = a*b
    nir_numpy = (a*b)[::-1, ::-1]

    da_nir = xr.DataArray(nir_numpy, dims=['y', 'x'])
    da_red = xr.DataArray(red_numpy, dims=['y', 'x'])

    da_ndvi = ndvi(da_nir, da_red)

    assert da_ndvi.dims == da_nir.dims
    assert da_ndvi.attrs == da_nir.attrs
    for coord in da_nir.coords:
        assert np.all(da_nir[coord] == da_ndvi[coord])

    assert da_ndvi[0, 0] == -1
    assert da_ndvi[-1, -1] == 1
    assert da_ndvi[5, 10] == da_ndvi[10, 5] == -0.5
    assert da_ndvi[15, 10] == da_ndvi[10, 15] == 0.5


def test_ndvi_cpu():

    # vanilla numpy version
    nir_numpy = create_test_arr(nir_data)
    red_numpy = create_test_arr(red_data)
    numpy_result = ndvi(nir_numpy, red_numpy)
    general_output_checks(nir_numpy, numpy_result, ndvi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    red_dask = create_test_arr(red_data, backend='dask')
    dask_result = ndvi(nir_dask, red_dask)
    general_output_checks(nir_dask, dask_result, ndvi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndvi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    red_cupy = create_test_arr(red_data, backend='cupy')
    cupy_result = ndvi(nir_cupy, red_cupy)
    general_output_checks(nir_cupy, cupy_result, ndvi_expected_results)

    # dask + cupy
    nir_dask = create_test_arr(nir_data, backend='dask+cupy')
    red_dask = create_test_arr(red_data, backend='dask+cupy')
    dask_cupy_result = ndvi(nir_dask, red_dask)
    general_output_checks(nir_dask, dask_cupy_result, ndvi_expected_results)


# SAVI -------------
def test_savi_cpu():
    nir_numpy = create_test_arr(nir_data)
    red_numpy = create_test_arr(red_data)

    # savi should be same as ndvi at soil_factor=0
    result_savi = savi(nir_numpy, red_numpy, soil_factor=0.0)
    result_ndvi = ndvi(nir_numpy, red_numpy)
    assert np.isclose(result_savi.data, result_ndvi.data, equal_nan=True).all()

    # test default savi where soil_factor = 1.0
    numpy_result = savi(nir_numpy, red_numpy, soil_factor=1.0)
    general_output_checks(nir_numpy, numpy_result, savi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    red_dask = create_test_arr(red_data, backend='dask')
    dask_result = savi(nir_dask, red_dask)
    general_output_checks(nir_dask, dask_result, savi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_savi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    red_cupy = create_test_arr(red_data, backend='cupy')
    cupy_result = savi(nir_cupy, red_cupy)
    general_output_checks(nir_cupy, cupy_result, savi_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    red_dask_cupy = create_test_arr(red_data, backend='dask+cupy')
    dask_cupy_result = savi(nir_dask_cupy, red_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, savi_expected_results)


# arvi -------------
def test_arvi_cpu():
    nir_numpy = create_test_arr(nir_data)
    red_numpy = create_test_arr(red_data)
    blue_numpy = create_test_arr(blue_data)
    numpy_result = arvi(nir_numpy, red_numpy, blue_numpy)
    general_output_checks(nir_numpy, numpy_result, arvi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    red_dask = create_test_arr(red_data, backend='dask')
    blue_dask = create_test_arr(blue_data, backend='dask')
    dask_result = arvi(nir_dask, red_dask, blue_dask)
    general_output_checks(nir_dask, dask_result, arvi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_arvi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    red_cupy = create_test_arr(red_data, backend='cupy')
    blue_cupy = create_test_arr(blue_data, backend='cupy')
    cupy_result = arvi(nir_cupy, red_cupy, blue_cupy)
    general_output_checks(nir_cupy, cupy_result, arvi_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    red_dask_cupy = create_test_arr(red_data, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(blue_data, backend='dask+cupy')
    dask_cupy_result = arvi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, arvi_expected_results
    )


# EVI -------------
def test_evi_cpu():
    nir_numpy = create_test_arr(nir_data)
    red_numpy = create_test_arr(red_data)
    blue_numpy = create_test_arr(blue_data)
    numpy_result = evi(nir_numpy, red_numpy, blue_numpy)
    general_output_checks(nir_numpy, numpy_result, evi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    red_dask = create_test_arr(red_data, backend='dask')
    blue_dask = create_test_arr(blue_data, backend='dask')
    dask_result = evi(nir_dask, red_dask, blue_dask)
    general_output_checks(nir_dask, dask_result, evi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_evi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    red_cupy = create_test_arr(red_data, backend='cupy')
    blue_cupy = create_test_arr(blue_data, backend='cupy')
    cupy_result = evi(nir_cupy, red_cupy, blue_cupy)
    general_output_checks(nir_cupy, cupy_result, evi_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    red_dask_cupy = create_test_arr(red_data, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(blue_data, backend='dask+cupy')
    dask_cupy_result = evi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, evi_expected_results)


# GCI -------------
def test_gci_cpu():
    # vanilla numpy version
    nir_numpy = create_test_arr(nir_data)
    green_numpy = create_test_arr(green_data)
    numpy_result = gci(nir_numpy, green_numpy)
    general_output_checks(nir_numpy, numpy_result, gci_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    green_dask = create_test_arr(green_data, backend='dask')
    dask_result = gci(nir_dask, green_dask)
    general_output_checks(nir_dask, dask_result, gci_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_gci_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    green_cupy = create_test_arr(green_data, backend='cupy')
    cupy_result = gci(nir_cupy, green_cupy)
    general_output_checks(nir_cupy, cupy_result, gci_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    green_dask_cupy = create_test_arr(green_data, backend='dask+cupy')
    dask_cupy_result = gci(nir_dask_cupy, green_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, gci_expected_results)


# SIPI -------------
def test_sipi_cpu():
    nir_numpy = create_test_arr(nir_data)
    red_numpy = create_test_arr(red_data)
    blue_numpy = create_test_arr(blue_data)
    numpy_result = sipi(nir_numpy, red_numpy, blue_numpy)
    general_output_checks(nir_numpy, numpy_result, sipi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    red_dask = create_test_arr(red_data, backend='dask')
    blue_dask = create_test_arr(blue_data, backend='dask')
    dask_result = sipi(nir_dask, red_dask, blue_dask)
    general_output_checks(nir_dask, dask_result, sipi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_sipi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    red_cupy = create_test_arr(red_data, backend='cupy')
    blue_cupy = create_test_arr(blue_data, backend='cupy')
    cupy_result = sipi(nir_cupy, red_cupy, blue_cupy)
    general_output_checks(nir_cupy, cupy_result, sipi_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    red_dask_cupy = create_test_arr(red_data, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(blue_data, backend='dask+cupy')
    dask_cupy_result = sipi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, sipi_expected_results)


# NBR -------------
def test_nbr_cpu():
    nir_numpy = create_test_arr(nir_data)
    swir_numpy = create_test_arr(swir2_data)
    numpy_result = nbr(nir_numpy, swir_numpy)
    general_output_checks(nir_numpy, numpy_result, nbr_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    swir_dask = create_test_arr(swir2_data, backend='dask')
    dask_result = nbr(nir_dask, swir_dask)
    general_output_checks(nir_dask, dask_result, nbr_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    swir_cupy = create_test_arr(swir2_data, backend='cupy')
    cupy_result = nbr(nir_cupy, swir_cupy)
    general_output_checks(nir_cupy, cupy_result, nbr_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    swir_dask_cupy = create_test_arr(swir2_data, backend='dask+cupy')
    dask_cupy_result = nbr(nir_dask_cupy, swir_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, nbr_expected_results)


# NBR2 -------------
def test_nbr2_cpu():
    swir1_numpy = create_test_arr(swir1_data)
    swir2_numpy = create_test_arr(swir2_data)
    numpy_result = nbr2(swir1_numpy, swir2_numpy)
    general_output_checks(swir1_numpy, numpy_result, nbr2_expected_results)

    # dask
    swir1_dask = create_test_arr(swir1_data, backend='dask')
    swir2_dask = create_test_arr(swir2_data, backend='dask')
    dask_result = nbr2(swir1_dask, swir2_dask)
    general_output_checks(swir1_dask, dask_result, nbr2_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Dnbr2ce not Available")
def test_nbr2_gpu():
    # cupy
    swir1_cupy = create_test_arr(swir1_data, backend='cupy')
    swir2_cupy = create_test_arr(swir2_data, backend='cupy')
    cupy_result = nbr2(swir1_cupy, swir2_cupy)
    general_output_checks(swir2_cupy, cupy_result, nbr2_expected_results)

    # dask + cupy
    swir1_dask_cupy = create_test_arr(swir1_data, backend='dask+cupy')
    swir2_dask_cupy = create_test_arr(swir2_data, backend='dask+cupy')
    dask_cupy_result = nbr2(swir1_dask_cupy, swir2_dask_cupy)
    general_output_checks(
        swir1_dask_cupy, dask_cupy_result, nbr2_expected_results)


# NDMI -------------
def test_ndmi_cpu():
    nir_numpy = create_test_arr(nir_data)
    swir1_numpy = create_test_arr(swir1_data)
    numpy_result = ndmi(nir_numpy, swir1_numpy)
    general_output_checks(nir_numpy, numpy_result, ndmi_expected_results)

    # dask
    nir_dask = create_test_arr(nir_data, backend='dask')
    swir1_dask = create_test_arr(swir1_data, backend='dask')
    dask_result = ndmi(nir_dask, swir1_dask)
    general_output_checks(nir_dask, dask_result, ndmi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndmi_gpu():
    # cupy
    nir_cupy = create_test_arr(nir_data, backend='cupy')
    swir1_cupy = create_test_arr(swir1_data, backend='cupy')
    cupy_result = ndmi(nir_cupy, swir1_cupy)
    general_output_checks(nir_cupy, cupy_result, ndmi_expected_results)

    # dask + cupy
    nir_dask_cupy = create_test_arr(nir_data, backend='dask+cupy')
    swir1_dask_cupy = create_test_arr(swir1_data, backend='dask+cupy')
    dask_cupy_result = ndmi(nir_dask_cupy, swir1_dask_cupy)
    general_output_checks(
        nir_dask_cupy, dask_cupy_result, ndmi_expected_results)


# EBBI -------------
def test_ebbi_cpu():
    # vanilla numpy version
    red_numpy = create_test_arr(red_data)
    swir_numpy = create_test_arr(swir1_data)
    tir_numpy = create_test_arr(tir_data)
    numpy_result = ebbi(red_numpy, swir_numpy, tir_numpy)
    general_output_checks(red_numpy, numpy_result, ebbi_expected_results)

    # dask
    red_dask = create_test_arr(red_data, backend='dask')
    swir_dask = create_test_arr(swir1_data, backend='dask')
    tir_dask = create_test_arr(tir_data, backend='dask')
    dask_result = ebbi(red_dask, swir_dask, tir_dask)
    general_output_checks(red_dask, dask_result, ebbi_expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ebbi_gpu():
    # cupy
    red_cupy = create_test_arr(red_data, backend='cupy')
    swir_cupy = create_test_arr(swir1_data, backend='cupy')
    tir_cupy = create_test_arr(tir_data, backend='cupy')
    cupy_result = ebbi(red_cupy, swir_cupy, tir_cupy)
    general_output_checks(red_cupy, cupy_result, ebbi_expected_results)

    # dask + cupy
    red_dask_cupy = create_test_arr(red_data, backend='dask+cupy')
    swir_dask_cupy = create_test_arr(swir1_data, backend='dask+cupy')
    tir_dask_cupy = create_test_arr(tir_data, backend='dask+cupy')
    dask_cupy_result = ebbi(red_dask_cupy, swir_dask_cupy, tir_dask_cupy)
    general_output_checks(
        red_dask_cupy, dask_cupy_result, ebbi_expected_results)


def test_true_color_cpu():
    # vanilla numpy version
    red_numpy = create_test_arr(red_data)
    green_numpy = create_test_arr(green_data)
    blue_numpy = create_test_arr(blue_data)
    numpy_result = true_color(
        red_numpy, green_numpy, blue_numpy, name='np_true_color'
    )
    assert numpy_result.name == 'np_true_color'
    general_output_checks(red_numpy, numpy_result, verify_attrs=False)

    # dask
    red_dask = create_test_arr(red_data, backend='dask')
    green_dask = create_test_arr(green_data, backend='dask')
    blue_dask = create_test_arr(blue_data, backend='dask')
    dask_result = true_color(
        red_dask, green_dask, blue_dask, name='dask_true_color'
    )
    assert dask_result.name == 'dask_true_color'
    general_output_checks(red_numpy, numpy_result, verify_attrs=False)

    np.testing.assert_allclose(
        numpy_result.data, dask_result.compute().data, equal_nan=True
    )
