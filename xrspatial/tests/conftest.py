import numpy as np
import pytest


@pytest.fixture
def random_data(size, dtype):
    rng = np.random.default_rng(2841)
    data = rng.integers(-100, 100, size=size)
    data = data.astype(dtype)
    return data


@pytest.fixture
def elevation_raster():
    elevation = np.array([
        [   np.nan,    np.nan,    np.nan,    np.nan,    np.nan,    np.nan],
        [704.237  , 242.24084, 429.3324 , 779.8816 , 193.29506, 984.6926 ],
        [226.56795, 815.7483 , 290.6041 ,  76.49687, 820.89716,  32.27882],
        [344.8238 , 256.34998, 806.8326 , 602.0442 , 721.1633 , 496.95636],
        [185.43515, 834.10425, 387.0871 , 716.0262 ,  49.61273, 752.95483],
        [302.4271 , 151.49211, 442.32797, 358.4702 , 659.8187 , 447.1241 ],
        [148.04834, 819.2133 , 468.97913, 977.11694, 597.69666, 999.14185],
        [268.1575 , 625.96466, 840.26483, 448.28333, 859.2699 , 528.04095]
    ], dtype=np.float32)
    return elevation


@pytest.fixture
def raster():
    data = np.array([
        [6., 7., 3., 4., 8., 1.],
        [4., 9., 7., 5., 6., 9.],
        [4., 3., 3., 1., 3., 7.],
        [3., 4., 9., 3., 7., 0.],
        [2., 1., 6., 5., 6., 2.],
        [4., 2., 4., 3., 8., 5.],
        [4., 1., 8., 5., 7., 0.],
        [7., 4., 6., 4., 1., 1.]
    ], dtype=np.float32)
    return data
