from xrspatial.multispectral import arvi, ebbi, evi, gci, nbr, nbr2, ndmi, ndvi, savi, sipi

from .common import Benchmarking, get_xr_dataarray


class Multispectral(Benchmarking):
    def __init__(self):
        super().__init__()

    def setup(self, nx, type):
        ny = nx // 2
        self.red = get_xr_dataarray((ny, nx), type, seed=100)
        self.green = get_xr_dataarray((ny, nx), type, seed=200)
        self.blue = get_xr_dataarray((ny, nx), type, seed=300)
        self.nir = get_xr_dataarray((ny, nx), type, seed=400)
        self.swir1 = get_xr_dataarray((ny, nx), type, seed=500)
        self.swir2 = get_xr_dataarray((ny, nx), type, seed=600)
        self.tir = get_xr_dataarray((ny, nx), type, seed=700)


class Arvi(Multispectral):
    def time_arvi(self, nx, type):
        arvi(self.nir, self.red, self.blue)


class Evi(Multispectral):
    def time_evi(self, nx, type):
        evi(self.nir, self.red, self.blue)


class Gci(Multispectral):
    def time_gci(self, nx, type):
        gci(self.nir, self.green)


class Nbr(Multispectral):
    def time_nbr(self, nx, type):
        nbr(self.nir, self.swir2)


class Nbr2(Multispectral):
    def time_nbr2(self, nx, type):
        nbr2(self.swir1, self.swir2)


class Ndvi(Multispectral):
    def time_ndvi(self, nx, type):
        ndvi(self.nir, self.red)


class Ndmi(Multispectral):
    def time_ndmi(self, nx, type):
        ndmi(self.nir, self.swir1)


class Savi(Multispectral):
    def time_savi(self, nx, type):
        savi(self.nir, self.red)


class Sipi(Multispectral):
    def time_sipi(self, nx, type):
        sipi(self.nir, self.red, self.blue)


class Ebbi(Multispectral):
    def time_ebbi(self, nx, type):
        ebbi(self.red, self.swir1, self.tir)
