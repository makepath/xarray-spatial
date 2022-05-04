import numpy as np

from xrspatial.classify import equal_interval, natural_breaks, quantile, reclassify

from .common import get_xr_dataarray


class Classify:
    params = ([100, 300, 1000, 3000, 10000], [1, 10, 100], ["numpy", "cupy"])
    param_names = ("nx", "nbins", "type")

    def setup(self, nx, nbins, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        min_val = np.nanmin(self.agg.data)
        max_val = np.nanmax(self.agg.data)
        self.nbins = nbins
        self.bins = np.linspace(min_val, max_val, self.nbins)
        self.new_values = np.arange(nbins)


class Reclassify(Classify):
    def time_reclassify(self, nx, nbins, type):
        reclassify(self.agg, self.bins, self.new_values)


class Quantile(Classify):
    def time_quantile(self, nx, nbins, type):
        quantile(self.agg, k=self.nbins)


class NaturalBreaks(Classify):
    params = ([100, 300, 1000, 3000, 10000], [1, 10], ["numpy"])

    def time_natural_breaks(self, nx, nbins, type):
        natural_breaks(self.agg, k=self.nbins)


class EqualInterval(Classify):
    def time_equal_interval(self, nx, nbins, type):
        equal_interval(self.agg, k=self.nbins)
