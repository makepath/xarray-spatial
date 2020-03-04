import datashader as ds

from math import pi as PI


class MercatorCanvas(ds.Canvas):

    def __init__(self, level=5):
        self.level = level
        size = 256 * 2 ** level
        mercator = PI * 6378137
        self.plot_width = size
        self.plot_height = size
        self.x_range = (-mercator, mercator)
        self.y_range = (-mercator, mercator)


cvs = MercatorCanvas(level=30)
