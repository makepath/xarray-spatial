import numpy as np
from shapely.geometry import box, LineString, MultiPoint, Point, Polygon

from xrspatial.rasterize import rasterize

try:
    import cupy
    _has_cupy = True
except ImportError:
    _has_cupy = False

try:
    import geopandas as gpd
    _has_geopandas = True
except ImportError:
    _has_geopandas = False


def _make_polygon_grid(nx, ny, n_polys_x=10, n_polys_y=5):
    """Generate a grid of non-overlapping rectangular polygons."""
    rng = np.random.default_rng(55128)
    dx = 360.0 / n_polys_x
    dy = 180.0 / n_polys_y
    geoms = []
    vals = []
    for iy in range(n_polys_y):
        for ix in range(n_polys_x):
            # Slightly inset so polygons don't share edges
            x0 = -180.0 + ix * dx + 0.1
            y0 = -90.0 + iy * dy + 0.1
            x1 = x0 + dx - 0.2
            y1 = y0 + dy - 0.2
            geoms.append(box(x0, y0, x1, y1))
            vals.append(rng.uniform(1, 100))
    return geoms, vals


def _make_complex_polygons(n_polys=20, n_vertices=64):
    """Generate irregular polygons with many vertices."""
    rng = np.random.default_rng(99231)
    geoms = []
    vals = []
    for i in range(n_polys):
        cx = rng.uniform(-150, 150)
        cy = rng.uniform(-70, 70)
        r = rng.uniform(5, 20)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
        radii = r * (0.7 + 0.3 * rng.uniform(size=n_vertices))
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        coords = list(zip(xs, ys))
        coords.append(coords[0])
        geoms.append(Polygon(coords))
        vals.append(float(i + 1))
    return geoms, vals


def _make_line_network(n_lines=50, n_vertices=10):
    """Generate a set of polylines with multiple vertices."""
    rng = np.random.default_rng(31782)
    geoms = []
    vals = []
    for i in range(n_lines):
        xs = np.cumsum(rng.uniform(-5, 5, n_vertices)) + rng.uniform(-150, 150)
        ys = np.cumsum(rng.uniform(-5, 5, n_vertices)) + rng.uniform(-70, 70)
        xs = np.clip(xs, -180, 180)
        ys = np.clip(ys, -90, 90)
        geoms.append(LineString(zip(xs, ys)))
        vals.append(float(i + 1))
    return geoms, vals


def _make_point_cloud(n_points=500):
    """Generate scattered points."""
    rng = np.random.default_rng(44091)
    xs = rng.uniform(-180, 180, n_points)
    ys = rng.uniform(-90, 90, n_points)
    geoms = [Point(x, y) for x, y in zip(xs, ys)]
    vals = rng.uniform(1, 100, n_points).tolist()
    return geoms, vals


# -------------------------------------------------------------------------
# Polygon rasterization benchmarks
# -------------------------------------------------------------------------

class RasterizePolygons:
    params = ([100, 300, 1000, 3000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        self.geoms, self.vals = _make_polygon_grid(nx, ny)
        self.pairs = list(zip(self.geoms, self.vals))
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny
        self.use_cuda = (backend == "cupy")

    def time_rasterize_polygons(self, nx, backend):
        rasterize(self.pairs, width=self.width, height=self.height,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)


class RasterizeComplexPolygons:
    params = ([100, 300, 1000, 3000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        self.geoms, self.vals = _make_complex_polygons(n_polys=20,
                                                       n_vertices=64)
        self.pairs = list(zip(self.geoms, self.vals))
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny
        self.use_cuda = (backend == "cupy")

    def time_rasterize_complex_polygons(self, nx, backend):
        rasterize(self.pairs, width=self.width, height=self.height,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)


# -------------------------------------------------------------------------
# Line rasterization benchmarks
# -------------------------------------------------------------------------

class RasterizeLines:
    params = ([100, 300, 1000, 3000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        self.geoms, self.vals = _make_line_network(n_lines=50, n_vertices=10)
        self.pairs = list(zip(self.geoms, self.vals))
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny
        self.use_cuda = (backend == "cupy")

    def time_rasterize_lines(self, nx, backend):
        rasterize(self.pairs, width=self.width, height=self.height,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)


# -------------------------------------------------------------------------
# Point rasterization benchmarks
# -------------------------------------------------------------------------

class RasterizePoints:
    params = ([100, 300, 1000, 3000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        self.geoms, self.vals = _make_point_cloud(n_points=500)
        self.pairs = list(zip(self.geoms, self.vals))
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny
        self.use_cuda = (backend == "cupy")

    def time_rasterize_points(self, nx, backend):
        rasterize(self.pairs, width=self.width, height=self.height,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)


# -------------------------------------------------------------------------
# Mixed geometry rasterization
# -------------------------------------------------------------------------

class RasterizeMixed:
    params = ([100, 300, 1000, 3000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        poly_geoms, poly_vals = _make_polygon_grid(nx, ny, n_polys_x=5,
                                                   n_polys_y=3)
        line_geoms, line_vals = _make_line_network(n_lines=30, n_vertices=8)
        pt_geoms, pt_vals = _make_point_cloud(n_points=200)
        self.pairs = (list(zip(poly_geoms, poly_vals))
                      + list(zip(line_geoms, line_vals))
                      + list(zip(pt_geoms, pt_vals)))
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny
        self.use_cuda = (backend == "cupy")

    def time_rasterize_mixed(self, nx, backend):
        rasterize(self.pairs, width=self.width, height=self.height,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)


# -------------------------------------------------------------------------
# GeoDataFrame input (measures parsing overhead)
# -------------------------------------------------------------------------

class RasterizeGeoDataFrame:
    params = ([100, 300, 1000, 3000],)
    param_names = ("nx",)

    def setup(self, nx):
        if not _has_geopandas:
            raise NotImplementedError("geopandas not available")
        ny = nx // 2
        geoms, vals = _make_polygon_grid(nx, ny)
        self.gdf = gpd.GeoDataFrame({'value': vals}, geometry=geoms)
        self.bounds = (-180, -90, 180, 90)
        self.width = nx
        self.height = ny

    def time_rasterize_geodataframe(self, nx):
        rasterize(self.gdf, width=self.width, height=self.height,
                  bounds=self.bounds, column='value', fill=0)


# -------------------------------------------------------------------------
# Scaling: many polygons
# -------------------------------------------------------------------------

class RasterizeScaling:
    """Measure how rasterize scales with polygon count at fixed resolution."""
    params = ([10, 50, 200, 1000], ["numpy", "cupy"])
    param_names = ("n_polys", "backend")

    def setup(self, n_polys, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        rng = np.random.default_rng(82714)
        geoms = []
        vals = []
        for i in range(n_polys):
            cx = rng.uniform(-150, 150)
            cy = rng.uniform(-70, 70)
            r = rng.uniform(2, 10)
            geoms.append(Point(cx, cy).buffer(r, resolution=16))
            vals.append(float(i + 1))
        self.pairs = list(zip(geoms, vals))
        self.bounds = (-180, -90, 180, 90)
        self.use_cuda = (backend == "cupy")

    def time_rasterize_scaling(self, n_polys, backend):
        rasterize(self.pairs, width=1000, height=500,
                  bounds=self.bounds, fill=0, use_cuda=self.use_cuda)
