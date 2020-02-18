import datashader as ds

from xrspatial import generate_terrain

W = 25
H = 30

X_RANGE = (0, 500)
Y_RANGE = (0, 500)


def test_generate_terrain():
    csv = ds.Canvas(x_range=X_RANGE, y_range=Y_RANGE,
                    plot_width=W, plot_height=H)
    terrain = generate_terrain(canvas=csv)
    assert terrain is not None
