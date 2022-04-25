from xrspatial.datasets import make_terrain
from xrspatial.utils import canvas_like


def test_canvas_like():
    # aspect ratio is 1:1
    terrain_shape = (1000, 1000)
    terrain = make_terrain(shape=terrain_shape)
    terrain_res = canvas_like(terrain, width=50)
    assert terrain_res.shape == (50, 50)
