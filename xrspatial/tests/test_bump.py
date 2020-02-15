from xrspatial import bump


def test_bump():
    bumps = bump(20, 20)
    assert bumps is not None
