import numpy as np
import pytest


@pytest.fixture
def random_data(size, dtype):
    rng = np.random.default_rng(2841)
    data = rng.integers(-100, 100, size=size)
    data = data.astype(dtype)
    return data
