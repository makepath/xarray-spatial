import pytest

import numpy as np


@pytest.fixture
def random_data(size, dtype):
    data = np.random.randint(low=-100, high=100, size=size)
    data = data.astype(dtype)
    return data
