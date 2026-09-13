import numpy as np

from utils.seed import seed_everything

def test_numpy_draws_reproducible_across_calls():
    seed_everything(0)
    first = np.random.rand(5)

    seed_everything(0)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)