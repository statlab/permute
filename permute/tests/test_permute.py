from ..permute import stratifiedPermutationTest as spt
from nose.tools import assert_equal, assert_less

import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_almost_equal)


def test_stratifiedPermutationTest():
    group = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    condition = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    response = np.array(
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    res = spt(group, condition, response, iterations=1000) 
    res1 = spt(group, condition, response, iterations=1000) 
    assert_less(res[1], 0.01)
    assert_almost_equal(res[3], res1[3])
