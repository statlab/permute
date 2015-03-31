from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import assert_almost_equal, assert_less

from ..stratified import stratifiedPermutationTest as spt


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
