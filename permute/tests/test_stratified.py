from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import assert_almost_equal, assert_less

from ..stratified import stratified_permutationtest as spt


def test_stratified_permutationtest():
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3]*3, 3)
    response = np.zeros_like(group)
    response[[0,  1,  3,  9, 10, 11, 18, 19, 20]] = 1

    res = spt(group, condition, response, iterations=1000, seed=42)
    res1 = spt(group, condition, response, iterations=1000, seed=42)
    assert_less(res[1], 0.01)
    assert_almost_equal(res[3], res1[3])
