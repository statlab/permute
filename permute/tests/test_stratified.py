from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState

from nose.tools import assert_almost_equal, assert_less
from nose.plugins.attrib import attr

from ..stratified import stratified_permutationtest as spt
from ..stratified import corrcoef, sim_corr


def test_stratified_permutationtest():
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3]*3, 3)
    response = np.zeros_like(group)
    response[[0, 1, 3, 9, 10, 11, 18, 19, 20]] = 1

    res = spt(group, condition, response, iterations=1000, seed=42)
    res1 = spt(group, condition, response, iterations=1000, seed=42)
    assert_less(res[1], 0.01)
    assert_almost_equal(res[3], res1[3])


def test_corrcoef():
    prng = RandomState(42)
    x = prng.rand(10)
    y = x
    group = prng.randint(3, size=10)
    res1 = corrcoef(x, y, group)
    res2 = corrcoef(x, y, group)
    np.testing.assert_equal(res1, res2)


@attr('slow')
def test_sim_corr():
    prng = RandomState(42)
    x = prng.rand(10)
    y = x
    group = prng.randint(3, size=10)
    res1 = sim_corr(x, y, group, prng=prng)
    res2 = sim_corr(x, y, group)
    np.testing.assert_equal(res1[0], res2[0])
