from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState


from ..core import (binom_conf_interval,
                    permute_within_groups,
                    two_sample)


def test_permute_within_group():
    group = np.repeat([1, 2, 3], 9)
    condition = np.repeat([1, 2, 3]*3, 3)
    response = np.zeros_like(group)
    response[[0,  1,  3,  9, 10, 11, 18, 19, 20]] = 1
    groups = np.unique(group)

    prng1 = RandomState(42)
    prng2 = RandomState(42)
    res1 = permute_within_groups(group, condition, groups, prng1)
    res2 = permute_within_groups(group, condition, groups, prng2)
    np.testing.assert_equal(res1, res2)

@np.testing.dec.skipif(True)
def test_binom_conf_interval():
    res = binom_conf_interval(10, 3)

def test_two_sample():
    prng = RandomState(42)

    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (1.0, -2.9053234460477784)
    np.testing.assert_equal(res, expected)

    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.96975, -0.54460818906623765)
    np.testing.assert_equal(res, expected)

    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    np.testing.assert_equal(res, expected)

