from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState


from ..core import (binom_conf_interval,
                    permute_within_groups)


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
