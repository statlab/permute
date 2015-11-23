from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises

from ..regression import simulate_ts_dist

def test_simulate_ts_dist():
    prng = RandomState(42)
    X = np.array([np.ones(10), prng.randint(1, 4, 10)]).T
    beta = np.array([1.2, 2])
    epsilon = prng.normal(0, .15, 10)
    y = X.dot(beta) + epsilon

    expected_res1 = {'dist': None,
                     'geq': 5,
                     'obs_ts': 33.440639668967926,
                     'pvalue': 0.00050000000000000001,
                     'num_perm': 10000}
    res1 = simulate_ts_dist(X[:, 1], y, seed=42)
    assert_equal(res1, expected_res1)
    #expected_res2 = {'geq': 9457,
    #                 'obs_ts': 0.46285714285714286,
    #                 'num_perm': 10000}
    #res2 = simulate_ts_dist(res[:5], seed=42, keep_dist=True)
    #assert_equal(res2['geq'], expected_res2['geq'])
    #assert_equal(res2['obs_ts'], expected_res2['obs_ts'])
    #assert_equal(res2['num_perm'], expected_res2['num_perm'])
    #assert_equal(res2['dist'].shape, (10000,))
