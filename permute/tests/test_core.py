from __future__ import division, print_function, absolute_import

from nose.plugins.attrib import attr
from nose.tools import assert_raises

import numpy as np
from numpy.random import RandomState


from ..core import (binom_conf_interval,
                    corr,
                    two_sample,
                    one_sample)


def test_corr():
    prng = RandomState(42)
    x = prng.randint(5, size=10)
    y = x
    res1 = corr(x, y, seed=prng)
    res2 = corr(x, y)
    np.testing.assert_equal(len(res1), 5)
    np.testing.assert_equal(len(res2), 5)
    np.testing.assert_equal(res1[0], res2[0])
    np.testing.assert_equal(res1[1], res2[1])
    #np.testing.assert_equal(res1[2], res2[2])
    #np.testing.assert_equal(res1[3], res2[3])

    y = prng.randint(5, size=10)
    res1 = corr(x, y, seed=prng)
    res2 = corr(x, y)
    np.testing.assert_equal(len(res1), 5)
    np.testing.assert_equal(len(res2), 5)
    np.testing.assert_equal(res1[0], res2[0])
    #np.testing.assert_equal(res1[1], res2[1])
    #np.testing.assert_equal(res1[2], res2[2])
    #np.testing.assert_equal(res1[3], res2[3])


def test_binom_conf_interval():
    res = binom_conf_interval(10, 3)
    expected = (0.05154625578928545, 0.6915018049393984)
    np.testing.assert_equal(res, expected)

    res2 = binom_conf_interval(10, 5, cl=0.95, alternative="upper")
    expected2 = (0.0, 0.7775588989918742)
    np.testing.assert_equal(res2, expected2)

    res3 = binom_conf_interval(10, 5, cl=0.95, alternative="lower")
    expected3 = (0.22244110100812578, 1.0)
    np.testing.assert_equal(res3, expected3)


@attr('slow')
def test_two_sample():
    prng = RandomState(42)

    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (1.0, -2.90532344604777)
    np.testing.assert_almost_equal(res, expected)

    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.96975, -0.54460818906623765)
    np.testing.assert_equal(res, expected)

    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    np.testing.assert_equal(res, expected)

    res = two_sample(x, y, seed=42, interval="upper")
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    expected_ci = (0.0, 0.6675064023707297)
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_almost_equal(res[2], expected_ci)

    res = two_sample(x, y, seed=42, interval="lower")
    expected_ci = (0.6625865251964975, 1.0)
    np.testing.assert_almost_equal(res[2], expected_ci)
    res = two_sample(x, y, seed=42, interval="two-sided")
    expected_ci = (0.6621149803107692, 0.6679754440683887)
    np.testing.assert_almost_equal(res[2], expected_ci)

    res = two_sample(x, y, seed=42, keep_dist=True)
    exp_dist_firstfive = [0.089396492796047111,
                          0.17390295863272254,
                         -0.034211921065956274,
                          0.29103960535095719,
                         -0.76420778601368644]
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_equal(len(res[2]), 100000)
    np.testing.assert_almost_equal(res[2][:5], exp_dist_firstfive)

    res = two_sample(x, y, seed=42, interval="two-sided", keep_dist=True)
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_almost_equal(res[2], expected_ci)
    np.testing.assert_equal(len(res[3]), 100000)
    np.testing.assert_almost_equal(res[3][:5], exp_dist_firstfive)

def test_one_sample():
    prng = RandomState(42)
    
    x = np.array(range(5))
    y = x-1
    
    # case 1: one sample only
    res = one_sample(x, seed = 42, reps = 100)
    np.testing.assert_almost_equal(res[0], 0.05999999)
    np.testing.assert_equal(res[1], 2)
    
    # case 2: paired sample
    res = one_sample(x, y, seed = 42, reps = 100)
    np.testing.assert_equal(res[0], 0.02)
    np.testing.assert_equal(res[1], 1)
    
    # case 3: break it - supply x and y, but not paired
    y = np.append(y, 10)
    assert_raises(ValueError, one_sample, x, y)
    
    # case 4: say keep_dist=True
    res = one_sample(x, seed = 42, reps = 100, keep_dist=True)
    np.testing.assert_almost_equal(res[0], 0.05999999)
    np.testing.assert_equal(res[1], 2)
    np.testing.assert_equal(min(res[2]), -2)
    np.testing.assert_equal(max(res[2]), 2)
    np.testing.assert_equal(np.median(res[2]), 0)

    # case 5: use t as test statistic
    y = x + prng.normal(size=5)
    res = one_sample(x, y, seed = 42, reps = 100, stat = "t", alternative = "less")
    np.testing.assert_equal(res[0], 0.07)
    np.testing.assert_almost_equal(res[1], 1.4491883)