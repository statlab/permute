from __future__ import division, print_function, absolute_import

from nose.plugins.attrib import attr
from nose.tools import assert_raises

import numpy as np
from numpy.random import RandomState


from ..core import (corr,
                    two_sample,
                    two_sample_conf_int,
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


@attr('slow')
def test_two_sample():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    res = two_sample(x, y, seed=42)
    expected = (1.0, -2.90532344604777)
    np.testing.assert_almost_equal(res, expected)
    
    # this one has keep_dist = True
    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    res2 = two_sample(x, y, seed=42, keep_dist=True)
    expected = (0.96975, -0.54460818906623765)
    np.testing.assert_equal(res, expected)
    np.testing.assert_equal(res2[:2], expected)

    # Normal-normal, same means
    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    np.testing.assert_equal(res, expected)

    # Ask for an upper confidence limit
    res = two_sample(x, y, seed=42, interval="upper")
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    expected_ci = (0.0, 0.6675064023707297)
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_almost_equal(res[2], expected_ci)
    
    # Ask for a lower confidence limit
    res = two_sample(x, y, seed=42, interval="lower")
    expected_ci = (0.6625865251964975, 1.0)
    np.testing.assert_almost_equal(res[2], expected_ci)
    res = two_sample(x, y, seed=42, interval="two-sided")
    expected_ci = (0.6621149803107692, 0.6679754440683887)
    np.testing.assert_almost_equal(res[2], expected_ci)

    # Check the permutation distribution
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

    # Ask for two-sided interval and keep_dist
    res = two_sample(x, y, seed=42, interval="two-sided", keep_dist=True)
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_almost_equal(res[2], expected_ci)
    np.testing.assert_equal(len(res[3]), 100000)
    np.testing.assert_almost_equal(res[3][:5], exp_dist_firstfive)
    
    # Define a lambda function (K-S test)
    f = lambda u: np.max( \
        [abs(sum(u[:len(x)]<=v)/len(x)-sum(u[len(x):]<=v)/len(y)) for v in u]\
        )
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (0.68999999999999995, 0.20000000000000007)
    np.testing.assert_equal(res[0], expected[0])
    np.testing.assert_equal(res[1], expected[1])

    # Test null with shift other than zero
    res = two_sample(x, y, seed=42, shift=2)
    np.testing.assert_equal(res[0], 1)
    np.testing.assert_equal(res[1], expected_ts)
    res = two_sample(x, y, seed=42, shift=2, alternative="less")
    np.testing.assert_equal(res[0], 0)
    np.testing.assert_equal(res[1], -expected_ts)

def test_two_sample_conf_int():
    prng = RandomState(42)
    
    # Shift is -1
    x = range(5)
    y = range(1,6)
    res = two_sample_conf_int(x, y, seed=prng)
    expected_ci =  (-3.039627925495512, 1.3180643632398046)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="upper")
    expected_ci = (-5, 1)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="lower")
    expected_ci = (-3, 5)
    np.testing.assert_almost_equal(res, expected_ci)
    
    
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