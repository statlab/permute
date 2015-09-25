from __future__ import division, print_function, absolute_import

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState


from ..core import (corr,
                    two_sample,
                    two_sample_shift,
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
    
    # This one has keep_dist = True
    y = prng.normal(1.4, size=20)
    res = two_sample(x, y, seed=42)
    res2 = two_sample(x, y, seed=42, keep_dist=True)
    expected = (0.96975, -0.54460818906623765)
    np.testing.assert_approx_equal(res[0], expected[0], 2)
    np.testing.assert_equal(res[1], expected[1])
    np.testing.assert_approx_equal(res2[0], expected[0], 2)
    np.testing.assert_equal(res2[1], expected[1])

    # Normal-normal, same means
    y = prng.normal(1, size=20)
    res = two_sample(x, y, seed=42)
    expected = (0.66505000000000003, -0.13990200413154097)
    np.testing.assert_approx_equal(res[0], expected[0], 2)
    np.testing.assert_equal(res[1], expected[1])

    # Check the permutation distribution
    res = two_sample(x, y, seed=42, keep_dist=True)
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    exp_dist_firstfive = [0.08939649,
                         -0.26323896,
                         0.15428355,
                         -0.0294264,
                         0.03318078]
    np.testing.assert_approx_equal(res[0], expected_pv, 2)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_equal(len(res[2]), 100000)
    np.testing.assert_almost_equal(res[2][:5], exp_dist_firstfive)

    # Define a lambda function (K-S test)
    f = lambda u,v: np.max( \
        [abs(sum(u<=val)/len(u)-sum(v<=val)/len(v)) \
        for val in np.concatenate([u,v])])
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (0.68, 0.20000000000000007)
    np.testing.assert_equal(res[0], expected[0])
    np.testing.assert_equal(res[1], expected[1])


def test_two_sample_shift():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    f = lambda u: u - 3
    finv = lambda u: u + 3
    f_err = lambda u: 2*u
    f_err_inv = lambda u: u/2
    expected_ts = -2.9053234460477784
    
    # Test null with shift other than zero
    res = two_sample_shift(x, y, seed=42, shift=2)
    np.testing.assert_equal(res[0], 1)
    np.testing.assert_equal(res[1], expected_ts)
    res2 = two_sample_shift(x, y, seed=42, shift=2, keep_dist=True)
    np.testing.assert_equal(res2[0], 1)
    np.testing.assert_equal(res2[1], expected_ts)
    np.testing.assert_almost_equal(res2[2][:3], np.array([1.55886506,  0.87281296,  1.13611123]))
    res = two_sample_shift(x, y, seed=42, shift=2, alternative="less")
    np.testing.assert_equal(res[0], 0)
    np.testing.assert_equal(res[1], expected_ts)
    
    # Test null with shift -3
    res = two_sample_shift(x, y, seed=42, shift=(f, finv))
    np.testing.assert_equal(res[0], 0.38074999999999998)
    np.testing.assert_equal(res[1], expected_ts)
    res = two_sample_shift(x, y, seed=42, shift=(f, finv), alternative="less")
    np.testing.assert_equal(res[0], 0.61924999999999997)
    np.testing.assert_equal(res[1], expected_ts)
    
    # Test null with multiplicative shift
    res = two_sample_shift(x, y, seed=42, shift=(f_err, f_err_inv), alternative="two-sided")
    np.testing.assert_equal(res[0], 0.17139)
    np.testing.assert_equal(res[1], expected_ts)

    # Define a lambda function
    f = lambda u,v: np.max(u) - np.max(v)
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (1, -3.2730653690015465)
    np.testing.assert_equal(res[0], expected[0])
    np.testing.assert_equal(res[1], expected[1])


@raises(ValueError)
def test_two_sample_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1,6))
    shift = lambda u: u + 3
    two_sample_shift(x, y, seed=5, shift=shift)
        

@attr('slow')
def test_two_sample_conf_int():
    prng = RandomState(42)
    
    # Shift is -1
    x = np.array(range(5))
    y = np.array(range(1,6))
    res = two_sample_conf_int(x, y, seed=prng)
    expected_ci =  (-3.0442353, 1.0000037)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="upper")
    expected_ci = (-5, 1)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="lower")
    expected_ci = (-3, 5)
    np.testing.assert_almost_equal(res, expected_ci)
    
    # Specify shift with a function pair
    shift = (lambda u, d: u + d, lambda u,d: u - d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-3, 1))
    
    # Specify shift with a multiplicative pair
    shift = (lambda u, d: u*d, lambda u,d: u/d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-1, -1))


@raises(AssertionError)
def test_two_sample_conf_int_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1,6))
    shift = (lambda u,d: -d*u, lambda u,d: -u/d)
    two_sample_conf_int(x, y, seed=5, shift=shift)


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