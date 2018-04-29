from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises

import numpy as np
from numpy.random import RandomState
from scipy.stats import binom


from ..core import (corr,
                    spearman_corr,
                    two_sample,
                    two_sample_shift,
                    two_sample_conf_int,
                    two_sample_fit,
                    one_sample,
                    one_sample,
                    one_sample_conf_int,
                    one_sample_percentile,
                    one_sample_percentile_ci)


def test_corr():
    prng = RandomState(42)
    x = prng.randint(5, size=10)
    y = x
    res1 = corr(x, y, seed=prng)
    res2 = corr(x, y)
    np.testing.assert_equal(len(res1), 3)
    np.testing.assert_equal(len(res2), 3)
    np.testing.assert_equal(res1[0], 1)
    np.testing.assert_equal(res2[0], 1)
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)

    y = prng.randint(5, size=10)
    res1 = corr(x, y, alternative="less", seed=prng)
    res2 = corr(x, y, alternative="less")
    np.testing.assert_equal(len(res1), 3)
    np.testing.assert_equal(len(res2), 3)
    np.testing.assert_equal(res1[0], res2[0])
    np.testing.assert_almost_equal(res1[1], res2[1], decimal=1)

    res1 = corr(x, y, alternative="two-sided", seed=prng)
    res2 = corr(x, y, alternative="greater")
    np.testing.assert_equal(len(res1), 3)
    np.testing.assert_equal(len(res2), 3)
    np.testing.assert_equal(res1[0], res2[0])
    np.testing.assert_almost_equal(res1[1], res2[1]*2, decimal=1)


def test_spearman_corr():
    prng = RandomState(42)
    x = np.array([2, 4, 6, 8, 10])
    y = np.array([1, 3, 5, 6, 9])
    xorder = np.array([1, 2, 3, 4, 5])
    res1 = corr(xorder, xorder, seed=prng)
    
    prng = RandomState(42)
    res2 = spearman_corr(x, y, seed=prng)
    np.testing.assert_equal(res1[0], res2[0])
    np.testing.assert_equal(res1[1], res2[1])
    np.testing.assert_array_equal(res1[2], res2[2])


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
    f = lambda u, v: np.max(
        [abs(sum(u <= val) / len(u) - sum(v <= val) / len(v))
         for val in np.concatenate([u, v])])
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (0.68, 0.20000000000000007)
    np.testing.assert_equal(res[0], expected[0])
    np.testing.assert_equal(res[1], expected[1])

def test_two_sample_fit():
    x = np.array(range(10))
    y = np.array(range(10))
    alpha = 0.01
    res = two_sample_fit(x, y, alpha)
    np.testing.assert_equal(res, True)

    y = np.array(range(10, 20))
    alpha = 0.1
    res = two_sample_fit(x, y, alpha)
    np.testing.assert_equal(res, False)

def test_two_sample_shift():
    prng = RandomState(42)

    # Normal-normal, different means examples
    x = prng.normal(1, size=20)
    y = prng.normal(4, size=20)
    f = lambda u: u - 3
    finv = lambda u: u + 3
    f_err = lambda u: 2 * u
    f_err_inv = lambda u: u / 2
    expected_ts = -2.9053234460477784

    # Test null with shift other than zero
    res = two_sample_shift(x, y, seed=42, shift=2)
    np.testing.assert_equal(res[0], 1)
    np.testing.assert_equal(res[1], expected_ts)
    res2 = two_sample_shift(x, y, seed=42, shift=2, keep_dist=True)
    np.testing.assert_equal(res2[0], 1)
    np.testing.assert_equal(res2[1], expected_ts)
    np.testing.assert_almost_equal(res2[2][:3], np.array(
        [1.55886506,  0.87281296,  1.13611123]))
    res = two_sample_shift(x, y, seed=42, shift=2, alternative="less")
    np.testing.assert_equal(res[0], 0)
    np.testing.assert_equal(res[1], expected_ts)

    # Test null with shift -3
    res = two_sample_shift(x, y, seed=42, shift=(f, finv))
    np.testing.assert_equal(res[0], 0.38074999999999998)
    np.testing.assert_equal(res[1], expected_ts)
    res = two_sample_shift(x, y, seed=42, shift=(f, finv), alternative="less")
    np.testing.assert_almost_equal(res[0], 0.61925)
    np.testing.assert_equal(res[1], expected_ts)

    # Test null with multiplicative shift
    res = two_sample_shift(x, y, seed=42,
        shift=(f_err, f_err_inv), alternative="two-sided")
    np.testing.assert_equal(res[0], 0)
    np.testing.assert_equal(res[1], expected_ts)

    # Define a lambda function
    f = lambda u, v: np.max(u) - np.max(v)
    res = two_sample(x, y, seed=42, stat=f, reps=100)
    expected = (1, -3.2730653690015465)
    np.testing.assert_equal(res[0], expected[0])
    np.testing.assert_equal(res[1], expected[1])


@raises(ValueError)
def test_two_sample_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = lambda u: u + 3
    two_sample_shift(x, y, seed=5, shift=shift)


@attr('slow')
def test_two_sample_conf_int():
    prng = RandomState(42)

    # Shift is -1
    x = np.array(range(5))
    y = np.array(range(1, 6))
    res = two_sample_conf_int(x, y, seed=prng)
    expected_ci = (-3.5, 1.012957978810817)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="upper")
    expected_ci = (-5, 1)
    np.testing.assert_almost_equal(res, expected_ci)
    res = two_sample_conf_int(x, y, seed=prng, alternative="lower")
    expected_ci = (-3, 5)
    np.testing.assert_almost_equal(res, expected_ci)

    # Specify shift with a function pair
    shift = (lambda u, d: u + d, lambda u, d: u - d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-3.5, 1))

    # Specify shift with a multiplicative pair
    shift = (lambda u, d: u * d, lambda u, d: u / d)
    res = two_sample_conf_int(x, y, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-1, -1))


@raises(AssertionError)
def test_two_sample_conf_int_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = (lambda u, d: -d * u, lambda u, d: -u / d)
    two_sample_conf_int(x, y, seed=5, shift=shift)


def test_one_sample():
    prng = RandomState(42)

    x = np.array(range(5))
    y = x - 1

    # case 1: one sample only
    res = one_sample(x, seed=42, reps=100)
    np.testing.assert_almost_equal(res[0], 0.05999999)
    np.testing.assert_equal(res[1], 2)

    # case 2: paired sample
    res = one_sample(x, y, seed=42, reps=100)
    np.testing.assert_equal(res[0], 0.02)
    np.testing.assert_equal(res[1], 1)

    # case 3: break it - supply x and y, but not paired
    y = np.append(y, 10)
    assert_raises(ValueError, one_sample, x, y)

    # case 4: say keep_dist=True
    res = one_sample(x, seed=42, reps=100, keep_dist=True)
    np.testing.assert_almost_equal(res[0], 0.05999999)
    np.testing.assert_equal(res[1], 2)
    np.testing.assert_equal(min(res[2]), -2)
    np.testing.assert_equal(max(res[2]), 2)
    np.testing.assert_equal(np.median(res[2]), 0)

    # case 5: use t as test statistic
    y = x + prng.normal(size=5)
    res = one_sample(x, y, seed=42, reps=100, stat="t", alternative="less")
    np.testing.assert_almost_equal(res[0], 0.05)
    np.testing.assert_almost_equal(res[1], -1.4491883)

    # case 6: use median as test statistic
    res = one_sample(x, seed=42, reps=100, stat="median")
    np.testing.assert_almost_equal(res[0], 0.14)
    np.testing.assert_equal(res[1], 2)

    # case 7: Test statistic is a function
    pcntl = lambda x: np.percentile(x, 20)
    res = one_sample(x, seed=42, reps=100, stat=pcntl)
    np.testing.assert_almost_equal(res[0], 0.059999999999)
    np.testing.assert_almost_equal(res[1], 0.8)

    prng = RandomState(42)

    x = np.array(range(10))
    y = x - 1

    # case 1:
    res0 = one_sample(x, seed=prng, reps=100, center=3)
    res1 = one_sample(x, seed=prng, reps=100, center=7)
    np.testing.assert_almost_equal(res0[0], 0.07)
    np.testing.assert_equal(res0[1], np.mean(x))
    np.testing.assert_almost_equal(res1[0], 1)

    # case 2: paired sample
    res = one_sample(x, y, seed=42, reps=100, center=1, keep_dist=True)
    np.testing.assert_almost_equal(res[0], 1)
    np.testing.assert_equal(res[1], 1)
    dist_unique = np.unique(res[2]) # Distribution should be all 1's
    np.testing.assert_equal(len(dist_unique), 1)
    np.testing.assert_equal(dist_unique[0], 1)

    # case 3: t test statistic
    x = np.array(range(5))
    res = one_sample(x, seed=42, reps=100, stat="t", alternative="less", center=0.1)
    np.testing.assert_almost_equal(res[0], 0.93999999999999995)
    np.testing.assert_almost_equal(res[1], 2.8284271247461898)

    # case 4: break it - supply x and y, but not paired
    y = np.append(y, 10)
    assert_raises(ValueError, one_sample, x, y)

    # case 5: use median as test statistic
    x = np.array(range(10))
    res = one_sample(x, seed=42, reps=100, stat="median", center=4.5)
    np.testing.assert_almost_equal(res[0], 0.53)
    np.testing.assert_equal(res[1], 4.5)

    # case 6: Test statistic is a function
    pcntl = lambda x: np.percentile(x, 20)
    res = one_sample(x, seed=42, reps=100, stat=pcntl, center=2)
    np.testing.assert_almost_equal(res[0], 0.029999999999999999)
    np.testing.assert_almost_equal(res[1], 1.8)


@attr('slow')
def test_one_sample_conf_int():
    prng = RandomState(42)

    # Standard confidence interval
    x = np.array(range(10))
    res = one_sample_conf_int(x, seed=prng)
    expected_ci = (2.2696168, 6.6684788)
    np.testing.assert_almost_equal(res, expected_ci)
    res = one_sample_conf_int(x, seed=prng, alternative="upper")
    expected_ci = (-4.5, 6.255232305502077)
    np.testing.assert_almost_equal(res, expected_ci)
    res = one_sample_conf_int(x, seed=prng, alternative="lower")
    expected_ci = (2.7680067828582393, 13.5)
    np.testing.assert_almost_equal(res, expected_ci)

    # Normal distribution shift centered at 1
    norm = prng.normal(0, 1, size=100)
    shift = prng.normal(1, 1, size=100)
    diff = norm - shift
    res = one_sample_conf_int(norm, diff, seed=prng)
    expected_ci = (0.8, 1.2)
    np.testing.assert_almost_equal(res, expected_ci, decimal=1)
    

    # Specify shift with a function pair
    shift = (lambda u, d: u + d, lambda u, d: u - d)
    res = one_sample_conf_int(x, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (2.333333333333319, 6.749999999999943))

    # Specify shift with a multiplicative pair
    shift = (lambda u, d: u * d, lambda u, d: u / d)
    res = one_sample_conf_int(norm, seed=5, shift=shift)
    np.testing.assert_almost_equal(res, (-0.0653441,  0.309073 ))

    # break it - supply x and y, but not paired
    y = np.append(x, 10)
    assert_raises(ValueError, one_sample_conf_int, x, y)

    # Testing with sample statistic of median
    res = one_sample_conf_int(x, seed=42, reps=100, stat="median")
    np.testing.assert_almost_equal(res, (2.499999999999458, 6.999999999999045))

    # Testing with t statistic
    prng = RandomState(42)
    x = np.arange(20)
    y = x + prng.normal(size=20)
    res = one_sample_conf_int(x, y, reps=100, seed=prng, stat='t')
    np.testing.assert_almost_equal(res, (-0.3018817477447192, 0.6510547144565948))

    min_func = lambda x: np.min(x)
    res = one_sample_conf_int(x, y, reps=100, seed=42, stat=min_func)
    np.testing.assert_almost_equal(res, (-0.5084626431347878, 0.167033714575861))


@raises(AssertionError)
def test_one_sample_conf_int_bad_shift():
    # Break it with a bad shift
    x = np.array(range(5))
    y = np.array(range(1, 6))
    shift = (lambda u, d: -d * u, lambda u, d: -u / d)
    one_sample_conf_int(x, y, seed=5, shift=shift)


def test_one_sample_percentile():
    prng = RandomState(42)
    x = np.arange(1, 101)
    res = one_sample_percentile(x, 50, p=50, alternative="less")
    np.testing.assert_equal(res[1], 50)
    expected_pval = binom.cdf(res[1], len(x), 0.5)
    np.testing.assert_equal(res[0], expected_pval)

    res = one_sample_percentile(x, 75, p=70, alternative="greater")
    np.testing.assert_equal(res[1], 75)
    expected_pval = 1 - binom.cdf(res[1], len(x), 0.7)
    np.testing.assert_equal(res[0], expected_pval)

    res = one_sample_percentile(x, 20, p=30, alternative="two-sided")
    np.testing.assert_equal(res[1], 20)
    expected_pval = 2 * binom.cdf(res[1], len(x), 0.3)
    np.testing.assert_equal(res[0], expected_pval)

    np.testing.assert_raises(ValueError, one_sample_percentile, x, x_p = 50, p=101)

def test_one_sample_percentile_ci():
    x = np.arange(0, 100)
    res = one_sample_percentile_ci(x)
    np.testing.assert_equal(res[0], 39)
    np.testing.assert_almost_equal(res[1], 59)

    res = one_sample_percentile_ci(x, p=50, alternative="upper")
    np.testing.assert_equal(res[0], 0)
    np.testing.assert_equal(res[1], 58)

    res = one_sample_percentile_ci(x, p=50, alternative="lower")
    np.testing.assert_equal(res[0], 40)
    np.testing.assert_equal(res[1], 99)

    y = np.append(x, 10)
    np.testing.assert_raises(ValueError, one_sample_percentile_ci, x, p=101)

    prng = RandomState(42)
    z = prng.normal(0, 5, 100)
    res = one_sample_percentile_ci(z, p=50)
    expected_ci = (-1.546061879256073, 0.55461294854933041)

