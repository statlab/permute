import sys
import pytest

import numpy as np
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256
from cryptorandom.sample import random_sample, random_permutation

from ..utils import (binom_conf_interval,
                     hypergeom_conf_interval,
                     hypergeometric,
                     binomial_p,
                     get_prng,
                     permute,
                     permute_rows,
                     permute_within_groups,
                     permute_incidence_fixed_sums,
                     potential_outcomes)


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

    res4 = binom_conf_interval(10, 5, cl=0.95, alternative="upper", p=1)
    expected4 = (0.0, 0.7775588989918742)
    np.testing.assert_equal(res4, expected4)

    res5 = binom_conf_interval(10, 5, cl=0.95, alternative="lower", p=0)
    expected5 = (0.22244110100812578, 1.0)
    np.testing.assert_equal(res5, expected5)


def test_hypergeom_conf_interval():
    res = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="two-sided")
    expected = (1.0, 4.0)
    np.testing.assert_equal(res, expected)

    res2 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="upper")
    expected2 = (0.0, 4.0)
    np.testing.assert_equal(res2, expected2)

    res3 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="lower")
    expected3 = (1.0, 5.0)
    np.testing.assert_equal(res3, expected3)

    res4 = hypergeom_conf_interval(2, 2, 5, cl=0.95, alternative="two-sided")
    expected4 = (2.0, 5.0)
    np.testing.assert_equal(res4, expected4)

    cl = 0.95
    n = 10
    x = 5
    N = 20
    [lot, hit] = [6, 14]
    alternative = "two-sided"
    [lo, hi] = hypergeom_conf_interval(n, x, N, cl=cl, alternative=alternative)
    np.testing.assert_equal(lo, lot)
    np.testing.assert_equal(hi, hit)

    res5 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="upper", G=5)
    np.testing.assert_equal(res5, expected2)

    res6 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="lower", G=0)
    np.testing.assert_equal(res6, expected3)


def test_hypergeometric():
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'greater'), 
                        1-hypergeom.cdf(3, 10, 5, 6))
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'less'), 
                        hypergeom.cdf(4, 10, 5, 6))
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'two-sided'), 
                        2*(1-hypergeom.cdf(3, 10, 5, 6)))


def test_hypergeometric_badinput1():
    pytest.raises(ValueError, hypergeometric, 5, 10, 2, 6)


def test_hypergeometric_badinput2():
    pytest.raises(ValueError, hypergeometric, 5, 10, 18, 6)


def test_hypergeometric_badinput3():
    pytest.raises(ValueError, hypergeometric, 5, 10, 6, 16)


def test_hypergeometric_badinput4():
    pytest.raises(ValueError, hypergeometric, 5, 10, 6, 2)


def test_binomial_p():
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'greater'), 
                        1-binom.cdf(4, 10, 0.5))
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'less'), 
                        binom.cdf(5, 10, 0.5))
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'two-sided'), 1)


def test_binomial_badinput():
    pytest.raises(ValueError, binomial_p, 10, 5, 0.5)


def test_get_random_state():
    prng1 = np.random.RandomState(42)
    prng2 = get_prng(42)
    prng3 = get_prng(prng1)
    prng4 = get_prng(prng2)
    prng5 = get_prng()
    prng6 = get_prng(None)
    prng7 = get_prng(np.random)
    prng8 = get_prng(SHA256(42))
    assert(isinstance(prng1, np.random.RandomState))
    assert(isinstance(prng2, SHA256))
    assert(isinstance(prng3, np.random.RandomState))
    assert(isinstance(prng4, SHA256))
    assert(isinstance(prng5, SHA256))
    assert(isinstance(prng6, SHA256))
    assert(isinstance(prng7, np.random.RandomState))
    x1 = prng1.randint(0, 5, size=10)
    x2 = prng2.randint(0, 5, size=10)
    x3 = prng3.randint(0, 5, size=10)
    x4 = prng4.randint(0, 5, size=10)
    x5 = prng5.randint(0, 5, size=10)
    x6 = prng6.randint(0, 5, size=10)
    x7 = prng7.randint(0, 5, size=10)
    x8 = prng8.randint(0, 5, size=10)
    np.testing.assert_equal(x2, x8)
    assert prng2.counter == 1
    assert prng2.baseseed == 42
    assert prng2.baseseed == prng4.baseseed
    assert len(x5) == 10
    assert len(x6) == 10
    assert len(x7) == 10


def test_get_random_state_error():
    pytest.raises(ValueError, get_prng, [1, 1.11])


def test_permute_within_group():
    x = np.repeat([1, 2, 3] * 3, 3)
    group = np.repeat([1, 2, 3], 9)
    #response = np.zeros_like(group)
    #response[[0,  1,  3,  9, 10, 11, 18, 19, 20]] = 1
    
    res1 = permute_within_groups(x, group, seed=42)
    res2 = permute_within_groups(x, group, seed=SHA256(42))
    np.testing.assert_equal(res1, res2)

    res3 = permute_within_groups(x, group)
    np.testing.assert_equal(res3.max(), 3)
    res3.sort()
    np.testing.assert_equal(group, res3)


@pytest.mark.skipif(sys.platform == "win32", reason="need updated cryptorandom")
def test_permute():
    prng = SHA256(42)

    x = prng.randint(0, 10, size=20)
    actual = permute(x, prng)
    expected = np.array([6, 9, 5, 1, 3, 1, 4, 7, 6, 9, 
                         8, 7, 2, 1, 9, 7, 8, 1, 8, 1])
    np.testing.assert_array_equal(actual, expected)

    actual = permute(x)
    np.testing.assert_equal(actual.max(), 9)
    np.testing.assert_equal(actual.min(), 1)


def test_permute_rows():
    prng = SHA256(42)

    x = prng.randint(0, 10, size=20).reshape(2, 10)
    actual = permute_rows(x, prng)
    expected = np.array([[9, 1, 8, 6, 9, 1, 5, 4, 3, 6], 
                         [1, 7, 2, 1, 9, 7, 8, 7, 8, 1]])
    np.testing.assert_array_equal(actual, expected)

    a = permute_rows(x)
    np.testing.assert_equal(a.max(), 9)
    np.testing.assert_equal(a.min(), 1)


def test_permute_incidence_fixed_sums():
    prng = np.random.RandomState(42)
    x0 = prng.randint(2, size=80).reshape((8, 10))
    x1 = permute_incidence_fixed_sums(x0)

    K = 5

    m = []
    for i in range(1000):
        x2 = permute_incidence_fixed_sums(x0, k=K)
        m.append(np.sum(x0 != x2))

    np.testing.assert_(max(m) <= K * 4,
                       "Too many swaps occurred")

    for axis in (0, 1):
        for test_arr in (x1, x2):
            np.testing.assert_array_equal(x0.sum(axis=axis),
                                          test_arr.sum(axis=axis))


def test_permute_incidence_fixed_sums_ND_arr():
    pytest.raises(ValueError, permute_incidence_fixed_sums, np.random.random((1, 1, 1)))


def test_permute_incidence_fixed_sums_non_binary():
    pytest.raises(ValueError, permute_incidence_fixed_sums, np.array([[1, 2], [3, 4]]))


def test_potential_outcomes():
    x = np.array(range(5)) + 1
    y = x + 4.5
    f = lambda u: u + 3.5
    finv = lambda u: u - 3.5
    g = lambda u: np.exp(u * 2)
    ginv = lambda u: np.log(u) / 2

    resf = potential_outcomes(x, y, f, finv)
    resg = potential_outcomes(x, y, g, ginv)
    expectedf = np.array([[1.,  -2.5],
                          [2.,  -1.5],
                          [3.,  -0.5],
                          [4.,   0.5],
                          [5.,   1.5],
                          [9.,   5.5],
                          [10.,   6.5],
                          [11.,   7.5],
                          [12.,   8.5],
                          [13.,   9.5]])
    expectedg = np.array([[1.00000000e+00,   0.00000000e+00],
                          [2.00000000e+00,   3.46573590e-01],
                          [3.00000000e+00,   5.49306144e-01],
                          [4.00000000e+00,   6.93147181e-01],
                          [5.00000000e+00,   8.04718956e-01],
                          [5.98741417e+04,   5.50000000e+00],
                          [4.42413392e+05,   6.50000000e+00],
                          [3.26901737e+06,   7.50000000e+00],
                          [2.41549528e+07,   8.50000000e+00],
                          [1.78482301e+08,   9.50000000e+00]])
    np.testing.assert_equal(resf, expectedf)
    np.testing.assert_almost_equal(resg[:5, :], expectedg[:5, :])
    np.testing.assert_almost_equal(resg[5:, :], expectedg[5:, :], 1)


def test_potential_outcomes_bad_inverse():
    f = lambda u: u + 3.5
    ginv = lambda u: np.log(u) / 2
    pytest.raises(AssertionError, potential_outcomes, np.array([1, 2]), np.array([3, 4]), f, ginv)
