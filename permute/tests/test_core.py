from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.random import RandomState


from ..core import (binom_conf_interval,
                    corr,
                    permute_within_groups,
                    permute_rows,
                    two_sample)


def test_permute_within_group():
    x = np.repeat([1, 2, 3]*3, 3)
    group = np.repeat([1, 2, 3], 9)
    #response = np.zeros_like(group)
    #response[[0,  1,  3,  9, 10, 11, 18, 19, 20]] = 1

    prng1 = RandomState(42)
    prng2 = RandomState(42)
    res1 = permute_within_groups(x, group, prng1)
    res2 = permute_within_groups(x, group, prng2)
    np.testing.assert_equal(res1, res2)

    res3 = permute_within_groups(x, group)
    np.testing.assert_equal(res3.max(), 3)
    res3.sort()
    np.testing.assert_equal(group, res3)


def test_corr():
    prng = RandomState(42)
    x = prng.randint(5, size=10)
    y = x
    res1 = corr(x, y, prng=prng)
    res2 = corr(x, y)
    np.testing.assert_equal(len(res1), 5)
    np.testing.assert_equal(len(res2), 5)
    np.testing.assert_equal(res1[0], res2[0])
    np.testing.assert_equal(res1[1], res2[1])
    #np.testing.assert_equal(res1[2], res2[2])
    #np.testing.assert_equal(res1[3], res2[3])

    y = prng.randint(5, size=10)
    res1 = corr(x, y, prng=prng)
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

    res2 = binom_conf_interval(10, 5, cl=0.95, alternative="greater")
    expected2 = (0.0, 0.7775588989918742)
    np.testing.assert_equal(res2, expected2)

    res3 = binom_conf_interval(10, 5, cl=0.95, alternative="less")
    expected3 = (0.22244110100812578, 1.0)
    np.testing.assert_equal(res3, expected3)


def test_permute_rows():
    prng = RandomState(42)

    x = prng.randint(10, size=20).reshape(2, 10)
    permute_rows(x, prng)
    expected = np.array([[2, 7, 7, 6, 4, 9, 3, 4, 6, 6],
                         [7, 4, 5, 5, 3, 7, 1, 2, 7, 1]])
    np.testing.assert_array_equal(x, expected)

    permute_rows(x)
    np.testing.assert_equal(x.max(), 9)
    np.testing.assert_equal(x.min(), 1)


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

    res = two_sample(x, y, seed=42, interval="two-sided")
    expected_pv = 0.66505000000000003
    expected_ts = -0.13990200413154097
    expected_ci = (0.0, 0.6675064023707297)
    np.testing.assert_equal(res[0], expected_pv)
    np.testing.assert_equal(res[1], expected_ts)
    np.testing.assert_almost_equal(res[2], expected_ci)
