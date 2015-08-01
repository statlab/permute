from nose.tools import raises

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal

from ..utils import (get_prng,
                     permute_rows,
                     permute_within_groups,
                     permute_incidence_fixed_sums,
                     hypergeom_conf_interval)


def test_get_random_state():
    prng1 = RandomState(42)
    prng2 = get_prng(42)
    prng3 = get_prng(prng1)
    prng4 = get_prng(prng2)
    prng5 = get_prng()
    prng6 = get_prng(None)
    prng7 = get_prng(np.random)
    assert(isinstance(prng1, RandomState))
    assert(isinstance(prng3, RandomState))
    assert(isinstance(prng5, RandomState))
    assert(isinstance(prng6, RandomState))
    assert(isinstance(prng7, RandomState))
    x1 = prng1.randint(5, size=10)
    x2 = prng2.randint(5, size=10)
    x3 = prng3.randint(5, size=10)
    x4 = prng4.randint(5, size=10)
    x5 = prng5.randint(5, size=10)
    x6 = prng6.randint(5, size=10)
    x7 = prng7.randint(5, size=10)
    assert_equal(x1, x2)
    assert_equal(x3, x4)
    assert_equal(len(x5), 10)
    assert_equal(len(x6), 10)
    assert_equal(len(x7), 10)


def test_hypergeom_conf_interval():
    cl = 0.95
    n = 10
    x = 5
    [lot, hit] = [6, 14]
    alternative = "two-sided"
    [lo, hi] = hypergeom_conf_interval(n, x, N, cl=cl, alternative = alternative)
    np.testing.assert_equal(lo, lot)
    np.testing.assert_equal(hi, hit)

@raises(ValueError)
def test_get_random_state_error():
    get_prng(1.11)


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


def test_permute_incidence_fixed_sums():
    prng = RandomState(42)
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


@raises(ValueError)
def test_permute_incidence_fixed_sums_ND_arr():
    permute_incidence_fixed_sums(np.random.random((1, 1, 1)))


@raises(ValueError)
def test_permute_incidence_fixed_sums_non_binary():
    permute_incidence_fixed_sums(np.array([[1, 2], [3, 4]]))
