from nose.tools import raises

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal

from ..utils import (get_prng,
                     permute_rows,
                     permute_within_groups)


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
