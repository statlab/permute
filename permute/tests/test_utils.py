from nose.tools import raises

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_equal

from ..utils import get_prng


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
