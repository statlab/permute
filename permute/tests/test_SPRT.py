"""
Unit Tests for sprt.py
"""
import math
import numpy as np 

from ..sprt import (sprt, bernoulli_lh_ratio)


def test_sprt():
    np.random.seed(5)
    # random_order and do not reject ho
    res = sprt(lambda x: bernoulli_lh_ratio(x, .5, .1), .05, .05, [0, 1]*10, True)
    assert res[0] == [False, True]
    # random_order and do reject ho
    res = sprt(lambda x: bernoulli_lh_ratio(x, .1, .5), .05, .05, [0, 1]*10, True)
    assert res[0] == [True, False]
    # not random_order and do not reject ho
    res = sprt(lambda x: bernoulli_lh_ratio(x, .5, .1), .05, .05, [1, 1], False)
    assert res[0] == [False, True]
    assert res[1] == 0.1**2/0.5**2
    # not random_order and inconclusive test
    res = sprt(lambda x: bernoulli_lh_ratio(x, .5, .1), .05, .05, [0, 0], False)
    assert res[0] == [False, False]
    assert res[1] == 0.9**2/0.5**2
