"""
Unit Tests for sprt.py
"""
import math
import numpy as np 

from ..sprt import (sprt, bernoulli_lh_ratio)

def test_sprt():
    obs1 = sprt(lambda x: bernoulli_lh_ratio(x, .5, 0.9), .10, .01, [1, 1, 1, 1, 1, 1, 1], True)
    assert obs1[0] == [True, False]
    
    obs2 = sprt(lambda x: bernoulli_lh_ratio(x, .1, 0.5), .01, .10, [0, 0, 0, 0, 0, 0, 0], True)
    assert obs2[0] == [False, True]
    
def test_bernoulli_lh_ratio():
    assert bernoulli_lh_ratio([1,1,1,1,1], 1.0, 0.0)==0.0
    
    assert bernoulli_lh_ratio([0,0,0,0,0], 0.0, 1.0)==0.0