"""
Unit Tests for seq_prob_ratio.py
"""
import math


from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr



from ..sprt import seq_prob_ratio_test
from ..sprt import bernoulli_lh
from ..sprt import bernoulli_seq_ratio_test
from ..sprt import normal_lh
from ..sprt import normal_seq_ratio_test
from ..sprt import hypergeom_lh
from ..sprt import hypergeom_seq_ratio_test

import math
import numpy as np 
import scipy
from scipy.special import comb



def test_normalTRUE():
	assert_equal(seq_prob_ratio_test(0.5, 0.1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3], 0.05, 0.6, "normal", True)[0], 1)
	
def test_normalFALSE():
	assert_equal(seq_prob_ratio_test(0.5, 0.1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3], 0.05, 0.6, "normal", True)[0], 1)

def test_bernoulliTRUE():
	assert_equal(math.trunc(seq_prob_ratio_test(0.5, 0.1, [0, 1], 0.05, 0.05, "bernoulli", True)[0]), 0)

def test_bernoulliFALSE():
	assert_equal(seq_prob_ratio_test(0.5, 0.1, [0, 1], 0.05, 0.05, "bernoulli", False), None)






