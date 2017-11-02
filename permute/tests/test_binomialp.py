"""
Unit tests for binomialp.py
"""
import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr
from scipy.stats import binom

from ..binomialp import binomial_p

def test_binomial_p():
    assert_almost_equal(binomial_p(5, 10, 0.5, 10**5, 'greater')[0], 1-binom.cdf(4, 10, 0.5), 2)
    assert_almost_equal(binomial_p(5, 10, 0.5, 10**5, 'less')[0], binom.cdf(5, 10, 0.5), 2)
    assert_almost_equal(binomial_p(5, 10, 0.5, 10**5, 'two-sided')[0], 1, 2)
    assert_equal(len(binomial_p(5, 10, 0.5, 10, 'greater', keep_dist=True)), 3)
    
    res1 = binomial_p(5, 10, 0.5, 10**2, 'greater', keep_dist=True, seed=12345)
    res2 = binomial_p(5, 10, 0.5, 10**2, 'greater', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])
    
    res1 = binomial_p(5, 10, 0.5, 10**2, 'less', keep_dist=True, seed=12345)
    res2 = binomial_p(5, 10, 0.5, 10**2, 'less', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])
    
    res1 = binomial_p(5, 10, 0.5, 10**2, 'two-sided', keep_dist=True, seed=12345)
    res2 = binomial_p(5, 10, 0.5, 10**2, 'two-sided', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])
    
@raises(ValueError)
def test_binomial_badinput():
    binomial_p(10, 5, 0.5)