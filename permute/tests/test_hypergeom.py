"""
Unit Tests for hypergeom.oy
"""

import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr
from scipy.stats import hypergeom

from ..hypergeom import hypergeometric


def test_hypergeometric():
    assert_almost_equal(hypergeometric(4, 10, 5, 6, 10**5, 'greater', False, 12345)[0], 1-hypergeom.cdf(3, 10, 5, 6), 2)
    assert_almost_equal(hypergeometric(4, 10, 5, 6, 10**5, 'less', False, 12345)[0], hypergeom.cdf(4, 10, 5, 6), 2)
    assert_almost_equal(hypergeometric(4, 10, 5, 6, 10**5, 'two-sided', False, 12345)[0], 2*(1-hypergeom.cdf(3, 10, 5, 6)), 2)
    
    res1 = hypergeometric(4, 10, 5, 6, 10**2, 'greater', keep_dist=True, seed=12345)
    res2 = hypergeometric(4, 10, 5, 6, 10**2, 'greater', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])

    res1 = hypergeometric(4, 10, 5, 6, 10**2, 'less', keep_dist=True, seed=12345)
    res2 = hypergeometric(4, 10, 5, 6, 10**2, 'less', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])
    
    res1 = hypergeometric(4, 10, 5, 6, 10**2, 'two-sided', keep_dist=True, seed=12345)
    res2 = hypergeometric(4, 10, 5, 6, 10**2, 'two-sided', keep_dist=False, seed=12345)
    assert_equal(res1[0], res2[0])
    
@raises(ValueError)
def test_hypergeometric_badinput1():
    hypergeometric(5, 10, 2, 6)

@raises(ValueError)
def test_hypergeometric_badinput2():
    hypergeometric(5, 10, 18, 6)

@raises(ValueError)
def test_hypergeometric_badinput3():
    hypergeometric(5, 10, 6, 16)

@raises(ValueError)
def test_hypergeometric_badinput4():
    hypergeometric(5, 10, 6, 2)