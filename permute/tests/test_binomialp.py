"""
Unit tests for binomialp.py
"""
import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..binomialp import binomialp

def less():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'greater')[0], 0.14)

def greater():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'less')[0], 0.23)

def twoSided():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'two-sided')[0], 0.37)

def lessWKD():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'greater')[0], 0.14)
	assert_equal(len(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'greater', True)), 4)

def greaterWKD():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'less')[0], 0.23)
	assert_equal(len(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'less', True)), 4)

def twoSidedWKD():
	assert_almost_equal(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'two-sided', True)[0], 0.37)
	assert_equal(len(binomial_p([0, 1, 0, 1], 10, 5, 10**5, 'two-sided', True)), 4)
