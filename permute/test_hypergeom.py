"""
Unit Tests for hypergeom.oy
"""

import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from hypergeom import hypergeom

def less():
	assert_less(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, 'less')[0], 1)

def greater():
	assert_almost_equal(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, 'greater')[0], 0.50)

def twoSided():
	assert_almost_equal(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, "two-sided")[0], 0.8)

def lessWTS():
	assert_less(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, 'less', True)[0], 1)

def greaterWTS():
	assert_almost_equal(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, 'greater', True)[0], 0.50)

def twoSidedWTS():
	assert_almost_equal(hypergeom([0, 1, 0, 1, 0, 1, 0, 1], 5, 2, 10**5, "two-sided", True)[0], 0.85)



