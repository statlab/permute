"""
Unit Tests for sign.py
"""

import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..sign import sign

def equalTEST_GREATER():
	x = [0, 1, 0, 1]
	y = [1, 0, 1, 0]
	assert_almost_equal(sign(x, y, 10**5, "greater", False)[0], 0.5)
	assert_euqal(sign(x, y, 10**5, "greater", False)[1], 0.5)

def equalTEST_LESS():
	x = [0, 1, 0, 1]
	y = [1, 0, 1, 0]
	assert_almost_equal(sign(x, y, 10**5, "less", False)[0], 0.5)
	assert_euqla(sign(x, y, 10**5, "less", False)[1], 0.5)

def equalTEST_TWOSIDED():
	x = [0, 1, 0, 1]
	y = [1, 0, 1, 0]
	assert_almost_equal(sign(x, y, 10**5, "two-sided", False)[0], 0.99)
	assert_euqla(sign(x, y, 10**5, "two-sided", False)[1], 0.5)


def test_keepDist():
	x = [0, 1, 0, 1]
	y = [1, 0, 1, 0]
	assert_equal(len(sign(x, y, 10**5, "greater", True)[2]), 10**5)
	assert_equal(len(sign(x, y, 10**5, "less", True)[2]), 10**5)
	assert_equal(len(sign(x, y, 10**5, "two-sided", True)[2]), 10**5)

def oneSUCCESS_oneFAIL():
	x = [1, 0, 0, 0]
	y = [0, 1, 1, 1]
	assert_almost_equal(sign(x, y, 10**5, "greater", False)[0], 0.25)
	assert_almost_equal(sign(x, y, 10**5, "greater", False)[1], 0.25)
	assert_almost_equal(sign(x, y, 10**5, "less", False)[0], 0.75)
	assert_almost_equal(sign(x, y, 10**5, "less", False)[1], 0.25)
	assert_almost_equal(sign(x, y, 10**5, "two-sided", False)[0], 0.5)
	assert_almost_equal(sign(x, y, 10**5, "two-sided", False)[1], 0.25)

def test_keepDist2():
	x = [1, 0, 0, 0]
	y = [0, 1, 1, 1]
	assert_equal(len(sign(x, y, 10**5, "greater", True)[2]), 10**5)
	assert_equal(len(sign(x, y, 10**5, "less", True)[2]), 10**5)
	assert_equal(len(sign(x, y, 10**5, "two-sided", True)[2]), 10**5)









