"""
Unit Tests for signed_rank.py
"""

import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..signed_rank import signed_rank

def emptyNegatives():
	x = [1, 0, 1]
	y = [0, 1, 1]
	assert_equal(signed_rank(x, y), 0)
	assert_equal(signed_rank(x, y, "two"), 0)

def largeSAMP():
	x = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
	y = [x for x in range(38)]
	assert_equal(signed_rank(x, y), 0)
	assert_equal(len(signed_rank(x, y, 10**5, "one", True)[1]), 1)
	assert_equal(signed_rank(x, y, 10**5, "two"), 0)
	assert_equal(len(signed_rank(x, y, 10**5, "two", True)[1]), 1)

def randLARGE():
	x = [y for y in range(5, 70)]
	y = [x for x in range(4, 69)]
	assert_almost_equal(signed_rank(x, y), 0.00048)
	assert_almost_equal(len(signed_rank(x, y, 10**5, "one", True)[1]), 1)
	assert_almost_equal(signed_rank(x, y, 10**5, "two"), 0.00096)
	assert_almost_equal(len(signed_rank(x, y, 10**5, "two", True)[1]), 1)




