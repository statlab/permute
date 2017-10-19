"""
Unit Tests for wilcoxon_sum.py
"""
import math
import scipy.stats as stats

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..wilcoxon_sum import wilcoxon_sum


def oneTail():
	assert_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3]), 0.2)

def twoTail():
	assert_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3], "two"), 0.2)

def oneTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3]), 0.1666)

def twoTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3], "two"), 0.166)

def oneTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4]), 0.166)

def twoTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4], "two"), 0.26666)

def twoTailTestIgnoreReps():
	assert_equal(wilcoxon_sum([1, 3], [6, 4], 10**5, "two", True)[1], [3])

def largeOneSmallTWO():
	assert_equal(wilcoxon_sum([x for x in range(0, 10**3)], [1, 2, 3]), 1)

def smallOneLargeTWO():
	assert_almost_equal(wilcoxon_sum([x for x in range(0, 10)], [x for x in range(0, 10**3)]), 0.0003238)

def approxNORMAL():
	a = [x for x in range(0, 10**3)]
	b = [y for y in range(2, 2 + 10**3)]
	assert_almost_equal(wilcoxon_sum(a, b), 0.437)
	meanNORM = (len(a)*(len(a) + len(b) + 1))/2
	sdNORM = ((len(a) * len(b)*(len(a) + len(b) + 1))/12)**0.5
	observed_ts = sum(wilcoxon_sum(a, b, 10**5, "one", True)[1])/10**5
	zSCORE = (mean - ts) / sd
	assert_almost_equal(abs(zSCORE), 0)
	observed_ts = sum(wilcoxon_sum(a, b, 10**5, "two", True)[1])/10**5
	zSCORE_TWO = (mean - observed_ts) / sd
	assert_almost_equal(abs(zSCORE_TWO), 0)






