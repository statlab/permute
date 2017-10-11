"""
Unit Tests for wilcoxon_sum.py
"""
import math

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..wilcoxon_sum import wilcoxon_sum


def oneTail():
	assert_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3]), 0.2)

def twoTail():
	assert_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3], "two"), 0.4)

def oneTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3]), 0.333)

def twoTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3], "two"), 0.666)

def oneTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4]), 0.166)

def twoTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4], "two"), 0.333)

