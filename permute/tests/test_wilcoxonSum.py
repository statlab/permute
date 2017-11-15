"""
Unit Tests for wilcoxon_sum.py
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
import scipy.stats as stats

from nose.tools import assert_equal, assert_almost_equal, assert_less, raises
from nose.plugins.attrib import attr

from ..wilcoxon_sum import wilcoxon_sum




def testoneTail():
	assert_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3]), 0.2)

def testtwoTail():
	assert_almost_equal(wilcoxon_sum([1.3, 3.4], [4.9, 10.3, 3.3], 10**5, "two"), 0.4)

def testoneTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3]), 0.3333333)

def testtwoTailTie():
	assert_almost_equal(wilcoxon_sum([1, 2], [2, 3], 10**5, "two"), 0.66666666666)

def testoneTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4]), 0.1666666666)

def testtwoTailTestSort():
	assert_almost_equal(wilcoxon_sum([1, 3], [6, 4], 10**5, "two"), 0.3333333)

def testtwoTailTestIgnoreReps():
	assert_equal(wilcoxon_sum([1, 3], [6, 4], 10**5, "two", True)[1], [3, 4, 5, 5, 6, 7])

def testlargeOneSmallTWO():
	assert_equal(wilcoxon_sum([x for x in range(0, 10**3)], [1, 2, 3]), 1)

def testsmallOneLargeTWO():
	assert_almost_equal(round(wilcoxon_sum([x for x in range(0, 10)], [x for x in range(0, 10**3)]), 3), 0.005)

def testkeep_dist():
	a = [x for x in range(0, 10**3)]
	b = [y for y in range(5, 5 + 10**3)]
	pval, lst = wilcoxon_sum(a, b, 10**5, "two", True)
	assert_equal(len(lst), 10**5)
	assert_almost_equal(round(pval, 1), 0.7)

def testapproxNORMAL():
	a = [x for x in range(0, 10**3)]
	b = [y for y in range(2, 2 + 10**3)]
	observed_ts = wilcoxon_sum(a, b)
	meanNORM = (len(a)*(len(a) + len(b) + 1))/2
	sdNORM = ((len(a) * len(b)*(len(a) + len(b) + 1))/12)**0.5
	
	#calculates test statistic we should compare the distribution of permutations to
	#excerpt from wilcoxon_sum.py
	mn = sorted(a + b)
	values = set(mn)
	ranks = [x for x in range(1, len(mn) + 1)]
	duplicates = {}
	if len(values) < len(mn):
		#handles ties in ranks
		for i in values:
			dups = [y for y in mn if y == i]
			counter = len(dups)
			if counter > 1 and i not in duplicates.keys():
				duplicates[i] = counter
		for d in duplicates.keys():
			ind = mn.index(d)
			avg = sum([x for x in range(ind + 1, ind + 1 + duplicates[d])]) / duplicates[d]
			for i in range(ind, ind + duplicates[d]):
				ranks[i] = avg
	rankM = 0
	for i in a:
		rankM += ranks[mn.index(i)]


	ts_for_norm = rankM
	x = (ts_for_norm - meanNORM) / sdNORM
	assert_almost_equal(round(stats.norm.cdf(x), 1), round(observed_ts, 1))






	