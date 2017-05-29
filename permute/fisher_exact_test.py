"""
Fisher's Exact Test
"""

import numpy as numpy
import scipy 
from scipy.special import comb 

## User Function ##

def fisher(data, tail):

	"""
	Performs Fisher's Exact Test. 

	Parameters
	----------
	data : list
	   list of lists representing a 2 X 2 data table
	   (for example, 
		   1   2
		   3   4
	   would be represented as [[1, 2], [3, 4]]
	tail : int
	   one or two tailed 
	
	Returns
	-------
	float
	   p-value

	"""

	sum_a = sum(data[0])
	sum_b = sum(data[1])
	tops = []
	for i in range(sum_a + 1):
		tops += [[i, sum_a - i]]


	sum_c = data[0][0] + data[1][0]
	sum_d = data[0][1] + data[1][1]
	new = []
	for top in tops:
		new += [[top, [sum_c - top[0], sum_d - top[1]]]]
	new = sorted(new, data)
	
	ts = []
	for table in new:
		x = test_statistic(table) 
		ts.append(x)
	original = test_statistic(data)

	index = 0
	for i in range(len(ts)):
		if ts[i] == original:
			index = i

	one_p = original
	for f in range(index):
		if ts[f] <= original:
			one_p += ts[f]
	two_p = one_p
	for s in range(index + 1, len(ts)):
		if ts[s] <= original:
			two_p += ts[s]
	if tail ==1:
		return one_p
	return two_p
		



## Private Helper Functions ##

def test_statistic(data):
	
	"""
	Returns test statistic of data.

	Parameters
	----------
	data : list
	   list of lists representing a 2 X 2 data table

	Returns
	-------
	float
	   probability of obtaining the given set of values 	

	"""

	a = data[0][0]
	b = data[0][1]
	c = data[1][0]
	d = data[1][1]
	n = a + b + c + d

	ab = a + b
	cd = c + d
	ac = a + c
	bd = b + d

	return (comb(ac, a) * comb(bd, b)) / comb(n, ab)

def sorted(new, data):
	"""
	Returns a version of new sorted by the smallest marginal frequency.

	Parameters
	----------
	new : list 
	   list of all possible data sets with the same marginal frequencies as those in data
	data : list
	   original 2 X 2 data table 

	Returns
	-------
	list
	   sorted version of new
	"""


	f = {}
	f['a'] = data[0][0]
	f['b'] = data[0][1]
	f['c'] = data[1][0]
	f['d'] = data[1][1]
	min_f = min(f, key=f.get)

	if min_f == 'a':
		new.sort(key=lambda x: x[0][0])

	if min_f == 'b':
		new.sort(key=lambda x: x[0][1])

	if min_f == 'c':
		new_sort(key=lambda x: x[1][0])

	if min_f == 'd':
		new_sort(key=lambda x: x[1][1])

	return new
