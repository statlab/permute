"""
Sequential Probability Ratio Tests

"""
import math
from decimal import Decimal

import numpy as np

import operator as op
from functools import reduce 


def binomial_llh(ho, ha, s, n):
	"""
	Returns the log-likelihood ratio for independently distributed binomial variables.

	Parameters
	----------
	ho : float
	   null hypothesis
	ha : float
	   alternative hypothesis
	s : float or int
	   number of successes in sample
	n : float or int
	   total number of elements in sample

	Returns
	-------
	float
	   log-likelihood of model
	"""

	null_llh = comb(n, s) * (ho ** s) * (1 - ho)**(n - s)
	alt_llh = comb(n, s) * (ha ** s) * (1 - ha)**(n - s)

	return null_llh / alt_llh


def binomial_seq_ratio_test(ho, ha, a, b):
	"""
	Performs seqential probability ratio test on binomial random variables. 

	Parameters
	----------
	ho : float
	   null hypothesis
	ha : float
	   alternative hypothesis
	a : float
	   Type I Error
	b  float
	   Type II Error

	Returns
	-------
	String
	   Conclusion of test
	List
	   [proportion of successes when test haults, sample size when test haults]
	"""
		
	A, B = math.log(b / (1 - a)), math.log((1 - b) / a)

	assert (A < B)

	def test(ho, ha, s, A, B, ss, ts):
		"""
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		s : int or float
		   current number of successes in sample
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size 
		ts : float or int
		   cumulative product of log-likelihood 
		"""
		successes = s

		trial = np.random.randint(2)

		sample_size = ss + 1
		
		if trial == 0:
			successes += 1
		
		ts *= binomial_llh(ho, ha, successes, sample_size)

		d = Decimal(ts)

		if d.ln() > B:
			print("Accept null hypothesis.")
			return [successes / sample_size, sample_size]
		elif d.ln() < A:
			print("Reject null hypothesis.")
			return [successes / sample_size, sample_size]
		else:
			test(ho, ha, successes, A, B, sample_size, ts)
	
	return test(ho, ha, 0, A, B, 0, 1)

def normal_llh(ho, ha, o2, x):

	"""
	Returns log-likelihood ratio for indepedently distributed normal variables.

	Parameters
	----------
	ho : float
	   null hypothesis 
	ha : float
	   alternative hypothesis 
	o2 : float
	   variance 
	x : float or int
	   value of most recent sample

	Returns
	-------
	float
	   log-likelihood of model
	"""
	null_llh = (1 / math.sqrt(2 * math.pi * o2)) * math.exp(-((x - ho)**2) / (2 * o2))
	alt_llh = (1 / math.sqrt(2 * math.pi * o2)) * math.exp(-((x - ha)**2) / (2 * o2))

	return math.log(null_llh / alt_llh)

def normal_seq_ratio_test(ho, ha, a, b, n):

	"""
	Performs sequential probability ratio test on normal random variables Xi. 

	Parameters
	----------
	ho : float or int
	   null hypothesis
	ha : float or int
	   alternative hypothesis
	a : float
	   Type I Error
	b : float
	   Type II Error
	n : list
	   list of all possible values of Xi for i = {1, 2, .....n}

	Returns
	-------
	string
	   conclusion of test
	list
	   [sample size when test haults, mean when test haults, variance when test haults]
	"""
	A, B = math.log(b / (1 - a)), math.log((1 - b) / a)

	assert (A < B)

	def first(ho, ha, s, u, o2, A, B, ss, ts, n):
		"""
		Handles exception for first sampling since a sample size of one would result in a variance of 0 by skipping
		the step of calculating the likelihood statistic. 

		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		s : list
		   list of trial results
		u : float or int
		   current mean
		o2 : float or int
		   current variance 
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size
		ts : float or int
		   cumulative product of log-likelihood 
		n : list
		   list of all possible values of Xi for i = {1, 2, .....n}
		"""

		index = np.random.choice(len(n), 1, replace=True)[0]
		trial = n[index]
		s.append(trial)
		ss += 1
		u = sum(s) / ss

		return test(ho, ha, s, u, 0, A, B, ss, 1, n)

	def test(ho, ha, s, u, o2, A, B, ss, ts, n):
		
		"""
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		s : list
		   list of trial results
		u : float or int
		   current mean
		o2 : float or int
		   current variance 
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size
		ts : float or int
		   cumulative product of log-likelihood 
		n : list
		   list of all possible values of Xi where i = {1, 2....n}
		"""
		
		index = np.random.choice(len(n), 1, replace=True)[0]
		
		trial = n[index]

		s.append(trial)

		ss += 1

		u = sum(s) / ss

		o2 = sum([(f - u)**2 for f in s]) * (1 / (ss - 1))

		ts *= normal_llh(ho, ha, o2, trial)

		d = Decimal(ts)

		if d.ln() > B:
			print("Accept null hypothesis.")
			return [ss, u, o2]
		elif d.ln() < A:
			print("Reject null hypothesis.")
			return [ss, u, o2]
		else:
			test(ho, ha, s, u, o2, A, B, ss, ts, n)

	return first(ho, ha, [], 0, 0, A, B, 0, 1, n)








def comb(n, r):

	"""
	Computes combinatorial.

	Parameters
	----------
	n : int
	   number of objects selected 
	r : int
	   total number of objects

	Returns
	-------
	int 
	   number of ways to select r objects from n objects in which order doesn't matter

	Source
	------
	http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python

	"""
	r = min(r, n-r)
	if r == 0: 
		return 1
	num = reduce(op.mul, range(n, n-r, -1)) 
	den = reduce(op.mul, range(1, r+1))
	return num // den












