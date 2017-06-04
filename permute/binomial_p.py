"""
Binomial Permutation Test

"""
import sys
sys.setrecursionlimit(100000)
import scipy
import numpy as np
from scipy.special import comb


def binomial_p(sample, n, y, Ho_sign):
	"""
	Parameters
	----------
	sample : array-like
	   list of elements consisting of x in {0, 1} where 0 represents a failure and
	   1 represents a seccuess
	y : int
	   hypothesized number of successes in n trials
	n : int
	   number of trials 
	Ho_sign : string
       null hypothesis: random variable Z is "==", "<=", "!=", ">", "<", or ">=" y
	
	Returns
	-------
	float
	   p-value 

	"""
	complements = {
	"==": "!=", 
	"!=": "==", 
	"<": ">=", 
	">": "<=", 
	">=": "<",
	"<=": ">",
	}

	original_ts = sum([x for x in sample if x == 1]) / len(sample)

	

	def generate():

		return np.random.binomial(n, original_ts, 1)


	def permute(p, count):

		if count == 2000:
			return p
		
		return Link(p.first(), permute(p, count+1))

	
	permutations = permute(Link(generate, Link.empty), 0)


	alt_sign = complements[Ho_sign]

	
	def filter(linked, sign, count): 
	
		if linked == Link.empty:
			return count
		if sign == "!=":
			if linked.first != y:
				count += 1
		elif sign == "==":
			if linked.first == y:
				count += 1
		elif sign == ">=":
			if linked.first >= y:
				count += 1
		elif sign == "<=":
			if linked.first <= y:
				count += 1
		elif sign == "<":
			if linked.first < y:
				count += 1
		elif sign == ">=":
			if linked.first >= y:
				count += 1
		return filter(linked.rest, sign, count)


	return filter(permutations, alt_sign, 0) / len(permutations)

	




## Private Helper Function ##

class Link:
    """
    A linked list.

    """
    empty = ()

    def __init__(self, first, rest=empty):
        self.first = first
        self.rest = rest

    def __getitem__(self, i):
        if i == 0:
            return self.first
        else:
            return self.rest[i-1]

    def __len__(self):
        return 1 + len(self.rest)

    def __repr__(self):
        if self.rest:
            rest_str = ', ' + repr(self.rest)
        else:
            rest_str = ''
        return 'Link({0}{1})'.format(repr(self.first), rest_str)


	











