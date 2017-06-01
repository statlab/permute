"""
Binomial Permutation Test

"""
import scipy
import numpy as np
from scipy.special import comb


## User's Function ##

def binomial_p(population, successes, n, y, Ho_sign):
	"""
	Performs binomoial permutation test.

	Parameters
	----------
	population : list
	   list of all elements in population represented by {0, 1}
	successes : int
	   number of successes in population
	y : int
	   hypothesized number of successes in n trials
	n : int
	   number of trials
	Ho_sign : string of equality sign in null hypothesis
	   random variable Z is "==", "<=", "!=", ">", "<", or ">=" y

	
	Returns
	-------
	float
	   p-value binomial probability test
	float
	   p-value of permutation test 

	"""
	p = successes /len(population)
	original_statistic = binomial_test(n, p, y, Ho_sign)

	length = len(population)
	permutations = int(comb(length, n))

	values = []
	def permute():
		nonlocal values
		new_sample = []
		for i in range(n):
			x = np.random.choice(population)
			new_sample.append(x)
		values.append(len([x for x in new_sample if x == 0])) 


	for p in range(permutations):
		permute()


	count = 0
	for val in values:
		if Ho_sign == "==":
			if val == y:
				count += 1
		elif Ho_sign == "<=":
			if val <= y:
				count += 1
		elif Ho_sign == "!=":
			if val != y:
				count += 1
		elif Ho_sign == ">":
			if val > y:
				count += 1
		elif Ho_sign == "<":
			if val < y:
				count += 1
		elif Ho_sign == ">=":
			if val >= y:
				count += 1
	permute_p = count / permutations
	return (original_statistic, permute_p)




## Private Helper Function ##

def binomial_test(n, p, y, Ho_sign, t=2):

	"""
	Returns the p-value of the binomial probability test.

	Parameters
	----------
	n : int
	   number of trials
	p : float
	   success probability
	y : int
	   hypothesized number of successes in n trials
	t : int (default 2)
	   for one-tailed test, input 1
	   for two-tailed test, input 2
	Ho_sign : string of equality sign in null hypothesis
	   random variable Z is "==", "<=", "!=", ">", "<", or ">=" y

	Returns
	-------
	float
	   p-value of test

	"""

	complements = {
	"==": "!=", 
	"!=": "==", 
	"<": ">=", 
	">": "<=", 
	">=": "<",
	"<=": ">",
	}

	if n == 1:
		return p 

	alt_sign = complements[Ho_sign]

	prob, g_prob, s_prob = 0, 0, 0

	if alt_sign == "==":
		prob = comb(n, y) * (p ** y) * (1 - p)**(n - y)

	elif alt_sign == "!=":
		for i in range(1, y+1):
			s_prob += comb(n, i) * (p ** i) * (1 - p)**(n - i)
		for i in range(y, n+1):
			g_prob += comb(n, i) * (p ** i) * (1 - p)**(n - i)
		prob = min(s_prob, g_prob)

	elif alt_sign == "<" or alt_sign == "<=": 
		for i in range(1, y+1):
			prob += comb(n, i) * (p ** i) * (1 - p)**(n - i)

	elif alt_sign == ">" or alt_sign == ">=": 
		for i in range(y, n+1):
			prob += comb(n, i) * (p ** i) * (1 - p)**(n - i)

	if t == 2:
		prob = 2 * prob
	return prob

	











