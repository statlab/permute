"""
Hypergeometric Test

"""
import operator as op
from functools import reduce 


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

def hypergeom_test(n, N, G, g, a, Ho_sign):

	"""
	Returns the p-value and conclusion of the hypergeometric probability test.

	Parameters
	----------
	n : int
	   sample size
	N : int
	   total population size
	G : int
	   number of good elements in population
	g : int
	   hypothesized number of good elements
	a : float
	   significance level
	Ho_sign : string of equality sign in null hypothesis
	   random variable Z is "==", "<=", "!=", ">", "<", or ">=" g

	Returns
	-------
	float
	   p-value of test
	string
	   conclusion of test

	"""
	complements = {
	"==": "!=", 
	"!=": "==", 
	"<": ">=", 
	">": "<=", 
	">=": "<",
	"<=": ">",
	}

	alt_sign = complements[Ho_sign]

	prob, g_prob, s_prob = 0, 0, 0

	if alt_sign == "==":
		prob = (comb(G, g) * comb((N - G), (n - g))) / comb(N, n)

	elif alt_sign == "!=":
		for i in range(0, g+1):
			s_prob += (comb(G, i) * comb((N - G), (n - i))) / comb(N, n)
		for i in range(g, n+1):
			g_prob += (comb(G, i) * comb((N - G), (n - i))) / comb(N, n)
		prob = min(s_prob, g_prob)

	elif alt_sign == "<" or alt_sign == "<=": 
		for i in range(0, g+1):
			prob += (comb(G, i) * comb((N - G), (n - i))) / comb(N, n)

	elif alt_sign == ">" or alt_sign == ">=": 
		for i in range(g, n+1):
			prob += (comb(G, i) * comb((N - G), (n - i))) / comb(N, n)

	if prob <= a:
		print("Reject the null hypothesis.")
	else:
		print("Fail to reject the null hypothesis.")
	return prob






