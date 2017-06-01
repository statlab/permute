"""
Hypergeometric Test

"""
import scipy
from scipy.special import comb

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
	string
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
		prob = (comb(G, g) * comb(N - G, n - g)) / comb(N, n)

	elif alt_sign == "!=":
		for i in range(0, g+1):
			s_prob += (comb(G, i) * comb(N - G, n - i)) / comb(N, n)
		for i in range(g, n+1):
			g_prob += (comb(G, i) * comb(N - G, n - i)) / comb(N, n)
		prob = min(s_prob, g_prob)

	elif alt_sign == "<" or alt_sign == "<=": 
		for i in range(0, g+1):
			prob += (comb(G, i) * comb(N - G, n - i)) / comb(N, n)

	elif alt_sign == ">" or alt_sign == ">=": 
		for i in range(g, n+1):
			prob += (comb(G, i) * comb(N - G, n - i)) / comb(N, n)

	if prob <= a:
		print("Reject the null hypothesis.")
	else:
		print("Fail to reject the null hypothesis.")
	return "p-value: {}".format(prob)






