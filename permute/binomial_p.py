"""
Binomial Probability Test

"""
import scipy
from scipy.special import comb

def binomial_test(n, p, y, t, a, Ho_sign):

	"""
	Returns the p-value and conclusion of the binomial probability test.

	Parameters
	----------
	n : int
	   number of trials
	p : float
	   success probability
	y : int
	   hypothesized number of successes in n trials
	t : int
	   for one-tailed test, input 1
	   for two-tailed test, input 2
	a : float
	   significance level
	Ho_sign : string of equality sign in null hypothesis
	   random variable Z is "==", "<=", "!=", ">", "<", or ">=" y

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

	if n == 1:
		if p <= a:
			print("Reject the null hypothesis.")
		else:
			print("Fail to reject the null hypothesis.")
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

	if prob <= a:
		print("Reject the null hypothesis.")
	else:
		print("Fail to reject the null hypothesis.")
	return "p-value: {}".format(prob)










