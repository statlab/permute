"""
Sequential Probability Ratio Tests
"""
import numpy as np

def sprt(likelihood_ratio, alpha, beta, x, random_order = True):
	"""
	Performs sequential probability ratio test with desired likelihood ratio. 
	Parameters
	----------
    likelihood_ratio : function
       likelihood ratio function with one parameter, x, the sample values
    alpha : float
       Type I Error
    beta : float
       Type II Error
    x : list
       list of sample values 
    random_order : boolean (default : True)
       True: sample values in random order 
       False: sample values not in random order

	Returns
	-------
	Array
		ordered pair of booleans (reject ho, reject ha)
	Float
		Likelihood ratio
	"""

	# calculate stopping rule values
	A, B = beta / (1 - alpha), (1 - beta) / alpha

	ts = 1
	index = 0
	if random_order:
		while (ts > A and ts < B and index < len(x)):
			ts = likelihood_ratio(x[0:index])
			index += 1
	else:
		ts = likelihood_ratio(x)

	# get conclusion of test
	if ts >= B:
		conclusion = [True, False]
	elif ts <= A:
		conclusion = [False, True]
	else:
		conclusion = [False, False]

	return [conclusion, ts]


def bernoulli_lh_ratio(x, po, pa):
	"""
	Returns the likelihood ratio for independently distributed bernoulli random variables.
	Parameters
	----------
	x : float or int
	   sample
	po : float
	   probability of success under null hypothesis
	pa : float
	   probability of success under alternative hypothesis
       
	Returns
	-------
	float
	   likelihood ratio 
	"""
	return (pa ** np.sum(x)) * (1 - pa)**(len(x) - np.sum(x)) / ((po ** np.sum(x)) * (1 - po)**(len(x) - np.sum(x)))