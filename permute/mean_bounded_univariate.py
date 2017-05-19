"""
Mean of Univariate Bounded Distributions

"""

def uniform_mean(a, b):

	"""
	Returns mean of uniform distribution bounded with lower bound a and upper bound b. Function applies
	to both discrete and continuous distributions. 

	Parameters
	------------
	a : float or int
		lower bound
	b : float or int
		upper bound 

	Returns
	------------
	float 
		mean of uniform distribution bounded by a & b

	"""

	return (a + b) / 2

def geometric_z_mean(p):

	"""
	Returns mean of geometric(p) on {0, 1, 2.....}.

	Parameters
	-------------
	p : float
		success probability

	Returns
	-------------
	float
		mean of geometric(p) on {0, 1, 2...}

	"""
	
	return (1 - p) / p

def geometric_mean(p):

	"""
	Returns mean of geometric(p) on {1, 2, ......}. 

	Parameters
	----------
	p: float
		success probability

	Returns
	-------
	float
		mean of geometric(p) on {1, 2, .....}

	"""

	return 1 / p

def exponential_mean(lam):

	"""
	Returns mean of exponential distribution.

	Parameters
	----------
	lam : float
		the rate of an exponential random variable, (lam > 0)

	Returns
	-------
	float
		mean of expential(lam)

	"""

	assert (lam > 0), "Parameter must be greater than zero."

	return 1 / lam

def neg_binomial_mean(r, p):

	"""
	Returns mean of negative binomal (r, p) on {0, 1, 2.....}.

	Parameters
	----------
	p : float
		success probability

	r : float or int
		number of successes

	Returns
	-------
	float
		mean of negative binomial(r, p)

	"""


	return (r * (1 - p)) / p

def gamma_mean(r, lam):
	"""
	Returns mean of gamma(r, lam) distribution.

	Parameters
	----------
	r : float 
		shape, (r > 0)
	lam : float
		rate or inverse scale, (lam > 0)

	Returns
	-------
	float
		mean of gamma(r, lam)

	"""

	assert (r > 0), "Variable r (shape) must be greater than 0."
	assert (lam > 0), "Variable lam (rate) must be greater than 0."

	return r / lam


def beta_mean(r, s):

	"""
	Returns mean of beta(r, s).

	Parameters
	----------
	r : int or float
		shape
	s: int or float
		shape

	Returns
	-------
	float
		mean of beta(r, s)

	"""

	return r / (r + s)



