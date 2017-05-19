"""
Mean of Univariate Bounded Distributions

"""

def binomial_mean(n, p):

	"""
	Returns mean of binomial distribution on n trials and success probability p.

	Parameters
	----------
	n : float or int
	   number of trials
	p : float
	   probability of success on each trial (0 <= p <= 1)

	Returns
	-------
	float
	   mean of binomial(n, p)
	"""
	
	assert (p >= 0 and p <= 1), "Success probability p must be between 0 and 1, inclusive."
	
	return n * p


def hypergeometric_mean(n, N, G):

	"""
	Returns mean of hypergeometric distribution with n sample size, N total population size, and G number of good elements in population.
	
	Parameters
	----------
	n : float or int
	   sample size
	N : float or int
	   total population size
	G : float or int
	   number of good elements in population

	Returns
	-------
	float
	   mean of hypergeometric(n, N, G)

	"""

	return (n * G) / N



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
	Returns mean of geometric distribution with success probability p on {0, 1, 2.....}.

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



