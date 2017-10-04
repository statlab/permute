"""
Sequential Probability Ratio Tests
"""
import math
import numpy as np
import scipy
from scipy.special import comb


############# User Function #################

def seq_prob_ratio(ho, ha, n, a, b, distribution, with_replacement=True):
	"""
	Performs sequential probability ratio test of desired distribution type. 
	Parameters
	----------
	ho : float or int
       null hypothesis
    ha : float or int
       alternate hypothesis
    a : float
       Type I Error
    b : float
       Type II Error
    n : list
       list of population values 
    with_replacement : boolean (default : True)
       True: sample from n with replacement 
       False: sample from n without replacement 
	distribution : string
		type of distribution : {"normal", "bernoulli", "hypergeometric"}
	
	Returns
	-------
	String
	   Conclusion of test
	Other value types (different per type of test)
	"""
	if distribution == "normal":
		return normal_seq_ratio(ho, ha, a, b, n, with_replacement) 
	elif distribution == "bernoulli":
		return bernoulli_seq_ratio(ho, ha, n, a, b, with_replacement)
	elif distribution == "hypergeometric":
		return hypergeom_seq_ratio(ho, ha, a, b, n, with_replacement)
	else:
		raise ValueError 


####### Private Helper Functions #######

def bernoulli_lh(ho, ha, s, n):
	"""
	Returns the likelihood ratio for independently distributed bernoulli random variables.
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
	   likelihood ratio of model
	"""

	null_lh = (ho ** s) * (1 - ho)**(n - s)
	alt_lh = (ha ** s) * (1 - ha)**(n - s)

	return alt_lh / null_lh


def bernoulli_seq_ratio(ho, ha, n, a, b, with_replacement=True):
	"""
	Performs seqential probability ratio test on independently distributed bernoulli random variables. 
	Parameters
	----------
	ho : float
	   null hypothesis
	ha : float
	   alternative hypothesis
	n : list of {0, 1}
	   list of population values 
	a : float
	   Type I Error
	b : float
	   Type II Error
	Returns
	-------
	String
	   Conclusion of test
	List
	   [proportion of successes when test haults, sample size when test haults, string of conclusion]
	"""
		
	A, B = b / (1 - a), (1 - b) / a

	assert (A < B)

	def test(ho, ha, n, s, A, B, ss, ts):
		"""
		Samples from n with replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		n : list
		   list of population values 
		s : int or float
		   current number of successes in sample
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size 
		ts : float or int
		   current likelihood 
		"""
		successes = s

		index = np.random.choice(len(n), 1, replace=True)[0]

		trial = n[index]

		ss += ss + 1
		
		if trial == 0:
			s += 1
		
		ts *= bernoulli_lh(ho, ha, s, ss)

		if ts >= B:
			return [s / ss, ss, "Accept null hypothesis."]
		elif ts <= A:
			return [s / ss, ss, "Reject null hypothesis."]
		else:
			return test(ho, ha, n, s, A, B, ss, ts)

	def test_wo(ho, ha, n, s, A, B, ss, ts):
		"""
		Samples from n without replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		n : list
		   list of population values 
		s : int or float
		   current number of successes in sample
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size 
		ts : float or int
		   current likelihood 
		"""

		index = np.random.choice(len(n), 1, replace=True)[0]

		trial = n[index]

		n.pop(index)

		ss += ss + 1
		
		if trial == 0:
			s += 1
		
		ts *= bernoulli_lh(ho, ha, s, ss)

		if ts >= B:
			return [s / ss, ss, "Accept null hypothesis."]
		elif ts <= A:
			return [s / ss, ss, "Reject null hypothesis."]
		else:
			if ss == len(n):
				return [None, "No Decision."]
			return test(ho, ha, n, s, A, B, ss, ts)
	
	if with_replacement == True:	
		return test(ho, ha, n, 0, A, B, 0, 1)

	return test_wo(ho, ha, n, 0, A, B, 0, 1)

def normal_lh(ho, ha, x):

	"""
	Returns likelihood ratio for indepedently distributed normal variables.
	Parameters
	----------
	ho : float
	   null hypothesis 
	ha : float
	   alternative hypothesis  
	x : float or int
	   value of most recent sample
	Returns
	-------
	float
	   likelihood of model
	"""
	null_lh = math.exp(-((x - ho)**2))
	alt_lh = math.exp(-((x - ha)**2))

	return alt_lh / null_lh

def normal_seq_ratio(ho, ha, a, b, n, with_replacement=False):

	"""
	Performs sequential probability ratio test on independently distributed normal random variables Xi. 
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
	with_replacement : boolean
	   False: sample from n without replacement
	   True: sample from n with replacement 
	Returns
	-------
	string
	   conclusion of test
	list
	   [sample size when test haults, mean when test haults, string of conclusion]
	"""
	A, B = b / (1 - a), (1 - b) / a

	assert (A < B)

	def test(ho, ha, s, u, A, B, ss, ts, n):
		
		"""
		Samples from n with replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		s : int or float
		   cummulative sum of trial results
		u : float or int
		   current mean
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size
		ts : float or int
		   cumulative product of likelihood ratios
		n : list
		   list of all possible values of Xi where i = {1, 2....n}
		"""
		
		index = np.random.choice(len(n), 1, replace=True)[0]
		
		trial = n[index]

		s += trial

		ss += 1

		u = s / ss

		ts *= normal_lh(ho, ha, trial)

		if ts >= B:
			return [ss, u, "Accept null hypothesis."]
		elif ts <= A:
			return [ss, u, "Reject null hypothesis."]
		else:
			return test(ho, ha, s, u, A, B, ss, ts, n)

	def test_wo(ho, ha, s, u, A, B, ss, ts, n):
		
		"""
		Samples from n without replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		s : int or float
		   cummulative sum of trial results
		u : float or int
		   current mean
		A : float
		   lower bound parameter
		B : float
		   upper bound parameter
		ss : int
		   current sample size
		ts : float or int
		   cumulative product of likelihood ratios
		n : list
		   list of all possible values of Xi where i = {1, 2....n}
		"""
		
		index = np.random.choice(len(n), 1, replace=True)[0]
		
		trial = n[index]

		n.pop(index) 

		s += trial

		ss += 1

		u = s / ss

		ts *= normal_lh(ho, ha, trial)

		if ts >= B:
			return [ss, u, "Accept null hypothesis."]
		elif ts <= A:
			return [ss, u, "Reject null hypothesis."]
		else:
			if ss == len(n):
				return [None, "No Decision"]
			return test_wo(ho, ha, s, u, A, B, ss, ts, n)

	if with_replacement == True:
		
		return test(ho, ha, 0, 0, A, B, 0, 1, n)
	
	return test_wo(ho, ha, 0, 0, A, B, 0, 1, n)

	


def hypergeom_lh(ho, ha, trial, n, g, N):
	"""
	Returns likelihood ratio for independently distributed hypergeometric random variables. 
	
	Parameters
	----------
	ho : float
	   null hypothesis
	ha : float
	   alternative hypothesis
	trial : float
	   number of good elements in recent sample 
	n : float or int
	   sample size
	g : float or int
	   number of good elements in sample 
	N : float or int
	   total population size 
	Returns
	-------
	float
	   likelihood ratio of model
	
	"""
	ho_G, ha_G = ho * (N / n), ha * (N / n)

	null_lh = (comb(ho_G, g) * comb(N - ho_G, n - g)) 
	alt_lh = (comb(ha_G, g) * comb(N - ha_G, n - g))

	return alt_lh / null_lh



def hypergeom_seq_ratio(ho, ha, a, b, N, with_replacement=True):

	"""
	Performs sequential probability ratio test on independently distributed hypergeometric random variables.
	
	Parameters
	----------
	ho : float
	   null hypothesis
	ha : float
	   alternative hypothesis
	a : float
	   Type I Error
	b : float 
	   Type II Error 
	N : list
	   list of elements in population 
	with_replacement : boolean
	   True: sample from N with replacement
	   False: sample from N without replacement 
	Returns
	-------
	float
	   mean when test haults
	string
	   conclusion of test 
	"""

	A, B = b / (1 - a), (1 - b) / a

	assert (A < B)

	def test(ho, ha, A, B, g, n, N, ts):
		"""
		Samples from N with replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		A : float
		   lower parameter
		B : float
		   upper parameter
		g : float or int
		   number of good elements in sample
		n : float or int
		   current sample size
		N : list of {0, 1}
		   list of elements in total population
		ts : float 
		   cumulative product of likelihood ratios
		"""

		index = np.random.choice(len(N), 1)[0]

		trial = N[index]

		n += 1

		if trial == 0:
			g += 1

		ts *= hypergeom_lh(ho, ha, trial, n, g, len(N))

		if ts >= B:
			return [n * (g/n), "Accept null hypothesis."]
		elif ts <= A:
			return [n * (g/n), "Reject null hypothesis."]
		else:
			return test(ho, ha, A, B, g, n, N, ts)

	def test_wo(ho, ha, A, B, g, n, N, ts):
		"""
		Samples from N without replacement.
		Parameters
		----------
		ho : float
		   null hypothesis
		ha : float
		   alternative hypothesis
		g : float or int
		   number of good elements in sample
		n : float or int
		   current sample size
		N : list of {0, 1}
		   list of elements in total population
		ts : float 
		   cumulative product of likelihood ratios
		"""

		index = np.random.choice(len(N), 1)[0]

		trial = N[index]

		N.pop(index)

		n += 1

		if trial == 0:
			g += 1

		ts *= hypergeom_lh(ho, ha, trial, n, g, len(N))

		if ts >= B:
			return [n * (g/n), "Accept null hypothesis."]
		elif ts <= A:
			return [n * (g/n), "Reject null hypoethesis."]
		else:
			if n == len(N):
				return [None, 'No Decision']
			return test_wo(ho, ha, A, B, g, n, N, ts)

	if with_replacement == True:
		return test(ho, ha, A, B, 0, 0, N, 1)
	
	return test_wo(ho, ha, A, B, 0, 0, N, 1)






