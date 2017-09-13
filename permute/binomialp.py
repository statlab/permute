"""
Binomial Permutation Test
"""
import scipy
import numpy as np
from scipy.special import comb
from utils import get_prng


def binomial_p(sample, n, y, reps=10**5, alternative='greater', keep_dist=False, seed=None):
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
	reps : int
	   number of repetitions (default: 10**5)
	alternative : {'greater', 'less', 'two-sided'}
	   alternative hypothesis to test (default: 'greater')
	keep_dis : boolean
	   flag for whether to store and return the array of values of the test statistics (default: false)
	seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator
	Returns
	-------
	float
	   estimated p-value 
	float
	   test statistic
    list
       distribution of test statistics (only if keep_dist == True)
	"""

	original_ts, prng = sum([x for x in sample if x == 1]) / len(sample), get_prng(seed)
	

	thePvalue = {'greater': lambda p: p, 'less': lambda p: 1-p, 'two-sided': lambda p: 2* np.min([p, 1-p])}

	def generate():

		return prng.binomial(n, original_ts, 1)


	permutations = []

	while reps >= 0:
		ts = generate()
		permutations.append(ts[0])
		reps -= 1

	simulations = list(permutations)
	
	alternative_func = {
	'greater': lambda thing: thing > y,
	'less': lambda thing: thing < y,
	'two-sided': lambda thing: thing != y
	}

	count = 0
	while len(permutations) >0:
		val = permutations.pop()
		if alternative_func[alternative](val):
			count += 1

	test_statistic = count 
	p_value = count / len(simulations)

	if keep_dist == True:
		return thePvalue[alternative](p_value), test_statistic, simulations

	return p_value*thePvalue[alternative](p_value), test_statistic




	
