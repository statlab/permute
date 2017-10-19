"""
Binomial Permutation Test
"""
import scipy
import numpy as np
from scipy.special import comb
from .utils import get_prng


def binomial_p(sample, reps=10**5, alternative='greater', keep_dist=False, seed=None):
	"""
	Parameters
	----------
	sample : array-like
	   list of elements consisting of x in {0, 1} where 0 represents a failure and
	   1 represents a seccuess
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

	original_ts = sum([x for x in sample if x == 1]) / len(sample)

	prng = get_prng(seed)


	def generate():

		return prng.binomial(1, original_ts)


	permutations = []

	while reps > 0:
		ts = generate()
		permutations.append(ts)
		reps -= 1

	simulations = list(permutations)
	permutations2 = list(permutations)
	
	alternative_func = {
	'greater': lambda thing: thing > original_ts,
	'less': lambda thing: thing < original_ts,
	}

	if alternative == 'two-sided':
		count = 0
		while len(permutations) >0:
			val = permutations.pop()
			if alternative_func['greater'](val):
				count += 1
		p_valueG = count / len(simulations)
		counter = 0
		while len(permutations2) > 0:
			val = permutations2.pop()
			if alternative_func['less'](val):
				counter += 1
		p_valueL = counter / len(simulations)
		p_value = 2 * min(p_valueG, p_valueL)


	
	else:

		count = 0
		while len(permutations) >0:
			val = permutations.pop()
			if alternative_func[alternative](val):
				count += 1
		p_value = count / len(simulations)


	if keep_dist == True:
		return p_value, original_ts, simulations

	return p_value, original_ts




	
