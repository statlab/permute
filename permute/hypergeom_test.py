"""
Hypergeometric Test

"""
import scipy
import numpy as np
from scipy.special import comb
#from .utils import get_prng

def hypergeom_test(population, n, g, reps=10**5, alternative='greater', keep_dis=False, seed=None):
	
	"""
	Parameters
	----------
	population : array-like
	   list of elements consisting of x in {0, 1} where 0 represents a failure and 1 represents a success
	n : int
	   sample size
	g : int
	   hypothesized number of good elements in sample
	reps : int
	   number of repetitions (default: 10**5)
	alternative : {'greater', 'less', 'two-sided'}
	   alternative hypothesis to test (deafult: 'greater')
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
	   distribution of test statistics (only if keep_dis == True)

	"""

	prng = get_prng(seed)

	pop_G = sum(population)
	pop_B = len(population) - pop_G

	def generate():

		return prng.hypergeometric(pop_G, pop_B, n)

	while reps >= 0:
		ts = generate()
		permutations.append(ts)
		reps -= 1

	simulations = list(permutations)

	alternative_func = {
	'greater': lambda thing: thing > g,
	'less': lambda thing: thing < g,
	'two-sided': lambda thing: thing != g
	}

	count = 0
	while len(permutations) > 0:
		val = permutations.pop()
		if alternative_func[alternative](val):
			count += 1

	test_statistic = count 
	p_value = count / len(simulations)

	if keep_dis == True:
		return p_value, test_statistic, permutations

	return p_value, test_statistic

















