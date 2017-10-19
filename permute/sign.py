"""
Sign Test

"""

import scipy
import numpy as np
from scipy.special import comb
from .utils import get_prng
from .binomialp import binomial_p


def sign(x, y, reps=10**5, alternative="greater", keep_dist=False, seed=None):

	"""

	Parameters
	----------
	x : array-like
		array of observations in the first sample
	y : array-like
		array of observations in the second sample
	reps : int
		number of repetitions
	keep_dist : boolean
		flag for whether to store and return the array of values
	seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

	Returns
	-------
	float
		p-value of Sign Test
	float
		observed test statistic
	list
		distribution of test statistics if 'keep_dist' == True
	"""
	assert len(x) == len(y), "Samples must have the same sizes."

	def assignStat(a, b):
		"""
		Helper method for assigning -1, 0, 1 to original pairs from samples.
		"""
		if a < b:
			return -1
		if a > b:
			return 1
		else:
			return 0

	def specialZero(assigned):
		"""
		Handles case of zeros in the sample. If there is an even number of zeros,
		randomly assign them -1 or 1. If there is an odd number,
		randomnly drop one and reduce the sample size by 1. Then, randomnly assign the
		remaining observations with -1 or 1.
		"""
		
		zeros = []
		for i in range(len(assigned)):
			if assigned[i] == 0:
				zeros.append(i)

		if len(zeros) == 0:
			return assigned
		if len(zeros) % 2 == 0:
			for i in range(len(assigned)):
				if assigned[i] == 0:
					assigned[i] = np.random.choice([-1, 1], 1)[0]
		elif len(zeros) % 2 != 0:
			indToDelete = np.random.choice(zeros, 1)[0]
			assigned.pop(indToDelete)
			for i in range(len(assigned)):
				if assigned[i] == 0:
					assigned[i] = np.random.choice([-1, 1], 1)[0]
		return assigned

	
	
	zipped = zip(x, y)
	assigned = [assignStat(x, y) for x, y in zipped]
	assigned = specialZero(assigned)
	for i in range(len(assigned)):
		if assigned[i] == -1:
			assigned[i] = 0
	return binomial_p(assigned, reps, alternative, keep_dist, seed)

