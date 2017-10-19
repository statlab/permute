"""
Run Test
"""

import scipy
import numpy as np
from .utils import get_prng

def run(seq, reps=10**5, alternative='greater', keep_dist=False, seed=None):
	"""
	Parameters
	----------
	s : array-like
		sequence of {0, 1}
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
		p-value
	list
		distribution of test statistics if 'keep_dist' == True
	"""

	prng = get_prng(seed)
	
	#find number of runs in initial sequence
	originalRuns = 1
	currVar = seq[0]
	for s in range(len(seq)):
		if seq[s] != currVar:
			currVar = seq[s]
			originalRuns += 1
	

	alternative_func = {
        'greater': lambda thing: thing > originalRuns,
        'less': lambda thing: thing < originalRuns
    }


	#permutation test
	testStatistics = []
	countRanks = 1
	currVar = seq[0]
	for r in range(reps):
		countRanks = 1
		prng.shuffle(seq)
		currVar = seq[0]
		for i in range(len(seq)):
			if seq[i] != currVar:
				currVar = seq[i]
				countRanks += 1
		testStatistics.append(countRanks)
	#compute p-value
	if alternative == "two-sided":
		p1 = 0
		for ts in testStatistics:
			if alternative_func['greater'](ts):
				p1 += 1
		p1_val = p1 / len(testStatistics)
		p2 = 0
		for ts2 in testStatistics:
			if alternative_func['less'](ts):
				p2 += 1
		p2_val = p2 / len(testStatistics)
		finalPVALUE = 2 * min(p1_val, p2_val)
		if keep_dist:
			return finalPVALUE, testStatistics
		return finalPVALUE
	else: 
		p = 0
		for ts in testStatistics:
			if alternative_func[alternative](ts):
				p += 1
		finalPVALUE = p / len(testStatistics)
		if keep_dist:
			return finalPVALUE, testStatistics
		return finalPVALUE






