"""
Signed Rank Test
"""

import scipy
import numpy as np
from scipy.special import comb
from .utils import get_prng
from .wilcoxon_sum import wilcoxon_sum

def signed_rank(x, y, reps=10**5, tail="one", keep_dist=False, seed=None):
	"""
	Parameters
	----------
	x : array-like
		array of observations in the first sample
	y : array-like
		array of observations in the second sample
	reps : int
		number of repetitions
	tail: string
		tailed p-value; can either be "one" or "two" (defaults to "one")
	alternative : string
	   options: {'greater', 'less', 'two-sided'}
	   alternative hypothesis to test (default: 'greater')
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
	assert len(x) == len(y), "Samples must be the same size."
	
	prng = get_prng(seed)
	zipped = list(zip(x, y))
	differences = [y - x for x, y in zipped]
	absDifferences = sorted([abs(y-x) for x, y in zipped])
	rankABS = [x + 1 for x in range(len(absDifferences))]
	duplicates = {}
	if len(set(absDifferences)) < len(absDifferences):
		#handles ties in ranks
		for i in set(absDifferences):
			dups = [y for y in absDifferences if y == i]
			counter = len(dups)
			if counter > 1 and i not in duplicates.keys():
				duplicates[i] = counter
		for d in duplicates.keys():
			ind = absDifferences.index(d)
			avg = sum([x for x in range(ind + 1, ind + 1 + duplicates[d])]) / duplicates[d]
			for i in range(ind, ind + duplicates[d]):
				rankABS[i] = avg
	signs_in_order = []
	for a in absDifferences:
		if a not in differences:
			signs_in_order.append(-1)
		else:
			signs_in_order.append(1)
	negatives = []
	positives = []
	for i in range(len(signs_in_order)):
		if signs_in_order[i] == -1:
			negatives.append(rankABS[i])
		else:
			positives.append(rankABS[i])
	return wilcoxon_sum(negatives, positives, reps, tail, keep_dist, seed)







	









