"""
Wilcoxon Rank Sum Test

"""
import scipy
import numpy as np
from scipy.special import comb
from .utils import get_prng

def wilcoxon_sum(x, y, reps=10**5, tail="one", keep_dist=False, seed=None):
	"""
	Parameters
	----------
	x : array-like
		array of observations in the first sample; sum of ranks will be used as test statistic
	y : array-like
		array of observations in the second sample
	reps : int
		number of repetitions; if len(x) <= 10**2, the algorithm will disregard an input to
		reps and return a direct calculation since the sample size is small enough
	tail: string
		tailed p-value; can either be "one" or "two" (defaults to "one")
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
		p-value of Wilcoxon Sum Test
	list
		distribution of test statistics if 'keep_dist' == True

	"""
	m = len(x)
	n = len(y)
	mn = sorted(x + y)
	values = set(mn)
	ranks = [x for x in range(1, len(mn) + 1)]
	duplicates = {}
	if len(values) < len(mn):
		#handles ties in ranks
		for i in values:
			dups = [y for y in mn if y == i]
			counter = len(dups)
			if counter > 1 and i not in duplicates.keys():
				duplicates[i] = counter
		for d in duplicates.keys():
			ind = mn.index(d)
			avg = sum([x for x in range(ind + 1, ind + 1 + duplicates[d])]) / duplicates[d]
			for i in range(ind, ind + duplicates[d]):
				ranks[i] = avg
	rankM = 0
	for i in x:
		rankM += ranks[mn.index(i)]
	permutationRanks = []
	if m <= 10**2:
		#direct calculation for small sample
		for i in range(len(ranks)):
			for y in range(i + 1, len(ranks)):
				permutationRanks.append([ranks[i], ranks[y]])
		sumPerm = [sum(thing) for thing in permutationRanks]
		denom = len(sumPerm)
		testStatistics = []
		if keep_dist:
			for R in set(sumPerm):
				if R <=rankM:
					testStatistics.append(R)
			ts = len(testStatistics) / denom
			if tail == "two":
				ts = 2 * min(ts, 1-ts)
			return ts, testStatistics

		else:
			ts = 0
			for R in set(sumPerm):
				if R <= rankM:
					ts += 1
			ts = ts / denom
			if tail == "two":
				ts = 2 * np.min([ts, 1-ts])
			return ts

	else:
		#run random permutations "reps" times for larger samples
		if keep_dist:
			testStatistics = []
			prng = get_prng(seed)
			r = reps
			while r > 0:
				prng.shuffle(ranks)
				testStatistics.append(sum(ranks[0:m]))
				r -=1
			ts = len(testStatistics) / reps
			if tail == "two":
				ts = 2 * np.min([ts, 1-ts])
			return ts, testStatistics
		else:
			ts = 0
			prng = get_prng(seed)
			r = reps
			while r >0:
				prng.shuffle(ranks)
				temp = sum(ranks[0:m])
				if temp <= rankM:
					ts += 1
				r -=1
			ts = ts / reps
			if tail == "two":
				ts = 2 * np.min([ts, 1-ts])
			return ts
