"""
Wilcoxon Rank Sum Test 

"""

def wilcoxon_sum(sample1, sample2, tail="one"):
	"""
	Parameters
	----------
	sample1: array-like
		array of observations in the first sample; sum of ranks will be used as test statistic
	sample2: array-like
		array of observations in the second sample
	tail: string
		tailed p-value; can either be "one" or "two" (defaults to "one") 
	
	Returns
	-------
	float
		p-value of Wilcoxon Sum Test

	"""
	m = len(sample1)
	n = len(sample2)
	mn = sorted(sample1 + sample2)
	values = set(mn)
	ranks = [x for x in range(1, len(mn) + 1)]
	duplicates = {}
	if len(values) < len(mn):
		#handles ties in ranks
		for x in values:
			dups = [y for y in mn if y == x]
			counter = len(dups)
			if counter > 1 and x not in duplicates.keys():
				duplicates[x] = counter
		for d in duplicates.keys():
			avg = sum([x for x in range(d, d +duplicates[d])]) / duplicates[d]
			for i in range(d-1, d - 1 + duplicates[d]):
				ranks[i] = avg	
	rankM = 0
	for x in sample1:
		rankM += ranks[mn.index(x)]
	permutationRanks = []
	for i in range(len(ranks)):
		for y in range(i + 1, len(ranks)):
			permutationRanks.append([ranks[i], ranks[y]])
	sumPerm = [sum(thing) for thing in permutationRanks]
	denom = len(sumPerm)
	finalP = 0
	for R in set(sumPerm):
		if R <= rankM:
			finalP += len([x for x in sumPerm if x == R]) / denom
	if tail == "one":
		return finalP
	return finalP * 2

			



	