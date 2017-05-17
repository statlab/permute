

def mean(list):
	
	"""Returns mean (type: float) of python list"""
	
	return sum(list)/len(list)	



def median(list):
	
	"""Returns median (type: float) of python list"""
	
	list.sort()
	leng = len(list)

	if leng % 2 != 0:
			

		return list[leng//2]

	else:

		second, first = list[leng//2], list[(leng//2) - 1]

		return (second + first) / 2


def quantile(list, q):

	"""Input: python list and desired quantile (percent of values in decimale form that fall below it)
		Output: tuple of (index, list element at index)"""

	list.sort()

	leng = len(list)

	return (round(q * (leng + 1)), list[round(q * (leng + 1))])



















