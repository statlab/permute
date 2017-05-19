"""
One Sample Tests
"""

def mean(lst):
	
	"""
	Returns mean of list of elements.
	
	Parameters

	----------

	lst : list
		list of ints/floats

	Returns

	----------

	float
		mean of elements in lst

	"""
	
	return sum(lst)/len(lst)	



def median(lst):
	
	"""
	Returns median of list of elements

	Parameters

	----------

	lst : list
		list of ints/floats

	Returns

	----------

	float
		median of elements in lst

	"""
	
	lst.sort()
	
	leng = len(lst)

	if leng % 2 != 0:
			

		return lst[leng//2]

	else:

		second, first = lst[leng//2], lst[(leng//2) - 1]

		return (second + first) / 2


def quantile(lst, q):

	"""
	Returns index of element in lst and element value at desired quantile q (percent of values that fall below it).

	Parameters

	----------

	lst : list
		list of ints or floats

	q : float
		desired quantile

	Returns

	----------

	list of length 2
		index of element in lst : int
		element value in lst: int or float

	"""


	lst.sort()

	leng = len(lst)

	index = round(q * (leng + 1))

	return [index, lst[index]]



















