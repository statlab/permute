"""
Gaffke's bound
"""

import numpy as np
from scipy.stats import rankdata

def gaffke_bound(lst, alpha=0.05, n_sim=10000):
    """
    performs a Gaffke Conjecture test and returns the lower bound of the mean based on the given significance level.

    Parameters
	----------
    lst: a lsit of non-negative random variables
    alpha: optional significance level
    n_sim: optional number of simulations 

    Returns
	-------
	Float
		lower Gaffke bound
    """
    n = len(lst)
    sorted_lst = np.sort(lst)
    statistics = []
    
    for _ in range(n_sim):
        u = np.random.rand(n)
        u.sort()
        u = np.append(u, 1)
        u_diff = np.diff(u)
        statistic = np.sum(sorted_lst * u_diff)
        statistics.append(statistic)
    
    statistics .sort()
    lower_bound_index = int(np.ceil(alpha * n_sim)) - 1
    lower_bound = statistics[lower_bound_index]
    
    return lower_bound
