"""
Binomial Permutation Test
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from .utils import get_prng


def binomial_p(x, n, p0, reps=10**5, alternative='greater', keep_dist=False, seed=None):
    """
    Parameters
    ----------
    sample : array-like
       list of elements consisting of x in {0, 1} where 0 represents a failure and
       1 represents a seccuess
    p0 : int
       hypothesized number of successes in n trials
    n : int
       number of trials 
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

    if n < x:
        raise ValueError("Cannot observe more ones than the population size")

    prng = get_prng(seed)

    def generate():
        return prng.binomial(n, p0, 1)[0]
    
    if keep_dist:
        permutations = np.empty(reps)
        for i in range(reps):
            permutations[i] = generate()
        if alternative == 'two-sided':
            hits_up = np.sum(permutations >= x)
            hits_low = np.sum(permutations <= x)
            p_value = 2*np.min([hits_up/reps, hits_low/reps, 0.5])
        elif alternative == 'greater':
            p_value = np.mean(permutations >= x)
        else:
            p_value = np.mean(permutations <= x)
        return p_value, x, permutations

    else:
        hits_up = 0
        hits_low = 0
        for i in range(reps):
            ts = generate()
            hits_up += (ts >= x)
            hits_low += (ts <= x)
        
        if alternative == 'two-sided':
            p_value = 2*np.min([hits_up/reps, hits_low/reps, 0.5])
        elif alternative == 'greater':
            p_value = hits_up/reps
        else:
            p_value = hits_low/reps

        return p_value, x