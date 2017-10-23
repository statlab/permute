"""
Hypergeometric Test
"""
import numpy as np
from .utils import get_prng

def hypergeometric(x, N, n, G, reps=10**5, alternative='greater', keep_dist=False, seed=None):
    
    """
    Parameters
    ----------
    x : int
        number of `good` elements observed in the sample
    N : int
        population size
    n : int
       sample size
    G : int
       hypothesized number of good elements in population
    reps : int
       number of repetitions (default: 10**5)
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    keep_dist : boolean
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
        raise ValueError("Cannot observe more good elements than the sample size")
    if N < n:
        raise ValueError("Population size cannot be smaller than sample")
    if N < G:
        raise ValueError("Number of good elements can't exceed the population size")
    if G < x:
        raise ValueError("Number of observed good elements can't exceed the number in the population")

    prng = get_prng(seed)

    def generate():
        return prng.hypergeometric(G, N-G, n)

    if keep_dist:
        permutations = np.empty(reps)
        for i in range(reps):
            permutations[i] = generate()
        if alternative == 'two-sided':
            hits_up = np.sum(permutations >= x)
            hits_low = np.sum(permutations <= x)
            p_value = 2*np.min(hits_up/reps, hits_low/reps, 0.5)
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



