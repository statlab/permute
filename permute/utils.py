"""
Various utilities and helper functions.
"""


import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, hypergeom
from cryptorandom.cryptorandom import SHA256
from cryptorandom.sample import random_sample, random_permutation

def binom_conf_interval(n, x, cl=0.975, alternative="two-sided", p=None,
                        **kwargs):
    """
    Compute a confidence interval for a binomial p, the probability of success in each trial.

    Parameters
    ----------
    n : int
        The number of Bernoulli trials.
    x : int
        The number of successes.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    p : float in (0, 1)
        Starting point in search for confidence bounds for probability of success in each trial.
    kwargs : dict
        Key word arguments

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (approximately)
        1-alpha.

    Notes
    -----
    xtol : float
        Tolerance
    rtol : float
        Tolerance
    maxiter : int
        Maximum number of iterations.
    """
    assert alternative in ("two-sided", "lower", "upper")

    if p is None:
        p = x / n
    ci_low = 0.0
    ci_upp = 1.0

    if alternative == 'two-sided':
        cl = 1 - (1 - cl) / 2

    if alternative != "upper" and x > 0:
        f = lambda q: cl - binom.cdf(x - 1, n, q)
        while f(p) < 0:
            p = (p+1)/2
        ci_low = brentq(f, 0.0, p, *kwargs)
    if alternative != "lower" and x < n:
        f = lambda q: binom.cdf(x, n, q) - (1 - cl)
        while f(p) < 0:
            p = p/2
        ci_upp = brentq(f, 1.0, p, *kwargs)

    return ci_low, ci_upp


def hypergeom_conf_interval(n, x, N, cl=0.975, alternative="two-sided", G=None,
                            **kwargs):
    """
    Confidence interval for a hypergeometric distribution parameter G, the number of good
    objects in a population in size N, based on the number x of good objects in a simple
    random sample of size n.

    Parameters
    ----------
    n : int
        The number of draws without replacement.
    x : int
        The number of "good" objects in the sample.
    N : int
        The number of objects in the population.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    G : int in [0, N]
        Starting point in search for confidence bounds for the hypergeometric parameter G.
    kwargs : dict
        Key word arguments

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (at least)
        1-alpha.

    Notes
    -----
    xtol : float
        Tolerance
    rtol : float
        Tolerance
    maxiter : int
        Maximum number of iterations.
    """
    assert alternative in ("two-sided", "lower", "upper")

    if G is None:
        G = (x / n) * N
    ci_low = 0
    ci_upp = N

    if alternative == 'two-sided':
        cl = 1 - (1 - cl) / 2

    if alternative != "upper" and x > 0:
        f = lambda q: cl - hypergeom.cdf(x - 1, N, q, n)
        while f(G) < 0:
            G = (G+N)/2
        ci_low = math.ceil(brentq(f, 0.0, G, *kwargs))

    if alternative != "lower" and x < n:
        f = lambda q: hypergeom.cdf(x, N, q, n) - (1 - cl)
        while f(G) < 0:
            G = G/2
        ci_upp = math.floor(brentq(f, G, N, *kwargs))

    return ci_low, ci_upp


def hypergeometric(x, N, n, G, alternative='greater'):
    
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
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    Returns
    -------
    float
       estimated p-value
    """
    if n < x:
        raise ValueError("Cannot observe more good elements than the sample size")
    if N < n:
        raise ValueError("Population size cannot be smaller than sample")
    if N < G:
        raise ValueError("Number of good elements can't exceed the population size")
    if G < x:
        raise ValueError("Number of observed good elements can't exceed the number in the population")

    assert alternative in ("two-sided", "less", "greater")
    if n < x:
        raise ValueError("Cannot observe more successes than the population size")

    plower = hypergeom.cdf(x, N, G, n)
    pupper = hypergeom.sf(x-1, N, G, n)
    if alternative == 'two-sided':
        pvalue = 2*np.min([plower, pupper, 0.5])
    elif alternative == 'greater':
        pvalue = pupper
    elif alternative == 'less':
        pvalue = plower
    return pvalue


def binomial_p(x, n, p, alternative='greater'):
    """
    Parameters
    ----------
    x : array-like
       list of elements consisting of x in {0, 1} where 0 represents a failure and
       1 represents a seccuess
    p : int
       hypothesized number of successes in n trials
    n : int
       number of trials 
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    Returns
    -------
    float
       estimated p-value 
    """

    assert alternative in ("two-sided", "less", "greater")
    if n < x:
        raise ValueError("Cannot observe more successes than the population size")

    plower = binom.cdf(x, n, p)
    pupper = binom.sf(x-1, n, p)
    if alternative == 'two-sided':
        pvalue = 2*np.min([plower, pupper, 0.5])
    elif alternative == 'greater':
        pvalue = pupper
    elif alternative == 'less':
        pvalue = plower
    return pvalue


def get_prng(seed=None):
    """Turn seed into a cryptorandom instance

    Parameters
    ----------
    seed : {None, int, str, RandomState}
        If seed is None, return generate a pseudo-random 63-bit seed using np.random
        and return a new SHA256 instance seeded with it.
        If seed is a number or str, return a new cryptorandom instance seeded with seed.
        If seed is already a numpy.random RandomState or SHA256 instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    RandomState
    """
    if seed is None:
        # Need to specify dtype (Windows defaults to int32)
        seed = np.random.randint(0, 10**10, dtype=np.int64) # generate an integer
    if seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer, float, str)):
        return SHA256(seed)
    if isinstance(seed, (np.random.RandomState, SHA256)):
        return seed
    raise ValueError('%r cannot be used to seed cryptorandom' % seed)


def permute_within_groups(x, group, seed=None):
    """
    Permutation of condition within each group.

    Parameters
    ----------
    x : array-like
        A 1-d array indicating treatment.
    group : array-like
        A 1-d array indicating group membership
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    permuted : array-like
        The within group permutation of x.
    """
    prng = get_prng(seed)
    permuted = x.copy()

    for g in np.unique(group):
        gg = group == g
        permuted[gg] = random_permutation(permuted[gg], prng=prng)
    return permuted


def permute(x, seed=None):
    """
    Permute an array in-place

    Parameters
    ----------
    x : array-like
        A 1-d array
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    None
        Original array is permuted in-place, nothing is returned.
    """
    return random_permutation(x, prng=seed)


def permute_rows(m, seed=None):
    """
    Permute the rows of a matrix in-place

    Parameters
    ----------
    m : array-like
        A 2-d array
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    None
        Original matrix is permuted in-place, nothing is returned.
    """
    prng = get_prng(seed)

    mprime = []
    for row in m:
        mprime.append(random_permutation(row, prng=prng))
    return np.array(mprime)

def permute_incidence_fixed_sums(incidence, k=1, seed=None):
    """
    Permute elements of a (binary) incidence matrix, keeping the
    row and column sums in-tact.

    Parameters
    ----------
    incidence : 2D ndarray
        Incidence matrix to permute.
    k : int
        The number of successful pairwise swaps to perform.
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Notes
    -----
    The row and column sums are kept fixed by always swapping elements
    two pairs at a time.

    Returns
    -------
    permuted : 2D ndarray
        The permuted incidence matrix.
    """

    if not incidence.ndim == 2:
        raise ValueError("Incidence matrix must be 2D")

    if incidence.min() != 0 or incidence.max() != 1:
        raise ValueError("Incidence matrix must be binary")

    prng = get_prng(seed)

    incidence = incidence.copy()
    n, m = incidence.shape
    rows = np.arange(n)
    cols = np.arange(m)
    K, k = k, 0

    while k < K:
        swappable = False
        while not swappable:
            chosen_rows = np.random.choice(rows, 2, replace=False)
            s0, s1 = chosen_rows

            potential_cols0, = np.where((incidence[s0, :] == 1) &
                                        (incidence[s1, :] == 0))

            potential_cols1, = np.where((incidence[s0, :] == 0) &
                                        (incidence[s1, :] == 1))

            potential_cols0 = np.setdiff1d(potential_cols0, potential_cols1)

            if (len(potential_cols0) == 0) or (len(potential_cols1) == 0):
                continue

            p0 = prng.choice(potential_cols0)
            p1 = prng.choice(potential_cols1)

            # These statements should always be true, so we should
            # never raise an assertion here
            assert incidence[s0, p0] == 1
            assert incidence[s0, p1] == 0
            assert incidence[s1, p0] == 0
            assert incidence[s1, p1] == 1
            swappable = True
        i0 = incidence.copy()
        incidence[[s0, s0, s1, s1],
                  [p0, p1, p0, p1]] = [0, 1, 1, 0]
        k += 1
    return incidence


def potential_outcomes(x, y, f, finverse):
    """
    Given observations $x$ under treatment and $y$ under control conditions,
    returns the potential outcomes for units under their unobserved condition
    under the hypothesis that $x_i = f(y_i)$ for all units.

    Parameters
    ----------
    x : array-like
        Outcomes under treatment
    y : array-like
        Outcomes under control
    f : function
        An invertible function
    finverse : function
        The inverse function to f.

    Returns
    -------
    potential_outcomes : 2D array
        The first column contains all potential outcomes under the treatment,
        the second column contains all potential outcomes under the control.
    """

    tester = np.array(range(5)) + 1
    assert np.allclose(finverse(f(tester)),
                       tester), "f and finverse aren't inverses"
    assert np.allclose(f(finverse(tester)),
                       tester), "f and finverse aren't inverses"

    pot_treat = np.concatenate([x, f(y)])
    pot_ctrl = np.concatenate([finverse(x), y])

    return np.column_stack([pot_treat, pot_ctrl])
