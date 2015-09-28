"""
Various utilities and helper functions.
"""
from __future__ import division, print_function, absolute_import

import numbers
import math
import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import (binom, hypergeom, ttest_ind, ttest_1samp)


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
        cl = 1 - (1-cl)/2

    if alternative != "upper" and x > 0:
        f = lambda q: cl - binom.cdf(x - 1, n, q)
        ci_low = brentq(f, 0.0, p, *kwargs)
    if alternative != "lower" and x < n:
        f = lambda q: binom.cdf(x, n, q) - (1 - cl)
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
        G = (x / n)*N
    ci_low = 0
    ci_upp = N

    if alternative == 'two-sided':
        cl = 1 - (1-cl)/2

    if alternative != "upper" and x > 0:
        f = lambda q: cl - hypergeom.cdf(x-1, N, q, n)
        ci_low = math.ceil(brentq(f, 0.0, G, *kwargs))

    if alternative != "lower" and x < n:
        f = lambda q: hypergeom.cdf(x, N, q, n) - (1-cl)
        ci_upp = math.floor(brentq(f, G, N, *kwargs))

    return ci_low, ci_upp


def get_prng(seed=None):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : {None, int, RandomState}
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    RandomState
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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

    # (avoid additional flops) -- maybe memoize
    for g in np.unique(group):
        gg = group == g
        permuted[gg] = prng.permutation(permuted[gg])
    return permuted


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

    for row in m:
        prng.shuffle(row)


def permute_incidence_fixed_sums(incidence, k=1):
    """
    Permute elements of a (binary) incidence matrix, keeping the
    row and column sums in-tact.

    Parameters
    ----------
    incidence : 2D ndarray
        Incidence matrix to permute.
    k : int
        The number of successful pairwise swaps to perform.

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

    incidence = incidence.copy()

    n, m = incidence.shape
    rows = np.arange(n)
    cols = np.arange(m)

    K, k = k, 0
    while k < K:

        swappable = False
        while (not swappable):
            chosen_rows = np.random.choice(rows, 2, replace=False)
            s0, s1 = chosen_rows

            potential_cols0, = np.where((incidence[s0, :] == 1) &
                                        (incidence[s1, :] == 0))

            potential_cols1, = np.where((incidence[s0, :] == 0) &
                                        (incidence[s1, :] == 1))

            potential_cols0 = np.setdiff1d(potential_cols0, potential_cols1)

            if (len(potential_cols0) == 0) or (len(potential_cols1) == 0):
                continue

            p0 = np.random.choice(potential_cols0)
            p1 = np.random.choice(potential_cols1)

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
    Given observations x under treatment and y under control conditions,
    returns the potential outcomes for units under their unobserved condition
    under the hypothesis that x_i = f(y_i) for all units.

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
    assert np.allclose(finverse(f(tester)), tester), "f and finverse aren't inverses"
    assert np.allclose(f(finverse(tester)), tester), "f and finverse aren't inverses"
    
    pot_treat = np.concatenate([x, f(y)])
    pot_ctrl = np.concatenate([finverse(x), y])

    return np.column_stack([pot_treat, pot_ctrl])